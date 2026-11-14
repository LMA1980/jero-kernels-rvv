//! `128-byte`-aligned weight repacking (`256b` `X100` / `1024b` `A100` + `8x16` `XSMTVDotII` `IME`
//! tiles), dynamic `f32 -> Q8` activation quantization, and row-sliceable RVV/IME GEMV & batched
//! multi-token `M x K` GEMM kernels.
//!
//! # Dual-Cluster Alignment Contract
//!
//! [`Aligned128`] enforces `#[repr(C, align(128))]` (`1024` bits) without custom `GlobalAlloc`
//! machinery (preserving full Miri Stacked/Tree-Borrows provenance). A `128-byte` boundary is
//! simultaneously:
//! - `4 x vlenb` (`4 x 32B`) on Cluster 0 (`X100`, `zvl256b`, `0x00FF`),
//! - `1 x vlenb` (`1 x 128B`) on Cluster 1 (`A100`, `zvl1024b`, `0xFF00`), and
//! - `1 x TILE_BYTES` (`8 x 16` `i8` = `128B`) for `SpacemiT` `XSMTVDotII` (`smt.vmadot`).
//!
//! Quantized weight matrices (`Q4_K_M`, `Q8_0`, `Q4_0`, `Q6_K`, `Q5_K`, `TQ1_0`, `TQ2_0`) unpack
//! once at load time into exact `i8` `8 x 16` tiles paired with contiguous per-tile `f32` scales
//! (and asymmetric `f32` mins for `Q4_K_M` / `Q5_K`). Both `X100` prefill (RVV 1.0 widening
//! reductions) and `A100` decode/prefill (`smt.vmadot`) consume the exact same repacked buffer
//! with zero layout conversion between phases.

use std::ops::Range;

use jero_core::{K3Error, QuantType, TensorView};

#[cfg(all(target_arch = "riscv64", not(miri)))]
use crate::dot_product_f32_rvv;
#[cfg(any(not(target_arch = "riscv64"), miri))]
use crate::dot_product_f32_scalar;
#[cfg(any(not(target_arch = "riscv64"), miri))]
use crate::ime::tile_dot_scalar_rows;
use crate::ime::{
    TILE_BYTES, TILE_C, TILE_K, TILE_M, TILE_N, TQ1_0_BLOCK_BYTES, TQ1_0_BLOCK_ELEMENTS,
    TQ2_0_BLOCK_BYTES, TQ2_0_BLOCK_ELEMENTS, tile_count, unpack_tq1_0_block_i8,
    unpack_tq2_0_block_i8,
};
#[cfg(all(target_arch = "riscv64", not(miri)))]
use crate::ime::{has_ime, tile_dot_rvv_rows};
use crate::quant::{
    I2_S_BLOCK_BYTES, I2_S_BLOCK_ELEMENTS, IQ4_NL_BLOCK_BYTES, IQ4_NL_BLOCK_ELEMENTS,
    KVALUES_IQ4NL, Q2_K_BLOCK_BYTES, Q2_K_BLOCK_ELEMENTS, Q3_K_BLOCK_BYTES, Q3_K_BLOCK_ELEMENTS,
    Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMENTS, Q4_1_BLOCK_BYTES, Q4_1_BLOCK_ELEMENTS, Q5_0_BLOCK_BYTES,
    Q5_0_BLOCK_ELEMENTS, Q5_1_BLOCK_BYTES, Q5_1_BLOCK_ELEMENTS, Q5_K_BLOCK_BYTES,
    Q5_K_BLOCK_ELEMENTS, Q6_K_BLOCK_BYTES, Q6_K_BLOCK_ELEMENTS, Q8_0_BLOCK_BYTES,
    Q8_0_BLOCK_ELEMENTS, read_i2_s_trailer_scale, unpack_i2_s_block_i8, unpack_iq4_nl_block_i8,
    unpack_q2_k_block_i8, unpack_q3_k_block_i8, unpack_q4_1_block_i8, unpack_q5_0_block_i8,
    unpack_q5_1_block_i8,
};
use crate::{
    Q4_K_M_BLOCK_BYTES, Q4_K_M_BLOCK_ELEMENTS, bf16_to_f32_finite, dot_product_f32_vla_stripmined,
    f16_to_f32_finite,
};

/// Default element threshold (`16M` elements = `64 MiB` as `f32`) up to which `F16` and `BF16`
/// tensors expand once to `128B`-aligned `f32` rows at load time; larger tensors (such as
/// large-vocabulary LM heads) stay in compact `128B`-aligned `u16` storage to respect `RLIMIT_MEMLOCK`.
pub const DEFAULT_DENSE_F32_MAX_ELEMENTS: usize = 16 * 1024 * 1024;

/// Number of `f32` elements in one [`Aligned128`] block (`128 / 4 = 32`).
pub const F32_PER_ALIGNED128: usize = 32;

/// Number of `u16` elements in one [`Aligned128`] block (`128 / 2 = 64`).
pub const U16_PER_ALIGNED128: usize = 64;

/// Quantization group size (`32` elements) used for dynamic `f32 -> Q8` activation quantization.
pub const ACT_Q8_GROUP_ELEMENTS: usize = 32;

/// Super-block size (`256` elements) tracked alongside `32`-element scales in [`QuantizedActQ8`].
pub const ACT_Q8_SUPER_BLOCK_ELEMENTS: usize = 256;

/// A `128-byte`-aligned memory block (`1024` bits) compatible with `zvl256b` (`X100`),
/// `zvl1024b` (`A100`), and `8 x 16` `i8` `XSMTVDotII` (`IME`) tiles.
#[repr(C, align(128))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Aligned128(pub [u8; 128]);

impl Default for Aligned128 {
    fn default() -> Self {
        Self::zeroed()
    }
}

impl Aligned128 {
    /// Returns a zero-initialized `128-byte`-aligned block.
    #[must_use]
    pub const fn zeroed() -> Self {
        Self([0_u8; 128])
    }

    /// Borrows the underlying `128` bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 128] {
        &self.0
    }

    /// Mutably borrows the underlying `128` bytes.
    #[must_use]
    pub const fn as_bytes_mut(&mut self) -> &mut [u8; 128] {
        &mut self.0
    }

    /// Views a slice of [`Aligned128`] blocks as a contiguous `&[u8]` slice (`128 * buf.len()` bytes).
    #[must_use]
    pub fn slice_as_bytes(buf: &[Self]) -> &[u8] {
        // SAFETY: `Aligned128` is `#[repr(C, align(128))]` wrapping `[u8; 128]` with zero padding.
        unsafe { core::slice::from_raw_parts(buf.as_ptr().cast::<u8>(), buf.len() * 128) }
    }

    /// Views a mutable slice of [`Aligned128`] blocks as a contiguous `&mut [u8]` slice.
    #[must_use]
    pub fn slice_as_bytes_mut(buf: &mut [Self]) -> &mut [u8] {
        // SAFETY: `Aligned128` is `#[repr(C, align(128))]` wrapping `[u8; 128]` with zero padding.
        unsafe { core::slice::from_raw_parts_mut(buf.as_mut_ptr().cast::<u8>(), buf.len() * 128) }
    }

    /// Views a slice of [`Aligned128`] blocks as a contiguous `&[i8]` slice (`128 * buf.len()` elements).
    #[must_use]
    pub fn slice_as_i8(buf: &[Self]) -> &[i8] {
        // SAFETY: `Aligned128` has size 128, alignment 128 >= align_of::<i8>(), and all byte bit-patterns are valid `i8`.
        unsafe { core::slice::from_raw_parts(buf.as_ptr().cast::<i8>(), buf.len() * 128) }
    }

    /// Views a mutable slice of [`Aligned128`] blocks as a contiguous `&mut [i8]` slice.
    #[must_use]
    pub fn slice_as_i8_mut(buf: &mut [Self]) -> &mut [i8] {
        // SAFETY: `Aligned128` has size 128, alignment 128 >= align_of::<i8>(), and all byte bit-patterns are valid `i8`.
        unsafe { core::slice::from_raw_parts_mut(buf.as_mut_ptr().cast::<i8>(), buf.len() * 128) }
    }

    /// Views a slice of [`Aligned128`] blocks as a contiguous `&[f32]` slice (`32 * buf.len()` elements).
    #[must_use]
    pub fn slice_as_f32(buf: &[Self]) -> &[f32] {
        // SAFETY: `Aligned128` has alignment 128 >= align_of::<f32>() (4) and size `32 * size_of::<f32>()`;
        // every bit pattern is a valid Rust `f32`.
        unsafe {
            core::slice::from_raw_parts(buf.as_ptr().cast::<f32>(), buf.len() * F32_PER_ALIGNED128)
        }
    }

    /// Views a mutable slice of [`Aligned128`] blocks as a contiguous `&mut [f32]` slice.
    #[must_use]
    pub fn slice_as_f32_mut(buf: &mut [Self]) -> &mut [f32] {
        // SAFETY: `Aligned128` has alignment 128 >= align_of::<f32>() (4) and size `32 * size_of::<f32>()`.
        unsafe {
            core::slice::from_raw_parts_mut(
                buf.as_mut_ptr().cast::<f32>(),
                buf.len() * F32_PER_ALIGNED128,
            )
        }
    }

    /// Views a slice of [`Aligned128`] blocks as a contiguous `&[u16]` slice (`64 * buf.len()` elements).
    #[must_use]
    pub fn slice_as_u16(buf: &[Self]) -> &[u16] {
        // SAFETY: `Aligned128` has alignment 128 >= align_of::<u16>() (2) and size `64 * size_of::<u16>()`.
        unsafe {
            core::slice::from_raw_parts(buf.as_ptr().cast::<u16>(), buf.len() * U16_PER_ALIGNED128)
        }
    }

    /// Views a mutable slice of [`Aligned128`] blocks as a contiguous `&mut [u16]` slice.
    #[must_use]
    pub fn slice_as_u16_mut(buf: &mut [Self]) -> &mut [u16] {
        // SAFETY: `Aligned128` has alignment 128 >= align_of::<u16>() (2) and size `64 * size_of::<u16>()`.
        unsafe {
            core::slice::from_raw_parts_mut(
                buf.as_mut_ptr().cast::<u16>(),
                buf.len() * U16_PER_ALIGNED128,
            )
        }
    }
}

#[inline]
fn quantize_block32_scalar(src: &[f32; 32], dst_q8: &mut [i8; 32]) -> (f32, f32) {
    let mut clean = [0.0_f32; 32];
    let mut absmax = 0.0_f32;
    let mut sum = 0.0_f32;
    for i in 0..32 {
        let v = if src[i].is_finite() { src[i] } else { 0.0_f32 };
        clean[i] = v;
        sum += v;
        let a = v.abs();
        if a > absmax {
            absmax = a;
        }
    }

    if absmax < 1e-30_f32 {
        dst_q8.fill(0);
        return (0.0_f32, sum);
    }

    let d_x = absmax / 127.0_f32;
    let inv_d = 127.0_f32 / absmax;
    for i in 0..32 {
        let q = (clean[i] * inv_d).round().clamp(-127.0_f32, 127.0_f32);
        #[expect(clippy::cast_possible_truncation, reason = "clamped to [-127, 127]")]
        {
            dst_q8[i] = q as i8;
        }
    }
    (d_x, sum)
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[inline]
#[allow(unused_assignments)]
#[expect(
    clippy::too_many_lines,
    reason = "two-pass RVV absmax/sum reduction and RMM i8 narrowing assembly"
)]
fn quantize_block32_rvv(src: &[f32; 32], dst_q8: &mut [i8; 32]) -> (f32, f32) {
    let mut clean = [0.0_f32; 32];
    let mut ps = src.as_ptr();
    let mut pc = clean.as_mut_ptr();
    let mut rem = 32_usize;
    let zero_f = 0.0_f32;
    let fmax = f32::MAX;
    let absmax: f32;
    let x_sum: f32;

    // SAFETY: `src` and `clean` hold 32 `f32` elements. `vmfle.vf v0, |x|, f32::MAX` yields `1`
    // only for finite lanes (filtering both `NaN` and `+-Inf` to `0.0` via `vmerge.vvm`).
    unsafe {
        core::arch::asm!(
            "vsetivli zero, 1, e32, m1, ta, ma",
            "vfmv.s.f v2, {z}",
            "vfmv.s.f v3, {z}",
            "2:",
            "vsetvli t0, {rem}, e32, m4, ta, ma",
            "vle32.v v8, ({ps})",
            "vfabs.v v12, v8",
            "vmfle.vf v0, v12, {fmax}",
            "vfmv.v.f v16, {z}",
            "vmerge.vvm v8, v16, v8, v0",
            "vse32.v v8, ({pc})",
            "vfabs.v v12, v8",
            "vfredmax.vs v2, v12, v2",
            "vfredusum.vs v3, v8, v3",
            "slli t1, t0, 2",
            "add {ps}, {ps}, t1",
            "add {pc}, {pc}, t1",
            "sub {rem}, {rem}, t0",
            "bnez {rem}, 2b",
            "vfmv.f.s {absmax}, v2",
            "vfmv.f.s {x_sum}, v3",
            ps = inout(reg) ps,
            pc = inout(reg) pc,
            rem = inout(reg) rem,
            z = in(freg) zero_f,
            fmax = in(freg) fmax,
            absmax = out(freg) absmax,
            x_sum = out(freg) x_sum,
            out("t0") _,
            out("t1") _,
            out("v0") _,
            out("v2") _,
            out("v3") _,
            out("v8") _,
            out("v9") _,
            out("v10") _,
            out("v11") _,
            out("v12") _,
            out("v13") _,
            out("v14") _,
            out("v15") _,
            out("v16") _,
            out("v17") _,
            out("v18") _,
            out("v19") _,
            options(nostack),
        );
    }

    if absmax < 1e-30_f32 {
        dst_q8.fill(0);
        return (0.0_f32, x_sum);
    }

    let d_x = absmax / 127.0_f32;
    let inv_d = 127.0_f32 / absmax;
    let min_q = -127.0_f32;
    let max_q = 127.0_f32;
    let mut pc_in = clean.as_ptr();
    let mut pq = dst_q8.as_mut_ptr();
    let mut rem_q = 32_usize;

    // SAFETY: `clean` has 32 sanitized `f32`s and `dst_q8` has space for 32 `i8`s.
    // `csrrwi t2, frm, 4` temporarily sets `RMM` (ties-to-max-magnitude, matching `f32::round`)
    // and restores the caller's `frm` before returning.
    unsafe {
        core::arch::asm!(
            "csrrwi t2, frm, 4",
            "3:",
            "vsetvli t0, {rem}, e32, m4, ta, ma",
            "vle32.v v8, ({pc})",
            "vfmul.vf v8, v8, {inv_d}",
            "vfmax.vf v8, v8, {min_q}",
            "vfmin.vf v8, v8, {max_q}",
            "vfcvt.x.f.v v8, v8",
            "vsetvli zero, t0, e16, m2, ta, ma",
            "vnsra.wi v12, v8, 0",
            "vsetvli zero, t0, e8, m1, ta, ma",
            "vnsra.wi v16, v12, 0",
            "vse8.v v16, ({pq})",
            "add {pq}, {pq}, t0",
            "slli t1, t0, 2",
            "add {pc}, {pc}, t1",
            "sub {rem}, {rem}, t0",
            "bnez {rem}, 3b",
            "csrrw zero, frm, t2",
            pc = inout(reg) pc_in,
            pq = inout(reg) pq,
            rem = inout(reg) rem_q,
            inv_d = in(freg) inv_d,
            min_q = in(freg) min_q,
            max_q = in(freg) max_q,
            out("t0") _,
            out("t1") _,
            out("t2") _,
            out("v8") _,
            out("v9") _,
            out("v10") _,
            out("v11") _,
            out("v12") _,
            out("v13") _,
            out("v16") _,
            options(nostack),
        );
    }

    (d_x, x_sum)
}

fn validate_q8_row_buffers(
    input_len: usize,
    out_quants_len: usize,
    out_scales_len: usize,
    out_sums_len: usize,
) -> Result<usize, K3Error> {
    if input_len == 0 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "quantize_row_f32_to_q8_32".to_owned(),
            offset: 0,
            byte_len: 0,
            buffer_len: 0,
            reason: "activation slice must be non-empty".to_owned(),
        });
    }
    let blocks = input_len.div_ceil(ACT_Q8_GROUP_ELEMENTS);
    let req_quants = blocks * ACT_Q8_GROUP_ELEMENTS;
    if out_quants_len < req_quants || out_scales_len < blocks || out_sums_len < blocks {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "quantize_row_f32_to_q8_32".to_owned(),
            offset: 0,
            byte_len: req_quants,
            buffer_len: out_quants_len,
            reason: format!(
                "insufficient Q8 output buffers: quants={out_quants_len}/{req_quants}, scales={out_scales_len}/{blocks}, sums={out_sums_len}/{blocks}"
            ),
        });
    }
    Ok(blocks)
}

/// Dynamically quantizes an `f32` activation slice into `32`-element symmetric `Q8` (`i8`) blocks,
/// recording per-block `absmax / 127.0` scales and exact `f32` block sums using portable scalar arithmetic.
///
/// Tail elements beyond `input.len()` up to `input.len().next_multiple_of(32)` are zero-padded,
/// and non-finite inputs (`NaN`, `+-Inf`) are sanitized to `0.0`.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `input` is empty or output slices are too short.
pub fn quantize_row_f32_to_q8_32_scalar(
    input: &[f32],
    out_quants: &mut [i8],
    out_scales: &mut [f32],
    out_sums: &mut [f32],
) -> Result<(), K3Error> {
    let blocks = validate_q8_row_buffers(
        input.len(),
        out_quants.len(),
        out_scales.len(),
        out_sums.len(),
    )?;
    let mut dst_blk = [0_i8; ACT_Q8_GROUP_ELEMENTS];
    for b in 0..blocks {
        let start = b * ACT_Q8_GROUP_ELEMENTS;
        let end = (start + ACT_Q8_GROUP_ELEMENTS).min(input.len());
        let mut buf = [0.0_f32; ACT_Q8_GROUP_ELEMENTS];
        buf[..end - start].copy_from_slice(&input[start..end]);
        let (scale, sum) = quantize_block32_scalar(&buf, &mut dst_blk);
        out_quants[start..start + ACT_Q8_GROUP_ELEMENTS].copy_from_slice(&dst_blk);
        out_scales[b] = scale;
        out_sums[b] = sum;
    }
    Ok(())
}

/// Dynamically quantizes an `f32` activation slice into `32`-element `Q8` (`i8`) blocks using
/// VLA RVV 1.0 `vfredmax.vs` + reciprocal-scale `vfcvt.x.f.v` (`RMM`) rounding.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `input` is empty or output slices are too short.
#[cfg(all(target_arch = "riscv64", not(miri)))]
pub fn quantize_row_f32_to_q8_32_rvv(
    input: &[f32],
    out_quants: &mut [i8],
    out_scales: &mut [f32],
    out_sums: &mut [f32],
) -> Result<(), K3Error> {
    let blocks = validate_q8_row_buffers(
        input.len(),
        out_quants.len(),
        out_scales.len(),
        out_sums.len(),
    )?;
    let mut dst_blk = [0_i8; ACT_Q8_GROUP_ELEMENTS];
    for b in 0..blocks {
        let start = b * ACT_Q8_GROUP_ELEMENTS;
        let end = (start + ACT_Q8_GROUP_ELEMENTS).min(input.len());
        let mut buf = [0.0_f32; ACT_Q8_GROUP_ELEMENTS];
        buf[..end - start].copy_from_slice(&input[start..end]);
        let (scale, sum) = quantize_block32_rvv(&buf, &mut dst_blk);
        out_quants[start..start + ACT_Q8_GROUP_ELEMENTS].copy_from_slice(&dst_blk);
        out_scales[b] = scale;
        out_sums[b] = sum;
    }
    Ok(())
}

/// Dynamically quantizes an `f32` activation slice into `32`-element `Q8` (`i8`) blocks,
/// dispatching to RVV 1.0 on `riscv64` and portable scalar elsewhere.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `input` is empty or output slices are too short.
pub fn quantize_row_f32_to_q8_32(
    input: &[f32],
    out_quants: &mut [i8],
    out_scales: &mut [f32],
    out_sums: &mut [f32],
) -> Result<(), K3Error> {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        quantize_row_f32_to_q8_32_rvv(input, out_quants, out_scales, out_sums)
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        quantize_row_f32_to_q8_32_scalar(input, out_quants, out_scales, out_sums)
    }
}

/// Dynamically quantized `[M, K]` activation batch packed into `128-byte`-aligned `8 x 16` `i8`
/// tiles alongside `32`-element scales, exact `f32` sums (`32`-element for `Q4_K_M`/`Q5_K`/`Q4_1`/`Q5_1`
/// and `16`-element for `Q2_K` min offsets), `256`-element super-block scales, and `128-byte`-aligned
/// sanitized `f32` rows (for lossless `F32` / `F16` / `BF16` projections).
#[derive(Debug, Clone)]
pub struct QuantizedActQ8 {
    m: usize,
    k: usize,
    k_padded: usize,
    m_tiles: usize,
    k_chunks: usize,
    k_blocks32: usize,
    tiles: Vec<Aligned128>,
    /// Layout: `[m_tile: 0..m_tiles][b32: 0..k_blocks32][row_in_tile: 0..8]`.
    x_scales: Vec<f32>,
    /// Layout: `[m_tile: 0..m_tiles][b32: 0..k_blocks32][row_in_tile: 0..8]`.
    x_sums: Vec<f32>,
    /// Layout: `[m_tile: 0..m_tiles][ch: 0..k_chunks][row_in_tile: 0..8]` (`16`-element chunk sums).
    x_sums_16: Vec<f32>,
    /// Layout: `[row_m: 0..m][sb256: 0..k.div_ceil(256)]`.
    super_scales_256: Vec<f32>,
    /// Layout: `[row_m: 0..m][aligned_blk: 0..k_padded / 32]` (`32` `f32`s per `Aligned128`).
    f32_rows: Vec<Aligned128>,
}

impl Default for QuantizedActQ8 {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantizedActQ8 {
    /// Creates an empty reusable [`QuantizedActQ8`] scratch container.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            m: 0,
            k: 0,
            k_padded: 0,
            m_tiles: 0,
            k_chunks: 0,
            k_blocks32: 0,
            tiles: Vec::new(),
            x_scales: Vec::new(),
            x_sums: Vec::new(),
            x_sums_16: Vec::new(),
            super_scales_256: Vec::new(),
            f32_rows: Vec::new(),
        }
    }

    /// Quantizes a `[M, K]` row-major `f32` activation matrix into `128-byte`-aligned `Q8` tiles.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] if `m == 0`, `k == 0`, or `act.len() != m * k`.
    pub fn quantize(act: &[f32], m: usize, k: usize) -> Result<Self, K3Error> {
        let mut out = Self::new();
        out.quantize_into(act, m, k)?;
        Ok(out)
    }

    /// Quantizes a single `[1, K]` `f32` activation vector for single-token decode GEMV.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] if `act` is empty.
    pub fn quantize_row(act: &[f32]) -> Result<Self, K3Error> {
        Self::quantize(act, 1, act.len())
    }

    /// Quantizes `[M, K]` activations into `self`, reusing previously allocated heap buffers.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] if `m == 0`, `k == 0`, or `act.len() != m * k`.
    pub fn quantize_into(&mut self, act: &[f32], m: usize, k: usize) -> Result<(), K3Error> {
        let expected = m.checked_mul(k).ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: "QuantizedActQ8".to_owned(),
            offset: 0,
            byte_len: size_of_val(act),
            buffer_len: usize::MAX,
            reason: "m * k overflow in activation quantization".to_owned(),
        })?;
        if m == 0 || k == 0 || act.len() != expected {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "QuantizedActQ8".to_owned(),
                offset: 0,
                byte_len: size_of_val(act),
                buffer_len: expected.saturating_mul(size_of::<f32>()),
                reason: format!(
                    "activation length {} does not match m ({m}) * k ({k})",
                    act.len()
                ),
            });
        }

        let k_padded = k.next_multiple_of(ACT_Q8_GROUP_ELEMENTS);
        let m_tiles = tile_count(m);
        let k_chunks = k_padded / TILE_K;
        let k_blocks32 = k_padded / ACT_Q8_GROUP_ELEMENTS;
        let sb256_per_row = k.div_ceil(ACT_Q8_SUPER_BLOCK_ELEMENTS);

        self.m = m;
        self.k = k;
        self.k_padded = k_padded;
        self.m_tiles = m_tiles;
        self.k_chunks = k_chunks;
        self.k_blocks32 = k_blocks32;

        self.tiles.clear();
        self.tiles.resize(m_tiles * k_chunks, Aligned128::zeroed());

        let meta_len = m_tiles * k_blocks32 * TILE_M;
        self.x_scales.clear();
        self.x_scales.resize(meta_len, 0.0_f32);
        self.x_sums.clear();
        self.x_sums.resize(meta_len, 0.0_f32);
        self.x_sums_16.clear();
        self.x_sums_16.resize(m_tiles * k_chunks * TILE_M, 0.0_f32);

        self.super_scales_256.clear();
        self.super_scales_256.resize(m * sb256_per_row, 0.0_f32);

        self.f32_rows.clear();
        self.f32_rows.resize(m * k_blocks32, Aligned128::zeroed());

        let mut row_q8 = vec![0_i8; k_padded];
        let mut row_scales = vec![0.0_f32; k_blocks32];
        let mut row_sums = vec![0.0_f32; k_blocks32];

        for row in 0..m {
            let src_row = &act[row * k..(row + 1) * k];
            quantize_row_f32_to_q8_32(src_row, &mut row_q8, &mut row_scales, &mut row_sums)?;

            // Populate sanitized 128B-aligned f32 row for DenseF32 / DenseHalf kernels
            let dst_f32_blocks = &mut self.f32_rows[row * k_blocks32..(row + 1) * k_blocks32];
            let dst_f32 = Aligned128::slice_as_f32_mut(dst_f32_blocks);
            for (dst, &val) in dst_f32.iter_mut().zip(src_row) {
                *dst = if val.is_finite() { val } else { 0.0_f32 };
            }

            let mt = row / TILE_M;
            let r_in_tile = row % TILE_M;

            for b32 in 0..k_blocks32 {
                let meta_idx = (mt * k_blocks32 + b32) * TILE_M + r_in_tile;
                let sc = row_scales[b32];
                self.x_scales[meta_idx] = sc;
                self.x_sums[meta_idx] = row_sums[b32];

                let sb_idx = (b32 * ACT_Q8_GROUP_ELEMENTS) / ACT_Q8_SUPER_BLOCK_ELEMENTS;
                if sb_idx < sb256_per_row {
                    let super_slot = &mut self.super_scales_256[row * sb256_per_row + sb_idx];
                    if sc > *super_slot {
                        *super_slot = sc;
                    }
                }
            }

            let row_f32 =
                Aligned128::slice_as_f32(&self.f32_rows[row * k_blocks32..(row + 1) * k_blocks32]);
            for ch in 0..k_chunks {
                let tile_bytes = Aligned128::slice_as_i8_mut(core::slice::from_mut(
                    &mut self.tiles[mt * k_chunks + ch],
                ));
                let dst_off = r_in_tile * TILE_K;
                let src_off = ch * TILE_K;
                tile_bytes[dst_off..dst_off + TILE_K]
                    .copy_from_slice(&row_q8[src_off..src_off + TILE_K]);
                self.x_sums_16[(mt * k_chunks + ch) * TILE_M + r_in_tile] =
                    row_f32[src_off..src_off + TILE_K].iter().copied().sum();
            }
        }

        Ok(())
    }

    /// Number of token rows `M` in the quantized activation batch.
    #[must_use]
    pub const fn m(&self) -> usize {
        self.m
    }

    /// Logical inner dimension `K` (`cols`).
    #[must_use]
    pub const fn k(&self) -> usize {
        self.k
    }

    /// `32`-element padded inner dimension `K`.
    #[must_use]
    pub const fn k_padded(&self) -> usize {
        self.k_padded
    }

    /// Number of `8`-row `M` tiles (`m.div_ceil(8)`).
    #[must_use]
    pub const fn m_tiles(&self) -> usize {
        self.m_tiles
    }

    /// Number of `16`-element `K` chunks (`k_padded / 16`).
    #[must_use]
    pub const fn k_chunks(&self) -> usize {
        self.k_chunks
    }

    /// Packed `128-byte`-aligned `8 x 16` `i8` activation tiles (`[m_tile][k_chunk]`).
    #[must_use]
    pub fn tiles(&self) -> &[Aligned128] {
        &self.tiles
    }

    /// Per-`32`-element block activation scales (`absmax / 127.0`) in `[m_tile][b32][8]` layout.
    #[must_use]
    pub fn x_scales(&self) -> &[f32] {
        &self.x_scales
    }

    /// Exact `f32` per-`32`-element block sums in `[m_tile][b32][8]` layout.
    #[must_use]
    pub fn x_sums(&self) -> &[f32] {
        &self.x_sums
    }

    /// Per-`256`-element super-block activation scales (`[m][k.div_ceil(256)]`).
    #[must_use]
    pub fn super_scales(&self) -> &[f32] {
        &self.super_scales_256
    }

    #[inline]
    fn f32_row(&self, row: usize) -> &[f32] {
        let blocks = &self.f32_rows[row * self.k_blocks32..(row + 1) * self.k_blocks32];
        &Aligned128::slice_as_f32(blocks)[..self.k]
    }
}

#[derive(Debug, Clone)]
enum RepackedStorage {
    BlockScaledI8 {
        n_tiles: usize,
        k_chunks: usize,
        chunks_per_scale: usize,
        scale_groups: usize,
        /// Layout: `[n_tile: 0..n_tiles][k_chunk: 0..k_chunks]` (`8 x 16` `i8` per `Aligned128`).
        tiles: Vec<Aligned128>,
        /// Layout: `[n_tile: 0..n_tiles][sg: 0..scale_groups][j: 0..8]`.
        scales: Vec<f32>,
        /// Layout: `[n_tile: 0..n_tiles][sg: 0..scale_groups][j: 0..8]` (present for `Q4_K_M`,
        /// `Q5_K`, `Q4_1`, `Q5_1`, `Q2_K`).
        mins: Option<Vec<f32>>,
    },
    DenseF32 {
        blocks_per_row: usize,
        /// Layout: `[row: 0..rows][blk: 0..blocks_per_row]` (`32` `f32`s per `Aligned128`).
        data: Vec<Aligned128>,
    },
    DenseHalf {
        is_bf16: bool,
        blocks_per_row: usize,
        /// Layout: `[row: 0..rows][blk: 0..blocks_per_row]` (`64` `u16`s per `Aligned128`).
        data: Vec<Aligned128>,
    },
}

/// Load-time `128-byte`-aligned repacked weight matrix supporting both `X100` (`zvl256b`, `32B`)
/// and `A100` (`zvl1024b`, `128B` + `8x16` `XSMTVDotII` `IME`) execution without per-token
/// nibble extraction.
#[derive(Debug, Clone)]
pub struct RepackedWeight {
    rows: usize,
    cols: usize,
    k_padded: usize,
    quant: QuantType,
    storage: RepackedStorage,
}

/// Canonical type alias for [`RepackedWeight`].
pub type AlignedQuantMatrix = RepackedWeight;

/// Repacks a `Q5_0` weight tensor into `8x16` `i8` `A100` `XSMTVDotII` / `X100` `RVV` tiles.
///
/// # Errors
///
/// Returns [`K3Error::CapabilityMismatch`] if `weights.quant() != QuantType::Q5_0`, or
/// [`K3Error::TensorBoundsError`] if dimensions or backing byte bounds do not match.
pub fn repack_q5_0_to_ime_8x16(weights: &TensorView<'_>) -> Result<RepackedWeight, K3Error> {
    RepackedWeight::repack_q5_0_to_ime_8x16(weights)
}

#[inline]
fn unpack_k4_scales_mins(scales_12: &[u8]) -> ([u8; 8], [u8; 8]) {
    let mut sc = [0_u8; 8];
    let mut m = [0_u8; 8];
    for j in 0..4 {
        sc[j] = scales_12[j] & 0x3F;
        m[j] = scales_12[j + 4] & 0x3F;
    }
    for j in 4..8 {
        sc[j] = (scales_12[j + 4] & 0x0F) | ((scales_12[j - 4] >> 6) << 4);
        m[j] = (scales_12[j + 4] >> 4) | ((scales_12[j] >> 6) << 4);
    }
    (sc, m)
}

fn validate_matrix_shape(
    weights: &TensorView<'_>,
    rows: usize,
    cols: usize,
) -> Result<(), K3Error> {
    let shape = weights.shape();
    match shape.len() {
        1 => {
            if rows.checked_mul(cols) != Some(shape[0]) {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: weights.name().to_owned(),
                    offset: weights.offset(),
                    byte_len: weights.byte_len(),
                    buffer_len: weights.as_bytes().len(),
                    reason: format!(
                        "1D tensor length {} does not match rows ({rows}) * cols ({cols})",
                        shape[0]
                    ),
                });
            }
        }
        2 => {
            let (d0, d1) = (shape[0], shape[1]);
            if !(d0 == rows && d1 == cols || d0 == cols && d1 == rows) {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: weights.name().to_owned(),
                    offset: weights.offset(),
                    byte_len: weights.byte_len(),
                    buffer_len: weights.as_bytes().len(),
                    reason: format!(
                        "tensor shape [{d0}, {d1}] does not match requested rows={rows}, cols={cols}"
                    ),
                });
            }
        }
        other => {
            return Err(K3Error::TensorBoundsError {
                tensor_name: weights.name().to_owned(),
                offset: weights.offset(),
                byte_len: weights.byte_len(),
                buffer_len: weights.as_bytes().len(),
                reason: format!("expected 1D or 2D weight tensor, got {other}D"),
            });
        }
    }
    Ok(())
}

fn validate_matrix_view(
    weights: &TensorView<'_>,
    rows: usize,
    cols: usize,
) -> Result<usize, K3Error> {
    if rows == 0 || cols == 0 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: format!("repack dimensions must be non-zero, got [{rows}, {cols}]"),
        });
    }

    let quant = weights.quant();
    let block_elems = quant.block_elements();
    if cols % block_elems != 0 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: format!(
                "columns ({cols}) must be a multiple of {quant:?} block size ({block_elems})"
            ),
        });
    }

    validate_matrix_shape(weights, rows, cols)?;

    let row_bytes = (cols / block_elems)
        .checked_mul(quant.block_bytes())
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: "row byte stride overflow".to_owned(),
        })?;
    let payload_bytes = rows
        .checked_mul(row_bytes)
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: "matrix byte length overflow".to_owned(),
        })?;
    let total_bytes = payload_bytes
        .checked_add(quant.scale_trailer_bytes())
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: "matrix byte length overflow with scale trailer".to_owned(),
        })?;

    if weights.as_bytes().len() < total_bytes {
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: format!(
                "backing bytes {} < required {total_bytes}",
                weights.as_bytes().len()
            ),
        });
    }

    Ok(row_bytes)
}

/// Tile-grid geometry describing the buffers required by [`repack_into_tile_range`] and
/// [`repack_q4_k_m_parallel_slices`] for zero-lock parallel load-time repacking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockScaledTileLayout {
    quant: QuantType,
    rows: usize,
    cols: usize,
    row_bytes: usize,
    k_padded: usize,
    n_tiles: usize,
    k_chunks: usize,
    chunks_per_scale: usize,
    scale_groups: usize,
    has_mins: bool,
}

impl BlockScaledTileLayout {
    /// Computes the dual-cluster (`X100` + `A100`) tile layout for `weights` interpreted as
    /// `[rows, cols]`.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::CapabilityMismatch`] if `weights.quant()` is dense (`F32`/`F16`/`BF16`)
    /// or unsupported (`Q8_1`/`MXFP4`), or [`K3Error::TensorBoundsError`] if geometry or backing
    /// byte bounds do not match.
    pub fn for_tensor(weights: &TensorView<'_>, rows: usize, cols: usize) -> Result<Self, K3Error> {
        let quant = weights.quant();
        match quant {
            QuantType::Q8_0
            | QuantType::Q4_0
            | QuantType::Q4_1
            | QuantType::Q5_0
            | QuantType::Q5_1
            | QuantType::Q2_K
            | QuantType::Q3_K
            | QuantType::Q4_K_M
            | QuantType::Q5_K
            | QuantType::Q6_K
            | QuantType::IQ4_NL
            | QuantType::TQ1_0
            | QuantType::TQ2_0
            | QuantType::I2_S => {}
            QuantType::F32
            | QuantType::F16
            | QuantType::BF16
            | QuantType::Q8_1
            | QuantType::MXFP4 => {
                return Err(K3Error::unsupported_tensor_quant(
                    "block_scaled_layout",
                    weights.name(),
                    quant,
                ));
            }
        }
        let row_bytes = validate_matrix_view(weights, rows, cols)?;
        Ok(Self::from_validated_dims(quant, rows, cols, row_bytes))
    }

    const fn from_validated_dims(
        quant: QuantType,
        rows: usize,
        cols: usize,
        row_bytes: usize,
    ) -> Self {
        let k_padded = cols.next_multiple_of(ACT_Q8_GROUP_ELEMENTS);
        let n_tiles = tile_count(rows);
        let k_chunks = k_padded / TILE_K;
        let chunks_per_scale =
            if matches!(quant, QuantType::Q6_K | QuantType::Q3_K | QuantType::Q2_K) {
                1
            } else {
                2
            };
        let scale_groups = k_chunks / chunks_per_scale;
        let has_mins = matches!(
            quant,
            QuantType::Q4_K_M
                | QuantType::Q5_K
                | QuantType::Q4_1
                | QuantType::Q5_1
                | QuantType::Q2_K
        );
        Self {
            quant,
            rows,
            cols,
            row_bytes,
            k_padded,
            n_tiles,
            k_chunks,
            chunks_per_scale,
            scale_groups,
            has_mins,
        }
    }

    /// Source quantization format.
    #[must_use]
    pub const fn quant(self) -> QuantType {
        self.quant
    }

    /// Logical weight row count (`N`).
    #[must_use]
    pub const fn rows(self) -> usize {
        self.rows
    }

    /// Logical weight column count (`K`).
    #[must_use]
    pub const fn cols(self) -> usize {
        self.cols
    }

    /// Number of `8`-row output tiles (`rows.div_ceil(8)`).
    #[must_use]
    pub const fn n_tiles(self) -> usize {
        self.n_tiles
    }

    /// Number of `16`-element `K` chunks per output row tile (`k_padded / 16`).
    #[must_use]
    pub const fn k_chunks(self) -> usize {
        self.k_chunks
    }

    /// Number of scale groups along `K` per output row tile.
    #[must_use]
    pub const fn scale_groups(self) -> usize {
        self.scale_groups
    }

    /// Whether the quantization format stores asymmetric block-min offsets (`Q4_K_M`, `Q5_K`,
    /// `Q4_1`, `Q5_1`, `Q2_K`).
    #[must_use]
    pub const fn has_mins(self) -> bool {
        self.has_mins
    }

    /// Total [`Aligned128`] tile blocks across all `n_tiles`.
    #[must_use]
    pub const fn total_tiles(self) -> usize {
        self.n_tiles * self.k_chunks
    }

    /// Total `f32` scale elements across all `n_tiles`.
    #[must_use]
    pub const fn total_scales(self) -> usize {
        self.n_tiles * self.scale_groups * TILE_N
    }

    /// Total `f32` block-min elements across all `n_tiles` (`0` when [`Self::has_mins`] is `false`).
    #[must_use]
    pub const fn total_mins(self) -> usize {
        if self.has_mins {
            self.total_scales()
        } else {
            0
        }
    }

    /// Number of `f32` scale (and optional min) elements owned by `tile_count` consecutive tiles.
    #[must_use]
    pub const fn scales_for_tiles(self, count: usize) -> usize {
        count * self.scale_groups * TILE_N
    }
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[inline]
fn unpack_q4_0_32_rvv(qs: &[u8; 16], dst: &mut [i8; 32]) {
    // SAFETY: `qs` holds 16 bytes and `dst` holds 32 `i8` bytes; `vsetivli zero, 16, e8, m1`
    // guarantees `vl = 16 <= VLMAX` on every RVV 1.0 CPU (`VLEN >= 128`).
    unsafe {
        core::arch::asm!(
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vle8.v v0, ({qs})",
            "vand.vi v1, v0, 0x0F",
            "vadd.vi v1, v1, -8",
            "vsrl.vi v2, v0, 4",
            "vadd.vi v2, v2, -8",
            "vse8.v v1, ({dst_lo})",
            "vse8.v v2, ({dst_hi})",
            qs = in(reg) qs.as_ptr(),
            dst_lo = in(reg) dst.as_mut_ptr(),
            dst_hi = in(reg) dst.as_mut_ptr().add(16),
            out("v0") _,
            out("v1") _,
            out("v2") _,
            options(nostack),
        );
    }
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[inline]
fn unpack_q4_k_chunk64_rvv(qs: &[u8; 32], dst: &mut [i8; 64]) {
    // SAFETY: `qs` holds 32 bytes (`[low 16B | high 16B]`) and `dst` holds 64 `i8` bytes;
    // `vl = 16, e8, m1` fits inside `VLEN >= 128`.
    unsafe {
        core::arch::asm!(
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vle8.v v0, ({qs0})",
            "vle8.v v3, ({qs1})",
            "vand.vi v1, v0, 0x0F",
            "vsrl.vi v2, v0, 4",
            "vand.vi v4, v3, 0x0F",
            "vsrl.vi v5, v3, 4",
            "vse8.v v1, ({d0})",
            "vse8.v v4, ({d1})",
            "vse8.v v2, ({d2})",
            "vse8.v v5, ({d3})",
            qs0 = in(reg) qs.as_ptr(),
            qs1 = in(reg) qs.as_ptr().add(16),
            d0 = in(reg) dst.as_mut_ptr(),
            d1 = in(reg) dst.as_mut_ptr().add(16),
            d2 = in(reg) dst.as_mut_ptr().add(32),
            d3 = in(reg) dst.as_mut_ptr().add(48),
            out("v0") _,
            out("v1") _,
            out("v2") _,
            out("v3") _,
            out("v4") _,
            out("v5") _,
            options(nostack),
        );
    }
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[inline]
fn copy_i8_chunks_to_tile_rvv(
    tile_base_i8: &mut [i8],
    start_ch: usize,
    row_in_tile: usize,
    src_i8: &[i8],
) {
    let n_chunks = src_i8.len() / TILE_K;
    if n_chunks == 0 {
        return;
    }
    // SAFETY: caller supplies `src_i8` covering `n_chunks * 16` bytes and `tile_base_i8` covering
    // `(start_ch + n_chunks) * TILE_BYTES`; `vl = 16, e8, m1` fits inside `VLEN >= 128`.
    unsafe {
        let ps = src_i8.as_ptr();
        let pd = tile_base_i8
            .as_mut_ptr()
            .add(start_ch * TILE_BYTES + row_in_tile * TILE_K);
        core::arch::asm!(
            "vsetivli zero, 16, e8, m1, ta, ma",
            "2:",
            "vle8.v v0, ({ps})",
            "vse8.v v0, ({pd})",
            "addi {ps}, {ps}, 16",
            "addi {pd}, {pd}, 128",
            "addi {rem}, {rem}, -1",
            "bnez {rem}, 2b",
            ps = inout(reg) ps => _,
            pd = inout(reg) pd => _,
            rem = inout(reg) n_chunks => _,
            out("v0") _,
            options(nostack),
        );
    }
}

#[inline]
fn write_row_block_chunks_to_tile_i8(
    tile_base_i8: &mut [i8],
    start_ch: usize,
    row_in_tile: usize,
    block_i8: &[i8],
) {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        copy_i8_chunks_to_tile_rvv(tile_base_i8, start_ch, row_in_tile, block_i8);
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        let n_chunks = block_i8.len() / TILE_K;
        for ch in 0..n_chunks {
            let dst_off = (start_ch + ch) * TILE_BYTES + row_in_tile * TILE_K;
            let src_off = ch * TILE_K;
            tile_base_i8[dst_off..dst_off + TILE_K]
                .copy_from_slice(&block_i8[src_off..src_off + TILE_K]);
        }
    }
}

/// Repacks a contiguous tile slice (`tile_range` of [`BlockScaledTileLayout::n_tiles`]) into
/// caller-supplied disjoint mutable slices with zero heap allocations (`#89`).
///
/// # Errors
///
/// Returns [`K3Error::CapabilityMismatch`] on dense/unsupported formats, or
/// [`K3Error::TensorBoundsError`] if `tile_range`, `weights`, or output slice lengths mismatch.
pub fn repack_into_tile_range(
    weights: &TensorView<'_>,
    rows: usize,
    cols: usize,
    tile_range: Range<usize>,
    out_tiles: &mut [Aligned128],
    out_scales: &mut [f32],
    out_mins: Option<&mut [f32]>,
) -> Result<(), K3Error> {
    let layout = BlockScaledTileLayout::for_tensor(weights, rows, cols)?;
    if tile_range.start > tile_range.end || tile_range.end > layout.n_tiles {
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: tile_range.start,
            byte_len: tile_range.len(),
            buffer_len: layout.n_tiles,
            reason: format!(
                "tile_range {}..{} exceeds n_tiles ({})",
                tile_range.start, tile_range.end, layout.n_tiles
            ),
        });
    }
    let sub_tiles = tile_range.len();
    let expected_tiles = sub_tiles * layout.k_chunks;
    let expected_scales = sub_tiles * layout.scale_groups * TILE_N;
    if out_tiles.len() != expected_tiles || out_scales.len() != expected_scales {
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: tile_range.start,
            byte_len: out_tiles.len(),
            buffer_len: expected_tiles,
            reason: format!(
                "tile/scale slice lengths ({}, {}) != expected ({expected_tiles}, {expected_scales})",
                out_tiles.len(),
                out_scales.len()
            ),
        });
    }
    let mins_slice = if layout.has_mins {
        let Some(m) = out_mins else {
            return Err(K3Error::TensorBoundsError {
                tensor_name: weights.name().to_owned(),
                offset: tile_range.start,
                byte_len: 0,
                buffer_len: expected_scales,
                reason: format!("{:?} requires out_mins slice", layout.quant),
            });
        };
        if m.len() != expected_scales {
            return Err(K3Error::TensorBoundsError {
                tensor_name: weights.name().to_owned(),
                offset: tile_range.start,
                byte_len: m.len(),
                buffer_len: expected_scales,
                reason: format!("out_mins length {} != expected {expected_scales}", m.len()),
            });
        }
        Some(m)
    } else {
        None
    };

    repack_block_scaled_range_into_i8(
        layout,
        weights.as_bytes(),
        tile_range,
        Aligned128::slice_as_i8_mut(out_tiles),
        out_scales,
        mins_slice,
    );
    Ok(())
}

/// Zero-allocation row-tile slice repacking helper specialized for `Q4_K_M` operating on raw
/// `&mut [i8]` tile slices (`tile_range.len() * k_chunks * TILE_BYTES` bytes) (`#89`).
///
/// # Errors
///
/// Returns [`K3Error::CapabilityMismatch`] if `weights.quant() != QuantType::Q4_K_M`, or
/// [`K3Error::TensorBoundsError`] if dimensions, `tile_range`, or slice lengths mismatch.
pub fn repack_q4_k_m_parallel_slices(
    weights: &TensorView<'_>,
    rows: usize,
    cols: usize,
    tile_range: Range<usize>,
    out_tiles_i8: &mut [i8],
    out_scales: &mut [f32],
    out_mins: &mut [f32],
) -> Result<(), K3Error> {
    if weights.quant() != QuantType::Q4_K_M {
        return Err(K3Error::unsupported_tensor_quant(
            "repack_q4_k_m_parallel_slices",
            weights.name(),
            weights.quant(),
        ));
    }
    let layout = BlockScaledTileLayout::for_tensor(weights, rows, cols)?;
    if tile_range.start > tile_range.end || tile_range.end > layout.n_tiles {
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: tile_range.start,
            byte_len: tile_range.len(),
            buffer_len: layout.n_tiles,
            reason: format!(
                "tile_range {}..{} exceeds n_tiles ({})",
                tile_range.start, tile_range.end, layout.n_tiles
            ),
        });
    }
    let sub_tiles = tile_range.len();
    let expected_tile_bytes = sub_tiles * layout.k_chunks * TILE_BYTES;
    let expected_scales = sub_tiles * layout.scale_groups * TILE_N;
    if out_tiles_i8.len() != expected_tile_bytes
        || out_scales.len() != expected_scales
        || out_mins.len() != expected_scales
    {
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: tile_range.start,
            byte_len: out_tiles_i8.len(),
            buffer_len: expected_tile_bytes,
            reason: format!(
                "parallel slice lengths ({}, {}, {}) != expected ({expected_tile_bytes}, {expected_scales}, {expected_scales})",
                out_tiles_i8.len(),
                out_scales.len(),
                out_mins.len()
            ),
        });
    }

    repack_block_scaled_range_into_i8(
        layout,
        weights.as_bytes(),
        tile_range,
        out_tiles_i8,
        out_scales,
        Some(out_mins),
    );
    Ok(())
}

#[expect(
    clippy::too_many_lines,
    reason = "covers all 14 block-scaled quant formats with zero heap allocation per tile"
)]
fn repack_block_scaled_range_into_i8(
    layout: BlockScaledTileLayout,
    raw: &[u8],
    tile_range: Range<usize>,
    out_tile_i8: &mut [i8],
    out_scales: &mut [f32],
    mut out_mins: Option<&mut [f32]>,
) {
    out_tile_i8.fill(0);
    out_scales.fill(0.0);
    if let Some(mins_buf) = out_mins.as_deref_mut() {
        mins_buf.fill(0.0);
    }
    if tile_range.is_empty() {
        return;
    }

    let i2_s_scale = if layout.quant == QuantType::I2_S {
        read_i2_s_trailer_scale(&raw[layout.rows * layout.row_bytes..])
    } else {
        0.0_f32
    };

    let tile_i8_stride = layout.k_chunks * TILE_BYTES;
    let sc_stride = layout.scale_groups * TILE_N;

    for (local_nt, nt) in tile_range.enumerate() {
        let tile_base_i8 =
            &mut out_tile_i8[local_nt * tile_i8_stride..(local_nt + 1) * tile_i8_stride];
        let scales = &mut out_scales[local_nt * sc_stride..(local_nt + 1) * sc_stride];
        let mut mins_tile = out_mins
            .as_deref_mut()
            .map(|m| &mut m[local_nt * sc_stride..(local_nt + 1) * sc_stride]);

        for j in 0..TILE_N {
            let row = nt * TILE_N + j;
            if row >= layout.rows {
                break;
            }
            let row_raw = &raw[row * layout.row_bytes..(row + 1) * layout.row_bytes];

            match layout.quant {
                QuantType::Q8_0 => {
                    let n_blks = layout.cols / Q8_0_BLOCK_ELEMENTS;
                    let mut blk_q = [0_i8; 32];
                    for b in 0..n_blks {
                        let blk = &row_raw[b * Q8_0_BLOCK_BYTES..(b + 1) * Q8_0_BLOCK_BYTES];
                        let d = f16_to_f32_finite(u16::from_le_bytes([blk[0], blk[1]]));
                        scales[b * TILE_N + j] = d;
                        if d != 0.0 {
                            #[expect(clippy::cast_possible_wrap, reason = "u8 reinterpreted as i8")]
                            for i in 0..32 {
                                blk_q[i] = blk[2 + i] as i8;
                            }
                            write_row_block_chunks_to_tile_i8(tile_base_i8, b * 2, j, &blk_q);
                        }
                    }
                }
                QuantType::Q4_0 => {
                    let n_blks = layout.cols / Q4_0_BLOCK_ELEMENTS;
                    let mut blk_q = [0_i8; 32];
                    for b in 0..n_blks {
                        let blk = &row_raw[b * Q4_0_BLOCK_BYTES..(b + 1) * Q4_0_BLOCK_BYTES];
                        let d = f16_to_f32_finite(u16::from_le_bytes([blk[0], blk[1]]));
                        scales[b * TILE_N + j] = d;
                        if d != 0.0 {
                            #[cfg(all(target_arch = "riscv64", not(miri)))]
                            {
                                let qs: &[u8; 16] =
                                    blk[2..18].try_into().expect("16-byte Q4_0 nibbles");
                                unpack_q4_0_32_rvv(qs, &mut blk_q);
                            }
                            #[cfg(any(not(target_arch = "riscv64"), miri))]
                            {
                                for i in 0..16 {
                                    let byte = blk[2 + i];
                                    #[expect(
                                        clippy::cast_possible_wrap,
                                        reason = "nibble - 8 fits in i8"
                                    )]
                                    {
                                        blk_q[i] = ((byte & 0x0F) as i8) - 8;
                                        blk_q[i + 16] = (((byte >> 4) & 0x0F) as i8) - 8;
                                    }
                                }
                            }
                            write_row_block_chunks_to_tile_i8(tile_base_i8, b * 2, j, &blk_q);
                        }
                    }
                }
                QuantType::Q4_1 => {
                    let n_blks = layout.cols / Q4_1_BLOCK_ELEMENTS;
                    let mins_vec = mins_tile.as_deref_mut().expect("Q4_1 allocates mins");
                    let mut blk_q = [0_i8; Q4_1_BLOCK_ELEMENTS];
                    for b in 0..n_blks {
                        let blk: &[u8; Q4_1_BLOCK_BYTES] = row_raw
                            [b * Q4_1_BLOCK_BYTES..(b + 1) * Q4_1_BLOCK_BYTES]
                            .try_into()
                            .expect("20-byte Q4_1 block");
                        let (d, m) = unpack_q4_1_block_i8(blk, &mut blk_q);
                        let idx = b * TILE_N + j;
                        scales[idx] = d;
                        mins_vec[idx] = if m == 0.0 { 0.0 } else { -m };
                        write_row_block_chunks_to_tile_i8(tile_base_i8, b * 2, j, &blk_q);
                    }
                }
                QuantType::Q5_0 => {
                    let n_blks = layout.cols / Q5_0_BLOCK_ELEMENTS;
                    let mut blk_q = [0_i8; Q5_0_BLOCK_ELEMENTS];
                    for b in 0..n_blks {
                        let blk: &[u8; Q5_0_BLOCK_BYTES] = row_raw
                            [b * Q5_0_BLOCK_BYTES..(b + 1) * Q5_0_BLOCK_BYTES]
                            .try_into()
                            .expect("22-byte Q5_0 block");
                        let d = unpack_q5_0_block_i8(blk, &mut blk_q);
                        scales[b * TILE_N + j] = d;
                        if d != 0.0 {
                            write_row_block_chunks_to_tile_i8(tile_base_i8, b * 2, j, &blk_q);
                        }
                    }
                }
                QuantType::Q5_1 => {
                    let n_blks = layout.cols / Q5_1_BLOCK_ELEMENTS;
                    let mins_vec = mins_tile.as_deref_mut().expect("Q5_1 allocates mins");
                    let mut blk_q = [0_i8; Q5_1_BLOCK_ELEMENTS];
                    for b in 0..n_blks {
                        let blk: &[u8; Q5_1_BLOCK_BYTES] = row_raw
                            [b * Q5_1_BLOCK_BYTES..(b + 1) * Q5_1_BLOCK_BYTES]
                            .try_into()
                            .expect("24-byte Q5_1 block");
                        let (d, m) = unpack_q5_1_block_i8(blk, &mut blk_q);
                        let idx = b * TILE_N + j;
                        scales[idx] = d;
                        mins_vec[idx] = if m == 0.0 { 0.0 } else { -m };
                        write_row_block_chunks_to_tile_i8(tile_base_i8, b * 2, j, &blk_q);
                    }
                }
                QuantType::IQ4_NL => {
                    let n_blks = layout.cols / IQ4_NL_BLOCK_ELEMENTS;
                    let mut blk_q = [0_i8; IQ4_NL_BLOCK_ELEMENTS];
                    for b in 0..n_blks {
                        let blk: &[u8; IQ4_NL_BLOCK_BYTES] = row_raw
                            [b * IQ4_NL_BLOCK_BYTES..(b + 1) * IQ4_NL_BLOCK_BYTES]
                            .try_into()
                            .expect("18-byte IQ4_NL block");
                        let d = unpack_iq4_nl_block_i8(blk, &mut blk_q);
                        scales[b * TILE_N + j] = d;
                        if d != 0.0 {
                            write_row_block_chunks_to_tile_i8(tile_base_i8, b * 2, j, &blk_q);
                        }
                    }
                }
                QuantType::Q2_K => {
                    let n_sblks = layout.cols / Q2_K_BLOCK_ELEMENTS;
                    let mins_vec = mins_tile.as_deref_mut().expect("Q2_K allocates mins");
                    let mut sb_q = [0_i8; Q2_K_BLOCK_ELEMENTS];
                    let mut sb_sc = [0.0_f32; 16];
                    let mut sb_mn = [0.0_f32; 16];
                    for sb in 0..n_sblks {
                        let blk: &[u8; Q2_K_BLOCK_BYTES] = row_raw
                            [sb * Q2_K_BLOCK_BYTES..(sb + 1) * Q2_K_BLOCK_BYTES]
                            .try_into()
                            .expect("84-byte Q2_K block");
                        unpack_q2_k_block_i8(blk, &mut sb_q, &mut sb_sc, &mut sb_mn);
                        for sub in 0..16 {
                            let idx = (sb * 16 + sub) * TILE_N + j;
                            scales[idx] = sb_sc[sub];
                            mins_vec[idx] = sb_mn[sub];
                        }
                        write_row_block_chunks_to_tile_i8(tile_base_i8, sb * 16, j, &sb_q);
                    }
                }
                QuantType::Q3_K => {
                    let n_sblks = layout.cols / Q3_K_BLOCK_ELEMENTS;
                    let mut sb_q = [0_i8; Q3_K_BLOCK_ELEMENTS];
                    let mut sb_sc = [0.0_f32; 16];
                    for sb in 0..n_sblks {
                        let blk: &[u8; Q3_K_BLOCK_BYTES] = row_raw
                            [sb * Q3_K_BLOCK_BYTES..(sb + 1) * Q3_K_BLOCK_BYTES]
                            .try_into()
                            .expect("110-byte Q3_K block");
                        unpack_q3_k_block_i8(blk, &mut sb_q, &mut sb_sc);
                        for (sub, &sc) in sb_sc.iter().enumerate() {
                            scales[(sb * 16 + sub) * TILE_N + j] = sc;
                        }
                        write_row_block_chunks_to_tile_i8(tile_base_i8, sb * 16, j, &sb_q);
                    }
                }
                QuantType::Q4_K_M => {
                    let n_sblks = layout.cols / Q4_K_M_BLOCK_ELEMENTS;
                    let mins_vec = mins_tile.as_deref_mut().expect("Q4_K_M allocates mins");
                    let mut chunk_q = [0_i8; 64];
                    for sb in 0..n_sblks {
                        let blk = &row_raw[sb * Q4_K_M_BLOCK_BYTES..(sb + 1) * Q4_K_M_BLOCK_BYTES];
                        let d = f16_to_f32_finite(u16::from_le_bytes([blk[0], blk[1]]));
                        let dmin = f16_to_f32_finite(u16::from_le_bytes([blk[2], blk[3]]));
                        let (sc, m) = unpack_k4_scales_mins(&blk[4..16]);
                        let qs = &blk[16..144];
                        for chunk in 0..4 {
                            let is = chunk * 2;
                            let sg0 = sb * 8 + is;
                            let sg1 = sg0 + 1;
                            scales[sg0 * TILE_N + j] = d * f32::from(sc[is]);
                            mins_vec[sg0 * TILE_N + j] = dmin * f32::from(m[is]);
                            scales[sg1 * TILE_N + j] = d * f32::from(sc[is + 1]);
                            mins_vec[sg1 * TILE_N + j] = dmin * f32::from(m[is + 1]);

                            let q_off = chunk * 32;
                            #[cfg(all(target_arch = "riscv64", not(miri)))]
                            {
                                let q_slice: &[u8; 32] = qs[q_off..q_off + 32]
                                    .try_into()
                                    .expect("32-byte Q4_K chunk");
                                unpack_q4_k_chunk64_rvv(q_slice, &mut chunk_q);
                            }
                            #[cfg(any(not(target_arch = "riscv64"), miri))]
                            {
                                #[expect(
                                    clippy::cast_possible_wrap,
                                    reason = "unsigned 4-bit in 0..=15"
                                )]
                                for l in 0..32 {
                                    chunk_q[l] = (qs[q_off + l] & 0x0F) as i8;
                                    chunk_q[32 + l] = ((qs[q_off + l] >> 4) & 0x0F) as i8;
                                }
                            }
                            write_row_block_chunks_to_tile_i8(
                                tile_base_i8,
                                sb * 16 + chunk * 4,
                                j,
                                &chunk_q,
                            );
                        }
                    }
                }
                QuantType::Q5_K => {
                    let n_sblks = layout.cols / Q5_K_BLOCK_ELEMENTS;
                    let mins_vec = mins_tile.as_deref_mut().expect("Q5_K allocates mins");
                    let mut chunk_q = [0_i8; 64];
                    for sb in 0..n_sblks {
                        let blk = &row_raw[sb * Q5_K_BLOCK_BYTES..(sb + 1) * Q5_K_BLOCK_BYTES];
                        let d = f16_to_f32_finite(u16::from_le_bytes([blk[0], blk[1]]));
                        let dmin = f16_to_f32_finite(u16::from_le_bytes([blk[2], blk[3]]));
                        let (sc, m) = unpack_k4_scales_mins(&blk[4..16]);
                        let qh = &blk[16..48];
                        let qs = &blk[48..176];
                        let mut u1 = 1_u8;
                        let mut u2 = 2_u8;
                        for chunk in 0..4 {
                            let is = chunk * 2;
                            let sg0 = sb * 8 + is;
                            let sg1 = sg0 + 1;
                            scales[sg0 * TILE_N + j] = d * f32::from(sc[is]);
                            mins_vec[sg0 * TILE_N + j] = dmin * f32::from(m[is]);
                            scales[sg1 * TILE_N + j] = d * f32::from(sc[is + 1]);
                            mins_vec[sg1 * TILE_N + j] = dmin * f32::from(m[is + 1]);

                            let q_off = chunk * 32;
                            #[expect(
                                clippy::cast_possible_wrap,
                                reason = "unsigned 5-bit in 0..=31"
                            )]
                            for l in 0..32 {
                                let hb1 = if (qh[l] & u1) != 0 { 16_u8 } else { 0_u8 };
                                let hb2 = if (qh[l] & u2) != 0 { 16_u8 } else { 0_u8 };
                                chunk_q[l] = ((qs[q_off + l] & 0x0F) + hb1) as i8;
                                chunk_q[32 + l] = ((qs[q_off + l] >> 4) + hb2) as i8;
                            }
                            write_row_block_chunks_to_tile_i8(
                                tile_base_i8,
                                sb * 16 + chunk * 4,
                                j,
                                &chunk_q,
                            );
                            u1 <<= 2;
                            u2 <<= 2;
                        }
                    }
                }
                QuantType::Q6_K => {
                    let n_sblks = layout.cols / Q6_K_BLOCK_ELEMENTS;
                    let mut sb_q = [0_i8; 256];
                    for sb in 0..n_sblks {
                        let blk = &row_raw[sb * Q6_K_BLOCK_BYTES..(sb + 1) * Q6_K_BLOCK_BYTES];
                        let d = f16_to_f32_finite(u16::from_le_bytes([blk[208], blk[209]]));
                        let ql = &blk[0..128];
                        let qh = &blk[128..192];
                        let sc_bytes = &blk[192..208];
                        for sub in 0..16 {
                            #[expect(clippy::cast_possible_wrap, reason = "signed 8-bit scale")]
                            let sc_val = f32::from(sc_bytes[sub] as i8);
                            scales[(sb * 16 + sub) * TILE_N + j] = d * sc_val;
                        }
                        if d != 0.0 {
                            for half_idx in 0..2 {
                                let n = half_idx * 128;
                                let low_off = half_idx * 64;
                                let high_off = half_idx * 32;
                                #[expect(clippy::cast_possible_wrap, reason = "6-bit quant - 32")]
                                for l in 0..32 {
                                    sb_q[n + l] = (((ql[low_off + l] & 0x0F)
                                        | ((qh[high_off + l] & 0x03) << 4))
                                        as i8)
                                        - 32;
                                    sb_q[n + l + 32] = (((ql[low_off + l + 32] & 0x0F)
                                        | (((qh[high_off + l] >> 2) & 0x03) << 4))
                                        as i8)
                                        - 32;
                                    sb_q[n + l + 64] = ((((ql[low_off + l] >> 4) & 0x0F)
                                        | (((qh[high_off + l] >> 4) & 0x03) << 4))
                                        as i8)
                                        - 32;
                                    sb_q[n + l + 96] = ((((ql[low_off + l + 32] >> 4) & 0x0F)
                                        | (((qh[high_off + l] >> 6) & 0x03) << 4))
                                        as i8)
                                        - 32;
                                }
                            }
                            write_row_block_chunks_to_tile_i8(tile_base_i8, sb * 16, j, &sb_q);
                        }
                    }
                }
                QuantType::TQ1_0 => {
                    let n_sblks = layout.cols / TQ1_0_BLOCK_ELEMENTS;
                    let mut trits = [0_i8; TQ1_0_BLOCK_ELEMENTS];
                    for sb in 0..n_sblks {
                        let blk: &[u8; TQ1_0_BLOCK_BYTES] = row_raw
                            [sb * TQ1_0_BLOCK_BYTES..(sb + 1) * TQ1_0_BLOCK_BYTES]
                            .try_into()
                            .expect("54-byte TQ1_0 block");
                        let d = unpack_tq1_0_block_i8(blk, &mut trits);
                        for sub in 0..8 {
                            scales[(sb * 8 + sub) * TILE_N + j] = d;
                        }
                        write_row_block_chunks_to_tile_i8(tile_base_i8, sb * 16, j, &trits);
                    }
                }
                QuantType::TQ2_0 => {
                    let n_sblks = layout.cols / TQ2_0_BLOCK_ELEMENTS;
                    let mut trits = [0_i8; TQ2_0_BLOCK_ELEMENTS];
                    for sb in 0..n_sblks {
                        let blk: &[u8; TQ2_0_BLOCK_BYTES] = row_raw
                            [sb * TQ2_0_BLOCK_BYTES..(sb + 1) * TQ2_0_BLOCK_BYTES]
                            .try_into()
                            .expect("66-byte TQ2_0 block");
                        let d = unpack_tq2_0_block_i8(blk, &mut trits);
                        for sub in 0..8 {
                            scales[(sb * 8 + sub) * TILE_N + j] = d;
                        }
                        write_row_block_chunks_to_tile_i8(tile_base_i8, sb * 16, j, &trits);
                    }
                }
                QuantType::I2_S => {
                    let n_blks = layout.cols / I2_S_BLOCK_ELEMENTS;
                    let mut trits = [0_i8; I2_S_BLOCK_ELEMENTS];
                    for b in 0..n_blks {
                        let blk: &[u8; I2_S_BLOCK_BYTES] = row_raw
                            [b * I2_S_BLOCK_BYTES..(b + 1) * I2_S_BLOCK_BYTES]
                            .try_into()
                            .expect("32-byte I2_S block");
                        unpack_i2_s_block_i8(blk, &mut trits);
                        for sub in 0..4 {
                            scales[(b * 4 + sub) * TILE_N + j] = i2_s_scale;
                        }
                        if i2_s_scale != 0.0 {
                            write_row_block_chunks_to_tile_i8(tile_base_i8, b * 8, j, &trits);
                        }
                    }
                }
                QuantType::F32
                | QuantType::F16
                | QuantType::BF16
                | QuantType::Q8_1
                | QuantType::MXFP4 => {
                    unreachable!()
                }
            }
        }
    }
}

fn repack_block_scaled_i8(
    raw: &[u8],
    quant: QuantType,
    rows: usize,
    cols: usize,
    row_bytes: usize,
) -> RepackedWeight {
    let layout = BlockScaledTileLayout::from_validated_dims(quant, rows, cols, row_bytes);
    let mut tiles = vec![Aligned128::zeroed(); layout.total_tiles()];
    let mut scales = vec![0.0_f32; layout.total_scales()];
    let mut mins = layout.has_mins.then(|| vec![0.0_f32; layout.total_mins()]);
    repack_block_scaled_range_into_i8(
        layout,
        raw,
        0..layout.n_tiles,
        Aligned128::slice_as_i8_mut(&mut tiles),
        &mut scales,
        mins.as_deref_mut(),
    );
    RepackedWeight {
        rows: layout.rows,
        cols: layout.cols,
        k_padded: layout.k_padded,
        quant: layout.quant,
        storage: RepackedStorage::BlockScaledI8 {
            n_tiles: layout.n_tiles,
            k_chunks: layout.k_chunks,
            chunks_per_scale: layout.chunks_per_scale,
            scale_groups: layout.scale_groups,
            tiles,
            scales,
            mins,
        },
    }
}

fn repack_dense_f32_bytes(
    raw: &[u8],
    rows: usize,
    cols: usize,
    row_bytes: usize,
) -> RepackedWeight {
    let blocks_per_row = cols.div_ceil(F32_PER_ALIGNED128);
    let k_padded = blocks_per_row * F32_PER_ALIGNED128;
    let mut data = vec![Aligned128::zeroed(); rows * blocks_per_row];
    for r in 0..rows {
        let row_raw = &raw[r * row_bytes..(r + 1) * row_bytes];
        let dst =
            Aligned128::slice_as_f32_mut(&mut data[r * blocks_per_row..(r + 1) * blocks_per_row]);
        for (c, slot) in dst.iter_mut().enumerate().take(cols) {
            let off = c * 4;
            let val = f32::from_le_bytes([
                row_raw[off],
                row_raw[off + 1],
                row_raw[off + 2],
                row_raw[off + 3],
            ]);
            *slot = if val.is_finite() { val } else { 0.0_f32 };
        }
    }
    RepackedWeight {
        rows,
        cols,
        k_padded,
        quant: QuantType::F32,
        storage: RepackedStorage::DenseF32 {
            blocks_per_row,
            data,
        },
    }
}

fn repack_half_to_dense_f32(
    raw: &[u8],
    quant: QuantType,
    is_bf16: bool,
    rows: usize,
    cols: usize,
    row_bytes: usize,
) -> RepackedWeight {
    let blocks_per_row = cols.div_ceil(F32_PER_ALIGNED128);
    let k_padded = blocks_per_row * F32_PER_ALIGNED128;
    let mut data = vec![Aligned128::zeroed(); rows * blocks_per_row];
    for r in 0..rows {
        let row_raw = &raw[r * row_bytes..(r + 1) * row_bytes];
        let dst =
            Aligned128::slice_as_f32_mut(&mut data[r * blocks_per_row..(r + 1) * blocks_per_row]);
        for (c, slot) in dst.iter_mut().enumerate().take(cols) {
            let off = c * 2;
            let bits = u16::from_le_bytes([row_raw[off], row_raw[off + 1]]);
            *slot = if is_bf16 {
                bf16_to_f32_finite(bits)
            } else {
                f16_to_f32_finite(bits)
            };
        }
    }
    RepackedWeight {
        rows,
        cols,
        k_padded,
        quant,
        storage: RepackedStorage::DenseF32 {
            blocks_per_row,
            data,
        },
    }
}

fn repack_half_compact(
    raw: &[u8],
    quant: QuantType,
    is_bf16: bool,
    rows: usize,
    cols: usize,
    row_bytes: usize,
) -> RepackedWeight {
    let blocks_per_row = cols.div_ceil(U16_PER_ALIGNED128);
    let k_padded = blocks_per_row * U16_PER_ALIGNED128;
    let mut data = vec![Aligned128::zeroed(); rows * blocks_per_row];
    for r in 0..rows {
        let row_raw = &raw[r * row_bytes..(r + 1) * row_bytes];
        let dst =
            Aligned128::slice_as_u16_mut(&mut data[r * blocks_per_row..(r + 1) * blocks_per_row]);
        for (c, slot) in dst.iter_mut().enumerate().take(cols) {
            let off = c * 2;
            *slot = u16::from_le_bytes([row_raw[off], row_raw[off + 1]]);
        }
    }
    RepackedWeight {
        rows,
        cols,
        k_padded,
        quant,
        storage: RepackedStorage::DenseHalf {
            is_bf16,
            blocks_per_row,
            data,
        },
    }
}

impl RepackedWeight {
    /// Repacks a 2D [`TensorView`] by inferring `(rows, cols)` from its GGUF/row-major shape.
    ///
    /// For unambiguous square or custom orientations, prefer [`Self::from_tensor`].
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] if the tensor is not 2D or its byte buffer is malformed.
    pub fn repack(weights: &TensorView<'_>) -> Result<Self, K3Error> {
        let shape = weights.shape();
        if shape.len() != 2 {
            return Err(K3Error::TensorBoundsError {
                tensor_name: weights.name().to_owned(),
                offset: weights.offset(),
                byte_len: weights.byte_len(),
                buffer_len: weights.as_bytes().len(),
                reason: format!(
                    "RepackedWeight::repack requires a 2D tensor, got {}D",
                    shape.len()
                ),
            });
        }
        let (d0, d1) = (shape[0], shape[1]);
        let blk = weights.quant().block_elements();
        let (rows, cols) = if d0 % blk == 0 && d1 % blk != 0 {
            (d1, d0)
        } else if d1 % blk == 0 && d0 % blk != 0 {
            (d0, d1)
        } else {
            // Standard GGUF 2D layout: ne[0] = cols (contiguous), ne[1] = rows
            (d1, d0)
        };
        Self::from_tensor(weights, rows, cols)
    }

    /// Repacks `weights` with explicit `(rows, cols)` dimensions (`[N, K]`) into `128-byte`-aligned
    /// dual-cluster (`X100` `256b` + `A100` `1024b` / `IME`) buffers.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] if dimensions or backing byte bounds do not match.
    pub fn from_tensor(
        weights: &TensorView<'_>,
        rows: usize,
        cols: usize,
    ) -> Result<Self, K3Error> {
        Self::from_tensor_with_half_limit(weights, rows, cols, DEFAULT_DENSE_F32_MAX_ELEMENTS)
    }

    /// Repacks `weights` with an explicit element threshold controlling whether `F16`/`BF16`
    /// tensors expand to aligned `f32` (`rows * cols <= dense_f32_max_elems`) or remain in compact
    /// `128B`-aligned `u16` rows.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] if dimensions or backing byte bounds do not match.
    pub fn from_tensor_with_half_limit(
        weights: &TensorView<'_>,
        rows: usize,
        cols: usize,
        dense_f32_max_elems: usize,
    ) -> Result<Self, K3Error> {
        let row_bytes = validate_matrix_view(weights, rows, cols)?;
        let raw = weights.as_bytes();
        let quant = weights.quant();

        match quant {
            QuantType::F32 => Ok(repack_dense_f32_bytes(raw, rows, cols, row_bytes)),
            QuantType::F16 | QuantType::BF16 => {
                let is_bf16 = quant == QuantType::BF16;
                if rows.saturating_mul(cols) <= dense_f32_max_elems {
                    Ok(repack_half_to_dense_f32(
                        raw, quant, is_bf16, rows, cols, row_bytes,
                    ))
                } else {
                    Ok(repack_half_compact(
                        raw, quant, is_bf16, rows, cols, row_bytes,
                    ))
                }
            }
            QuantType::Q8_0
            | QuantType::Q4_0
            | QuantType::Q4_1
            | QuantType::Q5_0
            | QuantType::Q5_1
            | QuantType::Q2_K
            | QuantType::Q3_K
            | QuantType::Q4_K_M
            | QuantType::Q5_K
            | QuantType::Q6_K
            | QuantType::IQ4_NL
            | QuantType::TQ1_0
            | QuantType::TQ2_0
            | QuantType::I2_S => Ok(repack_block_scaled_i8(raw, quant, rows, cols, row_bytes)),
            QuantType::Q8_1 | QuantType::MXFP4 => Err(K3Error::unsupported_tensor_quant(
                "repack",
                weights.name(),
                quant,
            )),
        }
    }

    /// Repacks a `Q5_0` weight tensor into `8x16` `i8` `A100` `XSMTVDotII` / `X100` `RVV` tiles.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::CapabilityMismatch`] if `weights.quant() != QuantType::Q5_0`, or
    /// [`K3Error::TensorBoundsError`] if dimensions or backing byte bounds do not match.
    pub fn repack_q5_0_to_ime_8x16(weights: &TensorView<'_>) -> Result<Self, K3Error> {
        if weights.quant() != QuantType::Q5_0 {
            return Err(K3Error::unsupported_tensor_quant(
                "repack_q5_0_to_ime_8x16",
                weights.name(),
                weights.quant(),
            ));
        }
        Self::repack(weights)
    }

    /// Computes the [`BlockScaledTileLayout`] required to assemble a block-scaled
    /// [`RepackedWeight`] concurrently across worker slices (`#89`).
    ///
    /// # Errors
    ///
    /// Returns [`K3Error`] when `weights` is dense, unsupported, or shape-mismatched.
    pub fn block_scaled_layout(
        weights: &TensorView<'_>,
        rows: usize,
        cols: usize,
    ) -> Result<BlockScaledTileLayout, K3Error> {
        BlockScaledTileLayout::for_tensor(weights, rows, cols)
    }

    /// Constructs a [`RepackedWeight`] from externally populated `layout`-sized buffers produced by
    /// concurrent calls to [`repack_into_tile_range`] or [`repack_q4_k_m_parallel_slices`] (`#89`).
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] if any buffer length does not match `layout`.
    pub fn from_tile_buffers(
        layout: BlockScaledTileLayout,
        tiles: Vec<Aligned128>,
        scales: Vec<f32>,
        mins: Option<Vec<f32>>,
    ) -> Result<Self, K3Error> {
        if tiles.len() != layout.total_tiles() || scales.len() != layout.total_scales() {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "RepackedWeight::from_tile_buffers".to_owned(),
                offset: 0,
                byte_len: tiles.len(),
                buffer_len: layout.total_tiles(),
                reason: format!(
                    "buffer lengths ({}, {}) != expected ({}, {})",
                    tiles.len(),
                    scales.len(),
                    layout.total_tiles(),
                    layout.total_scales()
                ),
            });
        }
        if layout.has_mins {
            let valid = mins
                .as_ref()
                .is_some_and(|m| m.len() == layout.total_mins());
            if !valid {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: "RepackedWeight::from_tile_buffers".to_owned(),
                    offset: 0,
                    byte_len: mins.as_ref().map_or(0, Vec::len),
                    buffer_len: layout.total_mins(),
                    reason: format!(
                        "{:?} requires mins buffer of length {}",
                        layout.quant,
                        layout.total_mins()
                    ),
                });
            }
        } else if let Some(unexpected_mins) = mins {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "RepackedWeight::from_tile_buffers".to_owned(),
                offset: 0,
                byte_len: unexpected_mins.len(),
                buffer_len: 0,
                reason: format!("{:?} does not use mins buffer", layout.quant),
            });
        }
        Ok(Self {
            rows: layout.rows,
            cols: layout.cols,
            k_padded: layout.k_padded,
            quant: layout.quant,
            storage: RepackedStorage::BlockScaledI8 {
                n_tiles: layout.n_tiles,
                k_chunks: layout.k_chunks,
                chunks_per_scale: layout.chunks_per_scale,
                scale_groups: layout.scale_groups,
                tiles,
                scales,
                mins,
            },
        })
    }

    /// Number of output rows `N`.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Number of `8`-row `XSMTVDotII` output tiles (`rows.div_ceil(8)` for block-quantized storage, `0` for dense).
    #[must_use]
    pub const fn n_tiles(&self) -> usize {
        match &self.storage {
            RepackedStorage::BlockScaledI8 { n_tiles, .. } => *n_tiles,
            RepackedStorage::DenseF32 { .. } | RepackedStorage::DenseHalf { .. } => 0,
        }
    }

    /// Number of logical input columns `K`.
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Padded inner dimension aligned to [`Aligned128`] / [`TILE_K`].
    #[must_use]
    pub const fn k_padded(&self) -> usize {
        self.k_padded
    }

    /// Source quantization format.
    #[must_use]
    pub const fn quant(&self) -> QuantType {
        self.quant
    }

    /// Shared-scale group size along `K` (`16` for `Q6_K`, `Q3_K`, and `Q2_K`, `32` for other
    /// block-quantized formats, or `0` for dense `F32`/`F16`/`BF16`).
    #[must_use]
    pub const fn group_size(&self) -> usize {
        match &self.storage {
            RepackedStorage::BlockScaledI8 {
                chunks_per_scale, ..
            } => *chunks_per_scale * TILE_K,
            RepackedStorage::DenseF32 { .. } | RepackedStorage::DenseHalf { .. } => 0,
        }
    }

    /// Whether `F16`/`BF16` weights are stored in compact `u16` mode rather than pre-expanded `f32`.
    #[must_use]
    pub const fn is_compact_half(&self) -> bool {
        matches!(self.storage, RepackedStorage::DenseHalf { .. })
    }

    /// Total bytes occupied by the `128-byte`-aligned weight payload.
    #[must_use]
    pub fn aligned_bytes(&self) -> usize {
        match &self.storage {
            RepackedStorage::BlockScaledI8 { tiles, .. } => tiles.len() * size_of::<Aligned128>(),
            RepackedStorage::DenseF32 { data, .. } | RepackedStorage::DenseHalf { data, .. } => {
                data.len() * size_of::<Aligned128>()
            }
        }
    }

    /// Base pointer to the `128-byte`-aligned weight payload (`ptr as usize % 128 == 0`).
    #[must_use]
    pub fn payload_ptr(&self) -> *const u8 {
        match &self.storage {
            RepackedStorage::BlockScaledI8 { tiles, .. } => tiles.as_ptr().cast::<u8>(),
            RepackedStorage::DenseF32 { data, .. } | RepackedStorage::DenseHalf { data, .. } => {
                data.as_ptr().cast::<u8>()
            }
        }
    }

    /// Reconstructs logical weight row `row` into `out` (`out.len() == self.cols()`), enabling
    /// lossless round-trip verification against raw GGUF block dequantizers.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] if `row >= self.rows()` or `out.len() != self.cols()`.
    pub fn dequantize_row(&self, row: usize, out: &mut [f32]) -> Result<(), K3Error> {
        if row >= self.rows || out.len() != self.cols {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "RepackedWeight::dequantize_row".to_owned(),
                offset: row,
                byte_len: size_of_val(out),
                buffer_len: self.cols * size_of::<f32>(),
                reason: format!(
                    "invalid row ({row}/{}) or output length ({}/{})",
                    self.rows,
                    out.len(),
                    self.cols
                ),
            });
        }

        match &self.storage {
            RepackedStorage::DenseF32 {
                blocks_per_row,
                data,
            } => {
                let row_blocks = &data[row * blocks_per_row..(row + 1) * blocks_per_row];
                let row_f32 = Aligned128::slice_as_f32(row_blocks);
                out.copy_from_slice(&row_f32[..self.cols]);
            }
            RepackedStorage::DenseHalf {
                is_bf16,
                blocks_per_row,
                data,
            } => {
                let row_blocks = &data[row * blocks_per_row..(row + 1) * blocks_per_row];
                let row_u16 = Aligned128::slice_as_u16(row_blocks);
                for (dst, &bits) in out.iter_mut().zip(&row_u16[..self.cols]) {
                    *dst = if *is_bf16 {
                        bf16_to_f32_finite(bits)
                    } else {
                        f16_to_f32_finite(bits)
                    };
                }
            }
            RepackedStorage::BlockScaledI8 {
                k_chunks,
                chunks_per_scale,
                scale_groups,
                tiles,
                scales,
                mins,
                ..
            } => {
                let nt = row / TILE_N;
                let j = row % TILE_N;
                for ch in 0..*k_chunks {
                    let sg = ch / *chunks_per_scale;
                    let w_sc = scales[(nt * *scale_groups + sg) * TILE_N + j];
                    let w_mn = mins
                        .as_ref()
                        .map_or(0.0_f32, |m| m[(nt * *scale_groups + sg) * TILE_N + j]);
                    let tile_i8 =
                        Aligned128::slice_as_i8(core::slice::from_ref(&tiles[nt * *k_chunks + ch]));
                    let q_slice = &tile_i8[j * TILE_K..(j + 1) * TILE_K];
                    let col_base = ch * TILE_K;
                    let valid = (self.cols - col_base).min(TILE_K);
                    for k_idx in 0..valid {
                        out[col_base + k_idx] = w_sc * f32::from(q_slice[k_idx]) - w_mn;
                    }
                }
            }
        }
        Ok(())
    }

    /// Single-token GEMV (`[1, K] x [N, K]^T -> [N]`) over pre-quantized [`QuantizedActQ8`].
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] on dimension mismatch.
    pub fn gemv(&self, out: &mut [f32], act: &QuantizedActQ8) -> Result<(), K3Error> {
        gemv_repacked(out, self, act)
    }

    /// Row-sliced single-token GEMV (`row_range` within `0..self.rows()`).
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] on dimension or range mismatch.
    pub fn gemv_rows(
        &self,
        out: &mut [f32],
        act: &QuantizedActQ8,
        row_range: Range<usize>,
    ) -> Result<(), K3Error> {
        gemv_repacked_rows(out, self, act, row_range)
    }

    /// Batched multi-token GEMM (`[M, K] x [N, K]^T -> [M, N]`) over pre-quantized [`QuantizedActQ8`].
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] on dimension mismatch.
    pub fn gemm(&self, out: &mut [f32], act: &QuantizedActQ8) -> Result<(), K3Error> {
        gemm_repacked(out, self, act)
    }

    /// Row-sliced batched multi-token GEMM writing `[M, row_range.len()]` row-major output.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] on dimension or range mismatch.
    pub fn gemm_rows(
        &self,
        out: &mut [f32],
        act: &QuantizedActQ8,
        row_range: Range<usize>,
    ) -> Result<(), K3Error> {
        gemm_repacked_rows(out, self, act, row_range)
    }
}

#[inline]
fn dot_f32_dispatch(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        dot_product_f32_rvv(a, b)
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        dot_product_f32_scalar(a, b)
    }
}

#[inline]
fn eval_half_row_dot(
    row_u16: &[u16],
    act_row: &[f32],
    cols: usize,
    is_bf16: bool,
    vlen_sim_bits: Option<usize>,
) -> f32 {
    let mut chunk_buf = [0.0_f32; 256];
    let mut sum = 0.0_f32;
    let mut col = 0usize;
    while col < cols {
        let n = (cols - col).min(chunk_buf.len());
        for (i, slot) in chunk_buf.iter_mut().enumerate().take(n) {
            let bits = row_u16[col + i];
            *slot = if is_bf16 {
                bf16_to_f32_finite(bits)
            } else {
                f16_to_f32_finite(bits)
            };
        }
        sum += if let Some(vbits) = vlen_sim_bits {
            dot_product_f32_vla_stripmined(&chunk_buf[..n], &act_row[col..col + n], vbits)
        } else {
            dot_f32_dispatch(&chunk_buf[..n], &act_row[col..col + n])
        };
        col += n;
    }
    sum
}

#[expect(
    clippy::too_many_lines,
    reason = "dispatches BlockScaledI8 (A100 IME / X100 RVV / Scalar), DenseF32, and DenseHalf over arbitrary M and row_range"
)]
fn execute_gemm_repacked_rows(
    out: &mut [f32],
    weights: &RepackedWeight,
    act: &QuantizedActQ8,
    row_range: Range<usize>,
    vlen_sim_bits: Option<usize>,
) -> Result<(), K3Error> {
    if act.k() != weights.cols() {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "gemm_repacked".to_owned(),
            offset: 0,
            byte_len: size_of_val(out),
            buffer_len: weights.aligned_bytes(),
            reason: format!(
                "activation K ({}) != weight cols ({})",
                act.k(),
                weights.cols()
            ),
        });
    }
    if row_range.start > row_range.end || row_range.end > weights.rows() {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "gemm_repacked".to_owned(),
            offset: row_range.start,
            byte_len: row_range.len(),
            buffer_len: weights.rows(),
            reason: format!(
                "invalid row_range {}..{} for weight rows {}",
                row_range.start,
                row_range.end,
                weights.rows()
            ),
        });
    }

    let m = act.m();
    let sub_n = row_range.len();
    let expected_out = m
        .checked_mul(sub_n)
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: "gemm_repacked".to_owned(),
            offset: 0,
            byte_len: size_of_val(out),
            buffer_len: usize::MAX,
            reason: "m * row_range.len() overflow".to_owned(),
        })?;
    if out.len() != expected_out {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "gemm_repacked".to_owned(),
            offset: 0,
            byte_len: size_of_val(out),
            buffer_len: expected_out * size_of::<f32>(),
            reason: format!(
                "output length {} != m ({m}) * sub_rows ({sub_n})",
                out.len()
            ),
        });
    }
    if sub_n == 0 {
        return Ok(());
    }

    match &weights.storage {
        RepackedStorage::DenseF32 {
            blocks_per_row,
            data,
        } => {
            for r_m in 0..m {
                let x_row = act.f32_row(r_m);
                let out_row = &mut out[r_m * sub_n..(r_m + 1) * sub_n];
                for (local_n, r_n) in row_range.clone().enumerate() {
                    let w_blocks = &data[r_n * blocks_per_row..(r_n + 1) * blocks_per_row];
                    let w_row = &Aligned128::slice_as_f32(w_blocks)[..weights.cols];
                    out_row[local_n] = if let Some(vbits) = vlen_sim_bits {
                        dot_product_f32_vla_stripmined(x_row, w_row, vbits)
                    } else {
                        dot_f32_dispatch(x_row, w_row)
                    };
                }
            }
        }
        RepackedStorage::DenseHalf {
            is_bf16,
            blocks_per_row,
            data,
        } => {
            for r_m in 0..m {
                let x_row = act.f32_row(r_m);
                let out_row = &mut out[r_m * sub_n..(r_m + 1) * sub_n];
                for (local_n, r_n) in row_range.clone().enumerate() {
                    let w_blocks = &data[r_n * blocks_per_row..(r_n + 1) * blocks_per_row];
                    let w_u16 = &Aligned128::slice_as_u16(w_blocks)[..weights.cols];
                    out_row[local_n] =
                        eval_half_row_dot(w_u16, x_row, weights.cols, *is_bf16, vlen_sim_bits);
                }
            }
        }
        RepackedStorage::BlockScaledI8 {
            k_chunks,
            chunks_per_scale,
            scale_groups,
            tiles: b_tiles,
            scales: w_scales,
            mins: w_mins,
            ..
        } => {
            #[cfg(all(target_arch = "riscv64", not(miri)))]
            let use_hw_ime = vlen_sim_bits.is_none() && has_ime();

            let a_tiles = act.tiles();
            let x_scales = act.x_scales();
            let x_sums = act.x_sums();
            let x_sums_16 = &act.x_sums_16;
            let k_blocks32 = act.k_blocks32;

            let first_nt = row_range.start / TILE_N;
            let last_nt = (row_range.end - 1) / TILE_N;
            let mut acc_f32 = [0.0_f32; TILE_C];

            // Outer loop over weight tiles `nt` keeps each weight row-tile hot in L1/L2 cache
            // across all `m_tiles` prompt tokens.
            for nt in first_nt..=last_nt {
                let j_start = row_range.start.saturating_sub(nt * TILE_N);
                let j_end = (row_range.end - nt * TILE_N).min(TILE_N);
                let b_row_tiles = &b_tiles[nt * *k_chunks..(nt + 1) * *k_chunks];
                let w_scales_tile = &w_scales[nt * *scale_groups * TILE_N..];
                let w_mins_tile = w_mins
                    .as_deref()
                    .map(|mins_buf| &mins_buf[nt * *scale_groups * TILE_N..]);

                for mt in 0..act.m_tiles() {
                    let active_m = (m - mt * TILE_M).min(TILE_M);
                    let a_row_tiles = &a_tiles[mt * *k_chunks..(mt + 1) * *k_chunks];
                    let x_scales_tile = &x_scales[mt * k_blocks32 * TILE_M..];
                    let x_sums_tile = w_mins_tile.map(|_| {
                        if *chunks_per_scale == 2 {
                            &x_sums[mt * k_blocks32 * TILE_M..]
                        } else {
                            &x_sums_16[mt * *k_chunks * TILE_M..]
                        }
                    });

                    if let Some(vbits) = vlen_sim_bits {
                        acc_f32.fill(0.0_f32);
                        let mut c_i32 = [0_i32; TILE_C];
                        for sg in 0..*scale_groups {
                            let ch_start = sg * *chunks_per_scale;
                            let ch_end = ch_start + *chunks_per_scale;
                            let a_sub = Aligned128::slice_as_i8(&a_row_tiles[ch_start..ch_end]);
                            let b_sub = Aligned128::slice_as_i8(&b_row_tiles[ch_start..ch_end]);
                            tile_dot_vla_sim(
                                a_sub,
                                b_sub,
                                *chunks_per_scale,
                                active_m,
                                vbits,
                                &mut c_i32,
                            );
                            accumulate_scaled_tile_scalar(
                                &mut acc_f32,
                                &c_i32,
                                x_scales_tile,
                                w_scales_tile,
                                x_sums_tile,
                                w_mins_tile,
                                *chunks_per_scale,
                                sg,
                                active_m,
                                j_start..j_end,
                            );
                        }
                    } else {
                        #[cfg(all(target_arch = "riscv64", not(miri)))]
                        if use_hw_ime {
                            let a_i8 = Aligned128::slice_as_i8(a_row_tiles);
                            let b_i8 = Aligned128::slice_as_i8(b_row_tiles);
                            let p_x_sums = x_sums_tile.map_or(core::ptr::null(), <[f32]>::as_ptr);
                            let p_w_mins = w_mins_tile.map_or(core::ptr::null(), <[f32]>::as_ptr);
                            // SAFETY: `a_i8`/`b_i8` hold `scale_groups * chunks_per_scale * 128` bytes,
                            // `x_scales_tile`/`w_scales_tile`/`p_x_sums`/`p_w_mins` cover `scale_groups` groups,
                            // `acc_f32` has 64 `f32` lanes, and `has_ime()` verified `A100` (`vlenb == 128`).
                            unsafe {
                                fused_tile_gemm_ime(
                                    a_i8.as_ptr(),
                                    b_i8.as_ptr(),
                                    x_scales_tile.as_ptr(),
                                    w_scales_tile.as_ptr(),
                                    p_x_sums,
                                    p_w_mins,
                                    *scale_groups,
                                    *chunks_per_scale,
                                    acc_f32.as_mut_ptr(),
                                );
                            }
                        } else {
                            let a_i8 = Aligned128::slice_as_i8(a_row_tiles);
                            let b_i8 = Aligned128::slice_as_i8(b_row_tiles);
                            let p_x_sums_base =
                                x_sums_tile.map_or(core::ptr::null(), <[f32]>::as_ptr);
                            let p_w_mins = w_mins_tile.map_or(core::ptr::null(), <[f32]>::as_ptr);
                            let mut half_acc = [0.0_f32; 32];
                            // Top 4 rows of `mt` (`0..active_m.min(4)`)
                            // SAFETY: `a_i8`/`b_i8` contain `k_chunks * 128` bytes, `x_*`/`w_*` have
                            // full 8-lane groups, `half_acc` has 32 `f32`s, and RVV uses `vl <= 16`
                            // at `e8, m1` and `vl = 4` at `e32, m1` (safe for all `VLEN >= 128`).
                            unsafe {
                                fused_tile_gemm_rvv_4x8(
                                    a_i8.as_ptr(),
                                    b_i8.as_ptr(),
                                    x_scales_tile.as_ptr(),
                                    w_scales_tile.as_ptr(),
                                    p_x_sums_base,
                                    p_w_mins,
                                    *scale_groups,
                                    *chunks_per_scale,
                                    half_acc.as_mut_ptr(),
                                );
                            }
                            acc_f32[..32].copy_from_slice(&half_acc);
                            if active_m > 4 {
                                let p_x_sums_hi = if p_x_sums_base.is_null() {
                                    core::ptr::null()
                                } else {
                                    // SAFETY: every group in `x_sums` / `x_sums_16` stores `TILE_M = 8` floats.
                                    unsafe { p_x_sums_base.add(4) }
                                };
                                // SAFETY: `a_i8.add(64)` addresses rows `4..7` inside every `128B` `A` tile.
                                unsafe {
                                    fused_tile_gemm_rvv_4x8(
                                        a_i8.as_ptr().add(64),
                                        b_i8.as_ptr(),
                                        x_scales_tile.as_ptr().add(4),
                                        w_scales_tile.as_ptr(),
                                        p_x_sums_hi,
                                        p_w_mins,
                                        *scale_groups,
                                        *chunks_per_scale,
                                        half_acc.as_mut_ptr(),
                                    );
                                }
                                acc_f32[32..].copy_from_slice(&half_acc);
                            }
                        }

                        #[cfg(any(not(target_arch = "riscv64"), miri))]
                        {
                            acc_f32.fill(0.0_f32);
                            let mut c_i32 = [0_i32; TILE_C];
                            for sg in 0..*scale_groups {
                                let ch_start = sg * *chunks_per_scale;
                                let ch_end = ch_start + *chunks_per_scale;
                                let a_sub = Aligned128::slice_as_i8(&a_row_tiles[ch_start..ch_end]);
                                let b_sub = Aligned128::slice_as_i8(&b_row_tiles[ch_start..ch_end]);
                                tile_dot_scalar_rows(
                                    a_sub,
                                    b_sub,
                                    *chunks_per_scale,
                                    active_m,
                                    &mut c_i32,
                                );
                                accumulate_scaled_tile_scalar(
                                    &mut acc_f32,
                                    &c_i32,
                                    x_scales_tile,
                                    w_scales_tile,
                                    x_sums_tile,
                                    w_mins_tile,
                                    *chunks_per_scale,
                                    sg,
                                    active_m,
                                    j_start..j_end,
                                );
                            }
                        }
                    }

                    for i in 0..active_m {
                        let global_m = mt * TILE_M + i;
                        let out_row = &mut out[global_m * sub_n..(global_m + 1) * sub_n];
                        for j in j_start..j_end {
                            let local_n = (nt * TILE_N + j) - row_range.start;
                            out_row[local_n] = acc_f32[i * TILE_N + j];
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn accumulate_scaled_tile_scalar(
    acc_f32: &mut [f32; TILE_C],
    c_i32: &[i32; TILE_C],
    x_scales_tile: &[f32],
    w_scales_tile: &[f32],
    x_sums_tile: Option<&[f32]>,
    w_mins_tile: Option<&[f32]>,
    chunks_per_scale: usize,
    sg: usize,
    active_m: usize,
    j_range: Range<usize>,
) {
    let b32 = if chunks_per_scale == 2 { sg } else { sg / 2 };
    let xs = &x_scales_tile[b32 * TILE_M..][..TILE_M];
    let ws = &w_scales_tile[sg * TILE_N..][..TILE_N];

    if let (Some(sums_buf), Some(mins_buf)) = (x_sums_tile, w_mins_tile) {
        let xsm = if chunks_per_scale == 2 {
            &sums_buf[b32 * TILE_M..][..TILE_M]
        } else {
            &sums_buf[sg * TILE_M..][..TILE_M]
        };
        let wm = &mins_buf[sg * TILE_N..][..TILE_N];
        for i in 0..active_m {
            let x_sc = xs[i];
            let x_sum_i = xsm[i];
            for j in j_range.clone() {
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "16..32-element i8*i8 dot fits exactly in 24-bit f32 mantissa"
                )]
                let dot_f = c_i32[i * TILE_N + j] as f32;
                acc_f32[i * TILE_N + j] += dot_f * (x_sc * ws[j]) - x_sum_i * wm[j];
            }
        }
    } else {
        for i in 0..active_m {
            let x_sc = xs[i];
            for j in j_range.clone() {
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "16..32-element i8*i8 dot fits exactly in 24-bit f32 mantissa"
                )]
                let dot_f = c_i32[i * TILE_N + j] as f32;
                acc_f32[i * TILE_N + j] += dot_f * (x_sc * ws[j]);
            }
        }
    }
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
const TILE_IDX_ROW: [u32; 64] = {
    let mut t = [0_u32; 64];
    let mut i = 0_u32;
    while i < 8 {
        let mut j = 0_u32;
        while j < 8 {
            t[(i * 8 + j) as usize] = i;
            j += 1;
        }
        i += 1;
    }
    t
};

#[cfg(all(target_arch = "riscv64", not(miri)))]
const TILE_IDX_COL: [u32; 64] = {
    let mut t = [0_u32; 64];
    let mut i = 0_u32;
    while i < 8 {
        let mut j = 0_u32;
        while j < 8 {
            t[(i * 8 + j) as usize] = j;
            j += 1;
        }
        i += 1;
    }
    t
};

/// Full-`K` `A100` (`VLEN = 1024`) `XSMTVDotII` fused GEMM micro-kernel (`#85`): maintains the
/// `8 x 8` `f32` accumulator in `v18:v19` across all `scale_groups` without intermediate
/// `[i32; 64]` stack spills.
#[cfg(all(target_arch = "riscv64", not(miri)))]
#[expect(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "raw kernel register bindings and inline asm"
)]
unsafe fn fused_tile_gemm_ime(
    pa: *const i8,
    pb: *const i8,
    pxs: *const f32,
    pws: *const f32,
    pxsm: *const f32,
    pwm: *const f32,
    scale_groups: usize,
    chunks_per_scale: usize,
    p_out: *mut f32,
) {
    if scale_groups == 0 {
        // SAFETY: `p_out` points to 64 valid `f32`s per caller contract.
        unsafe {
            core::ptr::write_bytes(p_out, 0, TILE_C);
        }
        return;
    }
    // SAFETY: caller guarantees buffer extents matching `scale_groups * chunks_per_scale`,
    // `p_out` holds 64 `f32`s, and `has_ime()` verified `A100` (`vlenb == 128`).
    unsafe {
        core::arch::asm!(
            "li {vl}, 64",
            "vsetvli zero, {vl}, e32, m2, ta, ma",
            "vmv.v.i v18, 0",
            "vle32.v v20, ({p_idx_row})",
            "vle32.v v22, ({p_idx_col})",
            "li {xs_half}, 0",
            "2:",
            "li {vl}, 64",
            "vsetvli zero, {vl}, e32, m2, ta, ma",
            "vmv.v.i v0, 0",
            "mv {ch}, {cps}",
            "li {vl}, 128",
            "vsetvli zero, {vl}, e8, m1, ta, ma",
            "3:",
            "vle8.v v8, ({pa})",
            "vle8.v v16, ({pb})",
            ".word 0xe304302b",
            "addi {pa}, {pa}, 128",
            "addi {pb}, {pb}, 128",
            "addi {ch}, {ch}, -1",
            "bnez {ch}, 3b",
            "vsetivli zero, 8, e32, m1, ta, ma",
            "vle32.v v4, ({pxs})",
            "vle32.v v10, ({pws})",
            "addi {pws}, {pws}, 32",
            "li {vl}, 1",
            "bne {cps}, {vl}, 4f",
            "xori {xs_half}, {xs_half}, 1",
            "bnez {xs_half}, 5f",
            "4:",
            "addi {pxs}, {pxs}, 32",
            "5:",
            "li {vl}, 64",
            "vsetvli zero, {vl}, e32, m2, ta, ma",
            "vfcvt.f.x.v v2, v0",
            "vrgather.vv v6, v4, v20",
            "vrgather.vv v12, v10, v22",
            "vfmul.vv v4, v6, v12",
            "vfmacc.vv v18, v2, v4",
            "beqz {pxsm}, 6f",
            "vsetivli zero, 8, e32, m1, ta, ma",
            "vle32.v v4, ({pxsm})",
            "vle32.v v10, ({pwm})",
            "addi {pxsm}, {pxsm}, 32",
            "addi {pwm}, {pwm}, 32",
            "li {vl}, 64",
            "vsetvli zero, {vl}, e32, m2, ta, ma",
            "vrgather.vv v6, v4, v20",
            "vrgather.vv v12, v10, v22",
            "vfnmsac.vv v18, v6, v12",
            "6:",
            "addi {sg_rem}, {sg_rem}, -1",
            "bnez {sg_rem}, 2b",
            "li {vl}, 64",
            "vsetvli zero, {vl}, e32, m2, ta, ma",
            "vse32.v v18, ({p_out})",
            pa = inout(reg) pa => _,
            pb = inout(reg) pb => _,
            pxs = inout(reg) pxs => _,
            pws = inout(reg) pws => _,
            pxsm = inout(reg) pxsm => _,
            pwm = inout(reg) pwm => _,
            sg_rem = inout(reg) scale_groups => _,
            cps = in(reg) chunks_per_scale,
            p_idx_row = in(reg) TILE_IDX_ROW.as_ptr(),
            p_idx_col = in(reg) TILE_IDX_COL.as_ptr(),
            p_out = in(reg) p_out,
            vl = out(reg) _,
            ch = out(reg) _,
            xs_half = out(reg) _,
            out("v0") _,
            out("v1") _,
            out("v2") _,
            out("v3") _,
            out("v4") _,
            out("v5") _,
            out("v6") _,
            out("v7") _,
            out("v8") _,
            out("v9") _,
            out("v10") _,
            out("v11") _,
            out("v12") _,
            out("v13") _,
            out("v16") _,
            out("v18") _,
            out("v19") _,
            out("v20") _,
            out("v21") _,
            out("v22") _,
            out("v23") _,
            options(nostack),
        );
    }
}

/// Full-`K` `X100` (`zvl256b`, `VLEN >= 128`-safe) `4 x 8` fused RVV 1.0 micro-kernel (`#86`):
/// keeps all `32` `f32` accumulators (`4 rows x 8 cols`) resident in `v24..v31` across all
/// `scale_groups` with zero stack spills per `32`-element block.
#[cfg(all(target_arch = "riscv64", not(miri)))]
#[expect(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "fused 4x8 RVV assembly kernel"
)]
unsafe fn fused_tile_gemm_rvv_4x8(
    pa: *const i8,
    pb: *const i8,
    pxs: *const f32,
    pws: *const f32,
    pxsm: *const f32,
    pwm: *const f32,
    scale_groups: usize,
    chunks_per_scale: usize,
    p_out: *mut f32,
) {
    if scale_groups == 0 {
        // SAFETY: `p_out` points to 32 valid `f32`s (`4 rows x 8 cols`).
        unsafe {
            core::ptr::write_bytes(p_out, 0, 32);
        }
        return;
    }
    // SAFETY: caller guarantees tile/scale buffers hold `scale_groups` groups, `p_out` holds 32
    // `f32`s, and all vector instructions strictly enforce `vl <= 16` at `e8, m1` / `e16, m2`
    // and `vl = 4` at `e32, m1` (`<= VLMAX` on any `VLEN >= 128` CPU).
    unsafe {
        core::arch::asm!(
            "vsetivli zero, 4, e32, m1, ta, ma",
            "vmv.v.i v24, 0",
            "vmv.v.i v25, 0",
            "vmv.v.i v26, 0",
            "vmv.v.i v27, 0",
            "vmv.v.i v28, 0",
            "vmv.v.i v29, 0",
            "vmv.v.i v30, 0",
            "vmv.v.i v31, 0",
            "li {xs_half}, 0",
            "2:",
            // ---- Half-panel 0 (cols 0..3): use copies ta = pa, tb = pb ----
            "vsetivli zero, 1, e32, m1, ta, ma",
            "vmv.v.i v0, 0",
            "vmv.v.i v1, 0",
            "vmv.v.i v2, 0",
            "vmv.v.i v3, 0",
            "vmv.v.i v4, 0",
            "vmv.v.i v5, 0",
            "vmv.v.i v6, 0",
            "vmv.v.i v7, 0",
            "vmv.v.i v8, 0",
            "vmv.v.i v9, 0",
            "vmv.v.i v10, 0",
            "vmv.v.i v11, 0",
            "vmv.v.i v12, 0",
            "vmv.v.i v13, 0",
            "vmv.v.i v14, 0",
            "vmv.v.i v15, 0",
            "mv {ta}, {pa}",
            "mv {tb}, {pb}",
            "mv {ch}, {cps}",
            "3:",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "addi t0, {tb}, 16",
            "addi t1, {tb}, 32",
            "addi t2, {tb}, 48",
            "vle8.v v16, ({tb})",
            "vle8.v v17, (t0)",
            "vle8.v v18, (t1)",
            "vle8.v v19, (t2)",
            // Row 0
            "vle8.v v20, ({ta})",
            "vwmul.vv v22, v20, v16",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v0, v22, v0",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vwmul.vv v22, v20, v17",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v1, v22, v1",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vwmul.vv v22, v20, v18",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v2, v22, v2",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vwmul.vv v22, v20, v19",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v3, v22, v3",
            // Row 1
            "vsetivli zero, 16, e8, m1, ta, ma",
            "addi t0, {ta}, 16",
            "vle8.v v20, (t0)",
            "vwmul.vv v22, v20, v16",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v4, v22, v4",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vwmul.vv v22, v20, v17",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v5, v22, v5",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vwmul.vv v22, v20, v18",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v6, v22, v6",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vwmul.vv v22, v20, v19",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v7, v22, v7",
            // Row 2
            "vsetivli zero, 16, e8, m1, ta, ma",
            "addi t0, {ta}, 32",
            "vle8.v v20, (t0)",
            "vwmul.vv v22, v20, v16",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v8, v22, v8",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vwmul.vv v22, v20, v17",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v9, v22, v9",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vwmul.vv v22, v20, v18",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v10, v22, v10",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vwmul.vv v22, v20, v19",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v11, v22, v11",
            // Row 3
            "vsetivli zero, 16, e8, m1, ta, ma",
            "addi t0, {ta}, 48",
            "vle8.v v20, (t0)",
            "vwmul.vv v22, v20, v16",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v12, v22, v12",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vwmul.vv v22, v20, v17",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v13, v22, v13",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vwmul.vv v22, v20, v18",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v14, v22, v14",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vwmul.vv v22, v20, v19",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v15, v22, v15",
            "addi {ta}, {ta}, 128",
            "addi {tb}, {tb}, 128",
            "addi {ch}, {ch}, -1",
            "bnez {ch}, 3b",

            // Pack cols 0..3 -> f32 and FMA into v24..v27
            "vsetivli zero, 4, e32, m1, ta, ma",
            "vle32.v v18, ({pws})",
            // Row 0 -> v24
            "vmv.x.s t0, v3",
            "vmv.s.x v17, t0",
            "vmv.x.s t0, v2",
            "vslide1up.vx v16, v17, t0",
            "vmv.x.s t0, v1",
            "vslide1up.vx v17, v16, t0",
            "vmv.x.s t0, v0",
            "vslide1up.vx v16, v17, t0",
            "vfcvt.f.x.v v16, v16",
            "flw ft0, 0({pxs})",
            "vfmul.vf v17, v18, ft0",
            "vfmacc.vv v24, v16, v17",
            // Row 1 -> v25
            "vmv.x.s t0, v7",
            "vmv.s.x v17, t0",
            "vmv.x.s t0, v6",
            "vslide1up.vx v16, v17, t0",
            "vmv.x.s t0, v5",
            "vslide1up.vx v17, v16, t0",
            "vmv.x.s t0, v4",
            "vslide1up.vx v16, v17, t0",
            "vfcvt.f.x.v v16, v16",
            "flw ft0, 4({pxs})",
            "vfmul.vf v17, v18, ft0",
            "vfmacc.vv v25, v16, v17",
            // Row 2 -> v26
            "vmv.x.s t0, v11",
            "vmv.s.x v17, t0",
            "vmv.x.s t0, v10",
            "vslide1up.vx v16, v17, t0",
            "vmv.x.s t0, v9",
            "vslide1up.vx v17, v16, t0",
            "vmv.x.s t0, v8",
            "vslide1up.vx v16, v17, t0",
            "vfcvt.f.x.v v16, v16",
            "flw ft0, 8({pxs})",
            "vfmul.vf v17, v18, ft0",
            "vfmacc.vv v26, v16, v17",
            // Row 3 -> v27
            "vmv.x.s t0, v15",
            "vmv.s.x v17, t0",
            "vmv.x.s t0, v14",
            "vslide1up.vx v16, v17, t0",
            "vmv.x.s t0, v13",
            "vslide1up.vx v17, v16, t0",
            "vmv.x.s t0, v12",
            "vslide1up.vx v16, v17, t0",
            "vfcvt.f.x.v v16, v16",
            "flw ft0, 12({pxs})",
            "vfmul.vf v17, v18, ft0",
            "vfmacc.vv v27, v16, v17",

            "beqz {pxsm}, 4f",
            "vle32.v v18, ({pwm})",
            "flw ft0, 0({pxsm})",
            "vfnmsac.vf v24, ft0, v18",
            "flw ft0, 4({pxsm})",
            "vfnmsac.vf v25, ft0, v18",
            "flw ft0, 8({pxsm})",
            "vfnmsac.vf v26, ft0, v18",
            "flw ft0, 12({pxsm})",
            "vfnmsac.vf v27, ft0, v18",
            "4:",

            // ---- Half-panel 1 (cols 4..7): advance pa, pb directly ----
            "vsetivli zero, 1, e32, m1, ta, ma",
            "vmv.v.i v0, 0",
            "vmv.v.i v1, 0",
            "vmv.v.i v2, 0",
            "vmv.v.i v3, 0",
            "vmv.v.i v4, 0",
            "vmv.v.i v5, 0",
            "vmv.v.i v6, 0",
            "vmv.v.i v7, 0",
            "vmv.v.i v8, 0",
            "vmv.v.i v9, 0",
            "vmv.v.i v10, 0",
            "vmv.v.i v11, 0",
            "vmv.v.i v12, 0",
            "vmv.v.i v13, 0",
            "vmv.v.i v14, 0",
            "vmv.v.i v15, 0",
            "mv {ch}, {cps}",
            "5:",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "addi t0, {pb}, 64",
            "addi t1, {pb}, 80",
            "addi t2, {pb}, 96",
            "addi t3, {pb}, 112",
            "vle8.v v16, (t0)",
            "vle8.v v17, (t1)",
            "vle8.v v18, (t2)",
            "vle8.v v19, (t3)",
            // Row 0
            "vle8.v v20, ({pa})",
            "vwmul.vv v22, v20, v16",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v0, v22, v0",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vwmul.vv v22, v20, v17",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v1, v22, v1",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vwmul.vv v22, v20, v18",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v2, v22, v2",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vwmul.vv v22, v20, v19",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v3, v22, v3",
            // Row 1
            "vsetivli zero, 16, e8, m1, ta, ma",
            "addi t0, {pa}, 16",
            "vle8.v v20, (t0)",
            "vwmul.vv v22, v20, v16",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v4, v22, v4",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vwmul.vv v22, v20, v17",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v5, v22, v5",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vwmul.vv v22, v20, v18",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v6, v22, v6",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vwmul.vv v22, v20, v19",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v7, v22, v7",
            // Row 2
            "vsetivli zero, 16, e8, m1, ta, ma",
            "addi t0, {pa}, 32",
            "vle8.v v20, (t0)",
            "vwmul.vv v22, v20, v16",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v8, v22, v8",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vwmul.vv v22, v20, v17",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v9, v22, v9",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vwmul.vv v22, v20, v18",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v10, v22, v10",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vwmul.vv v22, v20, v19",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v11, v22, v11",
            // Row 3
            "vsetivli zero, 16, e8, m1, ta, ma",
            "addi t0, {pa}, 48",
            "vle8.v v20, (t0)",
            "vwmul.vv v22, v20, v16",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v12, v22, v12",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vwmul.vv v22, v20, v17",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v13, v22, v13",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vwmul.vv v22, v20, v18",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v14, v22, v14",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vwmul.vv v22, v20, v19",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v15, v22, v15",
            "addi {pa}, {pa}, 128",
            "addi {pb}, {pb}, 128",
            "addi {ch}, {ch}, -1",
            "bnez {ch}, 5b",

            // Pack cols 4..7 -> f32 and FMA into v28..v31
            "vsetivli zero, 4, e32, m1, ta, ma",
            "addi t0, {pws}, 16",
            "vle32.v v18, (t0)",
            "addi {pws}, {pws}, 32",
            // Row 0 -> v28
            "vmv.x.s t0, v3",
            "vmv.s.x v17, t0",
            "vmv.x.s t0, v2",
            "vslide1up.vx v16, v17, t0",
            "vmv.x.s t0, v1",
            "vslide1up.vx v17, v16, t0",
            "vmv.x.s t0, v0",
            "vslide1up.vx v16, v17, t0",
            "vfcvt.f.x.v v16, v16",
            "flw ft0, 0({pxs})",
            "vfmul.vf v17, v18, ft0",
            "vfmacc.vv v28, v16, v17",
            // Row 1 -> v29
            "vmv.x.s t0, v7",
            "vmv.s.x v17, t0",
            "vmv.x.s t0, v6",
            "vslide1up.vx v16, v17, t0",
            "vmv.x.s t0, v5",
            "vslide1up.vx v17, v16, t0",
            "vmv.x.s t0, v4",
            "vslide1up.vx v16, v17, t0",
            "vfcvt.f.x.v v16, v16",
            "flw ft0, 4({pxs})",
            "vfmul.vf v17, v18, ft0",
            "vfmacc.vv v29, v16, v17",
            // Row 2 -> v30
            "vmv.x.s t0, v11",
            "vmv.s.x v17, t0",
            "vmv.x.s t0, v10",
            "vslide1up.vx v16, v17, t0",
            "vmv.x.s t0, v9",
            "vslide1up.vx v17, v16, t0",
            "vmv.x.s t0, v8",
            "vslide1up.vx v16, v17, t0",
            "vfcvt.f.x.v v16, v16",
            "flw ft0, 8({pxs})",
            "vfmul.vf v17, v18, ft0",
            "vfmacc.vv v30, v16, v17",
            // Row 3 -> v31
            "vmv.x.s t0, v15",
            "vmv.s.x v17, t0",
            "vmv.x.s t0, v14",
            "vslide1up.vx v16, v17, t0",
            "vmv.x.s t0, v13",
            "vslide1up.vx v17, v16, t0",
            "vmv.x.s t0, v12",
            "vslide1up.vx v16, v17, t0",
            "vfcvt.f.x.v v16, v16",
            "flw ft0, 12({pxs})",
            "vfmul.vf v17, v18, ft0",
            "vfmacc.vv v31, v16, v17",

            // Advance pxs: every sg when cps == 2, or on odd sg when cps == 1
            "li t0, 1",
            "bne {cps}, t0, 6f",
            "xori {xs_half}, {xs_half}, 1",
            "bnez {xs_half}, 7f",
            "6:",
            "addi {pxs}, {pxs}, 32",
            "7:",

            "beqz {pxsm}, 8f",
            "addi t0, {pwm}, 16",
            "vle32.v v18, (t0)",
            "addi {pwm}, {pwm}, 32",
            "flw ft0, 0({pxsm})",
            "vfnmsac.vf v28, ft0, v18",
            "flw ft0, 4({pxsm})",
            "vfnmsac.vf v29, ft0, v18",
            "flw ft0, 8({pxsm})",
            "vfnmsac.vf v30, ft0, v18",
            "flw ft0, 12({pxsm})",
            "vfnmsac.vf v31, ft0, v18",
            "addi {pxsm}, {pxsm}, 32",
            "8:",
            "addi {sg_rem}, {sg_rem}, -1",
            "bnez {sg_rem}, 2b",

            // Write 4 rows x 8 cols once at K completion
            "vsetivli zero, 4, e32, m1, ta, ma",
            "vse32.v v24, ({p_out})",
            "addi t0, {p_out}, 16",
            "vse32.v v28, (t0)",
            "addi t0, {p_out}, 32",
            "vse32.v v25, (t0)",
            "addi t0, {p_out}, 48",
            "vse32.v v29, (t0)",
            "addi t0, {p_out}, 64",
            "vse32.v v26, (t0)",
            "addi t0, {p_out}, 80",
            "vse32.v v30, (t0)",
            "addi t0, {p_out}, 96",
            "vse32.v v27, (t0)",
            "addi t0, {p_out}, 112",
            "vse32.v v31, (t0)",

            pa = inout(reg) pa => _,
            pb = inout(reg) pb => _,
            pxs = inout(reg) pxs => _,
            pws = inout(reg) pws => _,
            pxsm = inout(reg) pxsm => _,
            pwm = inout(reg) pwm => _,
            sg_rem = inout(reg) scale_groups => _,
            cps = in(reg) chunks_per_scale,
            p_out = in(reg) p_out,
            ta = out(reg) _,
            tb = out(reg) _,
            ch = out(reg) _,
            xs_half = out(reg) _,
            out("t0") _,
            out("t1") _,
            out("t2") _,
            out("t3") _,
            out("ft0") _,
            out("v0") _,
            out("v1") _,
            out("v2") _,
            out("v3") _,
            out("v4") _,
            out("v5") _,
            out("v6") _,
            out("v7") _,
            out("v8") _,
            out("v9") _,
            out("v10") _,
            out("v11") _,
            out("v12") _,
            out("v13") _,
            out("v14") _,
            out("v15") _,
            out("v16") _,
            out("v17") _,
            out("v18") _,
            out("v19") _,
            out("v20") _,
            out("v21") _,
            out("v22") _,
            out("v23") _,
            out("v24") _,
            out("v25") _,
            out("v26") _,
            out("v27") _,
            out("v28") _,
            out("v29") _,
            out("v30") _,
            out("v31") _,
            options(nostack),
        );
    }
}

/// Simulates `zvl256b` (`256-bit` `X100` widening RVV) vs `zvl1024b` (`1024-bit` `A100` `8x16` `IME`)
/// integer tile accumulation to verify exact cross-cluster numerical equivalence on any host.
fn tile_dot_vla_sim(
    a: &[i8],
    b: &[i8],
    k_chunks: usize,
    active_m: usize,
    vlen_bits: usize,
    c: &mut [i32; TILE_C],
) {
    c.fill(0);
    let m_bound = active_m.min(TILE_M);
    if vlen_bits >= 1024 {
        // Simulate A100 1024-bit 8x16x8 smt.vmadot full-tile accumulation
        for ch in 0..k_chunks {
            let at = &a[ch * TILE_BYTES..(ch + 1) * TILE_BYTES];
            let bt = &b[ch * TILE_BYTES..(ch + 1) * TILE_BYTES];
            for i in 0..TILE_M {
                for j in 0..TILE_N {
                    let mut acc = 0_i32;
                    for k in 0..TILE_K {
                        acc += i32::from(at[i * TILE_K + k]) * i32::from(bt[j * TILE_K + k]);
                    }
                    c[i * TILE_N + j] += acc;
                }
            }
        }
        for slot in &mut c[m_bound * TILE_N..] {
            *slot = 0;
        }
    } else {
        // Simulate X100 256-bit RVV strip-mined widening reduction per active row
        let vlmax_e8 = (vlen_bits / 8).max(1);
        for i in 0..m_bound {
            for ch in 0..k_chunks {
                let a_row = &a[ch * TILE_BYTES + i * TILE_K..ch * TILE_BYTES + (i + 1) * TILE_K];
                for j in 0..TILE_N {
                    let b_row =
                        &b[ch * TILE_BYTES + j * TILE_K..ch * TILE_BYTES + (j + 1) * TILE_K];
                    let mut idx = 0usize;
                    while idx < TILE_K {
                        let vl = (TILE_K - idx).min(vlmax_e8);
                        let mut lane_sum = 0_i32;
                        for l in 0..vl {
                            lane_sum += i32::from(a_row[idx + l]) * i32::from(b_row[idx + l]);
                        }
                        c[i * TILE_N + j] += lane_sum;
                        idx += vl;
                    }
                }
            }
        }
    }
}

/// Computes single-token GEMV (`[1, K] x [N, K]^T -> [N]`) over repacked weights and pre-quantized
/// [`QuantizedActQ8`] (`act.m() == 1`).
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `act.m() != 1`, `act.k() != weights.cols()`, or
/// `out.len() != weights.rows()`.
pub fn gemv_repacked(
    out: &mut [f32],
    weights: &RepackedWeight,
    act: &QuantizedActQ8,
) -> Result<(), K3Error> {
    gemv_repacked_rows(out, weights, act, 0..weights.rows())
}

/// Computes row-sliced single-token GEMV (`row_range` within `0..weights.rows()`) for lock-free
/// multi-threaded worker-pool partitioning.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `act.m() != 1`, `row_range` is out of bounds, or
/// dimensions mismatch.
pub fn gemv_repacked_rows(
    out: &mut [f32],
    weights: &RepackedWeight,
    act: &QuantizedActQ8,
    row_range: Range<usize>,
) -> Result<(), K3Error> {
    if act.m() != 1 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "gemv_repacked".to_owned(),
            offset: 0,
            byte_len: size_of_val(out),
            buffer_len: weights.aligned_bytes(),
            reason: format!("gemv_repacked requires act.m() == 1, got {}", act.m()),
        });
    }
    execute_gemm_repacked_rows(out, weights, act, row_range, None)
}

/// Computes batched multi-token prefill GEMM (`[M, K] x [N, K]^T -> [M, N]`) over repacked weights
/// and pre-quantized [`QuantizedActQ8`], amortizing each weight tile load across `M` tokens.
///
/// Dispatches to `A100` `smt.vmadot` (`XSMTVDotII`) when `vlenb == 128`, `X100`-safe RVV 1.0
/// widening reductions when `vlenb != 128` on `riscv64`, and portable scalar on host / Miri.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `act.k() != weights.cols()` or
/// `out.len() != act.m() * weights.rows()`.
pub fn gemm_repacked(
    out: &mut [f32],
    weights: &RepackedWeight,
    act: &QuantizedActQ8,
) -> Result<(), K3Error> {
    gemm_repacked_rows(out, weights, act, 0..weights.rows())
}

/// Computes row-sliced batched multi-token GEMM (`[M, K] x [row_range, K]^T -> [M, row_range.len()]`)
/// without shared mutable output contention across worker threads.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `row_range` is out of bounds or dimensions mismatch.
pub fn gemm_repacked_rows(
    out: &mut [f32],
    weights: &RepackedWeight,
    act: &QuantizedActQ8,
    row_range: Range<usize>,
) -> Result<(), K3Error> {
    execute_gemm_repacked_rows(out, weights, act, row_range, None)
}

/// Convenience single-token GEMV accepting a raw `&[f32]` activation slice (`[K]`).
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] on dimension mismatch.
pub fn gemv_repacked_f32(
    out: &mut [f32],
    weights: &RepackedWeight,
    input: &[f32],
) -> Result<(), K3Error> {
    gemv_repacked_f32_rows(out, weights, input, 0..weights.rows())
}

/// Convenience row-sliced single-token GEMV accepting a raw `&[f32]` activation slice (`[K]`).
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] on dimension or range mismatch.
pub fn gemv_repacked_f32_rows(
    out: &mut [f32],
    weights: &RepackedWeight,
    input: &[f32],
    row_range: Range<usize>,
) -> Result<(), K3Error> {
    let act = QuantizedActQ8::quantize_row(input)?;
    gemv_repacked_rows(out, weights, &act, row_range)
}

/// Convenience batched multi-token GEMM accepting a raw `&[f32]` activation matrix (`[M, K]`).
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] on dimension mismatch.
pub fn gemm_repacked_f32(
    out: &mut [f32],
    weights: &RepackedWeight,
    input: &[f32],
    m: usize,
) -> Result<(), K3Error> {
    gemm_repacked_f32_rows(out, weights, input, m, 0..weights.rows())
}

/// Convenience row-sliced batched multi-token GEMM accepting a raw `&[f32]` activation matrix (`[M, K]`).
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] on dimension or range mismatch.
pub fn gemm_repacked_f32_rows(
    out: &mut [f32],
    weights: &RepackedWeight,
    input: &[f32],
    m: usize,
    row_range: Range<usize>,
) -> Result<(), K3Error> {
    if m == 0 || weights.cols() == 0 || input.len() != m.saturating_mul(weights.cols()) {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "gemm_repacked_f32".to_owned(),
            offset: 0,
            byte_len: size_of_val(input),
            buffer_len: m.saturating_mul(weights.cols()) * size_of::<f32>(),
            reason: format!(
                "input length {} != m ({m}) * cols ({})",
                input.len(),
                weights.cols()
            ),
        });
    }
    let act = QuantizedActQ8::quantize(input, m, weights.cols())?;
    gemm_repacked_rows(out, weights, &act, row_range)
}

/// Executes [`gemm_repacked`] under an explicit simulated hardware vector width (`256` for `X100`
/// `zvl256b` or `1024` for `A100` `zvl1024b` / `IME`) to verify cross-cluster numerical parity.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] on dimension mismatch.
pub fn gemm_repacked_vla_sim(
    out: &mut [f32],
    weights: &RepackedWeight,
    act: &QuantizedActQ8,
    vlen_bits: usize,
) -> Result<(), K3Error> {
    execute_gemm_repacked_rows(out, weights, act, 0..weights.rows(), Some(vlen_bits))
}

/// Compact metadata storage backing [`PackedNibbleMatrix`] (`1.00x` raw GGUF footprint when
/// `rows % 8 == 0`, always `<= 1.05x` for realistic matrix dimensions).
#[derive(Debug, Clone)]
enum PackedNibbleMeta {
    Q4_0 {
        /// `[nt: 0..n_tiles][b32: 0..k_blocks32][j: 0..8]` IEEE `f16` bits.
        scales_f16: Vec<u16>,
    },
    Q4_1 {
        /// `[nt][b32][j]` IEEE `f16` bits for `d` and `m`.
        scales_f16: Vec<u16>,
        mins_f16: Vec<u16>,
    },
    IQ4NL {
        /// `[nt][b32][j]` IEEE `f16` bits.
        scales_f16: Vec<u16>,
    },
    Q4KM {
        /// `[nt: 0..n_tiles][sb: 0..k_blocks32 / 8][j: 0..8]` raw 16-byte `Q4_K_M` headers (`d`,
        /// `dmin`, and 12 packed 6-bit scale/min bytes).
        headers: Vec<[u8; 16]>,
    },
}

/// Compact 4-bit `128-byte`-aligned weight storage retaining two quantized weights per byte
/// (`#87`/`#88`), expanding nibbles on the fly with `smt.vpack.vv` (`.word 0x6704102b`) +
/// `smt.vmadot` on `A100` (`VLEN = 1024`) to eliminate the `2x` RAM expansion of `i8` tiles.
#[derive(Debug, Clone)]
pub struct PackedNibbleMatrix {
    rows: usize,
    cols: usize,
    k_blocks32: usize,
    n_tiles: usize,
    quant: QuantType,
    /// Layout: `[nt: 0..n_tiles][b32: 0..k_blocks32]` (`128` bytes = `8 rows x 32 nibbles`).
    /// Bytes `0..64` store columns `0..15`; bytes `64..128` store columns `16..31`. Within each
    /// 64-byte half, segment `pair = 0..4` interleaves even row `2*pair` in low nibbles and odd
    /// row `2*pair + 1` in high nibbles so one `smt.vpack.vv v0, v8, v16, 1` (`0x6704102b`) builds
    /// both `8x16` `A100` `B` tiles simultaneously.
    nibble_tiles: Vec<Aligned128>,
    meta: PackedNibbleMeta,
}

fn validate_nibble_quant(weights: &TensorView<'_>, op: &str) -> Result<(), K3Error> {
    match weights.quant() {
        QuantType::Q4_0 | QuantType::Q4_K_M | QuantType::Q4_1 | QuantType::IQ4_NL => Ok(()),
        other => Err(K3Error::unsupported_tensor_quant(op, weights.name(), other)),
    }
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[inline]
fn pack_nibble_halves_16_rvv(
    even_bytes: &[u8; 16],
    odd_bytes: &[u8; 16],
    out_lo_tile: &mut [u8; 16],
    out_hi_tile: &mut [u8; 16],
) {
    // SAFETY: all slices have exact length 16; `vl = 16, e8, m1` satisfies `VLMAX` on all RVV
    // 1.0 CPUs (`VLEN >= 128`).
    unsafe {
        core::arch::asm!(
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vle8.v v0, ({pe})",
            "vle8.v v1, ({po})",
            "vand.vi v2, v0, 0x0F",
            "vand.vi v3, v1, 0x0F",
            "vsll.vi v3, v3, 4",
            "vor.vv v2, v2, v3",
            "vse8.v v2, ({d_lo})",
            "vsrl.vi v4, v0, 4",
            "li {mask}, 0xF0",
            "vand.vx v5, v1, {mask}",
            "vor.vv v4, v4, v5",
            "vse8.v v4, ({d_hi})",
            pe = in(reg) even_bytes.as_ptr(),
            po = in(reg) odd_bytes.as_ptr(),
            d_lo = in(reg) out_lo_tile.as_mut_ptr(),
            d_hi = in(reg) out_hi_tile.as_mut_ptr(),
            mask = out(reg) _,
            out("v0") _,
            out("v1") _,
            out("v2") _,
            out("v3") _,
            out("v4") _,
            out("v5") _,
            options(nostack),
        );
    }
}

#[inline]
fn pack_pair_split_nibbles_into_tile(
    even_nibbles: &[u8; 16],
    odd_nibbles: &[u8; 16],
    tile_lo: &mut [u8],
    tile_hi: &mut [u8],
    pair: usize,
) {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        let dst_lo: &mut [u8; 16] = (&mut tile_lo[pair * 16..(pair + 1) * 16])
            .try_into()
            .expect("16B pair lo slice");
        let dst_hi: &mut [u8; 16] = (&mut tile_hi[pair * 16..(pair + 1) * 16])
            .try_into()
            .expect("16B pair hi slice");
        pack_nibble_halves_16_rvv(even_nibbles, odd_nibbles, dst_lo, dst_hi);
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        let off = pair * 16;
        for k in 0..16 {
            let e = even_nibbles[k];
            let o = odd_nibbles[k];
            tile_lo[off + k] = (e & 0x0F) | ((o & 0x0F) << 4);
            tile_hi[off + k] = (e >> 4) | (o & 0xF0);
        }
    }
}

#[inline]
fn pack_q4k_pair_chunk32_into_tiles(
    even_qs: &[u8; 32],
    odd_qs: &[u8; 32],
    tile_lo_blk: &mut Aligned128,
    tile_hi_blk: &mut Aligned128,
    pair: usize,
) {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        // SAFETY: `even_qs`/`odd_qs` cover 32 bytes; `tile_{lo,hi}_blk` cover 128 bytes;
        // `vl = 16, e8, m1` fits inside `VLEN >= 128`.
        unsafe {
            let pe0 = even_qs.as_ptr();
            let pe1 = even_qs.as_ptr().add(16);
            let po0 = odd_qs.as_ptr();
            let po1 = odd_qs.as_ptr().add(16);
            let d_lo0 = tile_lo_blk.as_bytes_mut().as_mut_ptr().add(pair * 16);
            let d_lo1 = tile_lo_blk.as_bytes_mut().as_mut_ptr().add(64 + pair * 16);
            let d_hi0 = tile_hi_blk.as_bytes_mut().as_mut_ptr().add(pair * 16);
            let d_hi1 = tile_hi_blk.as_bytes_mut().as_mut_ptr().add(64 + pair * 16);
            core::arch::asm!(
                "vsetivli zero, 16, e8, m1, ta, ma",
                "li {mask}, 0xF0",
                // cols 0..15 -> tile_lo[0..64], tile_hi[0..64]
                "vle8.v v0, ({pe0})",
                "vle8.v v1, ({po0})",
                "vand.vi v2, v0, 0x0F",
                "vand.vi v3, v1, 0x0F",
                "vsll.vi v3, v3, 4",
                "vor.vv v2, v2, v3",
                "vse8.v v2, ({d_lo0})",
                "vsrl.vi v4, v0, 4",
                "vand.vx v5, v1, {mask}",
                "vor.vv v4, v4, v5",
                "vse8.v v4, ({d_hi0})",
                // cols 16..31 -> tile_lo[64..128], tile_hi[64..128]
                "vle8.v v0, ({pe1})",
                "vle8.v v1, ({po1})",
                "vand.vi v2, v0, 0x0F",
                "vand.vi v3, v1, 0x0F",
                "vsll.vi v3, v3, 4",
                "vor.vv v2, v2, v3",
                "vse8.v v2, ({d_lo1})",
                "vsrl.vi v4, v0, 4",
                "vand.vx v5, v1, {mask}",
                "vor.vv v4, v4, v5",
                "vse8.v v4, ({d_hi1})",
                pe0 = in(reg) pe0,
                pe1 = in(reg) pe1,
                po0 = in(reg) po0,
                po1 = in(reg) po1,
                d_lo0 = in(reg) d_lo0,
                d_lo1 = in(reg) d_lo1,
                d_hi0 = in(reg) d_hi0,
                d_hi1 = in(reg) d_hi1,
                mask = out(reg) _,
                out("v0") _,
                out("v1") _,
                out("v2") _,
                out("v3") _,
                out("v4") _,
                out("v5") _,
                options(nostack),
            );
        }
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        let lo_bytes = tile_lo_blk.as_bytes_mut();
        let hi_bytes = tile_hi_blk.as_bytes_mut();
        for k in 0..16 {
            let e0 = even_qs[k];
            let o0 = odd_qs[k];
            lo_bytes[pair * 16 + k] = (e0 & 0x0F) | ((o0 & 0x0F) << 4);
            hi_bytes[pair * 16 + k] = (e0 >> 4) | (o0 & 0xF0);

            let e1 = even_qs[16 + k];
            let o1 = odd_qs[16 + k];
            lo_bytes[64 + pair * 16 + k] = (e1 & 0x0F) | ((o1 & 0x0F) << 4);
            hi_bytes[64 + pair * 16 + k] = (e1 >> 4) | (o1 & 0xF0);
        }
    }
}

/// Repacks a `tile_range` slice of compact 4-bit `Aligned128` nibble tiles into caller-supplied
/// disjoint slices without heap allocations (`#88`, `#89`).
///
/// Metadata requirements by quant format:
/// - `Q4_0` / `IQ4_NL`: `out_scales_f16` (`tile_range.len() * k_blocks32 * 8`), `out_mins_f16 = None`,
///   `out_q4k_headers = None`.
/// - `Q4_1`: `out_scales_f16` and `out_mins_f16 = Some(...)` (both length `tile_range.len() * k_blocks32 * 8`).
/// - `Q4_K_M`: `out_q4k_headers = Some(...)` (length `tile_range.len() * (k_blocks32 / 8) * 8`),
///   with empty `&mut []` for `out_scales_f16`.
///
/// # Errors
///
/// Returns [`K3Error::CapabilityMismatch`] on unsupported formats, or [`K3Error::TensorBoundsError`]
/// on dimension or output slice mismatch.
#[expect(
    clippy::too_many_lines,
    clippy::too_many_arguments,
    clippy::missing_panics_doc,
    reason = "covers Q4_0, Q4_1, IQ4_NL, and Q4_K_M compact tile slices with fixed-size sub-block arrays"
)]
pub fn repack_nibble_into_tile_range(
    weights: &TensorView<'_>,
    rows: usize,
    cols: usize,
    tile_range: Range<usize>,
    out_tiles: &mut [Aligned128],
    out_scales_f16: &mut [u16],
    mut out_mins_f16: Option<&mut [u16]>,
    mut out_q4k_headers: Option<&mut [[u8; 16]]>,
) -> Result<(), K3Error> {
    validate_nibble_quant(weights, "repack_nibble_into_tile_range")?;
    let row_bytes = validate_matrix_view(weights, rows, cols)?;
    let n_tiles = tile_count(rows);
    let k_blocks32 = cols / ACT_Q8_GROUP_ELEMENTS;
    if tile_range.start > tile_range.end || tile_range.end > n_tiles {
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: tile_range.start,
            byte_len: tile_range.len(),
            buffer_len: n_tiles,
            reason: format!(
                "tile_range {}..{} exceeds n_tiles ({n_tiles})",
                tile_range.start, tile_range.end
            ),
        });
    }
    let sub_tiles = tile_range.len();
    let expected_tiles = sub_tiles * k_blocks32;
    if out_tiles.len() != expected_tiles {
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: tile_range.start,
            byte_len: out_tiles.len(),
            buffer_len: expected_tiles,
            reason: format!(
                "nibble out_tiles length {} != {expected_tiles}",
                out_tiles.len()
            ),
        });
    }

    let quant = weights.quant();
    let meta_per_tile = k_blocks32 * TILE_N;
    match quant {
        QuantType::Q4_0 | QuantType::IQ4_NL => {
            if out_scales_f16.len() != sub_tiles * meta_per_tile {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: weights.name().to_owned(),
                    offset: 0,
                    byte_len: out_scales_f16.len(),
                    buffer_len: sub_tiles * meta_per_tile,
                    reason: "out_scales_f16 slice length mismatch".to_owned(),
                });
            }
            out_scales_f16.fill(0);
        }
        QuantType::Q4_1 => {
            if out_scales_f16.len() != sub_tiles * meta_per_tile {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: weights.name().to_owned(),
                    offset: 0,
                    byte_len: out_scales_f16.len(),
                    buffer_len: sub_tiles * meta_per_tile,
                    reason: "out_scales_f16 slice length mismatch".to_owned(),
                });
            }
            let Some(mins) = out_mins_f16.as_deref_mut() else {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: weights.name().to_owned(),
                    offset: 0,
                    byte_len: 0,
                    buffer_len: sub_tiles * meta_per_tile,
                    reason: "Q4_1 requires out_mins_f16 slice".to_owned(),
                });
            };
            if mins.len() != sub_tiles * meta_per_tile {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: weights.name().to_owned(),
                    offset: 0,
                    byte_len: mins.len(),
                    buffer_len: sub_tiles * meta_per_tile,
                    reason: "out_mins_f16 slice length mismatch".to_owned(),
                });
            }
            out_scales_f16.fill(0);
            mins.fill(0);
        }
        QuantType::Q4_K_M => {
            let n_sblks = cols / Q4_K_M_BLOCK_ELEMENTS;
            let expected_hdrs = sub_tiles * n_sblks * TILE_N;
            let Some(hdrs) = out_q4k_headers.as_deref_mut() else {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: weights.name().to_owned(),
                    offset: 0,
                    byte_len: 0,
                    buffer_len: expected_hdrs,
                    reason: "Q4_K_M requires out_q4k_headers slice".to_owned(),
                });
            };
            if hdrs.len() != expected_hdrs {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: weights.name().to_owned(),
                    offset: 0,
                    byte_len: hdrs.len(),
                    buffer_len: expected_hdrs,
                    reason: "out_q4k_headers slice length mismatch".to_owned(),
                });
            }
            hdrs.fill([0_u8; 16]);
        }
        _ => unreachable!(),
    }

    for tile in out_tiles.iter_mut() {
        tile.0.fill(0);
    }

    let raw = weights.as_bytes();
    let zero_16 = [0_u8; 16];
    let zero_32 = [0_u8; 32];

    for (local_nt, nt) in tile_range.enumerate() {
        let tile_slice = &mut out_tiles[local_nt * k_blocks32..(local_nt + 1) * k_blocks32];
        for pair in 0..4 {
            let row_even = nt * TILE_N + 2 * pair;
            if row_even >= rows {
                break;
            }
            let row_odd = row_even + 1;
            let has_odd = row_odd < rows;
            let even_raw = &raw[row_even * row_bytes..(row_even + 1) * row_bytes];
            let odd_raw = if has_odd {
                &raw[row_odd * row_bytes..(row_odd + 1) * row_bytes]
            } else {
                &[][..]
            };

            match quant {
                QuantType::Q4_0 | QuantType::IQ4_NL => {
                    for b in 0..k_blocks32 {
                        let be = &even_raw[b * 18..(b + 1) * 18];
                        let even_bits = u16::from_le_bytes([be[0], be[1]]);
                        let even_zero = f16_to_f32_finite(even_bits) == 0.0;
                        out_scales_f16[(local_nt * k_blocks32 + b) * TILE_N + 2 * pair] =
                            if even_zero { 0 } else { even_bits };

                        let (odd_zero, nib_odd) = if has_odd {
                            let bo = &odd_raw[b * 18..(b + 1) * 18];
                            let odd_bits = u16::from_le_bytes([bo[0], bo[1]]);
                            let oz = f16_to_f32_finite(odd_bits) == 0.0;
                            out_scales_f16[(local_nt * k_blocks32 + b) * TILE_N + 2 * pair + 1] =
                                if oz { 0 } else { odd_bits };
                            if oz {
                                (true, &zero_16)
                            } else {
                                let o_arr: &[u8; 16] = bo[2..18].try_into().expect("16B");
                                (false, o_arr)
                            }
                        } else {
                            (true, &zero_16)
                        };
                        let nib_even: &[u8; 16] = if even_zero {
                            &zero_16
                        } else {
                            be[2..18].try_into().expect("16B")
                        };
                        if !even_zero || !odd_zero {
                            let (lo, hi) = tile_slice[b].0.split_at_mut(64);
                            pack_pair_split_nibbles_into_tile(nib_even, nib_odd, lo, hi, pair);
                        }
                    }
                }
                QuantType::Q4_1 => {
                    let mins = out_mins_f16.as_deref_mut().expect("Q4_1 mins");
                    for b in 0..k_blocks32 {
                        let be = &even_raw[b * 20..(b + 1) * 20];
                        let sc_idx_e = (local_nt * k_blocks32 + b) * TILE_N + 2 * pair;
                        out_scales_f16[sc_idx_e] = u16::from_le_bytes([be[0], be[1]]);
                        mins[sc_idx_e] = u16::from_le_bytes([be[2], be[3]]);
                        let nib_even: &[u8; 16] = be[4..20].try_into().expect("16B");
                        let nib_odd: &[u8; 16] = if has_odd {
                            let bo = &odd_raw[b * 20..(b + 1) * 20];
                            let sc_idx_o = sc_idx_e + 1;
                            out_scales_f16[sc_idx_o] = u16::from_le_bytes([bo[0], bo[1]]);
                            mins[sc_idx_o] = u16::from_le_bytes([bo[2], bo[3]]);
                            bo[4..20].try_into().expect("16B")
                        } else {
                            &zero_16
                        };
                        let (lo, hi) = tile_slice[b].0.split_at_mut(64);
                        pack_pair_split_nibbles_into_tile(nib_even, nib_odd, lo, hi, pair);
                    }
                }
                QuantType::Q4_K_M => {
                    let n_sblks = cols / Q4_K_M_BLOCK_ELEMENTS;
                    let hdrs = out_q4k_headers.as_deref_mut().expect("Q4_K_M headers");
                    for sb in 0..n_sblks {
                        let be = &even_raw[sb * Q4_K_M_BLOCK_BYTES..(sb + 1) * Q4_K_M_BLOCK_BYTES];
                        hdrs[(local_nt * n_sblks + sb) * TILE_N + 2 * pair]
                            .copy_from_slice(&be[..16]);
                        let bo = if has_odd {
                            let b_slice =
                                &odd_raw[sb * Q4_K_M_BLOCK_BYTES..(sb + 1) * Q4_K_M_BLOCK_BYTES];
                            hdrs[(local_nt * n_sblks + sb) * TILE_N + 2 * pair + 1]
                                .copy_from_slice(&b_slice[..16]);
                            Some(b_slice)
                        } else {
                            None
                        };
                        for chunk in 0..4 {
                            let b_lo = sb * 8 + chunk * 2;
                            let b_hi = b_lo + 1;
                            let (left, right) = tile_slice.split_at_mut(b_hi);
                            let tile_lo = &mut left[b_lo];
                            let tile_hi = &mut right[0];
                            let qs_off = 16 + chunk * 32;
                            let qe: &[u8; 32] = be[qs_off..qs_off + 32].try_into().expect("32B");
                            let qo: &[u8; 32] = bo.map_or(&zero_32, |b| {
                                b[qs_off..qs_off + 32].try_into().expect("32B")
                            });
                            pack_q4k_pair_chunk32_into_tiles(qe, qo, tile_lo, tile_hi, pair);
                        }
                    }
                }
                _ => unreachable!(),
            }
        }
    }

    Ok(())
}

impl PackedNibbleMatrix {
    /// Repacks a 2D [`TensorView`] (`Q4_0`, `Q4_K_M`, `Q4_1`, or `IQ4_NL`) into compact `1.00x`
    /// nibble storage for `A100` `smt.vpack.vv` + `smt.vmadot` GEMM (`#88`).
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::CapabilityMismatch`] on non-nibble formats or
    /// [`K3Error::TensorBoundsError`] on shape/byte mismatches.
    pub fn repack(weights: &TensorView<'_>) -> Result<Self, K3Error> {
        let shape = weights.shape();
        if shape.len() != 2 {
            return Err(K3Error::TensorBoundsError {
                tensor_name: weights.name().to_owned(),
                offset: weights.offset(),
                byte_len: weights.byte_len(),
                buffer_len: weights.as_bytes().len(),
                reason: format!(
                    "PackedNibbleMatrix::repack requires a 2D tensor, got {}D",
                    shape.len()
                ),
            });
        }
        let (d0, d1) = (shape[0], shape[1]);
        let blk = weights.quant().block_elements();
        let (rows, cols) = if d0 % blk == 0 && d1 % blk != 0 {
            (d1, d0)
        } else if d1 % blk == 0 && d0 % blk != 0 {
            (d0, d1)
        } else {
            (d1, d0)
        };
        Self::from_tensor(weights, rows, cols)
    }

    /// Assembles a `Q4_0` or `IQ4_NL` [`PackedNibbleMatrix`] from pre-packed buffers (`#88`/`#89`).
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::CapabilityMismatch`] on other quant types or [`K3Error::TensorBoundsError`]
    /// if dimensions or buffer lengths mismatch.
    pub fn from_scale_f16_buffers(
        quant: QuantType,
        rows: usize,
        cols: usize,
        nibble_tiles: Vec<Aligned128>,
        scales_f16: Vec<u16>,
    ) -> Result<Self, K3Error> {
        if !matches!(quant, QuantType::Q4_0 | QuantType::IQ4_NL) {
            return Err(K3Error::unsupported_tensor_quant(
                "PackedNibbleMatrix::from_scale_f16_buffers",
                "buffers",
                quant,
            ));
        }
        let n_tiles = tile_count(rows);
        if rows == 0 || cols == 0 || cols % quant.block_elements() != 0 {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "PackedNibbleMatrix".to_owned(),
                offset: 0,
                byte_len: rows,
                buffer_len: cols,
                reason: "invalid rows/cols for block format".to_owned(),
            });
        }
        let k_blocks32 = cols / ACT_Q8_GROUP_ELEMENTS;
        if nibble_tiles.len() != n_tiles * k_blocks32
            || scales_f16.len() != n_tiles * k_blocks32 * TILE_N
        {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "PackedNibbleMatrix".to_owned(),
                offset: 0,
                byte_len: nibble_tiles.len(),
                buffer_len: n_tiles * k_blocks32,
                reason: "nibble buffer lengths do not match geometry".to_owned(),
            });
        }
        let meta = if quant == QuantType::Q4_0 {
            PackedNibbleMeta::Q4_0 { scales_f16 }
        } else {
            PackedNibbleMeta::IQ4NL { scales_f16 }
        };
        Ok(Self {
            rows,
            cols,
            k_blocks32,
            n_tiles,
            quant,
            nibble_tiles,
            meta,
        })
    }

    /// Assembles a `Q4_K_M` [`PackedNibbleMatrix`] from pre-packed buffers (`#88`/`#89`).
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] if dimensions or buffer lengths mismatch.
    pub fn from_q4_k_buffers(
        rows: usize,
        cols: usize,
        nibble_tiles: Vec<Aligned128>,
        headers: Vec<[u8; 16]>,
    ) -> Result<Self, K3Error> {
        let n_tiles = tile_count(rows);
        if rows == 0 || cols == 0 || cols % Q4_K_M_BLOCK_ELEMENTS != 0 {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "PackedNibbleMatrix".to_owned(),
                offset: 0,
                byte_len: rows,
                buffer_len: cols,
                reason: "invalid rows/cols for Q4_K_M".to_owned(),
            });
        }
        let k_blocks32 = cols / ACT_Q8_GROUP_ELEMENTS;
        let n_sblks = cols / Q4_K_M_BLOCK_ELEMENTS;
        if nibble_tiles.len() != n_tiles * k_blocks32 || headers.len() != n_tiles * n_sblks * TILE_N
        {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "PackedNibbleMatrix".to_owned(),
                offset: 0,
                byte_len: nibble_tiles.len(),
                buffer_len: n_tiles * k_blocks32,
                reason: "Q4_K_M nibble buffer lengths do not match geometry".to_owned(),
            });
        }
        Ok(Self {
            rows,
            cols,
            k_blocks32,
            n_tiles,
            quant: QuantType::Q4_K_M,
            nibble_tiles,
            meta: PackedNibbleMeta::Q4KM { headers },
        })
    }

    /// Repacks `weights` interpreted as `[rows, cols]` into compact nibble tiles.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::CapabilityMismatch`] on non-nibble formats or
    /// [`K3Error::TensorBoundsError`] on dimension/byte mismatches.
    pub fn from_tensor(
        weights: &TensorView<'_>,
        rows: usize,
        cols: usize,
    ) -> Result<Self, K3Error> {
        validate_nibble_quant(weights, "PackedNibbleMatrix::from_tensor")?;
        validate_matrix_view(weights, rows, cols)?;
        let n_tiles = tile_count(rows);
        let k_blocks32 = cols / ACT_Q8_GROUP_ELEMENTS;
        let quant = weights.quant();
        let mut nibble_tiles = vec![Aligned128::zeroed(); n_tiles * k_blocks32];

        let meta = match quant {
            QuantType::Q4_0 => {
                let mut scales_f16 = vec![0_u16; n_tiles * k_blocks32 * TILE_N];
                repack_nibble_into_tile_range(
                    weights,
                    rows,
                    cols,
                    0..n_tiles,
                    &mut nibble_tiles,
                    &mut scales_f16,
                    None,
                    None,
                )?;
                PackedNibbleMeta::Q4_0 { scales_f16 }
            }
            QuantType::IQ4_NL => {
                let mut scales_f16 = vec![0_u16; n_tiles * k_blocks32 * TILE_N];
                repack_nibble_into_tile_range(
                    weights,
                    rows,
                    cols,
                    0..n_tiles,
                    &mut nibble_tiles,
                    &mut scales_f16,
                    None,
                    None,
                )?;
                PackedNibbleMeta::IQ4NL { scales_f16 }
            }
            QuantType::Q4_1 => {
                let mut scales_f16 = vec![0_u16; n_tiles * k_blocks32 * TILE_N];
                let mut mins_f16 = vec![0_u16; n_tiles * k_blocks32 * TILE_N];
                repack_nibble_into_tile_range(
                    weights,
                    rows,
                    cols,
                    0..n_tiles,
                    &mut nibble_tiles,
                    &mut scales_f16,
                    Some(&mut mins_f16),
                    None,
                )?;
                PackedNibbleMeta::Q4_1 {
                    scales_f16,
                    mins_f16,
                }
            }
            QuantType::Q4_K_M => {
                let n_sblks = cols / Q4_K_M_BLOCK_ELEMENTS;
                let mut headers = vec![[0_u8; 16]; n_tiles * n_sblks * TILE_N];
                repack_nibble_into_tile_range(
                    weights,
                    rows,
                    cols,
                    0..n_tiles,
                    &mut nibble_tiles,
                    &mut [],
                    None,
                    Some(&mut headers),
                )?;
                PackedNibbleMeta::Q4KM { headers }
            }
            _ => unreachable!(),
        };

        Ok(Self {
            rows,
            cols,
            k_blocks32,
            n_tiles,
            quant,
            nibble_tiles,
            meta,
        })
    }

    /// Output row count `N`.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Logical input column count `K`.
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Number of `8`-row output tiles (`rows.div_ceil(8)`).
    #[must_use]
    pub const fn n_tiles(&self) -> usize {
        self.n_tiles
    }

    /// Number of `32`-element blocks per row (`cols / 32`).
    #[must_use]
    pub const fn k_blocks32(&self) -> usize {
        self.k_blocks32
    }

    /// Quantization format of this compact nibble matrix.
    #[must_use]
    pub const fn quant(&self) -> QuantType {
        self.quant
    }

    /// Unpacked raw GGUF tensor size in bytes for comparison against [`Self::packed_bytes`].
    #[must_use]
    pub const fn raw_tensor_bytes(&self) -> usize {
        self.rows * (self.cols / self.quant.block_elements()) * self.quant.block_bytes()
    }

    /// Resident heap payload size in bytes (`<= 1.05x` raw GGUF when `rows % 8 == 0` or large).
    #[must_use]
    pub fn packed_bytes(&self) -> usize {
        let meta_bytes = match &self.meta {
            PackedNibbleMeta::Q4_0 { scales_f16 } | PackedNibbleMeta::IQ4NL { scales_f16 } => {
                scales_f16.len() * size_of::<u16>()
            }
            PackedNibbleMeta::Q4_1 {
                scales_f16,
                mins_f16,
            } => (scales_f16.len() + mins_f16.len()) * size_of::<u16>(),
            PackedNibbleMeta::Q4KM { headers } => headers.len() * 16,
        };
        self.nibble_tiles.len() * size_of::<Aligned128>() + meta_bytes
    }

    /// Alias for [`Self::packed_bytes`].
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.packed_bytes()
    }

    /// Base pointer to the `128-byte`-aligned packed nibble tiles (`ptr as usize % 128 == 0`).
    #[must_use]
    pub fn payload_ptr(&self) -> *const u8 {
        self.nibble_tiles.as_ptr().cast::<u8>()
    }

    /// Computes batched GEMM `[M, K] x [N, K]^T -> [M, N]` unpacking nibbles on the fly.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] on dimension mismatch.
    pub fn gemm(&self, out: &mut [f32], act: &QuantizedActQ8) -> Result<(), K3Error> {
        self.gemm_rows(out, act, 0..self.rows)
    }

    /// Computes row-sliced batched GEMM `[M, K] x [row_range, K]^T -> [M, row_range.len()]`.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] if `row_range` or buffer lengths mismatch.
    pub fn gemm_rows(
        &self,
        out: &mut [f32],
        act: &QuantizedActQ8,
        row_range: Range<usize>,
    ) -> Result<(), K3Error> {
        execute_nibble_gemm_rows(out, self, act, row_range, None)
    }

    /// Computes single-token GEMV (`act.m() == 1`).
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] if `act.m() != 1` or dimensions mismatch.
    pub fn gemv(&self, out: &mut [f32], act: &QuantizedActQ8) -> Result<(), K3Error> {
        if act.m() != 1 {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "PackedNibbleMatrix::gemv".to_owned(),
                offset: 0,
                byte_len: act.m(),
                buffer_len: 1,
                reason: format!("gemv requires act.m() == 1, got {}", act.m()),
            });
        }
        self.gemm_rows(out, act, 0..self.rows)
    }

    /// Executes [`Self::gemm`] under a simulated vector width (`256` or `1024`) to verify cluster
    /// parity on any host.
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] on dimension mismatch.
    pub fn gemm_vla_sim(
        &self,
        out: &mut [f32],
        act: &QuantizedActQ8,
        vlen_bits: usize,
    ) -> Result<(), K3Error> {
        execute_nibble_gemm_rows(out, self, act, 0..self.rows, Some(vlen_bits))
    }

    /// Decodes a single logical row `row` into `out` (`out.len() == self.cols()`).
    ///
    /// # Errors
    ///
    /// Returns [`K3Error::TensorBoundsError`] if `row >= self.rows()` or `out.len() != self.cols()`.
    pub fn dequantize_row(&self, row: usize, out: &mut [f32]) -> Result<(), K3Error> {
        if row >= self.rows || out.len() != self.cols {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "PackedNibbleMatrix::dequantize_row".to_owned(),
                offset: row,
                byte_len: out.len(),
                buffer_len: self.cols,
                reason: format!(
                    "row ({row}) >= rows ({}) or out.len() ({}) != cols ({})",
                    self.rows,
                    out.len(),
                    self.cols
                ),
            });
        }
        let nt = row / TILE_N;
        let j = row % TILE_N;
        let mut tile_scales = vec![0.0_f32; self.k_blocks32 * TILE_N];
        let mut tile_mins = vec![0.0_f32; self.k_blocks32 * TILE_N];
        let has_mins = self.decode_tile_scales(nt, &mut tile_scales, &mut tile_mins);
        let mut chunks = [Aligned128::zeroed(); 2];
        for b in 0..self.k_blocks32 {
            unpack_nibble_tile_to_chunks(
                &self.nibble_tiles[nt * self.k_blocks32 + b],
                self.quant,
                &mut chunks,
            );
            let sc = tile_scales[b * TILE_N + j];
            let mn = if has_mins {
                tile_mins[b * TILE_N + j]
            } else {
                0.0
            };
            let c0 = Aligned128::slice_as_i8(&chunks[0..1]);
            let c1 = Aligned128::slice_as_i8(&chunks[1..2]);
            for k in 0..TILE_K {
                out[b * 32 + k] = f32::from(c0[j * TILE_K + k]) * sc - mn;
                out[b * 32 + TILE_K + k] = f32::from(c1[j * TILE_K + k]) * sc - mn;
            }
        }
        Ok(())
    }

    fn decode_tile_scales(&self, nt: usize, out_scales: &mut [f32], out_mins: &mut [f32]) -> bool {
        let len = self.k_blocks32 * TILE_N;
        let base = nt * len;
        match &self.meta {
            PackedNibbleMeta::Q4_0 { scales_f16 } | PackedNibbleMeta::IQ4NL { scales_f16 } => {
                for (dst, &bits) in out_scales.iter_mut().zip(&scales_f16[base..base + len]) {
                    *dst = f16_to_f32_finite(bits);
                }
                false
            }
            PackedNibbleMeta::Q4_1 {
                scales_f16,
                mins_f16,
            } => {
                for ((dst_s, dst_m), (&sb, &mb)) in
                    out_scales.iter_mut().zip(out_mins.iter_mut()).zip(
                        scales_f16[base..base + len]
                            .iter()
                            .zip(&mins_f16[base..base + len]),
                    )
                {
                    *dst_s = f16_to_f32_finite(sb);
                    let m = f16_to_f32_finite(mb);
                    *dst_m = if m == 0.0 { 0.0 } else { -m };
                }
                true
            }
            PackedNibbleMeta::Q4KM { headers } => {
                let n_sblks = self.k_blocks32 / 8;
                let hdr_base = nt * n_sblks * TILE_N;
                for sb in 0..n_sblks {
                    for j in 0..TILE_N {
                        let hdr = &headers[hdr_base + sb * TILE_N + j];
                        let d = f16_to_f32_finite(u16::from_le_bytes([hdr[0], hdr[1]]));
                        let dmin = f16_to_f32_finite(u16::from_le_bytes([hdr[2], hdr[3]]));
                        let (sc, m) = unpack_k4_scales_mins(&hdr[4..16]);
                        for is in 0..8 {
                            let idx = (sb * 8 + is) * TILE_N + j;
                            out_scales[idx] = d * f32::from(sc[is]);
                            out_mins[idx] = dmin * f32::from(m[is]);
                        }
                    }
                }
                true
            }
        }
    }
}

fn unpack_nibble_tile_to_chunks(
    packed: &Aligned128,
    quant: QuantType,
    out_chunks: &mut [Aligned128; 2],
) {
    let bytes = packed.as_bytes();
    let (chunk0_slice, chunk1_slice) = out_chunks.split_at_mut(1);
    let ch0 = Aligned128::slice_as_i8_mut(chunk0_slice);
    let ch1 = Aligned128::slice_as_i8_mut(chunk1_slice);
    let is_q4_0 = quant == QuantType::Q4_0;
    let is_iq4nl = quant == QuantType::IQ4_NL;

    for pair in 0..4 {
        for k in 0..16 {
            let b0 = bytes[pair * 16 + k];
            let b1 = bytes[64 + pair * 16 + k];
            let e0 = i8::from_ne_bytes([b0 & 0x0F]);
            let o0 = i8::from_ne_bytes([b0 >> 4]);
            let e1 = i8::from_ne_bytes([b1 & 0x0F]);
            let o1 = i8::from_ne_bytes([b1 >> 4]);
            if is_q4_0 {
                ch0[(2 * pair) * TILE_K + k] = e0 - 8;
                ch0[(2 * pair + 1) * TILE_K + k] = o0 - 8;
                ch1[(2 * pair) * TILE_K + k] = e1 - 8;
                ch1[(2 * pair + 1) * TILE_K + k] = o1 - 8;
            } else if is_iq4nl {
                ch0[(2 * pair) * TILE_K + k] = KVALUES_IQ4NL[usize::from(b0 & 0x0F)];
                ch0[(2 * pair + 1) * TILE_K + k] = KVALUES_IQ4NL[usize::from(b0 >> 4)];
                ch1[(2 * pair) * TILE_K + k] = KVALUES_IQ4NL[usize::from(b1 & 0x0F)];
                ch1[(2 * pair + 1) * TILE_K + k] = KVALUES_IQ4NL[usize::from(b1 >> 4)];
            } else {
                ch0[(2 * pair) * TILE_K + k] = e0;
                ch0[(2 * pair + 1) * TILE_K + k] = o0;
                ch1[(2 * pair) * TILE_K + k] = e1;
                ch1[(2 * pair + 1) * TILE_K + k] = o1;
            }
        }
    }
}

/// Computes `Q4_0` prefill GEMM directly from compact [`PackedNibbleMatrix`] (`1.00x` raw size)
/// using `smt.vpack.vv` + `smt.vmadot` on `A100` and RVV/scalar fallbacks elsewhere (`#88`).
///
/// # Errors
///
/// Returns [`K3Error::CapabilityMismatch`] if `weights.quant() != QuantType::Q4_0`, or
/// [`K3Error::TensorBoundsError`] on dimension mismatch.
pub fn gemm_q4_0_ime(
    out: &mut [f32],
    weights: &PackedNibbleMatrix,
    act: &QuantizedActQ8,
) -> Result<(), K3Error> {
    if weights.quant() != QuantType::Q4_0 {
        return Err(K3Error::unsupported_tensor_quant(
            "gemm_q4_0_ime",
            "PackedNibbleMatrix",
            weights.quant(),
        ));
    }
    weights.gemm(out, act)
}

/// Computes `Q4_K_M` prefill GEMM directly from compact [`PackedNibbleMatrix`] (`1.00x` raw size)
/// using `smt.vpack.vv` + `smt.vmadot` on `A100` and RVV/scalar fallbacks elsewhere (`#88`).
///
/// # Errors
///
/// Returns [`K3Error::CapabilityMismatch`] if `weights.quant() != QuantType::Q4_K_M`, or
/// [`K3Error::TensorBoundsError`] on dimension mismatch.
pub fn gemm_q4_k_ime(
    out: &mut [f32],
    weights: &PackedNibbleMatrix,
    act: &QuantizedActQ8,
) -> Result<(), K3Error> {
    if weights.quant() != QuantType::Q4_K_M {
        return Err(K3Error::unsupported_tensor_quant(
            "gemm_q4_k_ime",
            "PackedNibbleMatrix",
            weights.quant(),
        ));
    }
    weights.gemm(out, act)
}

#[expect(
    clippy::too_many_lines,
    reason = "handles A100 smt.vpack.vv+vmadot, X100 RVV fallback, and host/Miri verification uniformly"
)]
fn execute_nibble_gemm_rows(
    out: &mut [f32],
    weights: &PackedNibbleMatrix,
    act: &QuantizedActQ8,
    row_range: Range<usize>,
    vlen_sim_bits: Option<usize>,
) -> Result<(), K3Error> {
    if act.k() != weights.cols() {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "PackedNibbleMatrix::gemm".to_owned(),
            offset: 0,
            byte_len: size_of_val(out),
            buffer_len: weights.packed_bytes(),
            reason: format!(
                "activation K ({}) != weight cols ({})",
                act.k(),
                weights.cols()
            ),
        });
    }
    if row_range.start > row_range.end || row_range.end > weights.rows() {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "PackedNibbleMatrix::gemm".to_owned(),
            offset: row_range.start,
            byte_len: row_range.len(),
            buffer_len: weights.rows(),
            reason: format!(
                "invalid row_range {}..{} for weight rows {}",
                row_range.start,
                row_range.end,
                weights.rows()
            ),
        });
    }
    let m = act.m();
    let sub_n = row_range.len();
    let expected_out = m
        .checked_mul(sub_n)
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: "PackedNibbleMatrix::gemm".to_owned(),
            offset: 0,
            byte_len: size_of_val(out),
            buffer_len: usize::MAX,
            reason: "m * row_range.len() overflow".to_owned(),
        })?;
    if out.len() != expected_out {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "PackedNibbleMatrix::gemm".to_owned(),
            offset: 0,
            byte_len: size_of_val(out),
            buffer_len: expected_out * size_of::<f32>(),
            reason: format!(
                "output length {} != m ({m}) * sub_rows ({sub_n})",
                out.len()
            ),
        });
    }
    if sub_n == 0 {
        return Ok(());
    }

    #[cfg(all(target_arch = "riscv64", not(miri)))]
    let use_hw_ime = vlen_sim_bits.is_none() && has_ime();

    let k_chunks = weights.k_blocks32 * 2;
    let a_tiles = act.tiles();
    let x_scales = act.x_scales();
    let x_sums = act.x_sums();

    let first_nt = row_range.start / TILE_N;
    let last_nt = (row_range.end - 1) / TILE_N;
    let mut tile_scales = vec![0.0_f32; weights.k_blocks32 * TILE_N];
    let mut tile_mins = vec![0.0_f32; weights.k_blocks32 * TILE_N];
    let mut acc_f32 = [0.0_f32; TILE_C];
    let mut scratch_chunks = [Aligned128::zeroed(); 2];
    let mut c_i32 = [0_i32; TILE_C];

    #[cfg(all(target_arch = "riscv64", not(miri)))]
    let iq4_lut_bytes: [u8; 16] = core::array::from_fn(|i| KVALUES_IQ4NL[i].to_ne_bytes()[0]);

    for nt in first_nt..=last_nt {
        let j_start = row_range.start.saturating_sub(nt * TILE_N);
        let j_end = (row_range.end - nt * TILE_N).min(TILE_N);
        let has_mins = weights.decode_tile_scales(nt, &mut tile_scales, &mut tile_mins);
        let b_nib_tiles =
            &weights.nibble_tiles[nt * weights.k_blocks32..(nt + 1) * weights.k_blocks32];

        for mt in 0..act.m_tiles() {
            let active_m = (m - mt * TILE_M).min(TILE_M);
            let a_row_tiles = &a_tiles[mt * k_chunks..(mt + 1) * k_chunks];
            let act_scales_tile = &x_scales[mt * weights.k_blocks32 * TILE_M..];
            let act_sums_tile = has_mins.then(|| &x_sums[mt * weights.k_blocks32 * TILE_M..]);

            if let Some(vbits) = vlen_sim_bits {
                acc_f32.fill(0.0_f32);
                for b in 0..weights.k_blocks32 {
                    unpack_nibble_tile_to_chunks(
                        &b_nib_tiles[b],
                        weights.quant,
                        &mut scratch_chunks,
                    );
                    let a_sub = Aligned128::slice_as_i8(&a_row_tiles[b * 2..b * 2 + 2]);
                    let b_sub = Aligned128::slice_as_i8(&scratch_chunks);
                    tile_dot_vla_sim(a_sub, b_sub, 2, active_m, vbits, &mut c_i32);
                    accumulate_scaled_tile_scalar(
                        &mut acc_f32,
                        &c_i32,
                        act_scales_tile,
                        &tile_scales,
                        act_sums_tile,
                        has_mins.then_some(&tile_mins[..]),
                        2,
                        b,
                        active_m,
                        j_start..j_end,
                    );
                }
            } else {
                #[cfg(all(target_arch = "riscv64", not(miri)))]
                if use_hw_ime {
                    let a_i8 = Aligned128::slice_as_i8(a_row_tiles);
                    let b_u8 = Aligned128::slice_as_bytes(b_nib_tiles);
                    let p_act_sums = act_sums_tile.map_or(core::ptr::null(), <[f32]>::as_ptr);
                    let p_tile_mins = if has_mins {
                        tile_mins.as_ptr()
                    } else {
                        core::ptr::null()
                    };
                    let use_neg8 = usize::from(weights.quant == QuantType::Q4_0);
                    let p_lut = if weights.quant == QuantType::IQ4_NL {
                        iq4_lut_bytes.as_ptr()
                    } else {
                        core::ptr::null()
                    };
                    // SAFETY: all slices cover `k_blocks32` blocks, `acc_f32` has 64 `f32`s,
                    // `has_ime()` verified `A100` (`VLEN = 1024`) `smt.vpack.vv` + `smt.vmadot`.
                    unsafe {
                        fused_nibble_gemm_ime(
                            a_i8.as_ptr(),
                            b_u8.as_ptr(),
                            act_scales_tile.as_ptr(),
                            tile_scales.as_ptr(),
                            p_act_sums,
                            p_tile_mins,
                            weights.k_blocks32,
                            use_neg8,
                            p_lut,
                            acc_f32.as_mut_ptr(),
                        );
                    }
                } else {
                    acc_f32.fill(0.0_f32);
                    for b in 0..weights.k_blocks32 {
                        unpack_nibble_tile_to_chunks(
                            &b_nib_tiles[b],
                            weights.quant,
                            &mut scratch_chunks,
                        );
                        let a_sub = Aligned128::slice_as_i8(&a_row_tiles[b * 2..b * 2 + 2]);
                        let b_sub = Aligned128::slice_as_i8(&scratch_chunks);
                        tile_dot_rvv_rows(a_sub, b_sub, 2, active_m, &mut c_i32);
                        accumulate_scaled_tile_scalar(
                            &mut acc_f32,
                            &c_i32,
                            act_scales_tile,
                            &tile_scales,
                            act_sums_tile,
                            has_mins.then_some(&tile_mins[..]),
                            2,
                            b,
                            active_m,
                            j_start..j_end,
                        );
                    }
                }

                #[cfg(any(not(target_arch = "riscv64"), miri))]
                {
                    acc_f32.fill(0.0_f32);
                    for b in 0..weights.k_blocks32 {
                        unpack_nibble_tile_to_chunks(
                            &b_nib_tiles[b],
                            weights.quant,
                            &mut scratch_chunks,
                        );
                        let a_sub = Aligned128::slice_as_i8(&a_row_tiles[b * 2..b * 2 + 2]);
                        let b_sub = Aligned128::slice_as_i8(&scratch_chunks);
                        tile_dot_scalar_rows(a_sub, b_sub, 2, active_m, &mut c_i32);
                        accumulate_scaled_tile_scalar(
                            &mut acc_f32,
                            &c_i32,
                            act_scales_tile,
                            &tile_scales,
                            act_sums_tile,
                            has_mins.then_some(&tile_mins[..]),
                            2,
                            b,
                            active_m,
                            j_start..j_end,
                        );
                    }
                }
            }

            for i in 0..active_m {
                let global_m = mt * TILE_M + i;
                let out_row = &mut out[global_m * sub_n..(global_m + 1) * sub_n];
                for j in j_start..j_end {
                    let local_n = (nt * TILE_N + j) - row_range.start;
                    out_row[local_n] = acc_f32[i * TILE_N + j];
                }
            }
        }
    }

    Ok(())
}

/// Full-`K` `A100` (`VLEN = 1024`) compact nibble GEMM kernel (`#88`): unpacks packed nibbles via
/// `vand`/`vsrl`/`vadd` + `smt.vpack.vv` (`.word 0x6704102b`) directly into `v0:v1`, executes
/// `smt.vmadot` (`0xe204312b` / `0xe214312b` or `IQ4_NL` `vrgather` + `0xe244312b` / `0xe254312b`),
/// and maintains the `8x8` `f32` accumulator in `v18:v19` across all `k_blocks32`.
#[cfg(all(target_arch = "riscv64", not(miri)))]
#[expect(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "raw kernel register bindings and inline asm"
)]
unsafe fn fused_nibble_gemm_ime(
    pa: *const i8,
    pb_nib: *const u8,
    pxs: *const f32,
    pws: *const f32,
    pxsm: *const f32,
    pwm: *const f32,
    k_blocks32: usize,
    use_neg8_bias: usize,
    p_iq4nl_lut: *const u8,
    p_out: *mut f32,
) {
    if k_blocks32 == 0 {
        // SAFETY: `p_out` points to 64 valid `f32`s per caller contract.
        unsafe {
            core::ptr::write_bytes(p_out, 0, TILE_C);
        }
        return;
    }
    // SAFETY: caller guarantees buffers match `k_blocks32`, `p_out` holds 64 `f32`s, and
    // `has_ime()` verified `A100` (`VLEN = 1024`) `XSMTVDotII` support.
    unsafe {
        core::arch::asm!(
            "li {vl}, 64",
            "vsetvli zero, {vl}, e32, m2, ta, ma",
            "vmv.v.i v18, 0",
            "vle32.v v20, ({p_idx_row})",
            "vle32.v v22, ({p_idx_col})",
            "beqz {p_lut}, 2f",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vle8.v v14, ({p_lut})",
            "2:",
            "li {vl}, 128",
            "vsetvli zero, {vl}, e8, m1, ta, ma",
            "vle8.v v24, ({pb_nib})",
            "addi {pb_nib}, {pb_nib}, 128",
            "vand.vi v8, v24, 0x0F",
            "vsrl.vi v16, v24, 4",
            "beqz {use_neg8}, 3f",
            "vadd.vi v8, v8, -8",
            "vadd.vi v16, v16, -8",
            "3:",
            // smt.vpack.vv v0, v8, v16, 1 (128-bit interleave -> v0=chunk0 B tile, v1=chunk1 B tile)
            ".word 0x6704102b",
            "li {vl}, 64",
            "vsetvli zero, {vl}, e32, m2, ta, ma",
            "vmv.v.i v2, 0",
            "li {vl}, 128",
            "vsetvli zero, {vl}, e8, m1, ta, ma",
            "bnez {p_lut}, 4f",
            "vle8.v v8, ({pa})",
            ".word 0xe204312b", // smt.vmadot v2, v8, v0, i8
            "addi {pa}, {pa}, 128",
            "vle8.v v8, ({pa})",
            ".word 0xe214312b", // smt.vmadot v2, v8, v1, i8
            "j 5f",
            "4:",
            "vrgather.vv v4, v14, v0",
            "vrgather.vv v5, v14, v1",
            "vle8.v v8, ({pa})",
            ".word 0xe244312b", // smt.vmadot v2, v8, v4, i8
            "addi {pa}, {pa}, 128",
            "vle8.v v8, ({pa})",
            ".word 0xe254312b", // smt.vmadot v2, v8, v5, i8
            "5:",
            "addi {pa}, {pa}, 128",
            "vsetivli zero, 8, e32, m1, ta, ma",
            "vle32.v v4, ({pxs})",
            "vle32.v v10, ({pws})",
            "addi {pxs}, {pxs}, 32",
            "addi {pws}, {pws}, 32",
            "li {vl}, 64",
            "vsetvli zero, {vl}, e32, m2, ta, ma",
            "vfcvt.f.x.v v26, v2",
            "vrgather.vv v6, v4, v20",
            "vrgather.vv v12, v10, v22",
            "vfmul.vv v4, v6, v12",
            "vfmacc.vv v18, v26, v4",
            "beqz {pxsm}, 6f",
            "vsetivli zero, 8, e32, m1, ta, ma",
            "vle32.v v4, ({pxsm})",
            "vle32.v v10, ({pwm})",
            "addi {pxsm}, {pxsm}, 32",
            "addi {pwm}, {pwm}, 32",
            "li {vl}, 64",
            "vsetvli zero, {vl}, e32, m2, ta, ma",
            "vrgather.vv v6, v4, v20",
            "vrgather.vv v12, v10, v22",
            "vfnmsac.vv v18, v6, v12",
            "6:",
            "addi {rem}, {rem}, -1",
            "bnez {rem}, 2b",
            "li {vl}, 64",
            "vsetvli zero, {vl}, e32, m2, ta, ma",
            "vse32.v v18, ({p_out})",
            pa = inout(reg) pa => _,
            pb_nib = inout(reg) pb_nib => _,
            pxs = inout(reg) pxs => _,
            pws = inout(reg) pws => _,
            pxsm = inout(reg) pxsm => _,
            pwm = inout(reg) pwm => _,
            rem = inout(reg) k_blocks32 => _,
            use_neg8 = in(reg) use_neg8_bias,
            p_lut = in(reg) p_iq4nl_lut,
            p_idx_row = in(reg) TILE_IDX_ROW.as_ptr(),
            p_idx_col = in(reg) TILE_IDX_COL.as_ptr(),
            p_out = in(reg) p_out,
            vl = out(reg) _,
            out("v0") _,
            out("v1") _,
            out("v2") _,
            out("v3") _,
            out("v4") _,
            out("v5") _,
            out("v6") _,
            out("v7") _,
            out("v8") _,
            out("v10") _,
            out("v11") _,
            out("v12") _,
            out("v13") _,
            out("v14") _,
            out("v16") _,
            out("v18") _,
            out("v19") _,
            out("v20") _,
            out("v21") _,
            out("v22") _,
            out("v23") _,
            out("v24") _,
            out("v26") _,
            out("v27") _,
            options(nostack),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        dequantize_q4_0_block, dequantize_q4_k_m_block, dequantize_q6_k_block,
        dequantize_q8_0_block,
    };

    #[test]
    fn aligned128_guarantees_128_byte_alignment_and_zero_copy_views() {
        assert_eq!(size_of::<Aligned128>(), 128);
        assert_eq!(align_of::<Aligned128>(), 128);

        let mut buf = vec![Aligned128::zeroed(); 3];
        let addr = buf.as_ptr() as usize;
        assert_eq!(addr % 128, 0, "1024-bit A100 / IME alignment");
        assert_eq!(addr % 32, 0, "256-bit X100 alignment");

        Aligned128::slice_as_f32_mut(&mut buf)[0] = 3.5_f32;
        assert!((Aligned128::slice_as_f32(&buf)[0] - 3.5).abs() < 1e-6);
    }

    #[test]
    fn quantize_row_q8_sanitizes_non_finite_and_preserves_sums() {
        let mut x = [0.5_f32; 40];
        x[3] = f32::NAN;
        x[7] = f32::INFINITY;
        x[9] = f32::NEG_INFINITY;
        let mut q = [0_i8; 64];
        let mut sc = [0.0_f32; 2];
        let mut sums = [0.0_f32; 2];

        quantize_row_f32_to_q8_32(&x, &mut q, &mut sc, &mut sums).expect("quantize succeeds");
        assert_eq!(q[3], 0);
        assert_eq!(q[7], 0);
        assert_eq!(q[9], 0);
        // 29 valid 0.5 elements in block 0 -> sum = 14.5
        assert!((sums[0] - 14.5).abs() < 1e-5);
        // 8 valid 0.5 elements in block 1 -> sum = 4.0
        assert!((sums[1] - 4.0).abs() < 1e-5);
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "covers Q4_K_M, Q8_0, Q4_0, and Q6_K lossless round-trip assertions"
    )]
    fn repacked_roundtrip_matches_gguf_dequantizers() {
        // Q4_K_M [3, 256]
        let mut q4k_raw = vec![0_u8; 3 * Q4_K_M_BLOCK_BYTES];
        for r in 0..3 {
            let base = r * Q4_K_M_BLOCK_BYTES;
            q4k_raw[base..base + 2].copy_from_slice(&0x3800_u16.to_le_bytes()); // d = 0.5
            q4k_raw[base + 2..base + 4].copy_from_slice(&0x3400_u16.to_le_bytes()); // dmin = 0.25
            for idx in 0..12 {
                #[expect(clippy::cast_possible_truncation, reason = "small index")]
                {
                    q4k_raw[base + 4 + idx] = ((r * 7 + idx * 5 + 3) & 0xFF) as u8;
                }
            }
            for idx in 0..128 {
                #[expect(clippy::cast_possible_truncation, reason = "small index")]
                {
                    q4k_raw[base + 16 + idx] = ((r * 13 + idx * 11 + 1) & 0xFF) as u8;
                }
            }
        }
        let view_q4k =
            TensorView::from_bytes("w_q4k", QuantType::Q4_K_M, &[256, 3], 0, &q4k_raw).unwrap();
        let repacked_q4k = RepackedWeight::repack(&view_q4k).unwrap();
        assert_eq!(repacked_q4k.rows(), 3);
        assert_eq!(repacked_q4k.cols(), 256);
        assert_eq!(repacked_q4k.payload_ptr() as usize % 128, 0);

        for r in 0..3 {
            let blk: &[u8; Q4_K_M_BLOCK_BYTES] = q4k_raw
                [r * Q4_K_M_BLOCK_BYTES..(r + 1) * Q4_K_M_BLOCK_BYTES]
                .try_into()
                .unwrap();
            let mut expected = [0.0_f32; 256];
            let mut actual = [0.0_f32; 256];
            dequantize_q4_k_m_block(blk, &mut expected);
            repacked_q4k.dequantize_row(r, &mut actual).unwrap();
            for c in 0..256 {
                assert!((actual[c] - expected[c]).abs() < 1e-6);
            }
        }

        // Q8_0 [5, 32] and Q4_0 [5, 32] and Q6_K [2, 256]
        let mut q8_raw = vec![0_u8; 5 * Q8_0_BLOCK_BYTES];
        let mut q40_bytes = vec![0_u8; 5 * Q4_0_BLOCK_BYTES];
        for r in 0..5 {
            q8_raw[r * 34..r * 34 + 2].copy_from_slice(&0x3400_u16.to_le_bytes());
            q40_bytes[r * 18..r * 18 + 2].copy_from_slice(&0x3800_u16.to_le_bytes());
            for i in 0..32 {
                #[expect(clippy::cast_possible_truncation, reason = "bounded test byte")]
                {
                    q8_raw[r * 34 + 2 + i] = ((r * 17 + i * 3) as u8).wrapping_sub(64);
                }
            }
            for i in 0..16 {
                #[expect(clippy::cast_possible_truncation, reason = "bounded test byte")]
                {
                    q40_bytes[r * 18 + 2 + i] =
                        (((r + i) & 0x0F) as u8) | ((((r * 3 + i) & 0x0F) as u8) << 4);
                }
            }
        }
        let rep_q8 = RepackedWeight::from_tensor(
            &TensorView::from_bytes("q8", QuantType::Q8_0, &[32, 5], 0, &q8_raw).unwrap(),
            5,
            32,
        )
        .unwrap();
        let rep_q4 = RepackedWeight::from_tensor(
            &TensorView::from_bytes("q4", QuantType::Q4_0, &[32, 5], 0, &q40_bytes).unwrap(),
            5,
            32,
        )
        .unwrap();
        for r in 0..5 {
            let mut exp8 = [0.0_f32; 32];
            let mut act8 = [0.0_f32; 32];
            dequantize_q8_0_block(q8_raw[r * 34..(r + 1) * 34].try_into().unwrap(), &mut exp8);
            rep_q8.dequantize_row(r, &mut act8).unwrap();
            assert_eq!(act8, exp8);

            let mut exp4 = [0.0_f32; 32];
            let mut act4 = [0.0_f32; 32];
            dequantize_q4_0_block(
                q40_bytes[r * 18..(r + 1) * 18].try_into().unwrap(),
                &mut exp4,
            );
            rep_q4.dequantize_row(r, &mut act4).unwrap();
            assert_eq!(act4, exp4);
        }

        let mut q6_raw = vec![0_u8; 2 * Q6_K_BLOCK_BYTES];
        for r in 0..2 {
            let b = r * Q6_K_BLOCK_BYTES;
            q6_raw[b + 208..b + 210].copy_from_slice(&0x3800_u16.to_le_bytes());
            for i in 0..208 {
                #[expect(clippy::cast_possible_truncation, reason = "bounded test byte")]
                {
                    q6_raw[b + i] = ((r * 29 + i * 7) & 0xFF) as u8;
                }
            }
        }
        let rep_q6 = RepackedWeight::repack(
            &TensorView::from_bytes("q6", QuantType::Q6_K, &[256, 2], 0, &q6_raw).unwrap(),
        )
        .unwrap();
        for r in 0..2 {
            let mut exp6 = [0.0_f32; 256];
            let mut act6 = [0.0_f32; 256];
            dequantize_q6_k_block(
                q6_raw[r * Q6_K_BLOCK_BYTES..(r + 1) * Q6_K_BLOCK_BYTES]
                    .try_into()
                    .unwrap(),
                &mut exp6,
            );
            rep_q6.dequantize_row(r, &mut act6).unwrap();
            for c in 0..256 {
                assert!((act6[c] - exp6[c]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn miri_repacked_gemv_gemm_dense_and_block_scaled_slices() {
        // 1. BlockScaledI8 Q8_0 [11, 64] with M=5 and unaligned row_range 3..10
        let rows = 11_usize;
        let cols = 64_usize;
        let mut q8_raw = vec![0_u8; rows * 2 * Q8_0_BLOCK_BYTES];
        for r in 0..rows {
            for b in 0..2 {
                let off = (r * 2 + b) * Q8_0_BLOCK_BYTES;
                q8_raw[off..off + 2].copy_from_slice(&0x3400_u16.to_le_bytes()); // d = 0.25
                for i in 0..32 {
                    #[expect(clippy::cast_possible_truncation, reason = "bounded test quant")]
                    {
                        q8_raw[off + 2 + i] = ((r * 5 + b * 11 + i) as u8).wrapping_sub(32);
                    }
                }
            }
        }
        let shape_q8 = [cols, rows];
        let view_q8 =
            TensorView::from_bytes("q8_miri", QuantType::Q8_0, &shape_q8, 0, &q8_raw).unwrap();
        let rep_q8 = RepackedWeight::repack(&view_q8).unwrap();

        let m = 5_usize;
        let act_f32: Vec<f32> = (0..m * cols)
            .map(|idx| {
                #[expect(clippy::cast_precision_loss, reason = "small test index")]
                {
                    ((idx % 23) as f32) * 0.125 - 1.25
                }
            })
            .collect();
        let act = QuantizedActQ8::quantize(&act_f32, m, cols).unwrap();

        let mut full_out = vec![0.0_f32; m * rows];
        gemm_repacked(&mut full_out, &rep_q8, &act).unwrap();

        let mut sim256 = vec![0.0_f32; m * rows];
        let mut sim1024 = vec![0.0_f32; m * rows];
        gemm_repacked_vla_sim(&mut sim256, &rep_q8, &act, 256).unwrap();
        gemm_repacked_vla_sim(&mut sim1024, &rep_q8, &act, 1024).unwrap();
        assert_eq!(sim256, sim1024);
        assert_eq!(full_out, sim256);

        let sub_range = 3..10_usize;
        let sub_len = sub_range.len();
        let mut sub_out = vec![0.0_f32; m * sub_len];
        gemm_repacked_rows(&mut sub_out, &rep_q8, &act, sub_range.clone()).unwrap();
        for tok in 0..m {
            assert_eq!(
                &sub_out[tok * sub_len..(tok + 1) * sub_len],
                &full_out[tok * rows + sub_range.start..tok * rows + sub_range.end]
            );
        }

        // Single-token GEMV matches row 0 of batched GEMM
        let mut gemv_out = vec![0.0_f32; rows];
        gemv_repacked_f32(&mut gemv_out, &rep_q8, &act_f32[..cols]).unwrap();
        for r in 0..rows {
            assert!((gemv_out[r] - full_out[r]).abs() < 1e-5);
        }

        // 2. DenseF32 vs compact DenseHalf (BF16 with threshold = 0)
        let mut bf16_raw = Vec::with_capacity(6 * 40 * 2);
        for idx in 0..(6 * 40_u16) {
            let v = (f32::from(idx % 19) - 9.0) * 0.25;
            let bits = (v.to_bits() >> 16) as u16;
            bf16_raw.extend_from_slice(&bits.to_le_bytes());
        }
        let view_bf16 =
            TensorView::from_bytes("bf16_miri", QuantType::BF16, &[40, 6], 0, &bf16_raw).unwrap();
        let rep_expanded = RepackedWeight::from_tensor(&view_bf16, 6, 40).unwrap();
        let rep_compact =
            RepackedWeight::from_tensor_with_half_limit(&view_bf16, 6, 40, 0).unwrap();
        assert!(!rep_expanded.is_compact_half());
        assert!(rep_compact.is_compact_half());

        let x_bf16: Vec<f32> = (0..(2 * 40_u16))
            .map(|i| f32::from(i % 13) * 0.5 - 2.0)
            .collect();
        let mut out_exp = [0.0_f32; 12];
        let mut out_cmp = [0.0_f32; 12];
        gemm_repacked_f32(&mut out_exp, &rep_expanded, &x_bf16, 2).unwrap();
        gemm_repacked_f32(&mut out_cmp, &rep_compact, &x_bf16, 2).unwrap();
        for i in 0..12 {
            assert!((out_exp[i] - out_cmp[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn wave2_quants_miri_repack_dequant_gemm_and_fail_closed() {
        // Test asymmetric Q4_1 / Q5_1 zero-scale non-zero-min + Q2_K 16-elem min sums + Q5_0 helper
        let mut q4_1_raw = vec![0_u8; 2 * Q4_1_BLOCK_BYTES];
        // Row 0: d=0.5, m=-0.25, nibble=(2, 5)
        q4_1_raw[0..2].copy_from_slice(&0x3800_u16.to_le_bytes());
        q4_1_raw[2..4].copy_from_slice(&0xB400_u16.to_le_bytes());
        q4_1_raw[4..20].fill(0x52);
        // Row 1: d=0.0, m=0.5 (all elements must equal +0.5)
        q4_1_raw[22..24].copy_from_slice(&0x3800_u16.to_le_bytes());
        q4_1_raw[24..40].fill(0xF1);

        let v_q4_1 = TensorView::from_bytes("q4_1_m", QuantType::Q4_1, &[32, 2], 0, &q4_1_raw)
            .expect("Q4_1 view");
        let rep_q4_1 = RepackedWeight::repack(&v_q4_1).unwrap();
        let mut row0 = [0.0_f32; 32];
        let mut row1 = [0.0_f32; 32];
        rep_q4_1.dequantize_row(0, &mut row0).unwrap();
        rep_q4_1.dequantize_row(1, &mut row1).unwrap();
        assert!(row0[..16].iter().all(|&v| (v - 0.75).abs() < 1e-6));
        assert!(row0[16..].iter().all(|&v| (v - 2.25).abs() < 1e-6));
        assert!(row1.iter().all(|&v| (v - 0.5).abs() < 1e-6));

        // Q5_0 repack_q5_0_to_ime_8x16 helper + Q5_0 fail-closed when wrong quant
        let mut q5_0_raw = vec![0_u8; 2 * Q5_0_BLOCK_BYTES];
        q5_0_raw[0..2].copy_from_slice(&0x3800_u16.to_le_bytes());
        q5_0_raw[2..6].copy_from_slice(&0xFFFF_0000_u32.to_le_bytes());
        q5_0_raw[6..22].fill(0x12);
        q5_0_raw[22..24].copy_from_slice(&0x3C00_u16.to_le_bytes());
        let v_q5_0 =
            TensorView::from_bytes("q5_0_m", QuantType::Q5_0, &[32, 2], 0, &q5_0_raw).unwrap();
        let rep_q5_0 = repack_q5_0_to_ime_8x16(&v_q5_0).unwrap();
        let mut q5_rep = [0.0_f32; 32];
        let mut q5_ten = [0.0_f32; 32];
        rep_q5_0.dequantize_row(0, &mut q5_rep).unwrap();
        crate::quant::dequantize_tensor_row(&v_q5_0, 0, &mut q5_ten).unwrap();
        assert_eq!(q5_rep, q5_ten);
        assert!(matches!(
            repack_q5_0_to_ime_8x16(&v_q4_1),
            Err(K3Error::CapabilityMismatch { .. })
        ));

        // Q2_K 16-element sub-block GEMM verifies x_sums_16 matches oracle
        let mut q2k_raw = vec![0_u8; 2 * Q2_K_BLOCK_BYTES];
        for r in 0..2 {
            let off = r * Q2_K_BLOCK_BYTES;
            for sub in 0..16 {
                let sc = u8::try_from((r * 3 + sub) % 7 + 1).unwrap();
                let mn = u8::try_from((r * 5 + sub * 2) % 9 + 1).unwrap();
                q2k_raw[off + sub] = sc | (mn << 4);
            }
            for q in 0..64 {
                #[expect(clippy::cast_possible_truncation, reason = "bounded test pattern")]
                let byte = ((r * 13 + q * 7) % 256) as u8;
                q2k_raw[off + 16 + q] = byte;
            }
            q2k_raw[off + 80..off + 82].copy_from_slice(&0x3800_u16.to_le_bytes()); // d = 0.5
            q2k_raw[off + 82..off + 84].copy_from_slice(&0x3400_u16.to_le_bytes()); // dmin = 0.25
        }
        let v_q2k =
            TensorView::from_bytes("q2k_m", QuantType::Q2_K, &[256, 2], 0, &q2k_raw).unwrap();
        let rep_q2k = RepackedWeight::repack(&v_q2k).unwrap();
        assert_eq!(rep_q2k.group_size(), 16);
        let x: Vec<f32> = (0..256)
            .map(|i| {
                #[expect(clippy::cast_precision_loss, reason = "small test index")]
                {
                    ((i % 29) as f32 - 14.0) * 0.125
                }
            })
            .collect();
        let act = QuantizedActQ8::quantize(&x, 1, 256).unwrap();
        let mut out_256 = [0.0_f32; 2];
        let mut out_1024 = [0.0_f32; 2];
        gemm_repacked_vla_sim(&mut out_256, &rep_q2k, &act, 256).unwrap();
        gemm_repacked_vla_sim(&mut out_1024, &rep_q2k, &act, 1024).unwrap();
        assert_eq!(out_256, out_1024);

        let mut row_buf = [0.0_f32; 256];
        for (r, &got) in out_256.iter().enumerate() {
            rep_q2k.dequantize_row(r, &mut row_buf).unwrap();
            let exact: f32 = row_buf.iter().zip(&x).map(|(w, a)| w * a).sum();
            assert!((got - exact).abs() < 0.25, "Q2_K row {r}: {got} vs {exact}");
        }

        // Q8_1 / MXFP4 fail closed
        for &unsup in &[QuantType::Q8_1, QuantType::MXFP4] {
            let raw = vec![0_u8; unsup.block_bytes()];
            let view = TensorView::from_bytes("unsup", unsup, &[32, 1], 0, &raw).unwrap();
            assert!(matches!(
                RepackedWeight::repack(&view),
                Err(K3Error::CapabilityMismatch { .. })
            ));
        }
    }

    #[test]
    fn wave3_packed_nibble_footprint_and_gemm_parity() {
        // Test Q4_0 [16, 64] -> exact 1.00x raw GGUF size & matches dequantized f32 GEMM.
        let rows = 16;
        let cols = 64;
        let n_blks = cols / Q4_0_BLOCK_ELEMENTS;
        let mut q4_raw = vec![0_u8; rows * n_blks * Q4_0_BLOCK_BYTES];
        for r in 0..rows {
            for b in 0..n_blks {
                let off = (r * n_blks + b) * Q4_0_BLOCK_BYTES;
                q4_raw[off..off + 2].copy_from_slice(&0x3800_u16.to_le_bytes()); // 0.5
                for i in 0..16 {
                    #[expect(clippy::cast_possible_truncation, reason = "byte pattern")]
                    {
                        q4_raw[off + 2 + i] =
                            (((r + i + b * 3) & 0x0F) | (((r * 3 + i + 5) & 0x0F) << 4)) as u8;
                    }
                }
            }
        }
        let q4_dims = [cols, rows];
        let view_q4 =
            TensorView::from_bytes("w_q4_0", QuantType::Q4_0, &q4_dims, 0, &q4_raw).unwrap();
        let nib_q4 = PackedNibbleMatrix::repack(&view_q4).unwrap();
        assert_eq!(nib_q4.packed_bytes(), nib_q4.raw_tensor_bytes());
        assert!(nib_q4.packed_bytes() * 100 <= nib_q4.raw_tensor_bytes() * 105);
        assert_eq!(nib_q4.payload_ptr() as usize % 128, 0);

        let x: Vec<f32> = (0..4 * cols)
            .map(|i| (f32::from(u8::try_from(i % 19).expect("< 19")) - 9.0) * 0.25)
            .collect();
        let act = QuantizedActQ8::quantize(&x, 4, cols).unwrap();
        let mut out_ime = vec![0.0_f32; 4 * rows];
        let mut out_256 = vec![0.0_f32; 4 * rows];
        let mut out_1024 = vec![0.0_f32; 4 * rows];
        gemm_q4_0_ime(&mut out_ime, &nib_q4, &act).unwrap();
        nib_q4.gemm_vla_sim(&mut out_256, &act, 256).unwrap();
        nib_q4.gemm_vla_sim(&mut out_1024, &act, 1024).unwrap();
        assert_eq!(out_ime, out_256);
        assert_eq!(out_256, out_1024);

        let repacked_q4 = RepackedWeight::from_tensor(&view_q4, rows, cols).unwrap();
        let mut out_repacked = vec![0.0_f32; 4 * rows];
        repacked_q4.gemm(&mut out_repacked, &act).unwrap();
        for (a, b) in out_ime.iter().zip(&out_repacked) {
            assert!(
                (a - b).abs() < 1e-4,
                "PackedNibbleMatrix vs RepackedWeight mismatch: {a} vs {b}"
            );
        }

        let mut row_w = vec![0.0_f32; cols];
        for r in 0..rows {
            nib_q4.dequantize_row(r, &mut row_w).unwrap();
            let mut exact_row = vec![0.0_f32; cols];
            for b in 0..n_blks {
                let off = (r * n_blks + b) * Q4_0_BLOCK_BYTES;
                let blk: &[u8; Q4_0_BLOCK_BYTES] =
                    q4_raw[off..off + Q4_0_BLOCK_BYTES].try_into().expect("18B");
                let out_blk: &mut [f32; 32] = (&mut exact_row[b * 32..b * 32 + 32])
                    .try_into()
                    .expect("32f32");
                dequantize_q4_0_block(blk, out_blk);
            }
            assert_eq!(row_w, exact_row);
            for m_idx in 0..4 {
                let ref_dot: f32 = exact_row
                    .iter()
                    .zip(&x[m_idx * cols..(m_idx + 1) * cols])
                    .map(|(w, a)| w * a)
                    .sum();
                let err = (out_ime[m_idx * rows + r] - ref_dot).abs();
                let tol = 0.05 * ref_dot.abs().max(1.0) + 0.25;
                assert!(
                    err < tol,
                    "Q4_0 nibble GEMM [{m_idx},{r}]: {} vs {ref_dot}",
                    out_ime[m_idx * rows + r]
                );
            }
        }

        // Wrong quant fails closed
        let mut wrong_out = vec![0.0_f32; 4 * rows];
        assert!(matches!(
            gemm_q4_k_ime(&mut wrong_out, &nib_q4, &act),
            Err(K3Error::CapabilityMismatch { .. })
        ));
    }

    #[test]
    fn wave3_parallel_tile_repack_matches_single_thread_deterministically() {
        let rows = 19; // 3 tiles: [8, 8, 3]
        let cols = 256;
        let mut q4k_raw = vec![0_u8; rows * Q4_K_M_BLOCK_BYTES];
        for r in 0..rows {
            let base = r * Q4_K_M_BLOCK_BYTES;
            q4k_raw[base..base + 2].copy_from_slice(&0x3800_u16.to_le_bytes());
            q4k_raw[base + 2..base + 4].copy_from_slice(&0x3400_u16.to_le_bytes());
            for idx in 0..12 {
                #[expect(clippy::cast_possible_truncation, reason = "small index")]
                {
                    q4k_raw[base + 4 + idx] = ((r * 7 + idx * 5 + 3) & 0xFF) as u8;
                }
            }
            for idx in 0..128 {
                #[expect(clippy::cast_possible_truncation, reason = "small index")]
                {
                    q4k_raw[base + 16 + idx] = ((r * 13 + idx * 11 + 1) & 0xFF) as u8;
                }
            }
        }
        let q4k_dims = [cols, rows];
        let view =
            TensorView::from_bytes("w_q4k", QuantType::Q4_K_M, &q4k_dims, 0, &q4k_raw).unwrap();
        let seq = RepackedWeight::from_tensor(&view, rows, cols).unwrap();
        let layout = RepackedWeight::block_scaled_layout(&view, rows, cols).unwrap();

        let mut par_tiles = vec![Aligned128::zeroed(); layout.total_tiles()];
        let mut par_scales = vec![0.0_f32; layout.total_scales()];
        let mut par_mins = vec![0.0_f32; layout.total_mins()];

        let mid_nt = 1;
        let (t0, t1) = par_tiles.split_at_mut(mid_nt * layout.k_chunks());
        let (s0, s1) = par_scales.split_at_mut(layout.scales_for_tiles(mid_nt));
        let (m0, m1) = par_mins.split_at_mut(layout.scales_for_tiles(mid_nt));

        repack_into_tile_range(&view, rows, cols, 0..mid_nt, t0, s0, Some(m0)).unwrap();
        repack_q4_k_m_parallel_slices(
            &view,
            rows,
            cols,
            mid_nt..layout.n_tiles(),
            Aligned128::slice_as_i8_mut(t1),
            s1,
            m1,
        )
        .unwrap();

        let par = RepackedWeight::from_tile_buffers(layout, par_tiles, par_scales, Some(par_mins))
            .unwrap();
        let x: Vec<f32> = (0..5 * cols)
            .map(|i| (f32::from(u8::try_from(i % 23).expect("< 23")) - 11.0) * 0.125)
            .collect();
        let act = QuantizedActQ8::quantize(&x, 5, cols).unwrap();
        let mut out_seq = vec![0.0_f32; 5 * rows];
        let mut out_par = vec![0.0_f32; 5 * rows];
        gemm_repacked(&mut out_seq, &seq, &act).unwrap();
        gemm_repacked(&mut out_par, &par, &act).unwrap();
        assert_eq!(out_seq, out_par);

        let mut row_seq = vec![0.0_f32; cols];
        let mut row_par = vec![0.0_f32; cols];
        for r in 0..rows {
            seq.dequantize_row(r, &mut row_seq).unwrap();
            par.dequantize_row(r, &mut row_par).unwrap();
            assert_eq!(row_seq, row_par);
        }
    }
}
