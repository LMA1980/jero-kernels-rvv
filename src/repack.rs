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
use crate::ime::{is_vlen_1024, tile_dot_ime, tile_dot_rvv_rows};
use crate::quant::{
    Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMENTS, Q5_K_BLOCK_BYTES, Q5_K_BLOCK_ELEMENTS, Q6_K_BLOCK_BYTES,
    Q6_K_BLOCK_ELEMENTS, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMENTS,
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
/// tiles alongside `32`-element scales, exact `f32` sums (for asymmetric `Q4_K_M` / `Q5_K` min
/// offsets), `256`-element super-block scales, and `128-byte`-aligned sanitized `f32` rows (for
/// lossless `F32` / `F16` / `BF16` projections).
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

            for ch in 0..k_chunks {
                let tile_bytes = Aligned128::slice_as_i8_mut(core::slice::from_mut(
                    &mut self.tiles[mt * k_chunks + ch],
                ));
                let dst_off = r_in_tile * TILE_K;
                let src_off = ch * TILE_K;
                tile_bytes[dst_off..dst_off + TILE_K]
                    .copy_from_slice(&row_q8[src_off..src_off + TILE_K]);
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
        /// Layout: `[n_tile: 0..n_tiles][sg: 0..scale_groups][j: 0..8]` (present for `Q4_K_M`, `Q5_K`).
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
            let matches_rc = d0 == rows && d1 == cols;
            let matches_cr = d0 == cols && d1 == rows;
            if !matches_rc && !matches_cr {
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

    let row_bytes = (cols / block_elems)
        .checked_mul(quant.block_bytes())
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: "row byte stride overflow".to_owned(),
        })?;
    let total_bytes = rows
        .checked_mul(row_bytes)
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: "matrix byte length overflow".to_owned(),
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

#[expect(
    clippy::too_many_lines,
    reason = "handles exact GGML sub-block unpacking across Q8_0, Q4_0, Q4_K_M, Q5_K, Q6_K, TQ1_0, and TQ2_0"
)]
fn repack_block_scaled_i8(
    raw: &[u8],
    quant: QuantType,
    rows: usize,
    cols: usize,
    row_bytes: usize,
) -> RepackedWeight {
    let k_padded = cols.next_multiple_of(ACT_Q8_GROUP_ELEMENTS);
    let n_tiles = tile_count(rows);
    let k_chunks = k_padded / TILE_K;
    let chunks_per_scale = if quant == QuantType::Q6_K { 1 } else { 2 };
    let scale_groups = k_chunks / chunks_per_scale;
    let has_mins = matches!(quant, QuantType::Q4_K_M | QuantType::Q5_K);

    let mut tiles = vec![Aligned128::zeroed(); n_tiles * k_chunks];
    let mut scales = vec![0.0_f32; n_tiles * scale_groups * TILE_N];
    let mut mins = has_mins.then(|| vec![0.0_f32; n_tiles * scale_groups * TILE_N]);

    let mut row_i8 = vec![0_i8; k_padded];

    for row in 0..rows {
        row_i8.fill(0);
        let row_raw = &raw[row * row_bytes..(row + 1) * row_bytes];
        let nt = row / TILE_N;
        let j = row % TILE_N;

        match quant {
            QuantType::Q8_0 => {
                let n_blks = cols / Q8_0_BLOCK_ELEMENTS;
                for b in 0..n_blks {
                    let blk = &row_raw[b * Q8_0_BLOCK_BYTES..(b + 1) * Q8_0_BLOCK_BYTES];
                    let d = f16_to_f32_finite(u16::from_le_bytes([blk[0], blk[1]]));
                    scales[(nt * scale_groups + b) * TILE_N + j] = d;
                    if d != 0.0 {
                        let dst = &mut row_i8[b * 32..(b + 1) * 32];
                        #[expect(clippy::cast_possible_wrap, reason = "u8 reinterpreted as i8")]
                        for i in 0..32 {
                            dst[i] = blk[2 + i] as i8;
                        }
                    }
                }
            }
            QuantType::Q4_0 => {
                let n_blks = cols / Q4_0_BLOCK_ELEMENTS;
                for b in 0..n_blks {
                    let blk = &row_raw[b * Q4_0_BLOCK_BYTES..(b + 1) * Q4_0_BLOCK_BYTES];
                    let d = f16_to_f32_finite(u16::from_le_bytes([blk[0], blk[1]]));
                    scales[(nt * scale_groups + b) * TILE_N + j] = d;
                    if d != 0.0 {
                        let dst = &mut row_i8[b * 32..(b + 1) * 32];
                        for i in 0..16 {
                            let byte = blk[2 + i];
                            #[expect(clippy::cast_possible_wrap, reason = "nibble - 8 fits in i8")]
                            {
                                dst[i] = ((byte & 0x0F) as i8) - 8;
                                dst[i + 16] = (((byte >> 4) & 0x0F) as i8) - 8;
                            }
                        }
                    }
                }
            }
            QuantType::Q4_K_M => {
                let n_sblks = cols / Q4_K_M_BLOCK_ELEMENTS;
                let mins_vec = mins.as_mut().expect("Q4_K_M allocates mins");
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
                        scales[(nt * scale_groups + sg0) * TILE_N + j] = d * f32::from(sc[is]);
                        mins_vec[(nt * scale_groups + sg0) * TILE_N + j] = dmin * f32::from(m[is]);
                        scales[(nt * scale_groups + sg1) * TILE_N + j] = d * f32::from(sc[is + 1]);
                        mins_vec[(nt * scale_groups + sg1) * TILE_N + j] =
                            dmin * f32::from(m[is + 1]);

                        let base = sb * 256 + chunk * 64;
                        let q_off = chunk * 32;
                        #[expect(clippy::cast_possible_wrap, reason = "unsigned 4-bit in 0..=15")]
                        for l in 0..32 {
                            row_i8[base + l] = (qs[q_off + l] & 0x0F) as i8;
                            row_i8[base + 32 + l] = ((qs[q_off + l] >> 4) & 0x0F) as i8;
                        }
                    }
                }
            }
            QuantType::Q5_K => {
                let n_sblks = cols / Q5_K_BLOCK_ELEMENTS;
                let mins_vec = mins.as_mut().expect("Q5_K allocates mins");
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
                        scales[(nt * scale_groups + sg0) * TILE_N + j] = d * f32::from(sc[is]);
                        mins_vec[(nt * scale_groups + sg0) * TILE_N + j] = dmin * f32::from(m[is]);
                        scales[(nt * scale_groups + sg1) * TILE_N + j] = d * f32::from(sc[is + 1]);
                        mins_vec[(nt * scale_groups + sg1) * TILE_N + j] =
                            dmin * f32::from(m[is + 1]);

                        let base = sb * 256 + chunk * 64;
                        let q_off = chunk * 32;
                        #[expect(clippy::cast_possible_wrap, reason = "unsigned 5-bit in 0..=31")]
                        for l in 0..32 {
                            let hb1 = if (qh[l] & u1) != 0 { 16_u8 } else { 0_u8 };
                            let hb2 = if (qh[l] & u2) != 0 { 16_u8 } else { 0_u8 };
                            row_i8[base + l] = ((qs[q_off + l] & 0x0F) + hb1) as i8;
                            row_i8[base + 32 + l] = ((qs[q_off + l] >> 4) + hb2) as i8;
                        }
                        u1 <<= 2;
                        u2 <<= 2;
                    }
                }
            }
            QuantType::Q6_K => {
                let n_sblks = cols / Q6_K_BLOCK_ELEMENTS;
                for sb in 0..n_sblks {
                    let blk = &row_raw[sb * Q6_K_BLOCK_BYTES..(sb + 1) * Q6_K_BLOCK_BYTES];
                    let d = f16_to_f32_finite(u16::from_le_bytes([blk[208], blk[209]]));
                    let ql = &blk[0..128];
                    let qh = &blk[128..192];
                    let sc_bytes = &blk[192..208];
                    for sub in 0..16 {
                        #[expect(clippy::cast_possible_wrap, reason = "signed 8-bit scale")]
                        let sc_val = f32::from(sc_bytes[sub] as i8);
                        scales[(nt * scale_groups + sb * 16 + sub) * TILE_N + j] = d * sc_val;
                    }
                    if d != 0.0 {
                        for half_idx in 0..2 {
                            let n = sb * 256 + half_idx * 128;
                            let low_off = half_idx * 64;
                            let high_off = half_idx * 32;
                            #[expect(clippy::cast_possible_wrap, reason = "6-bit quant - 32")]
                            for l in 0..32 {
                                row_i8[n + l] = (((ql[low_off + l] & 0x0F)
                                    | ((qh[high_off + l] & 0x03) << 4))
                                    as i8)
                                    - 32;
                                row_i8[n + l + 32] = (((ql[low_off + l + 32] & 0x0F)
                                    | (((qh[high_off + l] >> 2) & 0x03) << 4))
                                    as i8)
                                    - 32;
                                row_i8[n + l + 64] = ((((ql[low_off + l] >> 4) & 0x0F)
                                    | (((qh[high_off + l] >> 4) & 0x03) << 4))
                                    as i8)
                                    - 32;
                                row_i8[n + l + 96] = ((((ql[low_off + l + 32] >> 4) & 0x0F)
                                    | (((qh[high_off + l] >> 6) & 0x03) << 4))
                                    as i8)
                                    - 32;
                            }
                        }
                    }
                }
            }
            QuantType::TQ1_0 => {
                let n_sblks = cols / TQ1_0_BLOCK_ELEMENTS;
                let mut trits = [0_i8; TQ1_0_BLOCK_ELEMENTS];
                for sb in 0..n_sblks {
                    let blk: &[u8; TQ1_0_BLOCK_BYTES] = row_raw
                        [sb * TQ1_0_BLOCK_BYTES..(sb + 1) * TQ1_0_BLOCK_BYTES]
                        .try_into()
                        .expect("54-byte TQ1_0 block");
                    let d = unpack_tq1_0_block_i8(blk, &mut trits);
                    for sub in 0..8 {
                        scales[(nt * scale_groups + sb * 8 + sub) * TILE_N + j] = d;
                    }
                    row_i8[sb * 256..(sb + 1) * 256].copy_from_slice(&trits);
                }
            }
            QuantType::TQ2_0 => {
                let n_sblks = cols / TQ2_0_BLOCK_ELEMENTS;
                let mut trits = [0_i8; TQ2_0_BLOCK_ELEMENTS];
                for sb in 0..n_sblks {
                    let blk: &[u8; TQ2_0_BLOCK_BYTES] = row_raw
                        [sb * TQ2_0_BLOCK_BYTES..(sb + 1) * TQ2_0_BLOCK_BYTES]
                        .try_into()
                        .expect("66-byte TQ2_0 block");
                    let d = unpack_tq2_0_block_i8(blk, &mut trits);
                    for sub in 0..8 {
                        scales[(nt * scale_groups + sb * 8 + sub) * TILE_N + j] = d;
                    }
                    row_i8[sb * 256..(sb + 1) * 256].copy_from_slice(&trits);
                }
            }
            QuantType::F32 | QuantType::F16 | QuantType::BF16 => unreachable!(),
        }

        for ch in 0..k_chunks {
            let tile_i8 =
                Aligned128::slice_as_i8_mut(core::slice::from_mut(&mut tiles[nt * k_chunks + ch]));
            let dst_off = j * TILE_K;
            let src_off = ch * TILE_K;
            tile_i8[dst_off..dst_off + TILE_K].copy_from_slice(&row_i8[src_off..src_off + TILE_K]);
        }
    }

    RepackedWeight {
        rows,
        cols,
        k_padded,
        quant,
        storage: RepackedStorage::BlockScaledI8 {
            n_tiles,
            k_chunks,
            chunks_per_scale,
            scale_groups,
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
            | QuantType::Q4_K_M
            | QuantType::Q5_K
            | QuantType::Q6_K
            | QuantType::TQ1_0
            | QuantType::TQ2_0 => Ok(repack_block_scaled_i8(raw, quant, rows, cols, row_bytes)),
        }
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

    /// Shared-scale group size along `K` (`16` for `Q6_K`, `32` for other block-quantized formats,
    /// or `0` for dense `F32`/`F16`/`BF16`).
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
            let use_hw_ime = vlen_sim_bits.is_none() && is_vlen_1024();

            let a_tiles = act.tiles();
            let x_scales = act.x_scales();
            let x_sums = act.x_sums();
            let k_blocks32 = act.k_blocks32;

            let first_nt = row_range.start / TILE_N;
            let last_nt = (row_range.end - 1) / TILE_N;
            let mut c_i32 = [0_i32; TILE_C];
            let mut acc_f32 = [0.0_f32; TILE_C];

            // Outer loop over weight tiles `nt` keeps each weight row-tile hot in L1/L2 cache
            // across all `m_tiles` prompt tokens.
            for nt in first_nt..=last_nt {
                let j_start = row_range.start.saturating_sub(nt * TILE_N);
                let j_end = (row_range.end - nt * TILE_N).min(TILE_N);
                let b_row_tiles = &b_tiles[nt * *k_chunks..(nt + 1) * *k_chunks];

                for mt in 0..act.m_tiles() {
                    let active_m = (m - mt * TILE_M).min(TILE_M);
                    let a_row_tiles = &a_tiles[mt * *k_chunks..(mt + 1) * *k_chunks];
                    acc_f32.fill(0.0_f32);

                    for sg in 0..*scale_groups {
                        let ch_start = sg * *chunks_per_scale;
                        let ch_end = ch_start + *chunks_per_scale;
                        let a_sub = Aligned128::slice_as_i8(&a_row_tiles[ch_start..ch_end]);
                        let b_sub = Aligned128::slice_as_i8(&b_row_tiles[ch_start..ch_end]);

                        if let Some(vbits) = vlen_sim_bits {
                            tile_dot_vla_sim(
                                a_sub,
                                b_sub,
                                *chunks_per_scale,
                                active_m,
                                vbits,
                                &mut c_i32,
                            );
                        } else {
                            #[cfg(all(target_arch = "riscv64", not(miri)))]
                            if use_hw_ime {
                                // SAFETY: `a_sub` and `b_sub` hold `chunks_per_scale * 128` bytes,
                                // `c_i32` holds 64 `i32`s, and `is_vlen_1024()` verified `A100` execution.
                                unsafe {
                                    tile_dot_ime(
                                        a_sub.as_ptr(),
                                        b_sub.as_ptr(),
                                        *chunks_per_scale,
                                        c_i32.as_mut_ptr(),
                                    );
                                }
                            } else {
                                tile_dot_rvv_rows(
                                    a_sub,
                                    b_sub,
                                    *chunks_per_scale,
                                    active_m,
                                    &mut c_i32,
                                );
                            }
                            #[cfg(any(not(target_arch = "riscv64"), miri))]
                            {
                                tile_dot_scalar_rows(
                                    a_sub,
                                    b_sub,
                                    *chunks_per_scale,
                                    active_m,
                                    &mut c_i32,
                                );
                            }
                        }

                        let b32 = if *chunks_per_scale == 2 { sg } else { sg / 2 };
                        let xs = &x_scales[(mt * k_blocks32 + b32) * TILE_M..][..TILE_M];
                        let ws = &w_scales[(nt * *scale_groups + sg) * TILE_N..][..TILE_N];

                        if let Some(mins_buf) = w_mins {
                            let xsm = &x_sums[(mt * k_blocks32 + b32) * TILE_M..][..TILE_M];
                            let wm = &mins_buf[(nt * *scale_groups + sg) * TILE_N..][..TILE_N];
                            for i in 0..active_m {
                                let x_sc = xs[i];
                                let x_sum_i = xsm[i];
                                for j in j_start..j_end {
                                    #[expect(
                                        clippy::cast_precision_loss,
                                        reason = "16..32-element i8*i8 dot fits exactly in 24-bit f32 mantissa"
                                    )]
                                    let dot_f = c_i32[i * TILE_N + j] as f32;
                                    acc_f32[i * TILE_N + j] +=
                                        dot_f * (x_sc * ws[j]) - x_sum_i * wm[j];
                                }
                            }
                        } else {
                            for i in 0..active_m {
                                let x_sc = xs[i];
                                for j in j_start..j_end {
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
}
