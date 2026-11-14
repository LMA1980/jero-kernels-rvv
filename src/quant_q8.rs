//! Activation quantization (`BlockQ8_0`, `BlockQ8K`) and direct integer `Q*_x_Q8` multi-row
//! `M = 1` GEMV kernels for VLA RVV 1.0 (`VLEN=128/256/1024`) and portable scalar fallback.

use jero_core::{K3Error, QuantType, TensorView};

use crate::f16_to_f32_finite;
#[cfg(any(not(target_arch = "riscv64"), miri))]
use crate::ime::unpack_tq2_0_block_i8;
use crate::ime::{
    TQ1_0_BLOCK_BYTES, TQ1_0_BLOCK_ELEMENTS, TQ2_0_BLOCK_BYTES, TQ2_0_BLOCK_ELEMENTS, dot_i8_vector,
};
use crate::quant::{
    Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMENTS, Q5_0_BLOCK_BYTES, Q5_0_BLOCK_ELEMENTS, Q5_K_BLOCK_BYTES,
    Q5_K_BLOCK_ELEMENTS, Q6_K_BLOCK_BYTES, Q6_K_BLOCK_ELEMENTS, Q8_0_BLOCK_BYTES,
    Q8_0_BLOCK_ELEMENTS, validate_q5_k_dimensions, validate_q6_k_dimensions,
    validate_q8_0_dimensions, validate_quant_matvec_dimensions,
};
use crate::{Q4_K_M_BLOCK_BYTES, Q4_K_M_BLOCK_ELEMENTS, validate_q4_k_m_dimensions};

/// Number of scalar elements in a single `BlockQ8K` activation super-block.
pub const Q8_K_BLOCK_ELEMENTS: usize = 256;

/// Stack scratch capacity in `BlockQ8_0` blocks (`128 * 32 = 4096` elements, ~4.6 KiB) for
/// zero-allocation `M = 1` `vec_dot` calls.
#[cfg(all(target_arch = "riscv64", not(miri)))]
pub(crate) const Q8_0_STACK_BLOCKS: usize = 128;

/// Stack scratch capacity in `BlockQ8K` super-blocks (`32 * 256 = 8192` elements, ~9.3 KiB) for
/// zero-allocation `M = 1` K-quant `vec_dot` calls.
#[cfg(all(target_arch = "riscv64", not(miri)))]
pub(crate) const Q8_K_STACK_BLOCKS: usize = 32;

/// Symmetrically quantized 32-element activation block (`f32` scale + `32 x i8`) used by
/// direct integer `Q4_0`/`Q5_0`/`Q8_0` x `Q8_0` `M = 1` GEMV kernels.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockQ8_0 {
    /// Positive block scale (`absmax / 127.0`), or `0.0` for all-zero / non-finite blocks.
    pub d: f32,
    /// Symmetrically quantized signed 8-bit activation values in `[-127, 127]`.
    pub qs: [i8; Q8_0_BLOCK_ELEMENTS],
}

impl BlockQ8_0 {
    /// Zero-initialized `BlockQ8_0` constant for stack scratch buffers.
    pub const ZERO: Self = Self {
        d: 0.0,
        qs: [0; Q8_0_BLOCK_ELEMENTS],
    };
}

impl Default for BlockQ8_0 {
    fn default() -> Self {
        Self::ZERO
    }
}

/// Symmetrically quantized 256-element activation super-block (`f32` scale + `256 x i8` +
/// `[i16; 16]` sub-block sums) used by direct integer `Q*_K` and ternary `M = 1` GEMV kernels.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockQ8K {
    /// Positive super-block scale (`absmax / 127.0`), or `0.0` for all-zero blocks.
    pub d: f32,
    /// Symmetrically quantized signed 8-bit activations across 256 elements in `[-127, 127]`.
    pub qs: [i8; Q8_K_BLOCK_ELEMENTS],
    /// Precomputed 16-element sub-block integer sums (`bsums[j] = sum(qs[16*j..16*(j+1)])`).
    pub bsums: [i16; 16],
}

impl BlockQ8K {
    /// Zero-initialized `BlockQ8K` constant for stack scratch buffers.
    pub const ZERO: Self = Self {
        d: 0.0,
        qs: [0; Q8_K_BLOCK_ELEMENTS],
        bsums: [0; 16],
    };
}

impl Default for BlockQ8K {
    fn default() -> Self {
        Self::ZERO
    }
}

#[inline]
#[expect(
    clippy::cast_precision_loss,
    reason = "24-bit significand matches GGML integer-to-f32 block accumulation semantics"
)]
const fn i32_to_f32(val: i32) -> f32 {
    val as f32
}

#[inline]
#[expect(
    clippy::cast_possible_truncation,
    reason = "clamped strictly to [-127.0, 127.0] before i8 conversion"
)]
fn round_scaled_to_i8(val: f32, inv_scale: f32) -> i8 {
    if val.is_finite() {
        (val * inv_scale).round().clamp(-127.0, 127.0) as i8
    } else {
        0
    }
}

#[inline]
fn validate_activation_quant_slice(
    name: &str,
    x_len: usize,
    out_len: usize,
    block_elems: usize,
) -> Result<usize, K3Error> {
    if x_len == 0 || x_len % block_elems != 0 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: name.to_owned(),
            offset: 0,
            byte_len: x_len * size_of::<f32>(),
            buffer_len: out_len,
            reason: format!(
                "{name}: input length {x_len} must be a positive multiple of {block_elems}"
            ),
        });
    }
    let n_blocks = x_len / block_elems;
    if out_len != n_blocks {
        return Err(K3Error::TensorBoundsError {
            tensor_name: name.to_owned(),
            offset: 0,
            byte_len: x_len * size_of::<f32>(),
            buffer_len: out_len,
            reason: format!("{name}: output block count {out_len} != expected {n_blocks}"),
        });
    }
    Ok(n_blocks)
}

#[inline]
fn quantize_block_q8_0_scalar(x: &[f32], blk: &mut BlockQ8_0) {
    let amax = x
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(0.0_f32, |acc, v| acc.max(v.abs()));
    if amax == 0.0 || !amax.is_finite() {
        *blk = BlockQ8_0::ZERO;
        return;
    }
    let d = amax / 127.0_f32;
    let inv_scale = 127.0_f32 / amax;
    blk.d = d;
    for (dst, &src) in blk.qs.iter_mut().zip(x.iter()) {
        *dst = round_scaled_to_i8(src, inv_scale);
    }
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[inline]
fn finite_absmax_rvv(x: &[f32]) -> f32 {
    let ptr_x = x.as_ptr();
    let rem = x.len();
    if rem == 0 {
        return 0.0;
    }
    let zero_f = 0.0_f32;
    let fmax = f32::MAX;
    let amax: f32;
    // SAFETY: `x` holds `rem` readable `f32` elements; `vmfle.vf v0, |x|, f32::MAX` yields `1`
    // strictly for finite lanes and replaces non-finite lanes (`NaN`, `+-Inf`) with `0.0` prior to
    // `vfredmax.vs` across strip-mined `e32, m4` (`VLEN >= 128` safe).
    unsafe {
        core::arch::asm!(
            "vsetivli zero, 1, e32, m1, ta, ma",
            "vfmv.s.f v2, {z}",
            "2:",
            "vsetvli {vl}, {rem}, e32, m4, ta, ma",
            "vle32.v v8, ({px})",
            "vfabs.v v12, v8",
            "vmfle.vf v0, v12, {fmax}",
            "vfmv.v.f v16, {z}",
            "vmerge.vvm v12, v16, v12, v0",
            "vfredmax.vs v2, v12, v2",
            "slli t0, {vl}, 2",
            "add {px}, {px}, t0",
            "sub {rem}, {rem}, {vl}",
            "bnez {rem}, 2b",
            "vsetivli zero, 1, e32, m1, ta, ma",
            "vfmv.f.s {amax}, v2",
            px = inout(reg) ptr_x => _,
            rem = inout(reg) rem => _,
            z = in(freg) zero_f,
            fmax = in(freg) fmax,
            amax = out(freg) amax,
            vl = out(reg) _,
            out("t0") _,
            out("v0") _,
            out("v2") _,
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
            options(nostack, readonly),
        );
    }
    amax
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[inline]
fn quantize_block_q8_0_rvv(x: &[f32], blk: &mut BlockQ8_0) {
    let amax = finite_absmax_rvv(x);
    if !amax.is_finite() || amax == 0.0 {
        *blk = BlockQ8_0::ZERO;
        return;
    }

    let inv_scale = 127.0_f32 / amax;
    if !inv_scale.is_finite() {
        quantize_block_q8_0_scalar(x, blk);
        return;
    }
    blk.d = amax / 127.0_f32;
    let ptr_x2 = x.as_ptr();
    let ptr_q = blk.qs.as_mut_ptr();
    let rem2 = Q8_0_BLOCK_ELEMENTS;
    let zero_f = 0.0_f32;
    let fmax = f32::MAX;
    let min_q = -127.0_f32;
    let max_q = 127.0_f32;
    // SAFETY: `x` has 32 readable `f32`s and `blk.qs` has 32 writable `i8`s; `vmfle.vf` +
    // `vmerge.vvm` zeroes non-finite lanes (`NaN`, `+-Inf`) prior to scaling and clamping to
    // `[-127.0, 127.0]`. `csrrwi {saved_frm}, frm, 4` selects `RMM` (ties-away-from-zero matching
    // `f32::round`) and restores the caller's `frm`. `e32,m4` -> `e16,m2` -> `e8,m1` groups are
    // strictly disjoint and `VLEN >= 128` safe.
    unsafe {
        core::arch::asm!(
            "csrrwi {saved_frm}, frm, 4",
            "2:",
            "vsetvli {vl}, {rem}, e32, m4, ta, ma",
            "vle32.v v8, ({px})",
            "vfabs.v v12, v8",
            "vmfle.vf v0, v12, {fmax}",
            "vfmv.v.f v16, {z}",
            "vmerge.vvm v8, v16, v8, v0",
            "vfmul.vf v8, v8, {inv_s}",
            "vfmax.vf v8, v8, {min_q}",
            "vfmin.vf v8, v8, {max_q}",
            "vfcvt.x.f.v v8, v8",
            "vsetvli zero, {vl}, e16, m2, ta, ma",
            "vnsra.wi v4, v8, 0",
            "vsetvli zero, {vl}, e8, m1, ta, ma",
            "vnsra.wi v2, v4, 0",
            "vse8.v v2, ({pq})",
            "slli t0, {vl}, 2",
            "add {px}, {px}, t0",
            "add {pq}, {pq}, {vl}",
            "sub {rem}, {rem}, {vl}",
            "bnez {rem}, 2b",
            "csrrw zero, frm, {saved_frm}",
            px = inout(reg) ptr_x2 => _,
            pq = inout(reg) ptr_q => _,
            rem = inout(reg) rem2 => _,
            z = in(freg) zero_f,
            fmax = in(freg) fmax,
            inv_s = in(freg) inv_scale,
            min_q = in(freg) min_q,
            max_q = in(freg) max_q,
            saved_frm = out(reg) _,
            vl = out(reg) _,
            out("t0") _,
            out("v0") _,
            out("v2") _,
            out("v4") _,
            out("v5") _,
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
}

/// Quantizes `x` into `Q8_0` activation blocks (`32` elements/block) in caller-provided scratch.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `x` is empty, not a multiple of `32`, or
/// `out.len() != x.len() / 32`.
pub fn quantize_row_q8_0_f32(x: &[f32], out: &mut [BlockQ8_0]) -> Result<(), K3Error> {
    let _n_blocks = validate_activation_quant_slice(
        "quantize_row_q8_0_f32",
        x.len(),
        out.len(),
        Q8_0_BLOCK_ELEMENTS,
    )?;
    for (chunk, blk) in x.chunks_exact(Q8_0_BLOCK_ELEMENTS).zip(out.iter_mut()) {
        #[cfg(all(target_arch = "riscv64", not(miri)))]
        {
            quantize_block_q8_0_rvv(chunk, blk);
        }
        #[cfg(any(not(target_arch = "riscv64"), miri))]
        {
            quantize_block_q8_0_scalar(chunk, blk);
        }
    }
    Ok(())
}

/// Alias for [`quantize_row_q8_0_f32`].
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] on invalid or mismatched slice lengths.
pub fn quantize_row_q8_0(x: &[f32], out: &mut [BlockQ8_0]) -> Result<(), K3Error> {
    quantize_row_q8_0_f32(x, out)
}

#[inline]
fn quantize_block_q8_k_scalar(x: &[f32], blk: &mut BlockQ8K) {
    let amax = x
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(0.0_f32, |acc, v| acc.max(v.abs()));
    if amax == 0.0 || !amax.is_finite() {
        *blk = BlockQ8K::ZERO;
        return;
    }
    let d = amax / 127.0_f32;
    let inv_scale = 127.0_f32 / amax;
    blk.d = d;
    for (sb, ((x_sub, q_sub), bsum_slot)) in x
        .chunks_exact(16)
        .zip(blk.qs.chunks_exact_mut(16))
        .zip(blk.bsums.iter_mut())
        .enumerate()
    {
        let _ = sb;
        let mut sub_sum = 0_i16;
        for (dst, &src) in q_sub.iter_mut().zip(x_sub.iter()) {
            let q = round_scaled_to_i8(src, inv_scale);
            *dst = q;
            sub_sum += i16::from(q);
        }
        *bsum_slot = sub_sum;
    }
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[inline]
fn quantize_block_q8_k_rvv(x: &[f32], blk: &mut BlockQ8K) {
    let amax = finite_absmax_rvv(x);
    if !amax.is_finite() || amax == 0.0 {
        *blk = BlockQ8K::ZERO;
        return;
    }

    let inv_scale = 127.0_f32 / amax;
    if !inv_scale.is_finite() {
        quantize_block_q8_k_scalar(x, blk);
        return;
    }
    blk.d = amax / 127.0_f32;
    let ptr_x2 = x.as_ptr();
    let ptr_q = blk.qs.as_mut_ptr();
    let ptr_bs = blk.bsums.as_mut_ptr();
    let sb_rem = 16_usize;
    let zero_f = 0.0_f32;
    let fmax = f32::MAX;
    let min_q = -127.0_f32;
    let max_q = 127.0_f32;
    // SAFETY: `x` holds 256 readable `f32`s (`16 x 16`), `blk.qs` has 256 writable `i8`s, and
    // `blk.bsums` has 16 2-byte-aligned `i16`s. `vmfle.vf` + `vmerge.vvm` zeroes non-finite lanes
    // (`NaN`, `+-Inf`) prior to scaling, clamping to `[-127.0, 127.0]`, and rounding with `frm = RMM`
    // (`4`), guaranteeing `v4` (`i16`) equals `v2` (`i8`) and `sum(v4)` fits cleanly in `i16`
    // (`<= 2032`). All vector lengths are `16` at `e32,m4`/`e16,m2`/`e8,m1` (`<= VLMAX` on
    // `VLEN >= 128`).
    unsafe {
        core::arch::asm!(
            "csrrwi {saved_frm}, frm, 4",
            "2:",
            "vsetivli zero, 16, e32, m4, ta, ma",
            "vle32.v v8, ({px})",
            "vfabs.v v12, v8",
            "vmfle.vf v0, v12, {fmax}",
            "vfmv.v.f v16, {z}",
            "vmerge.vvm v8, v16, v8, v0",
            "vfmul.vf v8, v8, {inv_s}",
            "vfmax.vf v8, v8, {min_q}",
            "vfmin.vf v8, v8, {max_q}",
            "vfcvt.x.f.v v8, v8",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vnsra.wi v4, v8, 0",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vnsra.wi v2, v4, 0",
            "vse8.v v2, ({pq})",
            "vsetivli zero, 1, e16, m1, ta, ma",
            "vmv.v.i v0, 0",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vredsum.vs v0, v4, v0",
            "vmv.x.s t0, v0",
            "sh t0, 0({pbs})",
            "addi {px}, {px}, 64",
            "addi {pq}, {pq}, 16",
            "addi {pbs}, {pbs}, 2",
            "addi {sb}, {sb}, -1",
            "bnez {sb}, 2b",
            "csrrw zero, frm, {saved_frm}",
            px = inout(reg) ptr_x2 => _,
            pq = inout(reg) ptr_q => _,
            pbs = inout(reg) ptr_bs => _,
            sb = inout(reg) sb_rem => _,
            z = in(freg) zero_f,
            fmax = in(freg) fmax,
            inv_s = in(freg) inv_scale,
            min_q = in(freg) min_q,
            max_q = in(freg) max_q,
            saved_frm = out(reg) _,
            out("t0") _,
            out("v0") _,
            out("v2") _,
            out("v4") _,
            out("v5") _,
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
}

/// Quantizes `x` into `BlockQ8K` activation super-blocks (`256` elements/block + 16-element
/// `bsums: [i16; 16]`) once per projection in caller-provided scratch.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `x` is empty, not a multiple of `256`, or
/// `out.len() != x.len() / 256`.
pub fn quantize_row_q8_k(x: &[f32], out: &mut [BlockQ8K]) -> Result<(), K3Error> {
    let _n_blocks = validate_activation_quant_slice(
        "quantize_row_q8_k",
        x.len(),
        out.len(),
        Q8_K_BLOCK_ELEMENTS,
    )?;
    for (chunk, blk) in x.chunks_exact(Q8_K_BLOCK_ELEMENTS).zip(out.iter_mut()) {
        #[cfg(all(target_arch = "riscv64", not(miri)))]
        {
            quantize_block_q8_k_rvv(chunk, blk);
        }
        #[cfg(any(not(target_arch = "riscv64"), miri))]
        {
            quantize_block_q8_k_scalar(chunk, blk);
        }
    }
    Ok(())
}

#[inline]
fn validate_q8_block_row_bytes(
    name: &str,
    w_blocks: &[u8],
    x_len: usize,
    w_blk_bytes: usize,
) -> Result<(), K3Error> {
    if x_len == 0 || w_blocks.len() != x_len * w_blk_bytes {
        return Err(K3Error::TensorBoundsError {
            tensor_name: name.to_owned(),
            offset: 0,
            byte_len: w_blocks.len(),
            buffer_len: x_len * w_blk_bytes,
            reason: format!(
                "{name}: expected {} weight bytes for {x_len} blocks, got {}",
                x_len * w_blk_bytes,
                w_blocks.len()
            ),
        });
    }
    Ok(())
}

// ================================================================================================
// Direct Q4_0 / Q5_0 / Q8_0 x Q8_0 block & 4-row M=1 GEMV kernels
// ================================================================================================

#[cfg(any(not(target_arch = "riscv64"), miri))]
#[inline]
fn dot_q4_0_q8_0_block_scalar(w_blk: &[u8], x_blk: &BlockQ8_0) -> f32 {
    let dw = f16_to_f32_finite(u16::from_le_bytes([w_blk[0], w_blk[1]]));
    if dw == 0.0 || x_blk.d == 0.0 {
        return 0.0;
    }
    let mut sumi = 0_i32;
    for (i, &byte) in w_blk[2..18].iter().enumerate() {
        let q0 = i32::from(byte & 0x0F) - 8;
        let q1 = i32::from((byte >> 4) & 0x0F) - 8;
        sumi += q0 * i32::from(x_blk.qs[i]) + q1 * i32::from(x_blk.qs[i + 16]);
    }
    i32_to_f32(sumi) * (dw * x_blk.d)
}

#[cfg(any(not(target_arch = "riscv64"), miri))]
#[inline]
fn dot_q5_0_q8_0_block_scalar(w_blk: &[u8], x_blk: &BlockQ8_0) -> f32 {
    let dw = f16_to_f32_finite(u16::from_le_bytes([w_blk[0], w_blk[1]]));
    if dw == 0.0 || x_blk.d == 0.0 {
        return 0.0;
    }
    let qh = u32::from_le_bytes([w_blk[2], w_blk[3], w_blk[4], w_blk[5]]);
    let mut sumi = 0_i32;
    for (j, &byte) in w_blk[6..22].iter().enumerate() {
        let xh0 = i32::from(((qh >> j) & 1) as u8) << 4;
        let xh1 = i32::from(((qh >> (j + 16)) & 1) as u8) << 4;
        let q0 = (i32::from(byte & 0x0F) | xh0) - 16;
        let q1 = (i32::from((byte >> 4) & 0x0F) | xh1) - 16;
        sumi += q0 * i32::from(x_blk.qs[j]) + q1 * i32::from(x_blk.qs[j + 16]);
    }
    i32_to_f32(sumi) * (dw * x_blk.d)
}

#[cfg(any(not(target_arch = "riscv64"), miri))]
#[inline]
fn dot_q8_0_q8_0_block_scalar(w_blk: &[u8], x_blk: &BlockQ8_0) -> f32 {
    let dw = f16_to_f32_finite(u16::from_le_bytes([w_blk[0], w_blk[1]]));
    if dw == 0.0 || x_blk.d == 0.0 {
        return 0.0;
    }
    let mut sumi = 0_i32;
    for (&wb, &xq) in w_blk[2..34].iter().zip(x_blk.qs.iter()) {
        sumi += i32::from(i8::from_ne_bytes([wb])) * i32::from(xq);
    }
    i32_to_f32(sumi) * (dw * x_blk.d)
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[inline]
fn dot_q4_0_q8_0_block_i32_rvv(weight_qs: *const u8, act_qs: *const i8) -> i32 {
    let out: i32;
    // SAFETY: `weight_qs` holds 16 readable bytes and `act_qs` holds 32 readable `i8` bytes;
    // `vl = 16` at `e8,m1` -> `e16,m2` -> `e32,m1` fits all `VLEN >= 128`; `2048 << i16::MAX`.
    unsafe {
        core::arch::asm!(
            "vsetivli zero, 1, e32, m1, ta, ma",
            "vmv.v.i v1, 0",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "addi t0, {px}, 16",
            "vle8.v v16, ({px})",
            "vle8.v v17, (t0)",
            "vle8.v v5, ({pw})",
            "vand.vi v6, v5, 15",
            "vadd.vi v6, v6, -8",
            "vsrl.vi v7, v5, 4",
            "vadd.vi v7, v7, -8",
            "vwmul.vv v18, v6, v16",
            "vwmacc.vv v18, v7, v17",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v1, v18, v1",
            "vsetivli zero, 1, e32, m1, ta, ma",
            "vmv.x.s {out}, v1",
            pw = in(reg) weight_qs,
            px = in(reg) act_qs,
            out = out(reg) out,
            out("t0") _,
            out("v1") _,
            out("v5") _,
            out("v6") _,
            out("v7") _,
            out("v16") _,
            out("v17") _,
            out("v18") _,
            out("v19") _,
            options(nostack, readonly),
        );
    }
    out
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[inline]
fn dot_q4_0_q8_0_block_4x_rvv(
    pw0: *const u8,
    pw1: *const u8,
    pw2: *const u8,
    pw3: *const u8,
    px_qs: *const i8,
) -> [i32; 4] {
    let (o0, o1, o2, o3): (i32, i32, i32, i32);
    // SAFETY: Each `pw*` holds 16 readable bytes and `px_qs` holds 32 readable `i8` bytes.
    // Activation halves stay loaded in `v16`/`v17` across all 4 weight rows (`v1..v4` accumulators).
    unsafe {
        core::arch::asm!(
            "vsetivli zero, 1, e32, m1, ta, ma",
            "vmv.v.i v1, 0",
            "vmv.v.i v2, 0",
            "vmv.v.i v3, 0",
            "vmv.v.i v4, 0",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "addi t0, {px}, 16",
            "vle8.v v16, ({px})",
            "vle8.v v17, (t0)",
            // Row 0
            "vle8.v v5, ({pw0})",
            "vand.vi v6, v5, 15",
            "vadd.vi v6, v6, -8",
            "vsrl.vi v7, v5, 4",
            "vadd.vi v7, v7, -8",
            "vwmul.vv v18, v6, v16",
            "vwmacc.vv v18, v7, v17",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v1, v18, v1",
            // Row 1
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vle8.v v5, ({pw1})",
            "vand.vi v6, v5, 15",
            "vadd.vi v6, v6, -8",
            "vsrl.vi v7, v5, 4",
            "vadd.vi v7, v7, -8",
            "vwmul.vv v18, v6, v16",
            "vwmacc.vv v18, v7, v17",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v2, v18, v2",
            // Row 2
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vle8.v v5, ({pw2})",
            "vand.vi v6, v5, 15",
            "vadd.vi v6, v6, -8",
            "vsrl.vi v7, v5, 4",
            "vadd.vi v7, v7, -8",
            "vwmul.vv v18, v6, v16",
            "vwmacc.vv v18, v7, v17",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v3, v18, v3",
            // Row 3
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vle8.v v5, ({pw3})",
            "vand.vi v6, v5, 15",
            "vadd.vi v6, v6, -8",
            "vsrl.vi v7, v5, 4",
            "vadd.vi v7, v7, -8",
            "vwmul.vv v18, v6, v16",
            "vwmacc.vv v18, v7, v17",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v4, v18, v4",
            "vsetivli zero, 1, e32, m1, ta, ma",
            "vmv.x.s {o0}, v1",
            "vmv.x.s {o1}, v2",
            "vmv.x.s {o2}, v3",
            "vmv.x.s {o3}, v4",
            pw0 = in(reg) pw0,
            pw1 = in(reg) pw1,
            pw2 = in(reg) pw2,
            pw3 = in(reg) pw3,
            px = in(reg) px_qs,
            o0 = lateout(reg) o0,
            o1 = lateout(reg) o1,
            o2 = lateout(reg) o2,
            o3 = lateout(reg) o3,
            out("t0") _,
            out("v1") _,
            out("v2") _,
            out("v3") _,
            out("v4") _,
            out("v5") _,
            out("v6") _,
            out("v7") _,
            out("v16") _,
            out("v17") _,
            out("v18") _,
            out("v19") _,
            options(nostack, readonly),
        );
    }
    [o0, o1, o2, o3]
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[inline]
fn dot_q5_0_q8_0_block_i32_rvv(pw_blk: *const u8, px_qs: *const i8) -> i32 {
    let out: i32;
    let c16 = 16_usize;
    // SAFETY: `pw_blk` holds 22 readable bytes (`qh` at `+2..+6`, `qs` at `+6..+22`) and
    // `px_qs` holds 32 readable `i8`s; `v0` holds mask bits (`vle8.v` `vl=2`), `vadd.vi` `-16`
    // is in signed imm5 range, and max `|w*x|*2 <= 4096 << i16::MAX`.
    unsafe {
        core::arch::asm!(
            "vsetivli zero, 1, e32, m1, ta, ma",
            "vmv.v.i v1, 0",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "addi t0, {px}, 16",
            "vle8.v v16, ({px})",
            "vle8.v v17, (t0)",
            "addi t0, {pw}, 6",
            "vle8.v v5, (t0)",
            "vand.vi v6, v5, 15",
            "vsrl.vi v7, v5, 4",
            "addi t0, {pw}, 2",
            "vsetivli zero, 2, e8, m1, ta, ma",
            "vle8.v v0, (t0)",
            "vsetivli zero, 16, e8, m1, tu, mu",
            "vor.vx v6, v6, {c16}, v0.t",
            "addi t0, {pw}, 4",
            "vsetivli zero, 2, e8, m1, ta, ma",
            "vle8.v v0, (t0)",
            "vsetivli zero, 16, e8, m1, tu, mu",
            "vor.vx v7, v7, {c16}, v0.t",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vadd.vi v6, v6, -16",
            "vadd.vi v7, v7, -16",
            "vwmul.vv v18, v6, v16",
            "vwmacc.vv v18, v7, v17",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v1, v18, v1",
            "vsetivli zero, 1, e32, m1, ta, ma",
            "vmv.x.s {out}, v1",
            pw = in(reg) pw_blk,
            px = in(reg) px_qs,
            c16 = in(reg) c16,
            out = lateout(reg) out,
            out("t0") _,
            out("v0") _,
            out("v1") _,
            out("v5") _,
            out("v6") _,
            out("v7") _,
            out("v16") _,
            out("v17") _,
            out("v18") _,
            out("v19") _,
            options(nostack, readonly),
        );
    }
    out
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[inline]
#[allow(
    clippy::too_many_lines,
    reason = "4-row Q5_0 x Q8_0 RVV kernel keeps activation halves resident in v16..v17"
)]
fn dot_q5_0_q8_0_block_4x_rvv(
    pw0: *const u8,
    pw1: *const u8,
    pw2: *const u8,
    pw3: *const u8,
    px_qs: *const i8,
) -> [i32; 4] {
    let (o0, o1, o2, o3): (i32, i32, i32, i32);
    let c16 = 16_usize;
    // SAFETY: Each `pw*` holds 22 readable bytes and `px_qs` holds 32 readable `i8` bytes;
    // `v16`/`v17` activation halves stay resident across all 4 rows.
    unsafe {
        core::arch::asm!(
            "vsetivli zero, 1, e32, m1, ta, ma",
            "vmv.v.i v1, 0",
            "vmv.v.i v2, 0",
            "vmv.v.i v3, 0",
            "vmv.v.i v4, 0",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "addi t0, {px}, 16",
            "vle8.v v16, ({px})",
            "vle8.v v17, (t0)",
            // Row 0
            "addi t0, {pw0}, 6",
            "vle8.v v5, (t0)",
            "vand.vi v6, v5, 15",
            "vsrl.vi v7, v5, 4",
            "addi t0, {pw0}, 2",
            "vsetivli zero, 2, e8, m1, ta, ma",
            "vle8.v v0, (t0)",
            "vsetivli zero, 16, e8, m1, tu, mu",
            "vor.vx v6, v6, {c16}, v0.t",
            "addi t0, {pw0}, 4",
            "vsetivli zero, 2, e8, m1, ta, ma",
            "vle8.v v0, (t0)",
            "vsetivli zero, 16, e8, m1, tu, mu",
            "vor.vx v7, v7, {c16}, v0.t",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vadd.vi v6, v6, -16",
            "vadd.vi v7, v7, -16",
            "vwmul.vv v18, v6, v16",
            "vwmacc.vv v18, v7, v17",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v1, v18, v1",
            // Row 1
            "vsetivli zero, 16, e8, m1, ta, ma",
            "addi t0, {pw1}, 6",
            "vle8.v v5, (t0)",
            "vand.vi v6, v5, 15",
            "vsrl.vi v7, v5, 4",
            "addi t0, {pw1}, 2",
            "vsetivli zero, 2, e8, m1, ta, ma",
            "vle8.v v0, (t0)",
            "vsetivli zero, 16, e8, m1, tu, mu",
            "vor.vx v6, v6, {c16}, v0.t",
            "addi t0, {pw1}, 4",
            "vsetivli zero, 2, e8, m1, ta, ma",
            "vle8.v v0, (t0)",
            "vsetivli zero, 16, e8, m1, tu, mu",
            "vor.vx v7, v7, {c16}, v0.t",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vadd.vi v6, v6, -16",
            "vadd.vi v7, v7, -16",
            "vwmul.vv v18, v6, v16",
            "vwmacc.vv v18, v7, v17",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v2, v18, v2",
            // Row 2
            "vsetivli zero, 16, e8, m1, ta, ma",
            "addi t0, {pw2}, 6",
            "vle8.v v5, (t0)",
            "vand.vi v6, v5, 15",
            "vsrl.vi v7, v5, 4",
            "addi t0, {pw2}, 2",
            "vsetivli zero, 2, e8, m1, ta, ma",
            "vle8.v v0, (t0)",
            "vsetivli zero, 16, e8, m1, tu, mu",
            "vor.vx v6, v6, {c16}, v0.t",
            "addi t0, {pw2}, 4",
            "vsetivli zero, 2, e8, m1, ta, ma",
            "vle8.v v0, (t0)",
            "vsetivli zero, 16, e8, m1, tu, mu",
            "vor.vx v7, v7, {c16}, v0.t",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vadd.vi v6, v6, -16",
            "vadd.vi v7, v7, -16",
            "vwmul.vv v18, v6, v16",
            "vwmacc.vv v18, v7, v17",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v3, v18, v3",
            // Row 3
            "vsetivli zero, 16, e8, m1, ta, ma",
            "addi t0, {pw3}, 6",
            "vle8.v v5, (t0)",
            "vand.vi v6, v5, 15",
            "vsrl.vi v7, v5, 4",
            "addi t0, {pw3}, 2",
            "vsetivli zero, 2, e8, m1, ta, ma",
            "vle8.v v0, (t0)",
            "vsetivli zero, 16, e8, m1, tu, mu",
            "vor.vx v6, v6, {c16}, v0.t",
            "addi t0, {pw3}, 4",
            "vsetivli zero, 2, e8, m1, ta, ma",
            "vle8.v v0, (t0)",
            "vsetivli zero, 16, e8, m1, tu, mu",
            "vor.vx v7, v7, {c16}, v0.t",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vadd.vi v6, v6, -16",
            "vadd.vi v7, v7, -16",
            "vwmul.vv v18, v6, v16",
            "vwmacc.vv v18, v7, v17",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v4, v18, v4",
            "vsetivli zero, 1, e32, m1, ta, ma",
            "vmv.x.s {o0}, v1",
            "vmv.x.s {o1}, v2",
            "vmv.x.s {o2}, v3",
            "vmv.x.s {o3}, v4",
            pw0 = in(reg) pw0,
            pw1 = in(reg) pw1,
            pw2 = in(reg) pw2,
            pw3 = in(reg) pw3,
            px = in(reg) px_qs,
            c16 = in(reg) c16,
            o0 = lateout(reg) o0,
            o1 = lateout(reg) o1,
            o2 = lateout(reg) o2,
            o3 = lateout(reg) o3,
            out("t0") _,
            out("v0") _,
            out("v1") _,
            out("v2") _,
            out("v3") _,
            out("v4") _,
            out("v5") _,
            out("v6") _,
            out("v7") _,
            out("v16") _,
            out("v17") _,
            out("v18") _,
            out("v19") _,
            options(nostack, readonly),
        );
    }
    [o0, o1, o2, o3]
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[inline]
fn dot_q8_0_q8_0_block_i32_rvv(weight_qs: *const u8, act_qs: *const i8) -> i32 {
    let out: i32;
    // SAFETY: `weight_qs` and `act_qs` hold 32 readable `i8` bytes; each 16-element half reduces
    // immediately via `vwredsum.vs` to `i32` so `128*128 + 128*128 = 32768` never overflows `i16`.
    unsafe {
        core::arch::asm!(
            "vsetivli zero, 1, e32, m1, ta, ma",
            "vmv.v.i v1, 0",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "addi t0, {px}, 16",
            "vle8.v v16, ({px})",
            "vle8.v v17, (t0)",
            "vle8.v v5, ({pw})",
            "addi t0, {pw}, 16",
            "vle8.v v6, (t0)",
            "vwmul.vv v18, v5, v16",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v1, v18, v1",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vwmul.vv v18, v6, v17",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v1, v18, v1",
            "vsetivli zero, 1, e32, m1, ta, ma",
            "vmv.x.s {out}, v1",
            pw = in(reg) weight_qs,
            px = in(reg) act_qs,
            out = out(reg) out,
            out("t0") _,
            out("v1") _,
            out("v5") _,
            out("v6") _,
            out("v16") _,
            out("v17") _,
            out("v18") _,
            out("v19") _,
            options(nostack, readonly),
        );
    }
    out
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[inline]
fn dot_q8_0_q8_0_block_4x_rvv(
    pw0: *const u8,
    pw1: *const u8,
    pw2: *const u8,
    pw3: *const u8,
    px_qs: *const i8,
) -> [i32; 4] {
    let (o0, o1, o2, o3): (i32, i32, i32, i32);
    // SAFETY: Each `pw*` and `px_qs` hold 32 readable `i8` bytes; each 16-element half reduces
    // immediately to `i32` (`vwredsum.vs`) preventing `i16` overflow across all 4 weight rows.
    unsafe {
        core::arch::asm!(
            "vsetivli zero, 1, e32, m1, ta, ma",
            "vmv.v.i v1, 0",
            "vmv.v.i v2, 0",
            "vmv.v.i v3, 0",
            "vmv.v.i v4, 0",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "addi t0, {px}, 16",
            "vle8.v v16, ({px})",
            "vle8.v v17, (t0)",
            // Row 0
            "vle8.v v5, ({pw0})",
            "addi t0, {pw0}, 16",
            "vle8.v v6, (t0)",
            "vwmul.vv v18, v5, v16",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v1, v18, v1",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vwmul.vv v18, v6, v17",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v1, v18, v1",
            // Row 1
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vle8.v v5, ({pw1})",
            "addi t0, {pw1}, 16",
            "vle8.v v6, (t0)",
            "vwmul.vv v18, v5, v16",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v2, v18, v2",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vwmul.vv v18, v6, v17",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v2, v18, v2",
            // Row 2
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vle8.v v5, ({pw2})",
            "addi t0, {pw2}, 16",
            "vle8.v v6, (t0)",
            "vwmul.vv v18, v5, v16",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v3, v18, v3",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vwmul.vv v18, v6, v17",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v3, v18, v3",
            // Row 3
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vle8.v v5, ({pw3})",
            "addi t0, {pw3}, 16",
            "vle8.v v6, (t0)",
            "vwmul.vv v18, v5, v16",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v4, v18, v4",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vwmul.vv v18, v6, v17",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v4, v18, v4",
            "vsetivli zero, 1, e32, m1, ta, ma",
            "vmv.x.s {o0}, v1",
            "vmv.x.s {o1}, v2",
            "vmv.x.s {o2}, v3",
            "vmv.x.s {o3}, v4",
            pw0 = in(reg) pw0,
            pw1 = in(reg) pw1,
            pw2 = in(reg) pw2,
            pw3 = in(reg) pw3,
            px = in(reg) px_qs,
            o0 = lateout(reg) o0,
            o1 = lateout(reg) o1,
            o2 = lateout(reg) o2,
            o3 = lateout(reg) o3,
            out("t0") _,
            out("v1") _,
            out("v2") _,
            out("v3") _,
            out("v4") _,
            out("v5") _,
            out("v6") _,
            out("v16") _,
            out("v17") _,
            out("v18") _,
            out("v19") _,
            options(nostack, readonly),
        );
    }
    [o0, o1, o2, o3]
}

#[inline]
fn dot_q4_0_q8_0_row(w_blocks: &[u8], x_blocks: &[BlockQ8_0]) -> f32 {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        let mut acc = 0.0_f32;
        for (wb, xb) in w_blocks.chunks_exact(Q4_0_BLOCK_BYTES).zip(x_blocks.iter()) {
            let dw = f16_to_f32_finite(u16::from_le_bytes([wb[0], wb[1]]));
            if dw != 0.0 && xb.d != 0.0 {
                let d_i32 = dot_q4_0_q8_0_block_i32_rvv(wb[2..18].as_ptr(), xb.qs.as_ptr());
                acc += i32_to_f32(d_i32) * (dw * xb.d);
            }
        }
        acc
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        w_blocks
            .chunks_exact(Q4_0_BLOCK_BYTES)
            .zip(x_blocks.iter())
            .map(|(wb, xb)| dot_q4_0_q8_0_block_scalar(wb, xb))
            .sum()
    }
}

#[inline]
fn dot_q4_0_q8_0_4row(
    r0: &[u8],
    r1: &[u8],
    r2: &[u8],
    r3: &[u8],
    x_blocks: &[BlockQ8_0],
) -> [f32; 4] {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        let mut acc = [0.0_f32; 4];
        for (idx, xb) in x_blocks.iter().enumerate() {
            if xb.d == 0.0 {
                continue;
            }
            let off = idx * Q4_0_BLOCK_BYTES;
            let b0 = &r0[off..off + Q4_0_BLOCK_BYTES];
            let b1 = &r1[off..off + Q4_0_BLOCK_BYTES];
            let b2 = &r2[off..off + Q4_0_BLOCK_BYTES];
            let b3 = &r3[off..off + Q4_0_BLOCK_BYTES];
            let dots = dot_q4_0_q8_0_block_4x_rvv(
                b0[2..].as_ptr(),
                b1[2..].as_ptr(),
                b2[2..].as_ptr(),
                b3[2..].as_ptr(),
                xb.qs.as_ptr(),
            );
            let ds = [b0, b1, b2, b3];
            for k in 0..4 {
                let dw = f16_to_f32_finite(u16::from_le_bytes([ds[k][0], ds[k][1]]));
                acc[k] += i32_to_f32(dots[k]) * (dw * xb.d);
            }
        }
        acc
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        let mut acc = [0.0_f32; 4];
        for (idx, xb) in x_blocks.iter().enumerate() {
            let off = idx * Q4_0_BLOCK_BYTES;
            acc[0] += dot_q4_0_q8_0_block_scalar(&r0[off..off + Q4_0_BLOCK_BYTES], xb);
            acc[1] += dot_q4_0_q8_0_block_scalar(&r1[off..off + Q4_0_BLOCK_BYTES], xb);
            acc[2] += dot_q4_0_q8_0_block_scalar(&r2[off..off + Q4_0_BLOCK_BYTES], xb);
            acc[3] += dot_q4_0_q8_0_block_scalar(&r3[off..off + Q4_0_BLOCK_BYTES], xb);
        }
        acc
    }
}

#[inline]
fn dot_q5_0_q8_0_row(w_blocks: &[u8], x_blocks: &[BlockQ8_0]) -> f32 {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        let mut acc = 0.0_f32;
        for (wb, xb) in w_blocks.chunks_exact(Q5_0_BLOCK_BYTES).zip(x_blocks.iter()) {
            let dw = f16_to_f32_finite(u16::from_le_bytes([wb[0], wb[1]]));
            if dw != 0.0 && xb.d != 0.0 {
                let d_i32 = dot_q5_0_q8_0_block_i32_rvv(wb.as_ptr(), xb.qs.as_ptr());
                acc += i32_to_f32(d_i32) * (dw * xb.d);
            }
        }
        acc
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        w_blocks
            .chunks_exact(Q5_0_BLOCK_BYTES)
            .zip(x_blocks.iter())
            .map(|(wb, xb)| dot_q5_0_q8_0_block_scalar(wb, xb))
            .sum()
    }
}

#[inline]
fn dot_q5_0_q8_0_4row(
    r0: &[u8],
    r1: &[u8],
    r2: &[u8],
    r3: &[u8],
    x_blocks: &[BlockQ8_0],
) -> [f32; 4] {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        let mut acc = [0.0_f32; 4];
        for (idx, xb) in x_blocks.iter().enumerate() {
            if xb.d == 0.0 {
                continue;
            }
            let off = idx * Q5_0_BLOCK_BYTES;
            let b0 = &r0[off..off + Q5_0_BLOCK_BYTES];
            let b1 = &r1[off..off + Q5_0_BLOCK_BYTES];
            let b2 = &r2[off..off + Q5_0_BLOCK_BYTES];
            let b3 = &r3[off..off + Q5_0_BLOCK_BYTES];
            let dots = dot_q5_0_q8_0_block_4x_rvv(
                b0.as_ptr(),
                b1.as_ptr(),
                b2.as_ptr(),
                b3.as_ptr(),
                xb.qs.as_ptr(),
            );
            let ds = [b0, b1, b2, b3];
            for k in 0..4 {
                let dw = f16_to_f32_finite(u16::from_le_bytes([ds[k][0], ds[k][1]]));
                acc[k] += i32_to_f32(dots[k]) * (dw * xb.d);
            }
        }
        acc
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        let mut acc = [0.0_f32; 4];
        for (idx, xb) in x_blocks.iter().enumerate() {
            let off = idx * Q5_0_BLOCK_BYTES;
            acc[0] += dot_q5_0_q8_0_block_scalar(&r0[off..off + Q5_0_BLOCK_BYTES], xb);
            acc[1] += dot_q5_0_q8_0_block_scalar(&r1[off..off + Q5_0_BLOCK_BYTES], xb);
            acc[2] += dot_q5_0_q8_0_block_scalar(&r2[off..off + Q5_0_BLOCK_BYTES], xb);
            acc[3] += dot_q5_0_q8_0_block_scalar(&r3[off..off + Q5_0_BLOCK_BYTES], xb);
        }
        acc
    }
}

#[inline]
fn dot_q8_0_q8_0_row(w_blocks: &[u8], x_blocks: &[BlockQ8_0]) -> f32 {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        let mut acc = 0.0_f32;
        for (wb, xb) in w_blocks.chunks_exact(Q8_0_BLOCK_BYTES).zip(x_blocks.iter()) {
            let dw = f16_to_f32_finite(u16::from_le_bytes([wb[0], wb[1]]));
            if dw != 0.0 && xb.d != 0.0 {
                let d_i32 = dot_q8_0_q8_0_block_i32_rvv(wb[2..].as_ptr(), xb.qs.as_ptr());
                acc += i32_to_f32(d_i32) * (dw * xb.d);
            }
        }
        acc
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        w_blocks
            .chunks_exact(Q8_0_BLOCK_BYTES)
            .zip(x_blocks.iter())
            .map(|(wb, xb)| dot_q8_0_q8_0_block_scalar(wb, xb))
            .sum()
    }
}

#[inline]
fn dot_q8_0_q8_0_4row(
    r0: &[u8],
    r1: &[u8],
    r2: &[u8],
    r3: &[u8],
    x_blocks: &[BlockQ8_0],
) -> [f32; 4] {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        let mut acc = [0.0_f32; 4];
        for (idx, xb) in x_blocks.iter().enumerate() {
            if xb.d == 0.0 {
                continue;
            }
            let off = idx * Q8_0_BLOCK_BYTES;
            let b0 = &r0[off..off + Q8_0_BLOCK_BYTES];
            let b1 = &r1[off..off + Q8_0_BLOCK_BYTES];
            let b2 = &r2[off..off + Q8_0_BLOCK_BYTES];
            let b3 = &r3[off..off + Q8_0_BLOCK_BYTES];
            let dots = dot_q8_0_q8_0_block_4x_rvv(
                b0[2..].as_ptr(),
                b1[2..].as_ptr(),
                b2[2..].as_ptr(),
                b3[2..].as_ptr(),
                xb.qs.as_ptr(),
            );
            let ds = [b0, b1, b2, b3];
            for k in 0..4 {
                let dw = f16_to_f32_finite(u16::from_le_bytes([ds[k][0], ds[k][1]]));
                acc[k] += i32_to_f32(dots[k]) * (dw * xb.d);
            }
        }
        acc
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        let mut acc = [0.0_f32; 4];
        for (idx, xb) in x_blocks.iter().enumerate() {
            let off = idx * Q8_0_BLOCK_BYTES;
            acc[0] += dot_q8_0_q8_0_block_scalar(&r0[off..off + Q8_0_BLOCK_BYTES], xb);
            acc[1] += dot_q8_0_q8_0_block_scalar(&r1[off..off + Q8_0_BLOCK_BYTES], xb);
            acc[2] += dot_q8_0_q8_0_block_scalar(&r2[off..off + Q8_0_BLOCK_BYTES], xb);
            acc[3] += dot_q8_0_q8_0_block_scalar(&r3[off..off + Q8_0_BLOCK_BYTES], xb);
        }
        acc
    }
}

/// Fast RVV/scalar single-row `Q5_0 x Q8_0` (raw GGUF byte blocks) dot product helper.
#[cfg(all(target_arch = "riscv64", not(miri)))]
pub(crate) fn vec_dot_q5_0_q8_0_raw_rvv(w_q5_0: &[u8], x_q8_0: &[u8]) -> f32 {
    let mut acc = 0.0_f32;
    let mut x_qs = [0_i8; Q8_0_BLOCK_ELEMENTS];
    for (w_blk, x_blk) in w_q5_0
        .chunks_exact(Q5_0_BLOCK_BYTES)
        .zip(x_q8_0.chunks_exact(Q8_0_BLOCK_BYTES))
    {
        let dw = f16_to_f32_finite(u16::from_le_bytes([w_blk[0], w_blk[1]]));
        let dx = f16_to_f32_finite(u16::from_le_bytes([x_blk[0], x_blk[1]]));
        if dw == 0.0 || dx == 0.0 {
            continue;
        }
        for (dst, &byte) in x_qs.iter_mut().zip(&x_blk[2..34]) {
            *dst = i8::from_ne_bytes([byte]);
        }
        let d_i32 = dot_q5_0_q8_0_block_i32_rvv(w_blk.as_ptr(), x_qs.as_ptr());
        acc += i32_to_f32(d_i32) * (dw * dx);
    }
    acc
}

/// Computes single-row `Q4_0 x BlockQ8_0` dot product.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `x_blocks` is empty or `w_blocks.len() != x_blocks.len() * 18`.
pub fn vec_dot_q4_0_q8_0(w_blocks: &[u8], x_blocks: &[BlockQ8_0]) -> Result<f32, K3Error> {
    validate_q8_block_row_bytes(
        "vec_dot_q4_0_q8_0",
        w_blocks,
        x_blocks.len(),
        Q4_0_BLOCK_BYTES,
    )?;
    Ok(dot_q4_0_q8_0_row(w_blocks, x_blocks))
}

/// Computes single-row `Q5_0 x BlockQ8_0` dot product.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `x_blocks` is empty or `w_blocks.len() != x_blocks.len() * 22`.
pub fn vec_dot_q5_0_q8_0_blocks(w_blocks: &[u8], x_blocks: &[BlockQ8_0]) -> Result<f32, K3Error> {
    validate_q8_block_row_bytes(
        "vec_dot_q5_0_q8_0_blocks",
        w_blocks,
        x_blocks.len(),
        Q5_0_BLOCK_BYTES,
    )?;
    Ok(dot_q5_0_q8_0_row(w_blocks, x_blocks))
}

/// Computes single-row `Q8_0 x BlockQ8_0` dot product.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `x_blocks` is empty or `w_blocks.len() != x_blocks.len() * 34`.
pub fn vec_dot_q8_0_q8_0(w_blocks: &[u8], x_blocks: &[BlockQ8_0]) -> Result<f32, K3Error> {
    validate_q8_block_row_bytes(
        "vec_dot_q8_0_q8_0",
        w_blocks,
        x_blocks.len(),
        Q8_0_BLOCK_BYTES,
    )?;
    Ok(dot_q8_0_q8_0_row(w_blocks, x_blocks))
}

/// Shared 4-row unrolled GEMV row-group driver supporting arbitrary chunked activation slices
/// without heap allocation.
#[inline]
#[expect(
    clippy::too_many_arguments,
    reason = "generic row driver parameterizes 4-row and 1-row kernels over chunk offsets"
)]
fn run_4row_chunked_gemv<X, F4, F1>(
    output: &mut [f32],
    raw_bytes: &[u8],
    rows: usize,
    row_stride_bytes: usize,
    blk_offset: usize,
    blk_bytes: usize,
    x_chunk: &[X],
    accumulate: bool,
    mut kernel_4x: F4,
    mut kernel_1x: F1,
) where
    F4: FnMut(&[u8], &[u8], &[u8], &[u8], &[X]) -> [f32; 4],
    F1: FnMut(&[u8], &[X]) -> f32,
{
    let byte_off = blk_offset * blk_bytes;
    let chunk_bytes = x_chunk.len() * blk_bytes;
    let rows4 = (rows / 4) * 4;
    let mut row = 0_usize;
    while row < rows4 {
        let b0 = row * row_stride_bytes + byte_off;
        let b1 = (row + 1) * row_stride_bytes + byte_off;
        let b2 = (row + 2) * row_stride_bytes + byte_off;
        let b3 = (row + 3) * row_stride_bytes + byte_off;
        let vals = kernel_4x(
            &raw_bytes[b0..b0 + chunk_bytes],
            &raw_bytes[b1..b1 + chunk_bytes],
            &raw_bytes[b2..b2 + chunk_bytes],
            &raw_bytes[b3..b3 + chunk_bytes],
            x_chunk,
        );
        for (dst, &val) in output[row..row + 4].iter_mut().zip(&vals) {
            if accumulate {
                *dst += val;
            } else {
                *dst = val;
            }
        }
        row += 4;
    }
    while row < rows {
        let b = row * row_stride_bytes + byte_off;
        let val = kernel_1x(&raw_bytes[b..b + chunk_bytes], x_chunk);
        if accumulate {
            output[row] += val;
        } else {
            output[row] = val;
        }
        row += 1;
    }
}

fn check_q8_block_len(
    name: &str,
    weights: &TensorView<'_>,
    blocks_per_row: usize,
    x_len: usize,
) -> Result<(), K3Error> {
    if x_len != blocks_per_row {
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: x_len,
            buffer_len: blocks_per_row,
            reason: format!("{name}: x_blocks len {x_len} != expected {blocks_per_row}"),
        });
    }
    Ok(())
}

/// Computes 4-row `M = 1` GEMV over `Q4_0` weights and pre-quantized [`BlockQ8_0`] activations
/// (`N % 4` tail handled cleanly, zero heap allocations).
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] on quantization type, dimension, or block count mismatch.
pub fn gemv_q4_0_q8_0(
    output: &mut [f32],
    weights: &TensorView<'_>,
    x_blocks: &[BlockQ8_0],
) -> Result<(), K3Error> {
    let cols = x_blocks.len() * Q4_0_BLOCK_ELEMENTS;
    let (rows, _) = validate_quant_matvec_dimensions(QuantType::Q4_0, output.len(), weights, cols)?;
    let blocks_per_row = cols / Q4_0_BLOCK_ELEMENTS;
    check_q8_block_len("gemv_q4_0_q8_0", weights, blocks_per_row, x_blocks.len())?;
    run_4row_chunked_gemv(
        output,
        weights.as_bytes(),
        rows,
        blocks_per_row * Q4_0_BLOCK_BYTES,
        0,
        Q4_0_BLOCK_BYTES,
        x_blocks,
        false,
        dot_q4_0_q8_0_4row,
        dot_q4_0_q8_0_row,
    );
    Ok(())
}

/// Alias for [`gemv_q4_0_q8_0`].
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] on invalid weights or activation block lengths.
pub fn gemv_q4_0_q8_0_4x(
    output: &mut [f32],
    weights: &TensorView<'_>,
    x_blocks: &[BlockQ8_0],
) -> Result<(), K3Error> {
    gemv_q4_0_q8_0(output, weights, x_blocks)
}

/// Computes 4-row `M = 1` GEMV over `Q5_0` weights and pre-quantized [`BlockQ8_0`] activations.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] on quantization type, dimension, or block count mismatch.
pub fn gemv_q5_0_q8_0(
    output: &mut [f32],
    weights: &TensorView<'_>,
    x_blocks: &[BlockQ8_0],
) -> Result<(), K3Error> {
    let cols = x_blocks.len() * Q5_0_BLOCK_ELEMENTS;
    let (rows, _) = validate_quant_matvec_dimensions(QuantType::Q5_0, output.len(), weights, cols)?;
    let blocks_per_row = cols / Q5_0_BLOCK_ELEMENTS;
    check_q8_block_len("gemv_q5_0_q8_0", weights, blocks_per_row, x_blocks.len())?;
    run_4row_chunked_gemv(
        output,
        weights.as_bytes(),
        rows,
        blocks_per_row * Q5_0_BLOCK_BYTES,
        0,
        Q5_0_BLOCK_BYTES,
        x_blocks,
        false,
        dot_q5_0_q8_0_4row,
        dot_q5_0_q8_0_row,
    );
    Ok(())
}

/// Alias for [`gemv_q5_0_q8_0`].
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] on invalid weights or activation block lengths.
pub fn gemv_q5_0_q8_0_4x(
    output: &mut [f32],
    weights: &TensorView<'_>,
    x_blocks: &[BlockQ8_0],
) -> Result<(), K3Error> {
    gemv_q5_0_q8_0(output, weights, x_blocks)
}

/// Computes 4-row `M = 1` GEMV over `Q8_0` weights and pre-quantized [`BlockQ8_0`] activations.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] on quantization type, dimension, or block count mismatch.
pub fn gemv_q8_0_q8_0(
    output: &mut [f32],
    weights: &TensorView<'_>,
    x_blocks: &[BlockQ8_0],
) -> Result<(), K3Error> {
    let cols = x_blocks.len() * Q8_0_BLOCK_ELEMENTS;
    let (rows, _) = validate_q8_0_dimensions(output.len(), weights, cols)?;
    let blocks_per_row = cols / Q8_0_BLOCK_ELEMENTS;
    check_q8_block_len("gemv_q8_0_q8_0", weights, blocks_per_row, x_blocks.len())?;
    run_4row_chunked_gemv(
        output,
        weights.as_bytes(),
        rows,
        blocks_per_row * Q8_0_BLOCK_BYTES,
        0,
        Q8_0_BLOCK_BYTES,
        x_blocks,
        false,
        dot_q8_0_q8_0_4row,
        dot_q8_0_q8_0_row,
    );
    Ok(())
}

/// Alias for [`gemv_q8_0_q8_0`].
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] on invalid weights or activation block lengths.
pub fn gemv_q8_0_q8_0_4x(
    output: &mut [f32],
    weights: &TensorView<'_>,
    x_blocks: &[BlockQ8_0],
) -> Result<(), K3Error> {
    gemv_q8_0_q8_0(output, weights, x_blocks)
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[allow(
    clippy::too_many_arguments,
    reason = "internal stack-chunked RVV dispatch helper parameterized over 4x and 1x block kernels"
)]
pub(crate) fn vec_dot_q8_0_chunked_rvv<F4, F1>(
    output: &mut [f32],
    raw_bytes: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
    blk_bytes: usize,
    mut kernel_4x: F4,
    mut kernel_1x: F1,
) where
    F4: FnMut(&[u8], &[u8], &[u8], &[u8], &[BlockQ8_0]) -> [f32; 4],
    F1: FnMut(&[u8], &[BlockQ8_0]) -> f32,
{
    let blocks_per_row = cols / Q8_0_BLOCK_ELEMENTS;
    let row_stride_bytes = blocks_per_row * blk_bytes;
    let mut x_scratch = [BlockQ8_0::ZERO; Q8_0_STACK_BLOCKS];
    let mut blk_start = 0_usize;
    while blk_start < blocks_per_row {
        let count = (blocks_per_row - blk_start).min(Q8_0_STACK_BLOCKS);
        let elem_start = blk_start * Q8_0_BLOCK_ELEMENTS;
        let elem_end = elem_start + count * Q8_0_BLOCK_ELEMENTS;
        let _ = quantize_row_q8_0_f32(&input[elem_start..elem_end], &mut x_scratch[..count]);
        let accumulate = blk_start > 0;
        run_4row_chunked_gemv(
            output,
            raw_bytes,
            rows,
            row_stride_bytes,
            blk_start,
            blk_bytes,
            &x_scratch[..count],
            accumulate,
            &mut kernel_4x,
            &mut kernel_1x,
        );
        blk_start += count;
    }
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
pub(crate) fn vec_dot_q4_0_4row_rvv(
    output: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
) -> Result<(), K3Error> {
    let (rows, cols) =
        validate_quant_matvec_dimensions(QuantType::Q4_0, output.len(), weights, input.len())?;
    vec_dot_q8_0_chunked_rvv(
        output,
        weights.as_bytes(),
        rows,
        cols,
        input,
        Q4_0_BLOCK_BYTES,
        dot_q4_0_q8_0_4row,
        dot_q4_0_q8_0_row,
    );
    Ok(())
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
pub(crate) fn vec_dot_q5_0_4row_rvv(
    output: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
) -> Result<(), K3Error> {
    let (rows, cols) =
        validate_quant_matvec_dimensions(QuantType::Q5_0, output.len(), weights, input.len())?;
    vec_dot_q8_0_chunked_rvv(
        output,
        weights.as_bytes(),
        rows,
        cols,
        input,
        Q5_0_BLOCK_BYTES,
        dot_q5_0_q8_0_4row,
        dot_q5_0_q8_0_row,
    );
    Ok(())
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
pub(crate) fn vec_dot_q8_0_4row_rvv(
    output: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
) -> Result<(), K3Error> {
    let (rows, cols) = validate_q8_0_dimensions(output.len(), weights, input.len())?;
    vec_dot_q8_0_chunked_rvv(
        output,
        weights.as_bytes(),
        rows,
        cols,
        input,
        Q8_0_BLOCK_BYTES,
        dot_q8_0_q8_0_4row,
        dot_q8_0_q8_0_row,
    );
    Ok(())
}

// ================================================================================================
// Direct Q4_K / Q5_K / Q6_K / TQ1_0 / TQ2_0 x BlockQ8K block & 4-row M=1 GEMV kernels
// ================================================================================================

#[inline]
pub(crate) fn unpack_q4_q5_k_scales_mins(scales: &[u8]) -> ([u8; 8], [u8; 8]) {
    let mut sc = [0_u8; 8];
    let mut m = [0_u8; 8];
    for j in 0..4 {
        sc[j] = scales[j] & 0x3F;
        m[j] = scales[j + 4] & 0x3F;
    }
    for j in 4..8 {
        sc[j] = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        m[j] = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
    }
    (sc, m)
}

#[cfg(any(not(target_arch = "riscv64"), miri))]
#[inline]
fn dot_q4_k_q8_k_block_scalar(w_blk: &[u8], x_blk: &BlockQ8K) -> f32 {
    let d = f16_to_f32_finite(u16::from_le_bytes([w_blk[0], w_blk[1]]));
    let dmin = f16_to_f32_finite(u16::from_le_bytes([w_blk[2], w_blk[3]]));
    if (d == 0.0 && dmin == 0.0) || x_blk.d == 0.0 {
        return 0.0;
    }
    let (sc, mins) = unpack_q4_q5_k_scales_mins(&w_blk[4..16]);
    let qs = &w_blk[16..144];
    let mut sumi = 0_i32;
    let mut sum_mins = 0_i32;
    for chunk in 0..4 {
        let is = chunk * 2;
        let q_sub = &qs[chunk * 32..(chunk + 1) * 32];
        let x_lo = &x_blk.qs[chunk * 64..chunk * 64 + 32];
        let x_hi = &x_blk.qs[chunk * 64 + 32..chunk * 64 + 64];
        let mut s_lo = 0_i32;
        let mut s_hi = 0_i32;
        for l in 0..32 {
            let b = q_sub[l];
            s_lo += i32::from(b & 0x0F) * i32::from(x_lo[l]);
            s_hi += i32::from(b >> 4) * i32::from(x_hi[l]);
        }
        sumi += i32::from(sc[is]) * s_lo + i32::from(sc[is + 1]) * s_hi;
        let bsum_lo = i32::from(x_blk.bsums[chunk * 4]) + i32::from(x_blk.bsums[chunk * 4 + 1]);
        let bsum_hi = i32::from(x_blk.bsums[chunk * 4 + 2]) + i32::from(x_blk.bsums[chunk * 4 + 3]);
        sum_mins += i32::from(mins[is]) * bsum_lo + i32::from(mins[is + 1]) * bsum_hi;
    }
    (d * x_blk.d) * i32_to_f32(sumi) - (dmin * x_blk.d) * i32_to_f32(sum_mins)
}

#[cfg(any(not(target_arch = "riscv64"), miri))]
#[inline]
fn dot_q5_k_q8_k_block_scalar(w_blk: &[u8], x_blk: &BlockQ8K) -> f32 {
    let d = f16_to_f32_finite(u16::from_le_bytes([w_blk[0], w_blk[1]]));
    let dmin = f16_to_f32_finite(u16::from_le_bytes([w_blk[2], w_blk[3]]));
    if (d == 0.0 && dmin == 0.0) || x_blk.d == 0.0 {
        return 0.0;
    }
    let (sc, mins) = unpack_q4_q5_k_scales_mins(&w_blk[4..16]);
    let qh = &w_blk[16..48];
    let qs = &w_blk[48..176];
    let mut sumi = 0_i32;
    let mut sum_mins = 0_i32;
    for chunk in 0..4 {
        let is = chunk * 2;
        let u1 = 1_u8 << (chunk * 2);
        let u2 = 2_u8 << (chunk * 2);
        let q_sub = &qs[chunk * 32..(chunk + 1) * 32];
        let x_lo = &x_blk.qs[chunk * 64..chunk * 64 + 32];
        let x_hi = &x_blk.qs[chunk * 64 + 32..chunk * 64 + 64];
        let mut s_lo = 0_i32;
        let mut s_hi = 0_i32;
        for l in 0..32 {
            let b = q_sub[l];
            let hb1 = if (qh[l] & u1) != 0 { 16_i32 } else { 0_i32 };
            let hb2 = if (qh[l] & u2) != 0 { 16_i32 } else { 0_i32 };
            s_lo += (i32::from(b & 0x0F) | hb1) * i32::from(x_lo[l]);
            s_hi += (i32::from(b >> 4) | hb2) * i32::from(x_hi[l]);
        }
        sumi += i32::from(sc[is]) * s_lo + i32::from(sc[is + 1]) * s_hi;
        let bsum_lo = i32::from(x_blk.bsums[chunk * 4]) + i32::from(x_blk.bsums[chunk * 4 + 1]);
        let bsum_hi = i32::from(x_blk.bsums[chunk * 4 + 2]) + i32::from(x_blk.bsums[chunk * 4 + 3]);
        sum_mins += i32::from(mins[is]) * bsum_lo + i32::from(mins[is + 1]) * bsum_hi;
    }
    (d * x_blk.d) * i32_to_f32(sumi) - (dmin * x_blk.d) * i32_to_f32(sum_mins)
}

#[cfg(any(not(target_arch = "riscv64"), miri))]
#[inline]
fn dot_q6_k_q8_k_block_scalar(w_blk: &[u8], x_blk: &BlockQ8K) -> f32 {
    let d = f16_to_f32_finite(u16::from_le_bytes([w_blk[208], w_blk[209]]));
    if d == 0.0 || x_blk.d == 0.0 {
        return 0.0;
    }
    let ql = &w_blk[0..128];
    let qh = &w_blk[128..192];
    let scales = &w_blk[192..208];
    let mut sumi = 0_i32;

    for half_idx in 0..2 {
        let n = half_idx * 128;
        let q_lo_off = half_idx * 64;
        let q_hi_off = half_idx * 32;
        let sc_base = half_idx * 8;
        for is in 0..2 {
            let l_base = is * 16;
            let sc1 = i32::from(i8::from_ne_bytes([scales[sc_base + is]]));
            let sc2 = i32::from(i8::from_ne_bytes([scales[sc_base + is + 2]]));
            let sc3 = i32::from(i8::from_ne_bytes([scales[sc_base + is + 4]]));
            let sc4 = i32::from(i8::from_ne_bytes([scales[sc_base + is + 6]]));
            let (mut s1, mut s2, mut s3, mut s4) = (0_i32, 0_i32, 0_i32, 0_i32);
            for l in l_base..l_base + 16 {
                let b_lo = ql[q_lo_off + l];
                let b_hi = ql[q_lo_off + l + 32];
                let hb = qh[q_hi_off + l];
                let q1 = (i32::from(b_lo & 0x0F) | (i32::from(hb & 0x03) << 4)) - 32;
                let q2 = (i32::from(b_hi & 0x0F) | (i32::from((hb >> 2) & 0x03) << 4)) - 32;
                let q3 = (i32::from(b_lo >> 4) | (i32::from((hb >> 4) & 0x03) << 4)) - 32;
                let q4 = (i32::from(b_hi >> 4) | (i32::from((hb >> 6) & 0x03) << 4)) - 32;
                s1 += q1 * i32::from(x_blk.qs[n + l]);
                s2 += q2 * i32::from(x_blk.qs[n + l + 32]);
                s3 += q3 * i32::from(x_blk.qs[n + l + 64]);
                s4 += q4 * i32::from(x_blk.qs[n + l + 96]);
            }
            sumi += sc1 * s1 + sc2 * s2 + sc3 * s3 + sc4 * s4;
        }
    }
    (d * x_blk.d) * i32_to_f32(sumi)
}

const TQ1_LUT: [[i8; 5]; 256] = {
    let mut table = [[0_i8; 5]; 256];
    let mut b = 0_usize;
    while b < 256 {
        let mut cur = b;
        let mut j = 0_usize;
        while j < 5 {
            let rem = match cur % 3 {
                0 => 0_i8,
                1 => 1_i8,
                _ => 2_i8,
            };
            table[b][j] = rem - 1;
            cur /= 3;
            j += 1;
        }
        b += 1;
    }
    table
};

#[inline]
fn unpack_tq1_0_lut(blk: &[u8], out: &mut [i8; TQ1_0_BLOCK_ELEMENTS]) -> f32 {
    let d = f16_to_f32_finite(u16::from_le_bytes([blk[0], blk[1]]));
    for (i, &byte) in blk[2..53].iter().enumerate() {
        let base = i * 5;
        out[base..base + 5].copy_from_slice(&TQ1_LUT[usize::from(byte)]);
    }
    out[255] = TQ1_LUT[usize::from(blk[53])][0];
    d
}

#[inline]
fn dot_tq1_0_q8_k_block(w_blk: &[u8], x_blk: &BlockQ8K) -> f32 {
    if x_blk.d == 0.0 {
        return 0.0;
    }
    let mut w_i8 = [0_i8; TQ1_0_BLOCK_ELEMENTS];
    let dw = unpack_tq1_0_lut(w_blk, &mut w_i8);
    if dw == 0.0 {
        return 0.0;
    }
    let d_i32 = dot_i8_vector(&w_i8, &x_blk.qs);
    i32_to_f32(d_i32) * (dw * x_blk.d)
}

#[cfg(any(not(target_arch = "riscv64"), miri))]
#[inline]
fn dot_tq2_0_q8_k_block_scalar(w_blk: &[u8; TQ2_0_BLOCK_BYTES], x_blk: &BlockQ8K) -> f32 {
    if x_blk.d == 0.0 {
        return 0.0;
    }
    let mut w_i8 = [0_i8; TQ2_0_BLOCK_ELEMENTS];
    let dw = unpack_tq2_0_block_i8(w_blk, &mut w_i8);
    if dw == 0.0 {
        return 0.0;
    }
    let d_i32 = dot_i8_vector(&w_i8, &x_blk.qs);
    i32_to_f32(d_i32) * (dw * x_blk.d)
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[inline]
fn dot_q4_k_chunk64_1x_rvv(pqs: *const u8, px: *const i8) -> (i32, i32) {
    let (s_lo, s_hi): (i32, i32);
    // SAFETY: `pqs` holds 32 packed nibble bytes and `px` holds 64 `i8` bytes; `vl = 16` at
    // `e8,m1` -> `e16,m2` -> `e32,m1`, `max sum in i16 <= 3810 << 32767`.
    unsafe {
        core::arch::asm!(
            "vsetivli zero, 1, e32, m1, ta, ma",
            "vmv.v.i v8, 0",
            "vmv.v.i v9, 0",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "addi t0, {px}, 16",
            "addi t1, {px}, 32",
            "addi t2, {px}, 48",
            "vle8.v v16, ({px})",
            "vle8.v v17, (t0)",
            "vle8.v v18, (t1)",
            "vle8.v v19, (t2)",
            "vle8.v v1, ({pqs})",
            "addi t0, {pqs}, 16",
            "vle8.v v2, (t0)",
            "vand.vi v3, v1, 15",
            "vand.vi v4, v2, 15",
            "vwmul.vv v20, v3, v16",
            "vwmacc.vv v20, v4, v17",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v8, v20, v8",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vsrl.vi v3, v1, 4",
            "vsrl.vi v4, v2, 4",
            "vwmul.vv v20, v3, v18",
            "vwmacc.vv v20, v4, v19",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v9, v20, v9",
            "vsetivli zero, 1, e32, m1, ta, ma",
            "vmv.x.s {slo}, v8",
            "vmv.x.s {shi}, v9",
            pqs = in(reg) pqs,
            px = in(reg) px,
            slo = lateout(reg) s_lo,
            shi = lateout(reg) s_hi,
            out("t0") _,
            out("t1") _,
            out("t2") _,
            out("v1") _,
            out("v2") _,
            out("v3") _,
            out("v4") _,
            out("v8") _,
            out("v9") _,
            out("v16") _,
            out("v17") _,
            out("v18") _,
            out("v19") _,
            out("v20") _,
            out("v21") _,
            options(nostack, readonly),
        );
    }
    (s_lo, s_hi)
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[inline]
#[allow(
    clippy::too_many_lines,
    reason = "4-row Q4_K RVV assembly keeps 64B activations pinned in v16..v19"
)]
fn dot_q4_k_chunk64_4x_rvv(
    pqs0: *const u8,
    pqs1: *const u8,
    pqs2: *const u8,
    pqs3: *const u8,
    px: *const i8,
    out_sums: &mut [i32; 8],
) {
    // SAFETY: `pqs0..3` hold 32 bytes each, `px` holds 64 `i8` bytes, and `out_sums` holds
    // 8 writable `i32` slots (`lo0..3`, `hi0..3`). `v16..v19` remain resident across all 4 rows.
    unsafe {
        core::arch::asm!(
            "vsetivli zero, 1, e32, m1, ta, ma",
            "vmv.v.i v8, 0",
            "vmv.v.i v9, 0",
            "vmv.v.i v10, 0",
            "vmv.v.i v11, 0",
            "vmv.v.i v12, 0",
            "vmv.v.i v13, 0",
            "vmv.v.i v14, 0",
            "vmv.v.i v15, 0",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "addi t0, {px}, 16",
            "addi t1, {px}, 32",
            "addi t2, {px}, 48",
            "vle8.v v16, ({px})",
            "vle8.v v17, (t0)",
            "vle8.v v18, (t1)",
            "vle8.v v19, (t2)",
            // Row 0 -> v8 (lo), v12 (hi)
            "vle8.v v1, ({q0})",
            "addi t0, {q0}, 16",
            "vle8.v v2, (t0)",
            "vand.vi v3, v1, 15",
            "vand.vi v4, v2, 15",
            "vwmul.vv v20, v3, v16",
            "vwmacc.vv v20, v4, v17",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v8, v20, v8",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vsrl.vi v3, v1, 4",
            "vsrl.vi v4, v2, 4",
            "vwmul.vv v20, v3, v18",
            "vwmacc.vv v20, v4, v19",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v12, v20, v12",
            // Row 1 -> v9 (lo), v13 (hi)
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vle8.v v1, ({q1})",
            "addi t0, {q1}, 16",
            "vle8.v v2, (t0)",
            "vand.vi v3, v1, 15",
            "vand.vi v4, v2, 15",
            "vwmul.vv v20, v3, v16",
            "vwmacc.vv v20, v4, v17",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v9, v20, v9",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vsrl.vi v3, v1, 4",
            "vsrl.vi v4, v2, 4",
            "vwmul.vv v20, v3, v18",
            "vwmacc.vv v20, v4, v19",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v13, v20, v13",
            // Row 2 -> v10 (lo), v14 (hi)
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vle8.v v1, ({q2})",
            "addi t0, {q2}, 16",
            "vle8.v v2, (t0)",
            "vand.vi v3, v1, 15",
            "vand.vi v4, v2, 15",
            "vwmul.vv v20, v3, v16",
            "vwmacc.vv v20, v4, v17",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v10, v20, v10",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vsrl.vi v3, v1, 4",
            "vsrl.vi v4, v2, 4",
            "vwmul.vv v20, v3, v18",
            "vwmacc.vv v20, v4, v19",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v14, v20, v14",
            // Row 3 -> v11 (lo), v15 (hi)
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vle8.v v1, ({q3})",
            "addi t0, {q3}, 16",
            "vle8.v v2, (t0)",
            "vand.vi v3, v1, 15",
            "vand.vi v4, v2, 15",
            "vwmul.vv v20, v3, v16",
            "vwmacc.vv v20, v4, v17",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v11, v20, v11",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vsrl.vi v3, v1, 4",
            "vsrl.vi v4, v2, 4",
            "vwmul.vv v20, v3, v18",
            "vwmacc.vv v20, v4, v19",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v15, v20, v15",
            // Store 8 scalar sums cleanly
            "vsetivli zero, 1, e32, m1, ta, ma",
            "vmv.x.s t0, v8",  "sw t0, 0({po})",
            "vmv.x.s t0, v9",  "sw t0, 4({po})",
            "vmv.x.s t0, v10", "sw t0, 8({po})",
            "vmv.x.s t0, v11", "sw t0, 12({po})",
            "vmv.x.s t0, v12", "sw t0, 16({po})",
            "vmv.x.s t0, v13", "sw t0, 20({po})",
            "vmv.x.s t0, v14", "sw t0, 24({po})",
            "vmv.x.s t0, v15", "sw t0, 28({po})",
            q0 = in(reg) pqs0,
            q1 = in(reg) pqs1,
            q2 = in(reg) pqs2,
            q3 = in(reg) pqs3,
            px = in(reg) px,
            po = in(reg) out_sums.as_mut_ptr(),
            out("t0") _,
            out("t1") _,
            out("t2") _,
            out("v1") _,
            out("v2") _,
            out("v3") _,
            out("v4") _,
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
            options(nostack),
        );
    }
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[inline]
fn dot_q5_k_chunk64_1x_rvv(
    pqs: *const u8,
    pqh: *const u8,
    px: *const i8,
    sh1: usize,
) -> (i32, i32) {
    let (s_lo, s_hi): (i32, i32);
    let sh2 = sh1 + 1;
    // SAFETY: `pqs` and `pqh` hold 32 readable bytes, `px` holds 64 `i8` bytes; `vsetivli 16` at
    // `e8,m1` -> `e16,m2` -> `e32,m1`; max `|w*x|*2 = 7874 << i16::MAX`.
    unsafe {
        core::arch::asm!(
            "vsetivli zero, 1, e32, m1, ta, ma",
            "vmv.v.i v8, 0",
            "vmv.v.i v9, 0",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "addi t0, {px}, 16",
            "addi t1, {px}, 32",
            "addi t2, {px}, 48",
            "vle8.v v16, ({px})",
            "vle8.v v17, (t0)",
            "vle8.v v18, (t1)",
            "vle8.v v19, (t2)",
            // Half 0 (elems 0..16 & 32..48)
            "vle8.v v1, ({pqs})",
            "vle8.v v5, ({pqh})",
            "vand.vi v3, v1, 15",
            "vsrl.vx v6, v5, {sh1}",
            "vand.vi v6, v6, 1",
            "vsll.vi v6, v6, 4",
            "vor.vv v3, v3, v6",
            "vsrl.vi v4, v1, 4",
            "vsrl.vx v6, v5, {sh2}",
            "vand.vi v6, v6, 1",
            "vsll.vi v6, v6, 4",
            "vor.vv v4, v4, v6",
            "vwmul.vv v20, v3, v16",
            "vwmul.vv v22, v4, v18",
            // Half 1 (elems 16..32 & 48..64)
            "addi t0, {pqs}, 16",
            "addi t1, {pqh}, 16",
            "vle8.v v1, (t0)",
            "vle8.v v5, (t1)",
            "vand.vi v3, v1, 15",
            "vsrl.vx v6, v5, {sh1}",
            "vand.vi v6, v6, 1",
            "vsll.vi v6, v6, 4",
            "vor.vv v3, v3, v6",
            "vsrl.vi v4, v1, 4",
            "vsrl.vx v6, v5, {sh2}",
            "vand.vi v6, v6, 1",
            "vsll.vi v6, v6, 4",
            "vor.vv v4, v4, v6",
            "vwmacc.vv v20, v3, v17",
            "vwmacc.vv v22, v4, v19",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v8, v20, v8",
            "vwredsum.vs v9, v22, v9",
            "vsetivli zero, 1, e32, m1, ta, ma",
            "vmv.x.s {slo}, v8",
            "vmv.x.s {shi}, v9",
            pqs = in(reg) pqs,
            pqh = in(reg) pqh,
            px = in(reg) px,
            sh1 = in(reg) sh1,
            sh2 = in(reg) sh2,
            slo = lateout(reg) s_lo,
            shi = lateout(reg) s_hi,
            out("t0") _,
            out("t1") _,
            out("t2") _,
            out("v1") _,
            out("v3") _,
            out("v4") _,
            out("v5") _,
            out("v6") _,
            out("v8") _,
            out("v9") _,
            out("v16") _,
            out("v17") _,
            out("v18") _,
            out("v19") _,
            out("v20") _,
            out("v21") _,
            out("v22") _,
            out("v23") _,
            options(nostack, readonly),
        );
    }
    (s_lo, s_hi)
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[inline]
fn dot_q5_k_chunk64_4x_rvv(
    pb0: *const u8,
    pb1: *const u8,
    pb2: *const u8,
    pb3: *const u8,
    chunk: usize,
    px: *const i8,
    out_sums: &mut [i32; 8],
) {
    let sh1 = chunk * 2;
    let q_off = chunk * 32;
    let r0 = dot_q5_k_chunk64_1x_rvv(
        // SAFETY: `chunk < 4`, `48 + chunk * 32 + 32 <= 176` within valid `Q5_K` block.
        unsafe { pb0.add(48 + q_off) },
        // SAFETY: `16 + 32 <= 176`.
        unsafe { pb0.add(16) },
        px,
        sh1,
    );
    let r1 = dot_q5_k_chunk64_1x_rvv(
        // SAFETY: `chunk < 4`, `48 + chunk * 32 + 32 <= 176` within valid `Q5_K` block.
        unsafe { pb1.add(48 + q_off) },
        // SAFETY: `16 + 32 <= 176`.
        unsafe { pb1.add(16) },
        px,
        sh1,
    );
    let r2 = dot_q5_k_chunk64_1x_rvv(
        // SAFETY: `chunk < 4`, `48 + chunk * 32 + 32 <= 176` within valid `Q5_K` block.
        unsafe { pb2.add(48 + q_off) },
        // SAFETY: `16 + 32 <= 176`.
        unsafe { pb2.add(16) },
        px,
        sh1,
    );
    let r3 = dot_q5_k_chunk64_1x_rvv(
        // SAFETY: `chunk < 4`, `48 + chunk * 32 + 32 <= 176` within valid `Q5_K` block.
        unsafe { pb3.add(48 + q_off) },
        // SAFETY: `16 + 32 <= 176`.
        unsafe { pb3.add(16) },
        px,
        sh1,
    );
    out_sums[0] = r0.0;
    out_sums[1] = r1.0;
    out_sums[2] = r2.0;
    out_sums[3] = r3.0;
    out_sums[4] = r0.1;
    out_sums[5] = r1.1;
    out_sums[6] = r2.1;
    out_sums[7] = r3.1;
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[inline]
fn dot_q6_k_quad16_1x_rvv(pql_lo: *const u8, pqh: *const u8, px: *const i8) -> [i32; 4] {
    let (s0, s1, s2, s3): (i32, i32, i32, i32);
    let c32 = 32_usize;
    // SAFETY: `pql_lo` holds 48 bytes (`ql[0..16]` and `ql[32..48]`), `pqh` holds 16 bytes,
    // `px` holds `4` 16-byte slices spaced by 32 bytes; `q - 32` uses `vsub.vx` (`32` exceeds
    // signed 5-bit `vadd.vi` range); `e8,m1` -> `e16,m2` -> `e32,m1`.
    unsafe {
        core::arch::asm!(
            "vsetivli zero, 1, e32, m1, ta, ma",
            "vmv.v.i v8, 0",
            "vmv.v.i v9, 0",
            "vmv.v.i v10, 0",
            "vmv.v.i v11, 0",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "addi t0, {px}, 32",
            "addi t1, {px}, 64",
            "addi t2, {px}, 96",
            "vle8.v v16, ({px})",
            "vle8.v v17, (t0)",
            "vle8.v v18, (t1)",
            "vle8.v v19, (t2)",
            "vle8.v v1, ({pql})",
            "addi t0, {pql}, 32",
            "vle8.v v2, (t0)",
            "vle8.v v3, ({pqh})",
            // q1: (ql[0] & 0xF) | ((qh & 3) << 4) - 32
            "vand.vi v4, v1, 15",
            "vand.vi v5, v3, 3",
            "vsll.vi v5, v5, 4",
            "vor.vv v4, v4, v5",
            "vsub.vx v4, v4, {c32}",
            "vwmul.vv v20, v4, v16",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v8, v20, v8",
            // q2: (ql[32] & 0xF) | (((qh >> 2) & 3) << 4) - 32
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vand.vi v4, v2, 15",
            "vsrl.vi v5, v3, 2",
            "vand.vi v5, v5, 3",
            "vsll.vi v5, v5, 4",
            "vor.vv v4, v4, v5",
            "vsub.vx v4, v4, {c32}",
            "vwmul.vv v20, v4, v17",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v9, v20, v9",
            // q3: (ql[0] >> 4) | (((qh >> 4) & 3) << 4) - 32
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vsrl.vi v4, v1, 4",
            "vsrl.vi v5, v3, 4",
            "vand.vi v5, v5, 3",
            "vsll.vi v5, v5, 4",
            "vor.vv v4, v4, v5",
            "vsub.vx v4, v4, {c32}",
            "vwmul.vv v20, v4, v18",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v10, v20, v10",
            // q4: (ql[32] >> 4) | (((qh >> 6) & 3) << 4) - 32
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vsrl.vi v4, v2, 4",
            "vsrl.vi v5, v3, 6",
            "vand.vi v5, v5, 3",
            "vsll.vi v5, v5, 4",
            "vor.vv v4, v4, v5",
            "vsub.vx v4, v4, {c32}",
            "vwmul.vv v20, v4, v19",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v11, v20, v11",
            "vsetivli zero, 1, e32, m1, ta, ma",
            "vmv.x.s {s0}, v8",
            "vmv.x.s {s1}, v9",
            "vmv.x.s {s2}, v10",
            "vmv.x.s {s3}, v11",
            pql = in(reg) pql_lo,
            pqh = in(reg) pqh,
            px = in(reg) px,
            c32 = in(reg) c32,
            s0 = lateout(reg) s0,
            s1 = lateout(reg) s1,
            s2 = lateout(reg) s2,
            s3 = lateout(reg) s3,
            out("t0") _,
            out("t1") _,
            out("t2") _,
            out("v1") _,
            out("v2") _,
            out("v3") _,
            out("v4") _,
            out("v5") _,
            out("v8") _,
            out("v9") _,
            out("v10") _,
            out("v11") _,
            out("v16") _,
            out("v17") _,
            out("v18") _,
            out("v19") _,
            out("v20") _,
            out("v21") _,
            options(nostack, readonly),
        );
    }
    [s0, s1, s2, s3]
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[inline]
fn dot_q4_k_q8_k_row_rvv(w_blocks: &[u8], x_blocks: &[BlockQ8K]) -> f32 {
    let mut acc = 0.0_f32;
    for (wb, xb) in w_blocks
        .chunks_exact(Q4_K_M_BLOCK_BYTES)
        .zip(x_blocks.iter())
    {
        let d = f16_to_f32_finite(u16::from_le_bytes([wb[0], wb[1]]));
        let dmin = f16_to_f32_finite(u16::from_le_bytes([wb[2], wb[3]]));
        if (d == 0.0 && dmin == 0.0) || xb.d == 0.0 {
            continue;
        }
        let (sc, mins) = unpack_q4_q5_k_scales_mins(&wb[4..16]);
        let mut sumi = 0_i32;
        let mut sum_mins = 0_i32;
        for chunk in 0..4 {
            let is = chunk * 2;
            let (s_lo, s_hi) = dot_q4_k_chunk64_1x_rvv(
                wb[16 + chunk * 32..].as_ptr(),
                xb.qs[chunk * 64..].as_ptr(),
            );
            sumi += i32::from(sc[is]) * s_lo + i32::from(sc[is + 1]) * s_hi;
            let bsum_lo = i32::from(xb.bsums[chunk * 4]) + i32::from(xb.bsums[chunk * 4 + 1]);
            let bsum_hi = i32::from(xb.bsums[chunk * 4 + 2]) + i32::from(xb.bsums[chunk * 4 + 3]);
            sum_mins += i32::from(mins[is]) * bsum_lo + i32::from(mins[is + 1]) * bsum_hi;
        }
        acc += (d * xb.d) * i32_to_f32(sumi) - (dmin * xb.d) * i32_to_f32(sum_mins);
    }
    acc
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[inline]
fn dot_q4_k_q8_k_4row_rvv(
    r0: &[u8],
    r1: &[u8],
    r2: &[u8],
    r3: &[u8],
    x_blocks: &[BlockQ8K],
) -> [f32; 4] {
    let mut acc = [0.0_f32; 4];
    let mut chunk_sums = [0_i32; 8];
    for (idx, xb) in x_blocks.iter().enumerate() {
        if xb.d == 0.0 {
            continue;
        }
        let off = idx * Q4_K_M_BLOCK_BYTES;
        let rows = [
            &r0[off..off + Q4_K_M_BLOCK_BYTES],
            &r1[off..off + Q4_K_M_BLOCK_BYTES],
            &r2[off..off + Q4_K_M_BLOCK_BYTES],
            &r3[off..off + Q4_K_M_BLOCK_BYTES],
        ];
        let mut scs = [[0_u8; 8]; 4];
        let mut mins = [[0_u8; 8]; 4];
        for r in 0..4 {
            let (s, m) = unpack_q4_q5_k_scales_mins(&rows[r][4..16]);
            scs[r] = s;
            mins[r] = m;
        }
        let mut sumi = [0_i32; 4];
        let mut sum_m = [0_i32; 4];
        for chunk in 0..4 {
            let is = chunk * 2;
            let q_off = 16 + chunk * 32;
            dot_q4_k_chunk64_4x_rvv(
                rows[0][q_off..].as_ptr(),
                rows[1][q_off..].as_ptr(),
                rows[2][q_off..].as_ptr(),
                rows[3][q_off..].as_ptr(),
                xb.qs[chunk * 64..].as_ptr(),
                &mut chunk_sums,
            );
            let bsum_lo = i32::from(xb.bsums[chunk * 4]) + i32::from(xb.bsums[chunk * 4 + 1]);
            let bsum_hi = i32::from(xb.bsums[chunk * 4 + 2]) + i32::from(xb.bsums[chunk * 4 + 3]);
            for r in 0..4 {
                sumi[r] += i32::from(scs[r][is]) * chunk_sums[r]
                    + i32::from(scs[r][is + 1]) * chunk_sums[r + 4];
                sum_m[r] += i32::from(mins[r][is]) * bsum_lo + i32::from(mins[r][is + 1]) * bsum_hi;
            }
        }
        for r in 0..4 {
            let d = f16_to_f32_finite(u16::from_le_bytes([rows[r][0], rows[r][1]]));
            let dmin = f16_to_f32_finite(u16::from_le_bytes([rows[r][2], rows[r][3]]));
            acc[r] += (d * xb.d) * i32_to_f32(sumi[r]) - (dmin * xb.d) * i32_to_f32(sum_m[r]);
        }
    }
    acc
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[inline]
fn dot_q5_k_q8_k_row(w_blocks: &[u8], x_blocks: &[BlockQ8K]) -> f32 {
    let mut acc = 0.0_f32;
    for (wb, xb) in w_blocks.chunks_exact(Q5_K_BLOCK_BYTES).zip(x_blocks.iter()) {
        let d = f16_to_f32_finite(u16::from_le_bytes([wb[0], wb[1]]));
        let dmin = f16_to_f32_finite(u16::from_le_bytes([wb[2], wb[3]]));
        if (d == 0.0 && dmin == 0.0) || xb.d == 0.0 {
            continue;
        }
        let (sc, mins) = unpack_q4_q5_k_scales_mins(&wb[4..16]);
        let mut sumi = 0_i32;
        let mut sum_mins = 0_i32;
        for chunk in 0..4 {
            let is = chunk * 2;
            let (s_lo, s_hi) = dot_q5_k_chunk64_1x_rvv(
                wb[48 + chunk * 32..].as_ptr(),
                wb[16..48].as_ptr(),
                xb.qs[chunk * 64..].as_ptr(),
                chunk * 2,
            );
            sumi += i32::from(sc[is]) * s_lo + i32::from(sc[is + 1]) * s_hi;
            let bsum_lo = i32::from(xb.bsums[chunk * 4]) + i32::from(xb.bsums[chunk * 4 + 1]);
            let bsum_hi = i32::from(xb.bsums[chunk * 4 + 2]) + i32::from(xb.bsums[chunk * 4 + 3]);
            sum_mins += i32::from(mins[is]) * bsum_lo + i32::from(mins[is + 1]) * bsum_hi;
        }
        acc += (d * xb.d) * i32_to_f32(sumi) - (dmin * xb.d) * i32_to_f32(sum_mins);
    }
    acc
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[inline]
fn dot_q5_k_q8_k_4row(
    r0: &[u8],
    r1: &[u8],
    r2: &[u8],
    r3: &[u8],
    x_blocks: &[BlockQ8K],
) -> [f32; 4] {
    let mut acc = [0.0_f32; 4];
    let mut chunk_sums = [0_i32; 8];
    for (idx, xb) in x_blocks.iter().enumerate() {
        if xb.d == 0.0 {
            continue;
        }
        let off = idx * Q5_K_BLOCK_BYTES;
        let rows = [
            &r0[off..off + Q5_K_BLOCK_BYTES],
            &r1[off..off + Q5_K_BLOCK_BYTES],
            &r2[off..off + Q5_K_BLOCK_BYTES],
            &r3[off..off + Q5_K_BLOCK_BYTES],
        ];
        let mut scs = [[0_u8; 8]; 4];
        let mut mins = [[0_u8; 8]; 4];
        for r in 0..4 {
            let (s, m) = unpack_q4_q5_k_scales_mins(&rows[r][4..16]);
            scs[r] = s;
            mins[r] = m;
        }
        let mut sumi = [0_i32; 4];
        let mut sum_m = [0_i32; 4];
        for chunk in 0..4 {
            let is = chunk * 2;
            dot_q5_k_chunk64_4x_rvv(
                rows[0].as_ptr(),
                rows[1].as_ptr(),
                rows[2].as_ptr(),
                rows[3].as_ptr(),
                chunk,
                xb.qs[chunk * 64..].as_ptr(),
                &mut chunk_sums,
            );
            let bsum_lo = i32::from(xb.bsums[chunk * 4]) + i32::from(xb.bsums[chunk * 4 + 1]);
            let bsum_hi = i32::from(xb.bsums[chunk * 4 + 2]) + i32::from(xb.bsums[chunk * 4 + 3]);
            for r in 0..4 {
                sumi[r] += i32::from(scs[r][is]) * chunk_sums[r]
                    + i32::from(scs[r][is + 1]) * chunk_sums[r + 4];
                sum_m[r] += i32::from(mins[r][is]) * bsum_lo + i32::from(mins[r][is + 1]) * bsum_hi;
            }
        }
        for r in 0..4 {
            let d = f16_to_f32_finite(u16::from_le_bytes([rows[r][0], rows[r][1]]));
            let dmin = f16_to_f32_finite(u16::from_le_bytes([rows[r][2], rows[r][3]]));
            acc[r] += (d * xb.d) * i32_to_f32(sumi[r]) - (dmin * xb.d) * i32_to_f32(sum_m[r]);
        }
    }
    acc
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[inline]
fn dot_q6_k_q8_k_row(w_blocks: &[u8], x_blocks: &[BlockQ8K]) -> f32 {
    let mut acc = 0.0_f32;
    for (wb, xb) in w_blocks.chunks_exact(Q6_K_BLOCK_BYTES).zip(x_blocks.iter()) {
        let d = f16_to_f32_finite(u16::from_le_bytes([wb[208], wb[209]]));
        if d == 0.0 || xb.d == 0.0 {
            continue;
        }
        let scales = &wb[192..208];
        let mut sumi = 0_i32;
        for half_idx in 0..2 {
            let n = half_idx * 128;
            let q_lo_off = half_idx * 64;
            let q_hi_off = 128 + half_idx * 32;
            let sc_base = half_idx * 8;
            for is in 0..2 {
                let l_base = is * 16;
                let [s1, s2, s3, s4] = dot_q6_k_quad16_1x_rvv(
                    wb[q_lo_off + l_base..].as_ptr(),
                    wb[q_hi_off + l_base..].as_ptr(),
                    xb.qs[n + l_base..].as_ptr(),
                );
                let sc1 = i32::from(i8::from_ne_bytes([scales[sc_base + is]]));
                let sc2 = i32::from(i8::from_ne_bytes([scales[sc_base + is + 2]]));
                let sc3 = i32::from(i8::from_ne_bytes([scales[sc_base + is + 4]]));
                let sc4 = i32::from(i8::from_ne_bytes([scales[sc_base + is + 6]]));
                sumi += sc1 * s1 + sc2 * s2 + sc3 * s3 + sc4 * s4;
            }
        }
        acc += (d * xb.d) * i32_to_f32(sumi);
    }
    acc
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[inline]
fn dot_q6_k_q8_k_4row(
    r0: &[u8],
    r1: &[u8],
    r2: &[u8],
    r3: &[u8],
    x_blocks: &[BlockQ8K],
) -> [f32; 4] {
    let mut acc = [0.0_f32; 4];
    for (idx, xb) in x_blocks.iter().enumerate() {
        if xb.d == 0.0 {
            continue;
        }
        let off = idx * Q6_K_BLOCK_BYTES;
        let rows = [
            &r0[off..off + Q6_K_BLOCK_BYTES],
            &r1[off..off + Q6_K_BLOCK_BYTES],
            &r2[off..off + Q6_K_BLOCK_BYTES],
            &r3[off..off + Q6_K_BLOCK_BYTES],
        ];
        let mut sumi = [0_i32; 4];
        for half_idx in 0..2 {
            let n = half_idx * 128;
            let q_lo_off = half_idx * 64;
            let q_hi_off = 128 + half_idx * 32;
            let sc_base = half_idx * 8;
            for is in 0..2 {
                let l_base = is * 16;
                let px = xb.qs[n + l_base..].as_ptr();
                for r in 0..4 {
                    let [s1, s2, s3, s4] = dot_q6_k_quad16_1x_rvv(
                        rows[r][q_lo_off + l_base..].as_ptr(),
                        rows[r][q_hi_off + l_base..].as_ptr(),
                        px,
                    );
                    let scales = &rows[r][192..208];
                    let sc1 = i32::from(i8::from_ne_bytes([scales[sc_base + is]]));
                    let sc2 = i32::from(i8::from_ne_bytes([scales[sc_base + is + 2]]));
                    let sc3 = i32::from(i8::from_ne_bytes([scales[sc_base + is + 4]]));
                    let sc4 = i32::from(i8::from_ne_bytes([scales[sc_base + is + 6]]));
                    sumi[r] += sc1 * s1 + sc2 * s2 + sc3 * s3 + sc4 * s4;
                }
            }
        }
        for r in 0..4 {
            let d = f16_to_f32_finite(u16::from_le_bytes([rows[r][208], rows[r][209]]));
            acc[r] += (d * xb.d) * i32_to_f32(sumi[r]);
        }
    }
    acc
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[inline]
fn dot_tq2_0_q8_k_block_rvv(w_blk: &[u8], x_blk: &BlockQ8K) -> f32 {
    let dw = f16_to_f32_finite(u16::from_le_bytes([w_blk[0], w_blk[1]]));
    if dw == 0.0 || x_blk.d == 0.0 {
        return 0.0;
    }
    let out_i32: i32;
    let pw = w_blk[2..66].as_ptr();
    let px = x_blk.qs.as_ptr();
    let grp = 4_usize;
    let stride4 = 4_isize;
    // SAFETY: `w_blk[2..66]` holds 64 packed 2-bit-trit bytes (`4 x 16B`) and `x_blk.qs` holds
    // 256 `i8`s (`4 x 64B`); `vl = 16` at `e8,m1` -> `e16,m2` -> `e32,m1`; `vlse8.v` uses
    // stride `4` to gather interleaved activation lanes matching `unpack_tq2_0_block_i8`.
    unsafe {
        core::arch::asm!(
            "vsetivli zero, 1, e32, m1, ta, ma",
            "vmv.v.i v1, 0",
            "2:",
            "vsetivli zero, 16, e8, m1, ta, ma",
            "vle8.v v5, ({pw})",
            "addi t0, {px}, 1",
            "addi t1, {px}, 2",
            "addi t2, {px}, 3",
            "vlse8.v v16, ({px}), {st}",
            "vlse8.v v17, (t0), {st}",
            "vlse8.v v18, (t1), {st}",
            "vlse8.v v19, (t2), {st}",
            "vand.vi v6, v5, 3",
            "vadd.vi v6, v6, -1",
            "vwmul.vv v20, v6, v16",
            "vsrl.vi v6, v5, 2",
            "vand.vi v6, v6, 3",
            "vadd.vi v6, v6, -1",
            "vwmacc.vv v20, v6, v17",
            "vsrl.vi v6, v5, 4",
            "vand.vi v6, v6, 3",
            "vadd.vi v6, v6, -1",
            "vwmacc.vv v20, v6, v18",
            "vsrl.vi v6, v5, 6",
            "vand.vi v6, v6, 3",
            "vadd.vi v6, v6, -1",
            "vwmacc.vv v20, v6, v19",
            "vsetivli zero, 16, e16, m2, ta, ma",
            "vwredsum.vs v1, v20, v1",
            "addi {pw}, {pw}, 16",
            "addi {px}, {px}, 64",
            "addi {grp}, {grp}, -1",
            "bnez {grp}, 2b",
            "vsetivli zero, 1, e32, m1, ta, ma",
            "vmv.x.s {out}, v1",
            pw = inout(reg) pw => _,
            px = inout(reg) px => _,
            grp = inout(reg) grp => _,
            st = in(reg) stride4,
            out = lateout(reg) out_i32,
            out("t0") _,
            out("t1") _,
            out("t2") _,
            out("v1") _,
            out("v5") _,
            out("v6") _,
            out("v16") _,
            out("v17") _,
            out("v18") _,
            out("v19") _,
            out("v20") _,
            out("v21") _,
            options(nostack, readonly),
        );
    }
    i32_to_f32(out_i32) * (dw * x_blk.d)
}

#[cfg(any(not(target_arch = "riscv64"), miri))]
#[inline]
fn dot_q4_k_q8_k_row(w_blocks: &[u8], x_blocks: &[BlockQ8K]) -> f32 {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        dot_q4_k_q8_k_row_rvv(w_blocks, x_blocks)
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        w_blocks
            .chunks_exact(Q4_K_M_BLOCK_BYTES)
            .zip(x_blocks.iter())
            .map(|(wb, xb)| dot_q4_k_q8_k_block_scalar(wb, xb))
            .sum()
    }
}

#[cfg(any(not(target_arch = "riscv64"), miri))]
#[inline]
fn dot_q4_k_q8_k_4row(
    r0: &[u8],
    r1: &[u8],
    r2: &[u8],
    r3: &[u8],
    x_blocks: &[BlockQ8K],
) -> [f32; 4] {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        dot_q4_k_q8_k_4row_rvv(r0, r1, r2, r3, x_blocks)
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        let mut acc = [0.0_f32; 4];
        for (idx, xb) in x_blocks.iter().enumerate() {
            let off = idx * Q4_K_M_BLOCK_BYTES;
            acc[0] += dot_q4_k_q8_k_block_scalar(&r0[off..off + Q4_K_M_BLOCK_BYTES], xb);
            acc[1] += dot_q4_k_q8_k_block_scalar(&r1[off..off + Q4_K_M_BLOCK_BYTES], xb);
            acc[2] += dot_q4_k_q8_k_block_scalar(&r2[off..off + Q4_K_M_BLOCK_BYTES], xb);
            acc[3] += dot_q4_k_q8_k_block_scalar(&r3[off..off + Q4_K_M_BLOCK_BYTES], xb);
        }
        acc
    }
}

#[inline]
fn dot_q5_k_q8_k_row_dispatch(w_blocks: &[u8], x_blocks: &[BlockQ8K]) -> f32 {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        dot_q5_k_q8_k_row(w_blocks, x_blocks)
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        w_blocks
            .chunks_exact(Q5_K_BLOCK_BYTES)
            .zip(x_blocks.iter())
            .map(|(wb, xb)| dot_q5_k_q8_k_block_scalar(wb, xb))
            .sum()
    }
}

#[inline]
fn dot_q5_k_q8_k_4row_dispatch(
    r0: &[u8],
    r1: &[u8],
    r2: &[u8],
    r3: &[u8],
    x_blocks: &[BlockQ8K],
) -> [f32; 4] {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        dot_q5_k_q8_k_4row(r0, r1, r2, r3, x_blocks)
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        let mut acc = [0.0_f32; 4];
        for (idx, xb) in x_blocks.iter().enumerate() {
            let off = idx * Q5_K_BLOCK_BYTES;
            acc[0] += dot_q5_k_q8_k_block_scalar(&r0[off..off + Q5_K_BLOCK_BYTES], xb);
            acc[1] += dot_q5_k_q8_k_block_scalar(&r1[off..off + Q5_K_BLOCK_BYTES], xb);
            acc[2] += dot_q5_k_q8_k_block_scalar(&r2[off..off + Q5_K_BLOCK_BYTES], xb);
            acc[3] += dot_q5_k_q8_k_block_scalar(&r3[off..off + Q5_K_BLOCK_BYTES], xb);
        }
        acc
    }
}

#[inline]
fn dot_q6_k_q8_k_row_dispatch(w_blocks: &[u8], x_blocks: &[BlockQ8K]) -> f32 {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        dot_q6_k_q8_k_row(w_blocks, x_blocks)
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        w_blocks
            .chunks_exact(Q6_K_BLOCK_BYTES)
            .zip(x_blocks.iter())
            .map(|(wb, xb)| dot_q6_k_q8_k_block_scalar(wb, xb))
            .sum()
    }
}

#[inline]
fn dot_q6_k_q8_k_4row_dispatch(
    r0: &[u8],
    r1: &[u8],
    r2: &[u8],
    r3: &[u8],
    x_blocks: &[BlockQ8K],
) -> [f32; 4] {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        dot_q6_k_q8_k_4row(r0, r1, r2, r3, x_blocks)
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        let mut acc = [0.0_f32; 4];
        for (idx, xb) in x_blocks.iter().enumerate() {
            let off = idx * Q6_K_BLOCK_BYTES;
            acc[0] += dot_q6_k_q8_k_block_scalar(&r0[off..off + Q6_K_BLOCK_BYTES], xb);
            acc[1] += dot_q6_k_q8_k_block_scalar(&r1[off..off + Q6_K_BLOCK_BYTES], xb);
            acc[2] += dot_q6_k_q8_k_block_scalar(&r2[off..off + Q6_K_BLOCK_BYTES], xb);
            acc[3] += dot_q6_k_q8_k_block_scalar(&r3[off..off + Q6_K_BLOCK_BYTES], xb);
        }
        acc
    }
}

#[inline]
fn dot_tq1_0_q8_k_row(w_blocks: &[u8], x_blocks: &[BlockQ8K]) -> f32 {
    w_blocks
        .chunks_exact(TQ1_0_BLOCK_BYTES)
        .zip(x_blocks.iter())
        .map(|(wb, xb)| dot_tq1_0_q8_k_block(wb, xb))
        .sum()
}

#[inline]
fn dot_tq1_0_q8_k_4row(
    r0: &[u8],
    r1: &[u8],
    r2: &[u8],
    r3: &[u8],
    x_blocks: &[BlockQ8K],
) -> [f32; 4] {
    let mut acc = [0.0_f32; 4];
    for (idx, xb) in x_blocks.iter().enumerate() {
        let off = idx * TQ1_0_BLOCK_BYTES;
        acc[0] += dot_tq1_0_q8_k_block(&r0[off..off + TQ1_0_BLOCK_BYTES], xb);
        acc[1] += dot_tq1_0_q8_k_block(&r1[off..off + TQ1_0_BLOCK_BYTES], xb);
        acc[2] += dot_tq1_0_q8_k_block(&r2[off..off + TQ1_0_BLOCK_BYTES], xb);
        acc[3] += dot_tq1_0_q8_k_block(&r3[off..off + TQ1_0_BLOCK_BYTES], xb);
    }
    acc
}

#[inline]
fn dot_tq2_0_q8_k_block(w_blk: &[u8], x_blk: &BlockQ8K) -> f32 {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        dot_tq2_0_q8_k_block_rvv(w_blk, x_blk)
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        let Ok(arr) = w_blk.try_into() else {
            return 0.0;
        };
        dot_tq2_0_q8_k_block_scalar(arr, x_blk)
    }
}

#[inline]
fn dot_tq2_0_q8_k_row(w_blocks: &[u8], x_blocks: &[BlockQ8K]) -> f32 {
    w_blocks
        .chunks_exact(TQ2_0_BLOCK_BYTES)
        .zip(x_blocks.iter())
        .map(|(wb, xb)| dot_tq2_0_q8_k_block(wb, xb))
        .sum()
}

#[inline]
fn dot_tq2_0_q8_k_4row(
    r0: &[u8],
    r1: &[u8],
    r2: &[u8],
    r3: &[u8],
    x_blocks: &[BlockQ8K],
) -> [f32; 4] {
    let mut acc = [0.0_f32; 4];
    for (idx, xb) in x_blocks.iter().enumerate() {
        let off = idx * TQ2_0_BLOCK_BYTES;
        acc[0] += dot_tq2_0_q8_k_block(&r0[off..off + TQ2_0_BLOCK_BYTES], xb);
        acc[1] += dot_tq2_0_q8_k_block(&r1[off..off + TQ2_0_BLOCK_BYTES], xb);
        acc[2] += dot_tq2_0_q8_k_block(&r2[off..off + TQ2_0_BLOCK_BYTES], xb);
        acc[3] += dot_tq2_0_q8_k_block(&r3[off..off + TQ2_0_BLOCK_BYTES], xb);
    }
    acc
}

/// Computes single-row `Q4_K_M x BlockQ8K` dot product.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `x_blocks` is empty or `w_blocks.len() != x_blocks.len() * 144`.
pub fn vec_dot_q4_k_q8_k(w_blocks: &[u8], x_blocks: &[BlockQ8K]) -> Result<f32, K3Error> {
    validate_q8_block_row_bytes(
        "vec_dot_q4_k_q8_k",
        w_blocks,
        x_blocks.len(),
        Q4_K_M_BLOCK_BYTES,
    )?;
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        Ok(dot_q4_k_q8_k_row_rvv(w_blocks, x_blocks))
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        Ok(dot_q4_k_q8_k_row(w_blocks, x_blocks))
    }
}

/// Computes single-row `Q5_K x BlockQ8K` dot product.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `x_blocks` is empty or `w_blocks.len() != x_blocks.len() * 176`.
pub fn vec_dot_q5_k_q8_k(w_blocks: &[u8], x_blocks: &[BlockQ8K]) -> Result<f32, K3Error> {
    validate_q8_block_row_bytes(
        "vec_dot_q5_k_q8_k",
        w_blocks,
        x_blocks.len(),
        Q5_K_BLOCK_BYTES,
    )?;
    Ok(dot_q5_k_q8_k_row_dispatch(w_blocks, x_blocks))
}

/// Computes single-row `Q6_K x BlockQ8K` dot product.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `x_blocks` is empty or `w_blocks.len() != x_blocks.len() * 210`.
pub fn vec_dot_q6_k_q8_k(w_blocks: &[u8], x_blocks: &[BlockQ8K]) -> Result<f32, K3Error> {
    validate_q8_block_row_bytes(
        "vec_dot_q6_k_q8_k",
        w_blocks,
        x_blocks.len(),
        Q6_K_BLOCK_BYTES,
    )?;
    Ok(dot_q6_k_q8_k_row_dispatch(w_blocks, x_blocks))
}

/// Computes single-row `TQ1_0 x BlockQ8K` dot product.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `x_blocks` is empty or `w_blocks.len() != x_blocks.len() * 54`.
pub fn vec_dot_tq1_0_q8_k(w_blocks: &[u8], x_blocks: &[BlockQ8K]) -> Result<f32, K3Error> {
    validate_q8_block_row_bytes(
        "vec_dot_tq1_0_q8_k",
        w_blocks,
        x_blocks.len(),
        TQ1_0_BLOCK_BYTES,
    )?;
    Ok(dot_tq1_0_q8_k_row(w_blocks, x_blocks))
}

/// Computes single-row `TQ2_0 x BlockQ8K` dot product.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `x_blocks` is empty or `w_blocks.len() != x_blocks.len() * 66`.
pub fn vec_dot_tq2_0_q8_k(w_blocks: &[u8], x_blocks: &[BlockQ8K]) -> Result<f32, K3Error> {
    validate_q8_block_row_bytes(
        "vec_dot_tq2_0_q8_k",
        w_blocks,
        x_blocks.len(),
        TQ2_0_BLOCK_BYTES,
    )?;
    Ok(dot_tq2_0_q8_k_row(w_blocks, x_blocks))
}

/// Computes 4-row `M = 1` GEMV over `Q4_K_M` weights and pre-quantized [`BlockQ8K`] activations.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] on quantization type, dimension, or block count mismatch.
pub fn gemv_q4_k_q8_k(
    output: &mut [f32],
    weights: &TensorView<'_>,
    x_blocks: &[BlockQ8K],
) -> Result<(), K3Error> {
    let cols = x_blocks.len() * Q4_K_M_BLOCK_ELEMENTS;
    let (rows, _) = validate_q4_k_m_dimensions(output.len(), weights, cols)?;
    let blocks_per_row = cols / Q4_K_M_BLOCK_ELEMENTS;
    check_q8_block_len("gemv_q4_k_q8_k", weights, blocks_per_row, x_blocks.len())?;
    run_4row_chunked_gemv(
        output,
        weights.as_bytes(),
        rows,
        blocks_per_row * Q4_K_M_BLOCK_BYTES,
        0,
        Q4_K_M_BLOCK_BYTES,
        x_blocks,
        false,
        dot_q4_k_q8_k_4row_dispatch,
        dot_q4_k_q8_k_row_dispatch,
    );
    Ok(())
}

#[inline]
fn dot_q4_k_q8_k_row_dispatch(w_blocks: &[u8], x_blocks: &[BlockQ8K]) -> f32 {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        dot_q4_k_q8_k_row_rvv(w_blocks, x_blocks)
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        dot_q4_k_q8_k_row(w_blocks, x_blocks)
    }
}

#[inline]
fn dot_q4_k_q8_k_4row_dispatch(
    r0: &[u8],
    r1: &[u8],
    r2: &[u8],
    r3: &[u8],
    x_blocks: &[BlockQ8K],
) -> [f32; 4] {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        dot_q4_k_q8_k_4row_rvv(r0, r1, r2, r3, x_blocks)
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        dot_q4_k_q8_k_4row(r0, r1, r2, r3, x_blocks)
    }
}

/// Computes 4-row `M = 1` GEMV over `Q5_K` weights and pre-quantized [`BlockQ8K`] activations.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] on quantization type, dimension, or block count mismatch.
pub fn gemv_q5_k_q8_k(
    output: &mut [f32],
    weights: &TensorView<'_>,
    x_blocks: &[BlockQ8K],
) -> Result<(), K3Error> {
    let cols = x_blocks.len() * Q5_K_BLOCK_ELEMENTS;
    let (rows, _) = validate_q5_k_dimensions(output.len(), weights, cols)?;
    let blocks_per_row = cols / Q5_K_BLOCK_ELEMENTS;
    check_q8_block_len("gemv_q5_k_q8_k", weights, blocks_per_row, x_blocks.len())?;
    run_4row_chunked_gemv(
        output,
        weights.as_bytes(),
        rows,
        blocks_per_row * Q5_K_BLOCK_BYTES,
        0,
        Q5_K_BLOCK_BYTES,
        x_blocks,
        false,
        dot_q5_k_q8_k_4row_dispatch,
        dot_q5_k_q8_k_row_dispatch,
    );
    Ok(())
}

/// Computes 4-row `M = 1` GEMV over `Q6_K` weights and pre-quantized [`BlockQ8K`] activations.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] on quantization type, dimension, or block count mismatch.
pub fn gemv_q6_k_q8_k(
    output: &mut [f32],
    weights: &TensorView<'_>,
    x_blocks: &[BlockQ8K],
) -> Result<(), K3Error> {
    let cols = x_blocks.len() * Q6_K_BLOCK_ELEMENTS;
    let (rows, _) = validate_q6_k_dimensions(output.len(), weights, cols)?;
    let blocks_per_row = cols / Q6_K_BLOCK_ELEMENTS;
    check_q8_block_len("gemv_q6_k_q8_k", weights, blocks_per_row, x_blocks.len())?;
    run_4row_chunked_gemv(
        output,
        weights.as_bytes(),
        rows,
        blocks_per_row * Q6_K_BLOCK_BYTES,
        0,
        Q6_K_BLOCK_BYTES,
        x_blocks,
        false,
        dot_q6_k_q8_k_4row_dispatch,
        dot_q6_k_q8_k_row_dispatch,
    );
    Ok(())
}

/// Computes 4-row `M = 1` GEMV over `TQ1_0` ternary weights and pre-quantized [`BlockQ8K`]
/// activations.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] on quantization type, dimension, or block count mismatch.
pub fn gemv_tq1_0_q8_k(
    output: &mut [f32],
    weights: &TensorView<'_>,
    x_blocks: &[BlockQ8K],
) -> Result<(), K3Error> {
    let cols = x_blocks.len() * TQ1_0_BLOCK_ELEMENTS;
    let (rows, _) =
        validate_quant_matvec_dimensions(QuantType::TQ1_0, output.len(), weights, cols)?;
    let blocks_per_row = cols / TQ1_0_BLOCK_ELEMENTS;
    check_q8_block_len("gemv_tq1_0_q8_k", weights, blocks_per_row, x_blocks.len())?;
    run_4row_chunked_gemv(
        output,
        weights.as_bytes(),
        rows,
        blocks_per_row * TQ1_0_BLOCK_BYTES,
        0,
        TQ1_0_BLOCK_BYTES,
        x_blocks,
        false,
        dot_tq1_0_q8_k_4row,
        dot_tq1_0_q8_k_row,
    );
    Ok(())
}

/// Computes 4-row `M = 1` GEMV over `TQ2_0` ternary weights and pre-quantized [`BlockQ8K`]
/// activations.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] on quantization type, dimension, or block count mismatch.
pub fn gemv_tq2_0_q8_k(
    output: &mut [f32],
    weights: &TensorView<'_>,
    x_blocks: &[BlockQ8K],
) -> Result<(), K3Error> {
    let cols = x_blocks.len() * TQ2_0_BLOCK_ELEMENTS;
    let (rows, _) =
        validate_quant_matvec_dimensions(QuantType::TQ2_0, output.len(), weights, cols)?;
    let blocks_per_row = cols / TQ2_0_BLOCK_ELEMENTS;
    check_q8_block_len("gemv_tq2_0_q8_k", weights, blocks_per_row, x_blocks.len())?;
    run_4row_chunked_gemv(
        output,
        weights.as_bytes(),
        rows,
        blocks_per_row * TQ2_0_BLOCK_BYTES,
        0,
        TQ2_0_BLOCK_BYTES,
        x_blocks,
        false,
        dot_tq2_0_q8_k_4row,
        dot_tq2_0_q8_k_row,
    );
    Ok(())
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[allow(
    clippy::too_many_arguments,
    reason = "internal stack-chunked RVV dispatch helper parameterized over 4x and 1x block kernels"
)]
pub(crate) fn vec_dot_q8_k_chunked_rvv<F4, F1>(
    output: &mut [f32],
    raw_bytes: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
    blk_bytes: usize,
    mut kernel_4x: F4,
    mut kernel_1x: F1,
) where
    F4: FnMut(&[u8], &[u8], &[u8], &[u8], &[BlockQ8K]) -> [f32; 4],
    F1: FnMut(&[u8], &[BlockQ8K]) -> f32,
{
    let blocks_per_row = cols / Q8_K_BLOCK_ELEMENTS;
    let row_stride_bytes = blocks_per_row * blk_bytes;
    let mut x_scratch = [BlockQ8K::ZERO; Q8_K_STACK_BLOCKS];
    let mut blk_start = 0_usize;
    while blk_start < blocks_per_row {
        let count = (blocks_per_row - blk_start).min(Q8_K_STACK_BLOCKS);
        let elem_start = blk_start * Q8_K_BLOCK_ELEMENTS;
        let elem_end = elem_start + count * Q8_K_BLOCK_ELEMENTS;
        let _ = quantize_row_q8_k(&input[elem_start..elem_end], &mut x_scratch[..count]);
        let accumulate = blk_start > 0;
        run_4row_chunked_gemv(
            output,
            raw_bytes,
            rows,
            row_stride_bytes,
            blk_start,
            blk_bytes,
            &x_scratch[..count],
            accumulate,
            &mut kernel_4x,
            &mut kernel_1x,
        );
        blk_start += count;
    }
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
pub(crate) fn vec_dot_q4_k_m_4row_rvv(
    output: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
) -> Result<(), K3Error> {
    let (rows, cols) = validate_q4_k_m_dimensions(output.len(), weights, input.len())?;
    vec_dot_q8_k_chunked_rvv(
        output,
        weights.as_bytes(),
        rows,
        cols,
        input,
        Q4_K_M_BLOCK_BYTES,
        dot_q4_k_q8_k_4row_rvv,
        dot_q4_k_q8_k_row_rvv,
    );
    Ok(())
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
pub(crate) fn vec_dot_q5_k_4row_rvv(
    output: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
) -> Result<(), K3Error> {
    let (rows, cols) = validate_q5_k_dimensions(output.len(), weights, input.len())?;
    vec_dot_q8_k_chunked_rvv(
        output,
        weights.as_bytes(),
        rows,
        cols,
        input,
        Q5_K_BLOCK_BYTES,
        dot_q5_k_q8_k_4row,
        dot_q5_k_q8_k_row,
    );
    Ok(())
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
pub(crate) fn vec_dot_q6_k_4row_rvv(
    output: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
) -> Result<(), K3Error> {
    let (rows, cols) = validate_q6_k_dimensions(output.len(), weights, input.len())?;
    vec_dot_q8_k_chunked_rvv(
        output,
        weights.as_bytes(),
        rows,
        cols,
        input,
        Q6_K_BLOCK_BYTES,
        dot_q6_k_q8_k_4row,
        dot_q6_k_q8_k_row,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dequantize_q4_k_m_block;
    use crate::quant::{
        dequantize_q4_0_block, dequantize_q5_0_block, dequantize_q5_k_block, dequantize_q6_k_block,
        dequantize_q8_0_block,
    };

    fn test_f32_row(len: usize, seed: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(len);
        for i in 0..len {
            let v = u8::try_from((i.wrapping_mul(37).wrapping_add(seed * 13)) % 29).unwrap_or(0);
            out.push((f32::from(v) - 14.0) * 0.125);
        }
        out
    }

    fn make_q8_0_input_f32(blk: &BlockQ8_0) -> [f32; 32] {
        let mut out = [0.0_f32; 32];
        for (dst, &q) in out.iter_mut().zip(blk.qs.iter()) {
            *dst = f32::from(q) * blk.d;
        }
        out
    }

    fn make_q8_k_input_f32(blk: &BlockQ8K) -> [f32; 256] {
        let mut out = [0.0_f32; 256];
        for (dst, &q) in out.iter_mut().zip(blk.qs.iter()) {
            *dst = f32::from(q) * blk.d;
        }
        out
    }

    #[test]
    fn test_quantize_row_q8_0_and_q8_k_sanitization_and_bounds() {
        assert!(matches!(
            quantize_row_q8_0(&[], &mut []),
            Err(K3Error::TensorBoundsError { .. })
        ));
        assert!(matches!(
            quantize_row_q8_0(&[1.0; 31], &mut [BlockQ8_0::ZERO; 1]),
            Err(K3Error::TensorBoundsError { .. })
        ));
        assert!(matches!(
            quantize_row_q8_k(&[1.0; 256], &mut []),
            Err(K3Error::TensorBoundsError { .. })
        ));

        // Block 0: mixed finite + NaN (no Inf); Block 1: mixed finite + Inf/-Inf; Block 2: all non-finite.
        let mut x96 = test_f32_row(96, 1);
        x96[3] = f32::NAN;
        x96[11] = -f32::NAN;
        x96[32 + 7] = f32::INFINITY;
        x96[32 + 19] = f32::NEG_INFINITY;
        x96[64..96].fill(f32::NAN);
        x96[64] = f32::INFINITY;
        x96[95] = 0.0;
        let mut q8_0 = [BlockQ8_0::ZERO; 3];
        quantize_row_q8_0_f32(&x96, &mut q8_0).expect("quantize q8_0");
        assert!(q8_0[0].d > 0.0);
        assert_eq!(q8_0[0].qs[3], 0);
        assert_eq!(q8_0[0].qs[11], 0);
        assert!(q8_0[0].qs.iter().any(|&q| q != 0));
        let mut scalar_block_0 = BlockQ8_0::ZERO;
        quantize_block_q8_0_scalar(&x96[0..32], &mut scalar_block_0);
        assert_eq!(q8_0[0], scalar_block_0);
        assert!(q8_0[1].d > 0.0);
        assert_eq!(q8_0[1].qs[7], 0);
        assert_eq!(q8_0[1].qs[19], 0);
        assert!(q8_0[1].qs.iter().any(|&q| q != 0));
        assert_eq!(q8_0[2], BlockQ8_0::ZERO);

        // Block 0: mixed finite + NaN (no Inf); Block 1: mixed finite + +/-Inf; Block 2: all non-finite.
        let mut x768 = test_f32_row(768, 2);
        x768[3] = f32::NAN;
        x768[16 + 9] = -f32::NAN;
        x768[256 + 5] = f32::NEG_INFINITY;
        x768[256 + 37] = f32::INFINITY;
        x768[512..768].fill(f32::NAN);
        x768[512] = f32::NEG_INFINITY;
        let mut q8_k = [BlockQ8K::ZERO; 3];
        quantize_row_q8_k(&x768, &mut q8_k).expect("quantize q8_k");
        assert!(q8_k[0].d > 0.0);
        assert_eq!(q8_k[0].qs[3], 0);
        assert_eq!(q8_k[0].qs[16 + 9], 0);
        assert!(q8_k[0].qs.iter().any(|&q| q != 0));
        let mut scalar_block_k = BlockQ8K::ZERO;
        quantize_block_q8_k_scalar(&x768[0..256], &mut scalar_block_k);
        assert_eq!(q8_k[0], scalar_block_k);
        assert!(q8_k[1].d > 0.0);
        assert_eq!(q8_k[1].qs[5], 0);
        assert_eq!(q8_k[1].qs[37], 0);
        assert_eq!(q8_k[2], BlockQ8K::ZERO);
        for blk in &q8_k {
            for sb in 0..16 {
                let expected_bsum: i16 = blk.qs[sb * 16..(sb + 1) * 16]
                    .iter()
                    .map(|&q| i16::from(q))
                    .sum();
                assert_eq!(blk.bsums[sb], expected_bsum);
            }
        }
    }

    #[cfg(test)]
    fn test_q8_0_quantized_input(cols: usize, seed: usize) -> (Vec<BlockQ8_0>, Vec<f32>) {
        let blocks_per_row = cols / Q8_0_BLOCK_ELEMENTS;
        let x = test_f32_row(cols, seed);
        let mut x_blocks = vec![BlockQ8_0::ZERO; blocks_per_row];
        quantize_row_q8_0(&x, &mut x_blocks).expect("quantize input");
        let mut x_q8_dequant = Vec::with_capacity(cols);
        for xb in &x_blocks {
            x_q8_dequant.extend_from_slice(&make_q8_0_input_f32(xb));
        }
        (x_blocks, x_q8_dequant)
    }

    #[test]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "bounded modular test values fit in u8/u32"
    )]
    fn test_q4_0_q5_0_x_q8_0_block_and_4row_gemv_tails() {
        let (rows, blocks_per_row) = (5_usize, 2_usize);
        let cols = blocks_per_row * Q8_0_BLOCK_ELEMENTS;
        let shape = [cols, rows];
        let (x_blocks, x_q8_dequant) = test_q8_0_quantized_input(cols, 3);

        // Q4_0
        let mut w4 = vec![0_u8; rows * blocks_per_row * Q4_0_BLOCK_BYTES];
        let mut ref4 = vec![0.0_f32; rows];
        for r in 0..rows {
            for b in 0..blocks_per_row {
                let off = (r * blocks_per_row + b) * Q4_0_BLOCK_BYTES;
                w4[off..off + 2].copy_from_slice(&0x3400_u16.to_le_bytes()); // 0.25
                for j in 0..16 {
                    let lo = ((r * 3 + b * 5 + j) % 16) as u8;
                    let hi = ((r * 7 + b * 2 + j + 1) % 16) as u8;
                    w4[off + 2 + j] = (hi << 4) | lo;
                }
                let blk: &[u8; Q4_0_BLOCK_BYTES] = w4[off..off + Q4_0_BLOCK_BYTES]
                    .try_into()
                    .expect("q4_0 blk");
                let mut dq = [0.0_f32; 32];
                dequantize_q4_0_block(blk, &mut dq);
                ref4[r] += dq
                    .iter()
                    .zip(&x_q8_dequant[b * 32..(b + 1) * 32])
                    .map(|(&a, &c)| a * c)
                    .sum::<f32>();
            }
            let row_slice = &w4[r * blocks_per_row * Q4_0_BLOCK_BYTES
                ..(r + 1) * blocks_per_row * Q4_0_BLOCK_BYTES];
            let d1 = vec_dot_q4_0_q8_0(row_slice, &x_blocks).expect("vec_dot_q4_0_q8_0");
            assert!((d1 - ref4[r]).abs() < 1e-3);
        }
        let view4 =
            TensorView::from_bytes("w4", QuantType::Q4_0, &shape, 0, &w4).expect("q4_0 view");
        let mut out4 = vec![0.0_f32; rows];
        let mut out4_4x = vec![0.0_f32; rows];
        gemv_q4_0_q8_0(&mut out4, &view4, &x_blocks).expect("gemv_q4_0_q8_0");
        gemv_q4_0_q8_0_4x(&mut out4_4x, &view4, &x_blocks).expect("gemv_q4_0_q8_0_4x");
        assert_eq!(out4, out4_4x);
        for r in 0..rows {
            assert!((out4[r] - ref4[r]).abs() < 1e-3);
        }

        // Q5_0
        let mut w5 = vec![0_u8; rows * blocks_per_row * Q5_0_BLOCK_BYTES];
        let mut ref5 = vec![0.0_f32; rows];
        for r in 0..rows {
            for b in 0..blocks_per_row {
                let off = (r * blocks_per_row + b) * Q5_0_BLOCK_BYTES;
                w5[off..off + 2].copy_from_slice(&0x3400_u16.to_le_bytes());
                w5[off + 2..off + 6]
                    .copy_from_slice(&(0xA55A_0F01_u32.wrapping_add((r + b) as u32)).to_le_bytes());
                for j in 0..16 {
                    let lo = ((r * 5 + j) % 16) as u8;
                    let hi = ((b * 3 + j + 2) % 16) as u8;
                    w5[off + 6 + j] = (hi << 4) | lo;
                }
                let blk: &[u8; Q5_0_BLOCK_BYTES] = w5[off..off + Q5_0_BLOCK_BYTES]
                    .try_into()
                    .expect("q5_0 blk");
                let mut dq = [0.0_f32; 32];
                dequantize_q5_0_block(blk, &mut dq);
                ref5[r] += dq
                    .iter()
                    .zip(&x_q8_dequant[b * 32..(b + 1) * 32])
                    .map(|(&a, &c)| a * c)
                    .sum::<f32>();
            }
            let row_slice = &w5[r * blocks_per_row * Q5_0_BLOCK_BYTES
                ..(r + 1) * blocks_per_row * Q5_0_BLOCK_BYTES];
            let d1 = vec_dot_q5_0_q8_0_blocks(row_slice, &x_blocks).expect("q5_0_q8_0_blocks");
            assert!((d1 - ref5[r]).abs() < 1e-3);
        }
        let view5 =
            TensorView::from_bytes("w5", QuantType::Q5_0, &shape, 0, &w5).expect("q5_0 view");
        let mut out5 = vec![0.0_f32; rows];
        gemv_q5_0_q8_0(&mut out5, &view5, &x_blocks).expect("gemv_q5_0_q8_0");
        for r in 0..rows {
            assert!((out5[r] - ref5[r]).abs() < 1e-3);
        }
    }

    #[test]
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "bounded modular test values fit in i32"
    )]
    fn test_q8_0_x_q8_0_block_and_gemv_tails() {
        let (rows, blocks_per_row) = (5_usize, 2_usize);
        let cols = blocks_per_row * Q8_0_BLOCK_ELEMENTS;
        let shape = [cols, rows];
        let (x_blocks, x_q8_dequant) = test_q8_0_quantized_input(cols, 3);

        let mut w8 = vec![0_u8; rows * blocks_per_row * Q8_0_BLOCK_BYTES];
        let mut ref8 = vec![0.0_f32; rows];
        for r in 0..rows {
            for b in 0..blocks_per_row {
                let off = (r * blocks_per_row + b) * Q8_0_BLOCK_BYTES;
                w8[off..off + 2].copy_from_slice(&0x3000_u16.to_le_bytes()); // 0.125
                for j in 0..32 {
                    let q = i8::try_from((j as i32 * 3 + r as i32 * 7 + b as i32) % 31 - 15)
                        .unwrap_or(0);
                    w8[off + 2 + j] = q.to_ne_bytes()[0];
                }
                let blk: &[u8; Q8_0_BLOCK_BYTES] = w8[off..off + Q8_0_BLOCK_BYTES]
                    .try_into()
                    .expect("q8_0 blk");
                let mut dq = [0.0_f32; 32];
                dequantize_q8_0_block(blk, &mut dq);
                ref8[r] += dq
                    .iter()
                    .zip(&x_q8_dequant[b * 32..(b + 1) * 32])
                    .map(|(&a, &c)| a * c)
                    .sum::<f32>();
            }
            let row_slice = &w8[r * blocks_per_row * Q8_0_BLOCK_BYTES
                ..(r + 1) * blocks_per_row * Q8_0_BLOCK_BYTES];
            let d1 = vec_dot_q8_0_q8_0(row_slice, &x_blocks).expect("vec_dot_q8_0_q8_0");
            assert!((d1 - ref8[r]).abs() < 1e-3);
        }
        let view8 =
            TensorView::from_bytes("w8", QuantType::Q8_0, &shape, 0, &w8).expect("q8_0 view");
        let mut out8 = vec![0.0_f32; rows];
        gemv_q8_0_q8_0(&mut out8, &view8, &x_blocks).expect("gemv_q8_0_q8_0");
        for r in 0..rows {
            assert!((out8[r] - ref8[r]).abs() < 1e-3);
        }
    }

    #[test]
    fn test_q4_k_q5_k_x_q8_k_gemv_tails() {
        let (rows, cols) = (5_usize, 256_usize);
        let shape = [cols, rows];
        let x = test_f32_row(cols, 7);
        let mut x_blocks = [BlockQ8K::ZERO; 1];
        quantize_row_q8_k(&x, &mut x_blocks).expect("quantize q8_k");
        let x_deq = make_q8_k_input_f32(&x_blocks[0]);

        // Q4_K_M
        let mut w4k = vec![0_u8; rows * Q4_K_M_BLOCK_BYTES];
        let mut r4k = vec![0.0_f32; rows];
        for r in 0..rows {
            let blk = &mut w4k[r * Q4_K_M_BLOCK_BYTES..(r + 1) * Q4_K_M_BLOCK_BYTES];
            blk[0..2].copy_from_slice(&0x3000_u16.to_le_bytes());
            blk[2..4].copy_from_slice(&0x2C00_u16.to_le_bytes());
            for (idx, s) in blk[4..16].iter_mut().enumerate() {
                *s = u8::try_from((idx + r * 3 + 1) % 31 + 1).unwrap_or(1);
            }
            for (idx, q) in blk[16..144].iter_mut().enumerate() {
                let lo = u8::try_from((idx + r) % 16).unwrap_or(0);
                let hi = u8::try_from((idx * 3 + r + 2) % 16).unwrap_or(0);
                *q = (hi << 4) | lo;
            }
            let arr: &[u8; Q4_K_M_BLOCK_BYTES] = (&*blk).try_into().expect("q4_k");
            let mut dq = [0.0_f32; 256];
            dequantize_q4_k_m_block(arr, &mut dq);
            r4k[r] = dq.iter().zip(x_deq.iter()).map(|(&a, &b)| a * b).sum();
            let d1 = vec_dot_q4_k_q8_k(blk, &x_blocks).expect("vec_dot_q4_k_q8_k");
            assert!((d1 - r4k[r]).abs() < 1e-2);
        }
        let v4k = TensorView::from_bytes("q4k", QuantType::Q4_K_M, &shape, 0, &w4k).expect("v4k");
        let mut o4k = vec![0.0_f32; rows];
        gemv_q4_k_q8_k(&mut o4k, &v4k, &x_blocks).expect("gemv_q4_k_q8_k");
        for r in 0..rows {
            assert!((o4k[r] - r4k[r]).abs() < 1e-2);
        }

        // Q5_K
        let mut w5k = vec![0_u8; rows * Q5_K_BLOCK_BYTES];
        let mut r5k = vec![0.0_f32; rows];
        for r in 0..rows {
            let blk = &mut w5k[r * Q5_K_BLOCK_BYTES..(r + 1) * Q5_K_BLOCK_BYTES];
            blk[0..2].copy_from_slice(&0x3000_u16.to_le_bytes());
            blk[2..4].copy_from_slice(&0x2C00_u16.to_le_bytes());
            for (idx, s) in blk[4..16].iter_mut().enumerate() {
                *s = u8::try_from((idx * 2 + r + 1) % 31 + 1).unwrap_or(1);
            }
            for (idx, h) in blk[16..48].iter_mut().enumerate() {
                *h = u8::try_from((idx * 17 + r * 11) % 256).unwrap_or(0);
            }
            for (idx, q) in blk[48..176].iter_mut().enumerate() {
                let lo = u8::try_from((idx + r * 3) % 16).unwrap_or(0);
                let hi = u8::try_from((idx * 5 + r) % 16).unwrap_or(0);
                *q = (hi << 4) | lo;
            }
            let arr: &[u8; Q5_K_BLOCK_BYTES] = (&*blk).try_into().expect("q5_k");
            let mut dq = [0.0_f32; 256];
            dequantize_q5_k_block(arr, &mut dq);
            r5k[r] = dq.iter().zip(x_deq.iter()).map(|(&a, &b)| a * b).sum();
        }
        let v5k = TensorView::from_bytes("q5k", QuantType::Q5_K, &shape, 0, &w5k).expect("v5k");
        let mut o5k = vec![0.0_f32; rows];
        gemv_q5_k_q8_k(&mut o5k, &v5k, &x_blocks).expect("gemv_q5_k_q8_k");
        for r in 0..rows {
            assert!((o5k[r] - r5k[r]).abs() < 1e-2);
        }
    }

    #[test]
    fn test_q6_k_tq1_tq2_x_q8_k_gemv_tails() {
        let (rows, cols) = (5_usize, 256_usize);
        let shape = [cols, rows];
        let x = test_f32_row(cols, 7);
        let mut x_blocks = [BlockQ8K::ZERO; 1];
        quantize_row_q8_k(&x, &mut x_blocks).expect("quantize q8_k");
        let x_deq = make_q8_k_input_f32(&x_blocks[0]);

        // Q6_K
        let mut w6k = vec![0_u8; rows * Q6_K_BLOCK_BYTES];
        let mut r6k = vec![0.0_f32; rows];
        for r in 0..rows {
            let blk = &mut w6k[r * Q6_K_BLOCK_BYTES..(r + 1) * Q6_K_BLOCK_BYTES];
            blk[208..210].copy_from_slice(&0x3000_u16.to_le_bytes());
            for (idx, q) in blk[0..192].iter_mut().enumerate() {
                *q = u8::try_from((idx * 29 + r * 7) % 256).unwrap_or(0);
            }
            for (idx, sc) in blk[192..208].iter_mut().enumerate() {
                let s =
                    i8::try_from(i32::try_from((idx + r * 3) % 15).unwrap_or(0) - 7).unwrap_or(1);
                *sc = s.to_ne_bytes()[0];
            }
            let arr: &[u8; Q6_K_BLOCK_BYTES] = (&*blk).try_into().expect("q6_k");
            let mut dq = [0.0_f32; 256];
            dequantize_q6_k_block(arr, &mut dq);
            r6k[r] = dq.iter().zip(x_deq.iter()).map(|(&a, &b)| a * b).sum();
        }
        let v6k = TensorView::from_bytes("q6k", QuantType::Q6_K, &shape, 0, &w6k).expect("v6k");
        let mut o6k = vec![0.0_f32; rows];
        gemv_q6_k_q8_k(&mut o6k, &v6k, &x_blocks).expect("gemv_q6_k_q8_k");
        for r in 0..rows {
            assert!((o6k[r] - r6k[r]).abs() < 1e-2);
        }

        // TQ1_0 & TQ2_0
        let mut wt1 = vec![0_u8; rows * TQ1_0_BLOCK_BYTES];
        let mut wt2 = vec![0_u8; rows * TQ2_0_BLOCK_BYTES];
        for r in 0..rows {
            let b1 = &mut wt1[r * TQ1_0_BLOCK_BYTES..(r + 1) * TQ1_0_BLOCK_BYTES];
            b1[0..2].copy_from_slice(&0x3400_u16.to_le_bytes());
            for (idx, q) in b1[2..].iter_mut().enumerate() {
                *q = u8::try_from((idx * 19 + r * 23) % 243).unwrap_or(0);
            }
            let b2 = &mut wt2[r * TQ2_0_BLOCK_BYTES..(r + 1) * TQ2_0_BLOCK_BYTES];
            b2[0..2].copy_from_slice(&0x3400_u16.to_le_bytes());
            for (idx, q) in b2[2..].iter_mut().enumerate() {
                *q = u8::try_from((idx * 13 + r * 31) % 256).unwrap_or(0);
            }
        }
        let vt1 = TensorView::from_bytes("tq1", QuantType::TQ1_0, &shape, 0, &wt1).expect("vt1");
        let vt2 = TensorView::from_bytes("tq2", QuantType::TQ2_0, &shape, 0, &wt2).expect("vt2");
        let mut ot1 = vec![0.0_f32; rows];
        let mut ot2 = vec![0.0_f32; rows];
        gemv_tq1_0_q8_k(&mut ot1, &vt1, &x_blocks).expect("gemv_tq1_0_q8_k");
        gemv_tq2_0_q8_k(&mut ot2, &vt2, &x_blocks).expect("gemv_tq2_0_q8_k");
        for r in 0..rows {
            let s1 = vec_dot_tq1_0_q8_k(
                &wt1[r * TQ1_0_BLOCK_BYTES..(r + 1) * TQ1_0_BLOCK_BYTES],
                &x_blocks,
            )
            .expect("s1");
            let s2 = vec_dot_tq2_0_q8_k(
                &wt2[r * TQ2_0_BLOCK_BYTES..(r + 1) * TQ2_0_BLOCK_BYTES],
                &x_blocks,
            )
            .expect("s2");
            assert!((ot1[r] - s1).abs() < 1e-4);
            assert!((ot2[r] - s2).abs() < 1e-4);
        }
    }
}
