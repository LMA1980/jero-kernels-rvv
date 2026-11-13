//! Block dequantization and quantized vector dot product primitives for `Q4_0`, `Q8_0`, `Q6_K`,
//! `Q5_K`, `F16`, and `BF16`.
//!
//! Provides bit-equivalent dequantization routines matching GGUF / `llama.cpp` definitions
//! and matrix-vector dot dispatchers for both VLA RVV 1.0 and portable scalar execution.

use jero_core::{BlockQ5_K, K3Error, QuantType, TensorView};

use crate::bf16_to_f32_finite;
#[cfg(all(target_arch = "riscv64", not(miri)))]
use crate::dot_product_f32_rvv;
use crate::dot_product_f32_scalar;
pub use crate::f16_to_f32_finite;

/// Number of scalar elements in a single `Q4_0` block.
pub const Q4_0_BLOCK_ELEMENTS: usize = 32;

/// Storage size in bytes of a single `Q4_0` block (2B `f16` scale + 16B packed 4-bit nibbles).
pub const Q4_0_BLOCK_BYTES: usize = 18;

/// Number of scalar elements in a single `Q8_0` block.
pub const Q8_0_BLOCK_ELEMENTS: usize = 32;

/// Storage size in bytes of a single `Q8_0` block.
pub const Q8_0_BLOCK_BYTES: usize = 34;

/// Number of scalar elements in a single `Q6_K` super-block.
pub const Q6_K_BLOCK_ELEMENTS: usize = 256;

/// Storage size in bytes of a single `Q6_K` super-block.
pub const Q6_K_BLOCK_BYTES: usize = 210;

/// Number of scalar elements in a single `Q5_K` super-block.
pub const Q5_K_BLOCK_ELEMENTS: usize = 256;

/// Storage size in bytes of a single `Q5_K` super-block.
pub const Q5_K_BLOCK_BYTES: usize = 176;

/// Dequantizes an 18-byte `Q4_0` block (2B `f16` scale + 16 packed nibble bytes) into 32 `f32` elements
/// matching GGML `dequantize_row_q4_0` (`out[0..16]` from low nibbles, `out[16..32]` from high nibbles).
pub fn dequantize_q4_0_block(block: &[u8; Q4_0_BLOCK_BYTES], out: &mut [f32; Q4_0_BLOCK_ELEMENTS]) {
    let d_bits = u16::from_le_bytes([block[0], block[1]]);
    let d = f16_to_f32_finite(d_bits);

    if d == 0.0 {
        out.fill(0.0);
        return;
    }

    for i in 0..16 {
        let byte = block[2 + i];
        #[expect(clippy::cast_possible_wrap, reason = "4-bit nibble offset by -8")]
        let q0 = ((byte & 0x0F) as i8) - 8;
        #[expect(clippy::cast_possible_wrap, reason = "4-bit nibble offset by -8")]
        let q1 = (((byte >> 4) & 0x0F) as i8) - 8;
        out[i] = d * f32::from(q0);
        out[i + 16] = d * f32::from(q1);
    }
}

/// Dequantizes a 34-byte `Q8_0` block (2B `f16` scale + 32 signed `i8` values) into 32 `f32` elements.
pub fn dequantize_q8_0_block(block: &[u8; Q8_0_BLOCK_BYTES], out: &mut [f32; Q8_0_BLOCK_ELEMENTS]) {
    let d_bits = u16::from_le_bytes([block[0], block[1]]);
    let d = f16_to_f32_finite(d_bits);

    if d == 0.0 {
        out.fill(0.0);
        return;
    }

    #[expect(
        clippy::cast_possible_wrap,
        reason = "reinterpreting u8 quant as signed i8"
    )]
    for i in 0..Q8_0_BLOCK_ELEMENTS {
        let q = block[2 + i] as i8;
        out[i] = d * f32::from(q);
    }
}

/// Dequantizes a 210-byte `Q6_K` super-block into 256 `f32` elements.
///
/// Unpacks quants structured as 128 bytes lower-nibbles (`ql`), 64 bytes 2-bit high bits (`qh`),
/// 16 bytes 8-bit scales, and 2 bytes `f16` super-scale `d`.
pub fn dequantize_q6_k_block(block: &[u8; Q6_K_BLOCK_BYTES], out: &mut [f32; Q6_K_BLOCK_ELEMENTS]) {
    let d_bits = u16::from_le_bytes([block[208], block[209]]);
    let d = f16_to_f32_finite(d_bits);

    if d == 0.0 {
        out.fill(0.0);
        return;
    }

    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales = &block[192..208];

    // Dequantize in 2 halves of 128 elements each
    for (half_idx, &n) in [0usize, 128usize].iter().enumerate() {
        let low_nibble_base = half_idx * 64;
        let high_bits_base = half_idx * 32;
        let sc_base = half_idx * 8;

        for l in 0..32 {
            let is = l / 16;

            #[expect(clippy::cast_possible_wrap, reason = "quant arithmetic offset by -32")]
            let q1 = (((ql[low_nibble_base + l] & 0x0F) | (((qh[high_bits_base + l]) & 0x03) << 4))
                as i8)
                - 32;
            #[expect(clippy::cast_possible_wrap, reason = "quant arithmetic offset by -32")]
            let q2 = (((ql[low_nibble_base + l + 32] & 0x0F)
                | (((qh[high_bits_base + l] >> 2) & 0x03) << 4)) as i8)
                - 32;
            #[expect(clippy::cast_possible_wrap, reason = "quant arithmetic offset by -32")]
            let q3 = ((((ql[low_nibble_base + l] >> 4) & 0x0F)
                | (((qh[high_bits_base + l] >> 4) & 0x03) << 4)) as i8)
                - 32;
            #[expect(clippy::cast_possible_wrap, reason = "quant arithmetic offset by -32")]
            let q4 = ((((ql[low_nibble_base + l + 32] >> 4) & 0x0F)
                | (((qh[high_bits_base + l] >> 6) & 0x03) << 4)) as i8)
                - 32;

            #[expect(clippy::cast_possible_wrap, reason = "signed scale byte")]
            let sc1 = f32::from(scales[sc_base + is] as i8);
            #[expect(clippy::cast_possible_wrap, reason = "signed scale byte")]
            let sc2 = f32::from(scales[sc_base + is + 2] as i8);
            #[expect(clippy::cast_possible_wrap, reason = "signed scale byte")]
            let sc3 = f32::from(scales[sc_base + is + 4] as i8);
            #[expect(clippy::cast_possible_wrap, reason = "signed scale byte")]
            let sc4 = f32::from(scales[sc_base + is + 6] as i8);

            out[n + l] = d * sc1 * f32::from(q1);
            out[n + l + 32] = d * sc2 * f32::from(q2);
            out[n + l + 64] = d * sc3 * f32::from(q3);
            out[n + l + 96] = d * sc4 * f32::from(q4);
        }
    }
}

/// Dequantizes a 176-byte `Q5_K` super-block into 256 `f32` elements matching GGML `dequantize_row_q5_K`.
pub fn dequantize_q5_k_block(block: &[u8; Q5_K_BLOCK_BYTES], out: &mut [f32; Q5_K_BLOCK_ELEMENTS]) {
    let d_bits = u16::from_le_bytes([block[0], block[1]]);
    let dmin_bits = u16::from_le_bytes([block[2], block[3]]);
    let d = f16_to_f32_finite(d_bits);
    let dmin = f16_to_f32_finite(dmin_bits);

    if d == 0.0 && dmin == 0.0 {
        out.fill(0.0);
        return;
    }

    let scales = &block[4..16];
    let qh = &block[16..48];
    let qs = &block[48..176];

    let mut sc = [0u8; 8];
    let mut m = [0u8; 8];
    for j in 0..4 {
        sc[j] = scales[j] & 0x3F;
        m[j] = scales[j + 4] & 0x3F;
    }
    for j in 4..8 {
        sc[j] = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        m[j] = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
    }

    let mut out_idx = 0usize;
    let mut q_offset = 0usize;
    let mut u1 = 1u8;
    let mut u2 = 2u8;

    for chunk in 0..4 {
        let is = chunk * 2;
        let d1 = d * f32::from(sc[is]);
        let m1 = dmin * f32::from(m[is]);
        let d2 = d * f32::from(sc[is + 1]);
        let m2 = dmin * f32::from(m[is + 1]);

        for l in 0..32 {
            let high_bit = if (qh[l] & u1) != 0 { 16_u8 } else { 0_u8 };
            let q1 = (qs[q_offset + l] & 0x0F) + high_bit;
            out[out_idx + l] = d1 * f32::from(q1) - m1;
        }
        for l in 0..32 {
            let high_bit = if (qh[l] & u2) != 0 { 16_u8 } else { 0_u8 };
            let q2 = (qs[q_offset + l] >> 4) + high_bit;
            out[out_idx + 32 + l] = d2 * f32::from(q2) - m2;
        }

        out_idx += 64;
        q_offset += 32;
        u1 <<= 2;
        u2 <<= 2;
    }
}

/// Dequantizes a structured [`BlockQ5_K`] into 256 `f32` elements.
pub fn dequantize_q5_k_block_struct(block: &BlockQ5_K, out: &mut [f32; Q5_K_BLOCK_ELEMENTS]) {
    let d = f16_to_f32_finite(block.d);
    let dmin = f16_to_f32_finite(block.dmin);

    if d == 0.0 && dmin == 0.0 {
        out.fill(0.0);
        return;
    }

    let scales = &block.scales;
    let qh = &block.qh;
    let qs = &block.qs;

    let mut sc = [0u8; 8];
    let mut m = [0u8; 8];
    for j in 0..4 {
        sc[j] = scales[j] & 0x3F;
        m[j] = scales[j + 4] & 0x3F;
    }
    for j in 4..8 {
        sc[j] = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        m[j] = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
    }

    let mut out_idx = 0usize;
    let mut q_offset = 0usize;
    let mut u1 = 1u8;
    let mut u2 = 2u8;

    for chunk in 0..4 {
        let is = chunk * 2;
        let d1 = d * f32::from(sc[is]);
        let m1 = dmin * f32::from(m[is]);
        let d2 = d * f32::from(sc[is + 1]);
        let m2 = dmin * f32::from(m[is + 1]);

        for l in 0..32 {
            let high_bit = if (qh[l] & u1) != 0 { 16_u8 } else { 0_u8 };
            let q1 = (qs[q_offset + l] & 0x0F) + high_bit;
            out[out_idx + l] = d1 * f32::from(q1) - m1;
        }
        for l in 0..32 {
            let high_bit = if (qh[l] & u2) != 0 { 16_u8 } else { 0_u8 };
            let q2 = (qs[q_offset + l] >> 4) + high_bit;
            out[out_idx + 32 + l] = d2 * f32::from(q2) - m2;
        }

        out_idx += 64;
        q_offset += 32;
        u1 <<= 2;
        u2 <<= 2;
    }
}

pub(crate) fn validate_q8_0_dimensions(
    output_len: usize,
    weights: &TensorView<'_>,
    input_len: usize,
) -> Result<(usize, usize), K3Error> {
    if weights.quant() != QuantType::Q8_0 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: format!("expected Q8_0 quant type, got {:?}", weights.quant()),
        });
    }

    let shape = weights.shape();
    if shape.is_empty() {
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: "empty tensor shape".to_owned(),
        });
    }

    if shape.len() == 2 {
        let dim0 = shape[0];
        let dim1 = shape[1];
        if output_len == dim0 && input_len == dim1 && dim1 % Q8_0_BLOCK_ELEMENTS == 0 {
            return Ok((dim0, dim1));
        }
        if output_len == dim1 && input_len == dim0 && dim0 % Q8_0_BLOCK_ELEMENTS == 0 {
            return Ok((dim1, dim0));
        }
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: format!(
                "matrix shape [{dim0}, {dim1}] does not match output len {output_len} and Q8_0-aligned input len {input_len}"
            ),
        });
    }

    let total_elems = weights.num_elements();
    if input_len == 0
        || output_len == 0
        || input_len % Q8_0_BLOCK_ELEMENTS != 0
        || output_len * input_len != total_elems
    {
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: format!(
                "tensor elements {total_elems} do not match output len {output_len} * input len {input_len}"
            ),
        });
    }

    Ok((output_len, input_len))
}

pub(crate) fn validate_q6_k_dimensions(
    output_len: usize,
    weights: &TensorView<'_>,
    input_len: usize,
) -> Result<(usize, usize), K3Error> {
    if weights.quant() != QuantType::Q6_K {
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: format!("expected Q6_K quant type, got {:?}", weights.quant()),
        });
    }

    let shape = weights.shape();
    if shape.is_empty() {
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: "empty tensor shape".to_owned(),
        });
    }

    if shape.len() == 2 {
        let dim0 = shape[0];
        let dim1 = shape[1];
        if output_len == dim0 && input_len == dim1 && dim1 % Q6_K_BLOCK_ELEMENTS == 0 {
            return Ok((dim0, dim1));
        }
        if output_len == dim1 && input_len == dim0 && dim0 % Q6_K_BLOCK_ELEMENTS == 0 {
            return Ok((dim1, dim0));
        }
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: format!(
                "matrix shape [{dim0}, {dim1}] does not match output len {output_len} and Q6_K-aligned input len {input_len}"
            ),
        });
    }

    let total_elems = weights.num_elements();
    if input_len == 0
        || output_len == 0
        || input_len % Q6_K_BLOCK_ELEMENTS != 0
        || output_len * input_len != total_elems
    {
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: format!(
                "tensor elements {total_elems} do not match output len {output_len} * input len {input_len}"
            ),
        });
    }

    Ok((output_len, input_len))
}

/// Computes matrix-vector multiplication with `Q8_0` quantized weights using portable scalar arithmetic.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if quantization is not `Q8_0` or dimensions mismatch.
pub fn vec_dot_q8_0_scalar(
    output: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
) -> Result<(), K3Error> {
    let (rows, cols) = validate_q8_0_dimensions(output.len(), weights, input.len())?;
    let blocks_per_row = cols / Q8_0_BLOCK_ELEMENTS;
    let raw_bytes = weights.as_bytes();
    let mut dequant_buf = [0.0_f32; Q8_0_BLOCK_ELEMENTS];

    for (row, out_slot) in output.iter_mut().enumerate().take(rows) {
        let mut row_sum = 0.0_f32;
        for blk_idx in 0..blocks_per_row {
            let block_index = row * blocks_per_row + blk_idx;
            let byte_start = block_index * Q8_0_BLOCK_BYTES;
            let byte_end = byte_start + Q8_0_BLOCK_BYTES;
            let block_slice: &[u8; Q8_0_BLOCK_BYTES] = raw_bytes[byte_start..byte_end]
                .try_into()
                .map_err(|_| K3Error::TensorBoundsError {
                tensor_name: weights.name().to_owned(),
                offset: byte_start,
                byte_len: Q8_0_BLOCK_BYTES,
                buffer_len: raw_bytes.len(),
                reason: "failed to slice Q8_0 block".to_owned(),
            })?;

            dequantize_q8_0_block(block_slice, &mut dequant_buf);
            let input_sub =
                &input[blk_idx * Q8_0_BLOCK_ELEMENTS..(blk_idx + 1) * Q8_0_BLOCK_ELEMENTS];
            row_sum += dot_product_f32_scalar(&dequant_buf, input_sub);
        }
        *out_slot = row_sum;
    }

    Ok(())
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[inline]
#[allow(unused_assignments)]
fn dot_i8_32_f32_rvv(quants: &[u8], input: &[f32]) -> f32 {
    debug_assert!(quants.len() >= Q8_0_BLOCK_ELEMENTS);
    debug_assert!(input.len() >= Q8_0_BLOCK_ELEMENTS);
    let mut pq = quants.as_ptr();
    let mut px = input.as_ptr();
    let mut rem = Q8_0_BLOCK_ELEMENTS;
    let zero_f = 0.0_f32;
    let sum: f32;

    // SAFETY: `pq` points to 32 valid signed bytes and `px` points to at least 32 `f32`s.
    // Strip-mined `e32, m4` (`vle8.v` -> `vsext.vf4` -> `vfcvt.f.x.v` -> `vfmul.vv` -> `vfredusum.vs`)
    // executes in 1 vector iteration on both `zvl256b` (`X100`) and `zvl1024b` (`A100`).
    unsafe {
        core::arch::asm!(
            "vsetivli zero, 1, e32, m1, ta, ma",
            "vfmv.s.f v0, {z}",
            "2:",
            "vsetvli t0, {rem}, e32, m4, ta, ma",
            "vle8.v v8, ({pq})",
            "vsext.vf4 v12, v8",
            "vfcvt.f.x.v v12, v12",
            "vle32.v v16, ({px})",
            "vfmul.vv v12, v12, v16",
            "vfredusum.vs v0, v12, v0",
            "add {pq}, {pq}, t0",
            "slli t1, t0, 2",
            "add {px}, {px}, t1",
            "sub {rem}, {rem}, t0",
            "bnez {rem}, 2b",
            "vfmv.f.s {sum}, v0",
            pq = inout(reg) pq,
            px = inout(reg) px,
            rem = inout(reg) rem,
            z = in(freg) zero_f,
            sum = out(freg) sum,
            out("t0") _,
            out("t1") _,
            out("v0") _,
            out("v8") _,
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
    sum
}

/// Computes matrix-vector multiplication with `Q8_0` quantized weights using RVV vector dot product.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if quantization is not `Q8_0` or dimensions mismatch.
#[cfg(all(target_arch = "riscv64", not(miri)))]
pub fn vec_dot_q8_0_rvv(
    output: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
) -> Result<(), K3Error> {
    let (rows, cols) = validate_q8_0_dimensions(output.len(), weights, input.len())?;
    let blocks_per_row = cols / Q8_0_BLOCK_ELEMENTS;
    let raw_bytes = weights.as_bytes();

    for (row, out_slot) in output.iter_mut().enumerate().take(rows) {
        let mut row_sum = 0.0_f32;
        for blk_idx in 0..blocks_per_row {
            let block_index = row * blocks_per_row + blk_idx;
            let byte_start = block_index * Q8_0_BLOCK_BYTES;
            let byte_end = byte_start + Q8_0_BLOCK_BYTES;
            let block_slice: &[u8; Q8_0_BLOCK_BYTES] = raw_bytes
                .get(byte_start..byte_end)
                .and_then(|s| s.try_into().ok())
                .ok_or_else(|| K3Error::TensorBoundsError {
                    tensor_name: weights.name().to_owned(),
                    offset: byte_start,
                    byte_len: Q8_0_BLOCK_BYTES,
                    buffer_len: raw_bytes.len(),
                    reason: "failed to slice Q8_0 block".to_owned(),
                })?;

            let d = f16_to_f32_finite(u16::from_le_bytes([block_slice[0], block_slice[1]]));
            if d != 0.0 {
                let input_sub =
                    &input[blk_idx * Q8_0_BLOCK_ELEMENTS..(blk_idx + 1) * Q8_0_BLOCK_ELEMENTS];
                row_sum += d * dot_i8_32_f32_rvv(&block_slice[2..34], input_sub);
            }
        }
        *out_slot = row_sum;
    }

    Ok(())
}

/// Computes matrix-vector multiplication with `Q6_K` quantized weights using portable scalar arithmetic.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if quantization is not `Q6_K` or dimensions mismatch.
pub fn vec_dot_q6_k_scalar(
    output: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
) -> Result<(), K3Error> {
    let (rows, cols) = validate_q6_k_dimensions(output.len(), weights, input.len())?;
    let blocks_per_row = cols / Q6_K_BLOCK_ELEMENTS;
    let raw_bytes = weights.as_bytes();
    let mut dequant_buf = [0.0_f32; Q6_K_BLOCK_ELEMENTS];

    for (row, out_slot) in output.iter_mut().enumerate().take(rows) {
        let mut row_sum = 0.0_f32;
        for blk_idx in 0..blocks_per_row {
            let block_index = row * blocks_per_row + blk_idx;
            let byte_start = block_index * Q6_K_BLOCK_BYTES;
            let byte_end = byte_start + Q6_K_BLOCK_BYTES;
            let block_slice: &[u8; Q6_K_BLOCK_BYTES] = raw_bytes[byte_start..byte_end]
                .try_into()
                .map_err(|_| K3Error::TensorBoundsError {
                tensor_name: weights.name().to_owned(),
                offset: byte_start,
                byte_len: Q6_K_BLOCK_BYTES,
                buffer_len: raw_bytes.len(),
                reason: "failed to slice Q6_K block".to_owned(),
            })?;

            dequantize_q6_k_block(block_slice, &mut dequant_buf);
            let input_sub =
                &input[blk_idx * Q6_K_BLOCK_ELEMENTS..(blk_idx + 1) * Q6_K_BLOCK_ELEMENTS];
            row_sum += dot_product_f32_scalar(&dequant_buf, input_sub);
        }
        *out_slot = row_sum;
    }

    Ok(())
}

/// Computes matrix-vector multiplication with `Q6_K` quantized weights using RVV vector dot product.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if quantization is not `Q6_K` or dimensions mismatch.
#[cfg(all(target_arch = "riscv64", not(miri)))]
pub fn vec_dot_q6_k_rvv(
    output: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
) -> Result<(), K3Error> {
    let (rows, cols) = validate_q6_k_dimensions(output.len(), weights, input.len())?;
    let blocks_per_row = cols / Q6_K_BLOCK_ELEMENTS;
    let raw_bytes = weights.as_bytes();
    let mut dequant_buf = [0.0_f32; Q6_K_BLOCK_ELEMENTS];

    for (row, out_slot) in output.iter_mut().enumerate().take(rows) {
        let mut row_sum = 0.0_f32;
        for blk_idx in 0..blocks_per_row {
            let block_index = row * blocks_per_row + blk_idx;
            let byte_start = block_index * Q6_K_BLOCK_BYTES;
            let byte_end = byte_start + Q6_K_BLOCK_BYTES;
            let block_slice: &[u8; Q6_K_BLOCK_BYTES] = raw_bytes[byte_start..byte_end]
                .try_into()
                .map_err(|_| K3Error::TensorBoundsError {
                tensor_name: weights.name().to_owned(),
                offset: byte_start,
                byte_len: Q6_K_BLOCK_BYTES,
                buffer_len: raw_bytes.len(),
                reason: "failed to slice Q6_K block".to_owned(),
            })?;

            dequantize_q6_k_block(block_slice, &mut dequant_buf);
            let input_sub =
                &input[blk_idx * Q6_K_BLOCK_ELEMENTS..(blk_idx + 1) * Q6_K_BLOCK_ELEMENTS];
            row_sum += dot_product_f32_rvv(&dequant_buf, input_sub);
        }
        *out_slot = row_sum;
    }

    Ok(())
}

/// Computes matrix-vector multiplication with `Q8_0` quantized weights.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if quantization is not `Q8_0` or dimensions mismatch.
pub fn vec_dot_q8_0(
    output: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
) -> Result<(), K3Error> {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        vec_dot_q8_0_rvv(output, weights, input)
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        vec_dot_q8_0_scalar(output, weights, input)
    }
}

/// Computes matrix-vector multiplication with `Q6_K` quantized weights.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if quantization is not `Q6_K` or dimensions mismatch.
pub fn vec_dot_q6_k(
    output: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
) -> Result<(), K3Error> {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        vec_dot_q6_k_rvv(output, weights, input)
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        vec_dot_q6_k_scalar(output, weights, input)
    }
}

/// Validates matrix dimensions and tensor bounds for `Q5_K` matrix-vector operations.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if quantization is not `Q5_K`, dimensions do not match,
/// or columns are not divisible by 256.
pub fn validate_q5_k_dimensions(
    output_len: usize,
    weights: &TensorView<'_>,
    input_len: usize,
) -> Result<(usize, usize), K3Error> {
    if weights.quant() != QuantType::Q5_K {
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: format!("expected Q5_K quant type, got {:?}", weights.quant()),
        });
    }

    let shape = weights.shape();
    if shape.is_empty() {
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: "empty tensor shape".to_owned(),
        });
    }

    if shape.len() == 2 {
        let dim0 = shape[0];
        let dim1 = shape[1];
        if output_len == dim0 && input_len == dim1 && dim1 % Q5_K_BLOCK_ELEMENTS == 0 {
            return Ok((dim0, dim1));
        }
        if output_len == dim1 && input_len == dim0 && dim0 % Q5_K_BLOCK_ELEMENTS == 0 {
            return Ok((dim1, dim0));
        }
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: format!(
                "matrix shape [{dim0}, {dim1}] does not match output len {output_len} and Q5_K-aligned input len {input_len}"
            ),
        });
    }

    let total_elems = weights.num_elements();
    if input_len == 0
        || output_len == 0
        || input_len % Q5_K_BLOCK_ELEMENTS != 0
        || output_len * input_len != total_elems
    {
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: format!(
                "tensor elements {total_elems} do not match output len {output_len} * input len {input_len}"
            ),
        });
    }

    Ok((output_len, input_len))
}

/// Computes matrix-vector multiplication with `Q5_K` quantized weights using portable scalar arithmetic.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if quantization is not `Q5_K` or dimensions mismatch.
pub fn vec_dot_q5_k_scalar(
    output: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
) -> Result<(), K3Error> {
    let (rows, cols) = validate_q5_k_dimensions(output.len(), weights, input.len())?;
    let blocks_per_row = cols / Q5_K_BLOCK_ELEMENTS;
    let raw_bytes = weights.as_bytes();
    let mut dequant_buf = [0.0_f32; Q5_K_BLOCK_ELEMENTS];

    for (row, out_slot) in output.iter_mut().enumerate().take(rows) {
        let mut row_sum = 0.0_f32;
        for blk_idx in 0..blocks_per_row {
            let block_index = row * blocks_per_row + blk_idx;
            let byte_start = block_index * Q5_K_BLOCK_BYTES;
            let byte_end = byte_start + Q5_K_BLOCK_BYTES;
            let block_slice: &[u8; Q5_K_BLOCK_BYTES] = raw_bytes[byte_start..byte_end]
                .try_into()
                .map_err(|_| K3Error::TensorBoundsError {
                tensor_name: weights.name().to_owned(),
                offset: byte_start,
                byte_len: Q5_K_BLOCK_BYTES,
                buffer_len: raw_bytes.len(),
                reason: "failed to slice Q5_K block".to_owned(),
            })?;

            dequantize_q5_k_block(block_slice, &mut dequant_buf);
            let input_sub =
                &input[blk_idx * Q5_K_BLOCK_ELEMENTS..(blk_idx + 1) * Q5_K_BLOCK_ELEMENTS];
            row_sum += dot_product_f32_scalar(&dequant_buf, input_sub);
        }
        *out_slot = row_sum;
    }

    Ok(())
}

/// Computes matrix-vector multiplication with `Q5_K` quantized weights using RVV vector dot product.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if quantization is not `Q5_K` or dimensions mismatch.
#[cfg(all(target_arch = "riscv64", not(miri)))]
pub fn vec_dot_q5_k_rvv(
    output: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
) -> Result<(), K3Error> {
    let (rows, cols) = validate_q5_k_dimensions(output.len(), weights, input.len())?;
    let blocks_per_row = cols / Q5_K_BLOCK_ELEMENTS;
    let raw_bytes = weights.as_bytes();
    let mut dequant_buf = [0.0_f32; Q5_K_BLOCK_ELEMENTS];

    for (row, out_slot) in output.iter_mut().enumerate().take(rows) {
        let mut row_sum = 0.0_f32;
        for blk_idx in 0..blocks_per_row {
            let block_index = row * blocks_per_row + blk_idx;
            let byte_start = block_index * Q5_K_BLOCK_BYTES;
            let byte_end = byte_start + Q5_K_BLOCK_BYTES;
            let block_slice: &[u8; Q5_K_BLOCK_BYTES] = raw_bytes[byte_start..byte_end]
                .try_into()
                .map_err(|_| K3Error::TensorBoundsError {
                tensor_name: weights.name().to_owned(),
                offset: byte_start,
                byte_len: Q5_K_BLOCK_BYTES,
                buffer_len: raw_bytes.len(),
                reason: "failed to slice Q5_K block".to_owned(),
            })?;

            dequantize_q5_k_block(block_slice, &mut dequant_buf);
            let input_sub =
                &input[blk_idx * Q5_K_BLOCK_ELEMENTS..(blk_idx + 1) * Q5_K_BLOCK_ELEMENTS];
            row_sum += dot_product_f32_rvv(&dequant_buf, input_sub);
        }
        *out_slot = row_sum;
    }

    Ok(())
}

/// Computes matrix-vector multiplication with `Q5_K` quantized weights.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if quantization is not `Q5_K` or dimensions mismatch.
pub fn vec_dot_q5_k(
    output: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
) -> Result<(), K3Error> {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        vec_dot_q5_k_rvv(output, weights, input)
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        vec_dot_q5_k_scalar(output, weights, input)
    }
}

/// Computes matrix-vector multiplication with `Q5_K` quantized weights (`y = W * x`).
///
/// Convenience entry point matching Issue #40 signature.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if dimensions do not align or weights are malformed.
pub fn matvec_q5_k(
    weights: &TensorView<'_>,
    input: &[f32],
    output: &mut [f32],
) -> Result<(), K3Error> {
    vec_dot_q5_k(output, weights, input)
}

/// Computes matrix-vector multiplication for a slice of [`BlockQ5_K`] structs.
///
/// # Panics
///
/// Panics if `cols % 256 != 0`, `weights.len() != rows * (cols / 256)`,
/// `x.len() < cols`, or `out.len() < rows`.
pub fn matvec_q5_k_slice(
    weights: &[BlockQ5_K],
    x: &[f32],
    out: &mut [f32],
    rows: usize,
    cols: usize,
) {
    assert!(
        cols % Q5_K_BLOCK_ELEMENTS == 0,
        "cols must be a multiple of 256"
    );
    let blocks_per_row = cols / Q5_K_BLOCK_ELEMENTS;
    assert_eq!(
        weights.len(),
        rows * blocks_per_row,
        "weight block count mismatch"
    );
    assert!(x.len() >= cols, "input slice too short");
    assert!(out.len() >= rows, "output slice too short");

    let mut dequant_buf = [0.0_f32; Q5_K_BLOCK_ELEMENTS];

    for (row, out_slot) in out.iter_mut().enumerate().take(rows) {
        let mut row_sum = 0.0_f32;
        for blk_idx in 0..blocks_per_row {
            let block = &weights[row * blocks_per_row + blk_idx];
            dequantize_q5_k_block_struct(block, &mut dequant_buf);
            let input_sub = &x[blk_idx * Q5_K_BLOCK_ELEMENTS..(blk_idx + 1) * Q5_K_BLOCK_ELEMENTS];
            #[cfg(all(target_arch = "riscv64", not(miri)))]
            {
                row_sum += dot_product_f32_rvv(&dequant_buf, input_sub);
            }
            #[cfg(any(not(target_arch = "riscv64"), miri))]
            {
                row_sum += dot_product_f32_scalar(&dequant_buf, input_sub);
            }
        }
        *out_slot = row_sum;
    }
}

pub(crate) fn validate_q4_0_dimensions(
    output_len: usize,
    weights: &TensorView<'_>,
    input_len: usize,
) -> Result<(usize, usize), K3Error> {
    if weights.quant() != QuantType::Q4_0 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: format!("expected Q4_0 quant type, got {:?}", weights.quant()),
        });
    }

    let shape = weights.shape();
    if shape.is_empty() {
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: "empty tensor shape".to_owned(),
        });
    }

    let (rows, cols) = if shape.len() == 2 {
        let dim0 = shape[0];
        let dim1 = shape[1];
        if output_len == dim0 && input_len == dim1 && dim1 % Q4_0_BLOCK_ELEMENTS == 0 {
            (dim0, dim1)
        } else if output_len == dim1 && input_len == dim0 && dim0 % Q4_0_BLOCK_ELEMENTS == 0 {
            (dim1, dim0)
        } else {
            return Err(K3Error::TensorBoundsError {
                tensor_name: weights.name().to_owned(),
                offset: weights.offset(),
                byte_len: weights.byte_len(),
                buffer_len: weights.as_bytes().len(),
                reason: format!(
                    "matrix shape [{dim0}, {dim1}] does not match output len {output_len} and Q4_0-aligned input len {input_len}"
                ),
            });
        }
    } else {
        let total_elems = weights.num_elements();
        if input_len == 0
            || output_len == 0
            || input_len % Q4_0_BLOCK_ELEMENTS != 0
            || output_len * input_len != total_elems
        {
            return Err(K3Error::TensorBoundsError {
                tensor_name: weights.name().to_owned(),
                offset: weights.offset(),
                byte_len: weights.byte_len(),
                buffer_len: weights.as_bytes().len(),
                reason: format!(
                    "tensor elements {total_elems} do not match output len {output_len} * input len {input_len}"
                ),
            });
        }
        (output_len, input_len)
    };

    let required_bytes = (rows * cols / Q4_0_BLOCK_ELEMENTS) * Q4_0_BLOCK_BYTES;
    if weights.as_bytes().len() < required_bytes {
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: format!(
                "Q4_0 buffer length {} < required {required_bytes}",
                weights.as_bytes().len()
            ),
        });
    }

    Ok((rows, cols))
}

/// Computes matrix-vector multiplication with `Q4_0` quantized weights using portable scalar arithmetic.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if quantization is not `Q4_0` or dimensions mismatch.
pub fn vec_dot_q4_0_scalar(
    output: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
) -> Result<(), K3Error> {
    let (rows, cols) = validate_q4_0_dimensions(output.len(), weights, input.len())?;
    let blocks_per_row = cols / Q4_0_BLOCK_ELEMENTS;
    let raw_bytes = weights.as_bytes();
    let mut dequant_buf = [0.0_f32; Q4_0_BLOCK_ELEMENTS];

    for (row, out_slot) in output.iter_mut().enumerate().take(rows) {
        let mut row_sum = 0.0_f32;
        for blk_idx in 0..blocks_per_row {
            let block_index = row * blocks_per_row + blk_idx;
            let byte_start = block_index * Q4_0_BLOCK_BYTES;
            let byte_end = byte_start + Q4_0_BLOCK_BYTES;
            let block_slice: &[u8; Q4_0_BLOCK_BYTES] = raw_bytes[byte_start..byte_end]
                .try_into()
                .map_err(|_| K3Error::TensorBoundsError {
                tensor_name: weights.name().to_owned(),
                offset: byte_start,
                byte_len: Q4_0_BLOCK_BYTES,
                buffer_len: raw_bytes.len(),
                reason: "failed to slice Q4_0 block".to_owned(),
            })?;

            dequantize_q4_0_block(block_slice, &mut dequant_buf);
            let input_sub =
                &input[blk_idx * Q4_0_BLOCK_ELEMENTS..(blk_idx + 1) * Q4_0_BLOCK_ELEMENTS];
            row_sum += dot_product_f32_scalar(&dequant_buf, input_sub);
        }
        *out_slot = row_sum;
    }

    Ok(())
}

/// Computes matrix-vector multiplication with `Q4_0` quantized weights using RVV vector dot product.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if quantization is not `Q4_0` or dimensions mismatch.
#[cfg(all(target_arch = "riscv64", not(miri)))]
pub fn vec_dot_q4_0_rvv(
    output: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
) -> Result<(), K3Error> {
    let (rows, cols) = validate_q4_0_dimensions(output.len(), weights, input.len())?;
    let blocks_per_row = cols / Q4_0_BLOCK_ELEMENTS;
    let raw_bytes = weights.as_bytes();
    let mut unpacked_i8 = [0_u8; Q4_0_BLOCK_ELEMENTS];

    for (row, out_slot) in output.iter_mut().enumerate().take(rows) {
        let mut row_sum = 0.0_f32;
        for blk_idx in 0..blocks_per_row {
            let block_index = row * blocks_per_row + blk_idx;
            let byte_start = block_index * Q4_0_BLOCK_BYTES;
            let byte_end = byte_start + Q4_0_BLOCK_BYTES;
            let block_slice: &[u8; Q4_0_BLOCK_BYTES] = raw_bytes[byte_start..byte_end]
                .try_into()
                .map_err(|_| K3Error::TensorBoundsError {
                tensor_name: weights.name().to_owned(),
                offset: byte_start,
                byte_len: Q4_0_BLOCK_BYTES,
                buffer_len: raw_bytes.len(),
                reason: "failed to slice Q4_0 block".to_owned(),
            })?;

            let d = f16_to_f32_finite(u16::from_le_bytes([block_slice[0], block_slice[1]]));
            if d != 0.0 {
                for i in 0..16 {
                    let byte = block_slice[2 + i];
                    unpacked_i8[i] = (byte & 0x0F).wrapping_sub(8);
                    unpacked_i8[i + 16] = ((byte >> 4) & 0x0F).wrapping_sub(8);
                }
                let input_sub =
                    &input[blk_idx * Q4_0_BLOCK_ELEMENTS..(blk_idx + 1) * Q4_0_BLOCK_ELEMENTS];
                row_sum += d * dot_i8_32_f32_rvv(&unpacked_i8, input_sub);
            }
        }
        *out_slot = row_sum;
    }

    Ok(())
}

/// Computes matrix-vector multiplication with `Q4_0` quantized weights.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if quantization is not `Q4_0` or dimensions mismatch.
pub fn vec_dot_q4_0(
    output: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
) -> Result<(), K3Error> {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        vec_dot_q4_0_rvv(output, weights, input)
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        vec_dot_q4_0_scalar(output, weights, input)
    }
}

fn validate_half_dimensions(
    expected_quant: QuantType,
    output_len: usize,
    weights: &TensorView<'_>,
    input_len: usize,
) -> Result<(usize, usize), K3Error> {
    if weights.quant() != expected_quant {
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: format!(
                "expected {expected_quant:?} quant type, got {:?}",
                weights.quant()
            ),
        });
    }

    let shape = weights.shape();
    if shape.is_empty() {
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: "empty tensor shape".to_owned(),
        });
    }

    let (rows, cols) = if shape.len() == 2 {
        let dim0 = shape[0];
        let dim1 = shape[1];
        if output_len == dim0 && input_len == dim1 {
            (dim0, dim1)
        } else if output_len == dim1 && input_len == dim0 {
            (dim1, dim0)
        } else {
            return Err(K3Error::TensorBoundsError {
                tensor_name: weights.name().to_owned(),
                offset: weights.offset(),
                byte_len: weights.byte_len(),
                buffer_len: weights.as_bytes().len(),
                reason: format!(
                    "matrix shape [{dim0}, {dim1}] does not match output len {output_len} and input len {input_len}"
                ),
            });
        }
    } else {
        let total_elems = weights.num_elements();
        if input_len == 0 || output_len == 0 || output_len * input_len != total_elems {
            return Err(K3Error::TensorBoundsError {
                tensor_name: weights.name().to_owned(),
                offset: weights.offset(),
                byte_len: weights.byte_len(),
                buffer_len: weights.as_bytes().len(),
                reason: format!(
                    "tensor elements {total_elems} do not match output len {output_len} * input len {input_len}"
                ),
            });
        }
        (output_len, input_len)
    };

    let required_bytes = rows
        .checked_mul(cols)
        .and_then(|n| n.checked_mul(2))
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: "half-precision byte length overflow".to_owned(),
        })?;
    if weights.as_bytes().len() < required_bytes {
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: format!(
                "{expected_quant:?} buffer length {} < required {required_bytes}",
                weights.as_bytes().len()
            ),
        });
    }

    Ok((rows, cols))
}

/// Computes matrix-vector multiplication with `F16` weights using portable scalar arithmetic.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if quantization is not `F16` or dimensions mismatch.
pub fn vec_dot_f16_scalar(
    output: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
) -> Result<(), K3Error> {
    let (rows, cols) =
        validate_half_dimensions(QuantType::F16, output.len(), weights, input.len())?;
    let raw_bytes = weights.as_bytes();
    let stride_bytes = cols * 2;
    let mut chunk_buf = [0.0_f32; 256];

    for (row, out_slot) in output.iter_mut().enumerate().take(rows) {
        let row_slice = &raw_bytes[row * stride_bytes..(row + 1) * stride_bytes];
        let mut row_sum = 0.0_f32;
        let mut col = 0usize;
        while col < cols {
            let n = (cols - col).min(chunk_buf.len());
            for (i, slot) in chunk_buf.iter_mut().enumerate().take(n) {
                let off = (col + i) * 2;
                let bits = u16::from_le_bytes([row_slice[off], row_slice[off + 1]]);
                *slot = f16_to_f32_finite(bits);
            }
            row_sum += dot_product_f32_scalar(&chunk_buf[..n], &input[col..col + n]);
            col += n;
        }
        *out_slot = row_sum;
    }

    Ok(())
}

/// Computes matrix-vector multiplication with `F16` weights using VLA RVV 1.0 dot product.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if quantization is not `F16` or dimensions mismatch.
#[cfg(all(target_arch = "riscv64", not(miri)))]
pub fn vec_dot_f16_rvv(
    output: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
) -> Result<(), K3Error> {
    let (rows, cols) =
        validate_half_dimensions(QuantType::F16, output.len(), weights, input.len())?;
    let raw_bytes = weights.as_bytes();
    let stride_bytes = cols * 2;
    let mut chunk_buf = [0.0_f32; 256];

    for (row, out_slot) in output.iter_mut().enumerate().take(rows) {
        let row_slice = &raw_bytes[row * stride_bytes..(row + 1) * stride_bytes];
        let mut row_sum = 0.0_f32;
        let mut col = 0usize;
        while col < cols {
            let n = (cols - col).min(chunk_buf.len());
            for (i, slot) in chunk_buf.iter_mut().enumerate().take(n) {
                let off = (col + i) * 2;
                let bits = u16::from_le_bytes([row_slice[off], row_slice[off + 1]]);
                *slot = f16_to_f32_finite(bits);
            }
            row_sum += dot_product_f32_rvv(&chunk_buf[..n], &input[col..col + n]);
            col += n;
        }
        *out_slot = row_sum;
    }

    Ok(())
}

/// Computes matrix-vector multiplication with `F16` half-precision weights.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if quantization is not `F16` or dimensions mismatch.
pub fn vec_dot_f16(
    output: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
) -> Result<(), K3Error> {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        vec_dot_f16_rvv(output, weights, input)
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        vec_dot_f16_scalar(output, weights, input)
    }
}

/// Computes matrix-vector multiplication with `BF16` weights using portable scalar arithmetic.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if quantization is not `BF16` or dimensions mismatch.
pub fn vec_dot_bf16_scalar(
    output: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
) -> Result<(), K3Error> {
    let (rows, cols) =
        validate_half_dimensions(QuantType::BF16, output.len(), weights, input.len())?;
    let raw_bytes = weights.as_bytes();
    let stride_bytes = cols * 2;
    let mut chunk_buf = [0.0_f32; 256];

    for (row, out_slot) in output.iter_mut().enumerate().take(rows) {
        let row_slice = &raw_bytes[row * stride_bytes..(row + 1) * stride_bytes];
        let mut row_sum = 0.0_f32;
        let mut col = 0usize;
        while col < cols {
            let n = (cols - col).min(chunk_buf.len());
            for (i, slot) in chunk_buf.iter_mut().enumerate().take(n) {
                let off = (col + i) * 2;
                let bits = u16::from_le_bytes([row_slice[off], row_slice[off + 1]]);
                *slot = bf16_to_f32_finite(bits);
            }
            row_sum += dot_product_f32_scalar(&chunk_buf[..n], &input[col..col + n]);
            col += n;
        }
        *out_slot = row_sum;
    }

    Ok(())
}

/// Computes matrix-vector multiplication with `BF16` weights using VLA RVV 1.0 dot product.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if quantization is not `BF16` or dimensions mismatch.
#[cfg(all(target_arch = "riscv64", not(miri)))]
pub fn vec_dot_bf16_rvv(
    output: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
) -> Result<(), K3Error> {
    let (rows, cols) =
        validate_half_dimensions(QuantType::BF16, output.len(), weights, input.len())?;
    let raw_bytes = weights.as_bytes();
    let stride_bytes = cols * 2;
    let mut chunk_buf = [0.0_f32; 256];

    for (row, out_slot) in output.iter_mut().enumerate().take(rows) {
        let row_slice = &raw_bytes[row * stride_bytes..(row + 1) * stride_bytes];
        let mut row_sum = 0.0_f32;
        let mut col = 0usize;
        while col < cols {
            let n = (cols - col).min(chunk_buf.len());
            for (i, slot) in chunk_buf.iter_mut().enumerate().take(n) {
                let off = (col + i) * 2;
                let bits = u16::from_le_bytes([row_slice[off], row_slice[off + 1]]);
                *slot = bf16_to_f32_finite(bits);
            }
            row_sum += dot_product_f32_rvv(&chunk_buf[..n], &input[col..col + n]);
            col += n;
        }
        *out_slot = row_sum;
    }

    Ok(())
}

/// Computes matrix-vector multiplication with `BF16` brain floating-point weights.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if quantization is not `BF16` or dimensions mismatch.
pub fn vec_dot_bf16(
    output: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
) -> Result<(), K3Error> {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        vec_dot_bf16_rvv(output, weights, input)
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        vec_dot_bf16_scalar(output, weights, input)
    }
}

fn validate_token_row_geometry(
    table: &TensorView<'_>,
    row_idx: usize,
    cols: usize,
) -> Result<usize, K3Error> {
    let raw = table.as_bytes();
    if cols == 0 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: table.name().to_owned(),
            offset: table.offset(),
            byte_len: table.byte_len(),
            buffer_len: raw.len(),
            reason: "dequantize_token_row output slice cannot be empty".to_owned(),
        });
    }

    let total_elems = table.num_elements();
    if total_elems == 0 || total_elems % cols != 0 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: table.name().to_owned(),
            offset: table.offset(),
            byte_len: table.byte_len(),
            buffer_len: raw.len(),
            reason: format!(
                "tensor element count {total_elems} is not a positive multiple of row width {cols}"
            ),
        });
    }

    let num_rows = total_elems / cols;
    let shape = table.shape();
    if shape.len() == 2 {
        let matches_orientation = (shape[0] == cols && shape[1] == num_rows)
            || (shape[1] == cols && shape[0] == num_rows);
        if !matches_orientation {
            return Err(K3Error::TensorBoundsError {
                tensor_name: table.name().to_owned(),
                offset: table.offset(),
                byte_len: table.byte_len(),
                buffer_len: raw.len(),
                reason: format!(
                    "tensor shape [{}, {}] does not match row width {cols} and row count {num_rows}",
                    shape[0], shape[1]
                ),
            });
        }
    }

    if row_idx >= num_rows {
        return Err(K3Error::TensorBoundsError {
            tensor_name: table.name().to_owned(),
            offset: row_idx,
            byte_len: table.byte_len(),
            buffer_len: raw.len(),
            reason: format!("row index {row_idx} out of bounds for {num_rows} rows"),
        });
    }

    Ok(num_rows)
}

fn dequantize_block_row<const BLOCK_ELEMS: usize, const BLOCK_BYTES: usize>(
    table: &TensorView<'_>,
    row_idx: usize,
    out: &mut [f32],
    decode_block: fn(&[u8; BLOCK_BYTES], &mut [f32; BLOCK_ELEMS]),
) -> Result<(), K3Error> {
    let cols = out.len();
    let raw = table.as_bytes();
    if cols % BLOCK_ELEMS != 0 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: table.name().to_owned(),
            offset: table.offset(),
            byte_len: table.byte_len(),
            buffer_len: raw.len(),
            reason: format!(
                "row width {cols} is not aligned to {:?} block size {BLOCK_ELEMS}",
                table.quant()
            ),
        });
    }

    let blocks_per_row = cols / BLOCK_ELEMS;
    let row_bytes =
        blocks_per_row
            .checked_mul(BLOCK_BYTES)
            .ok_or_else(|| K3Error::TensorBoundsError {
                tensor_name: table.name().to_owned(),
                offset: table.offset(),
                byte_len: table.byte_len(),
                buffer_len: raw.len(),
                reason: "row byte stride overflowed usize".to_owned(),
            })?;
    let row_start = row_idx
        .checked_mul(row_bytes)
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: table.name().to_owned(),
            offset: table.offset(),
            byte_len: table.byte_len(),
            buffer_len: raw.len(),
            reason: "row byte offset overflowed usize".to_owned(),
        })?;
    let row_end = row_start
        .checked_add(row_bytes)
        .filter(|&end| end <= raw.len())
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: table.name().to_owned(),
            offset: row_start,
            byte_len: row_bytes,
            buffer_len: raw.len(),
            reason: format!(
                "row {row_idx} byte range [{row_start}..{}] exceeds tensor buffer len {}",
                row_start.saturating_add(row_bytes),
                raw.len()
            ),
        })?;

    let row_slice = &raw[row_start..row_end];
    let mut block_out = [0.0_f32; BLOCK_ELEMS];
    for (blk_idx, out_chunk) in out.chunks_exact_mut(BLOCK_ELEMS).enumerate() {
        let start = blk_idx * BLOCK_BYTES;
        let end = start + BLOCK_BYTES;
        let block: &[u8; BLOCK_BYTES] =
            row_slice[start..end]
                .try_into()
                .map_err(|_| K3Error::TensorBoundsError {
                    tensor_name: table.name().to_owned(),
                    offset: row_start + start,
                    byte_len: BLOCK_BYTES,
                    buffer_len: raw.len(),
                    reason: "failed to slice quantized token block".to_owned(),
                })?;
        decode_block(block, &mut block_out);
        out_chunk.copy_from_slice(&block_out);
    }

    Ok(())
}

/// Zero-copy single-row dequantization directly from a memory-mapped embedding table [`TensorView`].
///
/// Decodes only the blocks corresponding to `row_idx` without materializing the full vocabulary table.
/// Supports `Q4_0`, `Q8_0`, `Q4_K_M`, `Q5_K`, `Q6_K`, `F32`, `F16`, and `BF16`.
///
/// # Errors
///
/// - Returns [`K3Error::TensorBoundsError`] if `out` is empty, `row_idx` exceeds row count, or
///   block alignment/buffer bounds fail.
/// - Returns [`K3Error::CapabilityMismatch`] if `table.quant()` is not supported for row lookup.
pub fn dequantize_token_row(
    table: &TensorView<'_>,
    row_idx: usize,
    out: &mut [f32],
) -> Result<(), K3Error> {
    let cols = out.len();
    let _num_rows = validate_token_row_geometry(table, row_idx, cols)?;
    let raw = table.as_bytes();

    match table.quant() {
        QuantType::Q4_0 => dequantize_block_row::<Q4_0_BLOCK_ELEMENTS, Q4_0_BLOCK_BYTES>(
            table,
            row_idx,
            out,
            dequantize_q4_0_block,
        ),
        QuantType::Q8_0 => dequantize_block_row::<Q8_0_BLOCK_ELEMENTS, Q8_0_BLOCK_BYTES>(
            table,
            row_idx,
            out,
            dequantize_q8_0_block,
        ),
        QuantType::Q4_K_M => dequantize_block_row::<
            { crate::Q4_K_M_BLOCK_ELEMENTS },
            { crate::Q4_K_M_BLOCK_BYTES },
        >(table, row_idx, out, crate::dequantize_q4_k_m_block),
        QuantType::Q5_K => dequantize_block_row::<Q5_K_BLOCK_ELEMENTS, Q5_K_BLOCK_BYTES>(
            table,
            row_idx,
            out,
            dequantize_q5_k_block,
        ),
        QuantType::Q6_K => dequantize_block_row::<Q6_K_BLOCK_ELEMENTS, Q6_K_BLOCK_BYTES>(
            table,
            row_idx,
            out,
            dequantize_q6_k_block,
        ),
        QuantType::F32 => {
            let row_bytes = cols
                .checked_mul(4)
                .ok_or_else(|| K3Error::TensorBoundsError {
                    tensor_name: table.name().to_owned(),
                    offset: table.offset(),
                    byte_len: table.byte_len(),
                    buffer_len: raw.len(),
                    reason: "F32 row stride overflowed usize".to_owned(),
                })?;
            let row_start = row_idx.saturating_mul(row_bytes);
            let row_end = row_start.saturating_add(row_bytes);
            if row_end > raw.len() {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: table.name().to_owned(),
                    offset: row_start,
                    byte_len: row_bytes,
                    buffer_len: raw.len(),
                    reason: "F32 row exceeds tensor buffer".to_owned(),
                });
            }
            for (slot, chunk) in out.iter_mut().zip(raw[row_start..row_end].chunks_exact(4)) {
                *slot = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
            Ok(())
        }
        QuantType::F16 | QuantType::BF16 => {
            let row_bytes = cols
                .checked_mul(2)
                .ok_or_else(|| K3Error::TensorBoundsError {
                    tensor_name: table.name().to_owned(),
                    offset: table.offset(),
                    byte_len: table.byte_len(),
                    buffer_len: raw.len(),
                    reason: "half-precision row stride overflowed usize".to_owned(),
                })?;
            let row_start = row_idx.saturating_mul(row_bytes);
            let row_end = row_start.saturating_add(row_bytes);
            if row_end > raw.len() {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: table.name().to_owned(),
                    offset: row_start,
                    byte_len: row_bytes,
                    buffer_len: raw.len(),
                    reason: "half-precision row exceeds tensor buffer".to_owned(),
                });
            }
            let decode_half = if table.quant() == QuantType::F16 {
                f16_to_f32_finite
            } else {
                bf16_to_f32_finite
            };
            for (slot, chunk) in out.iter_mut().zip(raw[row_start..row_end].chunks_exact(2)) {
                *slot = decode_half(u16::from_le_bytes([chunk[0], chunk[1]]));
            }
            Ok(())
        }
        other => Err(K3Error::CapabilityMismatch {
            architecture: "gemma".to_owned(),
            reason: format!("dequantize_token_row does not support quant type {other:?}"),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dequantize_q8_0_block_and_vec_dot() {
        let mut raw = [0u8; Q8_0_BLOCK_BYTES];
        // d = 2.0 in f16 (0x4000)
        raw[0..2].copy_from_slice(&0x4000_u16.to_le_bytes());
        // quants: 1..=32
        for i in 0..32_u8 {
            raw[2 + usize::from(i)] = i + 1;
        }

        let mut out = [0.0_f32; Q8_0_BLOCK_ELEMENTS];
        dequantize_q8_0_block(&raw, &mut out);

        for (i, &val) in out.iter().enumerate() {
            #[expect(
                clippy::cast_precision_loss,
                reason = "loop index fits comfortably in f32"
            )]
            let expected = 2.0 * ((i + 1) as f32);
            assert!((val - expected).abs() < 1e-5);
        }

        // Test zero scale edge case
        let zero_block = [0u8; Q8_0_BLOCK_BYTES];
        let mut zero_out = [1.0_f32; Q8_0_BLOCK_ELEMENTS];
        dequantize_q8_0_block(&zero_block, &mut zero_out);
        assert!(zero_out.iter().all(|&v| v == 0.0));

        // Test vec_dot_q8_0
        let view = TensorView::from_bytes("test_q8", QuantType::Q8_0, &[1, 32], 0, &raw)
            .expect("valid Q8_0 view");
        let input = [1.0_f32; 32];
        let mut dot_out = [0.0_f32; 1];
        vec_dot_q8_0(&mut dot_out, &view, &input).expect("vec_dot_q8_0 succeeds");

        // Sum of 2.0 * (1..=32) = 2 * (32 * 33 / 2) = 1056
        assert!((dot_out[0] - 1056.0).abs() < 1e-4);
    }

    #[test]
    fn test_dequantize_q6_k_block_and_vec_dot() {
        let mut raw = [0u8; Q6_K_BLOCK_BYTES];
        // d = 1.0 in f16 (0x3C00)
        raw[208..210].copy_from_slice(&0x3C00_u16.to_le_bytes());
        // scales = 1 for all 16 sub-blocks
        for s in &mut raw[192..208] {
            *s = 1;
        }
        // ql: 0x22 (low nibble = 2, high nibble = 2)
        for q in &mut raw[0..128] {
            *q = 0x22;
        }
        // qh: 0x00 (high bits = 0)
        // Reconstructed q = 2 - 32 = -30
        let mut out = [0.0_f32; Q6_K_BLOCK_ELEMENTS];
        dequantize_q6_k_block(&raw, &mut out);

        for &val in &out {
            // d=1.0, scale=1, q=-30 -> val = -30.0
            assert!((val - (-30.0)).abs() < 1e-4);
        }

        // Test vec_dot_q6_k
        let view = TensorView::from_bytes("test_q6_k", QuantType::Q6_K, &[1, 256], 0, &raw)
            .expect("valid Q6_K view");
        let input = [1.0_f32; 256];
        let mut dot_out = [0.0_f32; 1];
        vec_dot_q6_k(&mut dot_out, &view, &input).expect("vec_dot_q6_k succeeds");

        // Sum of 256 * (-30.0) = -7680
        assert!((dot_out[0] - (-7680.0)).abs() < 1e-3);
    }

    #[test]
    fn test_dequantize_q5_k_block_and_vec_dot() {
        let mut raw = [0u8; Q5_K_BLOCK_BYTES];
        // d = 1.0 (0x3C00), dmin = 0.5 (0x3800)
        raw[0..2].copy_from_slice(&0x3C00_u16.to_le_bytes());
        raw[2..4].copy_from_slice(&0x3800_u16.to_le_bytes());

        raw[4] = 2;
        raw[8] = 1;
        raw[5] = 3;
        raw[9] = 2;
        raw[6] = 4;
        raw[10] = 3;
        raw[7] = 5;
        raw[11] = 4;

        for b in &mut raw[16..48] {
            *b = 0x09;
        }
        for b in &mut raw[48..80] {
            *b = 0x53;
        }
        for b in &mut raw[80..112] {
            *b = 0x24;
        }

        let mut out = [0.0_f32; Q5_K_BLOCK_ELEMENTS];
        dequantize_q5_k_block(&raw, &mut out);

        let sum: f32 = out.iter().sum();
        assert!((sum - 4928.0).abs() < 1e-3);

        let view = TensorView::from_bytes("test_q5_k", QuantType::Q5_K, &[1, 256], 0, &raw)
            .expect("valid Q5_K view");
        let input = [1.0_f32; 256];
        let mut dot_out = [0.0_f32; 1];
        matvec_q5_k(&view, &input, &mut dot_out).expect("matvec_q5_k succeeds");
        assert!((dot_out[0] - 4928.0).abs() < 1e-3);

        // Also test matvec_q5_k_slice
        let block_struct = BlockQ5_K {
            d: 0x3C00,
            dmin: 0x3800,
            scales: [2, 3, 4, 5, 1, 2, 3, 4, 0, 0, 0, 0],
            qh: [0x09; 32],
            qs: {
                let mut qs = [0u8; 128];
                for b in &mut qs[0..32] {
                    *b = 0x53;
                }
                for b in &mut qs[32..64] {
                    *b = 0x24;
                }
                qs
            },
        };
        let mut slice_out = [0.0_f32; 1];
        matvec_q5_k_slice(&[block_struct], &input, &mut slice_out, 1, 256);
        assert!((slice_out[0] - 4928.0).abs() < 1e-3);
    }

    #[test]
    fn test_dequantize_q4_0_block_and_vec_dot() {
        let mut raw = [0u8; Q4_0_BLOCK_BYTES];
        // d = 0.5 in f16 (0x3800)
        raw[0..2].copy_from_slice(&0x3800_u16.to_le_bytes());
        // Low nibbles = 10 -> q0 = +2, high nibbles = 5 -> q1 = -3
        for b in &mut raw[2..18] {
            *b = (0x05_u8 << 4) | 0x0A_u8;
        }

        let mut out = [0.0_f32; Q4_0_BLOCK_ELEMENTS];
        dequantize_q4_0_block(&raw, &mut out);
        assert!(out[..16].iter().all(|&v| (v - 1.0).abs() < 1e-6));
        assert!(out[16..].iter().all(|&v| (v - (-1.5)).abs() < 1e-6));

        let view = TensorView::from_bytes("test_q4_0", QuantType::Q4_0, &[32, 1], 0, &raw)
            .expect("valid Q4_0 view");
        let input = [2.0_f32; 32];
        let mut dot_out = [0.0_f32; 1];
        vec_dot_q4_0(&mut dot_out, &view, &input).expect("vec_dot_q4_0 succeeds");
        // 16 * (1.0 * 2.0) + 16 * (-1.5 * 2.0) = 32 - 48 = -16
        assert!((dot_out[0] - (-16.0)).abs() < 1e-4);
    }

    #[test]
    fn test_vec_dot_f16_and_bf16() {
        let cols = 19usize;
        let rows = 2usize;
        let mut half_bytes = vec![0u8; rows * cols * 2];
        let mut bfloat_bytes = vec![0u8; rows * cols * 2];
        for r in 0..rows {
            for c in 0..cols {
                let off = (r * cols + c) * 2;
                // Row 0 = 1.0, Row 1 = -2.0
                let half_bits: u16 = if r == 0 { 0x3C00 } else { 0xC000 };
                let bfloat_bits: u16 = if r == 0 { 0x3F80 } else { 0xC000 };
                half_bytes[off..off + 2].copy_from_slice(&half_bits.to_le_bytes());
                bfloat_bytes[off..off + 2].copy_from_slice(&bfloat_bits.to_le_bytes());
            }
        }

        let input = vec![0.5_f32; cols];
        let shape = [cols, rows];
        let half_view =
            TensorView::from_bytes("w_f16", QuantType::F16, &shape, 0, &half_bytes).unwrap();
        let bfloat_view =
            TensorView::from_bytes("w_bf16", QuantType::BF16, &shape, 0, &bfloat_bytes).unwrap();

        let mut half_out = [0.0_f32; 2];
        let mut bfloat_out = [0.0_f32; 2];
        vec_dot_f16(&mut half_out, &half_view, &input).expect("vec_dot_f16");
        vec_dot_bf16(&mut bfloat_out, &bfloat_view, &input).expect("vec_dot_bf16");

        assert!((half_out[0] - 9.5).abs() < 1e-4);
        assert!((half_out[1] - (-19.0)).abs() < 1e-4);
        assert!((bfloat_out[0] - 9.5).abs() < 1e-4);
        assert!((bfloat_out[1] - (-19.0)).abs() < 1e-4);
    }

    #[test]
    fn test_dequantize_token_row_q4_0_q8_0() {
        // 1. Multi-row Q4_0 table (3 rows x 32 cols)
        let mut q4_bytes = vec![0_u8; 3 * Q4_0_BLOCK_BYTES];
        // Row 0: d = 0.5 (0x3800), low nibbles = 9 (+1), high nibbles = 7 (-1)
        q4_bytes[0..2].copy_from_slice(&0x3800_u16.to_le_bytes());
        q4_bytes[2..18].fill((0x07 << 4) | 0x09);
        // Row 2: d = 1.0 (0x3C00), low nibbles = 10 (+2), high nibbles = 6 (-2)
        let r2 = 2 * Q4_0_BLOCK_BYTES;
        q4_bytes[r2..r2 + 2].copy_from_slice(&0x3C00_u16.to_le_bytes());
        q4_bytes[r2 + 2..r2 + 18].fill((0x06 << 4) | 0x0A);

        let q4_view = TensorView::from_bytes(
            "per_layer_token_embd.weight",
            QuantType::Q4_0,
            &[32, 3],
            0,
            &q4_bytes,
        )
        .expect("Q4_0 embedding view");
        let mut row0 = [0.0_f32; 32];
        let mut row2 = [0.0_f32; 32];
        dequantize_token_row(&q4_view, 0, &mut row0).expect("dequantize Q4_0 row 0");
        dequantize_token_row(&q4_view, 2, &mut row2).expect("dequantize Q4_0 row 2");
        assert!(row0[..16].iter().all(|&v| (v - 0.5).abs() < 1e-6));
        assert!(row0[16..].iter().all(|&v| (v - (-0.5)).abs() < 1e-6));
        assert!(row2[..16].iter().all(|&v| (v - 2.0).abs() < 1e-6));
        assert!(row2[16..].iter().all(|&v| (v - (-2.0)).abs() < 1e-6));

        // 2. Multi-row Q8_0 table (2 rows x 32 cols)
        let mut q8_bytes = vec![0_u8; 2 * Q8_0_BLOCK_BYTES];
        q8_bytes[Q8_0_BLOCK_BYTES..Q8_0_BLOCK_BYTES + 2].copy_from_slice(&0x4000_u16.to_le_bytes()); // d = 2.0
        for col in 0..32_u8 {
            q8_bytes[Q8_0_BLOCK_BYTES + 2 + usize::from(col)] = col + 1;
        }
        let q8_view =
            TensorView::from_bytes("token_embd.weight", QuantType::Q8_0, &[32, 2], 0, &q8_bytes)
                .expect("Q8_0 embedding view");
        let mut q8_row1 = [0.0_f32; 32];
        dequantize_token_row(&q8_view, 1, &mut q8_row1).expect("dequantize Q8_0 row 1");
        assert!((q8_row1[0] - 2.0).abs() < 1e-5);
        assert!((q8_row1[31] - 64.0).abs() < 1e-5);

        // Out-of-bounds row fails cleanly
        assert!(matches!(
            dequantize_token_row(&q8_view, 2, &mut q8_row1),
            Err(K3Error::TensorBoundsError { .. })
        ));
    }
}
