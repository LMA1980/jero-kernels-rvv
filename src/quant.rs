//! Block dequantization and quantized vector dot product primitives for `Q8_0` and `Q6_K`.
//!
//! Provides bit-equivalent dequantization routines matching GGUF / `llama.cpp` definitions
//! and matrix-vector dot dispatchers for both VLA RVV 1.0 and portable scalar execution.

use k3_core::{K3Error, QuantType, TensorView};

#[cfg(all(target_arch = "riscv64", not(miri)))]
use crate::dot_product_f32_rvv;
use crate::dot_product_f32_scalar;
use crate::f16_to_f32_finite;

/// Number of scalar elements in a single `Q8_0` block.
pub const Q8_0_BLOCK_ELEMENTS: usize = 32;

/// Storage size in bytes of a single `Q8_0` block.
pub const Q8_0_BLOCK_BYTES: usize = 34;

/// Number of scalar elements in a single `Q6_K` super-block.
pub const Q6_K_BLOCK_ELEMENTS: usize = 256;

/// Storage size in bytes of a single `Q6_K` super-block.
pub const Q6_K_BLOCK_BYTES: usize = 210;

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
            row_sum += dot_product_f32_rvv(&dequant_buf, input_sub);
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
}
