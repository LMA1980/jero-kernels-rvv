//! Block dequantization and quantized vector dot product primitives across GGML block, super-block,
//! non-linear lookup (`IQ4_NL`), ternary (`TQ1_0`/`TQ2_0`/`I2_S`), and IEEE/brain half formats.
//!
//! Provides bit-equivalent dequantization routines matching GGUF / `llama.cpp` / `BitNet.cpp`
//! definitions and matrix-vector dot dispatchers for both VLA RVV 1.0 and portable scalar execution.

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

/// Number of scalar elements in a single `Q4_1` block.
pub const Q4_1_BLOCK_ELEMENTS: usize = 32;

/// Storage size in bytes of a single `Q4_1` block (2B `f16` `d` + 2B `f16` `m` + 16B packed nibbles).
pub const Q4_1_BLOCK_BYTES: usize = 20;

/// Number of scalar elements in a single `Q5_0` block.
pub const Q5_0_BLOCK_ELEMENTS: usize = 32;

/// Storage size in bytes of a single `Q5_0` block (2B `f16` `d` + 4B `qh` + 16B packed nibbles).
pub const Q5_0_BLOCK_BYTES: usize = 22;

/// Number of scalar elements in a single `Q5_1` block.
pub const Q5_1_BLOCK_ELEMENTS: usize = 32;

/// Storage size in bytes of a single `Q5_1` block (2B `f16` `d` + 2B `f16` `m` + 4B `qh` + 16B nibbles).
pub const Q5_1_BLOCK_BYTES: usize = 24;

/// Number of scalar elements in a single `Q8_0` block.
pub const Q8_0_BLOCK_ELEMENTS: usize = 32;

/// Storage size in bytes of a single `Q8_0` block.
pub const Q8_0_BLOCK_BYTES: usize = 34;

/// Number of scalar elements in a single `Q2_K` super-block.
pub const Q2_K_BLOCK_ELEMENTS: usize = 256;

/// Storage size in bytes of a single `Q2_K` super-block (16B `scales` + 64B `qs` + 2B `d` + 2B `dmin`).
pub const Q2_K_BLOCK_BYTES: usize = 84;

/// Number of scalar elements in a single `Q3_K` super-block.
pub const Q3_K_BLOCK_ELEMENTS: usize = 256;

/// Storage size in bytes of a single `Q3_K` super-block (32B `hmask` + 64B `qs` + 12B `scales` + 2B `d`).
pub const Q3_K_BLOCK_BYTES: usize = 110;

/// Number of scalar elements in a single `Q5_K` super-block.
pub const Q5_K_BLOCK_ELEMENTS: usize = 256;

/// Storage size in bytes of a single `Q5_K` super-block.
pub const Q5_K_BLOCK_BYTES: usize = 176;

/// Number of scalar elements in a single `Q6_K` super-block.
pub const Q6_K_BLOCK_ELEMENTS: usize = 256;

/// Storage size in bytes of a single `Q6_K` super-block.
pub const Q6_K_BLOCK_BYTES: usize = 210;

/// Number of scalar elements in a single `IQ4_NL` block.
pub const IQ4_NL_BLOCK_ELEMENTS: usize = 32;

/// Storage size in bytes of a single `IQ4_NL` block (2B `f16` `d` + 16B packed nibble indices).
pub const IQ4_NL_BLOCK_BYTES: usize = 18;

/// GGML 16-entry non-linear quantization table for `IQ4_NL` (`ggml_type = 20`).
pub const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// Number of scalar elements in a single `BitNet.cpp` `I2_S` (`ggml_type = 36`) block.
pub const I2_S_BLOCK_ELEMENTS: usize = 128;

/// Storage size in bytes of a single packed `BitNet.cpp` `I2_S` block (`128 * 2` bits = `32` bytes).
pub const I2_S_BLOCK_BYTES: usize = 32;

/// Trailing tensor-wide scale metadata size in bytes appended after `I2_S` packed blocks.
pub const I2_S_SCALE_TRAILER_BYTES: usize = 32;

#[inline]
fn scale_i8_into_f32<const N: usize>(scale: f32, quants: &[i8; N], out: &mut [f32; N]) {
    if scale == 0.0 {
        out.fill(0.0);
        return;
    }
    for (dst, &q) in out.iter_mut().zip(quants.iter()) {
        *dst = scale * f32::from(q);
    }
}

#[inline]
fn affine_i8_into_f32<const N: usize>(scale: f32, min: f32, quants: &[i8; N], out: &mut [f32; N]) {
    if scale == 0.0 && min == 0.0 {
        out.fill(0.0);
        return;
    }
    for (dst, &q) in out.iter_mut().zip(quants.iter()) {
        *dst = scale * f32::from(q) + min;
    }
}

/// Unpacks a 20-byte `Q4_1` block into 32 unsigned `0..=15` quants stored as `i8`, returning
/// `(d, m)`.
#[must_use]
pub fn unpack_q4_1_block_i8(
    block: &[u8; Q4_1_BLOCK_BYTES],
    out: &mut [i8; Q4_1_BLOCK_ELEMENTS],
) -> (f32, f32) {
    let d = f16_to_f32_finite(u16::from_le_bytes([block[0], block[1]]));
    let m = f16_to_f32_finite(u16::from_le_bytes([block[2], block[3]]));
    for (i, &byte) in block[4..20].iter().enumerate() {
        out[i] = i8::from_ne_bytes([byte & 0x0F]);
        out[i + 16] = i8::from_ne_bytes([(byte >> 4) & 0x0F]);
    }
    (d, m)
}

/// Dequantizes a 20-byte `Q4_1` block into 32 `f32` elements (`y = d * q + m`).
pub fn dequantize_q4_1_block(block: &[u8; Q4_1_BLOCK_BYTES], out: &mut [f32; Q4_1_BLOCK_ELEMENTS]) {
    let mut quants = [0_i8; Q4_1_BLOCK_ELEMENTS];
    let (d, m) = unpack_q4_1_block_i8(block, &mut quants);
    affine_i8_into_f32(d, m, &quants, out);
}

/// Unpacks a 22-byte `Q5_0` block into 32 signed `-16..=15` `i8` values and returns scale `d`.
#[must_use]
pub fn unpack_q5_0_block_i8(
    block: &[u8; Q5_0_BLOCK_BYTES],
    out: &mut [i8; Q5_0_BLOCK_ELEMENTS],
) -> f32 {
    let d = f16_to_f32_finite(u16::from_le_bytes([block[0], block[1]]));
    let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
    for (j, &byte) in block[6..22].iter().enumerate() {
        let xh0 = if ((qh >> j) & 1) != 0 { 16_i8 } else { 0_i8 };
        let xh1 = if ((qh >> (j + 16)) & 1) != 0 {
            16_i8
        } else {
            0_i8
        };
        out[j] = (i8::from_ne_bytes([byte & 0x0F]) | xh0) - 16;
        out[j + 16] = (i8::from_ne_bytes([(byte >> 4) & 0x0F]) | xh1) - 16;
    }
    d
}

/// Dequantizes a 22-byte `Q5_0` block (`ggml_type = 6`) into 32 `f32` elements.
pub fn dequantize_q5_0_block(block: &[u8; Q5_0_BLOCK_BYTES], out: &mut [f32; Q5_0_BLOCK_ELEMENTS]) {
    let mut quants = [0_i8; Q5_0_BLOCK_ELEMENTS];
    let d = unpack_q5_0_block_i8(block, &mut quants);
    scale_i8_into_f32(d, &quants, out);
}

/// Unpacks a 24-byte `Q5_1` block into 32 unsigned `0..=31` quants stored as `i8`, returning
/// `(d, m)`.
#[must_use]
pub fn unpack_q5_1_block_i8(
    block: &[u8; Q5_1_BLOCK_BYTES],
    out: &mut [i8; Q5_1_BLOCK_ELEMENTS],
) -> (f32, f32) {
    let d = f16_to_f32_finite(u16::from_le_bytes([block[0], block[1]]));
    let m = f16_to_f32_finite(u16::from_le_bytes([block[2], block[3]]));
    let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
    for (j, &byte) in block[8..24].iter().enumerate() {
        let xh0 = if ((qh >> j) & 1) != 0 { 16_i8 } else { 0_i8 };
        let xh1 = if ((qh >> (j + 16)) & 1) != 0 {
            16_i8
        } else {
            0_i8
        };
        out[j] = i8::from_ne_bytes([byte & 0x0F]) | xh0;
        out[j + 16] = i8::from_ne_bytes([(byte >> 4) & 0x0F]) | xh1;
    }
    (d, m)
}

/// Dequantizes a 24-byte `Q5_1` block (`ggml_type = 7`) into 32 `f32` elements (`y = d * q + m`).
pub fn dequantize_q5_1_block(block: &[u8; Q5_1_BLOCK_BYTES], out: &mut [f32; Q5_1_BLOCK_ELEMENTS]) {
    let mut quants = [0_i8; Q5_1_BLOCK_ELEMENTS];
    let (d, m) = unpack_q5_1_block_i8(block, &mut quants);
    affine_i8_into_f32(d, m, &quants, out);
}

/// Unpacks an 84-byte `Q2_K` super-block into 256 `0..=3` quants (`i8`) and 16 sub-block
/// `(scale, min)` pairs.
pub fn unpack_q2_k_block_i8(
    block: &[u8; Q2_K_BLOCK_BYTES],
    out_quants: &mut [i8; Q2_K_BLOCK_ELEMENTS],
    out_scales: &mut [f32; 16],
    out_mins: &mut [f32; 16],
) {
    let scales = &block[0..16];
    let qs = &block[16..80];
    let d = f16_to_f32_finite(u16::from_le_bytes([block[80], block[81]]));
    let dmin = f16_to_f32_finite(u16::from_le_bytes([block[82], block[83]]));

    let mut is = 0_usize;
    for half in 0..2 {
        let q_base = half * 32;
        let out_base = half * 128;
        for j in 0..4 {
            let shift = j * 2;
            let row_base = out_base + j * 32;

            let sc0 = scales[is];
            out_scales[is] = d * f32::from(sc0 & 0x0F);
            out_mins[is] = dmin * f32::from(sc0 >> 4);
            is += 1;
            for l in 0..16 {
                out_quants[row_base + l] = i8::from_ne_bytes([(qs[q_base + l] >> shift) & 0x03]);
            }

            let sc1 = scales[is];
            out_scales[is] = d * f32::from(sc1 & 0x0F);
            out_mins[is] = dmin * f32::from(sc1 >> 4);
            is += 1;
            for l in 0..16 {
                out_quants[row_base + 16 + l] =
                    i8::from_ne_bytes([(qs[q_base + 16 + l] >> shift) & 0x03]);
            }
        }
    }
}

/// Dequantizes an 84-byte `Q2_K` super-block (`ggml_type = 10`) into 256 `f32` elements.
pub fn dequantize_q2_k_block(block: &[u8; Q2_K_BLOCK_BYTES], out: &mut [f32; Q2_K_BLOCK_ELEMENTS]) {
    let mut quants = [0_i8; Q2_K_BLOCK_ELEMENTS];
    let mut scales = [0.0_f32; 16];
    let mut mins = [0.0_f32; 16];
    unpack_q2_k_block_i8(block, &mut quants, &mut scales, &mut mins);
    for sb in 0..16 {
        let base = sb * 16;
        let dl = scales[sb];
        let ml = mins[sb];
        for l in 0..16 {
            out[base + l] = dl * f32::from(quants[base + l]) - ml;
        }
    }
}

/// Unpacks a 110-byte `Q3_K` super-block into 256 `-4..=3` quants (`i8`) and 16 signed sub-block
/// scales (`d * (scale_i - 32)`).
pub fn unpack_q3_k_block_i8(
    block: &[u8; Q3_K_BLOCK_BYTES],
    out_quants: &mut [i8; Q3_K_BLOCK_ELEMENTS],
    out_scales: &mut [f32; 16],
) {
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0F0F_0F0F;

    let hmask = &block[0..32];
    let qs = &block[32..96];
    let scales_raw = &block[96..108];
    let d_all = f16_to_f32_finite(u16::from_le_bytes([block[108], block[109]]));

    let mut aux = [0_u32; 4];
    for (slot, chunk) in aux[..3].iter_mut().zip(scales_raw.chunks_exact(4)) {
        *slot = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
    aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
    aux[0] = (aux[0] & KMASK2) | ((tmp & KMASK1) << 4);
    aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);

    let mut scale_bytes = [0_u8; 16];
    for (i, &word) in aux.iter().enumerate() {
        scale_bytes[i * 4..(i + 1) * 4].copy_from_slice(&word.to_le_bytes());
    }

    let mut is = 0_usize;
    let mut m_mask = 1_u8;
    for half in 0..2 {
        let q_base = half * 32;
        let out_base = half * 128;
        for j in 0..4 {
            let shift = j * 2;
            let row_base = out_base + j * 32;

            let sc0 = i8::from_ne_bytes([scale_bytes[is]]);
            out_scales[is] = d_all * f32::from(i16::from(sc0) - 32);
            is += 1;
            for l in 0..16 {
                let q = i8::from_ne_bytes([(qs[q_base + l] >> shift) & 0x03]);
                let high_sub = if (hmask[l] & m_mask) != 0 { 0_i8 } else { 4_i8 };
                out_quants[row_base + l] = q - high_sub;
            }

            let sc1 = i8::from_ne_bytes([scale_bytes[is]]);
            out_scales[is] = d_all * f32::from(i16::from(sc1) - 32);
            is += 1;
            for l in 0..16 {
                let q = i8::from_ne_bytes([(qs[q_base + 16 + l] >> shift) & 0x03]);
                let high_sub = if (hmask[16 + l] & m_mask) != 0 {
                    0_i8
                } else {
                    4_i8
                };
                out_quants[row_base + 16 + l] = q - high_sub;
            }

            m_mask = m_mask.rotate_left(1);
        }
    }
}

/// Dequantizes a 110-byte `Q3_K` super-block (`ggml_type = 11`) into 256 `f32` elements.
pub fn dequantize_q3_k_block(block: &[u8; Q3_K_BLOCK_BYTES], out: &mut [f32; Q3_K_BLOCK_ELEMENTS]) {
    let mut quants = [0_i8; Q3_K_BLOCK_ELEMENTS];
    let mut scales = [0.0_f32; 16];
    unpack_q3_k_block_i8(block, &mut quants, &mut scales);
    for (sb, &dl) in scales.iter().enumerate() {
        let base = sb * 16;
        for l in 0..16 {
            out[base + l] = dl * f32::from(quants[base + l]);
        }
    }
}

/// Unpacks an 18-byte `IQ4_NL` block via [`KVALUES_IQ4NL`] into 32 `i8` values and returns `d`.
#[must_use]
pub fn unpack_iq4_nl_block_i8(
    block: &[u8; IQ4_NL_BLOCK_BYTES],
    out: &mut [i8; IQ4_NL_BLOCK_ELEMENTS],
) -> f32 {
    let d = f16_to_f32_finite(u16::from_le_bytes([block[0], block[1]]));
    for (j, &byte) in block[2..18].iter().enumerate() {
        out[j] = KVALUES_IQ4NL[usize::from(byte & 0x0F)];
        out[j + 16] = KVALUES_IQ4NL[usize::from(byte >> 4)];
    }
    d
}

/// Dequantizes an 18-byte `IQ4_NL` block (`ggml_type = 20`) into 32 `f32` elements matching `ggml`'s
/// split nibble layout (`qs[0..16]` low nibbles -> `out[0..16]`, high nibbles -> `out[16..32]`).
pub fn dequantize_iq4_nl_block(
    block: &[u8; IQ4_NL_BLOCK_BYTES],
    out: &mut [f32; IQ4_NL_BLOCK_ELEMENTS],
) {
    let mut quants = [0_i8; IQ4_NL_BLOCK_ELEMENTS];
    let d = unpack_iq4_nl_block_i8(block, &mut quants);
    scale_i8_into_f32(d, &quants, out);
}

/// Reads the tensor-wide `f32` scale from a 32-byte `I2_S` scale trailer, clamping non-finite values
/// safely to `0.0`.
#[must_use]
pub fn read_i2_s_trailer_scale(trailer: &[u8]) -> f32 {
    if trailer.len() < 4 {
        return 0.0;
    }
    let val = f32::from_le_bytes([trailer[0], trailer[1], trailer[2], trailer[3]]);
    if val.is_finite() { val } else { 0.0 }
}

/// Unpacks a 32-byte `BitNet.cpp` `I2_S` (`ggml_type = 36`) block into 128 ternary `i8` weights in
/// `{-1, 0, +1}` using the 4-group interleaved bit layout (`[7:6]`, `[5:4]`, `[3:2]`, `[1:0]`).
pub fn unpack_i2_s_block_i8(block: &[u8; I2_S_BLOCK_BYTES], out: &mut [i8; I2_S_BLOCK_ELEMENTS]) {
    for (gp, &byte) in block.iter().enumerate() {
        out[gp] = i8::from_ne_bytes([(byte >> 6) & 0x03]) - 1;
        out[32 + gp] = i8::from_ne_bytes([(byte >> 4) & 0x03]) - 1;
        out[64 + gp] = i8::from_ne_bytes([(byte >> 2) & 0x03]) - 1;
        out[96 + gp] = i8::from_ne_bytes([byte & 0x03]) - 1;
    }
}

/// Dequantizes a 32-byte `BitNet.cpp` `I2_S` block with tensor-wide `scale` into 128 `f32` values.
pub fn dequantize_i2_s_block(
    block: &[u8; I2_S_BLOCK_BYTES],
    scale: f32,
    out: &mut [f32; I2_S_BLOCK_ELEMENTS],
) {
    let mut quants = [0_i8; I2_S_BLOCK_ELEMENTS];
    unpack_i2_s_block_i8(block, &mut quants);
    scale_i8_into_f32(scale, &quants, out);
}

/// Dequantizes an entire `I2_S` packed row (without repeating the trailing 32-byte tensor scale)
/// using the pre-extracted tensor-wide `scale`.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `out` is empty, not divisible by `128`, or
/// `row_bytes.len()` does not match `(out.len() / 128) * 32`.
pub fn dequantize_i2_s_row_with_scale(
    row_bytes: &[u8],
    scale: f32,
    out: &mut [f32],
) -> Result<(), K3Error> {
    if out.is_empty() || out.len() % I2_S_BLOCK_ELEMENTS != 0 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "I2_S".to_owned(),
            offset: 0,
            byte_len: row_bytes.len(),
            buffer_len: out.len(),
            reason: format!(
                "I2_S row length {} must be a positive multiple of {I2_S_BLOCK_ELEMENTS}",
                out.len()
            ),
        });
    }
    let expected = (out.len() / I2_S_BLOCK_ELEMENTS) * I2_S_BLOCK_BYTES;
    if row_bytes.len() != expected {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "I2_S".to_owned(),
            offset: 0,
            byte_len: row_bytes.len(),
            buffer_len: expected,
            reason: format!(
                "I2_S row byte length {} != expected {expected}",
                row_bytes.len()
            ),
        });
    }
    for (chunk, dst) in row_bytes
        .chunks_exact(I2_S_BLOCK_BYTES)
        .zip(out.chunks_exact_mut(I2_S_BLOCK_ELEMENTS))
    {
        if let (Ok(blk), Ok(out_arr)) = (chunk.try_into(), dst.try_into()) {
            dequantize_i2_s_block(blk, scale, out_arr);
        }
    }
    Ok(())
}

/// Dequantizes an `IQ4_NL` row using RVV 1.0 `vrgather.vv` over an `LMUL = 1` lookup table register.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `out` is empty, not divisible by `32`, or
/// `row_bytes.len()` does not match `(out.len() / 32) * 18`.
#[cfg(all(target_arch = "riscv64", not(miri)))]
pub fn dequantize_iq4_nl_row_rvv(row_bytes: &[u8], out: &mut [f32]) -> Result<(), K3Error> {
    if out.is_empty() || out.len() % IQ4_NL_BLOCK_ELEMENTS != 0 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "IQ4_NL".to_owned(),
            offset: 0,
            byte_len: row_bytes.len(),
            buffer_len: out.len(),
            reason: format!(
                "IQ4_NL row length {} must be a positive multiple of {IQ4_NL_BLOCK_ELEMENTS}",
                out.len()
            ),
        });
    }
    let n_blocks = out.len() / IQ4_NL_BLOCK_ELEMENTS;
    let expected = n_blocks * IQ4_NL_BLOCK_BYTES;
    if row_bytes.len() != expected {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "IQ4_NL".to_owned(),
            offset: 0,
            byte_len: row_bytes.len(),
            buffer_len: expected,
            reason: format!(
                "IQ4_NL row byte length {} != expected {expected}",
                row_bytes.len()
            ),
        });
    }

    for (blk_bytes, out_chunk) in row_bytes
        .chunks_exact(IQ4_NL_BLOCK_BYTES)
        .zip(out.chunks_exact_mut(IQ4_NL_BLOCK_ELEMENTS))
    {
        let d = f16_to_f32_finite(u16::from_le_bytes([blk_bytes[0], blk_bytes[1]]));
        if d == 0.0 && d.is_sign_positive() {
            out_chunk.fill(0.0);
            continue;
        }
        let tbl_ptr = KVALUES_IQ4NL.as_ptr();
        let qs_ptr = blk_bytes[2..18].as_ptr();
        let out_lo_ptr = out_chunk.as_mut_ptr();
        // SAFETY: `KVALUES_IQ4NL` holds 16 valid `i8` bytes, `qs_ptr` holds 16 packed bytes, and
        // `out_lo_ptr` points to 32 writable `f32` elements (`[0..16]` low, `[16..32]` high).
        // Under `VLEN >= 128`, `LMUL = 1` at `e8` holds all 16 table entries in `v8`,
        // `vrgather.vv` destinations (`v4`, `v5`) do not overlap sources (`v8`, `v2`, `v3`), and
        // `e32, m4` guarantees `VLMAX >= 16` across all `VLEN >= 128` implementations
        // (`VLEN = 128/256/1024`) with disjoint 4-register aligned destination groups (`v12..v15`,
        // `v16..v19`).
        unsafe {
            let out_hi_ptr = out_lo_ptr.add(16);
            core::arch::asm!(
                "vsetivli zero, 16, e8, m1, ta, ma",
                "vle8.v v8, ({tbl})",
                "vle8.v v1, ({qs})",
                "vand.vi v2, v1, 15",
                "vsrl.vi v3, v1, 4",
                "vrgather.vv v4, v8, v2",
                "vrgather.vv v5, v8, v3",
                "vsetivli zero, 16, e32, m4, ta, ma",
                "vsext.vf4 v12, v4",
                "vfcvt.f.x.v v12, v12",
                "vfmul.vf v12, v12, {d}",
                "vse32.v v12, ({out_lo})",
                "vsext.vf4 v16, v5",
                "vfcvt.f.x.v v16, v16",
                "vfmul.vf v16, v16, {d}",
                "vse32.v v16, ({out_hi})",
                tbl = in(reg) tbl_ptr,
                qs = in(reg) qs_ptr,
                out_lo = in(reg) out_lo_ptr,
                out_hi = in(reg) out_hi_ptr,
                d = in(freg) d,
                out("v1") _,
                out("v2") _,
                out("v3") _,
                out("v4") _,
                out("v5") _,
                out("v8") _,
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
    Ok(())
}

/// Dequantizes a contiguous slice of `Q5_0` blocks (`blocks.len() == (out.len() / 32) * 22`) into
/// `out`.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `out` is empty, not divisible by `32`, or `blocks` length
/// does not match.
pub fn dequantize_q5_0(blocks: &[u8], out: &mut [f32]) -> Result<(), K3Error> {
    dequantize_row_bytes(QuantType::Q5_0, blocks, out.len(), out)
}

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

fn validate_quant_matvec_dimensions(
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

    let block_elems = expected_quant.block_elements();
    let (rows, cols) = if shape.len() == 2 {
        let dim0 = shape[0];
        let dim1 = shape[1];
        if output_len == dim0 && input_len == dim1 && dim1 % block_elems == 0 {
            (dim0, dim1)
        } else if output_len == dim1 && input_len == dim0 && dim0 % block_elems == 0 {
            (dim1, dim0)
        } else {
            return Err(K3Error::TensorBoundsError {
                tensor_name: weights.name().to_owned(),
                offset: weights.offset(),
                byte_len: weights.byte_len(),
                buffer_len: weights.as_bytes().len(),
                reason: format!(
                    "matrix shape [{dim0}, {dim1}] does not match output len {output_len} and {expected_quant:?}-aligned input len {input_len}"
                ),
            });
        }
    } else {
        let total_elems = weights.num_elements();
        if input_len == 0
            || output_len == 0
            || input_len % block_elems != 0
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

    let required_bytes = expected_quant.storage_bytes(weights.name(), rows * cols)?;
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

/// Computes the block-wise dot product of `Q5_0` weights against `Q8_0` quantized activations.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `n_elems` is empty or not divisible by `32`, or if
/// either byte slice length does not match `n_elems / 32` blocks.
pub fn vec_dot_q5_0_q8_0(w_q5_0: &[u8], x_q8_0: &[u8], n_elems: usize) -> Result<f32, K3Error> {
    if n_elems == 0 || n_elems % Q5_0_BLOCK_ELEMENTS != 0 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "vec_dot_q5_0_q8_0".to_owned(),
            offset: 0,
            byte_len: w_q5_0.len(),
            buffer_len: x_q8_0.len(),
            reason: format!(
                "n_elems ({n_elems}) must be a positive multiple of {Q5_0_BLOCK_ELEMENTS}"
            ),
        });
    }
    let n_blocks = n_elems / Q5_0_BLOCK_ELEMENTS;
    let expected_w = n_blocks * Q5_0_BLOCK_BYTES;
    let expected_x = n_blocks * Q8_0_BLOCK_BYTES;
    if w_q5_0.len() != expected_w || x_q8_0.len() != expected_x {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "vec_dot_q5_0_q8_0".to_owned(),
            offset: 0,
            byte_len: w_q5_0.len(),
            buffer_len: x_q8_0.len(),
            reason: format!(
                "block byte mismatch: expected ({expected_w}, {expected_x}), got ({}, {})",
                w_q5_0.len(),
                x_q8_0.len()
            ),
        });
    }

    let mut w_i8 = [0_i8; Q5_0_BLOCK_ELEMENTS];
    let mut x_i8 = [0_i8; Q8_0_BLOCK_ELEMENTS];
    let mut acc = 0.0_f32;
    for (w_blk, x_blk) in w_q5_0
        .chunks_exact(Q5_0_BLOCK_BYTES)
        .zip(x_q8_0.chunks_exact(Q8_0_BLOCK_BYTES))
    {
        let w_arr: &[u8; Q5_0_BLOCK_BYTES] =
            w_blk.try_into().map_err(|_| K3Error::TensorBoundsError {
                tensor_name: "vec_dot_q5_0_q8_0".to_owned(),
                offset: 0,
                byte_len: w_blk.len(),
                buffer_len: w_q5_0.len(),
                reason: "invalid Q5_0 block slice".to_owned(),
            })?;
        let x_arr: &[u8; Q8_0_BLOCK_BYTES] =
            x_blk.try_into().map_err(|_| K3Error::TensorBoundsError {
                tensor_name: "vec_dot_q5_0_q8_0".to_owned(),
                offset: 0,
                byte_len: x_blk.len(),
                buffer_len: x_q8_0.len(),
                reason: "invalid Q8_0 block slice".to_owned(),
            })?;

        let dw = unpack_q5_0_block_i8(w_arr, &mut w_i8);
        let dx = f16_to_f32_finite(u16::from_le_bytes([x_arr[0], x_arr[1]]));
        if dw == 0.0 || dx == 0.0 {
            continue;
        }
        for (dst, &byte) in x_i8.iter_mut().zip(&x_arr[2..34]) {
            *dst = i8::from_ne_bytes([byte]);
        }
        let dot_i32 = crate::ime::dot_i8_vector(&w_i8, &x_i8);
        #[expect(
            clippy::cast_precision_loss,
            reason = "32-element i8*i8 sum fits within 24-bit f32 mantissa"
        )]
        {
            acc += (dot_i32 as f32) * (dw * dx);
        }
    }

    Ok(acc)
}

fn vec_dot_generic_block<const BLOCK_ELEMS: usize, const BLOCK_BYTES: usize>(
    quant: QuantType,
    output: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
    mut dequant: impl FnMut(&[u8; BLOCK_BYTES], &mut [f32; BLOCK_ELEMS]),
) -> Result<(), K3Error> {
    let (rows, cols) = validate_quant_matvec_dimensions(quant, output.len(), weights, input.len())?;
    let blocks_per_row = cols / BLOCK_ELEMS;
    let raw_bytes = weights.as_bytes();
    let mut dequant_buf = [0.0_f32; BLOCK_ELEMS];

    for (row, out_slot) in output.iter_mut().enumerate().take(rows) {
        let mut row_sum = 0.0_f32;
        for blk_idx in 0..blocks_per_row {
            let block_index = row * blocks_per_row + blk_idx;
            let byte_start = block_index * BLOCK_BYTES;
            let byte_end = byte_start + BLOCK_BYTES;
            let block_slice: &[u8; BLOCK_BYTES] = raw_bytes[byte_start..byte_end]
                .try_into()
                .map_err(|_| K3Error::TensorBoundsError {
                    tensor_name: weights.name().to_owned(),
                    offset: byte_start,
                    byte_len: BLOCK_BYTES,
                    buffer_len: raw_bytes.len(),
                    reason: format!("failed to slice {quant:?} block"),
                })?;
            dequant(block_slice, &mut dequant_buf);
            let input_sub = &input[blk_idx * BLOCK_ELEMS..(blk_idx + 1) * BLOCK_ELEMS];
            row_sum += dot_f32_dispatch(&dequant_buf, input_sub);
        }
        *out_slot = row_sum;
    }
    Ok(())
}

/// Computes matrix-vector multiplication with `Q5_0` quantized weights (`f32` activations).
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if quantization is not `Q5_0` or dimensions mismatch.
pub fn vec_dot_q5_0(
    output: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
) -> Result<(), K3Error> {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        let (rows, cols) =
            validate_quant_matvec_dimensions(QuantType::Q5_0, output.len(), weights, input.len())?;
        let blocks_per_row = cols / Q5_0_BLOCK_ELEMENTS;
        let raw_bytes = weights.as_bytes();
        let mut unpacked_i8 = [0_i8; Q5_0_BLOCK_ELEMENTS];

        for (row, out_slot) in output.iter_mut().enumerate().take(rows) {
            let mut row_sum = 0.0_f32;
            for blk_idx in 0..blocks_per_row {
                let block_index = row * blocks_per_row + blk_idx;
                let byte_start = block_index * Q5_0_BLOCK_BYTES;
                let block_slice: &[u8; Q5_0_BLOCK_BYTES] = raw_bytes
                    [byte_start..byte_start + Q5_0_BLOCK_BYTES]
                    .try_into()
                    .map_err(|_| K3Error::TensorBoundsError {
                        tensor_name: weights.name().to_owned(),
                        offset: byte_start,
                        byte_len: Q5_0_BLOCK_BYTES,
                        buffer_len: raw_bytes.len(),
                        reason: "failed to slice Q5_0 block".to_owned(),
                    })?;
                let d = unpack_q5_0_block_i8(block_slice, &mut unpacked_i8);
                if d != 0.0 {
                    let q_bytes = unpacked_i8.map(|v| v.to_ne_bytes()[0]);
                    let input_sub =
                        &input[blk_idx * Q5_0_BLOCK_ELEMENTS..(blk_idx + 1) * Q5_0_BLOCK_ELEMENTS];
                    row_sum += d * dot_i8_32_f32_rvv(&q_bytes, input_sub);
                }
            }
            *out_slot = row_sum;
        }
        Ok(())
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        vec_dot_generic_block::<Q5_0_BLOCK_ELEMENTS, Q5_0_BLOCK_BYTES>(
            QuantType::Q5_0,
            output,
            weights,
            input,
            dequantize_q5_0_block,
        )
    }
}

/// Computes matrix-vector multiplication with `Q4_1` quantized weights.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if quantization is not `Q4_1` or dimensions mismatch.
pub fn vec_dot_q4_1(
    output: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
) -> Result<(), K3Error> {
    vec_dot_generic_block::<Q4_1_BLOCK_ELEMENTS, Q4_1_BLOCK_BYTES>(
        QuantType::Q4_1,
        output,
        weights,
        input,
        dequantize_q4_1_block,
    )
}

/// Computes matrix-vector multiplication with `Q5_1` quantized weights.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if quantization is not `Q5_1` or dimensions mismatch.
pub fn vec_dot_q5_1(
    output: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
) -> Result<(), K3Error> {
    vec_dot_generic_block::<Q5_1_BLOCK_ELEMENTS, Q5_1_BLOCK_BYTES>(
        QuantType::Q5_1,
        output,
        weights,
        input,
        dequantize_q5_1_block,
    )
}

/// Computes matrix-vector multiplication with `Q2_K` quantized weights.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if quantization is not `Q2_K` or dimensions mismatch.
pub fn vec_dot_q2_k(
    output: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
) -> Result<(), K3Error> {
    vec_dot_generic_block::<Q2_K_BLOCK_ELEMENTS, Q2_K_BLOCK_BYTES>(
        QuantType::Q2_K,
        output,
        weights,
        input,
        dequantize_q2_k_block,
    )
}

/// Computes matrix-vector multiplication with `Q3_K` quantized weights.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if quantization is not `Q3_K` or dimensions mismatch.
pub fn vec_dot_q3_k(
    output: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
) -> Result<(), K3Error> {
    vec_dot_generic_block::<Q3_K_BLOCK_ELEMENTS, Q3_K_BLOCK_BYTES>(
        QuantType::Q3_K,
        output,
        weights,
        input,
        dequantize_q3_k_block,
    )
}

/// Computes matrix-vector multiplication with `IQ4_NL` non-linear quantized weights.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if quantization is not `IQ4_NL` or dimensions mismatch.
pub fn vec_dot_iq4_nl(
    output: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
) -> Result<(), K3Error> {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        let (rows, cols) = validate_quant_matvec_dimensions(
            QuantType::IQ4_NL,
            output.len(),
            weights,
            input.len(),
        )?;
        let blocks_per_row = cols / IQ4_NL_BLOCK_ELEMENTS;
        let raw_bytes = weights.as_bytes();
        let mut unpacked_i8 = [0_i8; IQ4_NL_BLOCK_ELEMENTS];

        for (row, out_slot) in output.iter_mut().enumerate().take(rows) {
            let mut row_sum = 0.0_f32;
            for blk_idx in 0..blocks_per_row {
                let block_index = row * blocks_per_row + blk_idx;
                let byte_start = block_index * IQ4_NL_BLOCK_BYTES;
                let block_slice: &[u8; IQ4_NL_BLOCK_BYTES] = raw_bytes
                    [byte_start..byte_start + IQ4_NL_BLOCK_BYTES]
                    .try_into()
                    .map_err(|_| K3Error::TensorBoundsError {
                        tensor_name: weights.name().to_owned(),
                        offset: byte_start,
                        byte_len: IQ4_NL_BLOCK_BYTES,
                        buffer_len: raw_bytes.len(),
                        reason: "failed to slice IQ4_NL block".to_owned(),
                    })?;
                let d = unpack_iq4_nl_block_i8(block_slice, &mut unpacked_i8);
                if d != 0.0 {
                    let q_bytes = unpacked_i8.map(|v| v.to_ne_bytes()[0]);
                    let input_sub = &input
                        [blk_idx * IQ4_NL_BLOCK_ELEMENTS..(blk_idx + 1) * IQ4_NL_BLOCK_ELEMENTS];
                    row_sum += d * dot_i8_32_f32_rvv(&q_bytes, input_sub);
                }
            }
            *out_slot = row_sum;
        }
        Ok(())
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        vec_dot_generic_block::<IQ4_NL_BLOCK_ELEMENTS, IQ4_NL_BLOCK_BYTES>(
            QuantType::IQ4_NL,
            output,
            weights,
            input,
            dequantize_iq4_nl_block,
        )
    }
}

/// Computes matrix-vector multiplication with `BitNet.cpp` `I2_S` (`ggml_type = 36`) weights.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if quantization is not `I2_S` or dimensions mismatch.
pub fn vec_dot_i2_s(
    output: &mut [f32],
    weights: &TensorView<'_>,
    input: &[f32],
) -> Result<(), K3Error> {
    let (rows, cols) =
        validate_quant_matvec_dimensions(QuantType::I2_S, output.len(), weights, input.len())?;
    let blocks_per_row = cols / I2_S_BLOCK_ELEMENTS;
    let raw_bytes = weights.as_bytes();
    let payload_bytes = rows * blocks_per_row * I2_S_BLOCK_BYTES;
    let scale = read_i2_s_trailer_scale(&raw_bytes[payload_bytes..]);
    let mut dequant_buf = [0.0_f32; I2_S_BLOCK_ELEMENTS];

    for (row, out_slot) in output.iter_mut().enumerate().take(rows) {
        if scale == 0.0 {
            *out_slot = 0.0;
            continue;
        }
        let mut row_sum = 0.0_f32;
        for blk_idx in 0..blocks_per_row {
            let block_index = row * blocks_per_row + blk_idx;
            let byte_start = block_index * I2_S_BLOCK_BYTES;
            let block_slice: &[u8; I2_S_BLOCK_BYTES] = raw_bytes
                [byte_start..byte_start + I2_S_BLOCK_BYTES]
                .try_into()
                .map_err(|_| K3Error::TensorBoundsError {
                    tensor_name: weights.name().to_owned(),
                    offset: byte_start,
                    byte_len: I2_S_BLOCK_BYTES,
                    buffer_len: raw_bytes.len(),
                    reason: "failed to slice I2_S block".to_owned(),
                })?;
            dequantize_i2_s_block(block_slice, scale, &mut dequant_buf);
            let input_sub =
                &input[blk_idx * I2_S_BLOCK_ELEMENTS..(blk_idx + 1) * I2_S_BLOCK_ELEMENTS];
            row_sum += dot_f32_dispatch(&dequant_buf, input_sub);
        }
        *out_slot = row_sum;
    }

    Ok(())
}

fn dequantize_packed_blocks<const BLOCK_ELEMS: usize, const BLOCK_BYTES: usize>(
    row_bytes: &[u8],
    out: &mut [f32],
    mut dequant_block: impl FnMut(&[u8; BLOCK_BYTES], &mut [f32; BLOCK_ELEMS]),
) -> Result<(), K3Error> {
    let mut tmp = [0.0_f32; BLOCK_ELEMS];
    for (blk_bytes, out_chunk) in row_bytes
        .chunks_exact(BLOCK_BYTES)
        .zip(out.chunks_exact_mut(BLOCK_ELEMS))
    {
        let block_arr: &[u8; BLOCK_BYTES] =
            blk_bytes
                .try_into()
                .map_err(|_| K3Error::TensorBoundsError {
                    tensor_name: "dequantize_row_bytes".to_owned(),
                    offset: 0,
                    byte_len: blk_bytes.len(),
                    buffer_len: row_bytes.len(),
                    reason: "invalid block byte slice".to_owned(),
                })?;
        dequant_block(block_arr, &mut tmp);
        out_chunk.copy_from_slice(&tmp);
    }
    Ok(())
}

#[expect(
    clippy::too_many_lines,
    reason = "single exhaustive dispatch table across all 17 supported QuantType formats"
)]
fn dequantize_packed_row_bytes(
    tensor_name: &str,
    quant: QuantType,
    row_bytes: &[u8],
    n_cols: usize,
    i2_s_scale: f32,
    out: &mut [f32],
) -> Result<(), K3Error> {
    match quant {
        QuantType::F32 => {
            for (dst, chunk) in out.iter_mut().zip(row_bytes.chunks_exact(4)) {
                *dst = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
        }
        QuantType::F16 | QuantType::BF16 => {
            let decode_half = if quant == QuantType::F16 {
                f16_to_f32_finite
            } else {
                bf16_to_f32_finite
            };
            for (dst, chunk) in out.iter_mut().zip(row_bytes.chunks_exact(2)) {
                *dst = decode_half(u16::from_le_bytes([chunk[0], chunk[1]]));
            }
        }
        QuantType::Q4_0 => dequantize_packed_blocks::<Q4_0_BLOCK_ELEMENTS, Q4_0_BLOCK_BYTES>(
            row_bytes,
            out,
            dequantize_q4_0_block,
        )?,
        QuantType::Q4_1 => dequantize_packed_blocks::<Q4_1_BLOCK_ELEMENTS, Q4_1_BLOCK_BYTES>(
            row_bytes,
            out,
            dequantize_q4_1_block,
        )?,
        QuantType::Q5_0 => dequantize_packed_blocks::<Q5_0_BLOCK_ELEMENTS, Q5_0_BLOCK_BYTES>(
            row_bytes,
            out,
            dequantize_q5_0_block,
        )?,
        QuantType::Q5_1 => dequantize_packed_blocks::<Q5_1_BLOCK_ELEMENTS, Q5_1_BLOCK_BYTES>(
            row_bytes,
            out,
            dequantize_q5_1_block,
        )?,
        QuantType::Q8_0 => dequantize_packed_blocks::<Q8_0_BLOCK_ELEMENTS, Q8_0_BLOCK_BYTES>(
            row_bytes,
            out,
            dequantize_q8_0_block,
        )?,
        QuantType::Q2_K => dequantize_packed_blocks::<Q2_K_BLOCK_ELEMENTS, Q2_K_BLOCK_BYTES>(
            row_bytes,
            out,
            dequantize_q2_k_block,
        )?,
        QuantType::Q3_K => dequantize_packed_blocks::<Q3_K_BLOCK_ELEMENTS, Q3_K_BLOCK_BYTES>(
            row_bytes,
            out,
            dequantize_q3_k_block,
        )?,
        QuantType::Q4_K_M => dequantize_packed_blocks::<
            { crate::Q4_K_M_BLOCK_ELEMENTS },
            { crate::Q4_K_M_BLOCK_BYTES },
        >(row_bytes, out, crate::dequantize_q4_k_m_block)?,
        QuantType::Q5_K => dequantize_packed_blocks::<Q5_K_BLOCK_ELEMENTS, Q5_K_BLOCK_BYTES>(
            row_bytes,
            out,
            dequantize_q5_k_block,
        )?,
        QuantType::Q6_K => dequantize_packed_blocks::<Q6_K_BLOCK_ELEMENTS, Q6_K_BLOCK_BYTES>(
            row_bytes,
            out,
            dequantize_q6_k_block,
        )?,
        QuantType::IQ4_NL => {
            #[cfg(all(target_arch = "riscv64", not(miri)))]
            {
                dequantize_iq4_nl_row_rvv(row_bytes, out)?;
            }
            #[cfg(any(not(target_arch = "riscv64"), miri))]
            {
                dequantize_packed_blocks::<IQ4_NL_BLOCK_ELEMENTS, IQ4_NL_BLOCK_BYTES>(
                    row_bytes,
                    out,
                    dequantize_iq4_nl_block,
                )?;
            }
        }
        QuantType::TQ1_0 => {
            let mut q_i8 = [0_i8; crate::ime::TQ1_0_BLOCK_ELEMENTS];
            dequantize_packed_blocks::<
                { crate::ime::TQ1_0_BLOCK_ELEMENTS },
                { crate::ime::TQ1_0_BLOCK_BYTES },
            >(row_bytes, out, |blk, dst| {
                let d = crate::ime::unpack_tq1_0_block_i8(blk, &mut q_i8);
                scale_i8_into_f32(d, &q_i8, dst);
            })?;
        }
        QuantType::TQ2_0 => {
            let mut q_i8 = [0_i8; crate::ime::TQ2_0_BLOCK_ELEMENTS];
            dequantize_packed_blocks::<
                { crate::ime::TQ2_0_BLOCK_ELEMENTS },
                { crate::ime::TQ2_0_BLOCK_BYTES },
            >(row_bytes, out, |blk, dst| {
                let d = crate::ime::unpack_tq2_0_block_i8(blk, &mut q_i8);
                scale_i8_into_f32(d, &q_i8, dst);
            })?;
        }
        QuantType::I2_S => {
            dequantize_i2_s_row_with_scale(row_bytes, i2_s_scale, out)?;
        }
        QuantType::Q8_1 | QuantType::MXFP4 => {
            let _ = n_cols;
            return Err(K3Error::unsupported_tensor_quant(
                "dequantize",
                tensor_name,
                quant,
            ));
        }
    }
    Ok(())
}

/// Zero-allocation single-row byte slice dequantizer across `F32`/`F16`/`BF16`, `Q4_0`..`Q8_0`,
/// `Q2_K`..`Q6_K`, `IQ4_NL`, `TQ1_0`/`TQ2_0`, and `I2_S` (expecting the 32-byte scale trailer).
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] when `n_cols` is zero, `out.len() != n_cols`, `n_cols`
/// is not block-aligned, or `row_bytes.len()` mismatches exact `(n_cols / block_elements) *
/// block_bytes + scale_trailer_bytes()`. Returns [`K3Error::CapabilityMismatch`] on unsupported
/// quant formats (`Q8_1`, `MXFP4`).
pub fn dequantize_row_bytes(
    quant: QuantType,
    row_bytes: &[u8],
    n_cols: usize,
    out: &mut [f32],
) -> Result<(), K3Error> {
    if n_cols == 0 || out.len() != n_cols {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "dequantize_row_bytes".to_owned(),
            offset: 0,
            byte_len: row_bytes.len(),
            buffer_len: out.len(),
            reason: format!(
                "invalid row width n_cols ({n_cols}) or output slice length ({})",
                out.len()
            ),
        });
    }
    if matches!(quant, QuantType::Q8_1 | QuantType::MXFP4) {
        return Err(K3Error::unsupported_tensor_quant(
            "dequantize",
            "dequantize_row_bytes",
            quant,
        ));
    }
    let block_elems = quant.block_elements();
    if n_cols % block_elems != 0 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "dequantize_row_bytes".to_owned(),
            offset: 0,
            byte_len: row_bytes.len(),
            buffer_len: out.len(),
            reason: format!(
                "n_cols ({n_cols}) is not divisible by {quant:?} block size ({block_elems})"
            ),
        });
    }
    let packed_bytes = (n_cols / block_elems)
        .checked_mul(quant.block_bytes())
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: "dequantize_row_bytes".to_owned(),
            offset: 0,
            byte_len: row_bytes.len(),
            buffer_len: out.len(),
            reason: "row byte length overflow".to_owned(),
        })?;
    let expected_bytes = packed_bytes
        .checked_add(quant.scale_trailer_bytes())
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: "dequantize_row_bytes".to_owned(),
            offset: 0,
            byte_len: row_bytes.len(),
            buffer_len: out.len(),
            reason: "row byte + trailer length overflow".to_owned(),
        })?;
    if row_bytes.len() != expected_bytes {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "dequantize_row_bytes".to_owned(),
            offset: 0,
            byte_len: row_bytes.len(),
            buffer_len: expected_bytes,
            reason: format!(
                "{quant:?} row byte length {} != expected {expected_bytes}",
                row_bytes.len()
            ),
        });
    }
    let i2_s_scale = if quant == QuantType::I2_S {
        read_i2_s_trailer_scale(&row_bytes[packed_bytes..])
    } else {
        1.0_f32
    };
    dequantize_packed_row_bytes(
        "dequantize_row_bytes",
        quant,
        &row_bytes[..packed_bytes],
        n_cols,
        i2_s_scale,
        out,
    )
}

/// Zero-copy, zero-allocation single-row extraction from `view` into `out`, supporting
/// `F32`/`F16`/`BF16`, `Q4_0`..`Q8_0`, `Q2_K`..`Q6_K`, `IQ4_NL`, `TQ1_0`/`TQ2_0`, and `I2_S`.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] when dimensions, block alignment, `row_idx`, or backing
/// buffer bounds fail validation, and [`K3Error::CapabilityMismatch`] on unsupported quant types.
pub fn dequantize_tensor_row(
    view: &TensorView<'_>,
    row_idx: usize,
    out: &mut [f32],
) -> Result<(), K3Error> {
    let quant = view.quant();
    if matches!(quant, QuantType::Q8_1 | QuantType::MXFP4) {
        return Err(K3Error::unsupported_tensor_quant(
            "dequantize",
            view.name(),
            quant,
        ));
    }
    let cols = out.len();
    let num_rows = validate_token_row_geometry(view, row_idx, cols)?;
    let raw = view.as_bytes();
    let block_elems = quant.block_elements();
    if cols % block_elems != 0 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: view.name().to_owned(),
            offset: view.offset(),
            byte_len: view.byte_len(),
            buffer_len: raw.len(),
            reason: format!(
                "row width {cols} is not divisible by {quant:?} block size {block_elems}"
            ),
        });
    }
    let row_bytes = (cols / block_elems)
        .checked_mul(quant.block_bytes())
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: view.name().to_owned(),
            offset: view.offset(),
            byte_len: view.byte_len(),
            buffer_len: raw.len(),
            reason: "row byte length overflow".to_owned(),
        })?;
    let payload_bytes =
        num_rows
            .checked_mul(row_bytes)
            .ok_or_else(|| K3Error::TensorBoundsError {
                tensor_name: view.name().to_owned(),
                offset: view.offset(),
                byte_len: view.byte_len(),
                buffer_len: raw.len(),
                reason: "tensor payload byte length overflow".to_owned(),
            })?;
    let total_required = payload_bytes
        .checked_add(quant.scale_trailer_bytes())
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: view.name().to_owned(),
            offset: view.offset(),
            byte_len: view.byte_len(),
            buffer_len: raw.len(),
            reason: "tensor total byte length overflow".to_owned(),
        })?;
    if raw.len() < total_required {
        return Err(K3Error::TensorBoundsError {
            tensor_name: view.name().to_owned(),
            offset: view.offset(),
            byte_len: view.byte_len(),
            buffer_len: raw.len(),
            reason: format!(
                "{quant:?} tensor byte length {} < required {total_required}",
                raw.len()
            ),
        });
    }
    let start = row_idx * row_bytes;
    let end = start + row_bytes;
    let i2_s_scale = if quant == QuantType::I2_S {
        read_i2_s_trailer_scale(&raw[payload_bytes..])
    } else {
        1.0_f32
    };
    dequantize_packed_row_bytes(view.name(), quant, &raw[start..end], cols, i2_s_scale, out)
}

/// Zero-copy single-row dequantization directly from a memory-mapped embedding table [`TensorView`].
///
/// Decodes only the blocks corresponding to `row_idx` without materializing the full vocabulary table.
/// Delegates to [`dequantize_tensor_row`] across all supported GGUF formats.
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
    dequantize_tensor_row(table, row_idx, out)
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

    #[test]
    fn test_wave2_block_dequant_q4_1_q5_0_q5_1_q2_k_q3_k_iq4_nl_i2_s() {
        // 1. Q4_1: d = 0.5, m = -0.25, nibble 0 -> -0.25, nibble 15 -> 7.25
        let mut q4_1_blk = [0_u8; Q4_1_BLOCK_BYTES];
        q4_1_blk[0..2].copy_from_slice(&0x3800_u16.to_le_bytes());
        q4_1_blk[2..4].copy_from_slice(&0xB400_u16.to_le_bytes());
        for j in 0..16 {
            q4_1_blk[4 + j] = 0xF0; // lo = 0, hi = 15
        }
        let mut out32 = [0.0_f32; 32];
        dequantize_q4_1_block(&q4_1_blk, &mut out32);
        assert!(out32[..16].iter().all(|&v| (v - (-0.25)).abs() < 1e-6));
        assert!(out32[16..].iter().all(|&v| (v - 7.25).abs() < 1e-6));

        // Zero scale + non-zero min must preserve min
        let mut q4_1_zero_d = [0_u8; Q4_1_BLOCK_BYTES];
        q4_1_zero_d[2..4].copy_from_slice(&0xB800_u16.to_le_bytes()); // m = -0.5
        dequantize_q4_1_block(&q4_1_zero_d, &mut out32);
        assert!(out32.iter().all(|&v| (v - (-0.5)).abs() < 1e-6));

        // 2. Q5_0: d = 0.5, qh sets bit j=0 (hi=+16) and clears j=16; lo=1 (val 17-16=+1), hi=2 (2-16=-14)
        let mut q5_0_blk = [0_u8; Q5_0_BLOCK_BYTES];
        q5_0_blk[0..2].copy_from_slice(&0x3800_u16.to_le_bytes());
        q5_0_blk[2..6].copy_from_slice(&0x0000_0001_u32.to_le_bytes());
        q5_0_blk[6] = 0x21;
        dequantize_q5_0_block(&q5_0_blk, &mut out32);
        assert!((out32[0] - 0.5).abs() < 1e-6);
        assert!((out32[16] - (-7.0)).abs() < 1e-6);

        // 3. Q5_1: d = 0.5, m = 1.0, qh bit 0 + bit 16 set -> element 0 has hi=16, element 16 has hi=16
        let mut q5_1_blk = [0_u8; Q5_1_BLOCK_BYTES];
        q5_1_blk[0..2].copy_from_slice(&0x3800_u16.to_le_bytes());
        q5_1_blk[2..4].copy_from_slice(&0x3C00_u16.to_le_bytes());
        q5_1_blk[4..8].copy_from_slice(&0x0001_0001_u32.to_le_bytes());
        q5_1_blk[8] = 0x32;
        dequantize_q5_1_block(&q5_1_blk, &mut out32);
        assert!((out32[0] - 10.0).abs() < 1e-6); // 0.5 * (2 + 16) + 1.0
        assert!((out32[16] - 10.5).abs() < 1e-6); // 0.5 * (3 + 16) + 1.0

        // 4. Q2_K: sub-block 0 sc=2, m=1, q=3 -> d(1.0)*2*3 - dmin(0.5)*1 = 5.5
        let mut q2k_blk = [0_u8; Q2_K_BLOCK_BYTES];
        q2k_blk[0] = (1 << 4) | 2;
        q2k_blk[16] = 0x03; // elem 0 shift 0 = 3
        q2k_blk[80..82].copy_from_slice(&0x3C00_u16.to_le_bytes()); // d = 1.0
        q2k_blk[82..84].copy_from_slice(&0x3800_u16.to_le_bytes()); // dmin = 0.5
        let mut out256 = [0.0_f32; 256];
        dequantize_q2_k_block(&q2k_blk, &mut out256);
        assert!((out256[0] - 5.5).abs() < 1e-6);

        // 5. Q3_K: hmask=0 (all high bits inverted -> subtract 4 from 0..3), all raw scale=33 => dl = d * (33-32)
        let mut q3k_blk = [0_u8; Q3_K_BLOCK_BYTES];
        // Pack 6-bit scale 33 (= 0b10_0001) across scales[0..12]: low4 = 1 -> 0x11, top2 = 2
        q3k_blk[96..104].fill(0x11);
        q3k_blk[104..108].fill(0xAA);
        q3k_blk[32] = 0x03; // q0 = 3, with hmask bit 0 = 0 -> (3 - 4) = -1
        q3k_blk[108..110].copy_from_slice(&0x3800_u16.to_le_bytes()); // d = 0.5
        dequantize_q3_k_block(&q3k_blk, &mut out256);
        assert!((out256[0] - (-0.5)).abs() < 1e-6);

        // 6. IQ4_NL: nibble 0 -> -127, nibble 8 -> 1, nibble 15 -> 113
        let mut iq4_blk = [0_u8; IQ4_NL_BLOCK_BYTES];
        iq4_blk[0..2].copy_from_slice(&0x3C00_u16.to_le_bytes()); // d = 1.0
        iq4_blk[2] = 0xF0; // j=0 lo=0 (-127), j=16 hi=15 (113)
        iq4_blk[3] = 0x08; // j=1 lo=8 (1), j=17 hi=0 (-127)
        dequantize_iq4_nl_block(&iq4_blk, &mut out32);
        assert!((out32[0] - (-127.0)).abs() < 1e-6);
        assert!((out32[16] - 113.0).abs() < 1e-6);
        assert!((out32[1] - 1.0).abs() < 1e-6);
        assert!((out32[17] - (-127.0)).abs() < 1e-6);

        // 7. I2_S: byte (0<<6)|(1<<4)|(2<<2)|0 => {-1, 0, +1, -1} across 4 interleaved groups
        let mut i2s_blk = [0x55_u8; I2_S_BLOCK_BYTES]; // all 1 => 0
        i2s_blk[0] = (1 << 4) | (2 << 2);
        let mut out128 = [0.0_f32; 128];
        dequantize_i2_s_block(&i2s_blk, 0.25, &mut out128);
        assert!((out128[0] - (-0.25)).abs() < 1e-6);
        assert!(out128[32].abs() < 1e-6);
        assert!((out128[64] - 0.25).abs() < 1e-6);
        assert!((out128[96] - (-0.25)).abs() < 1e-6);
    }

    #[test]
    fn test_wave2_vec_dot_and_dequantize_q5_0() {
        let mut q5_0_row = [0_u8; 2 * Q5_0_BLOCK_BYTES];
        q5_0_row[0..2].copy_from_slice(&0x3800_u16.to_le_bytes()); // d = 0.5
        q5_0_row[2..6].copy_from_slice(&0xAAAA_5555_u32.to_le_bytes());
        for i in 0_u8..16 {
            q5_0_row[6 + usize::from(i)] = (i & 0x0F) | (((15 - i) & 0x0F) << 4);
        }
        q5_0_row[22..24].copy_from_slice(&0x3C00_u16.to_le_bytes()); // d = 1.0
        q5_0_row[24..28].copy_from_slice(&0x0F0F_F0F0_u32.to_le_bytes());
        q5_0_row[28..44].fill(0x5A);

        let mut dequant_row = [0.0_f32; 64];
        dequantize_q5_0(&q5_0_row, &mut dequant_row).unwrap();
        let x: Vec<f32> = (0..64)
            .map(|i| {
                #[expect(clippy::cast_precision_loss, reason = "small test index")]
                {
                    (i as f32 - 31.0) * 0.0625
                }
            })
            .collect();
        let ref_dot: f32 = dequant_row.iter().zip(&x).map(|(w, a)| w * a).sum();

        let view_q5 = TensorView::from_bytes("w_q5", QuantType::Q5_0, &[64, 1], 0, &q5_0_row)
            .expect("Q5_0 view");
        let mut dot_out = [0.0_f32; 1];
        vec_dot_q5_0(&mut dot_out, &view_q5, &x).unwrap();
        assert!((dot_out[0] - ref_dot).abs() < 1e-4);

        // Quantized Q8_0 activation dot helper vs ref_dot
        let mut q8_act = [0_u8; 2 * Q8_0_BLOCK_BYTES];
        for b in 0..2 {
            let blk = &mut q8_act[b * Q8_0_BLOCK_BYTES..(b + 1) * Q8_0_BLOCK_BYTES];
            blk[0..2].copy_from_slice(&0x3C00_u16.to_le_bytes()); // d = 1.0
            for i in 0..32 {
                let qi = i8::try_from(b * 32 + i).unwrap() - 31;
                blk[2 + i] = qi.to_ne_bytes()[0];
            }
        }
        let q8_f32: Vec<f32> = (0..64)
            .map(|i| {
                #[expect(clippy::cast_precision_loss, reason = "small test index")]
                {
                    (i as f32) - 31.0
                }
            })
            .collect();
        let exact_q8_dot: f32 = dequant_row.iter().zip(&q8_f32).map(|(w, a)| w * a).sum();
        let dot_q5_q8 = vec_dot_q5_0_q8_0(&q5_0_row, &q8_act, 64).unwrap();
        assert!((dot_q5_q8 - exact_q8_dot).abs() < 1e-4);

        // Bounds checks fail closed
        assert!(matches!(
            dequantize_q5_0(&q5_0_row[..10], &mut dequant_row),
            Err(K3Error::TensorBoundsError { .. })
        ));
        assert!(matches!(
            vec_dot_q5_0_q8_0(&q5_0_row, &q8_act, 48),
            Err(K3Error::TensorBoundsError { .. })
        ));
    }

    #[test]
    fn test_dequantize_tensor_row_and_row_bytes_all_formats_and_errors() {
        let rows = 3_usize;
        let formats = [
            QuantType::F32,
            QuantType::F16,
            QuantType::BF16,
            QuantType::Q8_0,
            QuantType::Q4_0,
            QuantType::Q4_1,
            QuantType::Q5_0,
            QuantType::Q5_1,
            QuantType::Q2_K,
            QuantType::Q3_K,
            QuantType::Q4_K_M,
            QuantType::Q5_K,
            QuantType::Q6_K,
            QuantType::IQ4_NL,
            QuantType::TQ1_0,
            QuantType::TQ2_0,
            QuantType::I2_S,
        ];

        for &quant in &formats {
            let cols = quant.block_elements().max(32);
            let row_stride = (cols / quant.block_elements()) * quant.block_bytes();
            let payload_len = rows * row_stride;
            let mut raw = vec![0_u8; payload_len + quant.scale_trailer_bytes()];
            for (idx, byte) in raw[..payload_len].iter_mut().enumerate() {
                #[expect(clippy::cast_possible_truncation, reason = "modulo 256 byte pattern")]
                let pattern = ((idx * 17 + 11) % 241) as u8;
                *byte = match quant {
                    QuantType::F32 => [0x00, 0x00, 0x80, 0x3E][idx % 4], // 0.25_f32
                    QuantType::F16 => [0x00, 0x38][idx % 2],             // 0.5_f16
                    QuantType::BF16 => [0x80, 0x3E][idx % 2],            // 0.25_bf16
                    QuantType::TQ1_0 | QuantType::TQ2_0 if idx % row_stride < 2 => {
                        [0x00, 0x38][idx % 2]
                    }
                    QuantType::TQ1_0 => pattern % 243,
                    _ => pattern,
                };
            }
            if quant == QuantType::I2_S {
                raw[payload_len..payload_len + 4].copy_from_slice(&0.5_f32.to_le_bytes());
            }

            let dims = [cols, rows];
            let view = TensorView::from_bytes("embd", quant, &dims, 0, &raw)
                .unwrap_or_else(|e| panic!("view for {quant:?}: {e}"));
            let mut out_tensor = vec![0.0_f32; cols];
            let mut out_row_bytes = vec![0.0_f32; cols];
            for r in [0_usize, 1, rows - 1] {
                dequantize_tensor_row(&view, r, &mut out_tensor)
                    .unwrap_or_else(|e| panic!("tensor row {quant:?} r={r}: {e}"));
                assert!(
                    out_tensor.iter().all(|v| v.is_finite()),
                    "non-finite output for {quant:?}"
                );
                if quant != QuantType::I2_S {
                    dequantize_row_bytes(
                        quant,
                        &raw[r * row_stride..(r + 1) * row_stride],
                        cols,
                        &mut out_row_bytes,
                    )
                    .unwrap();
                    assert_eq!(out_tensor, out_row_bytes);
                }
            }

            // Reject OOB row and short out buffer
            assert!(matches!(
                dequantize_tensor_row(&view, rows, &mut out_tensor),
                Err(K3Error::TensorBoundsError { .. })
            ));
            let mut short = vec![0.0_f32; cols - 1];
            assert!(matches!(
                dequantize_tensor_row(&view, 0, &mut short),
                Err(K3Error::TensorBoundsError { .. })
            ));
        }

        // 1D row 0 succeeds while row 1 and misaligned outputs fail closed cleanly
        let raw_q8 = vec![0_u8; Q8_0_BLOCK_BYTES];
        let view_1d = TensorView::from_bytes("vec1d", QuantType::Q8_0, &[32], 0, &raw_q8).unwrap();
        let mut out32 = [0.0_f32; 32];
        dequantize_tensor_row(&view_1d, 0, &mut out32).unwrap();
        assert!(matches!(
            dequantize_tensor_row(&view_1d, 1, &mut out32),
            Err(K3Error::TensorBoundsError { .. })
        ));
        let mut out16 = [0.0_f32; 16];
        assert!(matches!(
            dequantize_tensor_row(&view_1d, 0, &mut out16),
            Err(K3Error::TensorBoundsError { .. })
        ));

        for &unsup in &[QuantType::Q8_1, QuantType::MXFP4] {
            let raw = vec![0_u8; unsup.block_bytes()];
            assert!(matches!(
                dequantize_row_bytes(unsup, &raw, 32, &mut out32),
                Err(K3Error::CapabilityMismatch { .. })
            ));
        }
    }

    #[test]
    fn test_iq4_nl_all_lanes_sentinel_and_vlen128_geometry() {
        let rvv_vlmax =
            |vlen_bits: usize, sew_bits: usize, lmul: usize| (vlen_bits / sew_bits) * lmul;
        // Under VLEN = 128, e32/m2 gives VLMAX = 8 (insufficient for 16 elements), whereas e32/m4
        // yields VLMAX >= 16 across all RVV 1.0 VLEN >= 128 implementations.
        assert_eq!(rvv_vlmax(128, 32, 2), 8);
        for vlen in [128_usize, 256, 512, 1024] {
            assert!(rvv_vlmax(vlen, 8, 1) >= 16);
            assert!(rvv_vlmax(vlen, 32, 4) >= 16);
        }

        // Four blocks (128 elements): cover distinct nibbles across [0..8], [8..16], [16..24],
        // and [24..32], plus positive, negative, +0.0, -0.0, and non-finite (clamped) scales.
        let scales_f16: [u16; 4] = [0x3C00, 0xB800, 0x8000, 0x7E00];
        let mut raw = [0_u8; 4 * IQ4_NL_BLOCK_BYTES];
        let mut oracle = [-999.25_f32; 4 * IQ4_NL_BLOCK_ELEMENTS];
        for (b, &scale_bits) in scales_f16.iter().enumerate() {
            let base = b * IQ4_NL_BLOCK_BYTES;
            raw[base..base + 2].copy_from_slice(&scale_bits.to_le_bytes());
            for j in 0_u8..16 {
                let lo = (j.wrapping_add(u8::try_from(b * 3).unwrap())) & 0x0F;
                let hi = (15_u8.wrapping_sub(j).wrapping_add(u8::try_from(b).unwrap())) & 0x0F;
                raw[base + 2 + usize::from(j)] = (hi << 4) | lo;
            }
            let blk: &[u8; IQ4_NL_BLOCK_BYTES] = raw[base..base + IQ4_NL_BLOCK_BYTES]
                .try_into()
                .expect("18-byte IQ4_NL block");
            let dst: &mut [f32; IQ4_NL_BLOCK_ELEMENTS] = (&mut oracle
                [b * IQ4_NL_BLOCK_ELEMENTS..(b + 1) * IQ4_NL_BLOCK_ELEMENTS])
                .try_into()
                .expect("32-element output slice");
            dequantize_iq4_nl_block(blk, dst);
        }

        let mut out = [-999.25_f32; 4 * IQ4_NL_BLOCK_ELEMENTS];
        dequantize_row_bytes(QuantType::IQ4_NL, &raw, 4 * IQ4_NL_BLOCK_ELEMENTS, &mut out)
            .expect("IQ4_NL sentinel-prefilled row");
        for (idx, (&actual, &expected)) in out.iter().zip(&oracle).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "IQ4_NL lane {idx} mismatch"
            );
        }
    }

    #[cfg(all(target_arch = "riscv64", not(miri)))]
    #[test]
    fn test_iq4_nl_row_rvv_vlen128_contract_and_bit_exact_parity() {
        let mut vl: usize;
        // SAFETY: `vsetivli` updates VL/VTYPE CSRs without touching memory.
        unsafe {
            core::arch::asm!(
                "vsetivli {vl}, 16, e32, m4, ta, ma",
                vl = out(reg) vl,
                options(nomem, nostack),
            );
        }
        assert_eq!(vl, 16, "e32/m4 must grant vl=16 across all VLEN >= 128");

        let scales_f16: [u16; 3] = [0x3800, 0xBC00, 0x8000];
        let mut raw = [0_u8; 3 * IQ4_NL_BLOCK_BYTES];
        let mut oracle = [0.0_f32; 3 * IQ4_NL_BLOCK_ELEMENTS];
        for (b, &scale_bits) in scales_f16.iter().enumerate() {
            let base = b * IQ4_NL_BLOCK_BYTES;
            raw[base..base + 2].copy_from_slice(&scale_bits.to_le_bytes());
            for j in 0_u8..16 {
                let lo = (j + u8::try_from(b).unwrap()) & 0x0F;
                let hi = (15_u8 - j) & 0x0F;
                raw[base + 2 + usize::from(j)] = (hi << 4) | lo;
            }
            let blk: &[u8; IQ4_NL_BLOCK_BYTES] = raw[base..base + IQ4_NL_BLOCK_BYTES]
                .try_into()
                .expect("18-byte IQ4_NL block");
            let dst: &mut [f32; IQ4_NL_BLOCK_ELEMENTS] = (&mut oracle
                [b * IQ4_NL_BLOCK_ELEMENTS..(b + 1) * IQ4_NL_BLOCK_ELEMENTS])
                .try_into()
                .expect("32-element output slice");
            dequantize_iq4_nl_block(blk, dst);
        }

        let mut out = [-1234.5_f32; 3 * IQ4_NL_BLOCK_ELEMENTS];
        dequantize_iq4_nl_row_rvv(&raw, &mut out).expect("RVV IQ4_NL dequantize succeeds");
        for (idx, (&actual, &expected)) in out.iter().zip(&oracle).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "IQ4_NL RVV lane {idx} mismatch"
            );
        }
    }
}
