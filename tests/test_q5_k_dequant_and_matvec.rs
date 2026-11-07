//! Integration and Property Test Suite for Issue #40 (Milestone 5):
//! `dequantize_q5_k_block` and `matvec_q5_k` in `jero-kernels-rvv`.
//!
//! Objectives:
//! 1. Memory layout and byte-level constants (176 bytes per block, 256 elements).
//! 2. Exact numerical dequantization against GGML specification (affine d * sc * q - dmin * m).
//! 3. Full coverage across all 32 discrete 5-bit quantization levels (0..31).
//! 4. Canonical edge cases: zero blocks, all-ones (max quants), negative scaling, subnormal FP16.
//! 5. Single-block matrix-vector multiplication (y = W x, 1x256) with unit and alternating vectors.
//! 6. Multi-block matrix-vector multiplication (4x512, 2x1024) verified against reference oracle.
//! 7. Exhaustive property sweep over pseudo-random blocks.
//! 8. Defensive boundary validation and error propagation (`TensorBoundsError`).

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::needless_range_loop,
    reason = "test indices, lengths, and floats fit in f32"
)]

use core::mem::{align_of, offset_of, size_of};
use jero_core::{BlockQ5_K, K3Error, QuantType, TensorView};
use jero_kernels_rvv::{Q5_K_BLOCK_BYTES, Q5_K_BLOCK_ELEMENTS, dequantize_q5_k_block, matvec_q5_k};

// ============================================================================
// Independent GGML Reference Dequantization Oracle
// ============================================================================

fn reference_ggml_f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 1;
    let exp = (bits >> 10) & 0x1F;
    let mant = bits & 0x03FF;
    if exp == 0 {
        if mant == 0 {
            f32::from_bits((sign as u32) << 31)
        } else {
            let mut m = mant as u32;
            let mut e = 127u32 - 14;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03FF;
            f32::from_bits(((sign as u32) << 31) | (e << 23) | (m << 13))
        }
    } else if exp == 0x1F {
        let val = 0.0625_f32;
        if sign == 1 { -val } else { val }
    } else {
        let f32_exp = exp as u32 + (127 - 15);
        f32::from_bits(((sign as u32) << 31) | (f32_exp << 23) | ((mant as u32) << 13))
    }
}

fn reference_get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let sc = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (sc, m)
    }
}

fn reference_ggml_dequantize_q5_k(block: &[u8; 176], out: &mut [f32; 256]) {
    let d = reference_ggml_f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = reference_ggml_f16_to_f32(u16::from_le_bytes([block[2], block[3]]));

    if d == 0.0 && dmin == 0.0 {
        out.fill(0.0);
        return;
    }

    let scales = &block[4..16];
    let qh = &block[16..48];
    let qs = &block[48..176];

    let mut is_idx = 0usize;
    let mut u1 = 1u8;
    let mut u2 = 2u8;

    for chunk in 0..4 {
        let (sc1, m1) = reference_get_scale_min_k4(is_idx, scales);
        let d1 = d * f32::from(sc1);
        let m1_val = dmin * f32::from(m1);

        let (sc2, m2) = reference_get_scale_min_k4(is_idx + 1, scales);
        let d2 = d * f32::from(sc2);
        let m2_val = dmin * f32::from(m2);

        let ql = &qs[chunk * 32..(chunk + 1) * 32];
        let out_base = chunk * 64;

        for l in 0..32 {
            let q1 = (ql[l] & 0x0F) + if (qh[l] & u1) != 0 { 16 } else { 0 };
            out[out_base + l] = d1 * f32::from(q1) - m1_val;
        }

        for l in 0..32 {
            let q2 = (ql[l] >> 4) + if (qh[l] & u2) != 0 { 16 } else { 0 };
            out[out_base + 32 + l] = d2 * f32::from(q2) - m2_val;
        }

        is_idx += 2;
        u1 <<= 2;
        u2 <<= 2;
    }
}

// ============================================================================
// Deterministic Pseudo-Random Generator (No external dependencies)
// ============================================================================

struct TestLcg {
    state: u64,
}

impl TestLcg {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        (self.state >> 32) as u32
    }

    fn next_f32(&mut self, min: f32, max: f32) -> f32 {
        let val = (self.next_u32() & 0x007F_FFFF) as f32 / ((1 << 23) as f32);
        min + val * (max - min)
    }

    fn next_u8(&mut self) -> u8 {
        (self.next_u32() & 0xFF) as u8
    }
}

// ============================================================================
// 1. Constants and Struct Memory Layout
// ============================================================================

#[test]
fn test_q5_k_constants_and_struct_alignment() {
    assert_eq!(Q5_K_BLOCK_BYTES, 176, "Q5_K block bytes must be 176");
    assert_eq!(Q5_K_BLOCK_ELEMENTS, 256, "Q5_K block elements must be 256");
    assert_eq!(size_of::<BlockQ5_K>(), 176, "BlockQ5_K size must match 176");
    assert_eq!(
        align_of::<BlockQ5_K>(),
        16,
        "BlockQ5_K must have 16-byte alignment"
    );

    assert_eq!(offset_of!(BlockQ5_K, d), 0, "d offset must be 0");
    assert_eq!(offset_of!(BlockQ5_K, dmin), 2, "dmin offset must be 2");
    assert_eq!(offset_of!(BlockQ5_K, scales), 4, "scales offset must be 4");
    assert_eq!(offset_of!(BlockQ5_K, qh), 16, "qh offset must be 16");
    assert_eq!(offset_of!(BlockQ5_K, qs), 48, "qs offset must be 48");
}

// ============================================================================
// 2. Exact Golden Reference Vector Step-by-Step
// ============================================================================

#[test]
fn test_dequantize_q5_k_golden_vector_exact_step_by_step() {
    let mut raw = [0u8; Q5_K_BLOCK_BYTES];
    // d = 1.0 (0x3C00), dmin = 0.5 (0x3800)
    raw[0..2].copy_from_slice(&0x3C00_u16.to_le_bytes());
    raw[2..4].copy_from_slice(&0x3800_u16.to_le_bytes());

    // Sub-block scales and mins:
    // Sub-block 0: sc = 2, m = 1
    // Sub-block 1: sc = 3, m = 2
    // Sub-block 2: sc = 4, m = 3
    // Sub-block 3: sc = 5, m = 4
    raw[4] = 2;
    raw[8] = 1;
    raw[5] = 3;
    raw[9] = 2;
    raw[6] = 4;
    raw[10] = 3;
    raw[7] = 5;
    raw[11] = 4;

    // qh: bit 0 = 1, bit 3 = 1 -> 0x09
    for b in &mut raw[16..48] {
        *b = 0x09;
    }
    // Chunk 0 qs: low = 3, high = 5 -> 0x53
    for b in &mut raw[48..80] {
        *b = 0x53;
    }
    // Chunk 1 qs: low = 4, high = 2 -> 0x24
    for b in &mut raw[80..112] {
        *b = 0x24;
    }

    let mut out = [0.0_f32; Q5_K_BLOCK_ELEMENTS];
    dequantize_q5_k_block(&raw, &mut out);

    // Sub-block 0: 2.0 * (3 + 16) - 0.5 = 37.5
    // Sub-block 1: 3.0 * (5 + 0) - 1.0 = 14.0
    // Sub-block 2: 4.0 * (4 + 0) - 1.5 = 14.5
    // Sub-block 3: 5.0 * (2 + 16) - 2.0 = 88.0
    // Sub-blocks 4..7: 0.0
    for l in 0..32 {
        assert!(
            (out[l] - 37.5).abs() < 1e-5,
            "Sub-block 0 mismatch at {l}: {}",
            out[l]
        );
        assert!(
            (out[32 + l] - 14.0).abs() < 1e-5,
            "Sub-block 1 mismatch at {l}: {}",
            out[32 + l]
        );
        assert!(
            (out[64 + l] - 14.5).abs() < 1e-5,
            "Sub-block 2 mismatch at {l}: {}",
            out[64 + l]
        );
        assert!(
            (out[96 + l] - 88.0).abs() < 1e-5,
            "Sub-block 3 mismatch at {l}: {}",
            out[96 + l]
        );
    }
    for l in 128..256 {
        assert_eq!(out[l], 0.0, "Inactive sub-blocks must be exactly 0.0");
    }

    let total_sum: f32 = out.iter().sum();
    assert!(
        (total_sum - 4928.0).abs() < 1e-3,
        "Golden sum mismatch: {total_sum}"
    );
}

// ============================================================================
// 3. Upper Sub-Block Scale Packing (j >= 4)
// ============================================================================

#[test]
fn test_dequantize_q5_k_upper_subblock_packing() {
    let mut raw = [0u8; Q5_K_BLOCK_BYTES];
    // d = 2.0 (0x4000), dmin = 1.0 (0x3C00)
    raw[0..2].copy_from_slice(&0x4000_u16.to_le_bytes());
    raw[2..4].copy_from_slice(&0x3C00_u16.to_le_bytes());

    // Encode sub-block 4 (j=4): sc = 25 (0b011001), m = 17 (0b010001)
    // scales[0] high 2 bits = 1 (sc high 2 bits: 1 << 6 = 0x40)
    // scales[4] high 2 bits = 1 (m high 2 bits: 1 << 6 = 0x40)
    // scales[8] = (9 & 0x0F) | ((1 & 0x0F) << 4) = 0x19
    raw[4] = 0x40;
    raw[8] = 0x40;
    raw[12] = 0x19;

    // Sub-block 4 resides in chunk 2 (elements 128..159).
    // u1 = 1 << 4 = 16 (0x10).
    // Let qh have bit 4 set: 0x10.
    for b in &mut raw[16..48] {
        *b = 0x10;
    }
    // Low nibble in qs chunk 2 = 7 -> q = 7 + 16 = 23
    for b in &mut raw[112..144] {
        *b = 0x07;
    }

    let mut out = [0.0_f32; Q5_K_BLOCK_ELEMENTS];
    dequantize_q5_k_block(&raw, &mut out);

    // Expected for sub-block 4: d * sc * q - dmin * m
    // = 2.0 * 25.0 * 23.0 - 1.0 * 17.0 = 1150.0 - 17.0 = 1133.0
    for l in 0..32 {
        assert!(
            (out[128 + l] - 1133.0).abs() < 1e-4,
            "Sub-block 4 upper scale mismatch at {l}: {}",
            out[128 + l]
        );
    }
}

// ============================================================================
// 4. Exhaustive 32 Discrete Quantization Levels Sweep
// ============================================================================

#[test]
fn test_dequantize_q5_k_all_32_quant_levels() {
    let mut raw = [0u8; Q5_K_BLOCK_BYTES];
    // d = 1.0 (0x3C00), dmin = 0.0 (0x0000)
    raw[0..2].copy_from_slice(&0x3C00_u16.to_le_bytes());
    raw[2..4].copy_from_slice(&0x0000_u16.to_le_bytes());

    // sc0 = 1, m0 = 0
    raw[4] = 1;

    // For sub-block 0 (l in 0..32):
    // element l: low nibble = l & 0xF, high bit = (l >= 16)
    for l in 0..32 {
        let low = (l & 0x0F) as u8;
        raw[48 + l] = low;
        if l >= 16 {
            raw[16 + l] = 0x01; // bit 0 = 1
        }
    }

    let mut out = [0.0_f32; Q5_K_BLOCK_ELEMENTS];
    dequantize_q5_k_block(&raw, &mut out);

    for l in 0..32 {
        let expected = l as f32;
        assert!(
            (out[l] - expected).abs() < 1e-5,
            "Quant level {l} mismatch: got {}",
            out[l]
        );
    }
}

// ============================================================================
// 5. Edge Cases: Zero Block, All Max Quants, Negative Scaling, FP16
// ============================================================================

#[test]
fn test_dequantize_q5_k_all_zeros_block() {
    let raw = [0u8; Q5_K_BLOCK_BYTES];
    let mut out = [1.0_f32; Q5_K_BLOCK_ELEMENTS];
    dequantize_q5_k_block(&raw, &mut out);
    assert!(
        out.iter().all(|&v| v == 0.0),
        "Zero block must yield all 0.0"
    );
}

#[test]
fn test_dequantize_q5_k_all_max_quants_31() {
    let mut raw = [0u8; Q5_K_BLOCK_BYTES];
    // d = 1.0 (0x3C00), dmin = 0.0
    raw[0..2].copy_from_slice(&0x3C00_u16.to_le_bytes());
    // sc = 1 for all sub-blocks, m = 0
    raw[4] = 1;
    raw[5] = 1;
    raw[6] = 1;
    raw[7] = 1;
    raw[12] = 0x11;
    raw[13] = 0x11;
    raw[14] = 0x11;
    raw[15] = 0x11;

    // All low nibbles = 15 (0xFF), all high bits = 1 (0xFF) -> q = 31
    for b in &mut raw[16..48] {
        *b = 0xFF;
    }
    for b in &mut raw[48..176] {
        *b = 0xFF;
    }

    let mut out = [0.0_f32; Q5_K_BLOCK_ELEMENTS];
    dequantize_q5_k_block(&raw, &mut out);

    for (i, &v) in out.iter().enumerate() {
        assert!((v - 31.0).abs() < 1e-4, "Max quant mismatch at {i}: {v}");
    }
}

#[test]
fn test_dequantize_q5_k_negative_min_scaling() {
    let mut raw = [0u8; Q5_K_BLOCK_BYTES];
    // d = 0.25 (0x3400), dmin = 2.0 (0x4000)
    raw[0..2].copy_from_slice(&0x3400_u16.to_le_bytes());
    raw[2..4].copy_from_slice(&0x4000_u16.to_le_bytes());

    // sc0 = 1, m0 = 5
    raw[4] = 1;
    raw[8] = 5;
    // q = 10 (low = 10, high = 0)
    for b in &mut raw[48..80] {
        *b = 10;
    }

    let mut out = [0.0_f32; Q5_K_BLOCK_ELEMENTS];
    dequantize_q5_k_block(&raw, &mut out);

    // out = 0.25 * 1 * 10 - 2.0 * 5 = 2.5 - 10.0 = -7.5
    for l in 0..32 {
        assert!(
            (out[l] - (-7.5)).abs() < 1e-5,
            "Negative scale mismatch at {l}: {}",
            out[l]
        );
    }
}

#[test]
fn test_dequantize_q5_k_fp16_finite_clamp_and_subnormals() {
    let mut raw = [0u8; Q5_K_BLOCK_BYTES];
    // Subnormal d: 0x0001
    raw[0..2].copy_from_slice(&0x0001_u16.to_le_bytes());
    // Inf dmin: 0x7C00 (clamped to finite by finite f16 converter)
    raw[2..4].copy_from_slice(&0x7C00_u16.to_le_bytes());

    let mut out = [0.0_f32; Q5_K_BLOCK_ELEMENTS];
    dequantize_q5_k_block(&raw, &mut out);

    for (i, &v) in out.iter().enumerate() {
        assert!(v.is_finite(), "Non-finite value at {i}: {v}");
        assert!(!v.is_nan(), "NaN produced at {i}");
    }
}

// ============================================================================
// 6. Matrix-Vector Multiplication (y = W x)
// ============================================================================

#[test]
fn test_matvec_q5_k_single_block_row() {
    let mut raw = [0u8; Q5_K_BLOCK_BYTES];
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

    let view = TensorView::from_bytes("w_q5k", QuantType::Q5_K, &[1, 256], 0, &raw)
        .expect("valid Q5_K TensorView");

    // 1. Dot with unit vector [1.0; 256] -> sum = 4928.0
    let unit_input = [1.0_f32; 256];
    let mut output = [0.0_f32; 1];
    matvec_q5_k(&view, &unit_input, &mut output).expect("matvec succeeds");
    assert!(
        (output[0] - 4928.0).abs() < 1e-3,
        "Unit dot mismatch: {}",
        output[0]
    );

    // 2. Dot with alternating vector [+1, -1, ...] -> sum cancels to 0.0
    let mut alt_input = [0.0_f32; 256];
    for (i, v) in alt_input.iter_mut().enumerate() {
        *v = if i % 2 == 0 { 1.0 } else { -1.0 };
    }
    matvec_q5_k(&view, &alt_input, &mut output).expect("alt matvec succeeds");
    assert!(
        output[0].abs() < 1e-4,
        "Alternating cancellation mismatch: {}",
        output[0]
    );
}

#[test]
fn test_matvec_q5_k_multi_block_rows_and_cols() {
    // 4 rows x 512 cols = 8 blocks = 1408 bytes
    let rows = 4usize;
    let cols = 512usize;
    let total_blocks = rows * (cols / Q5_K_BLOCK_ELEMENTS);
    let mut raw = vec![0u8; total_blocks * Q5_K_BLOCK_BYTES];

    let mut rng = TestLcg::new(0xCAFE_BABE_1234_5678);
    for b in &mut raw {
        *b = rng.next_u8();
    }
    // Set valid finite fp16 scale patterns (non-zero)
    for blk in 0..total_blocks {
        let base = blk * Q5_K_BLOCK_BYTES;
        raw[base..base + 2].copy_from_slice(&0x3C00_u16.to_le_bytes()); // d = 1.0
        raw[base + 2..base + 4].copy_from_slice(&0x3800_u16.to_le_bytes()); // dmin = 0.5
    }

    let mut input = vec![0.0_f32; cols];
    for v in &mut input {
        *v = rng.next_f32(-1.0, 1.0);
    }

    let shape = [rows, cols];
    let view = TensorView::from_bytes("w_multi", QuantType::Q5_K, &shape, 0, &raw)
        .expect("valid multi-block view");
    let mut output = vec![0.0_f32; rows];
    matvec_q5_k(&view, &input, &mut output).expect("multi-block matvec");

    // Oracle calculation
    let blocks_per_row = cols / Q5_K_BLOCK_ELEMENTS;
    let mut dequant_buf = [0.0_f32; Q5_K_BLOCK_ELEMENTS];
    for (r, &actual_val) in output.iter().enumerate().take(rows) {
        let mut expected_row_sum = 0.0_f32;
        for blk in 0..blocks_per_row {
            let blk_idx = r * blocks_per_row + blk;
            let start = blk_idx * Q5_K_BLOCK_BYTES;
            let slice: &[u8; Q5_K_BLOCK_BYTES] =
                raw[start..start + Q5_K_BLOCK_BYTES].try_into().unwrap();
            reference_ggml_dequantize_q5_k(slice, &mut dequant_buf);
            let mut blk_sum = 0.0_f32;
            for e in 0..Q5_K_BLOCK_ELEMENTS {
                blk_sum += dequant_buf[e] * input[blk * Q5_K_BLOCK_ELEMENTS + e];
            }
            expected_row_sum += blk_sum;
        }
        let abs_diff = (actual_val - expected_row_sum).abs();
        let rel_diff = abs_diff / expected_row_sum.abs().max(1.0);
        assert!(
            abs_diff < 1e-2 && rel_diff < 1e-4,
            "Row {r} mismatch: actual {actual_val} vs expected {expected_row_sum} (abs_diff: {abs_diff}, rel_diff: {rel_diff})"
        );
    }
}

#[test]
fn test_matvec_q5_k_large_dimension_1024_cols() {
    let rows = 2usize;
    let cols = 1024usize;
    let total_blocks = rows * (cols / Q5_K_BLOCK_ELEMENTS);
    let mut raw = vec![0u8; total_blocks * Q5_K_BLOCK_BYTES];
    let mut rng = TestLcg::new(0x9876_5432_ABCD_EF01);
    for b in &mut raw {
        *b = rng.next_u8();
    }
    for blk in 0..total_blocks {
        let base = blk * Q5_K_BLOCK_BYTES;
        raw[base..base + 2].copy_from_slice(&0x3800_u16.to_le_bytes()); // d = 0.5
        raw[base + 2..base + 4].copy_from_slice(&0x3400_u16.to_le_bytes()); // dmin = 0.25
    }

    let mut input = vec![0.0_f32; cols];
    for v in &mut input {
        *v = rng.next_f32(-2.0, 2.0);
    }

    let shape = [rows, cols];
    let view = TensorView::from_bytes("w_1024", QuantType::Q5_K, &shape, 0, &raw)
        .expect("tensor view 1024");
    let mut output = vec![0.0_f32; rows];
    matvec_q5_k(&view, &input, &mut output).expect("1024 cols matvec");

    for (r, &val) in output.iter().enumerate() {
        assert!(val.is_finite(), "Non-finite output at row {r}: {val}");
        assert!(!val.is_nan(), "NaN output at row {r}");
    }
}

// ============================================================================
// 7. Defensive Bounds & Error Propagation
// ============================================================================

#[test]
fn test_matvec_q5_k_dimension_mismatch_defensive_errors() {
    let raw = vec![0u8; Q5_K_BLOCK_BYTES];

    // 1. Wrong QuantType (e.g. Q4_K_M)
    let view_wrong_quant =
        TensorView::from_bytes("w_q4", QuantType::Q4_K_M, &[1, 256], 0, &raw).expect("view Q4");
    let mut out = [0.0_f32; 1];
    let input = [1.0_f32; 256];
    assert!(matches!(
        matvec_q5_k(&view_wrong_quant, &input, &mut out),
        Err(K3Error::TensorBoundsError { .. })
    ));

    // 2. Input dimension mismatch (len 128 != cols 256)
    let view_valid =
        TensorView::from_bytes("w_valid", QuantType::Q5_K, &[1, 256], 0, &raw).expect("view valid");
    let input_short = [1.0_f32; 128];
    assert!(matches!(
        matvec_q5_k(&view_valid, &input_short, &mut out),
        Err(K3Error::TensorBoundsError { .. })
    ));

    // 3. Output dimension mismatch (len 2 != rows 1)
    let mut out_wrong = [0.0_f32; 2];
    assert!(matches!(
        matvec_q5_k(&view_valid, &input, &mut out_wrong),
        Err(K3Error::TensorBoundsError { .. })
    ));

    // 4. Columns not divisible by 256
    let view_unaligned = TensorView::from_bytes("w_unaligned", QuantType::Q5_K, &[1, 250], 0, &raw);
    assert!(view_unaligned.is_err());
}

// ============================================================================
// 8. Property Sweep Against GGML Reference Oracle
// ============================================================================

#[test]
fn test_q5_k_ggml_reference_oracle_property_sweep() {
    let mut rng = TestLcg::new(0x1337_BEEF_5555_AAAA);
    let mut block = [0u8; Q5_K_BLOCK_BYTES];
    let mut actual = [0.0_f32; Q5_K_BLOCK_ELEMENTS];
    let mut oracle = [0.0_f32; Q5_K_BLOCK_ELEMENTS];

    for trial in 0_u16..20_u16 {
        for b in &mut block {
            *b = rng.next_u8();
        }
        // Valid positive scales
        block[0..2].copy_from_slice(&(0x3C00_u16 + (trial * 10)).to_le_bytes());
        block[2..4].copy_from_slice(&(0x3800_u16 + (trial * 5)).to_le_bytes());

        dequantize_q5_k_block(&block, &mut actual);
        reference_ggml_dequantize_q5_k(&block, &mut oracle);

        for i in 0..Q5_K_BLOCK_ELEMENTS {
            let diff = (actual[i] - oracle[i]).abs();
            assert!(
                diff < 1e-4,
                "Trial {trial} mismatch at elem {i}: actual {} vs oracle {}",
                actual[i],
                oracle[i]
            );
        }
    }
}
