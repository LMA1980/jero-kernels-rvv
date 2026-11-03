//! Adversarial and Empirical Challenge Suite for Milestone 3 Compute Kernels:
//! TQ Unpacking, `BitLinear`, `SubLN`, and `ReLU^2`.
//!
//! Objectives:
//! 1. Exhaustive base-3 unpacking of all byte values 0..=255.
//! 2. Exhaustive 2-bit unpacking of all byte values 0..=255.
//! 3. `BitLinear` bit-identical accumulation between `bitlinear_i8_ime` and `bitlinear_i8_rvv`
//!    across prime rows/cols, small matrices, large multi-block matrices.
//! 4. Zero panics, zero NaN/Inf leaks, and defensive error returns on malformed inputs.
//! 5. Numerical stability and edge cases for `SubLN` and `ReLU^2`.

#![allow(
    clippy::all,
    clippy::pedantic,
    clippy::nursery,
    clippy::undocumented_unsafe_blocks
)]

use k3_core::{K3Error, QuantType, TensorView};
use kernels_rvv::f16_to_f32_finite;
use kernels_rvv::ime::{
    TQ1_0_BLOCK_BYTES, TQ1_0_BLOCK_ELEMENTS, TQ2_0_BLOCK_BYTES, TQ2_0_BLOCK_ELEMENTS,
    bitlinear_i8_ime, bitlinear_i8_rvv, unpack_tq1_0_block_i8, unpack_tq2_0_block_i8,
};
use kernels_rvv::quant::Q8_0_BLOCK_BYTES;
use kernels_rvv::{relu_squared_gate_f32, relu_squared_gate_scalar, subln_quantize_absmax_i8};

// Simple deterministic LCG pseudo-random generator for stress testing
struct SimpleLcg {
    state: u64,
}

impl SimpleLcg {
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

    fn next_u8(&mut self) -> u8 {
        (self.next_u32() & 0xFF) as u8
    }

    fn next_i8(&mut self) -> i8 {
        self.next_u8() as i8
    }

    fn next_f32(&mut self) -> f32 {
        let val = self.next_u32() & 0x007F_FFFF;
        (val as f32) / ((1 << 23) as f32)
    }
}

// ============================================================================
// 1. Exhaustive Base-3 Unpacking of all byte values 0..=255
// ============================================================================

#[test]
fn test_challenge_exhaustive_base3_unpack_all_256_bytes() {
    let mut block = [0u8; TQ1_0_BLOCK_BYTES];
    // Scale = 1.0 (FP16 0x3C00)
    block[0..2].copy_from_slice(&0x3C00_u16.to_le_bytes());

    let mut out = [0_i8; TQ1_0_BLOCK_ELEMENTS];

    // 1.1 Test all byte values 0..=255 across 5-trit positions
    for val in 0..=255_u8 {
        for chunk in 0..51 {
            block[2..53].fill(0);
            block[2 + chunk] = val;
            block[53] = 0;

            let scale = unpack_tq1_0_block_i8(&block, &mut out);
            assert_eq!(scale, 1.0);

            // Compute expected trits for `val`
            let mut b = val;
            let mut expected_trits = [0_i8; 5];
            for t in &mut expected_trits {
                *t = ((b % 3) as i8) - 1;
                b /= 3;
            }

            // Verify the 5 trits in this chunk
            let base = chunk * 5;
            for j in 0..5 {
                let trit = out[base + j];
                assert!(
                    trit == -1 || trit == 0 || trit == 1,
                    "Trit out of {{-1, 0, 1}} for byte {val} at chunk {chunk}, offset {j}: {trit}"
                );
                assert_eq!(
                    trit, expected_trits[j],
                    "Trit mismatch for byte {val} at chunk {chunk}, offset {j}: got {trit}, expected {}",
                    expected_trits[j]
                );
            }

            // Other chunks must be all -1 (since block bytes are 0, and (0 % 3) - 1 = -1)
            for other_chunk in 0..51 {
                if other_chunk != chunk {
                    let other_base = other_chunk * 5;
                    for j in 0..5 {
                        assert_eq!(out[other_base + j], -1);
                    }
                }
            }
            assert_eq!(out[255], -1);
        }
    }

    // 1.2 Test all byte values 0..=255 for the 54th byte (256th element, block[53])
    for val in 0..=255_u8 {
        block[2..53].fill(0);
        block[53] = val;

        let _ = unpack_tq1_0_block_i8(&block, &mut out);
        let expected_last = ((val % 3) as i8) - 1;
        assert!(
            out[255] == -1 || out[255] == 0 || out[255] == 1,
            "256th element out of bounds for byte {val}: {}",
            out[255]
        );
        assert_eq!(
            out[255], expected_last,
            "256th element mismatch for byte {val}: got {}, expected {expected_last}",
            out[255]
        );
    }
}

// ============================================================================
// 2. Exhaustive 2-bit Unpacking of all byte values 0..=255
// ============================================================================

#[test]
fn test_challenge_exhaustive_2bit_unpack_all_256_bytes() {
    let mut block = [0u8; TQ2_0_BLOCK_BYTES];
    // Scale = 2.5 (FP16 0x4100)
    block[0..2].copy_from_slice(&0x4100_u16.to_le_bytes());

    let mut out = [0_i8; TQ2_0_BLOCK_ELEMENTS];

    for val in 0..=255_u8 {
        for byte_idx in 0..64 {
            block[2..66].fill(0);
            block[2 + byte_idx] = val;

            let scale = unpack_tq2_0_block_i8(&block, &mut out);
            assert_eq!(scale, 2.5);

            let expected_q0 = ((val & 0x03) as i8) - 1;
            let expected_q1 = (((val >> 2) & 0x03) as i8) - 1;
            let expected_q2 = (((val >> 4) & 0x03) as i8) - 1;
            let expected_q3 = (((val >> 6) & 0x03) as i8) - 1;

            let base = byte_idx * 4;
            assert_eq!(out[base], expected_q0);
            assert_eq!(out[base + 1], expected_q1);
            assert_eq!(out[base + 2], expected_q2);
            assert_eq!(out[base + 3], expected_q3);

            // Other quants must be -1 (from 0x00)
            for other in 0..64 {
                if other != byte_idx {
                    let ob = other * 4;
                    assert_eq!(out[ob], -1);
                    assert_eq!(out[ob + 1], -1);
                    assert_eq!(out[ob + 2], -1);
                    assert_eq!(out[ob + 3], -1);
                }
            }
        }
    }
}

// ============================================================================
// 3. Stress Matrix Multiplications: Prime Rows/Cols & Bit-Identical Accumulation
// ============================================================================

#[test]
fn test_challenge_bitlinear_tq1_prime_and_irregular_shapes() {
    let prime_rows = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 31, 61];
    let multiblock_cols = [256, 512, 768, 1024]; // 1, 2, 3, 4 blocks
    let mut rng = SimpleLcg::new(0xDEAD_BEEF_CAFE_BABE);

    for &rows in &prime_rows {
        for &cols in &multiblock_cols {
            let blocks_per_row = cols / TQ1_0_BLOCK_ELEMENTS;
            let total_blocks = rows * blocks_per_row;
            let total_bytes = total_blocks * TQ1_0_BLOCK_BYTES;
            let mut raw_bytes = vec![0u8; total_bytes];

            // Fill blocks with arbitrary scales and base-3 bytes
            for blk in 0..total_blocks {
                let base = blk * TQ1_0_BLOCK_BYTES;
                // Generate a random valid normal FP16 scale
                let scale_bits = (rng.next_u32() % 0x7000 + 0x0400) as u16;
                raw_bytes[base..base + 2].copy_from_slice(&scale_bits.to_le_bytes());
                for i in 2..TQ1_0_BLOCK_BYTES {
                    raw_bytes[base + i] = rng.next_u8();
                }
            }

            let shape = [rows, cols];
            let weights =
                TensorView::from_bytes("challenge_w_tq1", QuantType::TQ1_0, &shape, 0, &raw_bytes)
                    .expect("valid tensor view");

            // Activations: mix of positive, negative, zero, extreme i8
            let mut act_i8 = vec![0_i8; cols];
            for a in &mut act_i8 {
                *a = rng.next_i8();
            }
            let act_scale = (rng.next_f32() + 0.01) * 0.1;

            let mut out_ime = vec![f32::NAN; rows];
            let mut out_rvv = vec![f32::NAN; rows];

            bitlinear_i8_ime(&mut out_ime, &weights, &act_i8, act_scale)
                .expect("bitlinear_i8_ime must succeed");
            bitlinear_i8_rvv(&mut out_rvv, &weights, &act_i8, act_scale)
                .expect("bitlinear_i8_rvv must succeed");

            for r in 0..rows {
                assert!(
                    !out_ime[r].is_nan(),
                    "out_ime[{r}] is NaN for shape [{rows}, {cols}]"
                );
                assert!(
                    !out_rvv[r].is_nan(),
                    "out_rvv[{r}] is NaN for shape [{rows}, {cols}]"
                );
                assert_eq!(
                    out_ime[r].to_bits(),
                    out_rvv[r].to_bits(),
                    "Bit-identity violation for TQ1_0 [{rows}, {cols}] at row {r}: IME={} RVV={}",
                    out_ime[r],
                    out_rvv[r]
                );
            }
        }
    }
}

#[test]
fn test_challenge_bitlinear_tq2_prime_and_irregular_shapes() {
    let prime_rows = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 31, 61];
    let multiblock_cols = [256, 512, 768, 1024]; // 1, 2, 3, 4 blocks
    let mut rng = SimpleLcg::new(0x0123_4567_89AB_CDEF);

    for &rows in &prime_rows {
        for &cols in &multiblock_cols {
            let blocks_per_row = cols / TQ2_0_BLOCK_ELEMENTS;
            let total_blocks = rows * blocks_per_row;
            let total_bytes = total_blocks * TQ2_0_BLOCK_BYTES;
            let mut raw_bytes = vec![0u8; total_bytes];

            for blk in 0..total_blocks {
                let base = blk * TQ2_0_BLOCK_BYTES;
                let scale_bits = (rng.next_u32() % 0x7000 + 0x0400) as u16;
                raw_bytes[base..base + 2].copy_from_slice(&scale_bits.to_le_bytes());
                for i in 2..TQ2_0_BLOCK_BYTES {
                    raw_bytes[base + i] = rng.next_u8();
                }
            }

            let shape = [rows, cols];
            let weights =
                TensorView::from_bytes("challenge_w_tq2", QuantType::TQ2_0, &shape, 0, &raw_bytes)
                    .expect("valid tensor view");

            let mut act_i8 = vec![0_i8; cols];
            for a in &mut act_i8 {
                *a = rng.next_i8();
            }
            let act_scale = (rng.next_f32() + 0.01) * 0.1;

            let mut out_ime = vec![f32::NAN; rows];
            let mut out_rvv = vec![f32::NAN; rows];

            bitlinear_i8_ime(&mut out_ime, &weights, &act_i8, act_scale)
                .expect("bitlinear_i8_ime must succeed");
            bitlinear_i8_rvv(&mut out_rvv, &weights, &act_i8, act_scale)
                .expect("bitlinear_i8_rvv must succeed");

            for r in 0..rows {
                assert!(
                    !out_ime[r].is_nan(),
                    "out_ime[{r}] is NaN for shape [{rows}, {cols}]"
                );
                assert!(
                    !out_rvv[r].is_nan(),
                    "out_rvv[{r}] is NaN for shape [{rows}, {cols}]"
                );
                assert_eq!(
                    out_ime[r].to_bits(),
                    out_rvv[r].to_bits(),
                    "Bit-identity violation for TQ2_0 [{rows}, {cols}] at row {r}: IME={} RVV={}",
                    out_ime[r],
                    out_rvv[r]
                );
            }
        }
    }
}

#[test]
fn test_challenge_bitlinear_q8_0_shapes() {
    let rows_list = [1, 2, 3, 7, 8, 9, 15, 16, 23];
    let cols_list = [32, 64, 96, 128, 256];
    let mut rng = SimpleLcg::new(0xFEED_FACE_CAFE_0001);

    for &rows in &rows_list {
        for &cols in &cols_list {
            let blocks_per_row = cols / 32;
            let total_blocks = rows * blocks_per_row;
            let total_bytes = total_blocks * Q8_0_BLOCK_BYTES;
            let mut raw_bytes = vec![0u8; total_bytes];

            for blk in 0..total_blocks {
                let base = blk * Q8_0_BLOCK_BYTES;
                let scale_bits = (rng.next_u32() % 0x7000 + 0x0400) as u16;
                raw_bytes[base..base + 2].copy_from_slice(&scale_bits.to_le_bytes());
                for i in 2..Q8_0_BLOCK_BYTES {
                    raw_bytes[base + i] = rng.next_u8();
                }
            }

            let shape = [rows, cols];
            let weights =
                TensorView::from_bytes("challenge_w_q8_0", QuantType::Q8_0, &shape, 0, &raw_bytes)
                    .expect("valid tensor view");

            let mut act_i8 = vec![0_i8; cols];
            for a in &mut act_i8 {
                *a = rng.next_i8();
            }
            let act_scale = 0.05_f32;

            let mut out_ime = vec![f32::NAN; rows];
            let mut out_rvv = vec![f32::NAN; rows];

            bitlinear_i8_ime(&mut out_ime, &weights, &act_i8, act_scale)
                .expect("bitlinear_i8_ime Q8_0 succeeds");
            bitlinear_i8_rvv(&mut out_rvv, &weights, &act_i8, act_scale)
                .expect("bitlinear_i8_rvv Q8_0 succeeds");

            for r in 0..rows {
                assert_eq!(
                    out_ime[r].to_bits(),
                    out_rvv[r].to_bits(),
                    "Q8_0 Bit-identity mismatch [{rows}, {cols}] at row {r}: IME={} RVV={}",
                    out_ime[r],
                    out_rvv[r]
                );
            }
        }
    }
}

// ============================================================================
// 4. Large Multi-Block Stress Test
// ============================================================================

#[test]
fn test_challenge_bitlinear_large_multiblock_stress() {
    let rows = 37;
    let cols = 4096; // 16 blocks per row = 592 blocks total
    let mut rng = SimpleLcg::new(0x9999_8888_7777_6666);

    // Test TQ1_0 large
    let total_bytes_tq1 = rows * 16 * TQ1_0_BLOCK_BYTES;
    let mut raw_tq1 = vec![0u8; total_bytes_tq1];
    for blk in 0..(rows * 16) {
        let base = blk * TQ1_0_BLOCK_BYTES;
        let scale_bits = (rng.next_u32() % 0x6000 + 0x0800) as u16;
        raw_tq1[base..base + 2].copy_from_slice(&scale_bits.to_le_bytes());
        for i in 2..TQ1_0_BLOCK_BYTES {
            raw_tq1[base + i] = rng.next_u8();
        }
    }
    let shape = [rows, cols];
    let view_tq1 = TensorView::from_bytes("large_w_tq1", QuantType::TQ1_0, &shape, 0, &raw_tq1)
        .expect("large view");

    // Test TQ2_0 large
    let total_bytes_tq2 = rows * 16 * TQ2_0_BLOCK_BYTES;
    let mut raw_tq2 = vec![0u8; total_bytes_tq2];
    for blk in 0..(rows * 16) {
        let base = blk * TQ2_0_BLOCK_BYTES;
        let scale_bits = (rng.next_u32() % 0x6000 + 0x0800) as u16;
        raw_tq2[base..base + 2].copy_from_slice(&scale_bits.to_le_bytes());
        for i in 2..TQ2_0_BLOCK_BYTES {
            raw_tq2[base + i] = rng.next_u8();
        }
    }
    let view_tq2 = TensorView::from_bytes("large_w_tq2", QuantType::TQ2_0, &shape, 0, &raw_tq2)
        .expect("large view");

    let act_i8: Vec<i8> = (0..cols).map(|_| rng.next_i8()).collect();
    let act_scale = 0.007_812_5_f32;

    let mut out_ime_tq1 = vec![0.0_f32; rows];
    let mut out_rvv_tq1 = vec![0.0_f32; rows];
    bitlinear_i8_ime(&mut out_ime_tq1, &view_tq1, &act_i8, act_scale).expect("large ime tq1");
    bitlinear_i8_rvv(&mut out_rvv_tq1, &view_tq1, &act_i8, act_scale).expect("large rvv tq1");

    for r in 0..rows {
        assert_eq!(
            out_ime_tq1[r].to_bits(),
            out_rvv_tq1[r].to_bits(),
            "Large TQ1 mismatch at row {r}"
        );
    }

    let mut out_ime_tq2 = vec![0.0_f32; rows];
    let mut out_rvv_tq2 = vec![0.0_f32; rows];
    bitlinear_i8_ime(&mut out_ime_tq2, &view_tq2, &act_i8, act_scale).expect("large ime tq2");
    bitlinear_i8_rvv(&mut out_rvv_tq2, &view_tq2, &act_i8, act_scale).expect("large rvv tq2");

    for r in 0..rows {
        assert_eq!(
            out_ime_tq2[r].to_bits(),
            out_rvv_tq2[r].to_bits(),
            "Large TQ2 mismatch at row {r}"
        );
    }
}

// ============================================================================
// 5. Adverse Inputs, Empty Buffers, Malformed Dimensions
// ============================================================================

#[test]
fn test_challenge_bitlinear_adverse_inputs_and_bounds_validation() {
    let raw = vec![0u8; 8 * TQ1_0_BLOCK_BYTES];
    let shape = [8, 256];
    let valid_view =
        TensorView::from_bytes("valid_tq1", QuantType::TQ1_0, &shape, 0, &raw).expect("valid view");

    let act_valid = [1_i8; 256];
    let mut out_valid = [0.0_f32; 8];

    // 5.1 Empty output
    let mut empty_out = [0.0_f32; 0];
    assert!(matches!(
        bitlinear_i8_ime(&mut empty_out, &valid_view, &act_valid, 1.0),
        Err(K3Error::TensorBoundsError { .. })
    ));
    assert!(matches!(
        bitlinear_i8_rvv(&mut empty_out, &valid_view, &act_valid, 1.0),
        Err(K3Error::TensorBoundsError { .. })
    ));

    // 5.2 Empty activation
    let empty_act: [i8; 0] = [];
    assert!(matches!(
        bitlinear_i8_ime(&mut out_valid, &valid_view, &empty_act, 1.0),
        Err(K3Error::TensorBoundsError { .. })
    ));
    assert!(matches!(
        bitlinear_i8_rvv(&mut out_valid, &valid_view, &empty_act, 1.0),
        Err(K3Error::TensorBoundsError { .. })
    ));

    // 5.3 Activation unaligned to block size (e.g. 255 elements instead of 256)
    let bad_act = [1_i8; 255];
    assert!(matches!(
        bitlinear_i8_ime(&mut out_valid, &valid_view, &bad_act, 1.0),
        Err(K3Error::TensorBoundsError { .. })
    ));
    assert!(matches!(
        bitlinear_i8_rvv(&mut out_valid, &valid_view, &bad_act, 1.0),
        Err(K3Error::TensorBoundsError { .. })
    ));

    // 5.4 Output length mismatch (e.g. 7 instead of 8)
    let mut bad_out = [0.0_f32; 7];
    assert!(matches!(
        bitlinear_i8_ime(&mut bad_out, &valid_view, &act_valid, 1.0),
        Err(K3Error::TensorBoundsError { .. })
    ));
    assert!(matches!(
        bitlinear_i8_rvv(&mut bad_out, &valid_view, &act_valid, 1.0),
        Err(K3Error::TensorBoundsError { .. })
    ));

    // 5.5 Unsupported quantization format (e.g. F32 or Q4_0 passed to BitLinear)
    let raw_f32 = vec![0u8; 8 * 256 * 4];
    let view_f32 =
        TensorView::from_bytes("f32_w", QuantType::F32, &shape, 0, &raw_f32).expect("f32 view");
    assert!(matches!(
        bitlinear_i8_ime(&mut out_valid, &view_f32, &act_valid, 1.0),
        Err(K3Error::TensorBoundsError { .. })
    ));
    assert!(matches!(
        bitlinear_i8_rvv(&mut out_valid, &view_f32, &act_valid, 1.0),
        Err(K3Error::TensorBoundsError { .. })
    ));

    // 5.6 Backing buffer truncated (smaller than required_blocks * block_bytes)
    let truncated_raw = vec![0u8; 7 * TQ1_0_BLOCK_BYTES]; // Only 7 blocks for 8 rows
    assert!(matches!(
        TensorView::from_bytes("trunc_tq1", QuantType::TQ1_0, &shape, 0, &truncated_raw),
        Err(K3Error::TensorBoundsError { .. })
    ));

    // 5.7 Zero scale and negative scale handling: must not panic or diverge
    let mut out_zero_ime = [f32::NAN; 8];
    let mut out_zero_rvv = [f32::NAN; 8];
    bitlinear_i8_ime(&mut out_zero_ime, &valid_view, &act_valid, 0.0).expect("zero scale ime");
    bitlinear_i8_rvv(&mut out_zero_rvv, &valid_view, &act_valid, 0.0).expect("zero scale rvv");
    assert_eq!(out_zero_ime, [0.0; 8]);
    assert_eq!(out_zero_rvv, [0.0; 8]);

    let mut out_neg_ime = [f32::NAN; 8];
    let mut out_neg_rvv = [f32::NAN; 8];
    bitlinear_i8_ime(&mut out_neg_ime, &valid_view, &act_valid, -0.5).expect("neg scale ime");
    bitlinear_i8_rvv(&mut out_neg_rvv, &valid_view, &act_valid, -0.5).expect("neg scale rvv");
    for r in 0..8 {
        assert_eq!(out_neg_ime[r].to_bits(), out_neg_rvv[r].to_bits());
    }
}

// ============================================================================
// 6. SubLN Dynamic Absmax & Numerical Stability
// ============================================================================

#[test]
fn test_challenge_subln_adversarial_numerical_stress() {
    let eps = 1e-5_f32;

    // 6.1 All zeroes
    let zero_in = [0.0_f32; 128];
    let norm_w = [1.0_f32; 128];
    let mut out_i8 = [99_i8; 128];
    let scale = subln_quantize_absmax_i8(&mut out_i8, &zero_in, &norm_w, eps)
        .expect("zero input must succeed");
    assert_eq!(scale, 1.0 / 127.0);
    assert!(out_i8.iter().all(|&v| v == 0));

    // 6.2 All NaN input: must trigger defensive guard, return safe scale, zero out_i8
    let nan_in = [f32::NAN; 64];
    let mut nan_out = [42_i8; 64];
    let nan_scale = subln_quantize_absmax_i8(&mut nan_out, &nan_in, &norm_w[..64], eps)
        .expect("NaN input must not panic");
    assert!(nan_scale > 0.0 && nan_scale.is_finite());
    assert!(!nan_scale.is_nan());
    assert!(nan_out.iter().all(|&v| v == 0));

    // 6.3 Input containing one NaN amongst valid floats: must not panic or leak NaN
    let mut mixed_nan_in = [1.0_f32; 64];
    mixed_nan_in[7] = f32::NAN;
    let mut mixed_out = [42_i8; 64];
    let mixed_scale = subln_quantize_absmax_i8(&mut mixed_out, &mixed_nan_in, &norm_w[..64], eps)
        .expect("mixed NaN must not panic");
    assert!(!mixed_scale.is_nan());
    assert!(mixed_scale.is_finite());

    // 6.4 Inf input
    let inf_in = [f32::INFINITY; 64];
    let mut inf_out = [42_i8; 64];
    let inf_scale = subln_quantize_absmax_i8(&mut inf_out, &inf_in, &norm_w[..64], eps)
        .expect("Inf input must not panic");
    assert!(!inf_scale.is_nan());
    assert!(inf_scale.is_finite());

    // 6.5 Extreme dynamic range (large numbers without overflow)
    let extreme_in = [1e15_f32, -1e15, 5e14, -5e14];
    let mut extreme_out = [0_i8; 4];
    let extreme_scale = subln_quantize_absmax_i8(&mut extreme_out, &extreme_in, &norm_w[..4], eps)
        .expect("extreme input succeeds");
    assert!(extreme_scale.is_finite() && extreme_scale > 0.0);
    assert_eq!(extreme_out[0], 127);
    assert_eq!(extreme_out[1], -127);

    // 6.6 Subnormal input numbers (1e-38)
    let subnormal_in = [1e-38_f32; 32];
    let mut subnormal_out = [42_i8; 32];
    let sub_scale = subln_quantize_absmax_i8(&mut subnormal_out, &subnormal_in, &norm_w[..32], eps)
        .expect("subnormal input succeeds");
    assert!(sub_scale.is_finite());
    assert!(subnormal_out.iter().all(|&v| v == 0));

    // 6.7 Slice dimension mismatches
    let mut mismatch_out = [0_i8; 32];
    assert!(matches!(
        subln_quantize_absmax_i8(&mut mismatch_out, &[1.0; 16], &norm_w[..32], eps),
        Err(K3Error::TensorBoundsError { .. })
    ));
    assert!(matches!(
        subln_quantize_absmax_i8(&mut mismatch_out, &[1.0; 32], &norm_w[..16], eps),
        Err(K3Error::TensorBoundsError { .. })
    ));
    assert!(matches!(
        subln_quantize_absmax_i8(&mut [0_i8; 16], &[1.0; 32], &norm_w[..32], eps),
        Err(K3Error::TensorBoundsError { .. })
    ));
    assert!(matches!(
        subln_quantize_absmax_i8(&mut [0_i8; 0], &[0.0; 0], &[0.0; 0], eps),
        Err(K3Error::TensorBoundsError { .. })
    ));
}

// ============================================================================
// 7. ReLU^2 Activation & Strip-Mining Length Stress
// ============================================================================

#[test]
fn test_challenge_relu_squared_adversarial_numerical_stress() {
    // 7.1 Strip-mining length boundary checks
    let test_lengths = [
        1, 2, 3, 7, 8, 9, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 513, 1024,
    ];

    for &len in &test_lengths {
        let gate: Vec<f32> = (0..len).map(|i| (i as f32) * 0.1 - 5.0).collect();
        let up: Vec<f32> = (0..len).map(|i| ((i % 5) as f32) * 0.5 + 0.2).collect();
        let mut out = vec![f32::NAN; len];
        let mut out_ref = vec![f32::NAN; len];

        relu_squared_gate_f32(&mut out, &gate, &up).expect("relu_squared dispatch succeeds");
        relu_squared_gate_scalar(&mut out_ref, &gate, &up);

        assert_eq!(out.len(), len);
        for i in 0..len {
            assert!(
                !out[i].is_nan(),
                "out[{i}] is NaN for len {len} with gate={}",
                gate[i]
            );
            assert_eq!(
                out[i].to_bits(),
                out_ref[i].to_bits(),
                "Mismatch between dispatch and scalar for len {len} at idx {i}: {} vs {}",
                out[i],
                out_ref[i]
            );
        }
    }

    // 7.2 Dimension mismatches
    let mut out_3 = [0.0_f32; 3];
    assert!(matches!(
        relu_squared_gate_f32(&mut out_3, &[1.0; 2], &[1.0; 3]),
        Err(K3Error::TensorBoundsError { .. })
    ));
    assert!(matches!(
        relu_squared_gate_f32(&mut out_3, &[1.0; 3], &[1.0; 4]),
        Err(K3Error::TensorBoundsError { .. })
    ));
    assert!(matches!(
        relu_squared_gate_f32(&mut [0.0; 0], &[0.0; 0], &[0.0; 0]),
        Err(K3Error::TensorBoundsError { .. })
    ));
}

// ============================================================================
// 8. Exhaustive FP16 Oracle Verification
// ============================================================================

#[test]
fn test_challenge_f16_to_f32_finite_exhaustive_oracle() {
    // Assert zero NaN and zero Inf across all 65,536 bit patterns
    for bits in 0..=u16::MAX {
        let val = f16_to_f32_finite(bits);
        assert!(
            val.is_finite(),
            "Non-finite value produced for bits 0x{bits:04X}: {val}"
        );
        assert!(!val.is_nan(), "NaN produced for bits 0x{bits:04X}");

        let exp = (bits >> 10) & 0x1F;
        if exp == 0x1F {
            // Clamped NaN/Inf
            assert!((val - 0.0625).abs() < 1e-6 || (val - -0.0625).abs() < 1e-6);
        }
    }
}
