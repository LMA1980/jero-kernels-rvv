//! Empirical Adversarial Challenge Test Suite for Milestone 3
//!
//! Stress-tests and empirical verification of:
//! 1. `subln_quantize_absmax_i8`: all zeros, constant inputs, extreme floats ($10^{-20}$, $10^6$),
//!    alternating signs, dynamic per-token absmax formula, NaN/Inf robustness, dimension bounds.
//! 2. `relu_squared_gate_f32`: negative truncation, zero values, large positive values,
//!    mathematical exactness $\max(0, gate)^2 \times up$, dimension mismatch error handling.
//! 3. BitLinear equivalence: `bitlinear_i8_rvv` vs `bitlinear_i8_ime` across irregular row dimensions
//!    (1, 7, 8, 15, 16, 32) and multi-block columns (256, 512, 1024) for both `TQ1_0` and `TQ2_0`.
//! 4. End-to-end chaining: SubLN -> BitLinear -> ReLU^2 -> SubLN -> BitLinear.
//! 5. Performance and throughput stability.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_lossless,
    clippy::cast_sign_loss,
    clippy::unreadable_literal,
    clippy::doc_markdown,
    clippy::needless_range_loop,
    clippy::float_cmp,
    clippy::too_many_lines
)]

use jero_core::{K3Error, QuantType, TensorView};
use jero_kernels_rvv::ime::{
    TQ1_0_BLOCK_BYTES, TQ1_0_BLOCK_ELEMENTS, TQ2_0_BLOCK_BYTES, TQ2_0_BLOCK_ELEMENTS,
    bitlinear_i8_ime, bitlinear_i8_rvv, unpack_tq1_0_block_i8, unpack_tq2_0_block_i8,
};
use jero_kernels_rvv::{relu_squared_gate_f32, relu_squared_gate_scalar, subln_quantize_absmax_i8};
use std::time::Instant;

// ============================================================================
// 1. STRESS TESTS: subln_quantize_absmax_i8
// ============================================================================

#[test]
fn test_challenge_subln_all_zeros_and_near_zeros() {
    let len = 256;
    let eps = 1e-5_f32;
    let norm_weight = vec![1.0_f32; len];

    // A. All-zero input: must NOT divide by zero or produce NaN
    let zero_input = vec![0.0_f32; len];
    let mut zero_out = vec![123_i8; len];
    let scale_zero = subln_quantize_absmax_i8(&mut zero_out, &zero_input, &norm_weight, eps)
        .expect("all-zero input must succeed");

    assert!(scale_zero.is_finite(), "scale must be finite");
    assert!(!scale_zero.is_nan(), "scale must not be NaN");
    assert!(scale_zero > 0.0, "scale must be strictly positive");
    assert_eq!(
        scale_zero,
        1.0_f32 / 127.0_f32,
        "all-zero input scale must equal default safe scale 1.0 / 127.0"
    );
    assert!(
        zero_out.iter().all(|&v| v == 0),
        "all-zero input must produce all-zero quantized output"
    );

    // B. All-zero weights with non-zero input: normalized values are all 0.0
    let non_zero_input: Vec<f32> = (0..len).map(|i| (i as f32) + 1.0).collect();
    let zero_weights = vec![0.0_f32; len];
    let mut out_zero_w = vec![42_i8; len];
    let scale_zero_w =
        subln_quantize_absmax_i8(&mut out_zero_w, &non_zero_input, &zero_weights, eps)
            .expect("all-zero weights must succeed");

    assert!(scale_zero_w.is_finite());
    assert_eq!(scale_zero_w, 1.0_f32 / 127.0_f32);
    assert!(out_zero_w.iter().all(|&v| v == 0));

    // C. Extreme subnormal / near-zero values (1e-20, 1e-15)
    for &tiny_val in &[1e-20_f32, 1e-15_f32, -1e-20_f32, -1e-15_f32] {
        let tiny_input = vec![tiny_val; len];
        let mut tiny_out = vec![77_i8; len];
        let scale_tiny = subln_quantize_absmax_i8(&mut tiny_out, &tiny_input, &norm_weight, eps)
            .expect("near-zero input must succeed");

        assert!(scale_tiny.is_finite());
        assert!(!scale_tiny.is_nan());
        assert!(scale_tiny > 0.0);
        assert!(
            tiny_out.iter().all(|&v| v == 0),
            "near-zero input must trigger defensive guard and fill 0"
        );
    }

    // D. Zero epsilon (eps = 0.0) with zero input
    let mut zero_out_eps0 = vec![55_i8; len];
    let scale_eps0 =
        subln_quantize_absmax_i8(&mut zero_out_eps0, &zero_input, &norm_weight, 0.0_f32)
            .expect("zero eps with zero input must not panic");
    assert!(scale_eps0.is_finite());
    assert!(zero_out_eps0.iter().all(|&v| v == 0));
}

#[test]
fn test_challenge_subln_constant_and_extreme_floats() {
    let len = 256;
    let eps = 1e-5_f32;
    let norm_weight = vec![1.0_f32; len];

    // A. Constant positive inputs: 10^6
    let large_val = 1e6_f32;
    let input_large = vec![large_val; len];
    let mut out_large = vec![0_i8; len];
    let scale_large = subln_quantize_absmax_i8(&mut out_large, &input_large, &norm_weight, eps)
        .expect("large constant input must succeed");

    // Analytical expectation:
    // mean_sq = 1e12, inv_rms = 1.0 / sqrt(1e12 + 1e-5) = 1e-6
    // x_norm = 1e6 * 1e-6 * 1.0 = 1.0
    // max_val = 1.0, scale = 1.0 / 127.0
    // out_i8 = round(1.0 / (1.0 / 127.0)) = 127
    assert!((scale_large - (1.0 / 127.0)).abs() < 1e-6);
    assert!(
        out_large.iter().all(|&v| v == 127),
        "constant large input must quantize uniformly to 127"
    );

    // B. Constant negative inputs: -10^6
    let input_neg_large = vec![-large_val; len];
    let mut out_neg_large = vec![0_i8; len];
    let scale_neg_large =
        subln_quantize_absmax_i8(&mut out_neg_large, &input_neg_large, &norm_weight, eps)
            .expect("large negative input must succeed");

    assert!((scale_neg_large - (1.0 / 127.0)).abs() < 1e-6);
    assert!(
        out_neg_large.iter().all(|&v| v == -127),
        "constant negative large input must quantize uniformly to -127"
    );

    // C. Alternating signs [+2.5, -2.5, +2.5, -2.5, ...]
    let alt_input: Vec<f32> = (0..len)
        .map(|i| if i % 2 == 0 { 2.5_f32 } else { -2.5_f32 })
        .collect();
    let mut alt_out = vec![0_i8; len];
    let scale_alt = subln_quantize_absmax_i8(&mut alt_out, &alt_input, &norm_weight, eps)
        .expect("alternating sign input must succeed");

    assert!((scale_alt - (1.0 / 127.0)).abs() < 1e-6);
    for (i, &v) in alt_out.iter().enumerate() {
        let expected = if i % 2 == 0 { 127 } else { -127 };
        assert_eq!(v, expected, "alternating element {i} mismatch");
    }

    // D. Heterogeneous weights (including negative and fractional weights)
    let heter_weight: Vec<f32> = (0..len)
        .map(|i| if i % 2 == 0 { 0.5_f32 } else { -2.0_f32 })
        .collect();
    let const_in = vec![1.0_f32; len];
    let mut heter_out = vec![0_i8; len];
    let scale_heter = subln_quantize_absmax_i8(&mut heter_out, &const_in, &heter_weight, eps)
        .expect("heterogeneous weight must succeed");

    // inv_rms approx 1.0. For even i: x_norm = 0.5. For odd i: x_norm = -2.0.
    // max_val = 2.0. scale = 2.0 / 127.0.
    // Even i: round(0.5 / (2.0 / 127.0)) = round(31.75) = 32.
    // Odd i: round(-2.0 / (2.0 / 127.0)) = -127.
    assert!((scale_heter - (2.0 / 127.0)).abs() < 1e-5);
    for (i, &v) in heter_out.iter().enumerate() {
        let expected = if i % 2 == 0 { 32 } else { -127 };
        assert_eq!(v, expected, "heterogeneous weight element {i} mismatch");
    }
}

#[test]
fn test_challenge_subln_nan_and_inf_poisoning_robustness() {
    let len = 256;
    let eps = 1e-5_f32;
    let norm_weight = vec![1.0_f32; len];

    // A. Input containing NaN
    let mut nan_input = vec![1.0_f32; len];
    nan_input[13] = f32::NAN;
    let mut nan_out = vec![99_i8; len];
    let scale_nan = subln_quantize_absmax_i8(&mut nan_out, &nan_input, &norm_weight, eps)
        .expect("NaN input must be handled gracefully without panic");

    assert!(scale_nan.is_finite(), "NaN input must yield finite scale");
    assert!(!scale_nan.is_nan(), "NaN input scale must not be NaN");
    assert!(
        nan_out.iter().all(|&v| v == 0),
        "NaN poisoned input must safely zero-fill output"
    );

    // B. Input containing Infinity
    let mut inf_input = vec![1.0_f32; len];
    inf_input[42] = f32::INFINITY;
    let mut inf_out = vec![88_i8; len];
    let scale_inf = subln_quantize_absmax_i8(&mut inf_out, &inf_input, &norm_weight, eps)
        .expect("Inf input must be handled gracefully without panic");

    assert!(scale_inf.is_finite());
    assert!(!scale_inf.is_nan());
    assert!(
        inf_out.iter().all(|&v| v == 0),
        "Inf poisoned input must safely zero-fill output"
    );
}

#[test]
fn test_challenge_subln_dynamic_absmax_scale_oracle() {
    // Generate 50 pseudo-random test vectors and verify exact formula adherence against f64 oracle
    let len = 256;
    let eps = 1e-5_f32;

    for iter in 0..50 {
        let seed = (iter as u32).wrapping_mul(1013904223).wrapping_add(1664525);
        let input: Vec<f32> = (0..len)
            .map(|i| {
                let r = ((seed ^ ((i as u32).wrapping_mul(7919))) % 10000) as f32 - 5000.0;
                r * 0.01_f32
            })
            .collect();
        let weight: Vec<f32> = (0..len)
            .map(|i| {
                (((seed.wrapping_add((i as u32).wrapping_mul(31))) % 1000) as f32 * 0.002) + 0.1
            })
            .collect();

        let mut actual_out = vec![0_i8; len];
        let actual_scale = subln_quantize_absmax_i8(&mut actual_out, &input, &weight, eps)
            .expect("subln oracle run");

        // F64 reference oracle
        let sum_sq: f64 = input.iter().map(|&x| (x as f64) * (x as f64)).sum();
        let mean_sq = sum_sq / (len as f64);
        let inv_rms = 1.0_f64 / (mean_sq + (eps as f64)).sqrt();

        let mut max_val = 0.0_f64;
        for i in 0..len {
            let x_norm = (input[i] as f64) * inv_rms * (weight[i] as f64);
            let abs_v = x_norm.abs();
            if abs_v > max_val {
                max_val = abs_v;
            }
        }

        let expected_scale = (max_val / 127.0) as f32;
        assert!(
            (actual_scale - expected_scale).abs() / expected_scale < 1e-4,
            "Iter {iter}: scale mismatch: actual={actual_scale}, expected={expected_scale}"
        );

        // Verify quantization accuracy: error between dequantized value and normalized float <= 0.6 * scale
        for i in 0..len {
            let x_norm_f32 = input[i] * (inv_rms as f32) * weight[i];
            let dequant = (actual_out[i] as f32) * actual_scale;
            let err = (dequant - x_norm_f32).abs();
            assert!(
                err <= actual_scale * 0.6 + 1e-5,
                "Iter {iter} elem {i}: dequant error {err} exceeds threshold (scale={actual_scale})"
            );
        }
    }
}

#[test]
fn test_challenge_subln_dimension_mismatches() {
    let eps = 1e-5_f32;

    // 1. Empty slices
    let mut empty_out: [i8; 0] = [];
    let empty_in: [f32; 0] = [];
    let empty_w: [f32; 0] = [];
    assert!(matches!(
        subln_quantize_absmax_i8(&mut empty_out, &empty_in, &empty_w, eps),
        Err(K3Error::TensorBoundsError { .. })
    ));

    // 2. Output length mismatch
    let mut short_out = [0_i8; 127];
    let normal_in = [1.0_f32; 128];
    let normal_w = [1.0_f32; 128];
    assert!(matches!(
        subln_quantize_absmax_i8(&mut short_out, &normal_in, &normal_w, eps),
        Err(K3Error::TensorBoundsError { .. })
    ));

    // 3. Weight length mismatch
    let mut normal_out = [0_i8; 128];
    let short_w = [1.0_f32; 127];
    assert!(matches!(
        subln_quantize_absmax_i8(&mut normal_out, &normal_in, &short_w, eps),
        Err(K3Error::TensorBoundsError { .. })
    ));
}

// ============================================================================
// 2. STRESS TESTS: relu_squared_gate_f32
// ============================================================================

#[test]
fn test_challenge_relu_squared_gate_extremes_and_negative_truncation() {
    // A. Strictly negative gates with positive/negative up values
    let neg_gate = [-1000.0_f32, -10.0, -1.0, -0.001, -1e-12, -1e-25];
    let up = [10.0_f32, -5.0, 2.0, 0.5, 100.0, -100.0];
    let mut out = [999.0_f32; 6];

    relu_squared_gate_f32(&mut out, &neg_gate, &up).expect("negative gate succeeds");
    for (i, &v) in out.iter().enumerate() {
        assert_eq!(
            v, 0.0,
            "negative gate element {i} must strictly evaluate to 0.0"
        );
    }

    // B. Zero and negative zero gates
    let zero_gate = [0.0_f32, -0.0_f32, 0.0_f32, -0.0_f32];
    let up_zero = [100.0_f32, -50.0_f32, 1e10_f32, -1e10_f32];
    let mut out_zero = [777.0_f32; 4];

    relu_squared_gate_f32(&mut out_zero, &zero_gate, &up_zero).expect("zero gate succeeds");
    for (i, &v) in out_zero.iter().enumerate() {
        assert_eq!(
            v, 0.0,
            "zero gate element {i} must strictly evaluate to 0.0"
        );
    }

    // C. Mathematical exactness: max(0, gate)^2 * up
    let test_cases = [
        (0.5_f32, 4.0_f32, 1.0_f32),      // 0.25 * 4 = 1.0
        (2.0_f32, 3.0_f32, 12.0_f32),     // 4 * 3 = 12.0
        (3.0_f32, -2.0_f32, -18.0_f32),   // 9 * (-2) = -18.0
        (10.0_f32, 0.05_f32, 5.0_f32),    // 100 * 0.05 = 5.0
        (100.0_f32, 0.0001_f32, 1.0_f32), // 10000 * 0.0001 = 1.0
        (1e4_f32, 1e-8_f32, 1.0_f32),     // 1e8 * 1e-8 = 1.0
    ];

    let gate_vec: Vec<f32> = test_cases.iter().map(|c| c.0).collect();
    let up_vec: Vec<f32> = test_cases.iter().map(|c| c.1).collect();
    let mut actual_out = vec![0.0_f32; test_cases.len()];

    relu_squared_gate_f32(&mut actual_out, &gate_vec, &up_vec).expect("exactness test");
    for (i, &(g, u, expected)) in test_cases.iter().enumerate() {
        let diff = (actual_out[i] - expected).abs();
        assert!(
            diff < 1e-5,
            "Case {i} (gate={g}, up={u}): expected {expected}, got {}",
            actual_out[i]
        );
    }

    // D. Fuzzing equivalence: scalar vs RVV dispatch across 1000 random floats
    let len = 1000;
    let gate_fuzz: Vec<f32> = (0..len)
        .map(|i| {
            let r = ((i * 104729) % 20000) as f32 - 10000.0;
            r * 0.01_f32
        })
        .collect();
    let up_fuzz: Vec<f32> = (0..len)
        .map(|i| {
            let r = ((i * 32452843) % 20000) as f32 - 10000.0;
            r * 0.01_f32
        })
        .collect();

    let mut out_dispatch = vec![0.0_f32; len];
    let mut out_scalar = vec![0.0_f32; len];

    relu_squared_gate_f32(&mut out_dispatch, &gate_fuzz, &up_fuzz).expect("dispatch fuzz");
    relu_squared_gate_scalar(&mut out_scalar, &gate_fuzz, &up_fuzz);

    for i in 0..len {
        assert_eq!(
            out_dispatch[i].to_bits(),
            out_scalar[i].to_bits(),
            "Mismatch between dispatch and scalar relu_squared at index {i}"
        );
    }
}

#[test]
fn test_challenge_relu_squared_dimension_mismatches() {
    // 1. Zero lengths
    let mut out_empty: [f32; 0] = [];
    let gate_empty: [f32; 0] = [];
    let up_empty: [f32; 0] = [];
    assert!(matches!(
        relu_squared_gate_f32(&mut out_empty, &gate_empty, &up_empty),
        Err(K3Error::TensorBoundsError { .. })
    ));

    // 2. Output dimension mismatch
    let mut out_short = [0.0_f32; 3];
    let gate_4 = [1.0_f32; 4];
    let up_4 = [1.0_f32; 4];
    assert!(matches!(
        relu_squared_gate_f32(&mut out_short, &gate_4, &up_4),
        Err(K3Error::TensorBoundsError { .. })
    ));

    // 3. Up dimension mismatch
    let mut out_4 = [0.0_f32; 4];
    let up_3 = [1.0_f32; 3];
    assert!(matches!(
        relu_squared_gate_f32(&mut out_4, &gate_4, &up_3),
        Err(K3Error::TensorBoundsError { .. })
    ));
}

// ============================================================================
// 3. STRESS TESTS: BitLinear Equivalence (bitlinear_i8_rvv vs bitlinear_i8_ime)
// ============================================================================

#[test]
fn test_challenge_bitlinear_tq1_tq2_ime_rvv_exhaustive_equivalence() {
    // Test across various row counts (including unaligned rows < 8, not multiple of 8, > 8)
    // and multi-block column sizes (256, 512, 1024).
    let test_configs = [
        (1, 256),  // Single row
        (3, 256),  // Unaligned < 8
        (7, 256),  // Unaligned < 8
        (8, 256),  // Exactly one TILE_N
        (11, 256), // Unaligned > 8
        (16, 256), // Multiple of 8
        (24, 256), // 3 tiles
        (4, 512),  // 2 blocks per row
        (8, 512),  // 2 blocks, 1 tile
        (7, 512),  // 2 blocks, unaligned rows
        (2, 1024), // 4 blocks per row
        (8, 1024), // 4 blocks, 1 tile
    ];

    for &(rows, cols) in &test_configs {
        let shape = [rows, cols];
        let act_i8: Vec<i8> = (0..cols)
            .map(|i| (((i * 17) % 255) as i16 - 128) as i8)
            .collect();
        let act_scale = 0.007_812_5_f32;

        // 1. Test TQ2_0
        let blocks_tq2 = (rows * cols) / TQ2_0_BLOCK_ELEMENTS;
        let mut raw_tq2 = vec![0u8; blocks_tq2 * TQ2_0_BLOCK_BYTES];
        for b in 0..blocks_tq2 {
            let base = b * TQ2_0_BLOCK_BYTES;
            let scale_val = (((b * 13) % 7 + 1) as f32) * 0.25;
            let d_bits = if (scale_val - 1.0).abs() < 1e-4 {
                0x3C00_u16
            } else {
                0x3800_u16
            };
            raw_tq2[base..base + 2].copy_from_slice(&d_bits.to_le_bytes());
            for k in 0..64 {
                raw_tq2[base + 2 + k] = ((b * 31 + k * 7) % 256) as u8;
            }
        }

        let view_tq2 = TensorView::from_bytes("w_tq2", QuantType::TQ2_0, &shape, 0, &raw_tq2)
            .expect("tq2 view");
        let mut out_ime_tq2 = vec![0.0_f32; rows];
        let mut out_rvv_tq2 = vec![0.0_f32; rows];

        bitlinear_i8_ime(&mut out_ime_tq2, &view_tq2, &act_i8, act_scale).expect("ime tq2");
        bitlinear_i8_rvv(&mut out_rvv_tq2, &view_tq2, &act_i8, act_scale).expect("rvv tq2");

        for r in 0..rows {
            assert_eq!(
                out_ime_tq2[r].to_bits(),
                out_rvv_tq2[r].to_bits(),
                "TQ2_0 bitwise mismatch at shape [{rows}, {cols}], row {r}: IME={} RVV={}",
                out_ime_tq2[r],
                out_rvv_tq2[r]
            );
        }

        // 2. Test TQ1_0
        let blocks_tq1 = (rows * cols) / TQ1_0_BLOCK_ELEMENTS;
        let mut raw_tq1 = vec![0u8; blocks_tq1 * TQ1_0_BLOCK_BYTES];
        for b in 0..blocks_tq1 {
            let base = b * TQ1_0_BLOCK_BYTES;
            let d_bits = 0x3C00_u16; // 1.0
            raw_tq1[base..base + 2].copy_from_slice(&d_bits.to_le_bytes());
            for k in 0..51 {
                raw_tq1[base + 2 + k] = ((b * 23 + k * 11) % 243) as u8;
            }
            raw_tq1[base + 53] = (b % 3) as u8;
        }

        let view_tq1 = TensorView::from_bytes("w_tq1", QuantType::TQ1_0, &shape, 0, &raw_tq1)
            .expect("tq1 view");
        let mut out_ime_tq1 = vec![0.0_f32; rows];
        let mut out_rvv_tq1 = vec![0.0_f32; rows];

        bitlinear_i8_ime(&mut out_ime_tq1, &view_tq1, &act_i8, act_scale).expect("ime tq1");
        bitlinear_i8_rvv(&mut out_rvv_tq1, &view_tq1, &act_i8, act_scale).expect("rvv tq1");

        for r in 0..rows {
            assert_eq!(
                out_ime_tq1[r].to_bits(),
                out_rvv_tq1[r].to_bits(),
                "TQ1_0 bitwise mismatch at shape [{rows}, {cols}], row {r}: IME={} RVV={}",
                out_ime_tq1[r],
                out_rvv_tq1[r]
            );
        }
    }
}

// ============================================================================
// 4. STRESS TESTS: Unpacking Robustness and Trit Extraction
// ============================================================================

#[test]
fn test_challenge_unpack_tq1_tq2_all_trits_and_fuzzing() {
    // A. TQ1_0: All -1s (byte = 0)
    let mut block_tq1_all_neg = [0u8; TQ1_0_BLOCK_BYTES];
    block_tq1_all_neg[0..2].copy_from_slice(&0x3C00_u16.to_le_bytes()); // 1.0
    let mut out_tq1 = [0_i8; TQ1_0_BLOCK_ELEMENTS];
    let s = unpack_tq1_0_block_i8(&block_tq1_all_neg, &mut out_tq1);
    assert_eq!(s, 1.0);
    assert!(
        out_tq1.iter().all(|&v| v == -1),
        "Byte 0 in TQ1_0 must unpack to all -1"
    );

    // B. TQ1_0: All 0s (digit 1 -> 1 + 3 + 9 + 27 + 81 = 121)
    let mut block_tq1_all_zero = [121u8; TQ1_0_BLOCK_BYTES];
    block_tq1_all_zero[0..2].copy_from_slice(&0x3C00_u16.to_le_bytes());
    block_tq1_all_zero[53] = 1; // 256th element
    let _ = unpack_tq1_0_block_i8(&block_tq1_all_zero, &mut out_tq1);
    assert!(
        out_tq1.iter().all(|&v| v == 0),
        "Byte 121 in TQ1_0 must unpack to all 0"
    );

    // C. TQ1_0: All +1s (digit 2 -> 2 + 6 + 18 + 54 + 162 = 242)
    let mut block_tq1_all_pos = [242u8; TQ1_0_BLOCK_BYTES];
    block_tq1_all_pos[0..2].copy_from_slice(&0x3C00_u16.to_le_bytes());
    block_tq1_all_pos[53] = 2; // 256th element
    let _ = unpack_tq1_0_block_i8(&block_tq1_all_pos, &mut out_tq1);
    assert!(
        out_tq1.iter().all(|&v| v == 1),
        "Byte 242 in TQ1_0 must unpack to all +1"
    );

    // D. TQ1_0: Out of range byte values (243..=255) must NOT panic and must stay in {-1, 0, 1}
    for bad_byte in 243..=255 {
        let mut block_fuzz = [bad_byte; TQ1_0_BLOCK_BYTES];
        block_fuzz[0..2].copy_from_slice(&0x3C00_u16.to_le_bytes());
        let _ = unpack_tq1_0_block_i8(&block_fuzz, &mut out_tq1);
        for &v in &out_tq1 {
            assert!(
                v == -1 || v == 0 || v == 1,
                "TQ1 out of range byte {bad_byte} unpacked to invalid trit {v}"
            );
        }
    }

    // E. TQ2_0: All -1s (byte = 0x00)
    let mut block_tq2_all_neg = [0u8; TQ2_0_BLOCK_BYTES];
    block_tq2_all_neg[0..2].copy_from_slice(&0x3C00_u16.to_le_bytes());
    let mut out_tq2 = [0_i8; TQ2_0_BLOCK_ELEMENTS];
    let _ = unpack_tq2_0_block_i8(&block_tq2_all_neg, &mut out_tq2);
    assert!(out_tq2.iter().all(|&v| v == -1));

    // F. TQ2_0: All 0s (byte = 0x55 = 0b01010101)
    let mut block_tq2_all_zero = [0x55_u8; TQ2_0_BLOCK_BYTES];
    block_tq2_all_zero[0..2].copy_from_slice(&0x3C00_u16.to_le_bytes());
    let _ = unpack_tq2_0_block_i8(&block_tq2_all_zero, &mut out_tq2);
    assert!(out_tq2.iter().all(|&v| v == 0));

    // G. TQ2_0: All +1s (byte = 0xAA = 0b10101010)
    let mut block_tq2_all_pos = [0xAA_u8; TQ2_0_BLOCK_BYTES];
    block_tq2_all_pos[0..2].copy_from_slice(&0x3C00_u16.to_le_bytes());
    let _ = unpack_tq2_0_block_i8(&block_tq2_all_pos, &mut out_tq2);
    assert!(out_tq2.iter().all(|&v| v == 1));
}

// ============================================================================
// 5. END-TO-END CHAINING: SubLN -> BitLinear -> ReLU^2 -> SubLN -> BitLinear
// ============================================================================

#[test]
fn test_challenge_end_to_end_chaining_bitnet_ffn() {
    let hidden_dim = 256;
    let intermediate_dim = 512;
    let eps = 1e-5_f32;

    // Test across 3 scenarios: (1) normal random, (2) all zeros, (3) extreme large
    let scenarios = [
        ("normal_random", {
            (0..hidden_dim)
                .map(|i| ((i as f32) * 0.05).sin() * 2.0)
                .collect::<Vec<f32>>()
        }),
        ("all_zeros", vec![0.0_f32; hidden_dim]),
        ("extreme_large", vec![1e5_f32; hidden_dim]),
    ];

    let norm_w1 = vec![1.0_f32; hidden_dim];
    let norm_w2 = vec![1.0_f32; intermediate_dim];

    // Build synthetic TQ1_0 and TQ2_0 weights
    let shape_gate_up = [intermediate_dim, hidden_dim];
    let shape_down = [hidden_dim, intermediate_dim];

    let mut raw_gate = vec![0xAA_u8; (intermediate_dim * hidden_dim / 256) * TQ2_0_BLOCK_BYTES];
    for b in 0..(intermediate_dim * hidden_dim / 256) {
        let base = b * TQ2_0_BLOCK_BYTES;
        raw_gate[base..base + 2].copy_from_slice(&0x3C00_u16.to_le_bytes()); // 1.0
    }
    let w_gate = TensorView::from_bytes("gate", QuantType::TQ2_0, &shape_gate_up, 0, &raw_gate)
        .expect("gate view");
    let w_up = TensorView::from_bytes("up", QuantType::TQ2_0, &shape_gate_up, 0, &raw_gate)
        .expect("up view");

    let mut raw_down = vec![0x55_u8; (hidden_dim * intermediate_dim / 256) * TQ1_0_BLOCK_BYTES];
    for b in 0..(hidden_dim * intermediate_dim / 256) {
        let base = b * TQ1_0_BLOCK_BYTES;
        raw_down[base..base + 2].copy_from_slice(&0x3C00_u16.to_le_bytes()); // 1.0
    }
    let w_down = TensorView::from_bytes("down", QuantType::TQ1_0, &shape_down, 0, &raw_down)
        .expect("down view");

    for (name, x_in) in scenarios {
        // --- PATH A: Baseline RVV forward chaining ---
        let mut act1_i8 = vec![0_i8; hidden_dim];
        let scale1 = subln_quantize_absmax_i8(&mut act1_i8, &x_in, &norm_w1, eps)
            .unwrap_or_else(|e| panic!("SubLN 1 failed for {name}: {e:?}"));

        let mut gate_f32 = vec![0.0_f32; intermediate_dim];
        let mut up_f32 = vec![0.0_f32; intermediate_dim];
        bitlinear_i8_rvv(&mut gate_f32, &w_gate, &act1_i8, scale1)
            .unwrap_or_else(|e| panic!("Gate BitLinear RVV failed for {name}: {e:?}"));
        bitlinear_i8_rvv(&mut up_f32, &w_up, &act1_i8, scale1)
            .unwrap_or_else(|e| panic!("Up BitLinear RVV failed for {name}: {e:?}"));

        let mut act_relu2 = vec![0.0_f32; intermediate_dim];
        relu_squared_gate_f32(&mut act_relu2, &gate_f32, &up_f32)
            .unwrap_or_else(|e| panic!("ReLU^2 failed for {name}: {e:?}"));

        let mut act2_i8 = vec![0_i8; intermediate_dim];
        let scale2 = subln_quantize_absmax_i8(&mut act2_i8, &act_relu2, &norm_w2, eps)
            .unwrap_or_else(|e| panic!("SubLN 2 failed for {name}: {e:?}"));

        let mut out_rvv = vec![0.0_f32; hidden_dim];
        bitlinear_i8_rvv(&mut out_rvv, &w_down, &act2_i8, scale2)
            .unwrap_or_else(|e| panic!("Down BitLinear RVV failed for {name}: {e:?}"));

        // --- PATH B: IME accelerated forward chaining ---
        let mut gate_ime = vec![0.0_f32; intermediate_dim];
        let mut up_ime = vec![0.0_f32; intermediate_dim];
        bitlinear_i8_ime(&mut gate_ime, &w_gate, &act1_i8, scale1)
            .unwrap_or_else(|e| panic!("Gate BitLinear IME failed for {name}: {e:?}"));
        bitlinear_i8_ime(&mut up_ime, &w_up, &act1_i8, scale1)
            .unwrap_or_else(|e| panic!("Up BitLinear IME failed for {name}: {e:?}"));

        let mut act_relu2_ime = vec![0.0_f32; intermediate_dim];
        relu_squared_gate_f32(&mut act_relu2_ime, &gate_ime, &up_ime)
            .unwrap_or_else(|e| panic!("ReLU^2 IME failed for {name}: {e:?}"));

        let mut out_ime = vec![0.0_f32; hidden_dim];
        bitlinear_i8_ime(&mut out_ime, &w_down, &act2_i8, scale2)
            .unwrap_or_else(|e| panic!("Down BitLinear IME failed for {name}: {e:?}"));

        // Assert numerical sanity & bit-identical equivalence across full chain
        assert_eq!(
            gate_f32, gate_ime,
            "Gate output RVV vs IME mismatch for {name}"
        );
        assert_eq!(up_f32, up_ime, "Up output RVV vs IME mismatch for {name}");
        assert_eq!(
            act_relu2, act_relu2_ime,
            "ReLU^2 output RVV vs IME mismatch for {name}"
        );
        for (i, (&v_rvv, &v_ime)) in out_rvv.iter().zip(&out_ime).enumerate() {
            assert!(
                v_rvv.is_finite(),
                "Output [{i}] is non-finite for scenario {name}: {v_rvv}"
            );
            assert_eq!(
                v_rvv.to_bits(),
                v_ime.to_bits(),
                "Full FFN Chain mismatch at index {i} for {name}: RVV={v_rvv}, IME={v_ime}"
            );
        }
    }
}

// ============================================================================
// 6. PERFORMANCE & THROUGHPUT BENCHMARKS
// ============================================================================

#[test]
fn test_challenge_m3_performance_and_throughput() {
    let dim = 256;
    let eps = 1e-5_f32;
    let iterations = 1000;

    let input = vec![1.23_f32; dim];
    let norm_w = vec![1.0_f32; dim];
    let mut out_i8 = vec![0_i8; dim];

    // Benchmark SubLN
    let t0 = Instant::now();
    for _ in 0..iterations {
        let _ = subln_quantize_absmax_i8(&mut out_i8, &input, &norm_w, eps);
    }
    let subln_elapsed = t0.elapsed();
    let subln_per_call = subln_elapsed.as_nanos() as f64 / (iterations as f64);

    // Benchmark ReLU^2
    let gate = vec![1.5_f32; dim];
    let up = vec![2.0_f32; dim];
    let mut out_relu = vec![0.0_f32; dim];
    let t1 = Instant::now();
    for _ in 0..iterations {
        let _ = relu_squared_gate_f32(&mut out_relu, &gate, &up);
    }
    let relu_elapsed = t1.elapsed();
    let relu_per_call = relu_elapsed.as_nanos() as f64 / (iterations as f64);

    // Benchmark BitLinear RVV
    let raw_w = vec![0xAA_u8; dim * TQ2_0_BLOCK_BYTES];
    let shape = [dim, dim];
    let weights =
        TensorView::from_bytes("bench_w", QuantType::TQ2_0, &shape, 0, &raw_w).expect("bench view");
    let mut out_linear = vec![0.0_f32; dim];

    let t2 = Instant::now();
    for _ in 0..100 {
        let _ = bitlinear_i8_rvv(&mut out_linear, &weights, &out_i8, 0.01);
    }
    let bitlinear_elapsed = t2.elapsed();
    let bitlinear_per_call = bitlinear_elapsed.as_nanos() as f64 / 100.0;

    eprintln!(
        "[PERF] SubLN (256): {subln_per_call:.1} ns/token | ReLU^2 (256): {relu_per_call:.1} ns/token | BitLinear GEMV (256x256): {bitlinear_per_call:.1} ns/call"
    );

    // Assert that latency is comfortably within real-time budgets (< 100 microseconds per call in release, relaxed in unoptimized debug)
    let max_subln = if cfg!(debug_assertions) {
        1_000_000.0
    } else {
        100_000.0
    };
    let max_relu = if cfg!(debug_assertions) {
        1_000_000.0
    } else {
        100_000.0
    };
    let max_bitlinear = if cfg!(debug_assertions) {
        10_000_000.0
    } else {
        1_000_000.0
    };

    assert!(
        subln_per_call < max_subln,
        "SubLN latency too high: {subln_per_call} ns"
    );
    assert!(
        relu_per_call < max_relu,
        "ReLU^2 latency too high: {relu_per_call} ns"
    );
    assert!(
        bitlinear_per_call < max_bitlinear,
        "BitLinear GEMV latency too high: {bitlinear_per_call} ns"
    );
}
