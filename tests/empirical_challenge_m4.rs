//! Empirical Challenge Test Suite for Milestone 4 Epic 1
//!
//! Stress-tests and empirical verification of RoPE, SwiGLU, BF16/F16, Q8_0/Q6_K, and slice_rows.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::doc_markdown,
    clippy::undocumented_unsafe_blocks,
    clippy::ptr_as_ptr,
    clippy::needless_range_loop,
    clippy::float_cmp,
    clippy::too_many_lines
)]

use jero_core::{K3Error, QuantType, TensorView};
use jero_kernels_rvv::{
    Q6_K_BLOCK_BYTES, Q6_K_BLOCK_ELEMENTS, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMENTS,
    RopeScalingConfig, apply_rope_inplace, bf16_to_f32_finite, dequantize_q6_k_block,
    dequantize_q8_0_block, f16_to_f32_finite, precompute_rope_inv_freq, swiglu_f32, vec_dot_q6_k,
    vec_dot_q8_0,
};
use std::f32::consts::PI;

// ============================================================================
// 1. RoPE FREQUENCY & BOUNDARY CHALLENGES
// ============================================================================

#[test]
fn test_challenge_rope_position_boundaries_and_extreme_positions() {
    let head_dim = 128;
    let theta_base = 10_000.0_f32;

    // A. pos = 0: Must be EXACT mathematical identity
    let mut qk_zero = vec![1.2345_f32; head_dim * 4];
    let qk_orig = qk_zero.clone();
    apply_rope_inplace(&mut qk_zero, head_dim, 0, 4, theta_base, None).expect("pos 0 must succeed");
    assert_eq!(qk_zero, qk_orig, "pos 0 must be exact bitwise identity");

    // B. pos = 1: Test analytical rotation for pair 0 and pair d/2 - 1
    let mut qk_pos1 = vec![0.0_f32; head_dim];
    // Head 0: pair 0 at index (0, 64), pair 63 at index (63, 127)
    qk_pos1[0] = 1.0;
    qk_pos1[64] = 0.0;
    qk_pos1[63] = 0.0;
    qk_pos1[127] = 1.0;

    apply_rope_inplace(&mut qk_pos1, head_dim, 1, 1, theta_base, None).expect("pos 1 must succeed");

    // Pair 0 has freq = 1.0. Angle = 1.0 rad.
    // x1' = 1.0 * cos(1) - 0.0 * sin(1) = cos(1)
    // x2' = 1.0 * sin(1) + 0.0 * cos(1) = sin(1)
    assert!((qk_pos1[0] - 1.0_f32.cos()).abs() < 1e-6);
    assert!((qk_pos1[64] - 1.0_f32.sin()).abs() < 1e-6);

    // C. Extreme positions: pos = 100_000 and pos = 1_000_000
    for &extreme_pos in &[10_000, 100_000, 1_000_000] {
        let mut qk_extreme = vec![0.5_f32; head_dim];
        apply_rope_inplace(
            &mut qk_extreme,
            head_dim,
            extreme_pos,
            extreme_pos,
            theta_base,
            None,
        )
        .unwrap_or_else(|_| panic!("extreme pos {extreme_pos} should succeed"));

        for (idx, &val) in qk_extreme.iter().enumerate() {
            assert!(
                val.is_finite(),
                "extreme pos {extreme_pos} produced non-finite value at index {idx}: {val}"
            );
            assert!(
                val.abs() <= 1.0,
                "extreme pos {extreme_pos} rotation magnitude exceeded bound: {val}"
            );
        }
    }
}

#[test]
fn test_challenge_llama3_smooth_rope_divergence_from_canonical_meta() {
    let head_dim = 128;
    let theta = 500_000.0_f32;
    let factor = 8.0_f32;
    let low_freq_factor = 1.0_f32;
    let high_freq_factor = 4.0_f32;
    let original_max_position_embeddings = 8192;

    let config = RopeScalingConfig::Llama3Smooth {
        factor,
        low_freq_factor,
        high_freq_factor,
        original_max_position_embeddings,
    };

    let k3_inv_freq = precompute_rope_inv_freq(head_dim, theta, Some(&config), 0)
        .expect("precompute llama3 smooth");

    let orig_ctx_f32 = original_max_position_embeddings as f32;
    let low_freq_wavelen = orig_ctx_f32 / low_freq_factor; // 8192.0
    let high_freq_wavelen = orig_ctx_f32 / high_freq_factor; // 2048.0

    let mut max_rel_divergence = 0.0_f32;
    let mut medium_freq_pairs = 0;

    for i in 0..(head_dim / 2) {
        let exponent = (2 * i) as f32 / head_dim as f32;
        let base_theta_i = 1.0_f32 / theta.powf(exponent);
        let wavelen = (2.0_f32 * PI) / base_theta_i;

        // Canonical Meta Llama 3 / llama.cpp reference formula:
        let meta_inv_freq = if wavelen < high_freq_wavelen {
            base_theta_i
        } else if wavelen > low_freq_wavelen {
            base_theta_i / factor
        } else {
            let smooth =
                (orig_ctx_f32 / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
            (1.0 - smooth) * (base_theta_i / factor) + smooth * base_theta_i
        };

        let k3_val = k3_inv_freq[i];

        if wavelen >= high_freq_wavelen && wavelen <= low_freq_wavelen {
            medium_freq_pairs += 1;
            let rel_diff = (meta_inv_freq - k3_val).abs() / meta_inv_freq;
            if rel_diff > max_rel_divergence {
                max_rel_divergence = rel_diff;
            }
        }
    }

    println!(
        "Llama-3 Smooth RoPE: {medium_freq_pairs} medium frequency pairs, max relative divergence = {:.2}%",
        max_rel_divergence * 100.0
    );

    // Empirical observation: k3 uses wavelength-linear scale inversion rather than frequency-linear interpolation,
    // resulting in up to 27.73% relative divergence in the medium-frequency transition band.
    assert!(
        max_rel_divergence < 1e-6,
        "Confirmed exact parity with canonical Meta Llama 3 frequency interpolation: {max_rel_divergence}"
    );
}

#[test]
fn test_challenge_longrope_context_switch_and_factor_lengths() {
    let head_dim = 64;
    let dim_pairs = head_dim / 2;
    let orig_ctx = 4096;

    // Test factor length mismatch
    let invalid_short = vec![1.0_f32; dim_pairs - 1];
    let valid_long = vec![2.0_f32; dim_pairs];
    let bad_config = RopeScalingConfig::LongRope {
        short_factor: invalid_short,
        long_factor: valid_long,
        original_max_position_embeddings: orig_ctx,
    };
    assert!(matches!(
        precompute_rope_inv_freq(head_dim, 10000.0, Some(&bad_config), 100),
        Err(K3Error::CapabilityMismatch { .. })
    ));

    // Test context switching boundary at pos = orig_ctx (4096) vs 4097
    let short_factors: Vec<f32> = (0..dim_pairs).map(|i| 1.0 + (i as f32) * 0.1).collect();
    let long_factors: Vec<f32> = (0..dim_pairs).map(|i| 2.0 + (i as f32) * 0.2).collect();

    let valid_config = RopeScalingConfig::LongRope {
        short_factor: short_factors.clone(),
        long_factor: long_factors.clone(),
        original_max_position_embeddings: orig_ctx,
    };

    let freq_at_bound = precompute_rope_inv_freq(head_dim, 10000.0, Some(&valid_config), orig_ctx)
        .expect("freq at 4096");
    let freq_past_bound =
        precompute_rope_inv_freq(head_dim, 10000.0, Some(&valid_config), orig_ctx + 1)
            .expect("freq at 4097");

    for i in 0..dim_pairs {
        let base = 1.0 / 10000.0_f32.powf((2 * i) as f32 / head_dim as f32);
        let expected_short = base / short_factors[i];
        let expected_long = base / long_factors[i];

        assert!((freq_at_bound[i] - expected_short).abs() < 1e-6);
        assert!((freq_past_bound[i] - expected_long).abs() < 1e-6);
        assert!(
            (freq_at_bound[i] - freq_past_bound[i]).abs() > 1e-6,
            "discontinuity exists at orig_ctx boundary"
        );
    }
}

// ============================================================================
// 2. SwiGLU NUMERICAL STABILITY & BOUNDARY CHALLENGES
// ============================================================================

#[test]
fn test_challenge_swiglu_boundary_continuity_and_extreme_values() {
    let mut out = [0.0_f32; 1];

    // 1. Boundary at g = -20.0
    // Test g just above -20.0 (e.g. -19.99999) vs just below (e.g. -20.00001)
    let g_above = -19.99999_f32;
    let g_below = -20.00001_f32;
    let u = 1.0_f32;

    swiglu_f32(&mut out, &[g_above], &[u]).unwrap();
    let out_above = out[0];

    swiglu_f32(&mut out, &[g_below], &[u]).unwrap();
    let out_below = out[0];

    let jump_step = (out_above - out_below).abs();
    println!("SwiGLU jump discontinuity at -20.0: {jump_step:.6e}");
    assert_eq!(out_below, 0.0);
    assert!(
        jump_step < 1e-7,
        "Jump at -20.0 is within 1e-7 precision: {jump_step}"
    );

    // 2. Boundary at g = +20.0
    let g_pos_below = 19.99999_f32;
    let g_pos_above = 20.00001_f32;
    swiglu_f32(&mut out, &[g_pos_below], &[u]).unwrap();
    let out_pos_below = out[0];
    swiglu_f32(&mut out, &[g_pos_above], &[u]).unwrap();
    let out_pos_above = out[0];
    assert!((out_pos_below - 19.99999).abs() < 1e-4);
    assert!((out_pos_above - 20.00001).abs() < 1e-4);

    // 3. Extreme gate values
    let extreme_gates = [-1e10_f32, -1000.0, -88.0, 50.0, 1000.0, 1e10];
    let extreme_ups = [1.0_f32, -2.5, 0.0, 10.0, -1.0, 0.001];
    let mut out_ext = vec![0.0_f32; extreme_gates.len()];

    swiglu_f32(&mut out_ext, &extreme_gates, &extreme_ups).expect("extreme swiglu succeeds");

    for (i, &g) in extreme_gates.iter().enumerate() {
        let val = out_ext[i];
        assert!(
            val.is_finite(),
            "swiglu output must be finite for finite inputs: {val}"
        );
        if g <= -20.0 {
            assert_eq!(
                val, 0.0,
                "swiglu output must be exactly 0.0 for gate <= -20.0"
            );
        } else if g >= 20.0 {
            let expected = g * extreme_ups[i];
            assert!((val - expected).abs() < 1e-4 * expected.abs().max(1.0));
        }
    }
}

#[test]
fn test_challenge_swiglu_nan_inf_poisoning() {
    let mut out = [0.0_f32; 3];
    let gates = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
    let ups = [1.0_f32, 1.0, 1.0];

    swiglu_f32(&mut out, &gates, &ups).expect("swiglu evaluates");

    // NaN gate produces NaN (unhandled)
    assert!(out[0].is_nan(), "NaN gate propagates NaN");
    // +Inf gate evaluates to +Inf because +Inf > 20.0
    assert_eq!(out[1], f32::INFINITY);
    // -Inf gate evaluates to 0.0 because -Inf < -20.0
    assert_eq!(out[2], 0.0);
}

// ============================================================================
// 3. BF16 & F16 EXHAUSTIVE ORACLES (65,536 PATTERNS)
// ============================================================================

#[test]
fn test_challenge_bf16_exhaustive_oracle() {
    let mut clamped_count = 0;
    let mut finite_count = 0;

    for bits in 0..=u16::MAX {
        let val = bf16_to_f32_finite(bits);
        let exp = (bits >> 7) & 0xFF;
        let sign = (bits >> 15) & 1;

        if exp == 0xFF {
            clamped_count += 1;
            let _ = sign;
            assert_eq!(
                val.to_bits(),
                0.0_f32.to_bits(),
                "BF16 non-finite 0x{bits:04X} must clamp to 0.0"
            );
        } else {
            finite_count += 1;
            assert!(
                val.is_finite(),
                "BF16 finite 0x{bits:04X} must produce finite f32"
            );
            let expected_f32 = f32::from_bits(u32::from(bits) << 16);
            assert_eq!(val.to_bits(), expected_f32.to_bits());
        }
    }

    assert_eq!(
        clamped_count, 256,
        "BF16 has exactly 256 non-finite patterns"
    );
    assert_eq!(
        finite_count, 65280,
        "BF16 has exactly 65280 finite patterns"
    );
}

#[test]
fn test_challenge_f16_exhaustive_oracle() {
    let mut clamped_count = 0;
    let mut subnormal_count = 0;
    let mut normal_count = 0;

    for bits in 0..=u16::MAX {
        let val = f16_to_f32_finite(bits);
        let sign = (bits >> 15) & 1;
        let exp = (bits >> 10) & 0x1F;
        let mant = bits & 0x03FF;

        if exp == 0x1F {
            clamped_count += 1;
            let expected = 0.0_f32;
            assert_eq!(
                val.to_bits(),
                expected.to_bits(),
                "F16 non-finite 0x{bits:04X} must clamp to {expected}"
            );
        } else if exp == 0 {
            if mant == 0 {
                // Zero
                assert_eq!(val.abs(), 0.0);
            } else {
                subnormal_count += 1;
                assert!(val.is_finite());
                // Subnormal analytical value: (-1)^sign * 2^(-24) * mant
                let analytical = (if sign == 1 { -1.0_f32 } else { 1.0_f32 })
                    * (2.0_f32.powi(-24))
                    * (mant as f32);
                assert_eq!(
                    val.to_bits(),
                    analytical.to_bits(),
                    "F16 subnormal 0x{bits:04X} must match analytical subnormal bitwise"
                );
            }
        } else {
            normal_count += 1;
            assert!(val.is_finite());
        }
    }

    assert_eq!(
        clamped_count, 2048,
        "F16 has exactly 2048 non-finite patterns (exp=0x1F, 2*1024 mantissas)"
    );
    assert_eq!(
        subnormal_count, 2046,
        "F16 has exactly 2046 subnormal patterns"
    );
    assert_eq!(
        normal_count, 61440,
        "F16 has exactly 61440 normal patterns (30*1024*2)"
    );
}

// ============================================================================
// 4. Q8_0 & Q6_K DEQUANTIZATION & VECTOR DOT CHALLENGES
// ============================================================================

#[test]
fn test_challenge_q8_0_scale_and_quants_bounds() {
    let mut block = [0u8; Q8_0_BLOCK_BYTES];
    let mut out = [0.0_f32; Q8_0_BLOCK_ELEMENTS];

    // 1. Zero scale (+0.0 and -0.0 in f16)
    block[0..2].copy_from_slice(&0x0000_u16.to_le_bytes());
    block[2..].fill(127);
    dequantize_q8_0_block(&block, &mut out);
    assert!(
        out.iter().all(|&v| v == 0.0),
        "zero scale must yield all zeros"
    );

    block[0..2].copy_from_slice(&0x8000_u16.to_le_bytes()); // -0.0
    dequantize_q8_0_block(&block, &mut out);
    assert!(
        out.iter().all(|&v| v == 0.0),
        "negative zero scale must yield all zeros"
    );

    // 2. Extreme quants: i8::MIN (-128) and i8::MAX (127)
    // Scale d = 1.0 in f16 (0x3C00)
    block[0..2].copy_from_slice(&0x3C00_u16.to_le_bytes());
    block[2] = 0x80; // -128 as i8
    block[3] = 0x7F; // 127 as i8
    block[4] = 0x00; // 0 as i8
    dequantize_q8_0_block(&block, &mut out);
    assert_eq!(out[0], -128.0);
    assert_eq!(out[1], 127.0);
    assert_eq!(out[2], 0.0);

    // 3. Scale NaN/Inf: Clamped to 0.0 by f16_to_f32_finite
    block[0..2].copy_from_slice(&0x7E00_u16.to_le_bytes()); // NaN
    dequantize_q8_0_block(&block, &mut out);
    assert_eq!(out[0], 0.0);
    assert_eq!(out[1], 0.0);
}

#[test]
fn test_challenge_q6_k_dequantization_comprehensive() {
    let mut block = [0u8; Q6_K_BLOCK_BYTES];
    let mut out = [0.0_f32; Q6_K_BLOCK_ELEMENTS];

    // Scale d = 2.0 in f16 (0x4000)
    block[208..210].copy_from_slice(&0x4000_u16.to_le_bytes());

    // Set scales to -128 (min i8) and +127 (max i8)
    block[192] = 0x80; // sc1 = -128
    block[193] = 0x7F; // sc1 for is=1

    // Lower nibbles and high bits:
    // For l = 0: q1 = (ql[0] & 0xF) | ((qh[0] & 3) << 4) - 32
    // Let ql[0] = 0x0F, qh[0] = 0x03 -> q1 = 15 | (3 << 4) - 32 = 63 - 32 = 31 (max 6-bit quant)
    block[0] = 0x0F;
    block[128] = 0x03;

    dequantize_q6_k_block(&block, &mut out);

    // Test vec_dot_q6_k on dequantized block
    let q6_shape = [1, Q6_K_BLOCK_ELEMENTS];
    let view_q6 = TensorView::from_bytes("test_q6_mat", QuantType::Q6_K, &q6_shape, 0, &block)
        .expect("valid q6 view");
    let input = vec![1.0_f32; Q6_K_BLOCK_ELEMENTS];
    let mut dot_out = [0.0_f32; 1];
    vec_dot_q6_k(&mut dot_out, &view_q6, &input).expect("vec_dot_q6_k succeeds");
    let expected_sum: f32 = out.iter().sum();
    assert!(
        (dot_out[0] - expected_sum).abs() < 1e-2,
        "vec_dot_q6_k ({}) must match sum of dequantized elements ({expected_sum})",
        dot_out[0]
    );
}

#[test]
fn test_challenge_q8_0_vec_dot_matrix() {
    const ROWS: usize = 3;
    const COLS: usize = 32;
    let mut backing = vec![0u8; ROWS * Q8_0_BLOCK_BYTES];
    // Fill row 0 with d = 1.0, quants = 1
    // Fill row 1 with d = 2.0, quants = 2
    // Fill row 2 with d = 0.5, quants = -4
    backing[0..2].copy_from_slice(&0x3C00_u16.to_le_bytes()); // 1.0
    backing[2..34].fill(1);

    let offset1 = Q8_0_BLOCK_BYTES;
    backing[offset1..offset1 + 2].copy_from_slice(&0x4000_u16.to_le_bytes()); // 2.0
    backing[offset1 + 2..offset1 + 34].fill(2);

    let offset2 = 2 * Q8_0_BLOCK_BYTES;
    backing[offset2..offset2 + 2].copy_from_slice(&0x3800_u16.to_le_bytes()); // 0.5
    backing[offset2 + 2..offset2 + 34].fill(0xFC); // -4 as i8

    let shape = [ROWS, COLS];
    let view = TensorView::from_bytes("q8_mat_3x32", QuantType::Q8_0, &shape, 0, &backing)
        .expect("valid q8 matrix view");

    let input = vec![1.0_f32; COLS];
    let mut output = [0.0_f32; ROWS];
    vec_dot_q8_0(&mut output, &view, &input).expect("vec_dot_q8_0 succeeds");

    // Row 0: 32 * (1.0 * 1) = 32.0
    assert!((output[0] - 32.0).abs() < 1e-4);
    // Row 1: 32 * (2.0 * 2) = 128.0
    assert!((output[1] - 128.0).abs() < 1e-4);
    // Row 2: 32 * (0.5 * -4) = -64.0
    assert!((output[2] - (-64.0)).abs() < 1e-4);
}

// ============================================================================
// 5. TensorView::slice_rows ZERO-COPY & STRUCTURAL CHALLENGES
// ============================================================================

#[test]
fn test_challenge_slice_rows_prime_and_non_standard_dimensions() {
    // Prime row counts: 17 rows, slice 7 rows
    let cols = 256;
    let rows = 17; // prime
    let byte_size = rows * 144; // Q4_K_M
    let mut buffer = vec![0u8; byte_size];

    // Fill each row with unique byte
    for r in 0..rows {
        buffer[r * 144..(r + 1) * 144].fill(r as u8 + 1);
    }

    let q4_shape = [cols, rows];
    let view = TensorView::from_bytes("q4_prime_matrix", QuantType::Q4_K_M, &q4_shape, 0, &buffer)
        .expect("valid tensor view");

    // Slice rows 5..12 (7 rows, prime count)
    let sliced = view.slice_rows(5, 7).expect("slice 7 rows from 17");
    assert_eq!(sliced.shape(), &[256, 7]);
    assert_eq!(sliced.offset(), 5 * 144);
    assert_eq!(sliced.byte_len(), 7 * 144);
    assert_eq!(sliced.num_elements(), 7 * 256);
    assert_eq!(sliced.as_bytes()[0], 6); // row 5 has fill byte 6
    assert_eq!(sliced.as_bytes()[7 * 144 - 1], 12); // row 11 has fill byte 12

    // Test prime columns with F32: [7, 13] (both prime)
    let f32_buffer = vec![0.0_f32; 7 * 13];
    let f32_bytes = unsafe {
        std::slice::from_raw_parts(f32_buffer.as_ptr() as *const u8, f32_buffer.len() * 4)
    };
    let f32_shape = [7, 13];
    let view_f32 =
        TensorView::from_bytes("f32_prime_mat", QuantType::F32, &f32_shape, 0, f32_bytes)
            .expect("valid f32 prime matrix");
    let sliced_f32 = view_f32.slice_rows(2, 5).expect("slice 5 rows of 7 cols");
    assert_eq!(sliced_f32.shape(), &[7, 5]);
    assert_eq!(sliced_f32.offset(), 2 * 7 * 4);
    assert_eq!(sliced_f32.byte_len(), 5 * 7 * 4);
    assert_eq!(sliced_f32.num_elements(), 35);
}

#[test]
fn test_challenge_slice_rows_boundary_and_zero_count() {
    let buffer = vec![0u8; 10 * 34]; // 10 rows of Q8_0
    let view = TensorView::from_bytes("q8_mat", QuantType::Q8_0, &[32, 10], 0, &buffer).unwrap();

    // 1. row_count = 0 at start
    let s0 = view.slice_rows(0, 0).expect("0 rows at 0");
    assert_eq!(s0.shape(), &[32, 0]);
    assert_eq!(s0.byte_len(), 0);
    assert_eq!(s0.num_elements(), 0);

    // 2. row_count = 0 at exact end (row_start = 10)
    let s_end = view.slice_rows(10, 0).expect("0 rows at end");
    assert_eq!(s_end.shape(), &[32, 0]);
    assert_eq!(s_end.offset(), 10 * 34);
    assert_eq!(s_end.byte_len(), 0);

    // 3. Overflow row_start = 11, count = 0
    assert!(matches!(
        view.slice_rows(11, 0),
        Err(K3Error::TensorBoundsError { .. })
    ));

    // 4. Exact full slice: row_start = 0, count = 10
    let s_full = view.slice_rows(0, 10).expect("full slice");
    assert_eq!(s_full.shape(), &[32, 10]);
    assert_eq!(s_full.byte_len(), 340);
    assert_eq!(s_full.num_elements(), 320);
}

#[test]
fn test_challenge_slice_rows_3d_tensor_structural_inconsistency() {
    // Structural challenge: A 3D tensor with shape [256, 4, 2] has 2048 elements.
    // In Q4_K_M, total storage is 2048 / 256 * 144 = 1152 bytes.
    let buffer = vec![0u8; 1152];
    let view_3d = TensorView::from_bytes("mat_3d", QuantType::Q4_K_M, &[256, 4, 2], 0, &buffer)
        .expect("valid 3D view");

    assert_eq!(view_3d.shape(), &[256, 4, 2]);
    assert_eq!(view_3d.num_elements(), 2048);
    assert_eq!(view_3d.byte_len(), 1152);

    // Slicing 2 rows must now be cleanly rejected:
    let err = view_3d
        .slice_rows(0, 2)
        .expect_err("slice_rows must reject 3D tensor");
    assert!(
        matches!(err, K3Error::TensorBoundsError { .. }),
        "slice_rows must return TensorBoundsError for >2D tensor"
    );
    assert!(
        err.to_string().contains("exactly 2 dimensions"),
        "error message must cite 2 dimensions requirement"
    );
}
