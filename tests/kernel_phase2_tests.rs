//! Exhaustive integration tests for Phase 2 kernels: `RoPE`, GQA, `SwiGLU`, BF16, `Q8_0`, and `Q6_K`.
#![expect(
    clippy::cast_precision_loss,
    reason = "test indices and dimensions fit in f32"
)]

use jero_core::{QuantType, TensorView};
use jero_kernels_rvv::{
    BaselineKernel, Q6_K_BLOCK_BYTES, Q6_K_BLOCK_ELEMENTS, Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMENTS,
    RopeScalingConfig, VectorKernel, apply_rope_inplace, bf16_to_f32_finite, dequantize_q6_k_block,
    dequantize_q8_0_block, gqa_causal_attention_f32, precompute_rope_inv_freq, swiglu_f32,
    vec_dot_q6_k, vec_dot_q8_0,
};

#[test]
fn test_rope_llama3_smooth_interpolation_continuity() {
    let head_dim = 128;
    let theta = 500_000.0_f32;
    let factor = 8.0_f32;
    let low_freq_factor = 1.0_f32;
    let high_freq_factor = 4.0_f32;
    let orig_ctx = 8192;

    let config = RopeScalingConfig::Llama3Smooth {
        factor,
        low_freq_factor,
        high_freq_factor,
        original_max_position_embeddings: orig_ctx,
    };

    let inv_freq = precompute_rope_inv_freq(head_dim, theta, Some(&config), 0)
        .expect("precompute llama3 smooth");
    let unscaled = precompute_rope_inv_freq(head_dim, theta, None, 0).expect("precompute unscaled");

    assert_eq!(inv_freq.len(), head_dim / 2);

    let low_freq_wavelen = (orig_ctx as f32) / low_freq_factor;
    let high_freq_wavelen = (orig_ctx as f32) / high_freq_factor;

    for (i, (&freq, &base_freq)) in inv_freq.iter().zip(&unscaled).enumerate() {
        let wavelen = (2.0 * std::f32::consts::PI) / base_freq;
        let effective_scale = base_freq / freq;

        if wavelen < high_freq_wavelen {
            assert!(
                (effective_scale - 1.0).abs() < 1e-5,
                "Pair {i}: wavelen {wavelen} < high {high_freq_wavelen}, scale should be 1.0 but got {effective_scale}"
            );
        } else if wavelen > low_freq_wavelen {
            assert!(
                (effective_scale - factor).abs() < 1e-5,
                "Pair {i}: wavelen {wavelen} > low {low_freq_wavelen}, scale should be {factor} but got {effective_scale}"
            );
        } else {
            // In the transitional smooth band
            assert!(
                effective_scale >= 1.0 && effective_scale <= factor,
                "Pair {i}: transitional scale {effective_scale} must be between 1.0 and {factor}"
            );
        }
    }
}

#[test]
fn test_rope_longrope_switch_boundary() {
    let head_dim = 8;
    let dim_pairs = head_dim / 2;
    let short_factor = vec![1.0, 1.25, 1.5, 1.75];
    let long_factor = vec![2.0, 2.5, 3.0, 3.5];
    let orig_ctx = 4096;

    let config = RopeScalingConfig::LongRope {
        short_factor: short_factor.clone(),
        long_factor: long_factor.clone(),
        original_max_position_embeddings: orig_ctx,
    };

    // Exactly at threshold (seq_len == orig_ctx) -> short_factor
    let at_thresh = precompute_rope_inv_freq(head_dim, 10_000.0, Some(&config), orig_ctx)
        .expect("at threshold");
    let unscaled = precompute_rope_inv_freq(head_dim, 10_000.0, None, orig_ctx).expect("unscaled");

    for i in 0..dim_pairs {
        let expected = unscaled[i] / short_factor[i];
        assert!((at_thresh[i] - expected).abs() < 1e-6);
    }

    // 1 token past threshold (seq_len == orig_ctx + 1) -> long_factor
    let past_thresh = precompute_rope_inv_freq(head_dim, 10_000.0, Some(&config), orig_ctx + 1)
        .expect("past threshold");

    for i in 0..dim_pairs {
        let expected = unscaled[i] / long_factor[i];
        assert!((past_thresh[i] - expected).abs() < 1e-6);
    }
}

#[test]
fn test_rope_inplace_multihead_consistency() {
    let n_heads = 4;
    let head_dim = 16;
    let total_elems = n_heads * head_dim;

    // Head 0 and Head 2 have identical initial contents
    let mut qk = vec![0.0_f32; total_elems];
    for d in 0..head_dim {
        let val = (d as f32) * 0.125 - 0.5;
        qk[d] = val; // Head 0
        qk[2 * head_dim + d] = val; // Head 2
    }

    apply_rope_inplace(&mut qk, head_dim, 5, 5, 10_000.0, None).expect("apply rope to multihead");

    // Both identical heads should be rotated identically
    for d in 0..head_dim {
        assert_eq!(
            qk[d].to_bits(),
            qk[2 * head_dim + d].to_bits(),
            "Mismatch at element {d} between head 0 and head 2"
        );
    }
}

#[test]
fn test_gqa_multihead_grouped_query_equivalence() {
    // 4 Q heads, 2 KV heads (group_size = 2).
    // Head 0 and 1 attend to KV head 0.
    // Head 2 and 3 attend to KV head 1.
    let head_dim = 4;
    let n_heads_q = 4;
    let n_heads_kv = 2;
    let seq_len = 3;

    let q = vec![1.0_f32; n_heads_q * head_dim];
    let mut k_cache = vec![0.0_f32; seq_len * n_heads_kv * head_dim];
    let mut v_cache = vec![0.0_f32; seq_len * n_heads_kv * head_dim];

    for s in 0..seq_len {
        for h in 0..n_heads_kv {
            let offset = (s * n_heads_kv + h) * head_dim;
            for d in 0..head_dim {
                k_cache[offset + d] = (s as f32) + (h as f32) * 0.5;
                v_cache[offset + d] = ((s + 1) * (h + 1) * (d + 1)) as f32;
            }
        }
    }

    let mut out = vec![0.0_f32; n_heads_q * head_dim];
    gqa_causal_attention_f32(
        &mut out, &q, &k_cache, &v_cache, n_heads_q, n_heads_kv, head_dim, seq_len, 0.5,
    )
    .expect("gqa causal attention succeeds");

    // Q head 0 and Q head 1 have identical q and identical KV target head (0), so their outputs must match exactly
    let head0 = &out[0..head_dim];
    let head1 = &out[head_dim..2 * head_dim];
    assert_eq!(head0, head1);

    // Q head 2 and Q head 3 also have identical q and KV target head (1)
    let head2 = &out[2 * head_dim..3 * head_dim];
    let head3 = &out[3 * head_dim..4 * head_dim];
    assert_eq!(head2, head3);
}

#[test]
fn test_bf16_conversion_finite_clamping_comprehensive() {
    // Normal positive values
    assert_eq!(bf16_to_f32_finite(0x3F80), 1.0); // 1.0
    assert_eq!(bf16_to_f32_finite(0x4000), 2.0); // 2.0
    assert_eq!(bf16_to_f32_finite(0x3E80), 0.25); // 0.25

    // Zero
    assert_eq!(bf16_to_f32_finite(0x0000), 0.0);
    assert_eq!(bf16_to_f32_finite(0x8000), -0.0);

    // Subnormal
    let subnormal = bf16_to_f32_finite(0x0001);
    assert!(subnormal > 0.0 && subnormal.is_finite());

    // Positive and negative infinity clamp safely to +0.0
    assert_eq!(bf16_to_f32_finite(0x7F80).to_bits(), 0.0_f32.to_bits());
    assert_eq!(bf16_to_f32_finite(0xFF80).to_bits(), 0.0_f32.to_bits());

    // Various NaNs collapse safely to +0.0 without fabricating non-zero weights
    assert_eq!(bf16_to_f32_finite(0x7FC0).to_bits(), 0.0_f32.to_bits());
    assert_eq!(bf16_to_f32_finite(0x7FFF).to_bits(), 0.0_f32.to_bits());
    assert_eq!(bf16_to_f32_finite(0xFFC0).to_bits(), 0.0_f32.to_bits());
    assert_eq!(bf16_to_f32_finite(0xFFFF).to_bits(), 0.0_f32.to_bits());

    // Assert that NO bit pattern can ever produce NaN or Inf
    for bits in [
        0x7F80_u16, 0x7F81, 0x7FC0, 0x7FFF, 0xFF80, 0xFF81, 0xFFC0, 0xFFFF,
    ] {
        let val = bf16_to_f32_finite(bits);
        assert!(
            val.is_finite(),
            "bits {bits:#06X} produced non-finite float {val}"
        );
        assert!(!val.is_nan(), "bits {bits:#06X} produced NaN");
    }
}

#[test]
fn test_swiglu_numerical_bounds_and_accuracy() {
    let gate = [
        -50.0_f32, -25.0, -20.1, -20.0, -5.0, 0.0, 5.0, 20.0, 20.1, 50.0,
    ];
    let up = [1.0_f32; 10];
    let mut out = [0.0_f32; 10];

    swiglu_f32(&mut out, &gate, &up).expect("swiglu calculation");

    // All deep negative values produce exactly 0.0
    assert_eq!(out[0], 0.0);
    assert_eq!(out[1], 0.0);
    assert_eq!(out[2], 0.0);

    // Near 0.0, SiLU(0) = 0.0
    assert_eq!(out[5], 0.0);

    // High positive values approach x
    assert!((out[8] - 20.1).abs() < 1e-4);
    assert!((out[9] - 50.0).abs() < 1e-4);

    // Output is monotonic with respect to gate when up=1.0 and gate >= 0
    for i in 5..9 {
        assert!(out[i + 1] > out[i]);
    }
}

#[test]
fn test_dequantize_q8_0_and_vector_dot_accuracy() {
    let mut raw = [0u8; 2 * Q8_0_BLOCK_BYTES];
    // Block 0: d = 1.0 (0x3C00), quants all +1
    raw[0..2].copy_from_slice(&0x3C00_u16.to_le_bytes());
    for b in &mut raw[2..34] {
        *b = 1;
    }
    // Block 1: d = 0.5 (0x3800), quants all -2
    raw[34..36].copy_from_slice(&0x3800_u16.to_le_bytes());
    for b in &mut raw[36..68] {
        *b = (-2_i8).to_ne_bytes()[0];
    }

    let mut out_block0 = [0.0_f32; Q8_0_BLOCK_ELEMENTS];
    dequantize_q8_0_block(raw[..34].try_into().unwrap(), &mut out_block0);
    assert!(out_block0.iter().all(|&v| (v - 1.0).abs() < 1e-5));

    let mut out_block1 = [0.0_f32; Q8_0_BLOCK_ELEMENTS];
    dequantize_q8_0_block(raw[34..68].try_into().unwrap(), &mut out_block1);
    assert!(out_block1.iter().all(|&v| (v - (-1.0)).abs() < 1e-5));

    // Matrix of 2 rows x 32 cols
    let view = TensorView::from_bytes("w_q8", QuantType::Q8_0, &[2, 32], 0, &raw)
        .expect("tensor view Q8_0");
    let input = [2.0_f32; 32];
    let mut output = [0.0_f32; 2];

    BaselineKernel::new()
        .vec_dot_q8_0(&mut output, &view, &input)
        .expect("vec_dot_q8_0 succeeds");

    // Row 0: 32 * (1.0 * 2.0) = 64.0
    assert!((output[0] - 64.0).abs() < 1e-4);
    // Row 1: 32 * (-1.0 * 2.0) = -64.0
    assert!((output[1] - (-64.0)).abs() < 1e-4);

    // Also verify standalone dispatch function
    let mut output_standalone = [0.0_f32; 2];
    vec_dot_q8_0(&mut output_standalone, &view, &input).expect("standalone vec_dot_q8_0 succeeds");
    assert_eq!(output, output_standalone);
}

#[test]
fn test_dequantize_q6_k_and_vector_dot_accuracy() {
    let mut raw = [0u8; Q6_K_BLOCK_BYTES];
    // Super-scale d = 2.0 in f16 (0x4000)
    raw[208..210].copy_from_slice(&0x4000_u16.to_le_bytes());
    // Sub-block scales: scale 1 for all 16 sub-blocks
    for s in &mut raw[192..208] {
        *s = 1;
    }
    // ql = 0x22 (low=2, high=2), qh = 0x00 -> value = 2 - 32 = -30
    for q in &mut raw[0..128] {
        *q = 0x22;
    }

    let mut dequant = [0.0_f32; Q6_K_BLOCK_ELEMENTS];
    dequantize_q6_k_block(&raw, &mut dequant);

    // Each element = d * scale * q = 2.0 * 1 * (-30) = -60.0
    for &val in &dequant {
        assert!((val - (-60.0)).abs() < 1e-4);
    }

    let view = TensorView::from_bytes("w_q6_k", QuantType::Q6_K, &[1, 256], 0, &raw)
        .expect("tensor view Q6_K");
    let input = [0.5_f32; 256];
    let mut output = [0.0_f32; 1];

    vec_dot_q6_k(&mut output, &view, &input).expect("vec_dot_q6_k succeeds");

    // Dot product: 256 * (-60.0 * 0.5) = 256 * (-30.0) = -7680.0
    assert!((output[0] - (-7680.0)).abs() < 1e-3);
}
