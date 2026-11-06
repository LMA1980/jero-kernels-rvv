//! Integration and Test-First Suite for Issue #41 (Milestone 3 / Milestone 4):
//! Dory 1.0 nGPT Kernels (`l2_norm_f32` and `ngpt_residual_lerp_f32`).
//!
//! Objectives:
//! 1. Test-First Red baseline compilation and functional verification.
//! 2. Exhaustive hardware length sweeps across Cluster 0 (X100 VLEN=256) and Cluster 1 (A100 VLEN=1024).
//! 3. Value distributions: constant, uniform pseudo-random, all-zeros, subnormals, and large dynamic ranges.
//! 4. Slicing and unaligned memory offsets.
//! 5. Exact mathematical boundaries for linear interpolation (alpha in {0.0, 0.5, 1.0} and extrapolation).
//! 6. Channelwise eigen-rate residual interpolation (`ngpt_residual_lerp_in_place`).
//! 7. Defensive bounds validation and error returns (`TensorBoundsError`).
//! 8. End-to-end recurrent nGPT hypersphere normalization simulation.

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    reason = "test indices, lengths, and floats fit in f32"
)]

use k3_core::K3Error;
use kernels_rvv::{
    l2_norm_f32, l2_norm_f32_in_place, l2_norm_f32_in_place_scalar, l2_norm_f32_scalar,
    l2_norm_inplace_f32, ngpt_residual_lerp_f32, ngpt_residual_lerp_f32_scalar,
    ngpt_residual_lerp_in_place, ngpt_residual_lerp_in_place_scalar,
    ngpt_residual_lerp_inplace_f32,
};

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
}

// ============================================================================
// High-Precision Double-Precision Reference Oracles
// ============================================================================
fn oracle_l2_norm_f64(x: &[f32], eps: f32) -> Vec<f32> {
    if x.is_empty() {
        return Vec::new();
    }
    let sum_sq: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let norm = (sum_sq + (eps as f64)).sqrt();
    if norm <= 0.0 || !norm.is_finite() {
        return vec![0.0_f32; x.len()];
    }
    x.iter().map(|&v| ((v as f64) / norm) as f32).collect()
}

fn oracle_lerp_f64(residual: &[f32], update: &[f32], alpha: f32) -> Vec<f32> {
    assert_eq!(residual.len(), update.len());
    let w_r = 1.0_f64 - (alpha as f64);
    let w_u = alpha as f64;
    residual
        .iter()
        .zip(update)
        .map(|(&r, &u)| (w_r * (r as f64) + w_u * (u as f64)) as f32)
        .collect()
}

// ============================================================================
// 1. Red Baseline Compilation & Basic Functionality
// ============================================================================
#[test]
fn test_red_baseline_compilation_and_basic_execution() {
    let x = [3.0_f32, 4.0, 0.0, 0.0];
    let mut out_norm = [0.0_f32; 4];
    // With eps = 0, sqrt(3^2 + 4^2) = 5.0 -> [0.6, 0.8, 0.0, 0.0]
    l2_norm_f32(&x, &mut out_norm, 0.0).expect("basic l2_norm succeeds");
    assert!((out_norm[0] - 0.6).abs() < 1e-5);
    assert!((out_norm[1] - 0.8).abs() < 1e-5);
    assert_eq!(out_norm[2], 0.0);
    assert_eq!(out_norm[3], 0.0);

    let r = [1.0_f32, 2.0, 3.0, 4.0];
    let u = [5.0_f32, 6.0, 7.0, 8.0];
    let mut out_lerp = [0.0_f32; 4];
    // With alpha = 0.25 -> 0.75 * [1,2,3,4] + 0.25 * [5,6,7,8] = [2, 3, 4, 5]
    ngpt_residual_lerp_f32(&r, &u, 0.25, &mut out_lerp).expect("basic lerp succeeds");
    assert!((out_lerp[0] - 2.0).abs() < 1e-5);
    assert!((out_lerp[1] - 3.0).abs() < 1e-5);
    assert!((out_lerp[2] - 4.0).abs() < 1e-5);
    assert!((out_lerp[3] - 5.0).abs() < 1e-5);
}

// ============================================================================
// 2. Exhaustive Length Sweeps across VLEN Boundaries
// ============================================================================
#[test]
fn test_l2_norm_length_sweep_and_vlen_boundaries() {
    let sweep_lengths = [
        1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511,
        512, 1023, 1024, 2048, 4096,
    ];
    let eps = 1e-6_f32;
    let mut rng = TestLcg::new(0x1234_5678_9ABC_DEF0);

    for &len in &sweep_lengths {
        let input: Vec<f32> = (0..len).map(|_| rng.next_f32(-2.0, 2.0)).collect();
        let mut out_dispatch = vec![f32::NAN; len];
        let mut out_scalar = vec![f32::NAN; len];
        let expected = oracle_l2_norm_f64(&input, eps);

        l2_norm_f32(&input, &mut out_dispatch, eps).expect("dispatch l2_norm succeeds");
        l2_norm_f32_scalar(&input, &mut out_scalar, eps).expect("scalar l2_norm succeeds");

        let tol = if len >= 2048 { 1e-4 } else { 1e-5 };
        for i in 0..len {
            assert!(
                !out_dispatch[i].is_nan(),
                "NaN produced in l2_norm at idx {i} for len {len}"
            );
            assert!(
                (out_dispatch[i] - expected[i]).abs() <= tol,
                "Mismatch at idx {i} for len {len}: dispatch={}, oracle={}, diff={}",
                out_dispatch[i],
                expected[i],
                (out_dispatch[i] - expected[i]).abs()
            );
            assert!(
                (out_dispatch[i] - out_scalar[i]).abs() <= tol,
                "Mismatch between dispatch and scalar at idx {i} for len {len}"
            );
        }

        // Verify output satisfies unit hypersphere constraint ||out||_2 ≈ 1.0
        let norm_res: f32 = out_dispatch.iter().map(|&v| v * v).sum::<f32>().sqrt();
        assert!(
            (norm_res - 1.0).abs() < 1e-4,
            "Unit norm violated for len {len}: got {norm_res}"
        );
    }
}

// ============================================================================
// 3. Value Distributions, Dynamic Ranges, and Edge Cases
// ============================================================================
#[test]
fn test_l2_norm_value_distributions_and_extremes() {
    let len = 256;
    let eps = 1e-6_f32;

    // A. Constant vectors: x_i = c -> out_i = sgn(c) / sqrt(N + eps/c^2)
    for &c in &[1.0_f32, -2.5, 0.25, 42.0] {
        let input = vec![c; len];
        let mut out = vec![0.0_f32; len];
        l2_norm_f32(&input, &mut out, eps).expect("constant input succeeds");

        let expected_val =
            (c as f64 / ((len as f64 * (c as f64).powi(2) + eps as f64).sqrt())) as f32;
        for (i, &val) in out.iter().enumerate() {
            assert!(
                (val - expected_val).abs() < 1e-5,
                "Constant c={c} mismatch at {i}: got {val}, expected {expected_val}"
            );
        }
    }

    // B. All zeros: with eps > 0 -> out = 0.0; with eps = 0.0 -> out = 0.0 (zero-norm clamp)
    let zero_input = vec![0.0_f32; len];
    let mut zero_out = vec![99.0_f32; len];
    l2_norm_f32(&zero_input, &mut zero_out, eps).expect("all-zero with eps succeeds");
    assert!(zero_out.iter().all(|&v| v == 0.0));

    let mut zero_out_eps0 = vec![99.0_f32; len];
    l2_norm_f32(&zero_input, &mut zero_out_eps0, 0.0).expect("all-zero with eps=0 clamped");
    assert!(zero_out_eps0.iter().all(|&v| v == 0.0));

    // C. Subnormals: values that square to 0 in f32
    let subnormal_input = vec![1e-38_f32; len];
    let mut sub_out = vec![0.0_f32; len];
    l2_norm_f32(&subnormal_input, &mut sub_out, eps).expect("subnormal input succeeds");
    assert!(sub_out.iter().all(|&v| v.is_finite() && !v.is_nan()));

    // D. In-Place normalization equivalence
    let mut in_place_buf: Vec<f32> = (0..len).map(|i| (i as f32) * 0.1 - 12.8).collect();
    let original = in_place_buf.clone();
    let mut separate_out = vec![0.0_f32; len];
    l2_norm_f32(&original, &mut separate_out, eps).expect("separate succeeds");
    l2_norm_inplace_f32(&mut in_place_buf, eps).expect("inplace succeeds");
    assert_eq!(in_place_buf, separate_out);

    // E. Issue #41 contract: l2_norm_f32_in_place
    let mut issue41_buf = original.clone();
    l2_norm_f32_in_place(&mut issue41_buf, eps);
    assert_eq!(issue41_buf, separate_out);

    // F. Scalar in-place equivalence
    let mut scalar_in_place_buf = original;
    l2_norm_f32_in_place_scalar(&mut scalar_in_place_buf, eps);
    assert_eq!(scalar_in_place_buf, separate_out);
}

// ============================================================================
// 4. Memory Alignment & Slicing Offsets
// ============================================================================
#[test]
fn test_l2_norm_unaligned_slices_and_offset_strides() {
    let base_len = 128;
    let eps = 1e-6_f32;
    let padding = 16;
    let full_len = base_len + padding;

    let mut full_input = vec![0.0_f32; full_len];
    for (i, v) in full_input.iter_mut().enumerate() {
        *v = (i as f32) * 0.25 - 16.0;
    }

    let mut full_out = vec![0.0_f32; full_len];

    // Sweep slice offsets 0..8 elements to verify unaligned vector load/store stability
    for offset in 0..8 {
        let input_slice = &full_input[offset..offset + base_len];
        let out_slice = &mut full_out[offset..offset + base_len];
        l2_norm_f32(input_slice, out_slice, eps).expect("unaligned slice succeeds");

        let expected = oracle_l2_norm_f64(input_slice, eps);
        for i in 0..base_len {
            assert!((out_slice[i] - expected[i]).abs() < 1e-5);
        }
    }
}

// ============================================================================
// 5. Defensive Bounds and Error Refusal (`l2_norm_f32`)
// ============================================================================
#[test]
fn test_l2_norm_defensive_bounds_and_errors() {
    let eps = 1e-6_f32;

    // A. Empty input
    let mut empty_out: [f32; 0] = [];
    assert!(matches!(
        l2_norm_f32(&[], &mut empty_out, eps),
        Err(K3Error::TensorBoundsError { .. })
    ));
    assert!(matches!(
        l2_norm_inplace_f32(&mut empty_out, eps),
        Err(K3Error::TensorBoundsError { .. })
    ));

    // l2_norm_f32_in_place on empty slice is a safe no-op per spec miner
    let mut empty_slice: [f32; 0] = [];
    l2_norm_f32_in_place(&mut empty_slice, eps);

    // B. Slice length mismatch
    let in_buf = [1.0_f32; 16];
    let mut short_out = [0.0_f32; 15];
    assert!(matches!(
        l2_norm_f32(&in_buf, &mut short_out, eps),
        Err(K3Error::TensorBoundsError { .. })
    ));

    let mut long_out = [0.0_f32; 17];
    assert!(matches!(
        l2_norm_f32(&in_buf, &mut long_out, eps),
        Err(K3Error::TensorBoundsError { .. })
    ));
}

// ============================================================================
// 6. Linear Interpolation Exact Alpha Boundaries (`ngpt_residual_lerp_f32`)
// ============================================================================
#[test]
fn test_ngpt_residual_lerp_alpha_boundaries_and_exact_identities() {
    let len = 64;
    let mut rng = TestLcg::new(0xCAFE_BABE_0000_1111);
    let r: Vec<f32> = (0..len).map(|_| rng.next_f32(-10.0, 10.0)).collect();
    let u: Vec<f32> = (0..len).map(|_| rng.next_f32(-10.0, 10.0)).collect();
    let mut out = vec![f32::NAN; len];

    // A. alpha = 0.0 -> exact bitwise identity with residual
    ngpt_residual_lerp_f32(&r, &u, 0.0, &mut out).expect("alpha=0.0 succeeds");
    for i in 0..len {
        assert_eq!(
            out[i].to_bits(),
            r[i].to_bits(),
            "alpha=0.0 must be exact bitwise identity of residual at {i}"
        );
    }

    // B. alpha = 1.0 -> exact bitwise identity with update
    ngpt_residual_lerp_f32(&r, &u, 1.0, &mut out).expect("alpha=1.0 succeeds");
    for i in 0..len {
        assert_eq!(
            out[i].to_bits(),
            u[i].to_bits(),
            "alpha=1.0 must be exact bitwise identity of update at {i}"
        );
    }

    // C. alpha = 0.5 -> arithmetic midpoint within 1 ULP
    ngpt_residual_lerp_f32(&r, &u, 0.5, &mut out).expect("alpha=0.5 succeeds");
    for i in 0..len {
        let expected = 0.5_f32 * r[i] + 0.5_f32 * u[i];
        assert!(
            (out[i] - expected).abs() < 1e-6,
            "alpha=0.5 midpoint mismatch at {i}: got {}, expected {expected}",
            out[i]
        );
    }

    // D. Out-of-bounds extrapolation: alpha = -0.5 and alpha = 1.5
    ngpt_residual_lerp_f32(&r, &u, -0.5, &mut out).expect("alpha=-0.5 succeeds");
    for i in 0..len {
        let expected = 1.5_f32 * r[i] - 0.5_f32 * u[i];
        assert!((out[i] - expected).abs() < 1e-5);
    }

    ngpt_residual_lerp_f32(&r, &u, 1.5, &mut out).expect("alpha=1.5 succeeds");
    for i in 0..len {
        let expected = -0.5_f32 * r[i] + 1.5_f32 * u[i];
        assert!((out[i] - expected).abs() < 1e-5);
    }

    // E. Equal vectors: r == u -> out == r for any alpha
    ngpt_residual_lerp_f32(&r, &r, 0.37, &mut out).expect("equal vectors succeed");
    for i in 0..len {
        assert!((out[i] - r[i]).abs() < 1e-6);
    }
}

// ============================================================================
// 7. Exhaustive Length Sweeps for Linear Interpolation
// ============================================================================
#[test]
fn test_ngpt_residual_lerp_length_sweep_and_vlen_boundaries() {
    let sweep_lengths = [
        1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511,
        512, 1023, 1024, 2048, 4096,
    ];
    let alpha = 0.333_333_f32;
    let mut rng = TestLcg::new(0x9876_5432_10FE_DCBA);

    for &len in &sweep_lengths {
        let r: Vec<f32> = (0..len).map(|_| rng.next_f32(-5.0, 5.0)).collect();
        let u: Vec<f32> = (0..len).map(|_| rng.next_f32(-5.0, 5.0)).collect();
        let mut out_dispatch = vec![f32::NAN; len];
        let mut out_scalar = vec![f32::NAN; len];
        let expected = oracle_lerp_f64(&r, &u, alpha);

        ngpt_residual_lerp_f32(&r, &u, alpha, &mut out_dispatch).expect("dispatch lerp succeeds");
        ngpt_residual_lerp_f32_scalar(&r, &u, alpha, &mut out_scalar)
            .expect("scalar lerp succeeds");

        for i in 0..len {
            assert!(
                !out_dispatch[i].is_nan(),
                "NaN produced in lerp at idx {i} for len {len}"
            );
            assert!(
                (out_dispatch[i] - expected[i]).abs() < 1e-6,
                "Mismatch at idx {i} for len {len}: got {}, expected {}",
                out_dispatch[i],
                expected[i]
            );
            assert_eq!(
                out_dispatch[i].to_bits(),
                out_scalar[i].to_bits(),
                "Bitwise difference between dispatch and scalar lerp at idx {i} for len {len}"
            );
        }

        // Test in-place lerp
        let mut in_place_r = r.clone();
        ngpt_residual_lerp_inplace_f32(&mut in_place_r, &u, alpha).expect("inplace lerp succeeds");
        assert_eq!(in_place_r, out_dispatch);
    }
}

// ============================================================================
// 8. Multi-Buffer Unaligned Slices for Interpolation
// ============================================================================
#[test]
fn test_ngpt_residual_lerp_unaligned_slices() {
    let base_len = 256;
    let padding = 16;
    let full_len = base_len + padding;

    let mut buf_r = vec![0.0_f32; full_len];
    let mut buf_u = vec![0.0_f32; full_len];
    let mut buf_out = vec![0.0_f32; full_len];

    for i in 0..full_len {
        buf_r[i] = (i as f32) * 0.1;
        buf_u[i] = (i as f32) * -0.2 + 5.0;
    }

    // Test different slice offsets for residual, update, and output buffers
    for offset_r in 0..4 {
        for offset_u in 0..4 {
            for offset_out in 0..4 {
                let r_slice = &buf_r[offset_r..offset_r + base_len];
                let u_slice = &buf_u[offset_u..offset_u + base_len];
                let out_slice = &mut buf_out[offset_out..offset_out + base_len];

                ngpt_residual_lerp_f32(r_slice, u_slice, 0.4, out_slice)
                    .expect("unaligned succeeds");
                let expected = oracle_lerp_f64(r_slice, u_slice, 0.4);
                for i in 0..base_len {
                    assert!((out_slice[i] - expected[i]).abs() < 1e-5);
                }
            }
        }
    }
}

// ============================================================================
// 9. Defensive Bounds and Error Refusal (`ngpt_residual_lerp_f32`)
// ============================================================================
#[test]
fn test_ngpt_residual_lerp_defensive_bounds_and_errors() {
    let mut empty_out: [f32; 0] = [];
    assert!(matches!(
        ngpt_residual_lerp_f32(&[], &[], 0.5, &mut empty_out),
        Err(K3Error::TensorBoundsError { .. })
    ));
    assert!(matches!(
        ngpt_residual_lerp_inplace_f32(&mut empty_out, &[], 0.5),
        Err(K3Error::TensorBoundsError { .. })
    ));

    let r = [1.0_f32; 16];
    let u_short = [2.0_f32; 15];
    let u_long = [2.0_f32; 17];
    let mut out = [0.0_f32; 16];
    let mut out_short = [0.0_f32; 15];

    // Update length mismatch
    assert!(matches!(
        ngpt_residual_lerp_f32(&r, &u_short, 0.5, &mut out),
        Err(K3Error::TensorBoundsError { .. })
    ));
    assert!(matches!(
        ngpt_residual_lerp_f32(&r, &u_long, 0.5, &mut out),
        Err(K3Error::TensorBoundsError { .. })
    ));

    // Output length mismatch
    assert!(matches!(
        ngpt_residual_lerp_f32(&r, &r, 0.5, &mut out_short),
        Err(K3Error::TensorBoundsError { .. })
    ));
}

// ============================================================================
// 10. Channelwise Eigen-Rate Residual Interpolation (Issue #41 Contract)
// ============================================================================
#[test]
fn test_ngpt_residual_lerp_channelwise_in_place_behavior() {
    let len = 64;
    let eps = 1e-6_f32;
    let mut rng = TestLcg::new(0xABCD_EF01_2345_6789);

    let residual: Vec<f32> = (0..len).map(|_| rng.next_f32(-2.0, 2.0)).collect();
    let cur_init: Vec<f32> = (0..len).map(|_| rng.next_f32(-2.0, 2.0)).collect();

    // A. alpha = 0.0 -> result is exactly L2Norm(residual)
    let mut cur = cur_init.clone();
    let alpha_zero = vec![0.0_f32; len];
    ngpt_residual_lerp_in_place(&residual, &mut cur, &alpha_zero, 1.0, eps)
        .expect("channelwise lerp succeeds");
    let mut expected_residual_norm = vec![0.0_f32; len];
    l2_norm_f32(&residual, &mut expected_residual_norm, eps).unwrap();
    for i in 0..len {
        assert!(
            (cur[i] - expected_residual_norm[i]).abs() < 1e-5,
            "Mismatch for alpha=0 at {i}: got {}, expected {}",
            cur[i],
            expected_residual_norm[i]
        );
    }

    // B. alpha = 1.0 -> result is exactly L2Norm(cur_init)
    let mut cur = cur_init.clone();
    let alpha_one = vec![1.0_f32; len];
    ngpt_residual_lerp_in_place(&residual, &mut cur, &alpha_one, 1.0, eps)
        .expect("channelwise lerp succeeds");
    let mut expected_cur_norm = vec![0.0_f32; len];
    l2_norm_f32(&cur_init, &mut expected_cur_norm, eps).unwrap();
    for i in 0..len {
        assert!(
            (cur[i] - expected_cur_norm[i]).abs() < 1e-5,
            "Mismatch for alpha=1 at {i}: got {}, expected {}",
            cur[i],
            expected_cur_norm[i]
        );
    }

    // C. Negative alpha_weight: should use absolute value |alpha_weight * alpha_scale|
    let mut cur_neg = cur_init.clone();
    let alpha_neg = vec![-0.5_f32; len];
    ngpt_residual_lerp_in_place(&residual, &mut cur_neg, &alpha_neg, 1.0, eps)
        .expect("negative alpha succeeds");

    let mut cur_pos = cur_init.clone();
    let alpha_pos = vec![0.5_f32; len];
    ngpt_residual_lerp_in_place(&residual, &mut cur_pos, &alpha_pos, 1.0, eps)
        .expect("positive alpha succeeds");

    for i in 0..len {
        assert!(
            (cur_neg[i] - cur_pos[i]).abs() < 1e-6,
            "Negative alpha did not match positive absolute value at {i}"
        );
    }

    // D. Equivalence between dispatch and scalar implementation
    let mut cur_dispatch = cur_init.clone();
    let mut cur_scalar = cur_init.clone();
    let alpha_rand: Vec<f32> = (0..len).map(|_| rng.next_f32(-1.5, 1.5)).collect();
    ngpt_residual_lerp_in_place(&residual, &mut cur_dispatch, &alpha_rand, 0.8, eps)
        .expect("dispatch channelwise succeeds");
    ngpt_residual_lerp_in_place_scalar(&residual, &mut cur_scalar, &alpha_rand, 0.8, eps)
        .expect("scalar channelwise succeeds");

    for i in 0..len {
        assert!(
            (cur_dispatch[i] - cur_scalar[i]).abs() < 1e-5,
            "Dispatch vs scalar mismatch at {i}: dispatch={}, scalar={}",
            cur_dispatch[i],
            cur_scalar[i]
        );
    }

    // E. Error handling: empty slices and dimension mismatch
    let mut empty_cur = [0.0_f32; 0];
    assert!(matches!(
        ngpt_residual_lerp_in_place(&[], &mut empty_cur, &[], 1.0, eps),
        Err(K3Error::TensorBoundsError { .. })
    ));

    let mut cur_short = vec![0.0_f32; len - 1];
    assert!(matches!(
        ngpt_residual_lerp_in_place(&residual, &mut cur_short, &alpha_zero, 1.0, eps),
        Err(K3Error::TensorBoundsError { .. })
    ));
}

// ============================================================================
// 11. Recurrent Block End-to-End Simulation on Unit Hypersphere
// ============================================================================
#[test]
fn test_ngpt_e2e_recurrent_block_simulation() {
    let dim = 1024; // Dory 1.0 nGPT standard hidden dimension
    let eps = 1e-6_f32;
    let mut rng = TestLcg::new(0x5555_AAAA_3333_7777);

    // Initial state vector x_0
    let x: Vec<f32> = (0..dim).map(|_| rng.next_f32(-1.0, 1.0)).collect();
    let mut x_norm = vec![0.0_f32; dim];
    l2_norm_f32(&x, &mut x_norm, eps).expect("initial norm");

    // Simulate 10 recurrent layer steps: x_{t+1} = L2Norm(LERP(x_t, L2Norm(W * x_t), alpha_t))
    for step in 0..10 {
        // Synthesize simulated transformation update u
        let u: Vec<f32> = (0..dim)
            .map(|i| x_norm[i] * 0.9 + rng.next_f32(-0.1, 0.1))
            .collect();
        let mut u_norm = vec![0.0_f32; dim];
        l2_norm_f32(&u, &mut u_norm, eps).expect("update norm");

        let alpha = 0.15_f32;
        let mut x_next = vec![0.0_f32; dim];
        ngpt_residual_lerp_f32(&x_norm, &u_norm, alpha, &mut x_next).expect("residual lerp");

        // Renormalize onto hypersphere
        l2_norm_f32(&x_next, &mut x_norm, eps).expect("re-normalization");

        // Invariant: ||x||_2 must remain strictly 1.0 ± 1e-4 across all recurrent steps
        let current_norm: f32 = x_norm.iter().map(|&v| v * v).sum::<f32>().sqrt();
        assert!(
            (current_norm - 1.0).abs() < 1e-4,
            "Hypersphere invariant violated at step {step}: norm={current_norm}"
        );
        for &v in &x_norm {
            assert!(
                v.is_finite() && !v.is_nan(),
                "Non-finite activation at step {step}"
            );
        }
    }
}
