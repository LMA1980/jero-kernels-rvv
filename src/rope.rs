//! Rotary Position Embeddings (`RoPE`) with standard linear, Llama-3 smooth, and `LongRoPE` scaling.
//!
//! Provides table precomputation and in-place vector rotation compatible with both
//! VLA RVV 1.0 (`zvl256b`/`zvl1024b`) strip-mining and portable scalar fallbacks.

use core::f32::consts::PI;
use k3_core::K3Error;

/// Configuration for `RoPE` frequency scaling methods.
#[derive(Debug, Clone, PartialEq)]
pub enum RopeScalingConfig {
    /// Linear frequency scaling (`inv_freq = inv_freq / factor`).
    Linear {
        /// Frequency scaling factor (> 0.0).
        factor: f32,
    },
    /// Llama-3 smooth frequency scaling with wavelength-dependent interpolation.
    Llama3Smooth {
        /// High-context frequency scaling factor (> 0.0).
        factor: f32,
        /// Low frequency bound factor (defines `lambda_low`).
        low_freq_factor: f32,
        /// High frequency bound factor (defines `lambda_high`).
        high_freq_factor: f32,
        /// Original pretraining context window limit.
        original_max_position_embeddings: usize,
    },
    /// Microsoft Phi-3 / Phi-4 `LongRoPE` with per-dimension short/long factor tables.
    LongRope {
        /// Per-dimension scaling factors applied when `seq_len <= original_max_position_embeddings`.
        short_factor: Vec<f32>,
        /// Per-dimension scaling factors applied when `seq_len > original_max_position_embeddings`.
        long_factor: Vec<f32>,
        /// Original pretraining context window threshold for factor switching.
        original_max_position_embeddings: usize,
    },
}

/// Precomputes inverse frequency table for rotary embeddings across dimension pairs $i \in [0, d/2)$.
///
/// # Errors
///
/// Returns [`K3Error::CapabilityMismatch`] if:
/// - `head_dim` is zero or odd,
/// - `theta` is non-positive or non-finite,
/// - scaling factors are non-positive, non-finite, or length does not match `head_dim / 2`.
#[expect(
    clippy::too_many_lines,
    reason = "comprehensive RoPE scaling dispatch across linear, smooth, and LongRoPE configs"
)]
pub fn precompute_rope_inv_freq(
    head_dim: usize,
    theta: f32,
    scaling: Option<&RopeScalingConfig>,
    seq_len: usize,
) -> Result<Vec<f32>, K3Error> {
    if head_dim == 0 || head_dim % 2 != 0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: "rope".to_owned(),
            reason: format!("head_dim must be positive and even, got {head_dim}"),
        });
    }

    if !theta.is_finite() || theta <= 0.0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: "rope".to_owned(),
            reason: format!("theta must be positive and finite, got {theta}"),
        });
    }

    let dim_pairs = head_dim / 2;
    let mut inv_freq = Vec::with_capacity(dim_pairs);

    #[expect(
        clippy::cast_precision_loss,
        reason = "head_dim fits comfortably in f32"
    )]
    let head_dim_f32 = head_dim as f32;

    for i in 0..dim_pairs {
        #[expect(
            clippy::cast_precision_loss,
            reason = "pair index fits comfortably in f32"
        )]
        let exponent = (2 * i) as f32 / head_dim_f32;
        let base_theta_i = 1.0_f32 / theta.powf(exponent);

        let scaled_theta_i = match scaling {
            None => base_theta_i,
            Some(RopeScalingConfig::Linear { factor }) => {
                if !factor.is_finite() || *factor <= 0.0 {
                    return Err(K3Error::CapabilityMismatch {
                        architecture: "rope".to_owned(),
                        reason: format!("linear factor must be positive and finite, got {factor}"),
                    });
                }
                base_theta_i / factor
            }
            Some(RopeScalingConfig::Llama3Smooth {
                factor,
                low_freq_factor,
                high_freq_factor,
                original_max_position_embeddings,
            }) => {
                if !factor.is_finite()
                    || *factor <= 0.0
                    || !low_freq_factor.is_finite()
                    || *low_freq_factor <= 0.0
                    || !high_freq_factor.is_finite()
                    || *high_freq_factor <= 0.0
                    || *original_max_position_embeddings == 0
                    || low_freq_factor >= high_freq_factor
                {
                    return Err(K3Error::CapabilityMismatch {
                        architecture: "rope".to_owned(),
                        reason: format!(
                            "invalid Llama3Smooth parameters: factor={factor}, low_freq_factor={low_freq_factor}, high_freq_factor={high_freq_factor}, orig_ctx={original_max_position_embeddings}"
                        ),
                    });
                }

                let wavelen = (2.0_f32 * PI) / base_theta_i;
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "orig_ctx fits comfortably in f32"
                )]
                let orig_ctx_f32 = *original_max_position_embeddings as f32;
                let low_freq_wavelen = orig_ctx_f32 / low_freq_factor;
                let high_freq_wavelen = orig_ctx_f32 / high_freq_factor;

                if wavelen < high_freq_wavelen {
                    base_theta_i
                } else if wavelen > low_freq_wavelen {
                    base_theta_i / factor
                } else {
                    let smooth = ((orig_ctx_f32 / wavelen) - low_freq_factor)
                        / (high_freq_factor - low_freq_factor);
                    let smooth = smooth.clamp(0.0, 1.0);
                    (1.0_f32 - smooth) * (base_theta_i / factor) + smooth * base_theta_i
                }
            }
            Some(RopeScalingConfig::LongRope {
                short_factor,
                long_factor,
                original_max_position_embeddings,
            }) => {
                if short_factor.len() != dim_pairs || long_factor.len() != dim_pairs {
                    return Err(K3Error::CapabilityMismatch {
                        architecture: "rope".to_owned(),
                        reason: format!(
                            "LongRope factor lengths (short: {}, long: {}) must equal head_dim / 2 ({dim_pairs})",
                            short_factor.len(),
                            long_factor.len()
                        ),
                    });
                }

                let s_fac = short_factor[i];
                let l_fac = long_factor[i];
                if !s_fac.is_finite() || s_fac <= 0.0 || !l_fac.is_finite() || l_fac <= 0.0 {
                    return Err(K3Error::CapabilityMismatch {
                        architecture: "rope".to_owned(),
                        reason: format!(
                            "LongRope factors must be positive and finite at index {i}: short={s_fac}, long={l_fac}"
                        ),
                    });
                }

                if seq_len <= *original_max_position_embeddings {
                    base_theta_i / s_fac
                } else {
                    base_theta_i / l_fac
                }
            }
        };

        inv_freq.push(scaled_theta_i);
    }

    Ok(inv_freq)
}

/// Applies rotary position embeddings in-place to query or key vectors.
///
/// Multi-head slices with `qk.len() % head_dim == 0` are transformed head-by-head using
/// the half-split rotation standard (`x1 = v[i]`, `x2 = v[i + d/2]`) shared by Llama, Mistral,
/// and Phi-3.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `head_dim` is odd, zero, or `qk.len() % head_dim != 0`.
/// Returns [`K3Error::CapabilityMismatch`] if frequency scaling parameters are invalid.
pub fn apply_rope_inplace(
    qk: &mut [f32],
    head_dim: usize,
    pos: usize,
    seq_len: usize,
    theta_base: f32,
    scaling: Option<&RopeScalingConfig>,
) -> Result<(), K3Error> {
    if head_dim == 0 || head_dim % 2 != 0 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "rope_qk".to_owned(),
            offset: 0,
            byte_len: size_of_val(qk),
            buffer_len: size_of_val(qk),
            reason: format!("head_dim must be positive and even, got {head_dim}"),
        });
    }

    if qk.len() % head_dim != 0 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "rope_qk".to_owned(),
            offset: 0,
            byte_len: size_of_val(qk),
            buffer_len: size_of_val(qk),
            reason: format!(
                "qk buffer length {} is not divisible by head_dim {head_dim}",
                qk.len()
            ),
        });
    }

    let inv_freq = precompute_rope_inv_freq(head_dim, theta_base, scaling, seq_len)?;
    apply_rope_with_inv_freq(qk, head_dim, pos, &inv_freq)
}

/// Applies rotary position embeddings in-place using a precomputed inverse frequency table.
///
/// Enables zero-allocation per-token rotation when `precompute_rope_inv_freq` is computed
/// once per sequence.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `head_dim` is odd, zero, or `qk.len() % head_dim != 0`.
/// Returns [`K3Error::CapabilityMismatch`] if `inv_freq.len() != head_dim / 2`.
pub fn apply_rope_with_inv_freq(
    qk: &mut [f32],
    head_dim: usize,
    pos: usize,
    inv_freq: &[f32],
) -> Result<(), K3Error> {
    if head_dim == 0 || head_dim % 2 != 0 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "rope_qk".to_owned(),
            offset: 0,
            byte_len: size_of_val(qk),
            buffer_len: size_of_val(qk),
            reason: format!("head_dim must be positive and even, got {head_dim}"),
        });
    }

    if qk.len() % head_dim != 0 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "rope_qk".to_owned(),
            offset: 0,
            byte_len: size_of_val(qk),
            buffer_len: size_of_val(qk),
            reason: format!(
                "qk buffer length {} is not divisible by head_dim {head_dim}",
                qk.len()
            ),
        });
    }

    let half = head_dim / 2;
    if inv_freq.len() != half {
        return Err(K3Error::CapabilityMismatch {
            architecture: "rope".to_owned(),
            reason: format!(
                "inv_freq table length {} must equal head_dim / 2 ({half})",
                inv_freq.len()
            ),
        });
    }

    // At position 0, cos(0) = 1.0 and sin(0) = 0.0, so rotation is exact identity
    if pos == 0 {
        return Ok(());
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "sequence position fits comfortably in f32"
    )]
    let pos_f32 = pos as f32;

    let mut cos_table = Vec::with_capacity(half);
    let mut sin_table = Vec::with_capacity(half);
    for &freq in inv_freq {
        let angle = pos_f32 * freq;
        cos_table.push(angle.cos());
        sin_table.push(angle.sin());
    }

    for head in qk.chunks_exact_mut(head_dim) {
        #[cfg(all(target_arch = "riscv64", not(miri)))]
        {
            // SAFETY: `head` has length `head_dim`, `cos_table` and `sin_table` have length `half`.
            // Pointers are non-overlapping and valid for `half` elements.
            unsafe {
                apply_rope_head_rvv(
                    head.as_mut_ptr(),
                    head[half..].as_mut_ptr(),
                    cos_table.as_ptr(),
                    sin_table.as_ptr(),
                    half,
                );
            }
        }
        #[cfg(any(not(target_arch = "riscv64"), miri))]
        {
            apply_rope_head_scalar(head, half, &cos_table, &sin_table);
        }
    }

    Ok(())
}

#[cfg(any(not(target_arch = "riscv64"), miri))]
fn apply_rope_head_scalar(head: &mut [f32], half: usize, cos_table: &[f32], sin_table: &[f32]) {
    for i in 0..half {
        let x1 = head[i];
        let x2 = head[i + half];
        let c = cos_table[i];
        let s = sin_table[i];
        head[i] = x1 * c - x2 * s;
        head[i + half] = x1 * s + x2 * c;
    }
}

/// RVV 1.0 vector length agnostic (`VLA`) strip-mining implementation of `RoPE` rotation.
#[cfg(all(target_arch = "riscv64", not(miri)))]
#[allow(unused_assignments)]
unsafe fn apply_rope_head_rvv(
    mut ptr1: *mut f32,
    mut ptr2: *mut f32,
    mut ptr_cos: *const f32,
    mut ptr_sin: *const f32,
    mut rem: usize,
) {
    // SAFETY: ptr1, ptr2, ptr_cos, and ptr_sin point to at least `rem` valid `f32` elements.
    // Strip-mined with `vsetvli` (`e32, m1, ta, ma`).
    unsafe {
        core::arch::asm!(
            "2:",
            "vsetvli t0, {rem}, e32, m1, ta, ma",
            "vle32.v v1, ({p1})",
            "vle32.v v2, ({p2})",
            "vle32.v v3, ({pcos})",
            "vle32.v v4, ({psin})",
            "vfmul.vv v5, v1, v3",
            "vfmul.vv v6, v2, v4",
            "vfsub.vv v5, v5, v6",
            "vfmul.vv v7, v1, v4",
            "vfmul.vv v8, v2, v3",
            "vfadd.vv v7, v7, v8",
            "vse32.v v5, ({p1})",
            "vse32.v v7, ({p2})",
            "slli t1, t0, 2",
            "add {p1}, {p1}, t1",
            "add {p2}, {p2}, t1",
            "add {pcos}, {pcos}, t1",
            "add {psin}, {psin}, t1",
            "sub {rem}, {rem}, t0",
            "bnez {rem}, 2b",
            p1 = inout(reg) ptr1,
            p2 = inout(reg) ptr2,
            pcos = inout(reg) ptr_cos,
            psin = inout(reg) ptr_sin,
            rem = inout(reg) rem,
            out("t0") _,
            out("t1") _,
            out("v1") _,
            out("v2") _,
            out("v3") _,
            out("v4") _,
            out("v5") _,
            out("v6") _,
            out("v7") _,
            out("v8") _,
            options(nostack),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precompute_rope_inv_freq_unscaled() {
        let head_dim = 64;
        let theta = 10000.0_f32;
        let inv_freq =
            precompute_rope_inv_freq(head_dim, theta, None, 0).expect("precompute unscaled rope");

        assert_eq!(inv_freq.len(), 32);
        assert!((inv_freq[0] - 1.0).abs() < 1e-6);
        let expected_last = 1.0 / theta.powf(62.0 / 64.0);
        assert!((inv_freq[31] - expected_last).abs() < 1e-6);
    }

    #[test]
    fn test_precompute_rope_inv_freq_linear() {
        let head_dim = 64;
        let theta = 10000.0_f32;
        let config = RopeScalingConfig::Linear { factor: 2.0 };
        let inv_freq = precompute_rope_inv_freq(head_dim, theta, Some(&config), 0)
            .expect("precompute linear rope");

        assert_eq!(inv_freq.len(), 32);
        assert!((inv_freq[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_precompute_rope_llama3_smooth() {
        let head_dim = 64;
        let theta = 500_000.0_f32;
        let config = RopeScalingConfig::Llama3Smooth {
            factor: 8.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position_embeddings: 8192,
        };
        let inv_freq = precompute_rope_inv_freq(head_dim, theta, Some(&config), 0)
            .expect("precompute llama3 smooth");

        assert_eq!(inv_freq.len(), 32);
        // Pair 0 has high frequency (short wavelength < high_freq_wavelen), so scale is 1.0
        assert!((inv_freq[0] - 1.0).abs() < 1e-6);
        // Later pairs have low frequency (long wavelength > low_freq_wavelen), so scale is factor
        let last_base = 1.0 / theta.powf(62.0 / 64.0);
        assert!((inv_freq[31] - last_base / 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_precompute_rope_longrope_context_switch() {
        let head_dim = 4;
        let short_factor = vec![1.0, 2.0];
        let long_factor = vec![3.0, 4.0];
        let config = RopeScalingConfig::LongRope {
            short_factor,
            long_factor,
            original_max_position_embeddings: 4096,
        };

        // Short context window
        let short_freq = precompute_rope_inv_freq(head_dim, 10000.0, Some(&config), 2048)
            .expect("precompute short longrope");
        assert!((short_freq[0] - 1.0).abs() < 1e-6);
        let base_1 = 1.0 / 10000.0_f32.powf(2.0 / 4.0);
        assert!((short_freq[1] - base_1 / 2.0).abs() < 1e-6);

        // Long context window (switches to long_factor)
        let long_freq = precompute_rope_inv_freq(head_dim, 10000.0, Some(&config), 8192)
            .expect("precompute long longrope");
        assert!((long_freq[0] - 1.0 / 3.0).abs() < 1e-6);
        assert!((long_freq[1] - base_1 / 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_apply_rope_inplace_identity_at_pos_zero() {
        let mut qk = [1.0_f32, 2.0, 3.0, 4.0];
        let copy = qk;
        apply_rope_inplace(&mut qk, 4, 0, 4, 10000.0, None).expect("pos 0 is exact identity");
        assert_eq!(qk, copy);
    }

    #[test]
    fn test_apply_rope_inplace_rotation() {
        let mut qk = [1.0_f32, 0.0, 0.0, 1.0];
        apply_rope_inplace(&mut qk, 4, 1, 4, 10000.0, None).expect("rotation at pos 1");

        // head_dim=4, half=2. Pair 0: (qk[0], qk[2]), Pair 1: (qk[1], qk[3])
        // inv_freq[0] = 1.0, angle0 = 1.0
        // qk'[0] = 1.0 * cos(1) - 0.0 * sin(1) = cos(1)
        // qk'[2] = 1.0 * sin(1) + 0.0 * cos(1) = sin(1)
        assert!((qk[0] - 1.0_f32.cos()).abs() < 1e-5);
        assert!((qk[2] - 1.0_f32.sin()).abs() < 1e-5);
    }

    #[test]
    fn test_rope_validation_errors() {
        let mut qk = [1.0_f32; 5];
        // Odd head_dim
        assert!(matches!(
            apply_rope_inplace(&mut qk, 5, 1, 5, 10000.0, None),
            Err(K3Error::TensorBoundsError { .. })
        ));

        let mut qk2 = [1.0_f32; 7];
        // qk.len() not divisible by head_dim
        assert!(matches!(
            apply_rope_inplace(&mut qk2, 4, 1, 4, 10000.0, None),
            Err(K3Error::TensorBoundsError { .. })
        ));

        // Invalid theta
        assert!(matches!(
            precompute_rope_inv_freq(4, -10.0, None, 0),
            Err(K3Error::CapabilityMismatch { .. })
        ));
    }
}
