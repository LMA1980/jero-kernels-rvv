//! Rotary Position Embeddings (`RoPE`) with standard linear, Llama-3 smooth, and `LongRoPE` scaling.
//!
//! Provides table precomputation and in-place vector rotation compatible with both
//! VLA RVV 1.0 (`zvl256b`/`zvl1024b`) strip-mining and portable scalar fallbacks.

use core::f32::consts::PI;
use jero_core::K3Error;

/// Maximum number of rotary dimension pairs (`rotary_dim / 2`) buffered on the stack
/// (`128` pairs = `rotary_dim <= 256`) so per-token `RoPE` avoids heap allocation on all
/// standard head dimensions (`64`, `96`, `128`, `256`).
const ROPE_STACK_PAIR_CAP: usize = 128;

/// Memory layout convention for paired Rotary Position Embedding (`RoPE`) dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RopeType {
    /// Interleaved adjacent pairs: `(v[2 * i], v[2 * i + 1])` (`LLAMA_ROPE_TYPE_NORM`).
    ///
    /// Used by standard Llama, Mistral, and GGUF `NORM` checkpoints.
    Norm,
    /// Split-half pairs: `(v[i], v[i + rotary_dim / 2])` (`LLAMA_ROPE_TYPE_NEOX`).
    ///
    /// Used by GPT-NeoX, Qwen2, Gemma, `DeepSeek`, and `HuggingFace` transformers checkpoints.
    #[default]
    NeoX,
}

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

    let dim_pairs = head_dim / 2;
    let mut inv_freq = vec![0.0_f32; dim_pairs];
    fill_rope_inv_freq(head_dim, theta, scaling, seq_len, &mut inv_freq)?;
    Ok(inv_freq)
}

#[expect(
    clippy::too_many_lines,
    reason = "comprehensive RoPE scaling dispatch across linear, smooth, and LongRoPE configs"
)]
fn fill_rope_inv_freq(
    rotary_dim: usize,
    theta: f32,
    scaling: Option<&RopeScalingConfig>,
    seq_len: usize,
    inv_freq_out: &mut [f32],
) -> Result<(), K3Error> {
    if rotary_dim == 0 || rotary_dim % 2 != 0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: "rope".to_owned(),
            reason: format!("rotary_dim must be positive and even, got {rotary_dim}"),
        });
    }

    let dim_pairs = rotary_dim / 2;
    if inv_freq_out.len() != dim_pairs {
        return Err(K3Error::CapabilityMismatch {
            architecture: "rope".to_owned(),
            reason: format!(
                "inv_freq_out length {} must equal rotary_dim / 2 ({dim_pairs})",
                inv_freq_out.len()
            ),
        });
    }

    if !theta.is_finite() || theta <= 0.0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: "rope".to_owned(),
            reason: format!("theta must be positive and finite, got {theta}"),
        });
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "rotary_dim fits comfortably in f32"
    )]
    let rotary_dim_f32 = rotary_dim as f32;

    for (i, slot) in inv_freq_out.iter_mut().enumerate() {
        #[expect(
            clippy::cast_precision_loss,
            reason = "pair index fits comfortably in f32"
        )]
        let exponent = (2 * i) as f32 / rotary_dim_f32;
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
                            "LongRope factor lengths (short: {}, long: {}) must equal rotary_dim / 2 ({dim_pairs})",
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

        *slot = scaled_theta_i;
    }

    Ok(())
}

fn validate_rope_geometry(
    buffer_byte_len: usize,
    elem_len: usize,
    head_dim: usize,
    rotary_dim: usize,
) -> Result<usize, K3Error> {
    if head_dim == 0 || head_dim % 2 != 0 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "rope_qk".to_owned(),
            offset: 0,
            byte_len: buffer_byte_len,
            buffer_len: buffer_byte_len,
            reason: format!("head_dim must be positive and even, got {head_dim}"),
        });
    }

    if rotary_dim == 0 || rotary_dim > head_dim || rotary_dim % 2 != 0 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "rope_qk".to_owned(),
            offset: 0,
            byte_len: buffer_byte_len,
            buffer_len: buffer_byte_len,
            reason: format!(
                "rotary_dim ({rotary_dim}) must be positive, even, and <= head_dim ({head_dim})"
            ),
        });
    }

    if elem_len % head_dim != 0 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "rope_qk".to_owned(),
            offset: 0,
            byte_len: buffer_byte_len,
            buffer_len: buffer_byte_len,
            reason: format!("qk buffer length {elem_len} is not divisible by head_dim {head_dim}"),
        });
    }

    Ok(elem_len / head_dim)
}

/// Computes `RoPE` cosine/sine lookup tables into caller-supplied scratch slices with zero heap allocation.
///
/// When `inv_freq` is `Some`, uses those precomputed inverse frequencies (requiring
/// `inv_freq.len() == cos_out.len()`). When `inv_freq` is `None`, computes frequencies from
/// `freq_base` assuming `rotary_dim = 2 * cos_out.len()`.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `cos_out` / `sin_out` are empty or have mismatched lengths.
/// Returns [`K3Error::CapabilityMismatch`] if `freq_base`, `freq_scale`, or `inv_freq.len()` are invalid.
pub fn compute_rope_cos_sin(
    pos: usize,
    freq_base: f32,
    freq_scale: f32,
    inv_freq: Option<&[f32]>,
    cos_out: &mut [f32],
    sin_out: &mut [f32],
) -> Result<(), K3Error> {
    let half = cos_out.len();
    if half == 0 || sin_out.len() != half {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "rope_cos_sin".to_owned(),
            offset: 0,
            byte_len: size_of_val(cos_out),
            buffer_len: size_of_val(sin_out),
            reason: format!(
                "cos/sin scratch slices must be non-empty and equal length (cos={}, sin={})",
                cos_out.len(),
                sin_out.len()
            ),
        });
    }

    if !freq_scale.is_finite() || freq_scale <= 0.0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: "rope".to_owned(),
            reason: format!("freq_scale must be positive and finite, got {freq_scale}"),
        });
    }

    if let Some(freqs) = inv_freq {
        if freqs.len() != half {
            return Err(K3Error::CapabilityMismatch {
                architecture: "rope".to_owned(),
                reason: format!(
                    "inv_freq table length {} must equal half ({half})",
                    freqs.len()
                ),
            });
        }
    } else if !freq_base.is_finite() || freq_base <= 0.0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: "rope".to_owned(),
            reason: format!("freq_base must be positive and finite, got {freq_base}"),
        });
    }

    if pos == 0 {
        cos_out.fill(1.0_f32);
        sin_out.fill(0.0_f32);
        return Ok(());
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "sequence position fits comfortably in f32"
    )]
    let pos_scaled = (pos as f32) * freq_scale;

    if let Some(freqs) = inv_freq {
        for i in 0..half {
            let angle = pos_scaled * freqs[i];
            cos_out[i] = angle.cos();
            sin_out[i] = angle.sin();
        }
    } else {
        let rotary_dim = half * 2;
        #[expect(
            clippy::cast_precision_loss,
            reason = "rotary_dim fits comfortably in f32"
        )]
        let rotary_dim_f32 = rotary_dim as f32;
        for i in 0..half {
            #[expect(
                clippy::cast_precision_loss,
                reason = "pair index fits comfortably in f32"
            )]
            let exponent = (2 * i) as f32 / rotary_dim_f32;
            let angle = pos_scaled / freq_base.powf(exponent);
            cos_out[i] = angle.cos();
            sin_out[i] = angle.sin();
        }
    }

    Ok(())
}

/// Applies rotary position embeddings in-place using precomputed `cos`/`sin` tables and explicit layout.
///
/// Only the leading `rotary_dim` dimensions of each head are rotated; tail dimensions
/// `head[rotary_dim..head_dim]` are preserved verbatim for partial-RoPE architectures.
/// Performs zero heap allocations.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `num_heads == 0`, `head_dim`/`rotary_dim` are invalid
/// (`rotary_dim == 0`, odd, or `rotary_dim > head_dim`), `x.len() != num_heads * head_dim`, or
/// `cos.len()` / `sin.len()` do not equal `rotary_dim / 2`.
pub fn apply_rope_with_cos_sin(
    x: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    rope_type: RopeType,
    cos: &[f32],
    sin: &[f32],
) -> Result<(), K3Error> {
    let expected_len = num_heads.checked_mul(head_dim);
    if num_heads == 0 || expected_len != Some(x.len()) {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "rope_qk".to_owned(),
            offset: 0,
            byte_len: size_of_val(x),
            buffer_len: size_of_val(x),
            reason: format!(
                "buffer length {} does not match num_heads ({num_heads}) * head_dim ({head_dim})",
                x.len()
            ),
        });
    }

    let _ = validate_rope_geometry(size_of_val(x), x.len(), head_dim, rotary_dim)?;

    let half = rotary_dim / 2;
    if cos.len() != half || sin.len() != half {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "rope_cos_sin".to_owned(),
            offset: 0,
            byte_len: size_of_val(cos),
            buffer_len: size_of_val(sin),
            reason: format!(
                "cos/sin lengths ({}, {}) must equal rotary_dim / 2 ({half})",
                cos.len(),
                sin.len()
            ),
        });
    }

    for head in x.chunks_exact_mut(head_dim) {
        let rotary_slice = &mut head[..rotary_dim];
        match rope_type {
            RopeType::Norm => {
                #[cfg(all(target_arch = "riscv64", not(miri)))]
                {
                    let base = rotary_slice.as_mut_ptr();
                    // SAFETY: `rotary_slice` has `2 * half` valid elements; strided pointers at offsets 0 and 1
                    // address non-overlapping even/odd elements strictly within `rotary_slice`.
                    unsafe {
                        apply_rope_norm_head_rvv(
                            base,
                            base.add(1),
                            cos.as_ptr(),
                            sin.as_ptr(),
                            half,
                        );
                    }
                }
                #[cfg(any(not(target_arch = "riscv64"), miri))]
                {
                    apply_rope_head_norm_scalar(rotary_slice, half, cos, sin);
                }
            }
            RopeType::NeoX => {
                #[cfg(all(target_arch = "riscv64", not(miri)))]
                {
                    let base = rotary_slice.as_mut_ptr();
                    // SAFETY: `rotary_slice` has `2 * half` valid elements; half-split pointers at offsets 0
                    // and `half` address disjoint halves of `rotary_slice`.
                    unsafe {
                        apply_rope_head_rvv(base, base.add(half), cos.as_ptr(), sin.as_ptr(), half);
                    }
                }
                #[cfg(any(not(target_arch = "riscv64"), miri))]
                {
                    apply_rope_head_neox_scalar(rotary_slice, half, cos, sin);
                }
            }
        }
    }

    Ok(())
}

/// Applies rotary position embeddings in-place to query or key vectors.
///
/// Multi-head slices with `qk.len() % head_dim == 0` are transformed head-by-head using
/// the half-split rotation standard (`RopeType::NeoX`, `x1 = v[i]`, `x2 = v[i + d/2]`)
/// shared by Llama, Mistral, and Phi-3, with zero heap allocations for `head_dim <= 256`.
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
    apply_rope_inplace_with_layout(
        qk,
        head_dim,
        head_dim,
        RopeType::NeoX,
        pos,
        seq_len,
        theta_base,
        scaling,
    )
}

/// Applies rotary position embeddings in-place with explicit `rotary_dim` and [`RopeType`].
///
/// Uses stack scratch buffers for `rotary_dim <= 256`, achieving zero heap allocations on all
/// standard transformer head sizes (`64`, `96`, `128`, `256`).
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `head_dim`, `rotary_dim`, or `qk.len()` bounds are invalid.
/// Returns [`K3Error::CapabilityMismatch`] if frequency scaling parameters are invalid.
#[expect(
    clippy::too_many_arguments,
    reason = "layout-aware RoPE helper mirrors GGML rope_ext parameters without wrapping struct overhead"
)]
pub fn apply_rope_inplace_with_layout(
    qk: &mut [f32],
    head_dim: usize,
    rotary_dim: usize,
    rope_type: RopeType,
    pos: usize,
    seq_len: usize,
    theta_base: f32,
    scaling: Option<&RopeScalingConfig>,
) -> Result<(), K3Error> {
    let _ = validate_rope_geometry(size_of_val(qk), qk.len(), head_dim, rotary_dim)?;
    let half = rotary_dim / 2;

    if half <= ROPE_STACK_PAIR_CAP {
        let mut inv_freq_stack = [0.0_f32; ROPE_STACK_PAIR_CAP];
        fill_rope_inv_freq(
            rotary_dim,
            theta_base,
            scaling,
            seq_len,
            &mut inv_freq_stack[..half],
        )?;
        apply_rope_with_inv_freq_and_layout(
            qk,
            head_dim,
            rotary_dim,
            rope_type,
            pos,
            &inv_freq_stack[..half],
        )
    } else {
        let mut inv_freq = vec![0.0_f32; half];
        fill_rope_inv_freq(rotary_dim, theta_base, scaling, seq_len, &mut inv_freq)?;
        apply_rope_with_inv_freq_and_layout(qk, head_dim, rotary_dim, rope_type, pos, &inv_freq)
    }
}

/// Applies rotary position embeddings in-place using a precomputed inverse frequency table.
///
/// Uses [`RopeType::NeoX`] across the full `head_dim`, with zero heap allocations for
/// `head_dim <= 256`.
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
    apply_rope_with_inv_freq_and_layout(qk, head_dim, head_dim, RopeType::NeoX, pos, inv_freq)
}

/// Applies rotary position embeddings in-place using a precomputed inverse frequency table,
/// partial `rotary_dim`, and explicit [`RopeType`] layout.
///
/// Zero heap allocations when `rotary_dim <= 256`.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `head_dim`, `rotary_dim`, or `qk.len()` bounds are invalid.
/// Returns [`K3Error::CapabilityMismatch`] if `inv_freq.len() != rotary_dim / 2`.
pub fn apply_rope_with_inv_freq_and_layout(
    qk: &mut [f32],
    head_dim: usize,
    rotary_dim: usize,
    rope_type: RopeType,
    pos: usize,
    inv_freq: &[f32],
) -> Result<(), K3Error> {
    let num_heads = validate_rope_geometry(size_of_val(qk), qk.len(), head_dim, rotary_dim)?;
    let half = rotary_dim / 2;

    if inv_freq.len() != half {
        return Err(K3Error::CapabilityMismatch {
            architecture: "rope".to_owned(),
            reason: format!(
                "inv_freq table length {} must equal rotary_dim / 2 ({half})",
                inv_freq.len()
            ),
        });
    }

    if pos == 0 || num_heads == 0 {
        return Ok(());
    }

    if half <= ROPE_STACK_PAIR_CAP {
        let mut cos_stack = [0.0_f32; ROPE_STACK_PAIR_CAP];
        let mut sin_stack = [0.0_f32; ROPE_STACK_PAIR_CAP];
        compute_rope_cos_sin(
            pos,
            10_000.0_f32,
            1.0_f32,
            Some(inv_freq),
            &mut cos_stack[..half],
            &mut sin_stack[..half],
        )?;
        apply_rope_with_cos_sin(
            qk,
            num_heads,
            head_dim,
            rotary_dim,
            rope_type,
            &cos_stack[..half],
            &sin_stack[..half],
        )
    } else {
        let mut cos_table = vec![0.0_f32; half];
        let mut sin_table = vec![0.0_f32; half];
        compute_rope_cos_sin(
            pos,
            10_000.0_f32,
            1.0_f32,
            Some(inv_freq),
            &mut cos_table,
            &mut sin_table,
        )?;
        apply_rope_with_cos_sin(
            qk, num_heads, head_dim, rotary_dim, rope_type, &cos_table, &sin_table,
        )
    }
}

#[cfg(any(not(target_arch = "riscv64"), miri))]
fn apply_rope_head_norm_scalar(
    head: &mut [f32],
    half: usize,
    cos_table: &[f32],
    sin_table: &[f32],
) {
    for i in 0..half {
        let x1 = head[2 * i];
        let x2 = head[2 * i + 1];
        let c = cos_table[i];
        let s = sin_table[i];
        head[2 * i] = x1 * c - x2 * s;
        head[2 * i + 1] = x1 * s + x2 * c;
    }
}

#[cfg(any(not(target_arch = "riscv64"), miri))]
fn apply_rope_head_neox_scalar(
    head: &mut [f32],
    half: usize,
    cos_table: &[f32],
    sin_table: &[f32],
) {
    for i in 0..half {
        let x1 = head[i];
        let x2 = head[i + half];
        let c = cos_table[i];
        let s = sin_table[i];
        head[i] = x1 * c - x2 * s;
        head[i + half] = x1 * s + x2 * c;
    }
}

/// RVV 1.0 vector length agnostic (`VLA`) strip-mining implementation of `RoPE` (`NeoX` half-split layout).
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

/// RVV 1.0 vector length agnostic (`VLA`) strip-mining implementation of `RoPE` (`Norm` interleaved layout).
#[cfg(all(target_arch = "riscv64", not(miri)))]
#[allow(unused_assignments)]
unsafe fn apply_rope_norm_head_rvv(
    mut ptr1: *mut f32,
    mut ptr2: *mut f32,
    mut ptr_cos: *const f32,
    mut ptr_sin: *const f32,
    mut rem: usize,
) {
    let stride_bytes: isize = 8;
    // SAFETY: `ptr1` and `ptr2` point to interleaved even/odd pairs spaced 8 bytes apart across `rem` pairs;
    // `ptr_cos` and `ptr_sin` are unit-stride slices of `rem` elements.
    unsafe {
        core::arch::asm!(
            "2:",
            "vsetvli t0, {rem}, e32, m1, ta, ma",
            "vlse32.v v1, ({p1}), {stride}",
            "vlse32.v v2, ({p2}), {stride}",
            "vle32.v v3, ({pcos})",
            "vle32.v v4, ({psin})",
            "vfmul.vv v5, v1, v3",
            "vfmul.vv v6, v2, v4",
            "vfsub.vv v5, v5, v6",
            "vfmul.vv v7, v1, v4",
            "vfmul.vv v8, v2, v3",
            "vfadd.vv v7, v7, v8",
            "vsse32.v v5, ({p1}), {stride}",
            "vsse32.v v7, ({p2}), {stride}",
            "slli t1, t0, 2",
            "add {pcos}, {pcos}, t1",
            "add {psin}, {psin}, t1",
            "slli t1, t0, 3",
            "add {p1}, {p1}, t1",
            "add {p2}, {p2}, t1",
            "sub {rem}, {rem}, t0",
            "bnez {rem}, 2b",
            p1 = inout(reg) ptr1,
            p2 = inout(reg) ptr2,
            pcos = inout(reg) ptr_cos,
            psin = inout(reg) ptr_sin,
            stride = in(reg) stride_bytes,
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

    #[test]
    fn test_rope_norm_and_neox_partial_rotary_golden_vectors() {
        let cases: &[(usize, usize)] = &[(64, 32), (96, 48), (128, 64), (256, 96), (128, 128)];
        let num_heads = 3usize;
        let pos = 17usize;
        let theta_base = 500_000.0_f32;
        let freq_scale = 0.75_f32;

        for &(head_dim, rotary_dim) in cases {
            let half = rotary_dim / 2;
            let mut cos = vec![0.0_f32; half];
            let mut sin = vec![0.0_f32; half];
            compute_rope_cos_sin(pos, theta_base, freq_scale, None, &mut cos, &mut sin)
                .expect("compute_rope_cos_sin succeeds");

            // Generate deterministic multi-head Q/K input tensor
            let mut orig = Vec::with_capacity(num_heads * head_dim);
            for h in 0..num_heads {
                for d in 0..head_dim {
                    #[expect(
                        clippy::cast_precision_loss,
                        reason = "small synthetic coordinates fit exact f32 mantissa"
                    )]
                    let val = ((h * 31 + d * 7) % 97) as f32 * 0.03125_f32 - 1.5_f32;
                    orig.push(val);
                }
            }

            // Double-precision llama.cpp reference calculation
            let pos_f64 =
                f64::from(u32::try_from(pos).expect("pos fits u32")) * f64::from(freq_scale);
            #[expect(clippy::cast_precision_loss, reason = "small dims fit exact f64")]
            let rotary_f64 = rotary_dim as f64;

            for rope_type in [RopeType::Norm, RopeType::NeoX] {
                let mut actual = orig.clone();
                apply_rope_with_cos_sin(
                    &mut actual,
                    num_heads,
                    head_dim,
                    rotary_dim,
                    rope_type,
                    &cos,
                    &sin,
                )
                .expect("apply_rope_with_cos_sin succeeds");

                let mut ref_buf = orig.clone();
                for head_idx in 0..num_heads {
                    let h_off = head_idx * head_dim;
                    for i in 0..half {
                        #[expect(clippy::cast_precision_loss, reason = "small index fits f64")]
                        let exp = (2 * i) as f64 / rotary_f64;
                        let angle = pos_f64 / f64::from(theta_base).powf(exp);
                        let (s_ref, c_ref) = angle.sin_cos();
                        let (i0, i1) = match rope_type {
                            RopeType::Norm => (h_off + 2 * i, h_off + 2 * i + 1),
                            RopeType::NeoX => (h_off + i, h_off + i + half),
                        };
                        let x0 = f64::from(ref_buf[i0]);
                        let x1 = f64::from(ref_buf[i1]);
                        #[expect(
                            clippy::cast_possible_truncation,
                            reason = "reference comparison downcasts back to f32"
                        )]
                        {
                            ref_buf[i0] = (x0 * c_ref - x1 * s_ref) as f32;
                            ref_buf[i1] = (x0 * s_ref + x1 * c_ref) as f32;
                        }
                    }

                    // Tail [rotary_dim..head_dim] must be completely untouched.
                    assert_eq!(
                        &actual[h_off + rotary_dim..h_off + head_dim],
                        &orig[h_off + rotary_dim..h_off + head_dim],
                        "partial RoPE modified unrotated tail for head_dim={head_dim}, rotary_dim={rotary_dim}"
                    );
                }

                for idx in 0..actual.len() {
                    let diff = (actual[idx] - ref_buf[idx]).abs();
                    assert!(
                        diff < 2e-4,
                        "parity failure ({rope_type:?}, head_dim={head_dim}, rotary_dim={rotary_dim}) at idx={idx}: actual={}, ref={}, diff={diff}",
                        actual[idx],
                        ref_buf[idx]
                    );
                }
            }
        }
    }

    #[test]
    fn test_rope_rotary_dim_and_cos_sin_bounds_return_tensor_bounds_error() {
        let mut x = [1.0_f32; 16];
        let cos = [1.0_f32; 4];
        let sin = [0.0_f32; 4];

        // rotary_dim == 0
        assert!(matches!(
            apply_rope_with_cos_sin(&mut x, 2, 8, 0, RopeType::Norm, &[], &[]),
            Err(K3Error::TensorBoundsError { .. })
        ));
        // odd rotary_dim
        assert!(matches!(
            apply_rope_with_cos_sin(&mut x, 2, 8, 6 - 1, RopeType::NeoX, &cos, &sin),
            Err(K3Error::TensorBoundsError { .. })
        ));
        // rotary_dim > head_dim
        assert!(matches!(
            apply_rope_with_cos_sin(&mut x, 2, 8, 10, RopeType::Norm, &[1.0; 5], &[0.0; 5]),
            Err(K3Error::TensorBoundsError { .. })
        ));
        // num_heads == 0
        assert!(matches!(
            apply_rope_with_cos_sin(&mut x, 0, 8, 8, RopeType::NeoX, &cos, &sin),
            Err(K3Error::TensorBoundsError { .. })
        ));
        // cos/sin len mismatch
        assert!(matches!(
            apply_rope_with_cos_sin(&mut x, 2, 8, 8, RopeType::Norm, &[1.0; 3], &[0.0; 4]),
            Err(K3Error::TensorBoundsError { .. })
        ));
        // empty scratch slices in compute_rope_cos_sin
        assert!(matches!(
            compute_rope_cos_sin(1, 10000.0, 1.0, None, &mut [], &mut []),
            Err(K3Error::TensorBoundsError { .. })
        ));
        // invalid freq_scale
        let mut cos_ok = [0.0_f32; 4];
        let mut sin_ok = [0.0_f32; 4];
        assert!(matches!(
            compute_rope_cos_sin(1, 10000.0, 0.0, None, &mut cos_ok, &mut sin_ok),
            Err(K3Error::CapabilityMismatch { .. })
        ));
        // inv_freq length mismatch in compute_rope_cos_sin
        assert!(matches!(
            compute_rope_cos_sin(1, 10000.0, 1.0, Some(&[1.0; 2]), &mut cos_ok, &mut sin_ok),
            Err(K3Error::CapabilityMismatch { .. })
        ));
    }

    #[test]
    fn test_vector_kernel_and_rvv_alias_rope_swiglu() {
        use crate::{RvvKernel, VectorKernel};

        let kernel = RvvKernel::new();
        let mut cos = [0.0_f32; 4];
        let mut sin = [0.0_f32; 4];
        kernel
            .compute_rope_cos_sin(0, 10000.0, 1.0, None, &mut cos, &mut sin)
            .expect("kernel compute_rope_cos_sin pos 0");
        assert_eq!(cos, [1.0; 4]);
        assert_eq!(sin, [0.0; 4]);

        let mut x = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let orig = x;
        kernel
            .apply_rope_with_cos_sin(&mut x, 1, 8, 8, RopeType::Norm, &cos, &sin)
            .expect("kernel apply_rope_with_cos_sin identity");
        assert_eq!(x, orig);

        let gate = [0.0_f32, 2.0, -30.0, 30.0];
        let up = [1.0_f32, 2.0, 3.0, 0.5];
        let mut out = [0.0_f32; 4];
        kernel
            .swiglu_f32(&mut out, &gate, &up)
            .expect("kernel swiglu_f32");
        assert_eq!(out[0], 0.0);
        assert!((out[3] - 15.0).abs() < 1e-4);
    }
}
