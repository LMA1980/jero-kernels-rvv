//! Non-token machine-model compute kernels (`FT-Transformer`, `TinyTimeMixers`, `TabPFN-v2`).
//!
//! Provides VLA `RVV 1.0` kernels (with portable scalar and strip-mined fallbacks) for
//! non-tokenized continuous tabular and multivariate time-series models:
//! - [`tokenize_continuous_features_f32`]: feature-wise affine tokenizer `x_i * W_i + b_i` for
//!   `FT-Transformer`.
//! - [`gelu_f32`], [`gelu_inplace_f32`], [`layernorm_f32`], and [`ttm_time_channel_mix_f32`]:
//!   `LayerNorm`, `GELU_tanh`, and two-stage residual patch/time and channel MLP mixer kernels for
//!   `TinyTimeMixers` (`TTM`).
//! - [`tabpfn_dual_axis_attention_f32`]: intra-row feature-axis and cross-row context-to-query
//!   sample-axis scaled dot-product attention primitives for `TabPFN-v2`.
//!
//! All public entry points validate non-zero geometry, `usize` multiplication bounds, slice lengths,
//! strictly positive parameters, and upfront element finiteness (`NaN`/`±Inf` rejection) before
//! mutating any output slice, failing closed with [`K3Error::TensorBoundsError`].

use jero_core::K3Error;

use crate::attention::{axpy_f32, dot_f32, softmax_scores_inplace};
pub use crate::swiglu::gelu_scalar_f32;

/// Layer normalization and two-layer `GELU` MLP weights for a single `TTM` mixer axis
/// (`dim -> hidden_dim -> dim`).
#[derive(Debug, Clone, Copy)]
pub struct AxisMlpWeights<'a> {
    /// `LayerNorm` affine scale (`[dim]`).
    pub norm_weight: &'a [f32],
    /// `LayerNorm` affine bias (`[dim]`).
    pub norm_bias: &'a [f32],
    /// First linear projection weight matrix in row-major `[hidden_dim, dim]` order.
    pub fc1_weight: &'a [f32],
    /// First linear projection bias (`[hidden_dim]`).
    pub fc1_bias: &'a [f32],
    /// Second linear projection weight matrix in row-major `[dim, hidden_dim]` order.
    pub fc2_weight: &'a [f32],
    /// Second linear projection bias (`[dim]`).
    pub fc2_bias: &'a [f32],
    /// Input and output vector length for this axis.
    pub dim: usize,
    /// Intermediate MLP hidden projection width.
    pub hidden_dim: usize,
}

impl AxisMlpWeights<'_> {
    fn validate(&self, axis_name: &str, expected_dim: usize) -> Result<(), K3Error> {
        let dim = check_positive_dim(&format!("{axis_name}_dim"), self.dim)?;
        if dim != expected_dim {
            return Err(bounds_error(
                &format!("{axis_name}_dim"),
                0,
                dim,
                expected_dim,
                format!("{axis_name} axis dim {dim} does not match expected {expected_dim}"),
            ));
        }
        let hidden_dim = check_positive_dim(&format!("{axis_name}_hidden_dim"), self.hidden_dim)?;
        let mat_len = checked_product(
            &format!("{axis_name}_mlp_weights"),
            hidden_dim,
            dim,
            "hidden_dim * dim",
        )?;

        check_exact_len(
            &format!("{axis_name}_norm_weight"),
            self.norm_weight.len(),
            dim,
        )?;
        check_exact_len(&format!("{axis_name}_norm_bias"), self.norm_bias.len(), dim)?;
        check_exact_len(
            &format!("{axis_name}_fc1_weight"),
            self.fc1_weight.len(),
            mat_len,
        )?;
        check_exact_len(
            &format!("{axis_name}_fc1_bias"),
            self.fc1_bias.len(),
            hidden_dim,
        )?;
        check_exact_len(
            &format!("{axis_name}_fc2_weight"),
            self.fc2_weight.len(),
            mat_len,
        )?;
        check_exact_len(&format!("{axis_name}_fc2_bias"), self.fc2_bias.len(), dim)?;

        check_all_finite(&format!("{axis_name}_norm_weight"), self.norm_weight)?;
        check_all_finite(&format!("{axis_name}_norm_bias"), self.norm_bias)?;
        check_all_finite(&format!("{axis_name}_fc1_weight"), self.fc1_weight)?;
        check_all_finite(&format!("{axis_name}_fc1_bias"), self.fc1_bias)?;
        check_all_finite(&format!("{axis_name}_fc2_weight"), self.fc2_weight)?;
        check_all_finite(&format!("{axis_name}_fc2_bias"), self.fc2_bias)?;

        Ok(())
    }
}

/// Attention axis selector for [`tabpfn_dual_axis_attention_f32`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TabPfnAttentionAxis {
    /// Intra-row attention across all `n_features` feature columns per sample.
    Feature,
    /// Cross-row attention across sample rows per feature column (`0..context_samples`).
    Sample,
}

/// Tensor geometry for `TabPFN-v2` dual-axis multi-head attention over row-major
/// `[n_samples, n_features, n_heads * head_dim]` tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TabPfnAttentionShape {
    /// Total sample rows (`context` rows followed by optional `query` rows).
    pub n_samples: usize,
    /// Number of feature columns per sample row.
    pub n_features: usize,
    /// Number of attention heads per position.
    pub n_heads: usize,
    /// Feature dimensionality per head (`d_model = n_heads * head_dim`).
    pub head_dim: usize,
    /// Optional prefix context row count visible as keys/values along [`TabPfnAttentionAxis::Sample`].
    pub context_samples: Option<usize>,
}

impl TabPfnAttentionShape {
    /// Constructs a new `TabPFN-v2` dual-axis attention geometry with all rows attending symmetrically.
    #[must_use]
    pub const fn new(n_samples: usize, n_features: usize, n_heads: usize, head_dim: usize) -> Self {
        Self {
            n_samples,
            n_features,
            n_heads,
            head_dim,
            context_samples: None,
        }
    }

    /// Sets the number of leading context sample rows visible as keys/values on the sample axis.
    #[must_use]
    pub const fn with_context_samples(mut self, context_samples: usize) -> Self {
        self.context_samples = Some(context_samples);
        self
    }

    /// Returns `n_heads * head_dim` when multiplication does not overflow `usize`.
    #[must_use]
    pub const fn d_model(self) -> Option<usize> {
        self.n_heads.checked_mul(self.head_dim)
    }

    /// Returns total element count `n_samples * n_features * n_heads * head_dim` when valid.
    #[must_use]
    pub const fn total_elements(self) -> Option<usize> {
        let Some(d_model) = self.d_model() else {
            return None;
        };
        let Some(row) = self.n_features.checked_mul(d_model) else {
            return None;
        };
        self.n_samples.checked_mul(row)
    }

    fn validated(self) -> Result<(usize, usize, usize), K3Error> {
        let n_samples = check_positive_dim("tabpfn_n_samples", self.n_samples)?;
        let n_features = check_positive_dim("tabpfn_n_features", self.n_features)?;
        let n_heads = check_positive_dim("tabpfn_n_heads", self.n_heads)?;
        let head_dim = check_positive_dim("tabpfn_head_dim", self.head_dim)?;

        let d_model = checked_product("tabpfn_shape", n_heads, head_dim, "n_heads * head_dim")?;
        let row_elems =
            checked_product("tabpfn_shape", n_features, d_model, "n_features * d_model")?;
        let total = checked_product(
            "tabpfn_shape",
            n_samples,
            row_elems,
            "n_samples * row_elems",
        )?;

        let n_ctx = match self.context_samples {
            None => n_samples,
            Some(ctx) if ctx == 0 || ctx > n_samples => {
                return Err(bounds_error(
                    "tabpfn_context_samples",
                    0,
                    ctx,
                    n_samples,
                    format!("context_samples ({ctx}) must lie within 1..={n_samples}"),
                ));
            }
            Some(ctx) => ctx,
        };

        Ok((d_model, total, n_ctx))
    }
}

pub(crate) fn bounds_error(
    tensor_name: &str,
    offset: usize,
    byte_len: usize,
    buffer_len: usize,
    reason: String,
) -> K3Error {
    K3Error::TensorBoundsError {
        tensor_name: tensor_name.to_owned(),
        offset,
        byte_len,
        buffer_len,
        reason,
    }
}

pub(crate) fn check_positive_dim(name: &str, value: usize) -> Result<usize, K3Error> {
    if value == 0 {
        return Err(bounds_error(
            name,
            0,
            0,
            0,
            format!("{name} must be non-zero"),
        ));
    }
    Ok(value)
}

pub(crate) fn checked_product(
    name: &str,
    a: usize,
    b: usize,
    desc: &str,
) -> Result<usize, K3Error> {
    a.checked_mul(b).ok_or_else(|| {
        bounds_error(
            name,
            0,
            usize::MAX,
            0,
            format!("{desc} overflowed usize ({a} * {b})"),
        )
    })
}

pub(crate) fn check_exact_len(name: &str, actual: usize, expected: usize) -> Result<(), K3Error> {
    if actual != expected {
        return Err(bounds_error(
            name,
            0,
            expected.saturating_mul(size_of::<f32>()),
            actual.saturating_mul(size_of::<f32>()),
            format!("expected {expected} elements for {name}, got {actual}"),
        ));
    }
    Ok(())
}

pub(crate) fn check_all_finite(name: &str, data: &[f32]) -> Result<(), K3Error> {
    for (idx, &val) in data.iter().enumerate() {
        if !val.is_finite() {
            return Err(bounds_error(
                name,
                idx.saturating_mul(size_of::<f32>()),
                size_of::<f32>(),
                size_of_val(data),
                format!("non-finite f32 value ({val}) in {name} at index {idx}"),
            ));
        }
    }
    Ok(())
}

pub(crate) fn check_positive_finite(name: &str, value: f32) -> Result<f32, K3Error> {
    if !value.is_finite() || value <= 0.0_f32 {
        return Err(bounds_error(
            name,
            0,
            size_of::<f32>(),
            size_of::<f32>(),
            format!("{name} ({value}) must be finite and strictly positive"),
        ));
    }
    Ok(value)
}

fn validate_feature_tokenizer(
    out: &[f32],
    raw: &[f32],
    weight: &[f32],
    bias: &[f32],
    n_features: usize,
    d_token: usize,
) -> Result<usize, K3Error> {
    let n_features = check_positive_dim("tokenize_n_features", n_features)?;
    let d_token = check_positive_dim("tokenize_d_token", d_token)?;
    let expected = checked_product(
        "tokenize_continuous",
        n_features,
        d_token,
        "n_features * d_token",
    )?;

    check_exact_len("tokenize_raw", raw.len(), n_features)?;
    check_exact_len("tokenize_weight", weight.len(), expected)?;
    check_exact_len("tokenize_bias", bias.len(), expected)?;
    check_exact_len("tokenize_out", out.len(), expected)?;

    check_all_finite("tokenize_raw", raw)?;
    check_all_finite("tokenize_weight", weight)?;
    check_all_finite("tokenize_bias", bias)?;

    Ok(expected)
}

/// Tokenizes scalar continuous features into dense `[n_features, d_token]` embeddings for
/// `FT-Transformer` using `out[i * d_token + d] = raw[i] * weight[i * d_token + d] + bias[i * d_token + d]`.
///
/// Dispatches to `VLA` `RVV 1.0` (`vfmacc.vf`) on `riscv64` targets and portable scalar arithmetic
/// elsewhere or under Miri.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `n_features` or `d_token` is zero, dimensions overflow
/// `usize`, slice lengths mismatch `[n_features]` / `[n_features * d_token]`, or any element in
/// `raw`, `weight`, or `bias` is non-finite (`NaN` or `±Inf`).
pub fn tokenize_continuous_features_f32(
    out: &mut [f32],
    raw: &[f32],
    weight: &[f32],
    bias: &[f32],
    n_features: usize,
    d_token: usize,
) -> Result<(), K3Error> {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        tokenize_continuous_features_f32_rvv(out, raw, weight, bias, n_features, d_token)
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        tokenize_continuous_features_f32_scalar(out, raw, weight, bias, n_features, d_token)
    }
}

/// Portable scalar reference implementation of [`tokenize_continuous_features_f32`].
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] on invalid dimensions, mismatched lengths, or non-finite inputs.
pub fn tokenize_continuous_features_f32_scalar(
    out: &mut [f32],
    raw: &[f32],
    weight: &[f32],
    bias: &[f32],
    n_features: usize,
    d_token: usize,
) -> Result<(), K3Error> {
    validate_feature_tokenizer(out, raw, weight, bias, n_features, d_token)?;
    for (i, &x) in raw.iter().enumerate() {
        let base = i * d_token;
        for d in 0..d_token {
            out[base + d] = x.mul_add(weight[base + d], bias[base + d]);
        }
    }
    Ok(())
}

/// Simulates `VLA` strip-mining for [`tokenize_continuous_features_f32`] at explicit `vlen_bits`
/// (`256` for `zvl256b` or `1024` for `zvl1024b`).
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] on invalid dimensions, mismatched lengths, or non-finite inputs.
pub fn tokenize_continuous_features_f32_vla_stripmined(
    out: &mut [f32],
    raw: &[f32],
    weight: &[f32],
    bias: &[f32],
    n_features: usize,
    d_token: usize,
    vlen_bits: usize,
) -> Result<(), K3Error> {
    validate_feature_tokenizer(out, raw, weight, bias, n_features, d_token)?;
    let vlmax = (vlen_bits / 32).max(1).saturating_mul(4);
    for (i, &x) in raw.iter().enumerate() {
        let base = i * d_token;
        let mut idx = 0usize;
        while idx < d_token {
            let vl = (d_token - idx).min(vlmax);
            for lane in 0..vl {
                let off = base + idx + lane;
                out[off] = x.mul_add(weight[off], bias[off]);
            }
            idx += vl;
        }
    }
    Ok(())
}

/// `RVV 1.0` (`vfmacc.vf` `e32, m4`) implementation of [`tokenize_continuous_features_f32`].
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] on invalid dimensions, mismatched lengths, or non-finite inputs.
#[cfg(all(target_arch = "riscv64", not(miri)))]
#[allow(unused_assignments)]
pub fn tokenize_continuous_features_f32_rvv(
    out: &mut [f32],
    raw: &[f32],
    weight: &[f32],
    bias: &[f32],
    n_features: usize,
    d_token: usize,
) -> Result<(), K3Error> {
    validate_feature_tokenizer(out, raw, weight, bias, n_features, d_token)?;
    for (i, &x) in raw.iter().enumerate() {
        let base = i * d_token;
        let mut pw = weight[base..base + d_token].as_ptr();
        let mut pb = bias[base..base + d_token].as_ptr();
        let mut pout = out[base..base + d_token].as_mut_ptr();
        let mut rem = d_token;

        // SAFETY: Validated above that `weight`, `bias`, and `out` all contain `base + d_token`
        // elements and do not alias (`out` is exclusively borrowed); `vfmacc.vf` computes
        // `x * weight + bias` cleanly across `e32, m4` strips.
        unsafe {
            core::arch::asm!(
                "2:",
                "vsetvli t0, {rem}, e32, m4, ta, ma",
                "vle32.v v8, ({pw})",
                "vle32.v v16, ({pb})",
                "vfmacc.vf v16, {x}, v8",
                "vse32.v v16, ({pout})",
                "slli t1, t0, 2",
                "add {pw}, {pw}, t1",
                "add {pb}, {pb}, t1",
                "add {pout}, {pout}, t1",
                "sub {rem}, {rem}, t0",
                "bnez {rem}, 2b",
                pw = inout(reg) pw,
                pb = inout(reg) pb,
                pout = inout(reg) pout,
                rem = inout(reg) rem,
                x = in(freg) x,
                out("t0") _,
                out("t1") _,
                out("v8") _,
                out("v9") _,
                out("v10") _,
                out("v11") _,
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

/// Evaluates element-wise `GELU_tanh` (`0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`)
/// from `input` into `out`.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `input` is empty, `out.len() != input.len()`, or any
/// element of `input` is non-finite (`NaN` or `±Inf`).
pub fn gelu_f32(out: &mut [f32], input: &[f32]) -> Result<(), K3Error> {
    let len = check_positive_dim("gelu_input", input.len())?;
    check_exact_len("gelu_out", out.len(), len)?;
    check_all_finite("gelu_input", input)?;

    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        // SAFETY: Both slices were validated non-empty and equal length (`len`).
        unsafe {
            gelu_rvv_raw(out.as_mut_ptr(), input.as_ptr(), len);
        }
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        for (dst, &x) in out.iter_mut().zip(input.iter()) {
            *dst = gelu_scalar_f32(x);
        }
    }
    Ok(())
}

/// Evaluates element-wise `GELU_tanh` (`out[i] = gelu(input[i])`) strictly via portable scalar math.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `input` is empty, `out.len() != input.len()`, or any
/// input element is non-finite.
pub fn gelu_f32_scalar(out: &mut [f32], input: &[f32]) -> Result<(), K3Error> {
    let len = check_positive_dim("gelu_input", input.len())?;
    check_exact_len("gelu_out", out.len(), len)?;
    check_all_finite("gelu_input", input)?;
    for (dst, &x) in out.iter_mut().zip(input.iter()) {
        *dst = gelu_scalar_f32(x);
    }
    Ok(())
}

/// Evaluates `GELU_tanh` in-place across `values`.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `values` is empty or contains non-finite entries.
pub fn gelu_inplace_f32(values: &mut [f32]) -> Result<(), K3Error> {
    let len = check_positive_dim("gelu_values", values.len())?;
    check_all_finite("gelu_values", values)?;

    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        let ptr = values.as_mut_ptr();
        // SAFETY: `ptr` points to a valid non-empty slice of `len` elements; `gelu_rvv_raw`
        // finishes reading all `vl` lanes into `v0` before storing `v24` back to `ptr`.
        unsafe {
            gelu_rvv_raw(ptr, ptr, len);
        }
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        let _ = len;
        for val in values.iter_mut() {
            *val = gelu_scalar_f32(*val);
        }
    }
    Ok(())
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[allow(unused_assignments)]
/// # Safety
///
/// `src` and `dst` must both be valid for `rem` `f32` elements (`src` readable, `dst` writable);
/// exact pointer aliasing (`src == dst`) is supported because loads complete before stores.
unsafe fn gelu_rvv_raw(mut pdst: *mut f32, mut psrc: *const f32, mut rem: usize) {
    // 0: cubic, 4: -2*sqrt(2/pi), 8: -88, 12: 88, 16: log2(e), 20: ln2_hi, 24: ln2_lo,
    // 28: 1/720, 32: 1/120, 36: 1/24, 40: 1/6, 44: 0.5, 48: 1.0
    const GELU_CONSTS: [f32; 13] = [
        0.044_715_f32,
        -1.595_769_2_f32,
        -88.0_f32,
        88.0_f32,
        core::f32::consts::LOG2_E,
        0.693_145_75_f32,
        1.428_606_8e-6_f32,
        1.0_f32 / 720.0_f32,
        1.0_f32 / 120.0_f32,
        1.0_f32 / 24.0_f32,
        1.0_f32 / 6.0_f32,
        0.5_f32,
        1.0_f32,
    ];
    let pc = GELU_CONSTS.as_ptr();
    let bias127: i32 = 127;
    let kmin: i32 = -126;
    let kmax: i32 = 127;

    // SAFETY: Caller upholds slice validity for `rem` elements; `e32, m4` registers `v0..v27`
    // do not overlap across arithmetic instructions and preserve `v0` (`x`) until `vfdiv.vv`.
    unsafe {
        core::arch::asm!(
            "flw ft0, 0({c})",
            "flw ft1, 4({c})",
            "flw ft2, 8({c})",
            "flw ft3, 12({c})",
            "flw ft4, 16({c})",
            "flw ft5, 20({c})",
            "flw ft6, 24({c})",
            "flw ft7, 48({c})",
            "2:",
            "vsetvli t0, {rem}, e32, m4, ta, ma",
            "vle32.v v0, ({psrc})",
            "vfmul.vv v4, v0, v0",
            "vfmul.vv v4, v4, v0",
            "vfmadd.vf v4, ft0, v0",
            "vfmul.vf v4, v4, ft1",
            "vfmax.vf v4, v4, ft2",
            "vfmin.vf v4, v4, ft3",
            "vfmul.vf v8, v4, ft4",
            "vfcvt.x.f.v v16, v8",
            "vfcvt.f.x.v v8, v16",
            "vfmul.vf v20, v8, ft5",
            "vfsub.vv v12, v4, v20",
            "vfmul.vf v20, v8, ft6",
            "vfsub.vv v12, v12, v20",
            "vmax.vx v16, v16, {kmin}",
            "vmin.vx v16, v16, {kmax}",
            "vadd.vx v16, v16, {bias127}",
            "vsll.vi v16, v16, 23",
            "flw ft8, 28({c})",
            "vfmv.v.f v20, ft8",
            "flw ft8, 32({c})",
            "vfmul.vv v20, v20, v12",
            "vfadd.vf v20, v20, ft8",
            "flw ft8, 36({c})",
            "vfmul.vv v20, v20, v12",
            "vfadd.vf v20, v20, ft8",
            "flw ft8, 40({c})",
            "vfmul.vv v20, v20, v12",
            "vfadd.vf v20, v20, ft8",
            "flw ft8, 44({c})",
            "vfmul.vv v20, v20, v12",
            "vfadd.vf v20, v20, ft8",
            "vfmul.vv v20, v20, v12",
            "vfadd.vf v20, v20, ft7",
            "vfmul.vv v20, v20, v12",
            "vfadd.vf v20, v20, ft7",
            "vfmul.vv v20, v20, v16",
            "vfadd.vf v20, v20, ft7",
            "vfdiv.vv v24, v0, v20",
            "vse32.v v24, ({pdst})",
            "slli t1, t0, 2",
            "add {psrc}, {psrc}, t1",
            "add {pdst}, {pdst}, t1",
            "sub {rem}, {rem}, t0",
            "bnez {rem}, 2b",
            psrc = inout(reg) psrc,
            pdst = inout(reg) pdst,
            rem = inout(reg) rem,
            c = in(reg) pc,
            bias127 = in(reg) bias127,
            kmin = in(reg) kmin,
            kmax = in(reg) kmax,
            out("t0") _,
            out("t1") _,
            out("ft0") _, out("ft1") _, out("ft2") _, out("ft3") _,
            out("ft4") _, out("ft5") _, out("ft6") _, out("ft7") _, out("ft8") _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            out("v24") _, out("v25") _, out("v26") _, out("v27") _,
            options(nostack),
        );
    }
}

fn validate_layernorm_args(
    out: &[f32],
    input: &[f32],
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    eps: f32,
) -> Result<usize, K3Error> {
    let len = check_positive_dim("layernorm_input", input.len())?;
    check_exact_len("layernorm_out", out.len(), len)?;
    check_positive_finite("layernorm_eps", eps)?;
    check_all_finite("layernorm_input", input)?;
    if let Some(w) = weight {
        check_exact_len("layernorm_weight", w.len(), len)?;
        check_all_finite("layernorm_weight", w)?;
    }
    if let Some(b) = bias {
        check_exact_len("layernorm_bias", b.len(), len)?;
        check_all_finite("layernorm_bias", b)?;
    }
    Ok(len)
}

/// Applies Layer Normalization (`(x - mean) / sqrt(var + eps) * weight + bias`) over a vector.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `input` is empty, lengths mismatch, `eps <= 0` or is
/// non-finite, or any input/parameter entry is `NaN` or `±Inf`.
pub fn layernorm_f32(
    out: &mut [f32],
    input: &[f32],
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    eps: f32,
) -> Result<(), K3Error> {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        layernorm_f32_rvv(out, input, weight, bias, eps)
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        layernorm_f32_scalar(out, input, weight, bias, eps)
    }
}

/// Portable scalar reference implementation of [`layernorm_f32`].
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] on invalid dimensions, lengths, `eps`, or non-finite values.
pub fn layernorm_f32_scalar(
    out: &mut [f32],
    input: &[f32],
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    eps: f32,
) -> Result<(), K3Error> {
    let len = validate_layernorm_args(out, input, weight, bias, eps)?;
    #[expect(
        clippy::cast_precision_loss,
        reason = "vector length fits comfortably in f32"
    )]
    let inv_n = 1.0_f32 / (len as f32);

    let sum: f32 = input.iter().copied().sum();
    let mean = sum * inv_n;

    let mut var_sum = 0.0_f32;
    for &x in input {
        let diff = x - mean;
        var_sum += diff * diff;
    }
    let inv_std = 1.0_f32 / (var_sum * inv_n + eps).sqrt();

    apply_layernorm_affine_scalar(out, input, mean, inv_std, weight, bias);
    Ok(())
}

fn apply_layernorm_affine_scalar(
    out: &mut [f32],
    input: &[f32],
    mean: f32,
    inv_std: f32,
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
) {
    for (i, &x) in input.iter().enumerate() {
        let mut norm = (x - mean) * inv_std;
        if let Some(w) = weight {
            norm *= w[i];
        }
        if let Some(b) = bias {
            norm += b[i];
        }
        out[i] = norm;
    }
}

/// Simulates `VLA` strip-mining for [`layernorm_f32`] at explicit `vlen_bits` (`256` or `1024`).
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] on invalid dimensions, lengths, `eps`, or non-finite values.
pub fn layernorm_f32_vla_stripmined(
    out: &mut [f32],
    input: &[f32],
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    eps: f32,
    vlen_bits: usize,
) -> Result<(), K3Error> {
    let len = validate_layernorm_args(out, input, weight, bias, eps)?;
    let vlmax = (vlen_bits / 32).max(1);
    #[expect(
        clippy::cast_precision_loss,
        reason = "vector length fits comfortably in f32"
    )]
    let inv_n = 1.0_f32 / (len as f32);

    let mut sum = 0.0_f32;
    for chunk in input.chunks(vlmax) {
        for &x in chunk {
            sum += x;
        }
    }
    let mean = sum * inv_n;

    let mut var_sum = 0.0_f32;
    for chunk in input.chunks(vlmax) {
        for &x in chunk {
            let d = x - mean;
            var_sum += d * d;
        }
    }
    let inv_std = 1.0_f32 / (var_sum * inv_n + eps).sqrt();

    apply_layernorm_affine_scalar(out, input, mean, inv_std, weight, bias);
    Ok(())
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[allow(unused_assignments)]
fn rvv_mean_and_inv_std(input: &[f32], inv_n: f32, eps: f32) -> (f32, f32) {
    let len = input.len();
    let zero = 0.0_f32;
    let mut ptr_sum = input.as_ptr();
    let mut rem_sum = len;
    let mut sum = 0.0_f32;
    // SAFETY: `input` is non-empty and valid for `len` elements; `e32, m1` ordered reduction.
    unsafe {
        core::arch::asm!(
            "vsetivli zero, 1, e32, m1, ta, ma",
            "vfmv.s.f v0, {zero}",
            "2:",
            "vsetvli t0, {rem}, e32, m1, ta, ma",
            "vle32.v v8, ({p})",
            "vfredusum.vs v0, v8, v0",
            "slli t1, t0, 2",
            "add {p}, {p}, t1",
            "sub {rem}, {rem}, t0",
            "bnez {rem}, 2b",
            "vsetivli zero, 1, e32, m1, ta, ma",
            "vfmv.f.s {out_sum}, v0",
            p = inout(reg) ptr_sum,
            rem = inout(reg) rem_sum,
            zero = in(freg) zero,
            out_sum = out(freg) sum,
            out("t0") _, out("t1") _,
            out("v0") _, out("v8") _,
            options(nostack),
        );
    }
    let mean = sum * inv_n;

    let mut ptr_var = input.as_ptr();
    let mut rem_var = len;
    let mut var_sum = 0.0_f32;
    // SAFETY: `input` is valid for `len` elements; computes `sum((x - mean)^2)` via `vfredusum.vs`.
    unsafe {
        core::arch::asm!(
            "vsetivli zero, 1, e32, m1, ta, ma",
            "vfmv.s.f v0, {zero}",
            "3:",
            "vsetvli t0, {rem}, e32, m1, ta, ma",
            "vle32.v v8, ({p})",
            "vfsub.vf v8, v8, {mean}",
            "vfmul.vv v8, v8, v8",
            "vfredusum.vs v0, v8, v0",
            "slli t1, t0, 2",
            "add {p}, {p}, t1",
            "sub {rem}, {rem}, t0",
            "bnez {rem}, 3b",
            "vsetivli zero, 1, e32, m1, ta, ma",
            "vfmv.f.s {out_var}, v0",
            p = inout(reg) ptr_var,
            rem = inout(reg) rem_var,
            zero = in(freg) zero,
            mean = in(freg) mean,
            out_var = out(freg) var_sum,
            out("t0") _, out("t1") _,
            out("v0") _, out("v8") _,
            options(nostack),
        );
    }
    let inv_std = 1.0_f32 / (var_sum * inv_n + eps).sqrt();
    (mean, inv_std)
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[allow(unused_assignments)]
fn rvv_mul_assign_f32(out: &mut [f32], weight: &[f32]) {
    let mut pw = weight.as_ptr();
    let mut pout = out.as_mut_ptr();
    let mut rem = out.len().min(weight.len());
    if rem == 0 {
        return;
    }
    // SAFETY: `out` and `weight` have at least `rem` elements and do not alias.
    unsafe {
        core::arch::asm!(
            "5:",
            "vsetvli t0, {rem}, e32, m4, ta, ma",
            "vle32.v v8, ({pout})",
            "vle32.v v16, ({pw})",
            "vfmul.vv v8, v8, v16",
            "vse32.v v8, ({pout})",
            "slli t1, t0, 2",
            "add {pw}, {pw}, t1",
            "add {pout}, {pout}, t1",
            "sub {rem}, {rem}, t0",
            "bnez {rem}, 5b",
            pw = inout(reg) pw,
            pout = inout(reg) pout,
            rem = inout(reg) rem,
            out("t0") _, out("t1") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            options(nostack),
        );
    }
}

/// `RVV 1.0` (`VLA` strip-mined) implementation of [`layernorm_f32`].
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] on invalid dimensions, lengths, `eps`, or non-finite values.
#[cfg(all(target_arch = "riscv64", not(miri)))]
#[allow(unused_assignments)]
pub fn layernorm_f32_rvv(
    out: &mut [f32],
    input: &[f32],
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    eps: f32,
) -> Result<(), K3Error> {
    let len = validate_layernorm_args(out, input, weight, bias, eps)?;
    #[expect(
        clippy::cast_precision_loss,
        reason = "vector length fits comfortably in f32"
    )]
    let inv_n = 1.0_f32 / (len as f32);
    let (mean, inv_std) = rvv_mean_and_inv_std(input, inv_n, eps);

    let mut pin = input.as_ptr();
    let mut pout = out.as_mut_ptr();
    let mut rem_out = len;
    // SAFETY: `input` and `out` are non-overlapping slices of length `len`.
    unsafe {
        core::arch::asm!(
            "4:",
            "vsetvli t0, {rem}, e32, m4, ta, ma",
            "vle32.v v8, ({pin})",
            "vfsub.vf v8, v8, {mean}",
            "vfmul.vf v8, v8, {inv_std}",
            "vse32.v v8, ({pout})",
            "slli t1, t0, 2",
            "add {pin}, {pin}, t1",
            "add {pout}, {pout}, t1",
            "sub {rem}, {rem}, t0",
            "bnez {rem}, 4b",
            pin = inout(reg) pin,
            pout = inout(reg) pout,
            rem = inout(reg) rem_out,
            mean = in(freg) mean,
            inv_std = in(freg) inv_std,
            out("t0") _, out("t1") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            options(nostack),
        );
    }

    if let Some(w) = weight {
        rvv_mul_assign_f32(out, w);
    }
    if let Some(b) = bias {
        axpy_f32(out, b, 1.0_f32);
    }
    Ok(())
}

/// Returns the caller scratch slice length (`f32` elements) needed by
/// [`ttm_time_channel_mix_with_scratch_f32`], or `None` on overflow/zero dimensions.
#[must_use]
pub fn ttm_time_channel_mix_scratch_len(
    n_patches: usize,
    n_channels: usize,
    time_hidden_dim: usize,
    channel_hidden_dim: usize,
) -> Option<usize> {
    if n_patches == 0 || n_channels == 0 || time_hidden_dim == 0 || channel_hidden_dim == 0 {
        return None;
    }
    let stage_elems = n_patches.checked_mul(n_channels)?;
    let time_work = n_patches.checked_mul(3)?.checked_add(time_hidden_dim)?;
    let chan_work = n_channels.checked_add(channel_hidden_dim)?;
    stage_elems.checked_add(time_work.max(chan_work))
}

fn run_axis_mlp_vector(
    norm_buf: &mut [f32],
    hidden_buf: &mut [f32],
    out_vec: &mut [f32],
    x_vec: &[f32],
    weights: &AxisMlpWeights<'_>,
    eps: f32,
) -> Result<(), K3Error> {
    let dim = weights.dim;
    let hidden_dim = weights.hidden_dim;
    layernorm_f32(
        &mut norm_buf[..dim],
        x_vec,
        Some(weights.norm_weight),
        Some(weights.norm_bias),
        eps,
    )?;

    for (h, &b1) in weights.fc1_bias.iter().enumerate() {
        let w_row = &weights.fc1_weight[h * dim..(h + 1) * dim];
        hidden_buf[h] = dot_f32(w_row, &norm_buf[..dim]) + b1;
    }
    gelu_inplace_f32(&mut hidden_buf[..hidden_dim])?;

    for (d, (&x_val, &b2)) in x_vec.iter().zip(weights.fc2_bias.iter()).enumerate() {
        let w_row = &weights.fc2_weight[d * hidden_dim..(d + 1) * hidden_dim];
        out_vec[d] = x_val + dot_f32(w_row, &hidden_buf[..hidden_dim]) + b2;
    }
    Ok(())
}

/// Applies the `TinyTimeMixers` (`TTM`) patch/time-axis residual MLP block along axis 0 of a
/// row-major `[n_patches, n_channels]` matrix.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if dimensions are zero or overflow, `weights` do not
/// match `n_patches`, `eps <= 0`, or any tensor/parameter value is non-finite.
pub fn ttm_patch_mix_f32(
    out: &mut [f32],
    input: &[f32],
    n_patches: usize,
    n_channels: usize,
    weights: &AxisMlpWeights<'_>,
    eps: f32,
) -> Result<(), K3Error> {
    let n_patches = check_positive_dim("ttm_n_patches", n_patches)?;
    let n_channels = check_positive_dim("ttm_n_channels", n_channels)?;
    let total = checked_product(
        "ttm_patch_mix",
        n_patches,
        n_channels,
        "n_patches * n_channels",
    )?;
    check_positive_finite("ttm_eps", eps)?;
    check_exact_len("ttm_input", input.len(), total)?;
    check_exact_len("ttm_out", out.len(), total)?;
    check_all_finite("ttm_input", input)?;
    weights.validate("ttm_time", n_patches)?;

    let mut x_col = vec![0.0_f32; n_patches];
    let mut out_col = vec![0.0_f32; n_patches];
    let mut norm_buf = vec![0.0_f32; n_patches];
    let mut hidden_buf = vec![0.0_f32; weights.hidden_dim];

    run_patch_mix_inner(
        out,
        input,
        n_patches,
        n_channels,
        weights,
        eps,
        &mut x_col,
        &mut out_col,
        &mut norm_buf,
        &mut hidden_buf,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "internal helper threads pre-validated scratch buffers for patch column gather/scatter"
)]
fn run_patch_mix_inner(
    out: &mut [f32],
    input: &[f32],
    n_patches: usize,
    n_channels: usize,
    weights: &AxisMlpWeights<'_>,
    eps: f32,
    x_col: &mut [f32],
    out_col: &mut [f32],
    norm_buf: &mut [f32],
    hidden_buf: &mut [f32],
) -> Result<(), K3Error> {
    for c in 0..n_channels {
        for p in 0..n_patches {
            x_col[p] = input[p * n_channels + c];
        }
        run_axis_mlp_vector(
            &mut norm_buf[..n_patches],
            &mut hidden_buf[..weights.hidden_dim],
            &mut out_col[..n_patches],
            &x_col[..n_patches],
            weights,
            eps,
        )?;
        for p in 0..n_patches {
            out[p * n_channels + c] = out_col[p];
        }
    }
    Ok(())
}

/// Applies the `TinyTimeMixers` (`TTM`) channel-axis residual MLP block along axis 1 of a
/// row-major `[n_patches, n_channels]` matrix.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if dimensions are zero or overflow, `weights` do not
/// match `n_channels`, `eps <= 0`, or any tensor/parameter value is non-finite.
pub fn ttm_channel_mix_f32(
    out: &mut [f32],
    input: &[f32],
    n_patches: usize,
    n_channels: usize,
    weights: &AxisMlpWeights<'_>,
    eps: f32,
) -> Result<(), K3Error> {
    let n_patches = check_positive_dim("ttm_n_patches", n_patches)?;
    let n_channels = check_positive_dim("ttm_n_channels", n_channels)?;
    let total = checked_product(
        "ttm_channel_mix",
        n_patches,
        n_channels,
        "n_patches * n_channels",
    )?;
    check_positive_finite("ttm_eps", eps)?;
    check_exact_len("ttm_input", input.len(), total)?;
    check_exact_len("ttm_out", out.len(), total)?;
    check_all_finite("ttm_input", input)?;
    weights.validate("ttm_channel", n_channels)?;

    let mut norm_buf = vec![0.0_f32; n_channels];
    let mut hidden_buf = vec![0.0_f32; weights.hidden_dim];
    run_channel_mix_inner(
        out,
        input,
        n_patches,
        n_channels,
        weights,
        eps,
        &mut norm_buf,
        &mut hidden_buf,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "internal helper threads pre-validated scratch buffers for contiguous channel rows"
)]
fn run_channel_mix_inner(
    out: &mut [f32],
    input: &[f32],
    n_patches: usize,
    n_channels: usize,
    weights: &AxisMlpWeights<'_>,
    eps: f32,
    norm_buf: &mut [f32],
    hidden_buf: &mut [f32],
) -> Result<(), K3Error> {
    for p in 0..n_patches {
        let base = p * n_channels;
        run_axis_mlp_vector(
            &mut norm_buf[..n_channels],
            &mut hidden_buf[..weights.hidden_dim],
            &mut out[base..base + n_channels],
            &input[base..base + n_channels],
            weights,
            eps,
        )?;
    }
    Ok(())
}

/// Executes a full `TinyTimeMixers` (`TTM`) dual-axis mixer block (`time/patch` residual MLP followed
/// by `channel` residual MLP) over row-major `[n_patches, n_channels]`.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if dimensions are zero or overflow, weight geometry mismatches,
/// `eps <= 0`, or any input/parameter entry is non-finite (`NaN`/`±Inf`).
pub fn ttm_time_channel_mix_f32(
    out: &mut [f32],
    input: &[f32],
    n_patches: usize,
    n_channels: usize,
    time_weights: &AxisMlpWeights<'_>,
    channel_weights: &AxisMlpWeights<'_>,
    eps: f32,
) -> Result<(), K3Error> {
    let req_scratch = ttm_time_channel_mix_scratch_len(
        n_patches,
        n_channels,
        time_weights.hidden_dim,
        channel_weights.hidden_dim,
    )
    .ok_or_else(|| {
        bounds_error(
            "ttm_time_channel_mix",
            0,
            0,
            0,
            "zero dimension or usize overflow computing TTM scratch size".to_owned(),
        )
    })?;
    let mut scratch = vec![0.0_f32; req_scratch];
    ttm_time_channel_mix_with_scratch_f32(
        out,
        input,
        n_patches,
        n_channels,
        time_weights,
        channel_weights,
        eps,
        &mut scratch,
    )
}

/// Zero-allocation variant of [`ttm_time_channel_mix_f32`] using a caller-supplied `scratch` slice
/// sized to at least [`ttm_time_channel_mix_scratch_len`].
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if dimensions/weights are invalid, `scratch` is too small,
/// or any input/parameter value is non-finite.
#[expect(
    clippy::too_many_arguments,
    reason = "zero-allocation TTM kernel takes explicit matrix dimensions, dual axis weights, eps, and caller scratch"
)]
pub fn ttm_time_channel_mix_with_scratch_f32(
    out: &mut [f32],
    input: &[f32],
    n_patches: usize,
    n_channels: usize,
    time_weights: &AxisMlpWeights<'_>,
    channel_weights: &AxisMlpWeights<'_>,
    eps: f32,
    scratch: &mut [f32],
) -> Result<(), K3Error> {
    let n_patches = check_positive_dim("ttm_n_patches", n_patches)?;
    let n_channels = check_positive_dim("ttm_n_channels", n_channels)?;
    let total = checked_product(
        "ttm_time_channel_mix",
        n_patches,
        n_channels,
        "n_patches * n_channels",
    )?;
    check_positive_finite("ttm_eps", eps)?;
    check_exact_len("ttm_input", input.len(), total)?;
    check_exact_len("ttm_out", out.len(), total)?;
    check_all_finite("ttm_input", input)?;
    time_weights.validate("ttm_time", n_patches)?;
    channel_weights.validate("ttm_channel", n_channels)?;

    let req_scratch = ttm_time_channel_mix_scratch_len(
        n_patches,
        n_channels,
        time_weights.hidden_dim,
        channel_weights.hidden_dim,
    )
    .ok_or_else(|| {
        bounds_error(
            "ttm_scratch",
            0,
            0,
            scratch.len().saturating_mul(size_of::<f32>()),
            "TTM scratch dimension overflow".to_owned(),
        )
    })?;
    if scratch.len() < req_scratch {
        return Err(bounds_error(
            "ttm_scratch",
            0,
            req_scratch.saturating_mul(size_of::<f32>()),
            scratch.len().saturating_mul(size_of::<f32>()),
            format!(
                "TTM scratch requires at least {req_scratch} f32 elements, got {}",
                scratch.len()
            ),
        ));
    }

    let (stage1_buf, work_buf) = scratch.split_at_mut(total);
    {
        let (col_in, rem1) = work_buf.split_at_mut(n_patches);
        let (col_out, rem2) = rem1.split_at_mut(n_patches);
        let (norm_buf, hidden_buf) = rem2.split_at_mut(n_patches);
        run_patch_mix_inner(
            stage1_buf,
            input,
            n_patches,
            n_channels,
            time_weights,
            eps,
            col_in,
            col_out,
            norm_buf,
            hidden_buf,
        )?;
    }
    let (norm_c, hidden_c) = work_buf.split_at_mut(n_channels);
    run_channel_mix_inner(
        out,
        stage1_buf,
        n_patches,
        n_channels,
        channel_weights,
        eps,
        norm_c,
        hidden_c,
    )
}

fn validate_tabpfn_buffers(
    out: &[f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    shape: TabPfnAttentionShape,
    scale: f32,
) -> Result<(usize, usize, usize), K3Error> {
    let (d_model, total, n_ctx) = shape.validated()?;
    check_positive_finite("tabpfn_scale", scale)?;
    check_exact_len("tabpfn_out", out.len(), total)?;
    check_exact_len("tabpfn_q", q.len(), total)?;
    check_exact_len("tabpfn_k", k.len(), total)?;
    check_exact_len("tabpfn_v", v.len(), total)?;
    check_all_finite("tabpfn_q", q)?;
    check_all_finite("tabpfn_k", k)?;
    check_all_finite("tabpfn_v", v)?;
    Ok((d_model, total, n_ctx))
}

#[expect(
    clippy::too_many_arguments,
    reason = "unified axis-agnostic attention kernel parameterized over outer/position strides"
)]
fn run_tabpfn_axis_attention(
    out: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_heads: usize,
    head_dim: usize,
    outer_len: usize,
    q_len: usize,
    kv_len: usize,
    outer_step: usize,
    pos_step: usize,
    scale: f32,
) -> Result<(), K3Error> {
    let mut scores = vec![0.0_f32; kv_len];

    for outer in 0..outer_len {
        let outer_base = outer * outer_step;
        for h in 0..n_heads {
            let head_base = outer_base + h * head_dim;
            for qi in 0..q_len {
                let q_off = head_base + qi * pos_step;
                let q_head = &q[q_off..q_off + head_dim];

                for (kj, score_slot) in scores[..kv_len].iter_mut().enumerate() {
                    let k_off = head_base + kj * pos_step;
                    let score = dot_f32(q_head, &k[k_off..k_off + head_dim]) * scale;
                    if !score.is_finite() {
                        return Err(bounds_error(
                            "tabpfn_attention_score",
                            kj.saturating_mul(size_of::<f32>()),
                            size_of::<f32>(),
                            kv_len.saturating_mul(size_of::<f32>()),
                            format!("non-finite attention score ({score}) encountered"),
                        ));
                    }
                    *score_slot = score;
                }

                softmax_scores_inplace(&mut scores[..kv_len]);

                let out_head = &mut out[q_off..q_off + head_dim];
                out_head.fill(0.0_f32);
                for (kj, &weight) in scores[..kv_len].iter().enumerate() {
                    if weight == 0.0_f32 {
                        continue;
                    }
                    let v_off = head_base + kj * pos_step;
                    axpy_f32(out_head, &v[v_off..v_off + head_dim], weight);
                }
            }
        }
    }

    Ok(())
}

/// Executes intra-row feature-axis multi-head scaled dot-product attention for `TabPFN-v2` over
/// row-major `[n_samples, n_features, n_heads * head_dim]` tensors.
///
/// For each sample row `s in 0..n_samples`, all `n_features` feature tokens attend bidirectionally.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `shape` is invalid or overflows `usize`, `scale <= 0` or
/// is non-finite, buffer lengths mismatch, any input is non-finite, or an attention score overflows.
pub fn tabpfn_feature_attention_f32(
    out: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    shape: TabPfnAttentionShape,
    scale: f32,
) -> Result<(), K3Error> {
    let (d_model, _, _) = validate_tabpfn_buffers(out, q, k, v, shape, scale)?;
    run_tabpfn_axis_attention(
        out,
        q,
        k,
        v,
        shape.n_heads,
        shape.head_dim,
        shape.n_samples,
        shape.n_features,
        shape.n_features,
        shape.n_features * d_model,
        d_model,
        scale,
    )
}

/// Executes cross-row sample-axis multi-head scaled dot-product attention for `TabPFN-v2` over
/// row-major `[n_samples, n_features, n_heads * head_dim]` tensors.
///
/// For each feature column `f in 0..n_features`, all sample rows attend strictly across context rows
/// `0..n_ctx` (`shape.context_samples.unwrap_or(shape.n_samples)`), preventing query-to-context or
/// query-to-query target leakage.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `shape` is invalid or overflows `usize`, `scale <= 0` or
/// is non-finite, buffer lengths mismatch, any input is non-finite, or an attention score overflows.
pub fn tabpfn_sample_attention_f32(
    out: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    shape: TabPfnAttentionShape,
    scale: f32,
) -> Result<(), K3Error> {
    let (d_model, _, n_ctx) = validate_tabpfn_buffers(out, q, k, v, shape, scale)?;
    run_tabpfn_axis_attention(
        out,
        q,
        k,
        v,
        shape.n_heads,
        shape.head_dim,
        shape.n_features,
        shape.n_samples,
        n_ctx,
        d_model,
        shape.n_features * d_model,
        scale,
    )
}

/// Dispatches `TabPFN-v2` dual-axis attention along either [`TabPfnAttentionAxis::Feature`] or
/// [`TabPfnAttentionAxis::Sample`].
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] on invalid shape/scale, slice mismatches, or non-finite values.
pub fn tabpfn_dual_axis_attention_f32(
    out: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    shape: TabPfnAttentionShape,
    axis: TabPfnAttentionAxis,
    scale: f32,
) -> Result<(), K3Error> {
    match axis {
        TabPfnAttentionAxis::Feature => tabpfn_feature_attention_f32(out, q, k, v, shape, scale),
        TabPfnAttentionAxis::Sample => tabpfn_sample_attention_f32(out, q, k, v, shape, scale),
    }
}

/// Executes sequential dual-axis self-attention (`Feature` axis followed by `Sample` axis) over a
/// shared `[n_samples, n_features, n_heads * head_dim]` hidden tensor (`Q = K = V`).
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] on invalid shape/scale, mismatched slice lengths, or
/// non-finite inputs/scores.
pub fn tabpfn_dual_axis_self_attention_f32(
    out: &mut [f32],
    hidden: &[f32],
    shape: TabPfnAttentionShape,
    scale: f32,
) -> Result<(), K3Error> {
    let (_, total, _) = validate_tabpfn_buffers(out, hidden, hidden, hidden, shape, scale)?;
    let mut mid = vec![0.0_f32; total];
    tabpfn_feature_attention_f32(&mut mid, hidden, hidden, hidden, shape, scale)?;
    tabpfn_sample_attention_f32(out, &mid, &mid, &mid, shape, scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BaselineKernel, VectorKernel};

    fn to_f32(val: usize) -> f32 {
        f32::from(u16::try_from(val).expect("test dimension fits in u16"))
    }

    struct OwnedMlp {
        norm_weight: Vec<f32>,
        norm_bias: Vec<f32>,
        fc1_weight: Vec<f32>,
        fc1_bias: Vec<f32>,
        fc2_weight: Vec<f32>,
        fc2_bias: Vec<f32>,
        dim: usize,
        hidden_dim: usize,
    }

    impl OwnedMlp {
        fn build(dim: usize, hidden_dim: usize, scale: f32) -> Self {
            let bias_sign = if scale == 0.0_f32 {
                0.0_f32
            } else {
                scale.signum()
            };
            Self {
                norm_weight: vec![1.0_f32; dim],
                norm_bias: vec![0.0_f32; dim],
                fc1_weight: (0..(hidden_dim * dim))
                    .map(|i| (to_f32(i % 5) - 2.0) * scale)
                    .collect(),
                fc1_bias: vec![0.05_f32 * bias_sign; hidden_dim],
                fc2_weight: (0..(dim * hidden_dim))
                    .map(|i| (to_f32(i % 7) - 3.0) * scale)
                    .collect(),
                fc2_bias: vec![-0.02_f32 * bias_sign; dim],
                dim,
                hidden_dim,
            }
        }

        fn as_weights(&self) -> AxisMlpWeights<'_> {
            AxisMlpWeights {
                norm_weight: &self.norm_weight,
                norm_bias: &self.norm_bias,
                fc1_weight: &self.fc1_weight,
                fc1_bias: &self.fc1_bias,
                fc2_weight: &self.fc2_weight,
                fc2_bias: &self.fc2_bias,
                dim: self.dim,
                hidden_dim: self.hidden_dim,
            }
        }
    }

    #[test]
    fn test_tokenize_continuous_features_parity_and_stripmining() {
        let n_features = 5usize;
        let d_token = 19usize; // odd length exercises VLA tails across VLEN=256 and VLEN=1024
        let raw = [0.5_f32, -1.25, 2.0, 0.0, 3.75];
        let mut weight = vec![0.0_f32; n_features * d_token];
        let mut bias = vec![0.0_f32; n_features * d_token];
        for i in 0..weight.len() {
            weight[i] = (to_f32(i % 11) - 5.0) * 0.25;
            bias[i] = (to_f32(i % 7) - 3.0) * 0.1;
        }

        let mut out_dispatch = vec![0.0_f32; n_features * d_token];
        let mut out_scalar = vec![0.0_f32; n_features * d_token];
        let mut out_vlen256 = vec![0.0_f32; n_features * d_token];
        let mut out_vlen1024 = vec![0.0_f32; n_features * d_token];
        let mut out_trait = vec![0.0_f32; n_features * d_token];

        tokenize_continuous_features_f32(
            &mut out_dispatch,
            &raw,
            &weight,
            &bias,
            n_features,
            d_token,
        )
        .unwrap();
        tokenize_continuous_features_f32_scalar(
            &mut out_scalar,
            &raw,
            &weight,
            &bias,
            n_features,
            d_token,
        )
        .unwrap();
        tokenize_continuous_features_f32_vla_stripmined(
            &mut out_vlen256,
            &raw,
            &weight,
            &bias,
            n_features,
            d_token,
            256,
        )
        .unwrap();
        tokenize_continuous_features_f32_vla_stripmined(
            &mut out_vlen1024,
            &raw,
            &weight,
            &bias,
            n_features,
            d_token,
            1024,
        )
        .unwrap();
        BaselineKernel::new()
            .tokenize_continuous_features_f32(
                &mut out_trait,
                &raw,
                &weight,
                &bias,
                n_features,
                d_token,
            )
            .unwrap();

        assert_eq!(out_vlen256, out_scalar);
        assert_eq!(out_vlen1024, out_scalar);
        assert_eq!(out_dispatch, out_trait);
        for (a, b) in out_dispatch.iter().zip(out_scalar.iter()) {
            assert!((a - b).abs() < 1e-6, "diff too large: {a} vs {b}");
        }
        // Row 3 has raw[3] == 0.0 -> output equals bias row verbatim
        assert_eq!(
            &out_dispatch[3 * d_token..4 * d_token],
            &bias[3 * d_token..4 * d_token]
        );
    }

    #[test]
    fn test_tokenize_continuous_features_fail_closed() {
        let mut out = vec![42.0_f32; 6];
        let raw = [1.0_f32, f32::NAN];
        let weight = [1.0_f32; 6];
        let bias = [0.0_f32; 6];
        let err =
            tokenize_continuous_features_f32(&mut out, &raw, &weight, &bias, 2, 3).unwrap_err();
        assert!(matches!(err, K3Error::TensorBoundsError { .. }));
        assert_eq!(
            out, [42.0_f32; 6],
            "out must stay unmodified on validation failure"
        );

        assert!(matches!(
            tokenize_continuous_features_f32(
                &mut out,
                &[1.0, 2.0],
                &[f32::INFINITY; 6],
                &bias,
                2,
                3
            ),
            Err(K3Error::TensorBoundsError { .. })
        ));
        assert!(matches!(
            tokenize_continuous_features_f32(
                &mut out,
                &[1.0, 2.0],
                &weight,
                &[f32::NEG_INFINITY; 6],
                2,
                3
            ),
            Err(K3Error::TensorBoundsError { .. })
        ));
        assert!(matches!(
            tokenize_continuous_features_f32(&mut [], &[], &[], &[], 0, 3),
            Err(K3Error::TensorBoundsError { .. })
        ));
        assert!(matches!(
            tokenize_continuous_features_f32(&mut out[..5], &[1.0, 2.0], &weight, &bias, 2, 3),
            Err(K3Error::TensorBoundsError { .. })
        ));
    }

    #[test]
    fn test_gelu_and_layernorm_parity_and_validation() {
        let x: Vec<f32> = (-12_i16..=12_i16).map(|v| f32::from(v) * 1.5).collect();
        let mut gelu_d = vec![0.0_f32; x.len()];
        let mut gelu_s = vec![0.0_f32; x.len()];
        gelu_f32(&mut gelu_d, &x).unwrap();
        gelu_f32_scalar(&mut gelu_s, &x).unwrap();
        for (a, b) in gelu_d.iter().zip(gelu_s.iter()) {
            assert!((a - b).abs() < 2e-5, "GELU mismatch: {a} vs {b}");
        }

        let mut ln_in = x.clone();
        gelu_inplace_f32(&mut ln_in).unwrap();
        for (a, b) in ln_in.iter().zip(gelu_d.iter()) {
            assert!((a - b).abs() < 2e-5, "GELU inplace mismatch: {a} vs {b}");
        }

        let gamma: Vec<f32> = (0..x.len()).map(|i| 0.5 + to_f32(i) * 0.05).collect();
        let beta: Vec<f32> = (0..x.len()).map(|i| -0.25 + to_f32(i) * 0.02).collect();
        let mut ln_out = vec![0.0_f32; x.len()];
        let mut ln_scalar = vec![0.0_f32; x.len()];
        let mut ln_256 = vec![0.0_f32; x.len()];
        let mut ln_1024 = vec![0.0_f32; x.len()];

        layernorm_f32(&mut ln_out, &x, Some(&gamma), Some(&beta), 1e-5).unwrap();
        layernorm_f32_scalar(&mut ln_scalar, &x, Some(&gamma), Some(&beta), 1e-5).unwrap();
        layernorm_f32_vla_stripmined(&mut ln_256, &x, Some(&gamma), Some(&beta), 1e-5, 256)
            .unwrap();
        layernorm_f32_vla_stripmined(&mut ln_1024, &x, Some(&gamma), Some(&beta), 1e-5, 1024)
            .unwrap();

        for i in 0..x.len() {
            assert!((ln_out[i] - ln_scalar[i]).abs() < 1e-5);
            assert!((ln_256[i] - ln_scalar[i]).abs() < 1e-5);
            assert!((ln_1024[i] - ln_scalar[i]).abs() < 1e-5);
        }

        // Unaffined zero-mean / unit-variance check on non-constant vector
        let mut unit = vec![0.0_f32; x.len()];
        layernorm_f32(&mut unit, &x, None, None, 1e-6).unwrap();
        let mean: f32 = unit.iter().sum::<f32>() / to_f32(unit.len());
        let var: f32 =
            unit.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / to_f32(unit.len());
        assert!(mean.abs() < 1e-5);
        assert!((var - 1.0).abs() < 1e-3);

        // Fail-closed checks preserve `out`
        let mut untouched = [9.0_f32; 4];
        assert!(matches!(
            layernorm_f32(&mut untouched, &[1.0, 2.0, 3.0, 4.0], None, None, 0.0),
            Err(K3Error::TensorBoundsError { .. })
        ));
        assert!(matches!(
            gelu_f32(&mut untouched, &[1.0, f32::NAN, 3.0, 4.0]),
            Err(K3Error::TensorBoundsError { .. })
        ));
        assert_eq!(untouched, [9.0_f32; 4]);
    }

    #[test]
    fn test_ttm_time_channel_mix_zero_alloc_and_residual() {
        let n_patches = 4usize;
        let n_channels = 3usize;
        let time_h = 6usize;
        let chan_h = 5usize;

        let input: Vec<f32> = (0..(n_patches * n_channels))
            .map(|i| (to_f32(i) - 5.5) * 0.2)
            .collect();

        let owned_time = OwnedMlp::build(n_patches, time_h, 0.1);
        let owned_chan = OwnedMlp::build(n_channels, chan_h, 0.12);
        let time_w = owned_time.as_weights();
        let chan_w = owned_chan.as_weights();

        let mut mid = vec![0.0_f32; n_patches * n_channels];
        let mut expected = vec![0.0_f32; n_patches * n_channels];
        ttm_patch_mix_f32(&mut mid, &input, n_patches, n_channels, &time_w, 1e-5).unwrap();
        ttm_channel_mix_f32(&mut expected, &mid, n_patches, n_channels, &chan_w, 1e-5).unwrap();

        let mut out_full = vec![0.0_f32; n_patches * n_channels];
        BaselineKernel::new()
            .ttm_time_channel_mix_f32(
                &mut out_full,
                &input,
                n_patches,
                n_channels,
                &time_w,
                &chan_w,
                1e-5,
            )
            .unwrap();
        #[cfg(not(miri))]
        assert_eq!(out_full, expected);
        for (a, b) in out_full.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "TTM full mismatch: {a} vs {b}");
        }

        let req = ttm_time_channel_mix_scratch_len(n_patches, n_channels, time_h, chan_h).unwrap();
        let mut scratch = vec![0.0_f32; req];
        let mut out_scratch = vec![0.0_f32; n_patches * n_channels];
        ttm_time_channel_mix_with_scratch_f32(
            &mut out_scratch,
            &input,
            n_patches,
            n_channels,
            &time_w,
            &chan_w,
            1e-5,
            &mut scratch,
        )
        .unwrap();
        #[cfg(not(miri))]
        assert_eq!(out_scratch, expected);
        for (a, b) in out_scratch.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "TTM scratch mismatch: {a} vs {b}");
        }

        // Zero MLP weights preserve pure residual identity `out == input`.
        let id_time = OwnedMlp::build(n_patches, time_h, 0.0);
        let id_chan = OwnedMlp::build(n_channels, chan_h, 0.0);
        let mut out_id = vec![0.0_f32; n_patches * n_channels];
        ttm_time_channel_mix_f32(
            &mut out_id,
            &input,
            n_patches,
            n_channels,
            &id_time.as_weights(),
            &id_chan.as_weights(),
            1e-5,
        )
        .unwrap();
        assert_eq!(out_id, input);

        // Fail-closed: tiny scratch and NaN weights keep `out_id` untouched.
        let snapshot = out_id.clone();
        assert!(matches!(
            ttm_time_channel_mix_with_scratch_f32(
                &mut out_id,
                &input,
                n_patches,
                n_channels,
                &time_w,
                &chan_w,
                1e-5,
                &mut scratch[..req - 1],
            ),
            Err(K3Error::TensorBoundsError { .. })
        ));
        let mut bad_chan = owned_chan;
        bad_chan.fc2_bias[0] = f32::NAN;
        assert!(matches!(
            ttm_time_channel_mix_f32(
                &mut out_id,
                &input,
                n_patches,
                n_channels,
                &time_w,
                &bad_chan.as_weights(),
                1e-5,
            ),
            Err(K3Error::TensorBoundsError { .. })
        ));
        assert_eq!(out_id, snapshot);
    }

    #[test]
    fn test_tabpfn_dual_axis_attention_isolation_and_bounds() {
        let n_samples = 4usize;
        let n_features = 3usize;
        let n_heads = 2usize;
        let head_dim = 4usize;
        let d_model = n_heads * head_dim;
        let row_elems = n_features * d_model;
        let total = n_samples * row_elems;
        let scale = 1.0_f32 / to_f32(head_dim).sqrt();

        let shape_ctx = TabPfnAttentionShape::new(n_samples, n_features, n_heads, head_dim)
            .with_context_samples(2);
        let q: Vec<f32> = (0..total).map(|i| (to_f32(i % 13) - 6.0) * 0.1).collect();
        let k: Vec<f32> = (0..total).map(|i| (to_f32(i % 11) - 5.0) * 0.12).collect();
        let v: Vec<f32> = (0..total).map(|i| (to_f32(i % 9) - 4.0) * 0.15).collect();

        // Feature-axis attention matches per-sample bidirectional_attention_f32.
        let mut out_feat = vec![0.0_f32; total];
        tabpfn_dual_axis_attention_f32(
            &mut out_feat,
            &q,
            &k,
            &v,
            shape_ctx,
            TabPfnAttentionAxis::Feature,
            scale,
        )
        .unwrap();
        for s in 0..n_samples {
            let mut ref_row = vec![0.0_f32; row_elems];
            let r = s * row_elems..(s + 1) * row_elems;
            crate::bidirectional_attention_f32(
                &mut ref_row,
                &q[r.clone()],
                &k[r.clone()],
                &v[r.clone()],
                n_features,
                n_heads,
                head_dim,
                None,
                scale,
            )
            .unwrap();
            for (a, b) in out_feat[r].iter().zip(ref_row.iter()) {
                assert!((a - b).abs() < 1e-6);
            }
        }

        // Context-to-query sample isolation: mutating query row 3 cannot affect rows 0..3.
        let mut out_sample_a = vec![0.0_f32; total];
        tabpfn_sample_attention_f32(&mut out_sample_a, &q, &k, &v, shape_ctx, scale).unwrap();

        let mut q_mut = q.clone();
        let mut k_mut = k.clone();
        let mut v_mut = v.clone();
        for idx in (3 * row_elems)..(4 * row_elems) {
            q_mut[idx] += 5.0;
            k_mut[idx] -= 5.0;
            v_mut[idx] *= 3.0;
        }
        let mut out_sample_b = vec![0.0_f32; total];
        BaselineKernel::new()
            .tabpfn_dual_axis_attention_f32(
                &mut out_sample_b,
                &q_mut,
                &k_mut,
                &v_mut,
                shape_ctx,
                TabPfnAttentionAxis::Sample,
                scale,
            )
            .unwrap();
        #[cfg(not(miri))]
        assert_eq!(out_sample_a[..3 * row_elems], out_sample_b[..3 * row_elems]);
        for (a, b) in out_sample_a[..3 * row_elems]
            .iter()
            .zip(out_sample_b[..3 * row_elems].iter())
        {
            assert!(
                (a - b).abs() < 1e-5,
                "TabPFN context isolation mismatch: {a} vs {b}"
            );
        }

        // Sequential dual-axis self-attention runs cleanly.
        let mut dual_out = vec![0.0_f32; total];
        tabpfn_dual_axis_self_attention_f32(&mut dual_out, &q, shape_ctx, scale).unwrap();
        assert!(dual_out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_tabpfn_dual_axis_attention_bounds() {
        let shape_ctx = TabPfnAttentionShape::new(4, 3, 2, 4).with_context_samples(2);
        let total = shape_ctx.total_elements().unwrap();
        let mut out = vec![7.0_f32; total];
        let q = vec![0.25_f32; total];
        let k = vec![0.25_f32; total];
        let mut v_nan = vec![0.25_f32; total];
        v_nan[3] = f32::NAN;

        assert!(matches!(
            tabpfn_sample_attention_f32(
                &mut out,
                &q,
                &k,
                &q,
                shape_ctx.with_context_samples(5),
                0.5
            ),
            Err(K3Error::TensorBoundsError { .. })
        ));
        assert!(matches!(
            tabpfn_sample_attention_f32(
                &mut out,
                &q,
                &k,
                &q,
                shape_ctx.with_context_samples(0),
                0.5
            ),
            Err(K3Error::TensorBoundsError { .. })
        ));
        assert!(matches!(
            tabpfn_feature_attention_f32(&mut out, &q, &k, &q, shape_ctx, 0.0),
            Err(K3Error::TensorBoundsError { .. })
        ));
        assert!(matches!(
            tabpfn_dual_axis_attention_f32(
                &mut out,
                &q,
                &k,
                &v_nan,
                shape_ctx,
                TabPfnAttentionAxis::Sample,
                0.5
            ),
            Err(K3Error::TensorBoundsError { .. })
        ));
        assert_eq!(out, vec![7.0_f32; total]);
    }
}
