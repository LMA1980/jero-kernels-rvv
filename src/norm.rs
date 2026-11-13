//! Unit hyperspherical normalization (`l2_norm_f32`) kernels.
//!
//! Provides parameter-free L2 normalization on the unit hypersphere:
//! `x_i <- x_i / sqrt(sum(x_k^2) + eps)`
//! with defensive zero-norm clamping and subnormal protection.
//! Includes RVV 1.0 VLA strip-mining vector implementations and portable scalar fallbacks.

use jero_core::K3Error;

/// Computes parameter-free unit hyperspherical normalization into a separate output slice:
/// `out_i = x_i / sqrt(sum(x_k^2) + eps)`
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `x` is empty or if `x.len() != out.len()`.
pub fn l2_norm_f32(x: &[f32], out: &mut [f32], eps: f32) -> Result<(), K3Error> {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        l2_norm_f32_rvv(x, out, eps)
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        l2_norm_f32_scalar(x, out, eps)
    }
}

/// Computes parameter-free unit hyperspherical normalization in-place (Issue #41 contract):
/// `values_i <- values_i / sqrt(sum(values_k^2) + eps)`
///
/// If `values` is empty, this operation is a safe no-op.
pub fn l2_norm_f32_in_place(values: &mut [f32], eps: f32) {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        l2_norm_f32_in_place_rvv(values, eps);
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        l2_norm_f32_in_place_scalar(values, eps);
    }
}

/// Computes parameter-free unit hyperspherical normalization in-place, returning a [`Result`].
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `values` is empty.
pub fn l2_norm_inplace_f32(values: &mut [f32], eps: f32) -> Result<(), K3Error> {
    if values.is_empty() {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "l2_norm_inplace".to_owned(),
            offset: 0,
            byte_len: 0,
            buffer_len: 0,
            reason: "empty input slice".to_owned(),
        });
    }
    l2_norm_f32_in_place(values, eps);
    Ok(())
}

/// Computes parameter-free unit hyperspherical normalization using portable scalar arithmetic.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `x` is empty or if `x.len() != out.len()`.
pub fn l2_norm_f32_scalar(x: &[f32], out: &mut [f32], eps: f32) -> Result<(), K3Error> {
    let len = x.len();
    if len == 0 || out.len() != len {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "l2_norm".to_owned(),
            offset: 0,
            byte_len: size_of_val(out),
            buffer_len: size_of_val(x),
            reason: format!("dimension mismatch: input={len}, output={}", out.len()),
        });
    }

    let eps = eps.max(1e-6);
    let sum_sq = crate::dot_product_f32_scalar(x, x);
    let denom = (sum_sq + eps).sqrt();
    let scale = if denom > 0.0_f32 && denom.is_finite() {
        1.0_f32 / denom
    } else {
        0.0_f32
    };

    for i in 0..len {
        out[i] = x[i] * scale;
    }

    Ok(())
}

/// Computes parameter-free unit hyperspherical normalization in-place using portable scalar arithmetic.
pub fn l2_norm_f32_in_place_scalar(values: &mut [f32], eps: f32) {
    let len = values.len();
    if len == 0 {
        return;
    }

    let eps = eps.max(1e-6);
    let sum_sq = crate::dot_product_f32_scalar(values, values);
    let denom = (sum_sq + eps).sqrt();
    let scale = if denom > 0.0_f32 && denom.is_finite() {
        1.0_f32 / denom
    } else {
        0.0_f32
    };

    for val in values.iter_mut() {
        *val *= scale;
    }
}

/// Computes parameter-free unit hyperspherical normalization using RVV 1.0 inline assembly.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `x` is empty or if `x.len() != out.len()`.
#[cfg(all(target_arch = "riscv64", not(miri)))]
#[allow(unused_assignments)]
pub fn l2_norm_f32_rvv(x: &[f32], out: &mut [f32], eps: f32) -> Result<(), K3Error> {
    let len = x.len();
    if len == 0 || out.len() != len {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "l2_norm".to_owned(),
            offset: 0,
            byte_len: size_of_val(out),
            buffer_len: size_of_val(x),
            reason: format!("dimension mismatch: input={len}, output={}", out.len()),
        });
    }

    let eps = eps.max(1e-6);
    let sum_sq = crate::dot_product_f32_rvv(x, x);
    let denom = (sum_sq + eps).sqrt();
    let scale = if denom > 0.0_f32 && denom.is_finite() {
        1.0_f32 / denom
    } else {
        0.0_f32
    };

    let mut ptr_in = x.as_ptr();
    let mut ptr_out = out.as_mut_ptr();
    let mut rem = len;

    // SAFETY: `ptr_in` and `ptr_out` point to valid contiguous slices of `rem` f32 elements.
    // Standard RVV 1.0 VLA strip-mining scales input by `scale` with vfmul.vf.
    unsafe {
        core::arch::asm!(
            "2:",
            "vsetvli t0, {rem}, e32, m1, ta, ma",
            "vle32.v v1, ({pin})",
            "vfmul.vf v1, v1, {scale}",
            "vse32.v v1, ({pout})",
            "slli t1, t0, 2",
            "add {pin}, {pin}, t1",
            "add {pout}, {pout}, t1",
            "sub {rem}, {rem}, t0",
            "bnez {rem}, 2b",
            pin = inout(reg) ptr_in,
            pout = inout(reg) ptr_out,
            rem = inout(reg) rem,
            scale = in(freg) scale,
            out("t0") _,
            out("t1") _,
            out("v1") _,
            options(nostack),
        );
    }

    Ok(())
}

/// Computes parameter-free unit hyperspherical normalization in-place using RVV 1.0 inline assembly.
#[cfg(all(target_arch = "riscv64", not(miri)))]
#[allow(unused_assignments)]
pub fn l2_norm_f32_in_place_rvv(values: &mut [f32], eps: f32) {
    let len = values.len();
    if len == 0 {
        return;
    }

    let eps = eps.max(1e-6);
    let sum_sq = crate::dot_product_f32_rvv(values, values);
    let denom = (sum_sq + eps).sqrt();
    let scale = if denom > 0.0_f32 && denom.is_finite() {
        1.0_f32 / denom
    } else {
        0.0_f32
    };

    let mut ptr = values.as_mut_ptr();
    let mut rem = len;

    // SAFETY: `ptr` points to a valid contiguous slice of `rem` f32 elements.
    // Standard RVV 1.0 VLA strip-mining scales elements in-place with vfmul.vf.
    unsafe {
        core::arch::asm!(
            "2:",
            "vsetvli t0, {rem}, e32, m1, ta, ma",
            "vle32.v v1, ({p})",
            "vfmul.vf v1, v1, {scale}",
            "vse32.v v1, ({p})",
            "slli t1, t0, 2",
            "add {p}, {p}, t1",
            "sub {rem}, {rem}, t0",
            "bnez {rem}, 2b",
            p = inout(reg) ptr,
            rem = inout(reg) rem,
            scale = in(freg) scale,
            out("t0") _,
            out("t1") _,
            out("v1") _,
            options(nostack),
        );
    }
}

fn validate_unit_offset_args(
    tensor_name: &str,
    output: &[f32],
    input: &[f32],
    weight: &[f32],
    eps: f32,
) -> Result<(), K3Error> {
    let len = input.len();
    if len == 0 || output.len() != len || weight.len() != len {
        return Err(K3Error::TensorBoundsError {
            tensor_name: tensor_name.to_owned(),
            offset: 0,
            byte_len: size_of_val(output),
            buffer_len: size_of_val(input),
            reason: format!(
                "dimension mismatch: input={len}, output={}, weight={}",
                output.len(),
                weight.len()
            ),
        });
    }
    if !eps.is_finite() || eps < 0.0 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: tensor_name.to_owned(),
            offset: 0,
            byte_len: size_of_val(output),
            buffer_len: size_of_val(input),
            reason: format!("invalid RMSNorm epsilon: {eps}"),
        });
    }
    Ok(())
}

/// Computes Gemma unit-offset Root Mean Square Normalization (`(1.0 + weight[i])`) into `output`.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if slices are empty, lengths mismatch, or `eps` is negative/non-finite.
pub fn rmsnorm_unit_offset(
    output: &mut [f32],
    input: &[f32],
    weight: &[f32],
    eps: f32,
) -> Result<(), K3Error> {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        rmsnorm_unit_offset_rvv(output, input, weight, eps)
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        rmsnorm_unit_offset_scalar(output, input, weight, eps)
    }
}

/// Computes Gemma unit-offset `RMSNorm` (`(1.0 + weight[i])`) using portable scalar arithmetic.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if slices are empty, lengths mismatch, or `eps` is negative/non-finite.
pub fn rmsnorm_unit_offset_scalar(
    output: &mut [f32],
    input: &[f32],
    weight: &[f32],
    eps: f32,
) -> Result<(), K3Error> {
    validate_unit_offset_args("rmsnorm_unit_offset", output, input, weight, eps)?;
    let len = input.len();
    let sum_sq = crate::dot_product_f32_scalar(input, input);
    #[expect(clippy::cast_precision_loss, reason = "length fits comfortably in f32")]
    let mean_sq = sum_sq / (len as f32);
    let denom = (mean_sq + eps).sqrt();
    let inv_rms = if denom > 0.0_f32 && denom.is_finite() {
        1.0_f32 / denom
    } else {
        0.0_f32
    };

    for i in 0..len {
        output[i] = input[i] * inv_rms * (1.0_f32 + weight[i]);
    }

    Ok(())
}

/// Computes Gemma unit-offset `RMSNorm` (`(1.0 + weight[i])`) using VLA RVV 1.0 inline assembly.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if slices are empty, lengths mismatch, or `eps` is negative/non-finite.
#[cfg(all(target_arch = "riscv64", not(miri)))]
#[allow(unused_assignments)]
pub fn rmsnorm_unit_offset_rvv(
    output: &mut [f32],
    input: &[f32],
    weight: &[f32],
    eps: f32,
) -> Result<(), K3Error> {
    validate_unit_offset_args("rmsnorm_unit_offset", output, input, weight, eps)?;
    let len = input.len();
    let sum_sq = crate::dot_product_f32_rvv(input, input);
    #[expect(clippy::cast_precision_loss, reason = "length fits comfortably in f32")]
    let mean_sq = sum_sq / (len as f32);
    let denom = (mean_sq + eps).sqrt();
    let inv_rms = if denom > 0.0_f32 && denom.is_finite() {
        1.0_f32 / denom
    } else {
        0.0_f32
    };

    let mut ptr_out = output.as_mut_ptr();
    let mut ptr_in = input.as_ptr();
    let mut ptr_w = weight.as_ptr();
    let mut rem = len;
    let one = 1.0_f32;

    // SAFETY: `ptr_out`, `ptr_in`, and `ptr_w` are valid for `rem` f32 elements.
    // VLA RVV 1.0 strip-mining scales `input` by `inv_rms` and multiplies by `(1.0 + weight)`.
    unsafe {
        core::arch::asm!(
            "2:",
            "vsetvli t0, {rem}, e32, m1, ta, ma",
            "vle32.v v1, ({pin})",
            "vfmul.vf v1, v1, {inv_rms}",
            "vle32.v v2, ({pw})",
            "vfadd.vf v2, v2, {one}",
            "vfmul.vv v1, v1, v2",
            "vse32.v v1, ({pout})",
            "slli t1, t0, 2",
            "add {pin}, {pin}, t1",
            "add {pw}, {pw}, t1",
            "add {pout}, {pout}, t1",
            "sub {rem}, {rem}, t0",
            "bnez {rem}, 2b",
            pout = inout(reg) ptr_out,
            pin = inout(reg) ptr_in,
            pw = inout(reg) ptr_w,
            rem = inout(reg) rem,
            inv_rms = in(freg) inv_rms,
            one = in(freg) one,
            out("t0") _,
            out("t1") _,
            out("v1") _,
            out("v2") _,
            options(nostack),
        );
    }

    Ok(())
}

/// Applies Gemma-style per-head QK-Normalization in-place using unit-offset `RMSNorm` (`1.0 + weight`).
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `head_dim == 0`, `heads` is empty, `heads.len()` is not a
/// multiple of `head_dim`, `weight.len() != head_dim`, or `eps` is negative/non-finite.
pub fn qk_norm_inplace(
    heads: &mut [f32],
    head_dim: usize,
    weight: &[f32],
    eps: f32,
) -> Result<(), K3Error> {
    if head_dim == 0
        || heads.is_empty()
        || heads.len() % head_dim != 0
        || weight.len() != head_dim
        || !eps.is_finite()
        || eps < 0.0
    {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "qk_norm_inplace".to_owned(),
            offset: 0,
            byte_len: size_of_val(heads),
            buffer_len: size_of_val(weight),
            reason: format!(
                "invalid QK-norm bounds: heads_len={}, head_dim={head_dim}, weight_len={}, eps={eps}",
                heads.len(),
                weight.len()
            ),
        });
    }

    #[expect(
        clippy::cast_precision_loss,
        reason = "head_dim fits comfortably in f32"
    )]
    let head_dim_f32 = head_dim as f32;
    for head in heads.chunks_exact_mut(head_dim) {
        #[cfg(all(target_arch = "riscv64", not(miri)))]
        let sum_sq = crate::dot_product_f32_rvv(head, head);
        #[cfg(any(not(target_arch = "riscv64"), miri))]
        let sum_sq = crate::dot_product_f32_scalar(head, head);

        let mean_sq = sum_sq / head_dim_f32;
        let denom = (mean_sq + eps).sqrt();
        let inv_rms = if denom > 0.0_f32 && denom.is_finite() {
            1.0_f32 / denom
        } else {
            0.0_f32
        };
        for (val, &w) in head.iter_mut().zip(weight.iter()) {
            *val = *val * inv_rms * (1.0_f32 + w);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rmsnorm_unit_offset() {
        // With zero weights, unit-offset RMSNorm applies gain (1.0 + 0.0) = 1.0 (never zero!).
        let input = [3.0_f32, 4.0, 3.0, 4.0];
        let zero_weights = [0.0_f32; 4];
        let mut out_zero_w = [0.0_f32; 4];
        rmsnorm_unit_offset(&mut out_zero_w, &input, &zero_weights, 1e-6)
            .expect("unit-offset rmsnorm with zero weights succeeds");

        let rms = (12.5_f32 + 1e-6).sqrt();
        for (i, &actual) in out_zero_w.iter().enumerate() {
            let expected = input[i] / rms;
            assert!((actual - expected).abs() < 1e-5);
        }

        // With non-zero weights, output scales by (1.0 + weight[i]).
        let weights = [0.5_f32, -0.25, 1.0, 0.0];
        let mut out = [0.0_f32; 4];
        rmsnorm_unit_offset(&mut out, &input, &weights, 1e-6)
            .expect("unit-offset rmsnorm with explicit weights succeeds");
        for (i, &actual) in out.iter().enumerate() {
            let expected = (input[i] / rms) * (1.0 + weights[i]);
            assert!((actual - expected).abs() < 1e-5);
        }

        // Per-head QK-norm normalizes each head independently.
        let mut heads = [3.0_f32, 4.0, 0.0, 6.0];
        let qk_weights = [0.2_f32, 0.5_f32];
        qk_norm_inplace(&mut heads, 2, &qk_weights, 1e-6).expect("qk_norm_inplace succeeds");
        let rms0 = (12.5_f32 + 1e-6).sqrt();
        let rms1 = (18.0_f32 + 1e-6).sqrt();
        assert!((heads[0] - (3.0 / rms0) * 1.2).abs() < 1e-5);
        assert!((heads[1] - (4.0 / rms0) * 1.5).abs() < 1e-5);
        assert!(heads[2].abs() < 1e-6);
        assert!((heads[3] - (6.0 / rms1) * 1.5).abs() < 1e-5);

        // Bounds mismatch fails closed.
        assert!(matches!(
            qk_norm_inplace(&mut heads, 3, &[0.0; 3], 1e-6),
            Err(K3Error::TensorBoundsError { .. })
        ));
    }
}
