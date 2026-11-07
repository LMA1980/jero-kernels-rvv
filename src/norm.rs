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
