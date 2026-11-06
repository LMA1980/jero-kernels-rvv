//! nGPT residual linear interpolation (`ngpt_residual_lerp_f32`) kernels.
//!
//! Provides residual linear interpolation on the unit hypersphere tangent space:
//! `LERP(a, b; alpha) = (1 - alpha) * a + alpha * b`
//! and nGPT channelwise eigen-rate residual interpolation (Issue #41 contract):
//! `L2Norm(a + |alpha| * (b - a))`
//! where `a = L2Norm(residual)` and `b = L2Norm(cur)`.

use k3_core::K3Error;

/// Computes residual linear interpolation on two slices into a separate output slice:
/// `out[i] = (1 - alpha) * residual[i] + alpha * update[i]`
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `residual` is empty or if slice lengths mismatch.
pub fn ngpt_residual_lerp_f32(
    residual: &[f32],
    update: &[f32],
    alpha: f32,
    out: &mut [f32],
) -> Result<(), K3Error> {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        ngpt_residual_lerp_f32_rvv(residual, update, alpha, out)
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        ngpt_residual_lerp_f32_scalar(residual, update, alpha, out)
    }
}

/// Computes residual linear interpolation in-place:
/// `residual[i] <- (1 - alpha) * residual[i] + alpha * update[i]`
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `residual` is empty or if `update.len() != residual.len()`.
pub fn ngpt_residual_lerp_inplace_f32(
    residual: &mut [f32],
    update: &[f32],
    alpha: f32,
) -> Result<(), K3Error> {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        ngpt_residual_lerp_inplace_rvv(residual, update, alpha)
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        ngpt_residual_lerp_inplace_scalar(residual, update, alpha)
    }
}

/// Computes nGPT channelwise eigen-rate residual linear interpolation on the unit hypersphere
/// (Issue #41 contract):
/// `L2Norm(a + |alpha| * (b - a))`
/// where `a = L2Norm(residual)` and `b = L2Norm(cur)`.
/// Updates `cur` in-place.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `cur` is empty or if slice lengths mismatch.
pub fn ngpt_residual_lerp_in_place(
    residual: &[f32],
    cur: &mut [f32],
    alpha_weight: &[f32],
    alpha_scale: f32,
    eps: f32,
) -> Result<(), K3Error> {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        ngpt_residual_lerp_in_place_rvv(residual, cur, alpha_weight, alpha_scale, eps)
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        ngpt_residual_lerp_in_place_scalar(residual, cur, alpha_weight, alpha_scale, eps)
    }
}

/// Computes residual linear interpolation using portable scalar arithmetic.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `residual` is empty or if slice lengths mismatch.
#[expect(
    clippy::float_cmp,
    reason = "exact boundary bypass for alpha = 1.0 identity"
)]
pub fn ngpt_residual_lerp_f32_scalar(
    residual: &[f32],
    update: &[f32],
    alpha: f32,
    out: &mut [f32],
) -> Result<(), K3Error> {
    let len = residual.len();
    if len == 0 || update.len() != len || out.len() != len {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "ngpt_residual_lerp".to_owned(),
            offset: 0,
            byte_len: size_of_val(out),
            buffer_len: size_of_val(residual),
            reason: format!(
                "dimension mismatch: residual={len}, update={}, out={}",
                update.len(),
                out.len()
            ),
        });
    }

    if alpha == 0.0_f32 {
        out.copy_from_slice(residual);
        return Ok(());
    }
    if alpha == 1.0_f32 {
        out.copy_from_slice(update);
        return Ok(());
    }

    let w0 = 1.0_f32 - alpha;
    let w1 = alpha;
    for i in 0..len {
        out[i] = w0 * residual[i] + w1 * update[i];
    }

    Ok(())
}

/// Computes in-place residual linear interpolation using portable scalar arithmetic.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `residual` is empty or if `update.len() != residual.len()`.
#[expect(
    clippy::float_cmp,
    reason = "exact boundary bypass for alpha = 1.0 identity"
)]
pub fn ngpt_residual_lerp_inplace_scalar(
    residual: &mut [f32],
    update: &[f32],
    alpha: f32,
) -> Result<(), K3Error> {
    let len = residual.len();
    if len == 0 || update.len() != len {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "ngpt_residual_lerp_inplace".to_owned(),
            offset: 0,
            byte_len: size_of_val(residual),
            buffer_len: size_of_val(update),
            reason: format!(
                "dimension mismatch: residual={len}, update={}",
                update.len()
            ),
        });
    }

    if alpha == 0.0_f32 {
        return Ok(());
    }
    if alpha == 1.0_f32 {
        residual.copy_from_slice(update);
        return Ok(());
    }

    let w0 = 1.0_f32 - alpha;
    let w1 = alpha;
    for i in 0..len {
        residual[i] = w0 * residual[i] + w1 * update[i];
    }

    Ok(())
}

/// Computes channelwise eigen-rate residual interpolation on the unit hypersphere
/// using portable scalar arithmetic.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `cur` is empty or if slice lengths mismatch.
pub fn ngpt_residual_lerp_in_place_scalar(
    residual: &[f32],
    cur: &mut [f32],
    alpha_weight: &[f32],
    alpha_scale: f32,
    eps: f32,
) -> Result<(), K3Error> {
    let len = cur.len();
    if len == 0 || residual.len() != len || alpha_weight.len() != len {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "ngpt_residual_lerp_in_place".to_owned(),
            offset: 0,
            byte_len: size_of_val(cur),
            buffer_len: size_of_val(residual),
            reason: format!(
                "dimension mismatch: cur={len}, residual={}, alpha_weight={}",
                residual.len(),
                alpha_weight.len()
            ),
        });
    }

    let eps = eps.max(1e-6);
    let sum_sq_r = crate::dot_product_f32_scalar(residual, residual);
    let sum_sq_c = crate::dot_product_f32_scalar(cur, cur);

    let denom_r = (sum_sq_r + eps).sqrt();
    let denom_c = (sum_sq_c + eps).sqrt();

    let scale_r = if denom_r > 0.0_f32 && denom_r.is_finite() {
        1.0_f32 / denom_r
    } else {
        0.0_f32
    };

    let scale_c = if denom_c > 0.0_f32 && denom_c.is_finite() {
        1.0_f32 / denom_c
    } else {
        0.0_f32
    };

    for i in 0..len {
        let a_i = residual[i] * scale_r;
        let b_i = cur[i] * scale_c;
        let alpha_i = (alpha_weight[i] * alpha_scale)
            .abs()
            .clamp(0.0_f32, 1.0_f32);
        cur[i] = (1.0_f32 - alpha_i) * a_i + alpha_i * b_i;
    }

    crate::norm::l2_norm_f32_in_place_scalar(cur, eps);

    Ok(())
}

/// Computes residual linear interpolation using RVV 1.0 inline assembly.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `residual` is empty or if slice lengths mismatch.
#[cfg(all(target_arch = "riscv64", not(miri)))]
#[allow(unused_assignments)]
#[expect(
    clippy::float_cmp,
    reason = "exact boundary bypass for alpha = 1.0 identity"
)]
pub fn ngpt_residual_lerp_f32_rvv(
    residual: &[f32],
    update: &[f32],
    alpha: f32,
    out: &mut [f32],
) -> Result<(), K3Error> {
    let len = residual.len();
    if len == 0 || update.len() != len || out.len() != len {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "ngpt_residual_lerp".to_owned(),
            offset: 0,
            byte_len: size_of_val(out),
            buffer_len: size_of_val(residual),
            reason: format!(
                "dimension mismatch: residual={len}, update={}, out={}",
                update.len(),
                out.len()
            ),
        });
    }

    if alpha == 0.0_f32 {
        out.copy_from_slice(residual);
        return Ok(());
    }
    if alpha == 1.0_f32 {
        out.copy_from_slice(update);
        return Ok(());
    }

    let w0 = 1.0_f32 - alpha;
    let w1 = alpha;

    let mut ptr_res = residual.as_ptr();
    let mut ptr_upd = update.as_ptr();
    let mut ptr_out = out.as_mut_ptr();
    let mut rem = len;

    // SAFETY: `ptr_res`, `ptr_upd`, and `ptr_out` point to valid contiguous slices of `rem` f32 elements.
    // RVV 1.0 strip-mining computes v3 = (residual * w0) + (update * w1) with vfmul.vf and vfmacc.vf.
    unsafe {
        core::arch::asm!(
            "2:",
            "vsetvli t0, {rem}, e32, m1, ta, ma",
            "vle32.v v1, ({pres})",
            "vle32.v v2, ({pupd})",
            "vfmul.vf v3, v1, {w0}",
            "vfmacc.vf v3, {w1}, v2",
            "vse32.v v3, ({pout})",
            "slli t1, t0, 2",
            "add {pres}, {pres}, t1",
            "add {pupd}, {pupd}, t1",
            "add {pout}, {pout}, t1",
            "sub {rem}, {rem}, t0",
            "bnez {rem}, 2b",
            pres = inout(reg) ptr_res,
            pupd = inout(reg) ptr_upd,
            pout = inout(reg) ptr_out,
            rem = inout(reg) rem,
            w0 = in(freg) w0,
            w1 = in(freg) w1,
            out("t0") _,
            out("t1") _,
            out("v1") _,
            out("v2") _,
            out("v3") _,
            options(nostack),
        );
    }

    Ok(())
}

/// Computes in-place residual linear interpolation using RVV 1.0 inline assembly.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `residual` is empty or if `update.len() != residual.len()`.
#[cfg(all(target_arch = "riscv64", not(miri)))]
#[allow(unused_assignments)]
#[expect(
    clippy::float_cmp,
    reason = "exact boundary bypass for alpha = 1.0 identity"
)]
pub fn ngpt_residual_lerp_inplace_rvv(
    residual: &mut [f32],
    update: &[f32],
    alpha: f32,
) -> Result<(), K3Error> {
    let len = residual.len();
    if len == 0 || update.len() != len {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "ngpt_residual_lerp_inplace".to_owned(),
            offset: 0,
            byte_len: size_of_val(residual),
            buffer_len: size_of_val(update),
            reason: format!(
                "dimension mismatch: residual={len}, update={}",
                update.len()
            ),
        });
    }

    if alpha == 0.0_f32 {
        return Ok(());
    }
    if alpha == 1.0_f32 {
        residual.copy_from_slice(update);
        return Ok(());
    }

    let w0 = 1.0_f32 - alpha;
    let w1 = alpha;

    let mut ptr_res = residual.as_mut_ptr();
    let mut ptr_upd = update.as_ptr();
    let mut rem = len;

    // SAFETY: `ptr_res` and `ptr_upd` point to valid contiguous slices of `rem` f32 elements.
    // RVV 1.0 strip-mining computes v3 = (residual * w0) + (update * w1) and stores back into residual.
    unsafe {
        core::arch::asm!(
            "2:",
            "vsetvli t0, {rem}, e32, m1, ta, ma",
            "vle32.v v1, ({pres})",
            "vle32.v v2, ({pupd})",
            "vfmul.vf v3, v1, {w0}",
            "vfmacc.vf v3, {w1}, v2",
            "vse32.v v3, ({pres})",
            "slli t1, t0, 2",
            "add {pres}, {pres}, t1",
            "add {pupd}, {pupd}, t1",
            "sub {rem}, {rem}, t0",
            "bnez {rem}, 2b",
            pres = inout(reg) ptr_res,
            pupd = inout(reg) ptr_upd,
            rem = inout(reg) rem,
            w0 = in(freg) w0,
            w1 = in(freg) w1,
            out("t0") _,
            out("t1") _,
            out("v1") _,
            out("v2") _,
            out("v3") _,
            options(nostack),
        );
    }

    Ok(())
}

/// Computes channelwise eigen-rate residual interpolation on the unit hypersphere
/// using RVV 1.0 acceleration.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `cur` is empty or if slice lengths mismatch.
#[cfg(all(target_arch = "riscv64", not(miri)))]
pub fn ngpt_residual_lerp_in_place_rvv(
    residual: &[f32],
    cur: &mut [f32],
    alpha_weight: &[f32],
    alpha_scale: f32,
    eps: f32,
) -> Result<(), K3Error> {
    let len = cur.len();
    if len == 0 || residual.len() != len || alpha_weight.len() != len {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "ngpt_residual_lerp_in_place".to_owned(),
            offset: 0,
            byte_len: size_of_val(cur),
            buffer_len: size_of_val(residual),
            reason: format!(
                "dimension mismatch: cur={len}, residual={}, alpha_weight={}",
                residual.len(),
                alpha_weight.len()
            ),
        });
    }

    let eps = eps.max(1e-6);
    let sum_sq_r = crate::dot_product_f32_rvv(residual, residual);
    let sum_sq_c = crate::dot_product_f32_rvv(cur, cur);

    let denom_r = (sum_sq_r + eps).sqrt();
    let denom_c = (sum_sq_c + eps).sqrt();

    let scale_r = if denom_r > 0.0_f32 && denom_r.is_finite() {
        1.0_f32 / denom_r
    } else {
        0.0_f32
    };

    let scale_c = if denom_c > 0.0_f32 && denom_c.is_finite() {
        1.0_f32 / denom_c
    } else {
        0.0_f32
    };

    for i in 0..len {
        let a_i = residual[i] * scale_r;
        let b_i = cur[i] * scale_c;
        let alpha_i = (alpha_weight[i] * alpha_scale)
            .abs()
            .clamp(0.0_f32, 1.0_f32);
        cur[i] = (1.0_f32 - alpha_i) * a_i + alpha_i * b_i;
    }

    crate::norm::l2_norm_f32_in_place_rvv(cur, eps);

    Ok(())
}
