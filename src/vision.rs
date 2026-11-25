//! Vision Transformer (`ViT`) preprocessing, patch embedding, 2D positional interpolation,
//! and `2x2` spatial pixel-shuffle token merging kernels with `RVV 1.0` and scalar parity.

use jero_core::K3Error;

use crate::machine::{
    bounds_error, check_all_finite, check_exact_len, check_positive_dim, check_positive_finite,
    checked_product,
};

#[derive(Debug, Clone, Copy)]
struct BilinearCoord {
    i0: usize,
    i1: usize,
    w: f32,
}

#[inline]
fn bilinear_coord(dst_idx: usize, src_len: usize, dst_len: usize) -> BilinearCoord {
    if src_len == dst_len {
        return BilinearCoord {
            i0: dst_idx,
            i1: dst_idx,
            w: 0.0_f32,
        };
    }
    if src_len <= 1 {
        return BilinearCoord {
            i0: 0,
            i1: 0,
            w: 0.0_f32,
        };
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "image and grid dimensions fit comfortably within f64 exact integer range"
    )]
    let pos = ((dst_idx as f64 + 0.5_f64) * (src_len as f64) / (dst_len as f64) - 0.5_f64)
        .clamp(0.0_f64, (src_len - 1) as f64);
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "pos is clamped to [0, src_len - 1]"
    )]
    let i0 = pos.floor() as usize;
    let i1 = (i0 + 1).min(src_len - 1);
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        reason = "fractional interpolation weight lies in [0.0, 1.0]"
    )]
    let w = (pos - (i0 as f64)) as f32;
    BilinearCoord { i0, i1, w }
}

#[expect(
    clippy::too_many_arguments,
    reason = "vision preprocessing validates explicit source/destination geometry and channel statistics"
)]
fn validate_resize_rgb_args(
    out_chw: &[f32],
    src_hwc: &[f32],
    src_h: usize,
    src_w: usize,
    dst_h: usize,
    dst_w: usize,
    mean: [f32; 3],
    std: [f32; 3],
) -> Result<usize, K3Error> {
    let src_h = check_positive_dim("resize_src_h", src_h)?;
    let src_w = check_positive_dim("resize_src_w", src_w)?;
    let dst_h = check_positive_dim("resize_dst_h", dst_h)?;
    let dst_w = check_positive_dim("resize_dst_w", dst_w)?;

    let src_pixels = checked_product("resize_src", src_h, src_w, "src_h * src_w")?;
    let src_elems = checked_product("resize_src", src_pixels, 3, "src_h * src_w * 3")?;
    let dst_pixels = checked_product("resize_dst", dst_h, dst_w, "dst_h * dst_w")?;
    let dst_elems = checked_product("resize_dst", dst_pixels, 3, "3 * dst_h * dst_w")?;

    check_exact_len("resize_src_hwc", src_hwc.len(), src_elems)?;
    check_exact_len("resize_out_chw", out_chw.len(), dst_elems)?;

    check_all_finite("resize_mean", &mean)?;
    for &s in &std {
        check_positive_finite("resize_std", s)?;
    }
    check_all_finite("resize_src_hwc", src_hwc)?;

    Ok(dst_pixels)
}

fn fill_bilinear_rgb_planes(
    out_chw: &mut [f32],
    src_hwc: &[f32],
    src_h: usize,
    src_w: usize,
    dst_h: usize,
    dst_w: usize,
) {
    let dst_pixels = dst_h * dst_w;
    for dy in 0..dst_h {
        let cy = bilinear_coord(dy, src_h, dst_h);
        let row0 = cy.i0 * src_w * 3;
        let row1 = cy.i1 * src_w * 3;
        let dst_row = dy * dst_w;

        for dx in 0..dst_w {
            let cx = bilinear_coord(dx, src_w, dst_w);
            let off00 = row0 + cx.i0 * 3;
            let off01 = row0 + cx.i1 * 3;
            let off10 = row1 + cx.i0 * 3;
            let off11 = row1 + cx.i1 * 3;
            let dst_idx = dst_row + dx;

            for c in 0..3 {
                let p00 = src_hwc[off00 + c];
                let p01 = src_hwc[off01 + c];
                let p10 = src_hwc[off10 + c];
                let p11 = src_hwc[off11 + c];
                let top = p00 + cx.w * (p01 - p00);
                let bot = p10 + cx.w * (p11 - p10);
                out_chw[c * dst_pixels + dst_idx] = top + cy.w * (bot - top);
            }
        }
    }
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[allow(unused_assignments)]
fn rvv_hwc_to_chw_norm_plane(
    out_plane: &mut [f32],
    src_chan_start: *const f32,
    mean: f32,
    inv_std: f32,
) {
    let mut rem = out_plane.len();
    if rem == 0 {
        return;
    }
    let mut ps = src_chan_start;
    let mut po = out_plane.as_mut_ptr();
    let stride_bytes = 12_isize;

    // SAFETY: `ps` points to `rem` RGB triplets (`stride = 12` bytes) inside `src_hwc` and `po`
    // points to `rem` contiguous `f32` elements in `out_plane`.
    unsafe {
        core::arch::asm!(
            "2:",
            "vsetvli t0, {rem}, e32, m4, ta, ma",
            "vlse32.v v8, ({ps}), {stride}",
            "vfsub.vf v8, v8, {mean}",
            "vfmul.vf v8, v8, {inv_std}",
            "vse32.v v8, ({po})",
            "mul t1, t0, {stride}",
            "add {ps}, {ps}, t1",
            "slli t2, t0, 2",
            "add {po}, {po}, t2",
            "sub {rem}, {rem}, t0",
            "bnez {rem}, 2b",
            ps = inout(reg) ps,
            po = inout(reg) po,
            rem = inout(reg) rem,
            stride = in(reg) stride_bytes,
            mean = in(freg) mean,
            inv_std = in(freg) inv_std,
            out("t0") _, out("t1") _, out("t2") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            options(nostack),
        );
    }
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[allow(unused_assignments)]
fn rvv_normalize_plane_inplace(plane: &mut [f32], mean: f32, inv_std: f32) {
    let mut rem = plane.len();
    if rem == 0 {
        return;
    }
    let mut po = plane.as_mut_ptr();

    // SAFETY: `po` points to `rem` contiguous writable `f32` elements.
    unsafe {
        core::arch::asm!(
            "2:",
            "vsetvli t0, {rem}, e32, m4, ta, ma",
            "vle32.v v8, ({po})",
            "vfsub.vf v8, v8, {mean}",
            "vfmul.vf v8, v8, {inv_std}",
            "vse32.v v8, ({po})",
            "slli t1, t0, 2",
            "add {po}, {po}, t1",
            "sub {rem}, {rem}, t0",
            "bnez {rem}, 2b",
            po = inout(reg) po,
            rem = inout(reg) rem,
            mean = in(freg) mean,
            inv_std = in(freg) inv_std,
            out("t0") _, out("t1") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            options(nostack),
        );
    }
}

/// Bilinearly resizes an interleaved `[src_h, src_w, 3]` RGB image into planar normalized
/// `[3, dst_h, dst_w]` `CHW` layout: `out_chw[c, y, x] = (resized[y, x, c] - mean[c]) / std[c]`.
///
/// Dispatches to `resize_rgb_bilinear_norm_f32_rvv` on `riscv64` targets and
/// [`resize_rgb_bilinear_norm_f32_scalar`] elsewhere or under Miri.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if any dimension is zero, dimensions overflow `usize`,
/// slice lengths mismatch, `std` contains non-positive or non-finite values, or `mean`/`src_hwc`
/// contain non-finite elements.
#[expect(
    clippy::too_many_arguments,
    reason = "vision resize+normalize takes explicit source/destination geometry and per-channel stats"
)]
pub fn resize_rgb_bilinear_norm_f32(
    out_chw: &mut [f32],
    src_hwc: &[f32],
    src_h: usize,
    src_w: usize,
    dst_h: usize,
    dst_w: usize,
    mean: [f32; 3],
    std: [f32; 3],
) -> Result<(), K3Error> {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        resize_rgb_bilinear_norm_f32_rvv(out_chw, src_hwc, src_h, src_w, dst_h, dst_w, mean, std)
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        resize_rgb_bilinear_norm_f32_scalar(out_chw, src_hwc, src_h, src_w, dst_h, dst_w, mean, std)
    }
}

/// Portable scalar reference implementation of [`resize_rgb_bilinear_norm_f32`].
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] on invalid dimensions, lengths, channel statistics, or
/// non-finite pixel values.
#[expect(
    clippy::too_many_arguments,
    reason = "vision resize+normalize takes explicit source/destination geometry and per-channel stats"
)]
pub fn resize_rgb_bilinear_norm_f32_scalar(
    out_chw: &mut [f32],
    src_hwc: &[f32],
    src_h: usize,
    src_w: usize,
    dst_h: usize,
    dst_w: usize,
    mean: [f32; 3],
    std: [f32; 3],
) -> Result<(), K3Error> {
    let dst_pixels =
        validate_resize_rgb_args(out_chw, src_hwc, src_h, src_w, dst_h, dst_w, mean, std)?;
    let inv_std = [1.0_f32 / std[0], 1.0_f32 / std[1], 1.0_f32 / std[2]];

    fill_bilinear_rgb_planes(out_chw, src_hwc, src_h, src_w, dst_h, dst_w);
    for c in 0..3 {
        let plane = &mut out_chw[c * dst_pixels..(c + 1) * dst_pixels];
        let m = mean[c];
        let is = inv_std[c];
        for slot in plane.iter_mut() {
            *slot = (*slot - m) * is;
        }
    }

    Ok(())
}

/// `RVV 1.0` (`vlse32.v` strided `HWC -> CHW` and `VLA` planar normalization) implementation of
/// [`resize_rgb_bilinear_norm_f32`].
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] on invalid dimensions, lengths, channel statistics, or
/// non-finite pixel values.
#[cfg(all(target_arch = "riscv64", not(miri)))]
#[expect(
    clippy::too_many_arguments,
    reason = "vision resize+normalize takes explicit source/destination geometry and per-channel stats"
)]
pub fn resize_rgb_bilinear_norm_f32_rvv(
    out_chw: &mut [f32],
    src_hwc: &[f32],
    src_h: usize,
    src_w: usize,
    dst_h: usize,
    dst_w: usize,
    mean: [f32; 3],
    std: [f32; 3],
) -> Result<(), K3Error> {
    let dst_pixels =
        validate_resize_rgb_args(out_chw, src_hwc, src_h, src_w, dst_h, dst_w, mean, std)?;
    let inv_std = [1.0_f32 / std[0], 1.0_f32 / std[1], 1.0_f32 / std[2]];

    if src_h == dst_h && src_w == dst_w {
        for c in 0..3 {
            let plane = &mut out_chw[c * dst_pixels..(c + 1) * dst_pixels];
            // SAFETY: `src_hwc` has `dst_pixels * 3` elements, so `&src_hwc[c..]` covers
            // `dst_pixels` elements at stride `3` (`12` bytes).
            let src_ptr = unsafe { src_hwc.as_ptr().add(c) };
            rvv_hwc_to_chw_norm_plane(plane, src_ptr, mean[c], inv_std[c]);
        }
        return Ok(());
    }

    fill_bilinear_rgb_planes(out_chw, src_hwc, src_h, src_w, dst_h, dst_w);
    for c in 0..3 {
        let plane = &mut out_chw[c * dst_pixels..(c + 1) * dst_pixels];
        rvv_normalize_plane_inplace(plane, mean[c], inv_std[c]);
    }

    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "patch embedding validates image, kernel, bias, and output tensor dimensions"
)]
fn validate_conv2d_patch_args(
    out_patches: &[f32],
    img_chw: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    patch_size: usize,
    kernel_oc_ic_kh_kw: &[f32],
    bias: Option<&[f32]>,
    hidden_dim: usize,
) -> Result<(usize, usize), K3Error> {
    let channels = check_positive_dim("patch_channels", channels)?;
    let height = check_positive_dim("patch_height", height)?;
    let width = check_positive_dim("patch_width", width)?;
    let patch_size = check_positive_dim("patch_size", patch_size)?;
    let hidden_dim = check_positive_dim("patch_hidden_dim", hidden_dim)?;

    if height % patch_size != 0 || width % patch_size != 0 {
        return Err(bounds_error(
            "conv2d_patch_embed",
            0,
            height,
            width,
            format!(
                "image dimensions ({height}x{width}) must be divisible by patch_size ({patch_size})"
            ),
        ));
    }

    let grid_h = height / patch_size;
    let grid_w = width / patch_size;
    let n_patches = checked_product("patch_grid", grid_h, grid_w, "grid_h * grid_w")?;

    let hw = checked_product("patch_img", height, width, "height * width")?;
    let img_elems = checked_product("patch_img", channels, hw, "channels * height * width")?;
    let patch_area = checked_product("patch_kernel", patch_size, patch_size, "patch_size^2")?;
    let patch_per_oc = checked_product(
        "patch_kernel",
        channels,
        patch_area,
        "channels * patch_size^2",
    )?;
    let kern_elems = checked_product(
        "patch_kernel",
        hidden_dim,
        patch_per_oc,
        "hidden_dim * channels * patch_size^2",
    )?;
    let out_elems = checked_product("patch_out", n_patches, hidden_dim, "n_patches * hidden_dim")?;

    check_exact_len("patch_img_chw", img_chw.len(), img_elems)?;
    check_exact_len("patch_kernel", kernel_oc_ic_kh_kw.len(), kern_elems)?;
    check_exact_len("patch_out", out_patches.len(), out_elems)?;
    if let Some(b) = bias {
        check_exact_len("patch_bias", b.len(), hidden_dim)?;
        check_all_finite("patch_bias", b)?;
    }

    check_all_finite("patch_img_chw", img_chw)?;
    check_all_finite("patch_kernel", kernel_oc_ic_kh_kw)?;

    Ok((grid_h, grid_w))
}

/// Computes non-overlapping `ViT` patch embedding via `Conv2D(kernel_size=patch_size, stride=patch_size)`
/// from `img_chw` (`[channels, height, width]`) and `kernel_oc_ic_kh_kw`
/// (`[hidden_dim, channels, patch_size, patch_size]`), producing `out_patches` (`[n_patches, hidden_dim]`).
///
/// Dispatches to `conv2d_patch_embed_f32_rvv` on `riscv64` targets and
/// [`conv2d_patch_embed_f32_scalar`] elsewhere or under Miri.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if any dimension is zero or overflows, `height` or `width`
/// is not divisible by `patch_size`, slice lengths mismatch, or any input/weight/bias is non-finite.
#[expect(
    clippy::too_many_arguments,
    reason = "Conv2D patch embedding accepts explicit CHW image, OIHW kernel, optional bias, and output slices"
)]
pub fn conv2d_patch_embed_f32(
    out_patches: &mut [f32],
    img_chw: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    patch_size: usize,
    kernel_oc_ic_kh_kw: &[f32],
    bias: Option<&[f32]>,
    hidden_dim: usize,
) -> Result<(), K3Error> {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        conv2d_patch_embed_f32_rvv(
            out_patches,
            img_chw,
            channels,
            height,
            width,
            patch_size,
            kernel_oc_ic_kh_kw,
            bias,
            hidden_dim,
        )
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        conv2d_patch_embed_f32_scalar(
            out_patches,
            img_chw,
            channels,
            height,
            width,
            patch_size,
            kernel_oc_ic_kh_kw,
            bias,
            hidden_dim,
        )
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "shared patch embedding loop parameterized by inner row dot product"
)]
fn conv2d_patch_embed_impl(
    out_patches: &mut [f32],
    img_chw: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    patch_size: usize,
    kernel_oc_ic_kh_kw: &[f32],
    bias: Option<&[f32]>,
    hidden_dim: usize,
    dot_fn: fn(&[f32], &[f32]) -> f32,
) -> Result<(), K3Error> {
    let (grid_h, grid_w) = validate_conv2d_patch_args(
        out_patches,
        img_chw,
        channels,
        height,
        width,
        patch_size,
        kernel_oc_ic_kh_kw,
        bias,
        hidden_dim,
    )?;

    let hw = height * width;
    let patch_area = patch_size * patch_size;
    let patch_per_oc = channels * patch_area;

    for py in 0..grid_h {
        for px in 0..grid_w {
            let patch_idx = py * grid_w + px;
            let out_row = &mut out_patches[patch_idx * hidden_dim..(patch_idx + 1) * hidden_dim];
            for (oc, out_slot) in out_row.iter_mut().enumerate() {
                let mut acc = bias.map_or(0.0_f32, |b| b[oc]);
                let kernel_oc_offset = oc * patch_per_oc;
                for ic in 0..channels {
                    let img_ic_base = ic * hw;
                    let kernel_ic_offset = kernel_oc_offset + ic * patch_area;
                    for ky in 0..patch_size {
                        let iy = py * patch_size + ky;
                        let ix = px * patch_size;
                        let img_row = &img_chw[img_ic_base + iy * width + ix
                            ..img_ic_base + iy * width + ix + patch_size];
                        let kern_row = &kernel_oc_ic_kh_kw[kernel_ic_offset + ky * patch_size
                            ..kernel_ic_offset + (ky + 1) * patch_size];
                        acc += dot_fn(img_row, kern_row);
                    }
                }
                *out_slot = acc;
            }
        }
    }

    Ok(())
}

/// Portable scalar reference implementation of [`conv2d_patch_embed_f32`].
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] on invalid dimensions, divisibility, lengths, or
/// non-finite values.
#[expect(
    clippy::too_many_arguments,
    reason = "Conv2D patch embedding accepts explicit CHW image, OIHW kernel, optional bias, and output slices"
)]
pub fn conv2d_patch_embed_f32_scalar(
    out_patches: &mut [f32],
    img_chw: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    patch_size: usize,
    kernel_oc_ic_kh_kw: &[f32],
    bias: Option<&[f32]>,
    hidden_dim: usize,
) -> Result<(), K3Error> {
    conv2d_patch_embed_impl(
        out_patches,
        img_chw,
        channels,
        height,
        width,
        patch_size,
        kernel_oc_ic_kh_kw,
        bias,
        hidden_dim,
        crate::dot_product_f32_scalar,
    )
}

/// `RVV 1.0` implementation of [`conv2d_patch_embed_f32`] using VLA `dot_product_f32_rvv`
/// reductions across contiguous patch rows.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] on invalid dimensions, divisibility, lengths, or
/// non-finite values.
#[cfg(all(target_arch = "riscv64", not(miri)))]
#[expect(
    clippy::too_many_arguments,
    reason = "Conv2D patch embedding accepts explicit CHW image, OIHW kernel, optional bias, and output slices"
)]
pub fn conv2d_patch_embed_f32_rvv(
    out_patches: &mut [f32],
    img_chw: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    patch_size: usize,
    kernel_oc_ic_kh_kw: &[f32],
    bias: Option<&[f32]>,
    hidden_dim: usize,
) -> Result<(), K3Error> {
    conv2d_patch_embed_impl(
        out_patches,
        img_chw,
        channels,
        height,
        width,
        patch_size,
        kernel_oc_ic_kh_kw,
        bias,
        hidden_dim,
        crate::dot_product_f32_rvv,
    )
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[allow(unused_assignments)]
#[expect(
    clippy::too_many_arguments,
    reason = "bilinear row blend passes four corner pointers and four interpolation weights"
)]
fn rvv_bilinear_blend_row(
    dst: &mut [f32],
    s00: &[f32],
    s01: &[f32],
    s10: &[f32],
    s11: &[f32],
    w00: f32,
    w01: f32,
    w10: f32,
    w11: f32,
) {
    let mut rem = dst.len();
    if rem == 0 {
        return;
    }
    let mut po = dst.as_mut_ptr();
    let mut p00 = s00.as_ptr();
    let mut p01 = s01.as_ptr();
    let mut p10 = s10.as_ptr();
    let mut p11 = s11.as_ptr();

    // SAFETY: All five pointers cover `rem` contiguous `f32` elements and `dst` does not alias inputs.
    unsafe {
        core::arch::asm!(
            "2:",
            "vsetvli t0, {rem}, e32, m4, ta, ma",
            "vle32.v v8, ({p00})",
            "vle32.v v12, ({p01})",
            "vle32.v v16, ({p10})",
            "vle32.v v20, ({p11})",
            "vfmul.vf v24, v8, {w00}",
            "vfmacc.vf v24, {w01}, v12",
            "vfmacc.vf v24, {w10}, v16",
            "vfmacc.vf v24, {w11}, v20",
            "vse32.v v24, ({po})",
            "slli t1, t0, 2",
            "add {p00}, {p00}, t1",
            "add {p01}, {p01}, t1",
            "add {p10}, {p10}, t1",
            "add {p11}, {p11}, t1",
            "add {po}, {po}, t1",
            "sub {rem}, {rem}, t0",
            "bnez {rem}, 2b",
            p00 = inout(reg) p00,
            p01 = inout(reg) p01,
            p10 = inout(reg) p10,
            p11 = inout(reg) p11,
            po = inout(reg) po,
            rem = inout(reg) rem,
            w00 = in(freg) w00,
            w01 = in(freg) w01,
            w10 = in(freg) w10,
            w11 = in(freg) w11,
            out("t0") _, out("t1") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            out("v24") _, out("v25") _, out("v26") _, out("v27") _,
            options(nostack),
        );
    }
}

/// Bilinearly resamples 2D spatial positional embeddings from `[base_grid_h, base_grid_w, hidden_dim]`
/// into `[dst_grid_h, dst_grid_w, hidden_dim]`.
///
/// When `base_grid_h == dst_grid_h` and `base_grid_w == dst_grid_w`, copies `base_pos`
/// bit-identically into `out_pos`.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if any dimension is zero or overflows `usize`,
/// slice lengths mismatch, or `base_pos` contains non-finite elements.
pub fn interpolate_2d_pos_embed_f32(
    out_pos: &mut [f32],
    base_pos: &[f32],
    base_grid_h: usize,
    base_grid_w: usize,
    dst_grid_h: usize,
    dst_grid_w: usize,
    hidden_dim: usize,
) -> Result<(), K3Error> {
    let base_grid_h = check_positive_dim("pos_base_grid_h", base_grid_h)?;
    let base_grid_w = check_positive_dim("pos_base_grid_w", base_grid_w)?;
    let dst_grid_h = check_positive_dim("pos_dst_grid_h", dst_grid_h)?;
    let dst_grid_w = check_positive_dim("pos_dst_grid_w", dst_grid_w)?;
    let hidden_dim = check_positive_dim("pos_hidden_dim", hidden_dim)?;

    let base_tokens = checked_product(
        "pos_base_grid",
        base_grid_h,
        base_grid_w,
        "base_grid_h * base_grid_w",
    )?;
    let base_elems = checked_product(
        "pos_base",
        base_tokens,
        hidden_dim,
        "base_tokens * hidden_dim",
    )?;
    let dst_tokens = checked_product(
        "pos_dst_grid",
        dst_grid_h,
        dst_grid_w,
        "dst_grid_h * dst_grid_w",
    )?;
    let dst_elems = checked_product("pos_dst", dst_tokens, hidden_dim, "dst_tokens * hidden_dim")?;

    check_exact_len("pos_base_pos", base_pos.len(), base_elems)?;
    check_exact_len("pos_out_pos", out_pos.len(), dst_elems)?;
    check_all_finite("pos_base_pos", base_pos)?;

    if base_grid_h == dst_grid_h && base_grid_w == dst_grid_w {
        out_pos.copy_from_slice(base_pos);
        return Ok(());
    }

    for dy in 0..dst_grid_h {
        let cy = bilinear_coord(dy, base_grid_h, dst_grid_h);
        for dx in 0..dst_grid_w {
            let cx = bilinear_coord(dx, base_grid_w, dst_grid_w);
            let w00 = (1.0_f32 - cy.w) * (1.0_f32 - cx.w);
            let w01 = (1.0_f32 - cy.w) * cx.w;
            let w10 = cy.w * (1.0_f32 - cx.w);
            let w11 = cy.w * cx.w;

            let off00 = (cy.i0 * base_grid_w + cx.i0) * hidden_dim;
            let off01 = (cy.i0 * base_grid_w + cx.i1) * hidden_dim;
            let off10 = (cy.i1 * base_grid_w + cx.i0) * hidden_dim;
            let off11 = (cy.i1 * base_grid_w + cx.i1) * hidden_dim;
            let dst_off = (dy * dst_grid_w + dx) * hidden_dim;

            let dst_slice = &mut out_pos[dst_off..dst_off + hidden_dim];
            let s00 = &base_pos[off00..off00 + hidden_dim];
            let s01 = &base_pos[off01..off01 + hidden_dim];
            let s10 = &base_pos[off10..off10 + hidden_dim];
            let s11 = &base_pos[off11..off11 + hidden_dim];

            #[cfg(all(target_arch = "riscv64", not(miri)))]
            {
                rvv_bilinear_blend_row(dst_slice, s00, s01, s10, s11, w00, w01, w10, w11);
            }
            #[cfg(any(not(target_arch = "riscv64"), miri))]
            {
                for d in 0..hidden_dim {
                    dst_slice[d] = w00 * s00[d] + w01 * s01[d] + w10 * s10[d] + w11 * s11[d];
                }
            }
        }
    }

    Ok(())
}

fn validate_pixel_shuffle_args(
    out_merged: &[f32],
    patches: &[f32],
    grid_h: usize,
    grid_w: usize,
    hidden_dim: usize,
) -> Result<(usize, usize), K3Error> {
    let grid_h = check_positive_dim("pixel_shuffle_grid_h", grid_h)?;
    let grid_w = check_positive_dim("pixel_shuffle_grid_w", grid_w)?;
    let hidden_dim = check_positive_dim("pixel_shuffle_hidden_dim", hidden_dim)?;

    if grid_h % 2 != 0 || grid_w % 2 != 0 {
        return Err(bounds_error(
            "pixel_shuffle_merge_2x2",
            0,
            grid_h,
            grid_w,
            format!("pixel_shuffle_merge_2x2 requires even grid_h={grid_h} and grid_w={grid_w}"),
        ));
    }

    let in_tokens = checked_product("pixel_shuffle_in", grid_h, grid_w, "grid_h * grid_w")?;
    let in_elems = checked_product(
        "pixel_shuffle_in",
        in_tokens,
        hidden_dim,
        "grid_h * grid_w * hidden_dim",
    )?;
    let out_h = grid_h / 2;
    let out_w = grid_w / 2;
    let out_tokens = checked_product("pixel_shuffle_out", out_h, out_w, "(grid_h/2) * (grid_w/2)")?;
    let merged_dim = checked_product("pixel_shuffle_out", 4, hidden_dim, "4 * hidden_dim")?;
    let out_elems = checked_product(
        "pixel_shuffle_out",
        out_tokens,
        merged_dim,
        "out_tokens * 4 * hidden_dim",
    )?;

    check_exact_len("pixel_shuffle_patches", patches.len(), in_elems)?;
    check_exact_len("pixel_shuffle_out_merged", out_merged.len(), out_elems)?;
    check_all_finite("pixel_shuffle_patches", patches)?;

    Ok((out_h, out_w))
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[allow(unused_assignments)]
fn rvv_copy_f32(dst: &mut [f32], src: &[f32]) {
    let mut rem = dst.len().min(src.len());
    if rem == 0 {
        return;
    }
    let mut ps = src.as_ptr();
    let mut pd = dst.as_mut_ptr();
    // SAFETY: `ps` and `pd` point to disjoint valid slices of length `rem`.
    unsafe {
        core::arch::asm!(
            "2:",
            "vsetvli t0, {rem}, e32, m4, ta, ma",
            "vle32.v v8, ({ps})",
            "vse32.v v8, ({pd})",
            "slli t1, t0, 2",
            "add {ps}, {ps}, t1",
            "add {pd}, {pd}, t1",
            "sub {rem}, {rem}, t0",
            "bnez {rem}, 2b",
            ps = inout(reg) ps,
            pd = inout(reg) pd,
            rem = inout(reg) rem,
            out("t0") _, out("t1") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            options(nostack),
        );
    }
}

/// Merges a `[grid_h, grid_w, hidden_dim]` patch token grid via `2x2` spatial pixel-shuffle into
/// `[grid_h / 2 * grid_w / 2, 4 * hidden_dim]`, ordering each `2x2` block row-major
/// `(top-left, top-right, bottom-left, bottom-right)`.
///
/// Dispatches to `pixel_shuffle_merge_2x2_f32_rvv` on `riscv64` targets and
/// [`pixel_shuffle_merge_2x2_f32_scalar`] elsewhere or under Miri.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if any dimension is zero or overflows `usize`, `grid_h`
/// or `grid_w` is odd, slice lengths mismatch, or any patch element is non-finite.
pub fn pixel_shuffle_merge_2x2_f32(
    out_merged: &mut [f32],
    patches: &[f32],
    grid_h: usize,
    grid_w: usize,
    hidden_dim: usize,
) -> Result<(), K3Error> {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        pixel_shuffle_merge_2x2_f32_rvv(out_merged, patches, grid_h, grid_w, hidden_dim)
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        pixel_shuffle_merge_2x2_f32_scalar(out_merged, patches, grid_h, grid_w, hidden_dim)
    }
}

/// Portable scalar reference implementation of [`pixel_shuffle_merge_2x2_f32`].
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] on invalid dimensions, odd grid sides, mismatched lengths,
/// or non-finite inputs.
pub fn pixel_shuffle_merge_2x2_f32_scalar(
    out_merged: &mut [f32],
    patches: &[f32],
    grid_h: usize,
    grid_w: usize,
    hidden_dim: usize,
) -> Result<(), K3Error> {
    let (out_h, out_w) =
        validate_pixel_shuffle_args(out_merged, patches, grid_h, grid_w, hidden_dim)?;
    let pair_dim = 2 * hidden_dim;
    let merged_dim = 4 * hidden_dim;

    for my in 0..out_h {
        for mx in 0..out_w {
            let dst_off = (my * out_w + mx) * merged_dim;
            let top_src_off = ((2 * my) * grid_w + (2 * mx)) * hidden_dim;
            let bot_src_off = ((2 * my + 1) * grid_w + (2 * mx)) * hidden_dim;
            out_merged[dst_off..dst_off + pair_dim]
                .copy_from_slice(&patches[top_src_off..top_src_off + pair_dim]);
            out_merged[dst_off + pair_dim..dst_off + merged_dim]
                .copy_from_slice(&patches[bot_src_off..bot_src_off + pair_dim]);
        }
    }

    Ok(())
}

/// `RVV 1.0` (`VLA` `vle32.v`/`vse32.v`) implementation of [`pixel_shuffle_merge_2x2_f32`].
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] on invalid dimensions, odd grid sides, mismatched lengths,
/// or non-finite inputs.
#[cfg(all(target_arch = "riscv64", not(miri)))]
pub fn pixel_shuffle_merge_2x2_f32_rvv(
    out_merged: &mut [f32],
    patches: &[f32],
    grid_h: usize,
    grid_w: usize,
    hidden_dim: usize,
) -> Result<(), K3Error> {
    let (out_h, out_w) =
        validate_pixel_shuffle_args(out_merged, patches, grid_h, grid_w, hidden_dim)?;
    let pair_dim = 2 * hidden_dim;
    let merged_dim = 4 * hidden_dim;

    for my in 0..out_h {
        for mx in 0..out_w {
            let dst_off = (my * out_w + mx) * merged_dim;
            let top_src_off = ((2 * my) * grid_w + (2 * mx)) * hidden_dim;
            let bot_src_off = ((2 * my + 1) * grid_w + (2 * mx)) * hidden_dim;
            rvv_copy_f32(
                &mut out_merged[dst_off..dst_off + pair_dim],
                &patches[top_src_off..top_src_off + pair_dim],
            );
            rvv_copy_f32(
                &mut out_merged[dst_off + pair_dim..dst_off + merged_dim],
                &patches[bot_src_off..bot_src_off + pair_dim],
            );
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resize_rgb_bilinear_norm_identity_constant_and_bounds() {
        let mean = [0.5_f32, 0.25, 0.125];
        let std = [0.25_f32, 0.5, 0.25];

        // Identity 2x3 HWC -> 3x2x3 CHW
        let src: Vec<f32> = (0..18_u8).map(|i| f32::from(i) * 0.05_f32).collect();
        let mut out_id = [0.0_f32; 18];
        resize_rgb_bilinear_norm_f32(&mut out_id, &src, 2, 3, 2, 3, mean, std).unwrap();
        for dy in 0..2 {
            for dx in 0..3 {
                for c in 0..3 {
                    let expected = (src[(dy * 3 + dx) * 3 + c] - mean[c]) / std[c];
                    assert!((out_id[c * 6 + dy * 3 + dx] - expected).abs() < 1e-5);
                }
            }
        }

        // Constant image preserves constant normalized values under upsample
        let const_img = [0.75_f32, 0.5, 0.375].repeat(4);
        let mut out_up = [0.0_f32; 3 * 5 * 7];
        resize_rgb_bilinear_norm_f32(&mut out_up, &const_img, 2, 2, 5, 7, mean, std).unwrap();
        for c in 0..3 {
            let expected = (const_img[c] - mean[c]) / std[c];
            for &v in &out_up[c * 35..(c + 1) * 35] {
                assert!((v - expected).abs() < 1e-5);
            }
        }

        // Validation errors preserve `out_up`
        let mut untouched = [42.0_f32; 12];
        assert!(matches!(
            resize_rgb_bilinear_norm_f32(
                &mut untouched,
                &const_img,
                2,
                2,
                2,
                2,
                mean,
                [0.5, 0.0, 0.5]
            ),
            Err(K3Error::TensorBoundsError { .. })
        ));
        assert!(matches!(
            resize_rgb_bilinear_norm_f32(&mut untouched, &const_img, 2, 2, 0, 2, mean, std),
            Err(K3Error::TensorBoundsError { .. })
        ));
        let mut nan_img = const_img;
        nan_img[5] = f32::NAN;
        assert!(matches!(
            resize_rgb_bilinear_norm_f32(&mut untouched, &nan_img, 2, 2, 2, 2, mean, std),
            Err(K3Error::TensorBoundsError { .. })
        ));
        assert_eq!(untouched, [42.0_f32; 12]);
    }

    #[test]
    fn test_conv2d_patch_embed_and_bounds() {
        // 1 channel, 4x4 image, patch_size=2 -> 2x2=4 patches, hidden_dim=2
        let img: Vec<f32> = (1..=16_u8).map(f32::from).collect();
        // oc=0: sum all 4 elements in patch; oc=1: [1, -1, 0, 2]
        let kernel = [
            1.0_f32, 1.0, 1.0, 1.0, // oc=0
            1.0, -1.0, 0.0, 2.0, // oc=1
        ];
        let bias = [0.5_f32, -1.0];
        let mut out = [0.0_f32; 4 * 2];

        conv2d_patch_embed_f32(&mut out, &img, 1, 4, 4, 2, &kernel, Some(&bias), 2).unwrap();
        // Top-left patch [[1, 2], [5, 6]]: sum=14+0.5=14.5; oc1=1-2+0+12-1=10.0
        assert!((out[0] - 14.5_f32).abs() < 1e-5);
        assert!((out[1] - 10.0_f32).abs() < 1e-5);
        // Bottom-right patch [[11, 12], [15, 16]]: sum=54+0.5=54.5; oc1=11-12+0+32-1=30.0
        assert!((out[6] - 54.5_f32).abs() < 1e-5);
        assert!((out[7] - 30.0_f32).abs() < 1e-5);

        // Without bias
        let mut out_nobias = [0.0_f32; 4 * 2];
        conv2d_patch_embed_f32(&mut out_nobias, &img, 1, 4, 4, 2, &kernel, None, 2).unwrap();
        assert!((out_nobias[0] - 14.0_f32).abs() < 1e-5);
        assert!((out_nobias[1] - 11.0_f32).abs() < 1e-5);

        // Non-divisible height/width fails closed
        assert!(matches!(
            conv2d_patch_embed_f32(&mut out, &img[..12], 1, 3, 4, 2, &kernel, None, 2),
            Err(K3Error::TensorBoundsError { .. })
        ));
    }

    #[test]
    fn test_interpolate_2d_pos_embed_identity_and_upsample() {
        let base = [
            1.0_f32, 10.0, 2.0, 20.0, // row 0
            3.0, 30.0, 4.0, 40.0, // row 1
        ];
        // Identity grid is bit-identical
        let mut out_id = [0.0_f32; 8];
        interpolate_2d_pos_embed_f32(&mut out_id, &base, 2, 2, 2, 2, 2).unwrap();
        for (a, b) in out_id.iter().zip(base.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }

        // Upsample 2x2 -> 4x4 stays bounded in [min, max] and preserves corners/symmetry
        let mut out_4x4 = [0.0_f32; 16 * 2];
        interpolate_2d_pos_embed_f32(&mut out_4x4, &base, 2, 2, 4, 4, 2).unwrap();
        for chunk in out_4x4.chunks_exact(2) {
            assert!((1.0_f32..=4.0_f32).contains(&chunk[0]));
            assert!((10.0_f32..=40.0_f32).contains(&chunk[1]));
            assert!((chunk[1] - chunk[0] * 10.0_f32).abs() < 1e-5);
        }
        // Top-left corner (dy=0, dx=0 -> sy=0, sx=0) matches [1.0, 10.0]
        assert!((out_4x4[0] - 1.0_f32).abs() < 1e-5);
        // Bottom-right corner (dy=3, dx=3 -> sy=1, sx=1) matches [4.0, 40.0]
        assert!((out_4x4[30] - 4.0_f32).abs() < 1e-5);
    }

    #[test]
    fn test_pixel_shuffle_merge_2x2_and_odd_rejection() {
        // 2x4 grid of dim=2 -> 1x2 merged grid of dim=8
        // Grid layout (h=2, w=4):
        // (0,0)=[1,2] (0,1)=[3,4] | (0,2)=[5,6] (0,3)=[7,8]
        // (1,0)=[9,10] (1,1)=[11,12] | (1,2)=[13,14] (1,3)=[15,16]
        let patches: Vec<f32> = (1..=16_u8).map(f32::from).collect();
        let mut merged = [0.0_f32; 16];
        pixel_shuffle_merge_2x2_f32(&mut merged, &patches, 2, 4, 2).unwrap();

        assert_eq!(
            merged,
            [
                1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0, // merged token 0
                5.0, 6.0, 7.0, 8.0, 13.0, 14.0, 15.0, 16.0, // merged token 1
            ]
        );

        // Odd grid_h or grid_w rejected
        assert!(matches!(
            pixel_shuffle_merge_2x2_f32(&mut merged, &patches[..12], 3, 2, 2),
            Err(K3Error::TensorBoundsError { .. })
        ));
        assert!(matches!(
            pixel_shuffle_merge_2x2_f32(&mut merged, &patches[..12], 2, 3, 2),
            Err(K3Error::TensorBoundsError { .. })
        ));
    }
}
