//! Whisper audio DSP preprocessing (`hann_window_f32`, `rfft_power_spectrum_512_f32`,
//! `log_mel_spectrogram_whisper_f32`) and `Conv1D(k=3, padding=1) + GELU` encoder stem kernels
//! with `RVV 1.0` and portable scalar fallbacks.

use core::f64::consts::PI;
use jero_core::K3Error;

use crate::machine::{
    bounds_error, check_all_finite, check_exact_len, check_positive_dim, checked_product,
};
use crate::swiglu::gelu_scalar_f32;

/// Whisper STFT frame / Hann window length in samples (`25 ms` at `16 kHz`).
pub const WHISPER_N_FFT: usize = 400;

/// Radix-2 FFT transform length used to evaluate zero-padded Whisper STFT frames.
pub const WHISPER_FFT_POINTS: usize = 512;

/// Whisper hop length in samples (`10 ms` at `16 kHz`).
pub const WHISPER_HOP_LENGTH: usize = 160;

/// Standard Whisper STFT unique real-FFT frequency bin count (`400 / 2 + 1`).
pub const WHISPER_N_BINS_201: usize = 201;

/// Full 512-point real-FFT unique frequency bin count (`512 / 2 + 1`).
pub const WHISPER_N_BINS_257: usize = 257;

/// Evaluates the periodic Hann window `w[n] = 0.5 * (1 - cos(2 * pi * n / n_fft))` for
/// `n in 0..n_fft` into `out`.
///
/// Dispatches to `hann_window_f32_rvv` on `riscv64` targets and [`hann_window_f32_scalar`]
/// elsewhere or under Miri.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `n_fft == 0` or `out.len() != n_fft`.
pub fn hann_window_f32(out: &mut [f32], n_fft: usize) -> Result<(), K3Error> {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        hann_window_f32_rvv(out, n_fft)
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        hann_window_f32_scalar(out, n_fft)
    }
}

/// Portable scalar reference implementation of [`hann_window_f32`].
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `n_fft == 0` or `out.len() != n_fft`.
pub fn hann_window_f32_scalar(out: &mut [f32], n_fft: usize) -> Result<(), K3Error> {
    let n_fft = check_positive_dim("hann_n_fft", n_fft)?;
    check_exact_len("hann_out", out.len(), n_fft)?;

    #[expect(
        clippy::cast_precision_loss,
        reason = "FFT length fits losslessly within f64 exact integer range"
    )]
    let denom = n_fft as f64;
    for (n, slot) in out.iter_mut().enumerate() {
        #[expect(
            clippy::cast_precision_loss,
            reason = "sample index fits losslessly within f64 exact integer range"
        )]
        let theta = 2.0_f64 * PI * (n as f64) / denom;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "Hann coefficient lies strictly in [0.0, 1.0]"
        )]
        {
            *slot = (0.5_f64 * (1.0_f64 - theta.cos())) as f32;
        }
    }

    Ok(())
}

/// `RVV 1.0` (`VLA` `vfrsub.vf` + `vfmul.vf`) implementation of [`hann_window_f32`].
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `n_fft == 0` or `out.len() != n_fft`.
#[cfg(all(target_arch = "riscv64", not(miri)))]
#[allow(unused_assignments)]
pub fn hann_window_f32_rvv(out: &mut [f32], n_fft: usize) -> Result<(), K3Error> {
    let n_fft = check_positive_dim("hann_n_fft", n_fft)?;
    check_exact_len("hann_out", out.len(), n_fft)?;

    #[expect(
        clippy::cast_precision_loss,
        reason = "FFT length fits losslessly within f64 exact integer range"
    )]
    let denom = n_fft as f64;
    for (n, slot) in out.iter_mut().enumerate() {
        #[expect(
            clippy::cast_precision_loss,
            reason = "sample index fits losslessly within f64 exact integer range"
        )]
        let theta = 2.0_f64 * PI * (n as f64) / denom;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "cosine lies strictly in [-1.0, 1.0]"
        )]
        {
            *slot = theta.cos() as f32;
        }
    }

    let mut po = out.as_mut_ptr();
    let mut rem = n_fft;
    // SAFETY: `po` covers `rem` contiguous `f32` elements; `vfrsub.vf` + `vfmul.vf` evaluates
    // `0.5 * (1.0 - cos)` in VLA `e32, m4` strips.
    unsafe {
        core::arch::asm!(
            "2:",
            "vsetvli t0, {rem}, e32, m4, ta, ma",
            "vle32.v v8, ({po})",
            "vfrsub.vf v8, v8, {one}",
            "vfmul.vf v8, v8, {half}",
            "vse32.v v8, ({po})",
            "slli t1, t0, 2",
            "add {po}, {po}, t1",
            "sub {rem}, {rem}, t0",
            "bnez {rem}, 2b",
            po = inout(reg) po,
            rem = inout(reg) rem,
            one = in(freg) 1.0_f32,
            half = in(freg) 0.5_f32,
            out("t0") _, out("t1") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            options(nostack),
        );
    }

    Ok(())
}

fn fft_512_power_bins(
    re: &mut [f32; WHISPER_FFT_POINTS],
    im: &mut [f32; WHISPER_FFT_POINTS],
    out_bins: &mut [f32],
) {
    // 9-bit bit-reversal permutation over 512 elements.
    for i in 0..WHISPER_FFT_POINTS {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "i < 512 fits in u16; reversing 16 bits and shifting by 7 yields 9-bit reversal"
        )]
        let j = usize::from((i as u16).reverse_bits() >> 7);
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    // 9 radix-2 Cooley-Tukey decimation-in-time butterfly stages.
    for stage in 1..=9_u32 {
        let m = 1_usize << stage;
        let half = m >> 1;
        #[expect(
            clippy::cast_precision_loss,
            reason = "radix-2 stage width <= 512 fits losslessly in f64"
        )]
        let m_f64 = m as f64;
        for j in 0..half {
            #[expect(
                clippy::cast_precision_loss,
                reason = "butterfly index < 256 fits losslessly in f64"
            )]
            let angle = -2.0_f64 * PI * (j as f64) / m_f64;
            #[expect(
                clippy::cast_possible_truncation,
                reason = "unit-circle twiddle factor fits in f32"
            )]
            let w_re = angle.cos() as f32;
            #[expect(
                clippy::cast_possible_truncation,
                reason = "unit-circle twiddle factor fits in f32"
            )]
            let w_im = angle.sin() as f32;

            let mut k = 0_usize;
            while k < WHISPER_FFT_POINTS {
                let i1 = k + j;
                let i2 = i1 + half;
                let r2 = re[i2];
                let im2 = im[i2];
                let t_re = w_re * r2 - w_im * im2;
                let t_im = w_re * im2 + w_im * r2;
                let u_re = re[i1];
                let u_im = im[i1];
                re[i1] = u_re + t_re;
                im[i1] = u_im + t_im;
                re[i2] = u_re - t_re;
                im[i2] = u_im - t_im;
                k += m;
            }
        }
    }

    for (k, bin_slot) in out_bins.iter_mut().enumerate() {
        let r = re[k];
        let x = im[k];
        *bin_slot = r * r + x * x;
    }
}

/// Computes an authentic radix-2 Cooley-Tukey 512-point real `FFT` on a zero-padded
/// 400-sample windowed frame, writing the first `201` power spectrum bins (`|X[k]|^2`) into
/// `out_bins_201` without heap allocation.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `windowed_frame_400.len() != 400`,
/// `out_bins_201.len() != 201`, or any input sample is non-finite.
pub fn rfft_power_spectrum_512_f32(
    out_bins_201: &mut [f32],
    windowed_frame_400: &[f32],
) -> Result<(), K3Error> {
    check_exact_len(
        "rfft_windowed_frame_400",
        windowed_frame_400.len(),
        WHISPER_N_FFT,
    )?;
    check_exact_len("rfft_out_bins_201", out_bins_201.len(), WHISPER_N_BINS_201)?;
    check_all_finite("rfft_windowed_frame_400", windowed_frame_400)?;

    let mut re = [0.0_f32; WHISPER_FFT_POINTS];
    let mut im = [0.0_f32; WHISPER_FFT_POINTS];
    re[..WHISPER_N_FFT].copy_from_slice(windowed_frame_400);
    fft_512_power_bins(&mut re, &mut im, out_bins_201);

    Ok(())
}

fn validate_log_mel_args(
    out_mel: &[f32],
    pcm_16khz: &[f32],
    mel_filters: &[f32],
    n_mels: usize,
    n_frames: usize,
) -> Result<usize, K3Error> {
    if n_mels != 80 && n_mels != 128 {
        return Err(bounds_error(
            "whisper_n_mels",
            0,
            n_mels,
            128,
            format!("Whisper log-mel expects n_mels in {{80, 128}}, got {n_mels}"),
        ));
    }
    let n_frames = check_positive_dim("whisper_n_frames", n_frames)?;
    let out_elems = checked_product("whisper_out_mel", n_mels, n_frames, "n_mels * n_frames")?;
    check_exact_len("whisper_out_mel", out_mel.len(), out_elems)?;

    let min_pcm = checked_product(
        "whisper_pcm_16khz",
        n_frames,
        WHISPER_HOP_LENGTH,
        "n_frames * 160",
    )?;
    if pcm_16khz.len() < min_pcm {
        return Err(bounds_error(
            "whisper_pcm_16khz",
            0,
            min_pcm.saturating_mul(size_of::<f32>()),
            size_of_val(pcm_16khz),
            format!(
                "pcm_16khz length {} is shorter than required {} samples for {n_frames} frames",
                pcm_16khz.len(),
                min_pcm
            ),
        ));
    }

    let expected_201 = n_mels * WHISPER_N_BINS_201;
    let expected_257 = n_mels * WHISPER_N_BINS_257;
    let n_bins = if mel_filters.len() == expected_201 {
        WHISPER_N_BINS_201
    } else if mel_filters.len() == expected_257 {
        WHISPER_N_BINS_257
    } else {
        return Err(bounds_error(
            "whisper_mel_filters",
            0,
            expected_201.saturating_mul(size_of::<f32>()),
            size_of_val(mel_filters),
            format!(
                "mel_filters length {} must equal {expected_201} ([{n_mels}, 201]) or {expected_257} ([{n_mels}, 257])",
                mel_filters.len()
            ),
        ));
    };

    check_all_finite("whisper_pcm_16khz", pcm_16khz)?;
    check_all_finite("whisper_mel_filters", mel_filters)?;

    Ok(n_bins)
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[allow(unused_assignments)]
fn rvv_mul_slices_f32(dst: &mut [f32], a: &[f32], b: &[f32]) {
    let mut rem = dst.len().min(a.len()).min(b.len());
    if rem == 0 {
        return;
    }
    let mut pd = dst.as_mut_ptr();
    let mut pa = a.as_ptr();
    let mut pb = b.as_ptr();
    // SAFETY: `pd`, `pa`, and `pb` point to valid slices of length `rem` (`dst` does not alias inputs).
    unsafe {
        core::arch::asm!(
            "2:",
            "vsetvli t0, {rem}, e32, m4, ta, ma",
            "vle32.v v8, ({pa})",
            "vle32.v v12, ({pb})",
            "vfmul.vv v8, v8, v12",
            "vse32.v v8, ({pd})",
            "slli t1, t0, 2",
            "add {pa}, {pa}, t1",
            "add {pb}, {pb}, t1",
            "add {pd}, {pd}, t1",
            "sub {rem}, {rem}, t0",
            "bnez {rem}, 2b",
            pa = inout(reg) pa,
            pb = inout(reg) pb,
            pd = inout(reg) pd,
            rem = inout(reg) rem,
            out("t0") _, out("t1") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            options(nostack),
        );
    }
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[allow(unused_assignments)]
fn rvv_max_f32(data: &[f32]) -> f32 {
    if data.is_empty() {
        return f32::NEG_INFINITY;
    }
    let mut p = data.as_ptr();
    let mut rem = data.len();
    let max_val: f32;
    // SAFETY: `p` covers `rem` readable `f32` elements; `vfredmax.vs` reduces across VLA `e32, m1`.
    unsafe {
        core::arch::asm!(
            "vsetivli zero, 1, e32, m1, ta, ma",
            "vfmv.s.f v0, {ninf}",
            "2:",
            "vsetvli t0, {rem}, e32, m1, ta, ma",
            "vle32.v v8, ({p})",
            "vfredmax.vs v0, v8, v0",
            "slli t1, t0, 2",
            "add {p}, {p}, t1",
            "sub {rem}, {rem}, t0",
            "bnez {rem}, 2b",
            "vsetivli zero, 1, e32, m1, ta, ma",
            "vfmv.f.s {maxv}, v0",
            p = inout(reg) p,
            rem = inout(reg) rem,
            ninf = in(freg) f32::NEG_INFINITY,
            maxv = out(freg) max_val,
            out("t0") _, out("t1") _,
            out("v0") _, out("v8") _,
            options(nostack, readonly),
        );
    }
    max_val
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[allow(unused_assignments)]
fn rvv_whisper_clamp_scale_inplace(out_mel: &mut [f32], min_floor: f32) {
    let mut rem = out_mel.len();
    if rem == 0 {
        return;
    }
    let mut po = out_mel.as_mut_ptr();
    // SAFETY: `po` covers `rem` contiguous writable `f32` elements.
    unsafe {
        core::arch::asm!(
            "2:",
            "vsetvli t0, {rem}, e32, m4, ta, ma",
            "vle32.v v8, ({po})",
            "vfmax.vf v8, v8, {floor}",
            "vfadd.vf v8, v8, {four}",
            "vfmul.vf v8, v8, {quarter}",
            "vse32.v v8, ({po})",
            "slli t1, t0, 2",
            "add {po}, {po}, t1",
            "sub {rem}, {rem}, t0",
            "bnez {rem}, 2b",
            po = inout(reg) po,
            rem = inout(reg) rem,
            floor = in(freg) min_floor,
            four = in(freg) 4.0_f32,
            quarter = in(freg) 0.25_f32,
            out("t0") _, out("t1") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            options(nostack),
        );
    }
}

/// Computes Whisper log-mel spectrogram `[n_mels, n_frames]` from 16 kHz PCM samples using
/// `n_fft = 400`, `hop = 160`, 512-point radix-2 FFT power bins, `log10(max(energy, 1e-10))`,
/// dynamic-range clamping (`>= max_val - 8.0`), and `(x + 4.0) / 4.0` normalization.
///
/// Supports `n_mels` of `80` or `128` and filterbanks shaped `[n_mels, 201]` or `[n_mels, 257]`.
///
/// Dispatches to `log_mel_spectrogram_whisper_f32_rvv` on `riscv64` targets and
/// [`log_mel_spectrogram_whisper_f32_scalar`] elsewhere or under Miri.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `n_mels` is not `80` or `128`, `n_frames == 0`,
/// dimensions overflow `usize`, slices mismatch, `pcm_16khz.len() < n_frames * 160`, or inputs
/// contain non-finite values.
pub fn log_mel_spectrogram_whisper_f32(
    out_mel: &mut [f32],
    pcm_16khz: &[f32],
    mel_filters: &[f32],
    n_mels: usize,
    n_frames: usize,
) -> Result<(), K3Error> {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        log_mel_spectrogram_whisper_f32_rvv(out_mel, pcm_16khz, mel_filters, n_mels, n_frames)
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        log_mel_spectrogram_whisper_f32_scalar(out_mel, pcm_16khz, mel_filters, n_mels, n_frames)
    }
}

/// Portable scalar reference implementation of [`log_mel_spectrogram_whisper_f32`].
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] on invalid `n_mels`, zero/overflow dimensions, slice
/// mismatches, or non-finite inputs.
pub fn log_mel_spectrogram_whisper_f32_scalar(
    out_mel: &mut [f32],
    pcm_16khz: &[f32],
    mel_filters: &[f32],
    n_mels: usize,
    n_frames: usize,
) -> Result<(), K3Error> {
    let n_bins = validate_log_mel_args(out_mel, pcm_16khz, mel_filters, n_mels, n_frames)?;

    let mut hann = [0.0_f32; WHISPER_N_FFT];
    hann_window_f32_scalar(&mut hann, WHISPER_N_FFT)?;

    let mut raw_frame = [0.0_f32; WHISPER_N_FFT];
    let mut windowed = [0.0_f32; WHISPER_N_FFT];
    let mut re = [0.0_f32; WHISPER_FFT_POINTS];
    let mut im = [0.0_f32; WHISPER_FFT_POINTS];
    let mut power = [0.0_f32; WHISPER_N_BINS_257];

    let mut max_val = f32::NEG_INFINITY;
    for frame_idx in 0..n_frames {
        let start = frame_idx * WHISPER_HOP_LENGTH;
        let avail = (pcm_16khz.len() - start).min(WHISPER_N_FFT);
        raw_frame[..avail].copy_from_slice(&pcm_16khz[start..start + avail]);
        raw_frame[avail..].fill(0.0_f32);
        for i in 0..WHISPER_N_FFT {
            windowed[i] = raw_frame[i] * hann[i];
        }

        re[..WHISPER_N_FFT].copy_from_slice(&windowed);
        re[WHISPER_N_FFT..].fill(0.0_f32);
        im.fill(0.0_f32);
        fft_512_power_bins(&mut re, &mut im, &mut power[..n_bins]);

        for m in 0..n_mels {
            let filter_row = &mel_filters[m * n_bins..(m + 1) * n_bins];
            let energy = crate::dot_product_f32_scalar(filter_row, &power[..n_bins]);
            let log_e = energy.max(1e-10_f32).log10();
            out_mel[m * n_frames + frame_idx] = log_e;
            if log_e > max_val {
                max_val = log_e;
            }
        }
    }

    let min_floor = max_val - 8.0_f32;
    for slot in out_mel.iter_mut() {
        *slot = (slot.max(min_floor) + 4.0_f32) * 0.25_f32;
    }

    Ok(())
}

/// `RVV 1.0` (`VLA` window multiplication, filterbank dot product, max reduction, and clamp/scale)
/// implementation of [`log_mel_spectrogram_whisper_f32`].
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] on invalid `n_mels`, zero/overflow dimensions, slice
/// mismatches, or non-finite inputs.
#[cfg(all(target_arch = "riscv64", not(miri)))]
pub fn log_mel_spectrogram_whisper_f32_rvv(
    out_mel: &mut [f32],
    pcm_16khz: &[f32],
    mel_filters: &[f32],
    n_mels: usize,
    n_frames: usize,
) -> Result<(), K3Error> {
    let n_bins = validate_log_mel_args(out_mel, pcm_16khz, mel_filters, n_mels, n_frames)?;

    let mut hann = [0.0_f32; WHISPER_N_FFT];
    hann_window_f32_rvv(&mut hann, WHISPER_N_FFT)?;

    let mut raw_frame = [0.0_f32; WHISPER_N_FFT];
    let mut windowed = [0.0_f32; WHISPER_N_FFT];
    let mut re = [0.0_f32; WHISPER_FFT_POINTS];
    let mut im = [0.0_f32; WHISPER_FFT_POINTS];
    let mut power = [0.0_f32; WHISPER_N_BINS_257];

    for frame_idx in 0..n_frames {
        let start = frame_idx * WHISPER_HOP_LENGTH;
        let avail = (pcm_16khz.len() - start).min(WHISPER_N_FFT);
        raw_frame[..avail].copy_from_slice(&pcm_16khz[start..start + avail]);
        raw_frame[avail..].fill(0.0_f32);
        rvv_mul_slices_f32(&mut windowed, &raw_frame, &hann);

        re[..WHISPER_N_FFT].copy_from_slice(&windowed);
        re[WHISPER_N_FFT..].fill(0.0_f32);
        im.fill(0.0_f32);
        fft_512_power_bins(&mut re, &mut im, &mut power[..n_bins]);

        for m in 0..n_mels {
            let filter_row = &mel_filters[m * n_bins..(m + 1) * n_bins];
            let energy = crate::dot_product_f32_rvv(filter_row, &power[..n_bins]);
            out_mel[m * n_frames + frame_idx] = energy.max(1e-10_f32).log10();
        }
    }

    let max_val = rvv_max_f32(out_mel);
    let min_floor = max_val - 8.0_f32;
    rvv_whisper_clamp_scale_inplace(out_mel, min_floor);

    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "Conv1D k=3 validation accepts input/weight/bias/output dimensions and stride"
)]
fn validate_conv1d_k3_args(
    out: &[f32],
    input: &[f32],
    in_channels: usize,
    in_len: usize,
    weight: &[f32],
    bias: &[f32],
    out_channels: usize,
    stride: usize,
) -> Result<usize, K3Error> {
    let in_channels = check_positive_dim("conv1d_in_channels", in_channels)?;
    let in_len = check_positive_dim("conv1d_in_len", in_len)?;
    let out_channels = check_positive_dim("conv1d_out_channels", out_channels)?;
    let stride = check_positive_dim("conv1d_stride", stride)?;

    let out_len = (in_len - 1) / stride + 1;
    let input_elems = checked_product("conv1d_input", in_channels, in_len, "in_channels * in_len")?;
    let oc_ic = checked_product(
        "conv1d_weight",
        out_channels,
        in_channels,
        "out_channels * in_channels",
    )?;
    let weight_elems =
        checked_product("conv1d_weight", oc_ic, 3, "out_channels * in_channels * 3")?;
    let out_elems = checked_product(
        "conv1d_out",
        out_channels,
        out_len,
        "out_channels * out_len",
    )?;

    check_exact_len("conv1d_input", input.len(), input_elems)?;
    check_exact_len("conv1d_weight", weight.len(), weight_elems)?;
    check_exact_len("conv1d_bias", bias.len(), out_channels)?;
    check_exact_len("conv1d_out", out.len(), out_elems)?;

    check_all_finite("conv1d_input", input)?;
    check_all_finite("conv1d_weight", weight)?;
    check_all_finite("conv1d_bias", bias)?;

    Ok(out_len)
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[allow(unused_assignments)]
fn rvv_strided_axpy_f32(dst: &mut [f32], src: &[f32], src_stride_elems: usize, weight: f32) {
    let mut rem = dst.len();
    if rem == 0 {
        return;
    }
    if src_stride_elems == 1 {
        crate::attention::axpy_f32(dst, &src[..rem], weight);
        return;
    }
    let mut pd = dst.as_mut_ptr();
    let mut ps = src.as_ptr();
    #[expect(
        clippy::cast_possible_wrap,
        reason = "Conv1D stride in bytes fits well within isize"
    )]
    let stride_bytes = (src_stride_elems * size_of::<f32>()) as isize;

    // SAFETY: Caller guarantees `src` contains at least `(rem - 1) * src_stride_elems + 1` elements
    // and `dst` is an exclusively borrowed slice of length `rem`.
    unsafe {
        core::arch::asm!(
            "2:",
            "vsetvli t0, {rem}, e32, m4, ta, ma",
            "vle32.v v8, ({pd})",
            "vlse32.v v16, ({ps}), {stride}",
            "vfmacc.vf v8, {w}, v16",
            "vse32.v v8, ({pd})",
            "slli t1, t0, 2",
            "add {pd}, {pd}, t1",
            "mul t2, t0, {stride}",
            "add {ps}, {ps}, t2",
            "sub {rem}, {rem}, t0",
            "bnez {rem}, 2b",
            pd = inout(reg) pd,
            ps = inout(reg) ps,
            rem = inout(reg) rem,
            stride = in(reg) stride_bytes,
            w = in(freg) weight,
            out("t0") _, out("t1") _, out("t2") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            options(nostack),
        );
    }
}

/// Evaluates `Conv1D(kernel_size=3, padding=1, stride=stride) + GELU_tanh` over `[in_channels, in_len]`
/// input with `[out_channels, in_channels, 3]` weights and `[out_channels]` bias into
/// `[out_channels, (in_len - 1) / stride + 1]`.
///
/// Dispatches to `conv1d_k3_gelu_f32_rvv` on `riscv64` targets and
/// [`conv1d_k3_gelu_f32_scalar`] elsewhere or under Miri.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if any dimension or `stride` is zero, dimensions overflow
/// `usize`, slice lengths mismatch, or any input/weight/bias element is non-finite.
#[expect(
    clippy::too_many_arguments,
    reason = "Conv1D k=3 + GELU kernel accepts input, geometry, weights, bias, stride, and output slice"
)]
pub fn conv1d_k3_gelu_f32(
    out: &mut [f32],
    input: &[f32],
    in_channels: usize,
    in_len: usize,
    weight: &[f32],
    bias: &[f32],
    out_channels: usize,
    stride: usize,
) -> Result<(), K3Error> {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        conv1d_k3_gelu_f32_rvv(
            out,
            input,
            in_channels,
            in_len,
            weight,
            bias,
            out_channels,
            stride,
        )
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        conv1d_k3_gelu_f32_scalar(
            out,
            input,
            in_channels,
            in_len,
            weight,
            bias,
            out_channels,
            stride,
        )
    }
}

/// Portable scalar reference implementation of [`conv1d_k3_gelu_f32`].
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] on invalid geometry/stride, mismatched slice lengths, or
/// non-finite values.
#[expect(
    clippy::too_many_arguments,
    reason = "Conv1D k=3 + GELU kernel accepts input, geometry, weights, bias, stride, and output slice"
)]
pub fn conv1d_k3_gelu_f32_scalar(
    out: &mut [f32],
    input: &[f32],
    in_channels: usize,
    in_len: usize,
    weight: &[f32],
    bias: &[f32],
    out_channels: usize,
    stride: usize,
) -> Result<(), K3Error> {
    let out_len = validate_conv1d_k3_args(
        out,
        input,
        in_channels,
        in_len,
        weight,
        bias,
        out_channels,
        stride,
    )?;

    for oc in 0..out_channels {
        let out_row = &mut out[oc * out_len..(oc + 1) * out_len];
        let b = bias[oc];
        for (t_out, out_slot) in out_row.iter_mut().enumerate() {
            let center = t_out * stride;
            let mut acc = b;
            for ic in 0..in_channels {
                let in_row = &input[ic * in_len..(ic + 1) * in_len];
                let w_base = (oc * in_channels + ic) * 3;
                let left = if center > 0 {
                    in_row[center - 1]
                } else {
                    0.0_f32
                };
                let mid = in_row[center];
                let right = if center + 1 < in_len {
                    in_row[center + 1]
                } else {
                    0.0_f32
                };
                acc +=
                    left * weight[w_base] + mid * weight[w_base + 1] + right * weight[w_base + 2];
            }
            *out_slot = gelu_scalar_f32(acc);
        }
    }

    Ok(())
}

/// `RVV 1.0` (`VLA` unit/strided `vfmacc.vf` taps + `gelu_inplace_f32`) implementation of
/// [`conv1d_k3_gelu_f32`].
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] on invalid geometry/stride, mismatched slice lengths, or
/// non-finite values.
#[cfg(all(target_arch = "riscv64", not(miri)))]
#[expect(
    clippy::too_many_arguments,
    reason = "Conv1D k=3 + GELU kernel accepts input, geometry, weights, bias, stride, and output slice"
)]
pub fn conv1d_k3_gelu_f32_rvv(
    out: &mut [f32],
    input: &[f32],
    in_channels: usize,
    in_len: usize,
    weight: &[f32],
    bias: &[f32],
    out_channels: usize,
    stride: usize,
) -> Result<(), K3Error> {
    let out_len = validate_conv1d_k3_args(
        out,
        input,
        in_channels,
        in_len,
        weight,
        bias,
        out_channels,
        stride,
    )?;

    let right_count = if in_len < 2 {
        0
    } else {
        ((in_len - 2) / stride + 1).min(out_len)
    };

    for oc in 0..out_channels {
        let out_row = &mut out[oc * out_len..(oc + 1) * out_len];
        out_row.fill(bias[oc]);

        for ic in 0..in_channels {
            let in_row = &input[ic * in_len..(ic + 1) * in_len];
            let w_base = (oc * in_channels + ic) * 3;
            let w0 = weight[w_base];
            let w1 = weight[w_base + 1];
            let w2 = weight[w_base + 2];

            // Center tap: out[t_out] += w1 * in[t_out * stride]
            rvv_strided_axpy_f32(out_row, in_row, stride, w1);
            // Left tap (t_out >= 1): out[t_out] += w0 * in[t_out * stride - 1]
            if out_len > 1 {
                rvv_strided_axpy_f32(&mut out_row[1..], &in_row[stride - 1..], stride, w0);
            }
            // Right tap (t_out * stride + 1 < in_len): out[t_out] += w2 * in[t_out * stride + 1]
            if right_count > 0 {
                rvv_strided_axpy_f32(&mut out_row[..right_count], &in_row[1..], stride, w2);
            }
        }
    }

    crate::machine::gelu_inplace_f32(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hann_window_periodic_properties() {
        let mut w = [0.0_f32; WHISPER_N_FFT];
        hann_window_f32(&mut w, WHISPER_N_FFT).unwrap();

        assert!((w[0] - 0.0_f32).abs() < 1e-6);
        assert!((w[WHISPER_N_FFT / 2] - 1.0_f32).abs() < 1e-6);
        for n in 1..WHISPER_N_FFT / 2 {
            assert!((w[n] - w[WHISPER_N_FFT - n]).abs() < 1e-6);
            assert!((0.0_f32..=1.0_f32).contains(&w[n]));
        }

        // n_fft == 1 and validation failures
        let mut w1 = [9.0_f32; 1];
        hann_window_f32(&mut w1, 1).unwrap();
        assert_eq!(w1[0], 0.0_f32);
        assert!(matches!(
            hann_window_f32(&mut [], 0),
            Err(K3Error::TensorBoundsError { .. })
        ));
        assert!(matches!(
            hann_window_f32(&mut w1, 4),
            Err(K3Error::TensorBoundsError { .. })
        ));
    }

    #[test]
    fn test_rfft_power_spectrum_512_pure_cosine_and_dc() {
        // Pure DC frame of ones: bin 0 = 400^2 = 160_000, positive power everywhere
        let dc_frame = [1.0_f32; WHISPER_N_FFT];
        let mut bins_dc = [0.0_f32; WHISPER_N_BINS_201];
        rfft_power_spectrum_512_f32(&mut bins_dc, &dc_frame).unwrap();
        assert!((bins_dc[0] - 160_000.0_f32).abs() < 1e-1);
        assert!(bins_dc.iter().all(|&b| b >= 0.0_f32 && b.is_finite()));

        // Pure cosine aligned to bin k0 = 32 of the 512-pt FFT
        let k0 = 32_usize;
        let mut tone = [0.0_f32; WHISPER_N_FFT];
        for (n, slot) in tone.iter_mut().enumerate() {
            #[expect(clippy::cast_precision_loss, reason = "n < 400 fits losslessly in f64")]
            let arg = 2.0_f64 * PI * (k0 as f64) * (n as f64) / 512.0_f64;
            #[expect(clippy::cast_possible_truncation, reason = "cosine lies in [-1, 1]")]
            {
                *slot = arg.cos() as f32;
            }
        }
        let mut bins_tone = [0.0_f32; WHISPER_N_BINS_201];
        rfft_power_spectrum_512_f32(&mut bins_tone, &tone).unwrap();
        let (peak_bin, _) = bins_tone
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .unwrap();
        assert_eq!(peak_bin, k0);
        assert!(bins_tone[k0] > 35_000.0_f32);
        assert!(bins_tone.iter().all(|&b| b >= 0.0_f32 && b.is_finite()));

        // Non-finite and length mismatch rejected
        let mut bad_frame = tone;
        bad_frame[10] = f32::NAN;
        assert!(matches!(
            rfft_power_spectrum_512_f32(&mut bins_tone, &bad_frame),
            Err(K3Error::TensorBoundsError { .. })
        ));
        assert!(matches!(
            rfft_power_spectrum_512_f32(&mut bins_tone[..200], &tone),
            Err(K3Error::TensorBoundsError { .. })
        ));
    }

    #[test]
    fn test_log_mel_spectrogram_whisper_dynamic_range_and_filters() {
        let n_mels = 80_usize;
        let n_frames = 4_usize;
        let pcm: Vec<f32> = (0..n_frames * WHISPER_HOP_LENGTH + WHISPER_N_FFT)
            .map(|i| {
                #[expect(clippy::cast_precision_loss, reason = "small test index")]
                let x = (i as f32) * 0.05_f32;
                0.5_f32 * x.sin()
            })
            .collect();

        // Synthetic normalized triangular-like filterbank across 201 bins
        let mut filters_201 = vec![0.0_f32; n_mels * WHISPER_N_BINS_201];
        for m in 0..n_mels {
            let bin = (m * 2 + 2).min(WHISPER_N_BINS_201 - 1);
            filters_201[m * WHISPER_N_BINS_201 + bin] = 1.0_f32;
        }

        let mut mel = vec![0.0_f32; n_mels * n_frames];
        log_mel_spectrogram_whisper_f32(&mut mel, &pcm, &filters_201, n_mels, n_frames).unwrap();
        assert!(mel.iter().all(|v| v.is_finite()));

        // Whisper clamp bounds: after (x + 4) / 4, max - min <= 8.0 / 4.0 = 2.0
        let max_v = mel.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min_v = mel.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(max_v - min_v <= 2.0_f32 + 1e-5_f32);

        // 128-bin mel filterbank also supported
        let filters_128 = vec![1.0_f32; 128 * WHISPER_N_BINS_201];
        let mut mel_128 = vec![0.0_f32; 128 * n_frames];
        log_mel_spectrogram_whisper_f32(&mut mel_128, &pcm, &filters_128, 128, n_frames).unwrap();
        assert!(mel_128.iter().all(|v| v.is_finite()));

        // Invalid n_mels and short PCM fail closed
        assert!(matches!(
            log_mel_spectrogram_whisper_f32(&mut mel, &pcm, &filters_201, 64, n_frames),
            Err(K3Error::TensorBoundsError { .. })
        ));
        assert!(matches!(
            log_mel_spectrogram_whisper_f32(&mut mel, &pcm[..100], &filters_201, n_mels, n_frames),
            Err(K3Error::TensorBoundsError { .. })
        ));
    }

    #[test]
    fn test_conv1d_k3_gelu_stride1_and_stride2() {
        // in_channels=2, in_len=5, out_channels=2
        let input = [
            1.0_f32, 2.0, 3.0, 4.0, 5.0, // ic=0
            -1.0, 0.5, 1.5, -2.0, 1.0, // ic=1
        ];
        // weight: [out_channels=2, in_channels=2, k=3]
        let weight = [
            0.5_f32, 1.0, -0.5, 0.25, -0.5, 0.75, // oc=0
            -1.0, 0.5, 0.5, 1.0, 1.0, -1.0, // oc=1
        ];
        let bias = [0.1_f32, -0.2];

        // Stride 1 -> out_len = 5
        let mut out_s1 = [0.0_f32; 2 * 5];
        conv1d_k3_gelu_f32(&mut out_s1, &input, 2, 5, &weight, &bias, 2, 1).unwrap();
        // Verify t_out = 0 for oc = 0:
        // center=0: ic0 (0*0.5 + 1*1.0 + 2*(-0.5) = 0) + ic1 (0*0.25 + (-1)*(-0.5) + 0.5*0.75 = 0.875) + 0.1 = 0.975
        let expected_0_0 = gelu_scalar_f32(0.975_f32);
        assert!((out_s1[0] - expected_0_0).abs() < 1e-5);
        // Verify t_out = 4 (right boundary padding=0) for oc = 0:
        // center=4: ic0 (4*0.5 + 5*1.0 + 0 = 7.0) + ic1 ((-2)*0.25 + 1*(-0.5) + 0 = -1.0) + 0.1 = 6.1
        let expected_0_4 = gelu_scalar_f32(6.1_f32);
        assert!((out_s1[4] - expected_0_4).abs() < 1e-5);

        // Stride 2 -> out_len = (5 - 1)/2 + 1 = 3 (matches t_out*2 = 0, 2, 4 from stride 1!)
        let mut out_s2 = [0.0_f32; 2 * 3];
        conv1d_k3_gelu_f32(&mut out_s2, &input, 2, 5, &weight, &bias, 2, 2).unwrap();
        assert!((out_s2[0] - out_s1[0]).abs() < 1e-5);
        assert!((out_s2[1] - out_s1[2]).abs() < 1e-5);
        assert!((out_s2[2] - out_s1[4]).abs() < 1e-5);
        assert!((out_s2[3] - out_s1[5]).abs() < 1e-5);
        assert!((out_s2[4] - out_s1[7]).abs() < 1e-5);
        assert!((out_s2[5] - out_s1[9]).abs() < 1e-5);

        // Edge case: in_len = 2, stride = 2 -> out_len = 1
        let mut out_len1 = [0.0_f32; 2];
        conv1d_k3_gelu_f32(
            &mut out_len1,
            &[1.0, 2.0, -1.0, 0.5],
            2,
            2,
            &weight,
            &bias,
            2,
            2,
        )
        .unwrap();
        assert!((out_len1[0] - expected_0_0).abs() < 1e-5);

        // Zero stride and non-finite inputs fail closed
        assert!(matches!(
            conv1d_k3_gelu_f32(&mut out_s1, &input, 2, 5, &weight, &bias, 2, 0),
            Err(K3Error::TensorBoundsError { .. })
        ));
        assert!(matches!(
            conv1d_k3_gelu_f32(&mut out_s1, &input, 2, 5, &weight, &[f32::NAN, 0.0], 2, 1),
            Err(K3Error::TensorBoundsError { .. })
        ));
    }
}
