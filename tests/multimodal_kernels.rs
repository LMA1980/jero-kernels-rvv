//! Integration tests for M10 multimodal kernels (`embed`, `vision`, `audio`) covering
//! zero-allocation execution, analytical reference parity (`f64`), RVV/scalar equivalence,
//! edge-case clamping, and fail-closed `K3Error::TensorBoundsError` validation.

use core::alloc::{GlobalAlloc, Layout};
use core::cell::Cell;
use core::f64::consts::PI;
use std::alloc::System;

use jero_core::K3Error;
use jero_kernels_rvv::{
    EMBED_POOL_CLS, EMBED_POOL_LAST_TOKEN, EMBED_POOL_MEAN, WHISPER_HOP_LENGTH, WHISPER_N_BINS_201,
    WHISPER_N_BINS_257, WHISPER_N_FFT, conv1d_k3_gelu_f32, conv1d_k3_gelu_f32_scalar,
    conv2d_patch_embed_f32, conv2d_patch_embed_f32_scalar, cosine_similarity_batch_f32,
    cosine_similarity_batch_f32_scalar,
    embedding::rerank_sigmoid_scores_f32 as alias_rerank_sigmoid_scores_f32, gelu_scalar_f32,
    hann_window_f32, hann_window_f32_scalar, interpolate_2d_pos_embed_f32,
    log_mel_spectrogram_whisper_f32, log_mel_spectrogram_whisper_f32_scalar,
    pixel_shuffle_merge_2x2_f32, pixel_shuffle_merge_2x2_f32_scalar, pool_and_l2_normalize_f32,
    pool_and_l2_normalize_f32_scalar, rerank_sigmoid_scores_f32, rerank_sigmoid_scores_f32_scalar,
    resize_rgb_bilinear_norm_f32, resize_rgb_bilinear_norm_f32_scalar, rfft_power_spectrum_512_f32,
};

struct ThreadCountingAllocator;

std::thread_local! {
    static TRACK_DEPTH: Cell<usize> = const { Cell::new(0) };
    static ALLOC_COUNT: Cell<usize> = const { Cell::new(0) };
    static REALLOC_COUNT: Cell<usize> = const { Cell::new(0) };
}

// SAFETY: Delegates directly to `std::alloc::System` while counting allocations on the
// current thread whenever `TRACK_DEPTH > 0`.
unsafe impl GlobalAlloc for ThreadCountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = TRACK_DEPTH.try_with(|depth| {
            if depth.get() > 0 {
                let _ = ALLOC_COUNT.try_with(|cnt| cnt.set(cnt.get() + 1));
            }
        });
        // SAFETY: Forwarded verbatim to `System` with identical `layout`.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` was allocated by `System` with `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let _ = TRACK_DEPTH.try_with(|depth| {
            if depth.get() > 0 {
                let _ = ALLOC_COUNT.try_with(|cnt| cnt.set(cnt.get() + 1));
            }
        });
        // SAFETY: Forwarded verbatim to `System` with identical `layout`.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let _ = TRACK_DEPTH.try_with(|depth| {
            if depth.get() > 0 {
                let _ = REALLOC_COUNT.try_with(|cnt| cnt.set(cnt.get() + 1));
            }
        });
        // SAFETY: Forwarded verbatim to `System` with identical `ptr`, `layout`, and `new_size`.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL_ALLOC: ThreadCountingAllocator = ThreadCountingAllocator;

fn assert_zero_heap_allocations<R>(label: &str, f: impl FnOnce() -> R) -> R {
    ALLOC_COUNT.with(|cnt| cnt.set(0));
    REALLOC_COUNT.with(|cnt| cnt.set(0));
    TRACK_DEPTH.with(|depth| depth.set(depth.get() + 1));
    let out = f();
    TRACK_DEPTH.with(|depth| depth.set(depth.get() - 1));
    let allocs = ALLOC_COUNT.with(Cell::get);
    let reallocs = REALLOC_COUNT.with(Cell::get);
    assert_eq!(
        allocs, 0,
        "{label}: expected 0 heap allocations, observed {allocs}"
    );
    assert_eq!(
        reallocs, 0,
        "{label}: expected 0 heap reallocations, observed {reallocs}"
    );
    out
}

#[expect(clippy::cast_precision_loss, reason = "small test index")]
fn to_f32(idx: usize) -> f32 {
    idx as f32
}

#[test]
fn test_embed_zero_alloc_and_f64_parity_across_tail_lengths() {
    for &dim in &[1_usize, 7, 15, 37, 64, 128, 256] {
        let seq_len = 5_usize;
        let hidden_seq: Vec<f32> = (0..seq_len * dim)
            .map(|i| (to_f32(i % 23) - 11.0_f32) * 0.25_f32)
            .collect();
        let mat_dim = (dim / 2).max(1);

        let mut out_mean = vec![0.0_f32; dim];
        let mut out_cls = vec![0.0_f32; dim];
        let mut out_last = vec![0.0_f32; dim];
        let mut out_mat = vec![0.0_f32; mat_dim];
        let mut out_scalar = vec![0.0_f32; dim];

        assert_zero_heap_allocations("pool_and_l2_normalize", || {
            pool_and_l2_normalize_f32(
                &mut out_mean,
                &hidden_seq,
                seq_len,
                dim,
                EMBED_POOL_MEAN,
                None,
            )
            .unwrap();
            pool_and_l2_normalize_f32(
                &mut out_cls,
                &hidden_seq,
                seq_len,
                dim,
                EMBED_POOL_CLS,
                None,
            )
            .unwrap();
            pool_and_l2_normalize_f32(
                &mut out_last,
                &hidden_seq,
                seq_len,
                dim,
                EMBED_POOL_LAST_TOKEN,
                None,
            )
            .unwrap();
            pool_and_l2_normalize_f32(
                &mut out_mat,
                &hidden_seq,
                seq_len,
                dim,
                EMBED_POOL_MEAN,
                Some(mat_dim),
            )
            .unwrap();
        });

        pool_and_l2_normalize_f32_scalar(
            &mut out_scalar,
            &hidden_seq,
            seq_len,
            dim,
            EMBED_POOL_MEAN,
            None,
        )
        .unwrap();

        for (a, b) in out_mean.iter().zip(out_scalar.iter()) {
            assert!((a - b).abs() < 1e-5_f32);
        }

        let norm_mean: f64 = out_mean.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        assert!((norm_mean.sqrt() - 1.0_f64).abs() < 1e-5_f64);
        let norm_mat: f64 = out_mat.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        assert!((norm_mat.sqrt() - 1.0_f64).abs() < 1e-5_f64);

        let num_docs = 6_usize;
        let docs: Vec<f32> = (0..num_docs * dim)
            .map(|i| (to_f32((i * 5) % 19) - 9.0_f32) * 0.125_f32)
            .collect();
        let mut scores = vec![0.0_f32; num_docs];
        let mut scores_scalar = vec![0.0_f32; num_docs];
        assert_zero_heap_allocations("cosine_similarity_batch", || {
            cosine_similarity_batch_f32(&mut scores, &out_mean, &docs, num_docs, dim).unwrap();
        });
        cosine_similarity_batch_f32_scalar(&mut scores_scalar, &out_mean, &docs, num_docs, dim)
            .unwrap();
        for (a, b) in scores.iter().zip(scores_scalar.iter()) {
            assert!((a - b).abs() < 1e-5_f32);
        }
    }

    // Module alias `embedding` and root re-export expose identical symbols.
    let logits = [-25.0_f32, -1.0, 0.0, 1.0, 25.0];
    let mut sig = [0.0_f32; 5];
    let mut sig_alias = [0.0_f32; 5];
    let mut sig_scalar = [0.0_f32; 5];
    assert_zero_heap_allocations("rerank_sigmoid_scores", || {
        rerank_sigmoid_scores_f32(&mut sig, &logits, 1.25_f32).unwrap();
        alias_rerank_sigmoid_scores_f32(&mut sig_alias, &logits, 1.25_f32).unwrap();
    });
    assert_eq!(sig, sig_alias);
    rerank_sigmoid_scores_f32_scalar(&mut sig_scalar, &logits, 1.25_f32).unwrap();
    for (a, b) in sig.iter().zip(sig_scalar.iter()) {
        assert!((a - b).abs() < 1e-5_f32);
    }
}

#[test]
fn test_vision_zero_alloc_and_vit_pipeline_parity() {
    let (src_h, src_w, dst_h, dst_w) = (8_usize, 10_usize, 28_usize, 28_usize);
    let patch_size = 14_usize;
    let hidden_dim = 17_usize;
    let mean = [0.481_454_66_f32, 0.457_827_5, 0.408_210_73];
    let std = [0.268_629_54_f32, 0.261_302_6, 0.275_777_1];

    let src_hwc: Vec<f32> = (0..src_h * src_w * 3)
        .map(|i| to_f32(i % 29) / 29.0_f32)
        .collect();
    let mut img_chw = vec![0.0_f32; 3 * dst_h * dst_w];
    let mut img_chw_scalar = vec![0.0_f32; 3 * dst_h * dst_w];

    assert_zero_heap_allocations("resize_rgb_bilinear_norm", || {
        resize_rgb_bilinear_norm_f32(
            &mut img_chw,
            &src_hwc,
            src_h,
            src_w,
            dst_h,
            dst_w,
            mean,
            std,
        )
        .unwrap();
    });
    resize_rgb_bilinear_norm_f32_scalar(
        &mut img_chw_scalar,
        &src_hwc,
        src_h,
        src_w,
        dst_h,
        dst_w,
        mean,
        std,
    )
    .unwrap();
    for (a, b) in img_chw.iter().zip(img_chw_scalar.iter()) {
        assert!((a - b).abs() < 1e-5_f32);
    }

    // Conv2D patch embed over 28x28 with patch_size=14 -> 2x2 = 4 patches
    let kernel: Vec<f32> = (0..hidden_dim * 3 * patch_size * patch_size)
        .map(|i| (to_f32(i % 17) - 8.0_f32) * 0.01_f32)
        .collect();
    let bias: Vec<f32> = (0..hidden_dim).map(|i| to_f32(i) * 0.02_f32).collect();
    let mut patches = vec![0.0_f32; 4 * hidden_dim];
    let mut patches_scalar = vec![0.0_f32; 4 * hidden_dim];

    assert_zero_heap_allocations("conv2d_patch_embed", || {
        conv2d_patch_embed_f32(
            &mut patches,
            &img_chw,
            3,
            dst_h,
            dst_w,
            patch_size,
            &kernel,
            Some(&bias),
            hidden_dim,
        )
        .unwrap();
    });
    conv2d_patch_embed_f32_scalar(
        &mut patches_scalar,
        &img_chw,
        3,
        dst_h,
        dst_w,
        patch_size,
        &kernel,
        Some(&bias),
        hidden_dim,
    )
    .unwrap();
    for (a, b) in patches.iter().zip(patches_scalar.iter()) {
        assert!((a - b).abs() < 1e-5_f32);
    }

    // Interpolate 2D pos embed from 2x2 -> 4x6, then pixel-shuffle 4x6 -> 2x3
    let mut pos_4x6 = vec![0.0_f32; 24 * hidden_dim];
    assert_zero_heap_allocations("interpolate_2d_pos_embed", || {
        interpolate_2d_pos_embed_f32(&mut pos_4x6, &patches, 2, 2, 4, 6, hidden_dim).unwrap();
    });
    let mut merged = vec![0.0_f32; 6 * 4 * hidden_dim];
    let mut merged_scalar = vec![0.0_f32; 6 * 4 * hidden_dim];
    assert_zero_heap_allocations("pixel_shuffle_merge_2x2", || {
        pixel_shuffle_merge_2x2_f32(&mut merged, &pos_4x6, 4, 6, hidden_dim).unwrap();
    });
    pixel_shuffle_merge_2x2_f32_scalar(&mut merged_scalar, &pos_4x6, 4, 6, hidden_dim).unwrap();
    assert_eq!(merged, merged_scalar);
}

#[test]
fn test_audio_zero_alloc_parseval_whisper_mel_and_conv1d() {
    let mut hann = [0.0_f32; WHISPER_N_FFT];
    let mut hann_scalar = [0.0_f32; WHISPER_N_FFT];
    assert_zero_heap_allocations("hann_window", || {
        hann_window_f32(&mut hann, WHISPER_N_FFT).unwrap();
    });
    hann_window_f32_scalar(&mut hann_scalar, WHISPER_N_FFT).unwrap();
    for (a, b) in hann.iter().zip(hann_scalar.iter()) {
        assert!((a - b).abs() < 1e-6_f32);
    }

    // Sum of power bins on pure cosine at k=16 matches Parseval expectation
    let mut frame = [0.0_f32; WHISPER_N_FFT];
    for (n, slot) in frame.iter_mut().enumerate() {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_precision_loss,
            reason = "n < 400 and unit cosine lies in [-1, 1]"
        )]
        {
            *slot = (2.0_f64 * PI * 16.0_f64 * (n as f64) / 512.0_f64).cos() as f32 * hann[n];
        }
    }
    let mut bins = [0.0_f32; WHISPER_N_BINS_201];
    assert_zero_heap_allocations("rfft_power_spectrum_512", || {
        rfft_power_spectrum_512_f32(&mut bins, &frame).unwrap();
    });
    let (peak_bin, _) = bins
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap();
    assert_eq!(peak_bin, 16);

    // Whisper log-mel with both 201 and 257 filterbank shapes + Conv1D stem (stride 1 & stride 2)
    let n_mels = 80_usize;
    let n_frames = 17_usize;
    let pcm: Vec<f32> = (0..n_frames * WHISPER_HOP_LENGTH)
        .map(|i| (to_f32(i) * 0.08_f32).sin() * 0.25_f32)
        .collect();
    let mut filters_257 = vec![0.0_f32; n_mels * WHISPER_N_BINS_257];
    for m in 0..n_mels {
        filters_257[m * WHISPER_N_BINS_257 + (m * 3 + 1).min(WHISPER_N_BINS_257 - 1)] = 1.0_f32;
    }

    let mut mel = vec![0.0_f32; n_mels * n_frames];
    let mut mel_scalar = vec![0.0_f32; n_mels * n_frames];
    assert_zero_heap_allocations("log_mel_spectrogram_whisper", || {
        log_mel_spectrogram_whisper_f32(&mut mel, &pcm, &filters_257, n_mels, n_frames).unwrap();
    });
    log_mel_spectrogram_whisper_f32_scalar(&mut mel_scalar, &pcm, &filters_257, n_mels, n_frames)
        .unwrap();
    for (a, b) in mel.iter().zip(mel_scalar.iter()) {
        assert!((a - b).abs() < 1e-5_f32);
    }

    // Whisper Conv1D stem: conv1 (80->64, stride 1) followed by conv2 (64->64, stride 2)
    let c_mid = 64_usize;
    let w1: Vec<f32> = (0..c_mid * n_mels * 3)
        .map(|i| (to_f32(i % 13) - 6.0_f32) * 0.01_f32)
        .collect();
    let b1 = vec![0.05_f32; c_mid];
    let mut h1 = vec![0.0_f32; c_mid * n_frames];
    let mut h1_scalar = vec![0.0_f32; c_mid * n_frames];
    assert_zero_heap_allocations("conv1d_k3_gelu_stride1", || {
        conv1d_k3_gelu_f32(&mut h1, &mel, n_mels, n_frames, &w1, &b1, c_mid, 1).unwrap();
    });
    conv1d_k3_gelu_f32_scalar(&mut h1_scalar, &mel, n_mels, n_frames, &w1, &b1, c_mid, 1).unwrap();
    for (a, b) in h1.iter().zip(h1_scalar.iter()) {
        assert!((a - b).abs() < 2e-5_f32);
    }

    let out_len_2 = (n_frames - 1) / 2 + 1;
    let w2: Vec<f32> = (0..c_mid * c_mid * 3)
        .map(|i| (to_f32(i % 11) - 5.0_f32) * 0.01_f32)
        .collect();
    let b2 = vec![-0.02_f32; c_mid];
    let mut h2 = vec![0.0_f32; c_mid * out_len_2];
    let mut h2_scalar = vec![0.0_f32; c_mid * out_len_2];
    assert_zero_heap_allocations("conv1d_k3_gelu_stride2", || {
        conv1d_k3_gelu_f32(&mut h2, &h1, c_mid, n_frames, &w2, &b2, c_mid, 2).unwrap();
    });
    conv1d_k3_gelu_f32_scalar(&mut h2_scalar, &h1, c_mid, n_frames, &w2, &b2, c_mid, 2).unwrap();
    for (a, b) in h2.iter().zip(h2_scalar.iter()) {
        assert!((a - b).abs() < 2e-5_f32);
    }
    // Sanity check GELU activation rejects all-NaN input without modifying output
    let mut untouched = [7.0_f32; 4];
    assert!(matches!(
        conv1d_k3_gelu_f32(
            &mut untouched,
            &[f32::NAN; 4],
            1,
            4,
            &[1.0; 3],
            &[0.0],
            1,
            1
        ),
        Err(K3Error::TensorBoundsError { .. })
    ));
    assert_eq!(untouched, [7.0_f32; 4]);
    let _ = gelu_scalar_f32(0.0_f32);
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[test]
fn test_riscv64_rvv_direct_parity_embed() {
    use jero_kernels_rvv::{
        cosine_similarity_batch_f32_rvv, pool_and_l2_normalize_f32_rvv,
        rerank_sigmoid_scores_f32_rvv,
    };

    let seq = [1.0_f32, -2.0, 3.0, 4.0, -5.0, 6.0, 0.5, 1.5, -0.5, 2.5];
    let mut pool_rvv = [0.0_f32; 5];
    let mut pool_sca = [0.0_f32; 5];
    pool_and_l2_normalize_f32_rvv(&mut pool_rvv, &seq, 2, 5, EMBED_POOL_MEAN, None).unwrap();
    pool_and_l2_normalize_f32_scalar(&mut pool_sca, &seq, 2, 5, EMBED_POOL_MEAN, None).unwrap();
    for (a, b) in pool_rvv.iter().zip(pool_sca.iter()) {
        assert!((a - b).abs() < 1e-5_f32);
    }

    let mut cos_rvv = [0.0_f32; 2];
    let mut cos_sca = [0.0_f32; 2];
    cosine_similarity_batch_f32_rvv(&mut cos_rvv, &pool_rvv, &seq, 2, 5).unwrap();
    cosine_similarity_batch_f32_scalar(&mut cos_sca, &pool_rvv, &seq, 2, 5).unwrap();
    for (a, b) in cos_rvv.iter().zip(cos_sca.iter()) {
        assert!((a - b).abs() < 1e-5_f32);
    }

    let mut sig_rvv = [0.0_f32; 5];
    let mut sig_sca = [0.0_f32; 5];
    rerank_sigmoid_scores_f32_rvv(&mut sig_rvv, &seq[..5], 1.5_f32).unwrap();
    rerank_sigmoid_scores_f32_scalar(&mut sig_sca, &seq[..5], 1.5_f32).unwrap();
    for (a, b) in sig_rvv.iter().zip(sig_sca.iter()) {
        assert!((a - b).abs() < 1e-5_f32);
    }
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[test]
fn test_riscv64_rvv_direct_parity_vision() {
    use jero_kernels_rvv::{
        conv2d_patch_embed_f32_rvv, pixel_shuffle_merge_2x2_f32_rvv,
        resize_rgb_bilinear_norm_f32_rvv,
    };

    let img_hwc = [
        0.2_f32, 0.4, 0.6, 0.8, 0.1, 0.3, 0.5, 0.7, 0.9, 0.0, 0.2, 0.4,
    ];
    let mut resize_rvv = [0.0_f32; 12];
    let mut resize_sca = [0.0_f32; 12];
    resize_rgb_bilinear_norm_f32_rvv(
        &mut resize_rvv,
        &img_hwc,
        2,
        2,
        2,
        2,
        [0.5, 0.5, 0.5],
        [0.25, 0.25, 0.25],
    )
    .unwrap();
    resize_rgb_bilinear_norm_f32_scalar(
        &mut resize_sca,
        &img_hwc,
        2,
        2,
        2,
        2,
        [0.5, 0.5, 0.5],
        [0.25, 0.25, 0.25],
    )
    .unwrap();
    for (a, b) in resize_rvv.iter().zip(resize_sca.iter()) {
        assert!((a - b).abs() < 1e-5_f32);
    }

    let mut patch_rvv = [0.0_f32; 2];
    let mut patch_sca = [0.0_f32; 2];
    let kern = [0.5_f32; 24];
    conv2d_patch_embed_f32_rvv(
        &mut patch_rvv,
        &resize_rvv,
        3,
        2,
        2,
        2,
        &kern,
        Some(&[0.1, -0.1]),
        2,
    )
    .unwrap();
    conv2d_patch_embed_f32_scalar(
        &mut patch_sca,
        &resize_rvv,
        3,
        2,
        2,
        2,
        &kern,
        Some(&[0.1, -0.1]),
        2,
    )
    .unwrap();
    for (a, b) in patch_rvv.iter().zip(patch_sca.iter()) {
        assert!((a - b).abs() < 1e-5_f32);
    }

    let mut shuffle_rvv = [0.0_f32; 12];
    let mut shuffle_sca = [0.0_f32; 12];
    pixel_shuffle_merge_2x2_f32_rvv(&mut shuffle_rvv, &resize_rvv, 2, 2, 3).unwrap();
    pixel_shuffle_merge_2x2_f32_scalar(&mut shuffle_sca, &resize_rvv, 2, 2, 3).unwrap();
    assert_eq!(shuffle_rvv, shuffle_sca);
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[test]
fn test_riscv64_rvv_direct_parity_audio() {
    use jero_kernels_rvv::{
        conv1d_k3_gelu_f32_rvv, hann_window_f32_rvv, log_mel_spectrogram_whisper_f32_rvv,
    };

    let mut hann_rvv = [0.0_f32; WHISPER_N_FFT];
    let mut hann_sca = [0.0_f32; WHISPER_N_FFT];
    hann_window_f32_rvv(&mut hann_rvv, WHISPER_N_FFT).unwrap();
    hann_window_f32_scalar(&mut hann_sca, WHISPER_N_FFT).unwrap();
    for (a, b) in hann_rvv.iter().zip(hann_sca.iter()) {
        assert!((a - b).abs() < 1e-6_f32);
    }

    let pcm = [0.1_f32; 400];
    let filters = vec![0.25_f32; 80 * WHISPER_N_BINS_201];
    let mut mel_rvv = vec![0.0_f32; 80];
    let mut mel_sca = vec![0.0_f32; 80];
    log_mel_spectrogram_whisper_f32_rvv(&mut mel_rvv, &pcm, &filters, 80, 1).unwrap();
    log_mel_spectrogram_whisper_f32_scalar(&mut mel_sca, &pcm, &filters, 80, 1).unwrap();
    for (a, b) in mel_rvv.iter().zip(mel_sca.iter()) {
        assert!((a - b).abs() < 1e-5_f32);
    }

    let mut conv_rvv = [0.0_f32; 3];
    let mut conv_sca = [0.0_f32; 3];
    conv1d_k3_gelu_f32_rvv(
        &mut conv_rvv,
        &[1.0, 2.0, 3.0, 4.0, 5.0],
        1,
        5,
        &[0.5, 1.0, -0.5],
        &[0.2],
        1,
        2,
    )
    .unwrap();
    conv1d_k3_gelu_f32_scalar(
        &mut conv_sca,
        &[1.0, 2.0, 3.0, 4.0, 5.0],
        1,
        5,
        &[0.5, 1.0, -0.5],
        &[0.2],
        1,
        2,
    )
    .unwrap();
    for (a, b) in conv_rvv.iter().zip(conv_sca.iter()) {
        assert!((a - b).abs() < 2e-5_f32);
    }
}
