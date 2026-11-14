//! Grouped-Query Attention (`GQA`) score calculation and weighted value accumulation.
//!
//! Supports Multi-Head Attention (`n_heads_q == n_heads_kv`), Multi-Query Attention
//! (`n_heads_kv == 1`), and Grouped-Query Attention (`n_heads_q % n_heads_kv == 0`)
//! with numerically stable softmax and token-interleaved KV cache indexing.

use jero_core::K3Error;
use std::ops::Range;

#[cfg(all(target_arch = "riscv64", not(miri)))]
use crate::dot_product_f32_rvv;
#[cfg(any(not(target_arch = "riscv64"), miri))]
use crate::dot_product_f32_scalar;

#[inline]
pub(crate) fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        dot_product_f32_rvv(a, b)
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        dot_product_f32_scalar(a, b)
    }
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[inline]
#[allow(unused_assignments)]
fn axpy_f32_rvv(out: &mut [f32], x: &[f32], alpha: f32) {
    let len = out.len().min(x.len());
    if len == 0 {
        return;
    }
    let mut po = out.as_mut_ptr();
    let mut px = x.as_ptr();
    let mut rem = len;

    // SAFETY: `po` and `px` are valid for `rem` f32 elements and do not alias (`out` is `&mut`).
    // Strip-mined `e32, m4` loop computes `out[i] += alpha * x[i]` on both `zvl256b` and `zvl1024b`.
    unsafe {
        core::arch::asm!(
            "2:",
            "vsetvli t0, {rem}, e32, m4, ta, ma",
            "vle32.v v8, ({po})",
            "vle32.v v16, ({px})",
            "vfmacc.vf v8, {alpha}, v16",
            "vse32.v v8, ({po})",
            "slli t1, t0, 2",
            "add {po}, {po}, t1",
            "add {px}, {px}, t1",
            "sub {rem}, {rem}, t0",
            "bnez {rem}, 2b",
            po = inout(reg) po,
            px = inout(reg) px,
            rem = inout(reg) rem,
            alpha = in(freg) alpha,
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

#[inline]
pub(crate) fn axpy_f32(out: &mut [f32], x: &[f32], alpha: f32) {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        axpy_f32_rvv(out, x, alpha);
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        for (dst, &val) in out.iter_mut().zip(x) {
            *dst += alpha * val;
        }
    }
}

/// Attention masking, positional offset, and score scaling configuration for multi-token GQA prefill (`#92`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttentionMaskConfig {
    /// Absolute sequence position of token `m = 0` in the current query batch.
    pub start_pos: usize,
    /// Optional local sliding-window span (`kv_pos + window >= query_pos + 1`).
    pub sliding_window: Option<usize>,
    /// Optional logit softcap (`cap * (score / cap).tanh()`).
    pub logit_softcap: Option<f32>,
    /// Dot-product scale factor (typically `1.0 / sqrt(head_dim)`).
    pub scale: f32,
}

impl AttentionMaskConfig {
    /// Creates a standard causal attention configuration starting at `start_pos` with `scale`.
    #[must_use]
    pub const fn causal(start_pos: usize, scale: f32) -> Self {
        Self {
            start_pos,
            sliding_window: None,
            logit_softcap: None,
            scale,
        }
    }

    /// Attaches a local sliding window bound (`None` disables windowing).
    #[must_use]
    pub const fn with_sliding_window(mut self, sliding_window: Option<usize>) -> Self {
        self.sliding_window = sliding_window;
        self
    }

    /// Attaches a finite positive logit soft-capping bound (`None` disables softcapping).
    #[must_use]
    pub const fn with_logit_softcap(mut self, logit_softcap: Option<f32>) -> Self {
        self.logit_softcap = logit_softcap;
        self
    }
}

/// Balances KV heads across `worker_count` parallel workers for contention-free GQA prefill (`#93`).
///
/// Returns `Some((kv_head_range, q_head_range))` for `worker_idx` when that worker has non-empty
/// work, or `None` if dimensions are invalid or `worker_idx >= n_heads_kv`.
#[must_use]
pub fn gqa_kv_head_partition(
    n_heads_kv: usize,
    n_heads_q: usize,
    worker_idx: usize,
    worker_count: usize,
) -> Option<(Range<usize>, Range<usize>)> {
    if worker_count == 0
        || worker_idx >= worker_count
        || n_heads_kv == 0
        || n_heads_q == 0
        || n_heads_q % n_heads_kv != 0
    {
        return None;
    }
    let base = n_heads_kv / worker_count;
    let rem = n_heads_kv % worker_count;
    let start_kv = worker_idx * base + worker_idx.min(rem);
    let count_kv = base + usize::from(worker_idx < rem);
    if count_kv == 0 {
        return None;
    }
    let group_size = n_heads_q / n_heads_kv;
    let end_kv = start_kv + count_kv;
    Some((start_kv..end_kv, start_kv * group_size..end_kv * group_size))
}

/// Computes Grouped-Query Attention (`GQA`) for a single query token across a sequence of cached keys and values.
///
/// Uses a stack scratch buffer for `seq_len <= 2048` (zero heap allocations) and delegates to
/// [`gqa_causal_attention_with_scratch_f32`]; prefer [`gqa_causal_attention_with_scratch_f32`] on
/// hot inference paths for arbitrary context lengths without allocation.
///
/// # Memory Layout
///
/// - `q`: query vector of shape `[n_heads_q, head_dim]`.
/// - `out`: output vector of shape `[n_heads_q, head_dim]`.
/// - `k_cache`: key buffer with shape `[seq_len, n_heads_kv, head_dim]`.
/// - `v_cache`: value buffer with shape `[seq_len, n_heads_kv, head_dim]`.
///
/// # Errors
///
/// - Returns [`K3Error::CapabilityMismatch`] if `n_heads_q % n_heads_kv != 0` or if any head dimension is zero.
/// - Returns [`K3Error::TensorBoundsError`] if buffer lengths do not match configured dimensions.
#[expect(
    clippy::too_many_arguments,
    reason = "attention kernel takes explicit buffer and geometry parameters"
)]
pub fn gqa_causal_attention_f32(
    out: &mut [f32],
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
) -> Result<(), K3Error> {
    if seq_len <= 2048 {
        let mut stack_scratch = [0.0_f32; 2048];
        return gqa_causal_attention_with_scratch_f32(
            out,
            q,
            k_cache,
            v_cache,
            n_heads_q,
            n_heads_kv,
            head_dim,
            seq_len,
            scale,
            &mut stack_scratch[..seq_len],
        );
    }
    let mut heap_scratch = vec![0.0_f32; seq_len];
    gqa_causal_attention_with_scratch_f32(
        out,
        q,
        k_cache,
        v_cache,
        n_heads_q,
        n_heads_kv,
        head_dim,
        seq_len,
        scale,
        &mut heap_scratch,
    )
}

/// Zero-allocation single-token Grouped-Query Attention (`GQA`) using a caller-supplied
/// `scores_scratch` buffer (`len >= seq_len`) (`#92`).
///
/// # Errors
///
/// - Returns [`K3Error::CapabilityMismatch`] if `n_heads_q % n_heads_kv != 0`, any dimension is zero,
///   or `scale` is non-finite.
/// - Returns [`K3Error::TensorBoundsError`] if tensor or scratch lengths are insufficient or mismatched.
#[expect(
    clippy::too_many_arguments,
    reason = "single-token GQA decode signature plus explicit scratch slice"
)]
pub fn gqa_causal_attention_with_scratch_f32(
    out: &mut [f32],
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
    scores_scratch: &mut [f32],
) -> Result<(), K3Error> {
    if seq_len == 0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: "gqa".to_owned(),
            reason: format!(
                "GQA dimensions must be non-zero (n_heads_q={n_heads_q}, n_heads_kv={n_heads_kv}, head_dim={head_dim}, seq_len={seq_len})"
            ),
        });
    }
    gqa_causal_attention_prefill_strided_kv_f32(
        out,
        q,
        k_cache,
        v_cache,
        1,
        n_heads_q,
        n_heads_kv,
        head_dim,
        seq_len,
        n_heads_kv.saturating_mul(head_dim),
        0..n_heads_kv,
        AttentionMaskConfig::causal(seq_len - 1, scale),
        scores_scratch,
    )
}

/// Zero-allocation multi-token causal Grouped-Query Attention (`GQA`) prefill across `m` query tokens
/// using a caller-supplied `scores_scratch` buffer (`#92`).
///
/// # Memory Layout
///
/// - `q`: `[m, n_heads_q, head_dim]`
/// - `out`: `[m, n_heads_q, head_dim]`
/// - `k_cache` / `v_cache`: `[total_seq_len, n_heads_kv, head_dim]`
///
/// # Errors
///
/// Returns [`K3Error::CapabilityMismatch`] on invalid dimensions, zero sliding window, or non-finite/non-positive
/// softcap/scale, or [`K3Error::TensorBoundsError`] on tensor or scratch length mismatch.
#[expect(
    clippy::too_many_arguments,
    reason = "explicit prefill geometry, mask config, and caller scratch"
)]
pub fn gqa_causal_attention_prefill_f32(
    out: &mut [f32],
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    m: usize,
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    total_seq_len: usize,
    config: AttentionMaskConfig,
    scores_scratch: &mut [f32],
) -> Result<(), K3Error> {
    gqa_causal_attention_prefill_strided_kv_f32(
        out,
        q,
        k_cache,
        v_cache,
        m,
        n_heads_q,
        n_heads_kv,
        head_dim,
        total_seq_len,
        n_heads_kv.saturating_mul(head_dim),
        0..n_heads_kv,
        config,
        scores_scratch,
    )
}

/// Multi-token causal GQA prefill restricted to a disjoint KV-head slice (`kv_head_range`), enabling
/// lock-free `WorkerPool` parallelism over `[0..n_heads_kv]` (`#93`).
///
/// - `q` holds all heads (`[m, n_heads_q, head_dim]`, shared read-only).
/// - `out` holds only the worker's output slice (`[m, kv_head_range.len() * (n_heads_q / n_heads_kv), head_dim]`),
///   matching `[m, n_heads_q, head_dim]` when `kv_head_range == 0..n_heads_kv`.
///
/// # Errors
///
/// Returns [`K3Error::CapabilityMismatch`] or [`K3Error::TensorBoundsError`] on invalid configuration,
/// out-of-range KV slice, or buffer bounds mismatch.
#[expect(
    clippy::too_many_arguments,
    reason = "explicit KV-head slice range and caller scratch for lock-free parallel workers"
)]
pub fn gqa_causal_attention_prefill_kv_heads_f32(
    out: &mut [f32],
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    m: usize,
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    total_seq_len: usize,
    kv_head_range: Range<usize>,
    config: AttentionMaskConfig,
    scores_scratch: &mut [f32],
) -> Result<(), K3Error> {
    gqa_causal_attention_prefill_strided_kv_f32(
        out,
        q,
        k_cache,
        v_cache,
        m,
        n_heads_q,
        n_heads_kv,
        head_dim,
        total_seq_len,
        n_heads_kv.saturating_mul(head_dim),
        kv_head_range,
        config,
        scores_scratch,
    )
}

#[expect(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "validated prefill geometry parameters"
)]
fn validate_prefill_args(
    out_len: usize,
    q_len: usize,
    k_len: usize,
    v_len: usize,
    scratch_len: usize,
    m: usize,
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    total_seq_len: usize,
    kv_row_stride: usize,
    kv_head_range: &Range<usize>,
    config: AttentionMaskConfig,
) -> Result<usize, K3Error> {
    if n_heads_q == 0 || n_heads_kv == 0 || head_dim == 0 || m == 0 || total_seq_len == 0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: "gqa".to_owned(),
            reason: format!(
                "GQA prefill dimensions must be non-zero (m={m}, n_heads_q={n_heads_q}, n_heads_kv={n_heads_kv}, head_dim={head_dim}, total_seq_len={total_seq_len})"
            ),
        });
    }
    if n_heads_q % n_heads_kv != 0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: "gqa".to_owned(),
            reason: format!(
                "n_heads_q ({n_heads_q}) must be divisible by n_heads_kv ({n_heads_kv})"
            ),
        });
    }
    if !config.scale.is_finite() {
        return Err(K3Error::CapabilityMismatch {
            architecture: "gqa".to_owned(),
            reason: format!("attention scale must be finite (got {})", config.scale),
        });
    }
    if config.sliding_window == Some(0) {
        return Err(K3Error::CapabilityMismatch {
            architecture: "gqa".to_owned(),
            reason: "sliding_window must be >= 1 when set".to_owned(),
        });
    }
    if let Some(cap) = config.logit_softcap {
        if !cap.is_finite() || cap <= 0.0 {
            return Err(K3Error::CapabilityMismatch {
                architecture: "gqa".to_owned(),
                reason: format!("logit_softcap must be finite and > 0 (got {cap})"),
            });
        }
    }
    if kv_head_range.start > kv_head_range.end || kv_head_range.end > n_heads_kv {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "gqa_kv_head_range".to_owned(),
            offset: kv_head_range.start,
            byte_len: kv_head_range.len(),
            buffer_len: n_heads_kv,
            reason: format!("kv_head_range {kv_head_range:?} exceeds n_heads_kv={n_heads_kv}"),
        });
    }

    let query_end = config
        .start_pos
        .checked_add(m)
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: "gqa_start_pos".to_owned(),
            offset: config.start_pos,
            byte_len: m,
            buffer_len: total_seq_len,
            reason: "start_pos + m overflow".to_owned(),
        })?;
    if query_end > total_seq_len {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "gqa_start_pos".to_owned(),
            offset: config.start_pos,
            byte_len: m,
            buffer_len: total_seq_len,
            reason: format!("start_pos + m ({query_end}) exceeds total_seq_len ({total_seq_len})"),
        });
    }

    let kv_width = n_heads_kv
        .checked_mul(head_dim)
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: "gqa_kv".to_owned(),
            offset: 0,
            byte_len: 0,
            buffer_len: usize::MAX,
            reason: "kv head width overflow".to_owned(),
        })?;
    if kv_row_stride < kv_width {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "gqa_kv_stride".to_owned(),
            offset: 0,
            byte_len: kv_row_stride,
            buffer_len: kv_width,
            reason: format!("kv_row_stride ({kv_row_stride}) < n_heads_kv * head_dim ({kv_width})"),
        });
    }

    let expected_q_len = m
        .checked_mul(n_heads_q)
        .and_then(|v| v.checked_mul(head_dim))
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: "gqa_q".to_owned(),
            offset: 0,
            byte_len: 0,
            buffer_len: usize::MAX,
            reason: "q dimension overflow".to_owned(),
        })?;
    if q_len != expected_q_len {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "gqa_q".to_owned(),
            offset: 0,
            byte_len: q_len * size_of::<f32>(),
            buffer_len: expected_q_len * size_of::<f32>(),
            reason: format!("query length {q_len} != expected {expected_q_len}"),
        });
    }

    let group_size = n_heads_q / n_heads_kv;
    let local_q_heads = kv_head_range.len() * group_size;
    let expected_out_len = m
        .checked_mul(local_q_heads)
        .and_then(|v| v.checked_mul(head_dim))
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: "gqa_out".to_owned(),
            offset: 0,
            byte_len: 0,
            buffer_len: usize::MAX,
            reason: "out dimension overflow".to_owned(),
        })?;
    if out_len != expected_out_len {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "gqa_out".to_owned(),
            offset: 0,
            byte_len: out_len * size_of::<f32>(),
            buffer_len: expected_out_len * size_of::<f32>(),
            reason: format!("output length {out_len} != expected {expected_out_len}"),
        });
    }

    let min_kv_len = (total_seq_len - 1)
        .checked_mul(kv_row_stride)
        .and_then(|v| v.checked_add(kv_width))
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: "gqa_kv".to_owned(),
            offset: 0,
            byte_len: 0,
            buffer_len: usize::MAX,
            reason: "kv dimension overflow".to_owned(),
        })?;
    if k_len < min_kv_len {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "gqa_k_cache".to_owned(),
            offset: 0,
            byte_len: k_len * size_of::<f32>(),
            buffer_len: min_kv_len * size_of::<f32>(),
            reason: format!("k_cache length {k_len} < required {min_kv_len}"),
        });
    }
    if v_len < min_kv_len {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "gqa_v_cache".to_owned(),
            offset: 0,
            byte_len: v_len * size_of::<f32>(),
            buffer_len: min_kv_len * size_of::<f32>(),
            reason: format!("v_cache length {v_len} < required {min_kv_len}"),
        });
    }

    let min_scratch = config
        .sliding_window
        .map_or(query_end, |w| w.min(query_end));
    if scratch_len < min_scratch {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "gqa_scores_scratch".to_owned(),
            offset: 0,
            byte_len: scratch_len * size_of::<f32>(),
            buffer_len: min_scratch * size_of::<f32>(),
            reason: format!("scores_scratch length {scratch_len} < required {min_scratch}"),
        });
    }

    Ok(group_size)
}

/// Strided-KV-cache GQA prefill over a KV-head range (`#93`), supporting KV caches whose token
/// row stride exceeds `n_heads_kv * head_dim` (padded/paged KV layouts) with KV-head outer grouping.
///
/// # Errors
///
/// Returns [`K3Error::CapabilityMismatch`] or [`K3Error::TensorBoundsError`] when dimensions,
/// stride, KV-head range, or scratch length are invalid.
#[expect(
    clippy::too_many_arguments,
    reason = "strided KV-cache prefill with KV-head range and caller scratch"
)]
pub fn gqa_causal_attention_prefill_strided_kv_f32(
    out: &mut [f32],
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    m: usize,
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    total_seq_len: usize,
    kv_row_stride: usize,
    kv_head_range: Range<usize>,
    config: AttentionMaskConfig,
    scores_scratch: &mut [f32],
) -> Result<(), K3Error> {
    let group_size = validate_prefill_args(
        out.len(),
        q.len(),
        k_cache.len(),
        v_cache.len(),
        scores_scratch.len(),
        m,
        n_heads_q,
        n_heads_kv,
        head_dim,
        total_seq_len,
        kv_row_stride,
        &kv_head_range,
        config,
    )?;

    if kv_head_range.is_empty() {
        return Ok(());
    }

    let local_q_heads = kv_head_range.len() * group_size;

    // Outer loop by KV head maximizes L1/L2 reuse of `[0..total_seq_len, kv_h, head_dim]`
    // across all `m` query tokens and `group_size` Q heads sharing that KV head.
    for kv_h in kv_head_range.clone() {
        let local_q_base = (kv_h - kv_head_range.start) * group_size;
        let kv_head_offset = kv_h * head_dim;

        for m_idx in 0..m {
            let query_pos = config.start_pos + m_idx;
            let valid_end = (query_pos + 1).min(total_seq_len);
            let valid_start = config
                .sliding_window
                .map_or(0, |w| query_pos.saturating_add(1).saturating_sub(w));
            let active_len = valid_end.saturating_sub(valid_start);
            let score_buf = &mut scores_scratch[..active_len];

            for g in 0..group_size {
                let global_q_h = kv_h * group_size + g;
                let local_q_h = local_q_base + g;
                let q_slice = &q[(m_idx * n_heads_q + global_q_h) * head_dim..][..head_dim];

                for (slot, s) in score_buf.iter_mut().zip(valid_start..valid_end) {
                    let kv_off = s * kv_row_stride + kv_head_offset;
                    let k_slice = &k_cache[kv_off..kv_off + head_dim];
                    let mut score = dot_f32(q_slice, k_slice) * config.scale;
                    if let Some(cap) = config.logit_softcap {
                        score = cap * (score / cap).tanh();
                    }
                    *slot = score;
                }

                if active_len > 0 {
                    softmax_scores_inplace(score_buf);
                }

                let out_slice =
                    &mut out[(m_idx * local_q_heads + local_q_h) * head_dim..][..head_dim];
                out_slice.fill(0.0_f32);

                for (s, &weight) in (valid_start..valid_end).zip(score_buf.iter()) {
                    if weight == 0.0_f32 {
                        continue;
                    }
                    let kv_off = s * kv_row_stride + kv_head_offset;
                    let v_slice = &v_cache[kv_off..kv_off + head_dim];
                    axpy_f32(out_slice, v_slice, weight);
                }
            }
        }
    }

    Ok(())
}

/// Computes unmasked bidirectional multi-head scaled dot-product attention over `seq_len` tokens.
///
/// Tensor layout for `out`, `q`, `k`, and `v` is row-major `[seq_len, n_heads, head_dim]`.
/// When `window_size` is `Some(w)`, query position `i` attends symmetrically to keys where
/// `|i - j| <= w`; when `None`, full global bidirectional attention over `0..seq_len` runs.
///
/// # Errors
/// Returns [`K3Error::CapabilityMismatch`] if `seq_len`, `n_heads`, `head_dim`, or element counts
/// overflow/are zero, and [`K3Error::TensorBoundsError`] if any buffer length mismatches.
#[allow(clippy::too_many_arguments)]
pub fn bidirectional_attention_f32(
    out: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
    window_size: Option<usize>,
    scale: f32,
) -> Result<(), K3Error> {
    if seq_len == 0 || n_heads == 0 || head_dim == 0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: "bidirectional_attention".to_owned(),
            reason: "seq_len, n_heads, and head_dim must be non-zero".to_owned(),
        });
    }

    let expected_len = seq_len
        .checked_mul(n_heads)
        .and_then(|n| n.checked_mul(head_dim))
        .ok_or_else(|| K3Error::CapabilityMismatch {
            architecture: "bidirectional_attention".to_owned(),
            reason: "attention buffer dimensions overflow usize".to_owned(),
        })?;
    let expected_bytes = expected_len.saturating_mul(size_of::<f32>());

    for (name, len) in [
        ("out", out.len()),
        ("q", q.len()),
        ("k", k.len()),
        ("v", v.len()),
    ] {
        if len != expected_len {
            return Err(K3Error::TensorBoundsError {
                tensor_name: name.to_owned(),
                offset: 0,
                byte_len: expected_bytes,
                buffer_len: len.saturating_mul(size_of::<f32>()),
                reason: format!("expected {expected_len} elements for {name}, got {len}"),
            });
        }
    }

    let mut scores = vec![0.0_f32; seq_len];

    for i in 0..seq_len {
        let (win_start, win_end) = match window_size {
            Some(w) => (i.saturating_sub(w), i.saturating_add(w).min(seq_len - 1)),
            None => (0, seq_len - 1),
        };
        let active_len = win_end - win_start + 1;

        for h in 0..n_heads {
            let q_off = (i * n_heads + h) * head_dim;
            let q_head = &q[q_off..q_off + head_dim];

            let mut max_score = f32::NEG_INFINITY;
            for (rel_idx, j) in (win_start..=win_end).enumerate() {
                let k_off = (j * n_heads + h) * head_dim;
                let k_head = &k[k_off..k_off + head_dim];
                let score = dot_f32(q_head, k_head) * scale;
                scores[rel_idx] = score;
                if score > max_score {
                    max_score = score;
                }
            }

            let mut weight_sum = 0.0_f32;
            for score in &mut scores[..active_len] {
                let w = (*score - max_score).exp();
                *score = w;
                weight_sum += w;
            }
            #[allow(clippy::cast_precision_loss)]
            let inv_sum = if weight_sum > 0.0 {
                1.0 / weight_sum
            } else {
                1.0 / (active_len as f32)
            };

            let out_head = &mut out[q_off..q_off + head_dim];
            out_head.fill(0.0);

            for (rel_idx, j) in (win_start..=win_end).enumerate() {
                let weight = scores[rel_idx] * inv_sum;
                if weight == 0.0 {
                    continue;
                }
                let v_off = (j * n_heads + h) * head_dim;
                let v_head = &v[v_off..v_off + head_dim];
                axpy_f32(out_head, v_head, weight);
            }
        }
    }

    Ok(())
}

/// Pools a `[seq_len, hidden_dim]` sequence tensor into a `[hidden_dim]` embedding vector.
///
/// When `mean_pool` is `true`, computes the arithmetic mean across all `seq_len` token vectors;
/// when `false`, extracts the first (`[CLS]`) token vector at index `0`.
pub fn pool_sequence_f32(
    out: &mut [f32],
    hidden: &[f32],
    seq_len: usize,
    hidden_dim: usize,
    mean_pool: bool,
) {
    out.fill(0.0);
    if seq_len == 0 || hidden_dim == 0 || out.is_empty() || hidden.is_empty() {
        return;
    }

    let dim = hidden_dim.min(out.len()).min(hidden.len());
    let max_rows = hidden.len() / hidden_dim;
    let active_seq = seq_len.min(max_rows);
    if active_seq == 0 {
        return;
    }

    let pooled = &mut out[..dim];
    if !mean_pool {
        pooled.copy_from_slice(&hidden[..dim]);
        return;
    }

    for s in 0..active_seq {
        let start = s * hidden_dim;
        let row = &hidden[start..start + dim];
        axpy_f32(pooled, row, 1.0);
    }
    #[allow(clippy::cast_precision_loss)]
    let inv_seq = 1.0 / (active_seq as f32);
    for val in pooled {
        *val *= inv_seq;
    }
}

/// Computes the `DeBERTa-v3` log-bucketed relative position index for `(q_idx, k_idx)` over
/// `2 * max_rel_pos` buckets centered at `max_rel_pos`.
///
/// Exact relative offsets inside `max_rel_pos / 2` map linearly, while larger distances map
/// logarithmically into the remaining buckets within `0..2 * max_rel_pos`. Returns `0` when
/// `max_rel_pos == 0`.
#[must_use]
pub fn compute_deberta_rel_pos_bucket(q_idx: usize, k_idx: usize, max_rel_pos: usize) -> usize {
    if max_rel_pos == 0 {
        return 0;
    }
    if q_idx == k_idx {
        return max_rel_pos;
    }

    let abs_d = q_idx.abs_diff(k_idx);
    let half = (max_rel_pos / 2).max(1);
    let mag = if abs_d < half || half >= max_rel_pos {
        abs_d.min(max_rel_pos)
    } else {
        #[allow(clippy::cast_precision_loss)]
        let ratio = ((abs_d as f32) / (half as f32)).ln()
            / (((2 * max_rel_pos) as f32) / (half as f32)).ln();
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            clippy::cast_precision_loss
        )]
        let scaled =
            half + ((ratio.clamp(0.0, 1.0) * ((max_rel_pos - half) as f32)).floor() as usize);
        scaled.clamp(half, max_rel_pos)
    };

    if q_idx < k_idx {
        max_rel_pos.saturating_sub(mag)
    } else {
        (max_rel_pos + mag).min(2 * max_rel_pos - 1)
    }
}

/// Computes the disentangled attention scaling factor `1.0 / sqrt(3.0 * head_dim)` for `DeBERTa-v3`.
///
/// # Errors
///
/// Returns [`K3Error::CapabilityMismatch`] if `head_dim == 0`.
pub fn deberta_attention_scale(head_dim: usize) -> Result<f32, K3Error> {
    if head_dim == 0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: "deberta-v3".to_owned(),
            reason: "head_dim must be non-zero for DeBERTa-v3 scale normalization".to_owned(),
        });
    }
    #[allow(clippy::cast_precision_loss)]
    let denom = (3.0_f32 * (head_dim as f32)).sqrt();
    Ok(1.0_f32 / denom)
}

#[allow(clippy::too_many_arguments)]
fn validate_deberta_buffers(
    out_len: usize,
    q_content_len: usize,
    k_content_len: usize,
    v_content_len: usize,
    q_pos_len: usize,
    k_pos_len: usize,
    rel_pos_buckets: &[usize],
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
) -> Result<(usize, usize), K3Error> {
    if seq_len == 0 || n_heads == 0 || head_dim == 0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: "deberta-v3".to_owned(),
            reason: "seq_len, n_heads, and head_dim must be non-zero".to_owned(),
        });
    }

    let hidden = n_heads
        .checked_mul(head_dim)
        .ok_or_else(|| K3Error::CapabilityMismatch {
            architecture: "deberta-v3".to_owned(),
            reason: "hidden dimension overflow usize".to_owned(),
        })?;
    let expected_len = seq_len
        .checked_mul(hidden)
        .ok_or_else(|| K3Error::CapabilityMismatch {
            architecture: "deberta-v3".to_owned(),
            reason: "sequence tensor length overflow usize".to_owned(),
        })?;
    let expected_bytes = expected_len.saturating_mul(size_of::<f32>());

    for (name, len) in [
        ("out", out_len),
        ("q_content", q_content_len),
        ("k_content", k_content_len),
        ("v_content", v_content_len),
    ] {
        if len != expected_len {
            return Err(K3Error::TensorBoundsError {
                tensor_name: name.to_owned(),
                offset: 0,
                byte_len: expected_bytes,
                buffer_len: len.saturating_mul(size_of::<f32>()),
                reason: format!("expected {expected_len} elements for {name}, got {len}"),
            });
        }
    }

    let expected_bucket_len =
        seq_len
            .checked_mul(seq_len)
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: "deberta-v3".to_owned(),
                reason: "rel_pos_buckets dimension overflow usize".to_owned(),
            })?;
    if rel_pos_buckets.len() != expected_bucket_len {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "rel_pos_buckets".to_owned(),
            offset: 0,
            byte_len: expected_bucket_len.saturating_mul(size_of::<usize>()),
            buffer_len: rel_pos_buckets.len().saturating_mul(size_of::<usize>()),
            reason: format!(
                "expected {expected_bucket_len} relative bucket entries, got {}",
                rel_pos_buckets.len()
            ),
        });
    }

    if q_pos_len != k_pos_len || q_pos_len < hidden || q_pos_len % hidden != 0 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "pos_embeddings".to_owned(),
            offset: 0,
            byte_len: k_pos_len.saturating_mul(size_of::<f32>()),
            buffer_len: q_pos_len.saturating_mul(size_of::<f32>()),
            reason: format!(
                "q_pos ({q_pos_len}) and k_pos ({k_pos_len}) must match and be non-zero multiples of hidden ({hidden})"
            ),
        });
    }

    let num_buckets = q_pos_len / hidden;
    for (idx, &bucket) in rel_pos_buckets.iter().enumerate() {
        if bucket >= num_buckets {
            return Err(K3Error::TensorBoundsError {
                tensor_name: "rel_pos_buckets".to_owned(),
                offset: idx,
                byte_len: size_of::<usize>(),
                buffer_len: num_buckets,
                reason: format!(
                    "relative position bucket {bucket} at index {idx} out of range 0..{num_buckets}"
                ),
            });
        }
    }

    Ok((hidden, num_buckets))
}

fn softmax_scores_inplace_scalar(scores: &mut [f32]) {
    if scores.is_empty() {
        return;
    }
    let mut max_score = f32::NEG_INFINITY;
    for &s in scores.iter() {
        if s > max_score {
            max_score = s;
        }
    }
    let mut sum_exp = 0.0_f32;
    for s in scores.iter_mut() {
        let e = (*s - max_score).exp();
        *s = e;
        sum_exp += e;
    }
    #[allow(clippy::cast_precision_loss)]
    let inv_sum = if sum_exp > 0.0 && sum_exp.is_finite() {
        1.0_f32 / sum_exp
    } else {
        1.0_f32 / (scores.len().max(1) as f32)
    };
    for s in scores.iter_mut() {
        *s *= inv_sum;
    }
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[expect(
    unused_assignments,
    clippy::too_many_lines,
    reason = "strip-mined RVV softmax mutates inout pointers and length counters across three passes"
)]
fn softmax_scores_inplace_rvv(scores: &mut [f32]) {
    const LN2_HI: f32 = f32::from_bits(0x3f31_7200);
    const LN2_LO: f32 = 1.428_606_8e-6_f32;
    const EXP_CONSTS: [f32; 10] = [
        -87.33_f32,
        88.0_f32,
        core::f32::consts::LOG2_E,
        LN2_HI,
        LN2_LO,
        1.0_f32 / 720.0_f32,
        1.0_f32 / 120.0_f32,
        1.0_f32 / 24.0_f32,
        1.0_f32 / 6.0_f32,
        0.5_f32,
    ];
    if scores.is_empty() {
        return;
    }

    let mut p_max = scores.as_ptr();
    let mut rem_max = scores.len();
    let mut max_score = f32::NEG_INFINITY;
    // SAFETY: `scores` is non-empty; VLA `e32, m1` `vfredmax.vs` scans `scores` safely.
    unsafe {
        core::arch::asm!(
            "vsetivli zero, 1, e32, m1, ta, ma",
            "vfmv.s.f v0, {init}",
            "2:",
            "vsetvli t0, {rem}, e32, m1, ta, ma",
            "vle32.v v8, ({p})",
            "vfredmax.vs v0, v8, v0",
            "slli t1, t0, 2",
            "add {p}, {p}, t1",
            "sub {rem}, {rem}, t0",
            "bnez {rem}, 2b",
            "vsetivli zero, 1, e32, m1, ta, ma",
            "vfmv.f.s {out_max}, v0",
            p = inout(reg) p_max,
            rem = inout(reg) rem_max,
            init = in(freg) f32::NEG_INFINITY,
            out_max = out(freg) max_score,
            out("t0") _,
            out("t1") _,
            out("v0") _,
            out("v8") _,
            options(nostack),
        );
    }

    if !max_score.is_finite() {
        softmax_scores_inplace_scalar(scores);
        return;
    }

    let mut p_exp = scores.as_mut_ptr();
    let mut rem_exp = scores.len();
    let mut sum_exp = 0.0_f32;
    let pc = EXP_CONSTS.as_ptr();
    // SAFETY: `scores` is exclusively borrowed and non-empty; Cody-Waite `expf` + `vfredusum.vs`
    // uses `e32, m1` dynamic `vl` (`VLMAX >= 4` on all RVV `VLEN >= 128`).
    unsafe {
        core::arch::asm!(
            "flw ft0, 0({c})",
            "flw ft1, 4({c})",
            "flw ft2, 8({c})",
            "flw ft3, 12({c})",
            "flw ft4, 16({c})",
            "li t2, 127",
            "vsetivli zero, 1, e32, m1, ta, ma",
            "vfmv.s.f v1, {zero}",
            "3:",
            "vsetvli t0, {rem}, e32, m1, ta, ma",
            "vle32.v v8, ({p})",
            "vfsub.vf v8, v8, {max_s}",
            "vmflt.vf v0, v8, ft0",
            "vfmax.vf v8, v8, ft0",
            "vfmin.vf v8, v8, ft1",
            "vfmul.vf v9, v8, ft2",
            "vfcvt.x.f.v v10, v9",
            "vfcvt.f.x.v v11, v10",
            "vfmul.vf v12, v11, ft3",
            "vfsub.vv v8, v8, v12",
            "vfmul.vf v12, v11, ft4",
            "vfsub.vv v8, v8, v12",
            "flw ft5, 20({c})",
            "vfmv.v.f v11, ft5",
            "flw ft6, 24({c})",
            "vfmul.vv v11, v11, v8",
            "vfadd.vf v11, v11, ft6",
            "flw ft5, 28({c})",
            "vfmul.vv v11, v11, v8",
            "vfadd.vf v11, v11, ft5",
            "flw ft6, 32({c})",
            "vfmul.vv v11, v11, v8",
            "vfadd.vf v11, v11, ft6",
            "flw ft5, 36({c})",
            "vfmul.vv v11, v11, v8",
            "vfadd.vf v11, v11, ft5",
            "vfmul.vv v11, v11, v8",
            "vfadd.vf v11, v11, {one}",
            "vfmul.vv v11, v11, v8",
            "vfadd.vf v11, v11, {one}",
            "vadd.vx v10, v10, t2",
            "vsll.vi v10, v10, 23",
            "vfmul.vv v11, v11, v10",
            "vfmerge.vfm v11, v11, {zero}, v0",
            "vse32.v v11, ({p})",
            "vfredusum.vs v1, v11, v1",
            "slli t1, t0, 2",
            "add {p}, {p}, t1",
            "sub {rem}, {rem}, t0",
            "bnez {rem}, 3b",
            "vsetivli zero, 1, e32, m1, ta, ma",
            "vfmv.f.s {out_sum}, v1",
            p = inout(reg) p_exp,
            rem = inout(reg) rem_exp,
            c = in(reg) pc,
            max_s = in(freg) max_score,
            one = in(freg) 1.0_f32,
            zero = in(freg) 0.0_f32,
            out_sum = out(freg) sum_exp,
            out("t0") _,
            out("t1") _,
            out("t2") _,
            out("ft0") _,
            out("ft1") _,
            out("ft2") _,
            out("ft3") _,
            out("ft4") _,
            out("ft5") _,
            out("ft6") _,
            out("v0") _,
            out("v1") _,
            out("v8") _,
            out("v9") _,
            out("v10") _,
            out("v11") _,
            out("v12") _,
            options(nostack),
        );
    }

    if !(sum_exp > 0.0 && sum_exp.is_finite()) {
        #[expect(clippy::cast_precision_loss, reason = "sequence length fits in f32")]
        let uniform = 1.0_f32 / (scores.len() as f32);
        scores.fill(uniform);
        return;
    }

    let inv_sum = 1.0_f32 / sum_exp;
    let mut p_norm = scores.as_mut_ptr();
    let mut rem_norm = scores.len();
    // SAFETY: `scores` is valid for `rem_norm` elements; `e32, m4` scale loop normalizes probabilities.
    unsafe {
        core::arch::asm!(
            "4:",
            "vsetvli t0, {rem}, e32, m4, ta, ma",
            "vle32.v v8, ({p})",
            "vfmul.vf v8, v8, {inv}",
            "vse32.v v8, ({p})",
            "slli t1, t0, 2",
            "add {p}, {p}, t1",
            "sub {rem}, {rem}, t0",
            "bnez {rem}, 4b",
            p = inout(reg) p_norm,
            rem = inout(reg) rem_norm,
            inv = in(freg) inv_sum,
            out("t0") _,
            out("t1") _,
            out("v8") _,
            out("v9") _,
            out("v10") _,
            out("v11") _,
            options(nostack),
        );
    }
}

fn softmax_scores_inplace(scores: &mut [f32]) {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        softmax_scores_inplace_rvv(scores);
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        softmax_scores_inplace_scalar(scores);
    }
}

/// Computes `DeBERTa-v3` disentangled relative positional attention across `seq_len` tokens:
/// `A_{i,j} = (Q^c_i (K^c_j)^T + Q^c_i (K^r_{b(i,j)})^T + K^c_j (Q^r_{b(j,i)})^T) / sqrt(3 * head_dim)`.
///
/// Tensor layout for `out`, `q_content`, `k_content`, and `v_content` is row-major
/// `[seq_len, n_heads, head_dim]`, while `q_pos` and `k_pos` have shape
/// `[num_buckets, n_heads, head_dim]` indexed by `rel_pos_buckets[i * seq_len + j]`.
///
/// # Errors
///
/// Returns [`K3Error::CapabilityMismatch`] if `seq_len`, `n_heads`, or `head_dim` is zero or overflows,
/// or [`K3Error::TensorBoundsError`] if tensor lengths or bucket indices are out of bounds.
#[allow(clippy::too_many_arguments)]
pub fn deberta_disentangled_attention_f32(
    out: &mut [f32],
    q_content: &[f32],
    k_content: &[f32],
    v_content: &[f32],
    q_pos: &[f32],
    k_pos: &[f32],
    rel_pos_buckets: &[usize],
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
) -> Result<(), K3Error> {
    let (hidden, _num_buckets) = validate_deberta_buffers(
        out.len(),
        q_content.len(),
        k_content.len(),
        v_content.len(),
        q_pos.len(),
        k_pos.len(),
        rel_pos_buckets,
        seq_len,
        n_heads,
        head_dim,
    )?;

    let scale = deberta_attention_scale(head_dim)?;
    let mut scores = vec![0.0_f32; seq_len];

    for i in 0..seq_len {
        for h in 0..n_heads {
            let head_base = i * hidden + h * head_dim;
            let qc = &q_content[head_base..head_base + head_dim];

            for (j, score_slot) in scores.iter_mut().enumerate().take(seq_len) {
                let key_offset = j * hidden + h * head_dim;
                let kc = &k_content[key_offset..key_offset + head_dim];

                let bucket_ij = rel_pos_buckets[i * seq_len + j];
                let k_pos_offset = bucket_ij * hidden + h * head_dim;
                let kr = &k_pos[k_pos_offset..k_pos_offset + head_dim];

                let bucket_ji = rel_pos_buckets[j * seq_len + i];
                let q_pos_offset = bucket_ji * hidden + h * head_dim;
                let qr = &q_pos[q_pos_offset..q_pos_offset + head_dim];

                let c2c = dot_f32(qc, kc);
                let c2p = dot_f32(qc, kr);
                let p2c = dot_f32(kc, qr);
                *score_slot = (c2c + c2p + p2c) * scale;
            }

            softmax_scores_inplace(&mut scores[..seq_len]);

            let out_head = &mut out[head_base..head_base + head_dim];
            out_head.fill(0.0);
            for (j, &weight) in scores.iter().enumerate().take(seq_len) {
                if weight == 0.0 {
                    continue;
                }
                let v_base = j * hidden + h * head_dim;
                let vc = &v_content[v_base..v_base + head_dim];
                axpy_f32(out_head, vc, weight);
            }
        }
    }

    Ok(())
}

/// Computes `DeepSeek` Multi-Head Latent Attention (`MLA`) in compressed latent space with
/// absorbed `W_UK` and `W_UV` projections.
///
/// Instead of materializing uncompressed `[seq_len, n_heads, head_dim]` K/V tensors, this kernel
/// scores absorbed nope queries (`q_nope: [n_heads, kv_lora_rank]`) + decoupled rotary queries
/// (`q_rotary: [n_heads, rope_dim]`) directly against the compressed latent cache
/// `kv_latent_cache: [seq_len, kv_lora_rank + rope_dim]`, and accumulates softmax-weighted latent
/// representations into `out_latent: [n_heads, kv_lora_rank]` via RVV `dot_f32` and `axpy_f32`.
///
/// # Errors
///
/// - Returns [`K3Error::CapabilityMismatch`] if `n_heads`, `kv_lora_rank`, or `seq_len` is zero,
///   if `scale` is non-finite, or if dimensions overflow `usize`.
/// - Returns [`K3Error::TensorBoundsError`] if any slice length mismatches expected bounds.
#[expect(
    clippy::too_many_arguments,
    reason = "MLA kernel takes explicit absorbed query, latent KV cache, and geometry parameters"
)]
pub fn mla_absorbed_attention_f32(
    out_latent: &mut [f32],
    q_nope: &[f32],
    q_rotary: &[f32],
    kv_latent_cache: &[f32],
    n_heads: usize,
    kv_lora_rank: usize,
    rope_dim: usize,
    seq_len: usize,
    scale: f32,
) -> Result<(), K3Error> {
    if n_heads == 0 || kv_lora_rank == 0 || seq_len == 0 || !scale.is_finite() {
        return Err(K3Error::CapabilityMismatch {
            architecture: "deepseek2".to_owned(),
            reason: format!(
                "invalid MLA parameters: n_heads={n_heads}, kv_lora_rank={kv_lora_rank}, seq_len={seq_len}, scale={scale}"
            ),
        });
    }

    let expected_latent =
        n_heads
            .checked_mul(kv_lora_rank)
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: "deepseek2".to_owned(),
                reason: "MLA out_latent dimension overflowed usize".to_owned(),
            })?;
    let expected_rope =
        n_heads
            .checked_mul(rope_dim)
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: "deepseek2".to_owned(),
                reason: "MLA q_rope dimension overflowed usize".to_owned(),
            })?;
    let stride = kv_lora_rank
        .checked_add(rope_dim)
        .ok_or_else(|| K3Error::CapabilityMismatch {
            architecture: "deepseek2".to_owned(),
            reason: "MLA latent cache stride overflowed usize".to_owned(),
        })?;
    let expected_cache =
        seq_len
            .checked_mul(stride)
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: "deepseek2".to_owned(),
                reason: "MLA cache length overflowed usize".to_owned(),
            })?;

    for (name, actual, expected) in [
        ("out_latent", out_latent.len(), expected_latent),
        ("q_nope", q_nope.len(), expected_latent),
        ("q_rotary", q_rotary.len(), expected_rope),
    ] {
        if actual != expected {
            return Err(K3Error::TensorBoundsError {
                tensor_name: name.to_owned(),
                offset: 0,
                byte_len: actual.saturating_mul(size_of::<f32>()),
                buffer_len: expected.saturating_mul(size_of::<f32>()),
                reason: format!("MLA buffer '{name}' length {actual} != expected {expected}"),
            });
        }
    }

    if kv_latent_cache.len() < expected_cache {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "kv_latent_cache".to_owned(),
            offset: 0,
            byte_len: kv_latent_cache.len().saturating_mul(size_of::<f32>()),
            buffer_len: expected_cache.saturating_mul(size_of::<f32>()),
            reason: format!(
                "kv_latent_cache length {} < expected {expected_cache}",
                kv_latent_cache.len()
            ),
        });
    }

    let mut scores = vec![0.0_f32; seq_len];

    for h in 0..n_heads {
        let q_abs_h = &q_nope[h * kv_lora_rank..(h + 1) * kv_lora_rank];
        let q_rot_h = &q_rotary[h * rope_dim..(h + 1) * rope_dim];

        for (t, score_slot) in scores.iter_mut().enumerate().take(seq_len) {
            let base = t * stride;
            let c_kv = &kv_latent_cache[base..base + kv_lora_rank];
            let score_nope = dot_f32(q_abs_h, c_kv);
            let score_rotary = if rope_dim > 0 {
                let k_rope = &kv_latent_cache[base + kv_lora_rank..base + stride];
                dot_f32(q_rot_h, k_rope)
            } else {
                0.0
            };
            *score_slot = (score_nope + score_rotary) * scale;
        }

        softmax_scores_inplace(&mut scores[..seq_len]);

        let out_h = &mut out_latent[h * kv_lora_rank..(h + 1) * kv_lora_rank];
        out_h.fill(0.0);

        for (t, &weight) in scores.iter().enumerate().take(seq_len) {
            if weight == 0.0 {
                continue;
            }
            let base = t * stride;
            let c_kv = &kv_latent_cache[base..base + kv_lora_rank];
            axpy_f32(out_h, c_kv, weight);
        }
    }

    Ok(())
}

/// Executes one autoregressive step of a Gated-`DeltaNet` linear-attention recurrent layer (`Qwen3.5`).
///
/// Updates the per-head recurrent state matrix `state: [n_heads, head_dim, head_dim]` in place:
/// `S_h = g_h * S_h + beta_h * (v_h - S_h * k_h) * k_h^T`, then computes `out_h = S_h * q_h`.
///
/// # Errors
///
/// - Returns [`K3Error::CapabilityMismatch`] if `n_heads == 0` or `head_dim == 0` or dimensions overflow.
/// - Returns [`K3Error::TensorBoundsError`] if buffer lengths mismatch `[n_heads, head_dim]` / `[n_heads]`.
#[expect(
    clippy::too_many_arguments,
    reason = "recurrent state update takes explicit QKV, decay gate, beta, and geometry parameters"
)]
pub fn gated_deltanet_step_f32(
    state: &mut [f32],
    out: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    decay_gate: &[f32],
    beta: &[f32],
    n_heads: usize,
    head_dim: usize,
) -> Result<(), K3Error> {
    if n_heads == 0 || head_dim == 0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: "qwen35".to_owned(),
            reason: format!(
                "gated_deltanet_step_f32 requires non-zero n_heads={n_heads} and head_dim={head_dim}"
            ),
        });
    }

    let expected_vec =
        n_heads
            .checked_mul(head_dim)
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: "qwen35".to_owned(),
                reason: "gated_deltanet_step_f32 vector length overflowed usize".to_owned(),
            })?;
    let expected_state =
        expected_vec
            .checked_mul(head_dim)
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: "qwen35".to_owned(),
                reason: "gated_deltanet_step_f32 state length overflowed usize".to_owned(),
            })?;

    for (name, actual, expected) in [
        ("state", state.len(), expected_state),
        ("out", out.len(), expected_vec),
        ("q", q.len(), expected_vec),
        ("k", k.len(), expected_vec),
        ("v", v.len(), expected_vec),
        ("decay_gate", decay_gate.len(), n_heads),
        ("beta", beta.len(), n_heads),
    ] {
        if actual != expected {
            return Err(K3Error::TensorBoundsError {
                tensor_name: name.to_owned(),
                offset: 0,
                byte_len: actual.saturating_mul(size_of::<f32>()),
                buffer_len: expected.saturating_mul(size_of::<f32>()),
                reason: format!("Gated-DeltaNet '{name}' length {actual} != expected {expected}"),
            });
        }
    }

    let head_state_len = head_dim * head_dim;
    for head_idx in 0..n_heads {
        let decay = decay_gate[head_idx];
        let gate_beta = beta[head_idx];
        let q_h = &q[head_idx * head_dim..(head_idx + 1) * head_dim];
        let k_h = &k[head_idx * head_dim..(head_idx + 1) * head_dim];
        let v_h = &v[head_idx * head_dim..(head_idx + 1) * head_dim];
        let out_h = &mut out[head_idx * head_dim..(head_idx + 1) * head_dim];
        let s_h = &mut state[head_idx * head_state_len..(head_idx + 1) * head_state_len];

        for row_idx in 0..head_dim {
            let s_row = &mut s_h[row_idx * head_dim..(row_idx + 1) * head_dim];
            for entry in &mut *s_row {
                *entry *= decay;
            }
            let retrieved = dot_f32(s_row, k_h);
            let delta = gate_beta * (v_h[row_idx] - retrieved);
            axpy_f32(s_row, k_h, delta);
            out_h[row_idx] = dot_f32(s_row, q_h);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gqa_single_token_decode_identity() {
        let head_dim = 4;
        let n_heads_q = 2;
        let n_heads_kv = 1;
        let seq_len = 1;

        let q = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let k_cache = [1.0_f32, 2.0, 3.0, 4.0];
        let v_cache = [10.0_f32, 20.0, 30.0, 40.0];
        let mut out = [0.0_f32; 8];

        gqa_causal_attention_f32(
            &mut out, &q, &k_cache, &v_cache, n_heads_q, n_heads_kv, head_dim, seq_len, 0.5,
        )
        .expect("gqa single token");

        // With seq_len=1, softmax weight is 1.0, so both heads output exactly v_cache[0]
        assert_eq!(&out[..4], &v_cache[..]);
        assert_eq!(&out[4..], &v_cache[..]);
    }

    #[test]
    fn test_gqa_multi_query_and_multi_head_equivalence() {
        let head_dim = 2;
        let n_heads_q = 4;
        let n_heads_kv = 2;
        let seq_len = 2;

        let q = [1.0_f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        // 2 tokens, 2 KV heads each:
        // Token 0: head 0=[1.0, 0.0], head 1=[0.0, 1.0]
        // Token 1: head 0=[0.0, 1.0], head 1=[1.0, 0.0]
        let k_cache = [
            1.0_f32, 0.0, 0.0, 1.0, // token 0
            0.0, 1.0, 1.0, 0.0, // token 1
        ];
        let v_cache = [
            1.0_f32, 0.0, 0.0, 1.0, // token 0
            0.0, 1.0, 1.0, 0.0, // token 1
        ];
        let mut out = [0.0_f32; 8];

        gqa_causal_attention_f32(
            &mut out, &q, &k_cache, &v_cache, n_heads_q, n_heads_kv, head_dim, seq_len, 1.0,
        )
        .expect("gqa causal attention");

        // Q head 0 attends to KV head 0: dot with tok 0 is 1.0, with tok 1 is 0.0.
        // Higher weight on tok 0 (w0 ≈ 0.731 > w1 ≈ 0.269), so out[0] > out[1].
        assert!(out[0] > out[1]);
    }

    #[test]
    fn test_gqa_validation_errors() {
        let mut out = [0.0_f32; 4];
        let q = [1.0_f32; 4];
        let k = [1.0_f32; 4];
        let v = [1.0_f32; 4];

        // n_heads_q not divisible by n_heads_kv (3 % 2 != 0)
        assert!(matches!(
            gqa_causal_attention_f32(&mut out, &q, &k, &v, 3, 2, 2, 1, 1.0),
            Err(K3Error::CapabilityMismatch { .. })
        ));

        // Buffer length mismatch
        assert!(matches!(
            gqa_causal_attention_f32(&mut out, &q[..2], &k, &v, 2, 1, 2, 1, 1.0),
            Err(K3Error::TensorBoundsError { .. })
        ));
    }

    #[test]
    fn test_bidirectional_attention_and_pooling() {
        // seq_len=3, n_heads=1, head_dim=2:
        // Keys/values are orthogonal one-hot directions scaled clearly:
        // tok 0 -> [1, 0], v0 = [10, 0]
        // tok 1 -> [0, 1], v1 = [0, 20]
        // tok 2 -> [1, 0], v2 = [100, 0]
        let q = [
            0.0_f32, 10.0, // tok 0 queries tok 1 (future token!) strongly
            10.0, 0.0, // tok 1 queries tok 0 and tok 2
            10.0, 0.0, // tok 2 queries tok 0 and tok 2
        ];
        let k = [
            1.0_f32, 0.0, // tok 0
            0.0, 1.0, // tok 1
            1.0, 0.0, // tok 2
        ];
        let v = [
            10.0_f32, 0.0, // tok 0
            0.0, 20.0, // tok 1
            100.0, 0.0, // tok 2
        ];

        // 1. Global unmasked bidirectional attention: tok 0 sees future tok 1 (v1[1] = 20).
        let mut out_global = [0.0_f32; 6];
        bidirectional_attention_f32(&mut out_global, &q, &k, &v, 3, 1, 2, None, 1.0)
            .expect("global bidirectional attention succeeds");
        assert!(
            out_global[1] > 19.0,
            "tok 0 must attend unmasked to future tok 1"
        );
        // tok 0 also has equal score (0.0) on tok 0 and tok 2, both within global context.
        assert!(out_global[0] > 0.0);

        // 2. Local sliding window w=0 (self-only): tok 0 only attends to tok 0 => out[0..2] == v[0..2]
        let mut out_w0 = [0.0_f32; 6];
        bidirectional_attention_f32(&mut out_w0, &q, &k, &v, 3, 1, 2, Some(0), 1.0)
            .expect("window=0 attention succeeds");
        assert!((out_w0[0] - 10.0).abs() <= 1e-5);
        assert!(out_w0[1].abs() <= 1e-5);

        // 3. Local sliding window w=1 on tok 0 excludes tok 2 (|0 - 2| > 1).
        let mut out_w1 = [0.0_f32; 6];
        bidirectional_attention_f32(&mut out_w1, &q, &k, &v, 3, 1, 2, Some(1), 1.0)
            .expect("window=1 attention succeeds");
        assert!(out_w1[0] < out_global[0]);

        // 4. Error validation
        assert!(matches!(
            bidirectional_attention_f32(&mut out_w1, &q[..4], &k, &v, 3, 1, 2, None, 1.0),
            Err(K3Error::TensorBoundsError { .. })
        ));
        assert!(matches!(
            bidirectional_attention_f32(&mut out_w1, &q, &k, &v, 0, 1, 2, None, 1.0),
            Err(K3Error::CapabilityMismatch { .. })
        ));

        // 5. Sequence pooling: CLS vs MeanPool
        let hidden = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut cls_out = [0.0_f32; 2];
        pool_sequence_f32(&mut cls_out, &hidden, 3, 2, false);
        assert!((cls_out[0] - 1.0).abs() <= 1e-6);
        assert!((cls_out[1] - 2.0).abs() <= 1e-6);

        let mut mean_out = [0.0_f32; 2];
        pool_sequence_f32(&mut mean_out, &hidden, 3, 2, true);
        assert!((mean_out[0] - 3.0).abs() <= 1e-6);
        assert!((mean_out[1] - 4.0).abs() <= 1e-6);
    }

    #[test]
    fn test_compute_deberta_rel_pos_bucket() {
        let max_rel_pos = 16;
        // Center bucket for identical query/key positions
        assert_eq!(
            compute_deberta_rel_pos_bucket(5, 5, max_rel_pos),
            max_rel_pos
        );
        // Edge case max_rel_pos == 0
        assert_eq!(compute_deberta_rel_pos_bucket(0, 3, 0), 0);

        // Exact small linear offsets around center (half = 8)
        assert_eq!(compute_deberta_rel_pos_bucket(5, 6, max_rel_pos), 15);
        assert_eq!(compute_deberta_rel_pos_bucket(6, 5, max_rel_pos), 17);

        // Antisymmetry & monotonic clamping in [0, 2 * max_rel_pos)
        let left_far = compute_deberta_rel_pos_bucket(0, 200, max_rel_pos);
        let right_far = compute_deberta_rel_pos_bucket(200, 0, max_rel_pos);
        assert_eq!(left_far, 0);
        assert_eq!(right_far, 2 * max_rel_pos - 1);

        // Scale factor 1 / sqrt(3 * head_dim)
        let scale = deberta_attention_scale(12).expect("valid scale");
        assert!((scale - (1.0_f32 / 6.0_f32)).abs() <= 1e-6);
        assert!(matches!(
            deberta_attention_scale(0),
            Err(K3Error::CapabilityMismatch { .. })
        ));
    }

    #[test]
    fn test_deberta_disentangled_attention_f32() {
        // Test C2P + P2C bias sensitivity when C2C content vectors are identical across keys.
        let seq_len = 2;
        let n_heads = 1;
        let head_dim = 2;
        let max_rel_pos = 2;
        let num_buckets = 2 * max_rel_pos; // 4

        let q_content = [1.0_f32, 0.0, 1.0, 0.0];
        let k_content = [1.0_f32, 0.0, 1.0, 0.0];
        let v_content = [10.0_f32, 0.0, 0.0, 20.0];

        let mut rel_pos_buckets = [0_usize; 4];
        for i in 0..seq_len {
            for j in 0..seq_len {
                rel_pos_buckets[i * seq_len + j] =
                    compute_deberta_rel_pos_bucket(i, j, max_rel_pos);
            }
        }
        // Bucket 1 = rel(-1) (i=0, j=1); Bucket 2 = rel(0); Bucket 3 = rel(+1) (i=1, j=0)
        let mut q_pos = vec![0.0_f32; num_buckets * head_dim];
        let mut k_pos = vec![0.0_f32; num_buckets * head_dim];
        // Make bucket 1 strongly attractive to q_content=[1, 0] via c2p
        k_pos[head_dim] = 10.0;
        // Make bucket 3 strongly unattractive via c2p
        k_pos[3 * head_dim] = -10.0;
        // Neutral self-position, positive p2c at bucket 3 when querying j=1 from i=0 (b_{1,0}=3)
        q_pos[3 * head_dim] = 5.0;

        let mut out = [0.0_f32; 4];
        deberta_disentangled_attention_f32(
            &mut out,
            &q_content,
            &k_content,
            &v_content,
            &q_pos,
            &k_pos,
            &rel_pos_buckets,
            seq_len,
            n_heads,
            head_dim,
        )
        .expect("disentangled attention succeeds");

        // Token 0 must focus overwhelmingly on token 1 (v_content[1] = [0, 20])
        assert!(
            out[1] > 18.0 && out[0] < 2.0,
            "tok 0 must attend to tok 1 via C2P+P2C: out={out:?}"
        );
        assert!(out.iter().all(|v| v.is_finite()));

        // Error checks
        assert!(matches!(
            deberta_disentangled_attention_f32(
                &mut out[..2],
                &q_content,
                &k_content,
                &v_content,
                &q_pos,
                &k_pos,
                &rel_pos_buckets,
                seq_len,
                n_heads,
                head_dim,
            ),
            Err(K3Error::TensorBoundsError { .. })
        ));
        let mut bad_buckets = rel_pos_buckets;
        bad_buckets[0] = num_buckets;
        assert!(matches!(
            deberta_disentangled_attention_f32(
                &mut out,
                &q_content,
                &k_content,
                &v_content,
                &q_pos,
                &k_pos,
                &bad_buckets,
                seq_len,
                n_heads,
                head_dim,
            ),
            Err(K3Error::TensorBoundsError { .. })
        ));
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_deberta_c2p_p2c_attention() {
        // Multi-head numerical parity check against reference scalar computation with 1/sqrt(3d)
        let seq_len = 3;
        let n_heads = 2;
        let head_dim = 4;
        let hidden = n_heads * head_dim;
        let max_rel_pos = 4;
        let num_buckets = 2 * max_rel_pos;

        let mut rel_pos_buckets = vec![0_usize; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                rel_pos_buckets[i * seq_len + j] =
                    compute_deberta_rel_pos_bucket(i, j, max_rel_pos);
            }
        }

        let q_c: Vec<f32> = (0..seq_len * hidden)
            .map(|x| ((x % 7) as f32 - 3.0) * 0.25)
            .collect();
        let k_c: Vec<f32> = (0..seq_len * hidden)
            .map(|x| ((x % 5) as f32 - 2.0) * 0.3)
            .collect();
        let v_c: Vec<f32> = (0..seq_len * hidden).map(|x| (x as f32) * 0.1).collect();
        let q_p: Vec<f32> = (0..num_buckets * hidden)
            .map(|x| ((x % 6) as f32 - 2.5) * 0.2)
            .collect();
        let k_p: Vec<f32> = (0..num_buckets * hidden)
            .map(|x| ((x % 4) as f32 - 1.5) * 0.35)
            .collect();

        let mut out = vec![0.0_f32; seq_len * hidden];
        deberta_disentangled_attention_f32(
            &mut out,
            &q_c,
            &k_c,
            &v_c,
            &q_p,
            &k_p,
            &rel_pos_buckets,
            seq_len,
            n_heads,
            head_dim,
        )
        .expect("multi-head c2p+p2c attention succeeds");

        // Compare head 1 of token 2 with pure content-only attention (zero position tables):
        // c2p + p2c must produce distinct contextual outputs.
        let zero_pos = vec![0.0_f32; num_buckets * hidden];
        let mut out_no_pos = vec![0.0_f32; seq_len * hidden];
        deberta_disentangled_attention_f32(
            &mut out_no_pos,
            &q_c,
            &k_c,
            &v_c,
            &zero_pos,
            &zero_pos,
            &rel_pos_buckets,
            seq_len,
            n_heads,
            head_dim,
        )
        .expect("c2c-only attention succeeds");

        let max_diff = out
            .iter()
            .zip(out_no_pos.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_diff > 1e-3, "c2p+p2c must contribute non-trivially");
    }

    #[test]
    fn test_deberta_disentangled_attention() {
        let mut out = [0.0_f32; 4];
        let q = [1.0_f32, 0.0, 0.0, 1.0];
        let k = [1.0_f32, 0.0, 0.0, 1.0];
        let v = [2.0_f32, 4.0, 6.0, 8.0];
        let q_pos = [0.5_f32, 0.0, 0.0, 0.5, 0.1, 0.2, 0.3, 0.4];
        let k_pos = [0.5_f32, 0.0, 0.0, 0.5, 0.1, 0.2, 0.3, 0.4];
        let buckets = [2_usize, 1, 3, 2];

        deberta_disentangled_attention_f32(&mut out, &q, &k, &v, &q_pos, &k_pos, &buckets, 2, 1, 2)
            .expect("disentangled attention integration smoke check");
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[allow(clippy::too_many_arguments)]
    fn compute_uncompressed_mla_reference(
        kv_latent_cache: &[f32],
        key_up_matrix: &[f32],
        value_up_matrix: &[f32],
        q_nope_raw: &[f32],
        q_rotary: &[f32],
        n_heads: usize,
        kv_lora_rank: usize,
        rope_dim: usize,
        d_nope: usize,
        d_v: usize,
        seq_len: usize,
        scale: f32,
    ) -> Vec<f32> {
        let stride = kv_lora_rank + rope_dim;
        let head_dim_k = d_nope + rope_dim;
        let mut q_full = vec![0.0_f32; n_heads * head_dim_k];
        let mut k_full = vec![0.0_f32; seq_len * n_heads * head_dim_k];
        let mut v_full = vec![0.0_f32; seq_len * n_heads * d_v];

        for h in 0..n_heads {
            q_full[h * head_dim_k..h * head_dim_k + d_nope]
                .copy_from_slice(&q_nope_raw[h * d_nope..(h + 1) * d_nope]);
            q_full[h * head_dim_k + d_nope..(h + 1) * head_dim_k]
                .copy_from_slice(&q_rotary[h * rope_dim..(h + 1) * rope_dim]);

            for t in 0..seq_len {
                let c_kv = &kv_latent_cache[t * stride..t * stride + kv_lora_rank];
                let k_rope = &kv_latent_cache[t * stride + kv_lora_rank..t * stride + stride];
                let k_dst =
                    &mut k_full[(t * n_heads + h) * head_dim_k..(t * n_heads + h + 1) * head_dim_k];
                for (d, slot) in k_dst[..d_nope].iter_mut().enumerate() {
                    *slot = c_kv
                        .iter()
                        .enumerate()
                        .map(|(r, &c_val)| {
                            c_val * key_up_matrix[(h * kv_lora_rank + r) * d_nope + d]
                        })
                        .sum();
                }
                k_dst[d_nope..].copy_from_slice(k_rope);

                let v_dst = &mut v_full[(t * n_heads + h) * d_v..(t * n_heads + h + 1) * d_v];
                for (vi, slot) in v_dst.iter_mut().enumerate() {
                    *slot = c_kv
                        .iter()
                        .enumerate()
                        .map(|(r, &c_val)| {
                            c_val * value_up_matrix[(h * kv_lora_rank + r) * d_v + vi]
                        })
                        .sum();
                }
            }
        }

        let mut ref_out = vec![0.0_f32; n_heads * d_v];
        let mut ref_scores = vec![0.0_f32; seq_len];
        for h in 0..n_heads {
            let qh = &q_full[h * head_dim_k..(h + 1) * head_dim_k];
            for (t, sc) in ref_scores.iter_mut().enumerate() {
                let kh =
                    &k_full[(t * n_heads + h) * head_dim_k..(t * n_heads + h + 1) * head_dim_k];
                *sc = dot_f32(qh, kh) * scale;
            }
            softmax_scores_inplace(&mut ref_scores);
            for (t, &w) in ref_scores.iter().enumerate() {
                let vh = &v_full[(t * n_heads + h) * d_v..(t * n_heads + h + 1) * d_v];
                axpy_f32(&mut ref_out[h * d_v..(h + 1) * d_v], vh, w);
            }
        }
        ref_out
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_mla_absorbed_attention_equivalence() {
        let (n_heads, kv_lora_rank, rope_dim, d_nope, d_v, seq_len) = (3, 8, 4, 6, 5, 4);
        let scale = 1.0_f32 / ((d_nope + rope_dim) as f32).sqrt();
        let stride = kv_lora_rank + rope_dim;
        let mut kv_latent_cache = vec![0.0_f32; seq_len * stride];
        for t in 0..seq_len {
            for r in 0..kv_lora_rank {
                kv_latent_cache[t * stride + r] = ((t * 7 + r * 3) % 11) as f32 * 0.1 - 0.5;
            }
            for d in 0..rope_dim {
                kv_latent_cache[t * stride + kv_lora_rank + d] =
                    ((t * 5 + d * 2) % 9) as f32 * 0.12 - 0.4;
            }
        }

        let key_up_matrix: Vec<f32> = (0..n_heads * kv_lora_rank * d_nope)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.08)
            .collect();
        let value_up_matrix: Vec<f32> = (0..n_heads * kv_lora_rank * d_v)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.07)
            .collect();
        let q_nope_raw: Vec<f32> = (0..n_heads * d_nope)
            .map(|i| ((i % 11) as f32 - 5.0) * 0.15)
            .collect();
        let q_rotary: Vec<f32> = (0..n_heads * rope_dim)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.2)
            .collect();

        let ref_out = compute_uncompressed_mla_reference(
            &kv_latent_cache,
            &key_up_matrix,
            &value_up_matrix,
            &q_nope_raw,
            &q_rotary,
            n_heads,
            kv_lora_rank,
            rope_dim,
            d_nope,
            d_v,
            seq_len,
            scale,
        );

        let mut q_nope_abs = vec![0.0_f32; n_heads * kv_lora_rank];
        for h in 0..n_heads {
            for r in 0..kv_lora_rank {
                q_nope_abs[h * kv_lora_rank + r] = (0..d_nope)
                    .map(|d| {
                        q_nope_raw[h * d_nope + d]
                            * key_up_matrix[(h * kv_lora_rank + r) * d_nope + d]
                    })
                    .sum();
            }
        }

        let mut out_latent = vec![0.0_f32; n_heads * kv_lora_rank];
        mla_absorbed_attention_f32(
            &mut out_latent,
            &q_nope_abs,
            &q_rotary,
            &kv_latent_cache,
            n_heads,
            kv_lora_rank,
            rope_dim,
            seq_len,
            scale,
        )
        .expect("mla absorbed attention succeeds");

        let mut abs_out = vec![0.0_f32; n_heads * d_v];
        for h in 0..n_heads {
            for vi in 0..d_v {
                abs_out[h * d_v + vi] = (0..kv_lora_rank)
                    .map(|r| {
                        out_latent[h * kv_lora_rank + r]
                            * value_up_matrix[(h * kv_lora_rank + r) * d_v + vi]
                    })
                    .sum();
            }
        }

        let max_diff = ref_out
            .iter()
            .zip(abs_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_diff < 1e-5, "max_diff={max_diff}");
    }

    #[test]
    fn test_mla_absorbed_attention_error_checks() {
        let mut out = [0.0_f32; 4];
        let qn = [0.0_f32; 4];
        let qr = [0.0_f32; 2];
        let kv = [0.0_f32; 6];
        assert!(matches!(
            mla_absorbed_attention_f32(&mut out, &qn, &qr, &kv, 0, 2, 1, 2, 1.0),
            Err(K3Error::CapabilityMismatch { .. })
        ));
        assert!(matches!(
            mla_absorbed_attention_f32(&mut out[..2], &qn, &qr, &kv, 2, 2, 1, 2, 1.0),
            Err(K3Error::TensorBoundsError { .. })
        ));
    }

    #[test]
    fn test_gated_deltanet_step_f32() {
        let n_heads = 2;
        let head_dim = 3;
        let mut state = vec![0.0_f32; n_heads * head_dim * head_dim];
        let mut out = vec![0.0_f32; n_heads * head_dim];
        let q = [1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0];
        let k = [1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0];
        let v = [2.0_f32, 4.0, 6.0, 1.0, 3.0, 5.0];
        let decay_gate = [0.9_f32, 0.5_f32];
        let beta = [1.0_f32, 0.5_f32];

        // Step 1 from zero state: S_h = beta_h * v_h * k_h^T, out_h = S_h * q_h = beta_h * v_h
        gated_deltanet_step_f32(
            &mut state,
            &mut out,
            &q,
            &k,
            &v,
            &decay_gate,
            &beta,
            n_heads,
            head_dim,
        )
        .expect("step 1 succeeds");
        assert!((out[0] - 2.0).abs() < 1e-6);
        assert!((out[1] - 4.0).abs() < 1e-6);
        assert!((out[3] - 0.5).abs() < 1e-6);
        assert!((out[4] - 1.5).abs() < 1e-6);

        // Error checks
        assert!(matches!(
            gated_deltanet_step_f32(&mut state, &mut out, &q, &k, &v, &decay_gate, &beta, 0, 3),
            Err(K3Error::CapabilityMismatch { .. })
        ));
    }

    fn make_wave3_seq(len: usize, mul: usize, modulo: usize, bias: f32, step: f32) -> Vec<f32> {
        (0..len)
            .map(|idx| (f32::from(u8::try_from((idx * mul) % modulo).unwrap()) - bias) * step)
            .collect()
    }

    #[test]
    fn wave3_gqa_prefill_matches_sequential_decode_with_swa_and_softcap() {
        let m_tokens = 6;
        let start_pos = 3;
        let total_seq = start_pos + m_tokens;
        let (n_q, n_kv, head_dim) = (4, 2, 8);
        let scale = 1.0_f32 / 8.0_f32.sqrt();
        let q_buf = make_wave3_seq(m_tokens * n_q * head_dim, 1, 13, 6.0, 0.15);
        let k_buf = make_wave3_seq(total_seq * n_kv * head_dim, 3, 11, 5.0, 0.12);
        let v_buf = make_wave3_seq(total_seq * n_kv * head_dim, 5, 17, 8.0, 0.1);

        let cfg = AttentionMaskConfig::causal(start_pos, scale);
        let mut scratch = vec![0.0_f32; total_seq];
        let mut out_prefill = vec![0.0_f32; m_tokens * n_q * head_dim];
        gqa_causal_attention_prefill_f32(
            &mut out_prefill,
            &q_buf,
            &k_buf,
            &v_buf,
            m_tokens,
            n_q,
            n_kv,
            head_dim,
            total_seq,
            cfg,
            &mut scratch,
        )
        .unwrap();

        let mut out_seq = vec![0.0_f32; n_q * head_dim];
        for tok in 0..m_tokens {
            let valid_seq = start_pos + tok + 1;
            gqa_causal_attention_with_scratch_f32(
                &mut out_seq,
                &q_buf[tok * n_q * head_dim..(tok + 1) * n_q * head_dim],
                &k_buf[..valid_seq * n_kv * head_dim],
                &v_buf[..valid_seq * n_kv * head_dim],
                n_q,
                n_kv,
                head_dim,
                valid_seq,
                scale,
                &mut scratch,
            )
            .unwrap();
            for idx in 0..n_q * head_dim {
                assert!(
                    (out_prefill[tok * n_q * head_dim + idx] - out_seq[idx]).abs() < 1e-6,
                    "tok={tok} idx={idx}"
                );
            }
        }

        let cfg_swa = AttentionMaskConfig::causal(start_pos, scale)
            .with_sliding_window(Some(3))
            .with_logit_softcap(Some(2.0));
        let mut out_swa = vec![0.0_f32; m_tokens * n_q * head_dim];
        gqa_causal_attention_prefill_f32(
            &mut out_swa,
            &q_buf,
            &k_buf,
            &v_buf,
            m_tokens,
            n_q,
            n_kv,
            head_dim,
            total_seq,
            cfg_swa,
            &mut scratch,
        )
        .unwrap();
        assert!(out_swa.iter().all(|x| x.is_finite()));

        let (kv_r0, q_r0) = gqa_kv_head_partition(n_kv, n_q, 0, 2).unwrap();
        assert_eq!(kv_r0, 0..1);
        assert_eq!(q_r0, 0..2);
        let mut part0 = vec![0.0_f32; m_tokens * 2 * head_dim];
        gqa_causal_attention_prefill_kv_heads_f32(
            &mut part0,
            &q_buf,
            &k_buf,
            &v_buf,
            m_tokens,
            n_q,
            n_kv,
            head_dim,
            total_seq,
            kv_r0,
            cfg_swa,
            &mut scratch,
        )
        .unwrap();
        for tok in 0..m_tokens {
            let lhs = &part0[tok * 2 * head_dim..(tok + 1) * 2 * head_dim];
            let rhs = &out_swa[tok * n_q * head_dim..tok * n_q * head_dim + 2 * head_dim];
            #[cfg(not(miri))]
            assert_eq!(lhs, rhs);
            for (idx, (a, b)) in lhs.iter().zip(rhs.iter()).enumerate() {
                assert!((a - b).abs() < 1e-6, "tok={tok} idx={idx}: {a} vs {b}");
            }
        }
        assert!(gqa_kv_head_partition(n_kv, n_q, 2, 2).is_none());
    }
}
