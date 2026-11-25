//! Dense embedding pooling, `Matryoshka` truncation, batch cosine similarity, and
//! cross-encoder sigmoid reranking kernels with portable scalar fallbacks and `RVV 1.0` paths.

use jero_core::K3Error;

use crate::machine::{
    bounds_error, check_all_finite, check_exact_len, check_positive_dim, check_positive_finite,
    checked_product,
};

/// Pooling mode code selecting token-axis mean pooling in [`pool_and_l2_normalize_f32`].
pub const EMBED_POOL_MEAN: u8 = 1;

/// Pooling mode code selecting row `0` (`[CLS]`) pooling in [`pool_and_l2_normalize_f32`].
pub const EMBED_POOL_CLS: u8 = 2;

/// Pooling mode code selecting row `seq_len - 1` (`LastToken`) pooling in [`pool_and_l2_normalize_f32`].
pub const EMBED_POOL_LAST_TOKEN: u8 = 3;

fn validate_pool_args(
    out: &[f32],
    hidden_seq: &[f32],
    seq_len: usize,
    hidden_dim: usize,
    mode_code: u8,
    matryoshka_dim: Option<usize>,
) -> Result<usize, K3Error> {
    if !matches!(
        mode_code,
        EMBED_POOL_MEAN | EMBED_POOL_CLS | EMBED_POOL_LAST_TOKEN
    ) {
        return Err(bounds_error(
            "embed_pooling_mode",
            0,
            usize::from(mode_code),
            3,
            format!(
                "invalid embedding pooling mode_code {mode_code}; expected 1 (Mean), 2 (Cls), or 3 (LastToken)"
            ),
        ));
    }

    let seq_len = check_positive_dim("embed_seq_len", seq_len)?;
    let hidden_dim = check_positive_dim("embed_hidden_dim", hidden_dim)?;
    let expected_seq_elems = checked_product(
        "embed_hidden_seq",
        seq_len,
        hidden_dim,
        "seq_len * hidden_dim",
    )?;
    check_exact_len("embed_hidden_seq", hidden_seq.len(), expected_seq_elems)?;

    let effective_dim = match matryoshka_dim {
        None => hidden_dim,
        Some(d) if d == 0 || d > hidden_dim => {
            return Err(bounds_error(
                "embed_matryoshka_dim",
                0,
                d,
                hidden_dim,
                format!("matryoshka_dim ({d}) must lie within 1..={hidden_dim}"),
            ));
        }
        Some(d) => d,
    };

    check_exact_len("embed_out", out.len(), effective_dim)?;
    check_all_finite("embed_hidden_seq", hidden_seq)?;

    Ok(effective_dim)
}

fn l2_normalize_overflow_safe_scalar(out: &mut [f32]) {
    let mut max_abs = 0.0_f32;
    for &val in out.iter() {
        let abs = val.abs();
        if abs > max_abs {
            max_abs = abs;
        }
    }
    if max_abs == 0.0_f32 || !max_abs.is_finite() {
        out.fill(0.0_f32);
        return;
    }

    let max_f64 = f64::from(max_abs);
    let mut sum_sq_scaled = 0.0_f64;
    for &val in out.iter() {
        let scaled = f64::from(val) / max_f64;
        sum_sq_scaled += scaled * scaled;
    }
    let norm = max_f64 * sum_sq_scaled.sqrt();
    if norm == 0.0_f64 || !norm.is_finite() {
        out.fill(0.0_f32);
        return;
    }

    let inv_norm = 1.0_f64 / norm;
    for slot in out.iter_mut() {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "L2-normalized element lies strictly in [-1.0, 1.0]"
        )]
        {
            *slot = (f64::from(*slot) * inv_norm) as f32;
        }
    }
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[allow(unused_assignments)]
fn rvv_add_inplace_f32(dst: &mut [f32], src: &[f32]) {
    let mut rem = dst.len().min(src.len());
    if rem == 0 {
        return;
    }
    let mut pd = dst.as_mut_ptr();
    let mut ps = src.as_ptr();
    // SAFETY: `pd` and `ps` are disjoint valid slices of length `rem` (`dst` is `&mut`).
    unsafe {
        core::arch::asm!(
            "2:",
            "vsetvli t0, {rem}, e32, m4, ta, ma",
            "vle32.v v8, ({pd})",
            "vle32.v v16, ({ps})",
            "vfadd.vv v8, v8, v16",
            "vse32.v v8, ({pd})",
            "slli t1, t0, 2",
            "add {pd}, {pd}, t1",
            "add {ps}, {ps}, t1",
            "sub {rem}, {rem}, t0",
            "bnez {rem}, 2b",
            pd = inout(reg) pd,
            ps = inout(reg) ps,
            rem = inout(reg) rem,
            out("t0") _,
            out("t1") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            options(nostack),
        );
    }
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[allow(unused_assignments)]
fn rvv_scale_inplace_f32(dst: &mut [f32], scale: f32) {
    let mut rem = dst.len();
    if rem == 0 {
        return;
    }
    let mut pd = dst.as_mut_ptr();
    // SAFETY: `pd` points to a valid mutable slice of `rem` `f32` elements.
    unsafe {
        core::arch::asm!(
            "2:",
            "vsetvli t0, {rem}, e32, m4, ta, ma",
            "vle32.v v8, ({pd})",
            "vfmul.vf v8, v8, {scale}",
            "vse32.v v8, ({pd})",
            "slli t1, t0, 2",
            "add {pd}, {pd}, t1",
            "sub {rem}, {rem}, t0",
            "bnez {rem}, 2b",
            pd = inout(reg) pd,
            rem = inout(reg) rem,
            scale = in(freg) scale,
            out("t0") _,
            out("t1") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            options(nostack),
        );
    }
}

/// Pools a `[seq_len, hidden_dim]` hidden-state matrix (`mode_code`: `1`=`Mean`, `2`=`Cls` row 0,
/// `3`=`LastToken` row `seq_len - 1`), truncates to `matryoshka_dim` if provided, and applies safe
/// L2 normalization (zero-norm inputs remain zero without `NaN`).
///
/// Dispatches to `pool_and_l2_normalize_f32_rvv` on `riscv64` targets and
/// [`pool_and_l2_normalize_f32_scalar`] elsewhere or under Miri.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `mode_code` is not `1..=3`, `seq_len` or `hidden_dim`
/// is zero or overflows `usize`, `matryoshka_dim` is `Some(0)` or `> hidden_dim`, slice lengths
/// mismatch, or any element in `hidden_seq` is non-finite.
pub fn pool_and_l2_normalize_f32(
    out: &mut [f32],
    hidden_seq: &[f32],
    seq_len: usize,
    hidden_dim: usize,
    mode_code: u8,
    matryoshka_dim: Option<usize>,
) -> Result<(), K3Error> {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        pool_and_l2_normalize_f32_rvv(
            out,
            hidden_seq,
            seq_len,
            hidden_dim,
            mode_code,
            matryoshka_dim,
        )
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        pool_and_l2_normalize_f32_scalar(
            out,
            hidden_seq,
            seq_len,
            hidden_dim,
            mode_code,
            matryoshka_dim,
        )
    }
}

/// Portable scalar reference implementation of [`pool_and_l2_normalize_f32`].
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] on invalid pooling mode, zero/overflow dimensions,
/// mismatched lengths, invalid `matryoshka_dim`, or non-finite inputs.
pub fn pool_and_l2_normalize_f32_scalar(
    out: &mut [f32],
    hidden_seq: &[f32],
    seq_len: usize,
    hidden_dim: usize,
    mode_code: u8,
    matryoshka_dim: Option<usize>,
) -> Result<(), K3Error> {
    let effective_dim = validate_pool_args(
        out,
        hidden_seq,
        seq_len,
        hidden_dim,
        mode_code,
        matryoshka_dim,
    )?;

    match mode_code {
        EMBED_POOL_MEAN => {
            out.copy_from_slice(&hidden_seq[..effective_dim]);
            for s in 1..seq_len {
                let row = &hidden_seq[s * hidden_dim..s * hidden_dim + effective_dim];
                for (dst, &val) in out.iter_mut().zip(row.iter()) {
                    *dst += val;
                }
            }
            #[expect(
                clippy::cast_precision_loss,
                reason = "sequence length fits comfortably in f32"
            )]
            let inv_seq = 1.0_f32 / (seq_len as f32);
            for dst in out.iter_mut() {
                *dst *= inv_seq;
            }
            if !out.iter().all(|v| v.is_finite()) {
                out.fill(0.0_f32);
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "sequence length fits within f64 exact integer range"
                )]
                let inv_seq_f64 = 1.0_f64 / (seq_len as f64);
                for d in 0..effective_dim {
                    let mut sum = 0.0_f64;
                    for s in 0..seq_len {
                        sum += f64::from(hidden_seq[s * hidden_dim + d]);
                    }
                    #[expect(
                        clippy::cast_possible_truncation,
                        reason = "mean of finite f32 values fits in finite f32"
                    )]
                    {
                        out[d] = (sum * inv_seq_f64) as f32;
                    }
                }
            }
        }
        EMBED_POOL_CLS => {
            out.copy_from_slice(&hidden_seq[..effective_dim]);
        }
        EMBED_POOL_LAST_TOKEN => {
            let start = (seq_len - 1) * hidden_dim;
            out.copy_from_slice(&hidden_seq[start..start + effective_dim]);
        }
        _ => unreachable!("mode_code validated upfront"),
    }

    let sum_sq = crate::dot_product_f32_scalar(out, out);
    if sum_sq > 0.0_f32 && sum_sq.is_finite() {
        let inv_norm = 1.0_f32 / sum_sq.sqrt();
        for slot in out.iter_mut() {
            *slot *= inv_norm;
        }
    } else {
        l2_normalize_overflow_safe_scalar(out);
    }

    Ok(())
}

/// `RVV 1.0` (`VLA` strip-mined) implementation of [`pool_and_l2_normalize_f32`].
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] on invalid pooling mode, zero/overflow dimensions,
/// mismatched lengths, invalid `matryoshka_dim`, or non-finite inputs.
#[cfg(all(target_arch = "riscv64", not(miri)))]
pub fn pool_and_l2_normalize_f32_rvv(
    out: &mut [f32],
    hidden_seq: &[f32],
    seq_len: usize,
    hidden_dim: usize,
    mode_code: u8,
    matryoshka_dim: Option<usize>,
) -> Result<(), K3Error> {
    let effective_dim = validate_pool_args(
        out,
        hidden_seq,
        seq_len,
        hidden_dim,
        mode_code,
        matryoshka_dim,
    )?;

    match mode_code {
        EMBED_POOL_MEAN => {
            out.copy_from_slice(&hidden_seq[..effective_dim]);
            for s in 1..seq_len {
                let row = &hidden_seq[s * hidden_dim..s * hidden_dim + effective_dim];
                rvv_add_inplace_f32(out, row);
            }
            #[expect(
                clippy::cast_precision_loss,
                reason = "sequence length fits comfortably in f32"
            )]
            let inv_seq = 1.0_f32 / (seq_len as f32);
            rvv_scale_inplace_f32(out, inv_seq);
            if !out.iter().all(|v| v.is_finite()) {
                out.fill(0.0_f32);
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "sequence length fits within f64 exact integer range"
                )]
                let inv_seq_f64 = 1.0_f64 / (seq_len as f64);
                for d in 0..effective_dim {
                    let mut sum = 0.0_f64;
                    for s in 0..seq_len {
                        sum += f64::from(hidden_seq[s * hidden_dim + d]);
                    }
                    #[expect(
                        clippy::cast_possible_truncation,
                        reason = "mean of finite f32 values fits in finite f32"
                    )]
                    {
                        out[d] = (sum * inv_seq_f64) as f32;
                    }
                }
            }
        }
        EMBED_POOL_CLS => {
            out.copy_from_slice(&hidden_seq[..effective_dim]);
        }
        EMBED_POOL_LAST_TOKEN => {
            let start = (seq_len - 1) * hidden_dim;
            out.copy_from_slice(&hidden_seq[start..start + effective_dim]);
        }
        _ => unreachable!("mode_code validated upfront"),
    }

    let sum_sq = crate::dot_product_f32_rvv(out, out);
    if sum_sq > 0.0_f32 && sum_sq.is_finite() {
        let inv_norm = 1.0_f32 / sum_sq.sqrt();
        rvv_scale_inplace_f32(out, inv_norm);
    } else {
        l2_normalize_overflow_safe_scalar(out);
    }

    Ok(())
}

fn validate_cosine_batch_args(
    out_scores: &[f32],
    query: &[f32],
    docs: &[f32],
    num_docs: usize,
    dim: usize,
) -> Result<(), K3Error> {
    let num_docs = check_positive_dim("cosine_num_docs", num_docs)?;
    let dim = check_positive_dim("cosine_dim", dim)?;
    let docs_elems = checked_product("cosine_docs", num_docs, dim, "num_docs * dim")?;

    check_exact_len("cosine_query", query.len(), dim)?;
    check_exact_len("cosine_docs", docs.len(), docs_elems)?;
    check_exact_len("cosine_out_scores", out_scores.len(), num_docs)?;

    check_all_finite("cosine_query", query)?;
    check_all_finite("cosine_docs", docs)?;

    Ok(())
}

fn cosine_overflow_safe_f64(query: &[f32], doc: &[f32]) -> f32 {
    let max_q = query.iter().fold(0.0_f32, |m, &x| m.max(x.abs()));
    let max_d = doc.iter().fold(0.0_f32, |m, &x| m.max(x.abs()));
    if max_q == 0.0_f32 || max_d == 0.0_f32 {
        return 0.0_f32;
    }
    let inv_q = 1.0_f64 / f64::from(max_q);
    let inv_d = 1.0_f64 / f64::from(max_d);

    let mut dot = 0.0_f64;
    let mut nq = 0.0_f64;
    let mut nd = 0.0_f64;
    for (&q, &d) in query.iter().zip(doc.iter()) {
        let qs = f64::from(q) * inv_q;
        let ds = f64::from(d) * inv_d;
        dot += qs * ds;
        nq += qs * qs;
        nd += ds * ds;
    }
    let denom = nq.sqrt() * nd.sqrt();
    if denom == 0.0_f64 || !denom.is_finite() {
        return 0.0_f32;
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "clamped cosine similarity lies strictly in [-1.0, 1.0]"
    )]
    {
        (dot / denom).clamp(-1.0_f64, 1.0_f64) as f32
    }
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[allow(unused_assignments)]
fn rvv_dot_and_norm_sq_f32(query: &[f32], doc: &[f32]) -> (f32, f32) {
    let mut rem = query.len().min(doc.len());
    if rem == 0 {
        return (0.0_f32, 0.0_f32);
    }
    let mut pq = query.as_ptr();
    let mut pd = doc.as_ptr();
    let cross_dot: f32;
    let doc_norm_sq: f32;
    // SAFETY: `pq` and `pd` point to valid contiguous slices of `rem` `f32` elements.
    // Two independent `e32, m4` accumulators (`v0` for `q*d` and `v4` for `d*d`) execute in one
    // memory pass using `tu, ma` tail-undisturbed accumulation and `vfredusum.vs` reduction.
    unsafe {
        core::arch::asm!(
            "vsetvli t0, zero, e32, m4, ta, ma",
            "vmv.v.i v0, 0",
            "vmv.v.i v4, 0",
            "2:",
            "vsetvli t1, {rem}, e32, m4, tu, ma",
            "vle32.v v8, ({pq})",
            "vle32.v v12, ({pd})",
            "vfmacc.vv v0, v8, v12",
            "vfmacc.vv v4, v12, v12",
            "slli t2, t1, 2",
            "add {pq}, {pq}, t2",
            "add {pd}, {pd}, t2",
            "sub {rem}, {rem}, t1",
            "bnez {rem}, 2b",
            "vsetivli zero, 1, e32, m1, ta, ma",
            "vmv.v.i v24, 0",
            "vmv.v.i v25, 0",
            "vsetvli zero, t0, e32, m4, ta, ma",
            "vfredusum.vs v24, v0, v24",
            "vfredusum.vs v25, v4, v25",
            "vsetivli zero, 1, e32, m1, ta, ma",
            "vfmv.f.s {cross_dot}, v24",
            "vfmv.f.s {doc_norm_sq}, v25",
            pq = inout(reg) pq,
            pd = inout(reg) pd,
            rem = inout(reg) rem,
            cross_dot = out(freg) cross_dot,
            doc_norm_sq = out(freg) doc_norm_sq,
            out("t0") _, out("t1") _, out("t2") _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            out("v24") _, out("v25") _,
            options(nostack, readonly),
        );
    }
    (cross_dot, doc_norm_sq)
}

/// Computes cosine similarity between `query` (`[dim]`) and `num_docs` rows of `docs` (`[num_docs, dim]`),
/// writing results to `out_scores` (`[num_docs]`). Zero-norm vectors yield `0.0` without `NaN`.
///
/// Dispatches to `cosine_similarity_batch_f32_rvv` on `riscv64` targets and
/// [`cosine_similarity_batch_f32_scalar`] elsewhere or under Miri.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] on zero/overflow dimensions, mismatched slice lengths,
/// or non-finite inputs.
pub fn cosine_similarity_batch_f32(
    out_scores: &mut [f32],
    query: &[f32],
    docs: &[f32],
    num_docs: usize,
    dim: usize,
) -> Result<(), K3Error> {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        cosine_similarity_batch_f32_rvv(out_scores, query, docs, num_docs, dim)
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        cosine_similarity_batch_f32_scalar(out_scores, query, docs, num_docs, dim)
    }
}

/// Portable scalar reference implementation of [`cosine_similarity_batch_f32`] using `f64`
/// accumulation for reference precision.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] on zero/overflow dimensions, mismatched slice lengths,
/// or non-finite inputs.
pub fn cosine_similarity_batch_f32_scalar(
    out_scores: &mut [f32],
    query: &[f32],
    docs: &[f32],
    num_docs: usize,
    dim: usize,
) -> Result<(), K3Error> {
    validate_cosine_batch_args(out_scores, query, docs, num_docs, dim)?;

    let q_norm_sq: f64 = query.iter().map(|&q| f64::from(q) * f64::from(q)).sum();
    if q_norm_sq == 0.0_f64 {
        out_scores.fill(0.0_f32);
        return Ok(());
    }
    if !q_norm_sq.is_finite() {
        for (doc_idx, score_slot) in out_scores.iter_mut().enumerate() {
            let doc = &docs[doc_idx * dim..(doc_idx + 1) * dim];
            *score_slot = cosine_overflow_safe_f64(query, doc);
        }
        return Ok(());
    }
    let q_norm = q_norm_sq.sqrt();

    for (doc_idx, score_slot) in out_scores.iter_mut().enumerate() {
        let doc = &docs[doc_idx * dim..(doc_idx + 1) * dim];
        let mut dot = 0.0_f64;
        let mut d_norm_sq = 0.0_f64;
        for (&q, &d) in query.iter().zip(doc.iter()) {
            let q64 = f64::from(q);
            let d64 = f64::from(d);
            dot += q64 * d64;
            d_norm_sq += d64 * d64;
        }
        if d_norm_sq == 0.0_f64 {
            *score_slot = 0.0_f32;
        } else if !d_norm_sq.is_finite() || !dot.is_finite() {
            *score_slot = cosine_overflow_safe_f64(query, doc);
        } else {
            let denom = q_norm * d_norm_sq.sqrt();
            #[expect(
                clippy::cast_possible_truncation,
                reason = "clamped cosine similarity lies strictly in [-1.0, 1.0]"
            )]
            {
                *score_slot = if denom > 0.0_f64 && denom.is_finite() {
                    (dot / denom).clamp(-1.0_f64, 1.0_f64) as f32
                } else {
                    0.0_f32
                };
            }
        }
    }

    Ok(())
}

/// `RVV 1.0` (`VLA` single-pass `vfmacc.vv` + `vfredusum.vs`) implementation of
/// [`cosine_similarity_batch_f32`].
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] on zero/overflow dimensions, mismatched slice lengths,
/// or non-finite inputs.
#[cfg(all(target_arch = "riscv64", not(miri)))]
pub fn cosine_similarity_batch_f32_rvv(
    out_scores: &mut [f32],
    query: &[f32],
    docs: &[f32],
    num_docs: usize,
    dim: usize,
) -> Result<(), K3Error> {
    validate_cosine_batch_args(out_scores, query, docs, num_docs, dim)?;

    let q_norm_sq = crate::dot_product_f32_rvv(query, query);
    if q_norm_sq == 0.0_f32 {
        let q_max_abs = query.iter().fold(0.0_f32, |m, &x| m.max(x.abs()));
        if q_max_abs == 0.0_f32 {
            out_scores.fill(0.0_f32);
            return Ok(());
        }
    }
    let q_norm_f64 = f64::from(q_norm_sq).sqrt();

    for (doc_idx, score_slot) in out_scores.iter_mut().enumerate() {
        let doc = &docs[doc_idx * dim..(doc_idx + 1) * dim];
        let (dot_qd, d_norm_sq) = rvv_dot_and_norm_sq_f32(query, doc);
        if q_norm_sq > 0.0_f32
            && q_norm_sq.is_finite()
            && d_norm_sq > 0.0_f32
            && d_norm_sq.is_finite()
            && dot_qd.is_finite()
        {
            let denom = q_norm_f64 * f64::from(d_norm_sq).sqrt();
            #[expect(
                clippy::cast_possible_truncation,
                reason = "clamped cosine similarity lies strictly in [-1.0, 1.0]"
            )]
            {
                *score_slot = if denom > 0.0_f64 && denom.is_finite() {
                    (f64::from(dot_qd) / denom).clamp(-1.0_f64, 1.0_f64) as f32
                } else {
                    0.0_f32
                };
            }
        } else {
            *score_slot = cosine_overflow_safe_f64(query, doc);
        }
    }

    Ok(())
}

fn validate_rerank_sigmoid_args(out: &[f32], logits: &[f32], scale: f32) -> Result<(), K3Error> {
    let len = check_positive_dim("rerank_logits", logits.len())?;
    check_exact_len("rerank_out", out.len(), len)?;
    check_positive_finite("rerank_scale", scale)?;
    check_all_finite("rerank_logits", logits)?;
    Ok(())
}

/// Applies numerically stable scaled sigmoid `1.0 / (1.0 + exp(-(logit * scale)))` for
/// cross-encoder reranking, guaranteeing outputs strictly within `[0.0, 1.0]` without `NaN`.
///
/// Dispatches to `rerank_sigmoid_scores_f32_rvv` on `riscv64` targets and
/// [`rerank_sigmoid_scores_f32_scalar`] elsewhere or under Miri.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `logits` is empty, `out.len() != logits.len()`,
/// `scale <= 0.0` or non-finite, or any logit is non-finite.
pub fn rerank_sigmoid_scores_f32(
    out: &mut [f32],
    logits: &[f32],
    scale: f32,
) -> Result<(), K3Error> {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        rerank_sigmoid_scores_f32_rvv(out, logits, scale)
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        rerank_sigmoid_scores_f32_scalar(out, logits, scale)
    }
}

/// Portable scalar reference implementation of [`rerank_sigmoid_scores_f32`].
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] on invalid lengths, invalid `scale`, or non-finite inputs.
pub fn rerank_sigmoid_scores_f32_scalar(
    out: &mut [f32],
    logits: &[f32],
    scale: f32,
) -> Result<(), K3Error> {
    validate_rerank_sigmoid_args(out, logits, scale)?;

    for (dst, &logit) in out.iter_mut().zip(logits.iter()) {
        let z = logit * scale;
        *dst = if z >= 88.0_f32 {
            1.0_f32
        } else if z <= -88.0_f32 {
            0.0_f32
        } else if z >= 0.0_f32 {
            (1.0_f32 / (1.0_f32 + (-z).exp())).clamp(0.0_f32, 1.0_f32)
        } else {
            let e = z.exp();
            (e / (1.0_f32 + e)).clamp(0.0_f32, 1.0_f32)
        };
    }

    Ok(())
}

/// `RVV 1.0` (`VLA` `e32, m1` Cody-Waite `exp(-|z|)` sigmoid) implementation of
/// [`rerank_sigmoid_scores_f32`].
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] on invalid lengths, invalid `scale`, or non-finite inputs.
#[cfg(all(target_arch = "riscv64", not(miri)))]
#[allow(unused_assignments)]
pub fn rerank_sigmoid_scores_f32_rvv(
    out: &mut [f32],
    logits: &[f32],
    scale: f32,
) -> Result<(), K3Error> {
    const EXP_CONSTS: [f32; 9] = [
        -87.33_f32,
        core::f32::consts::LOG2_E,
        0.693_145_75_f32,
        1.428_606_8e-6_f32,
        1.0_f32 / 720.0_f32,
        1.0_f32 / 120.0_f32,
        1.0_f32 / 24.0_f32,
        1.0_f32 / 6.0_f32,
        0.5_f32,
    ];

    validate_rerank_sigmoid_args(out, logits, scale)?;

    let mut pin = logits.as_ptr();
    let mut pout = out.as_mut_ptr();
    let mut rem = logits.len();
    let pc = EXP_CONSTS.as_ptr();

    // SAFETY: `pin` and `pout` point to disjoint valid slices of length `rem`.
    // Computes `a = exp(-|z|)` via Cody-Waite degree-6 approximation on `[-87.33, 0.0]`, then
    // evaluates `1 / (1 + a)` for `z >= 0` or `a / (1 + a)` for `z < 0`, clamped to `[0.0, 1.0]`.
    unsafe {
        core::arch::asm!(
            "flw ft0, 0({c})",
            "flw ft1, 4({c})",
            "flw ft2, 8({c})",
            "flw ft3, 12({c})",
            "li t2, 127",
            "2:",
            "vsetvli t0, {rem}, e32, m1, ta, ma",
            "vle32.v v8, ({pin})",
            "vfmul.vf v8, v8, {scale}",
            "vfsgnjx.vv v9, v8, v8",
            "vfsgnjn.vv v9, v9, v9",
            "vmflt.vf v0, v9, ft0",
            "vfmax.vf v9, v9, ft0",
            "vfmul.vf v10, v9, ft1",
            "vfcvt.x.f.v v11, v10",
            "vfcvt.f.x.v v10, v11",
            "vfmul.vf v12, v10, ft2",
            "vfsub.vv v9, v9, v12",
            "vfmul.vf v12, v10, ft3",
            "vfsub.vv v9, v9, v12",
            "flw ft4, 16({c})",
            "vfmv.v.f v10, ft4",
            "flw ft5, 20({c})",
            "vfmul.vv v10, v10, v9",
            "vfadd.vf v10, v10, ft5",
            "flw ft4, 24({c})",
            "vfmul.vv v10, v10, v9",
            "vfadd.vf v10, v10, ft4",
            "flw ft5, 28({c})",
            "vfmul.vv v10, v10, v9",
            "vfadd.vf v10, v10, ft5",
            "flw ft4, 32({c})",
            "vfmul.vv v10, v10, v9",
            "vfadd.vf v10, v10, ft4",
            "vfmul.vv v10, v10, v9",
            "vfadd.vf v10, v10, {one}",
            "vfmul.vv v10, v10, v9",
            "vfadd.vf v10, v10, {one}",
            "vadd.vx v11, v11, t2",
            "vsll.vi v11, v11, 23",
            "vfmul.vv v10, v10, v11",
            "vfmerge.vfm v10, v10, {zero}, v0",
            "vmflt.vf v0, v8, {zero}",
            "vfmv.v.f v11, {one}",
            "vmerge.vvm v11, v11, v10, v0",
            "vfadd.vf v10, v10, {one}",
            "vfdiv.vv v11, v11, v10",
            "vfmax.vf v11, v11, {zero}",
            "vfmin.vf v11, v11, {one}",
            "vse32.v v11, ({pout})",
            "slli t1, t0, 2",
            "add {pin}, {pin}, t1",
            "add {pout}, {pout}, t1",
            "sub {rem}, {rem}, t0",
            "bnez {rem}, 2b",
            pin = inout(reg) pin,
            pout = inout(reg) pout,
            rem = inout(reg) rem,
            c = in(reg) pc,
            scale = in(freg) scale,
            one = in(freg) 1.0_f32,
            zero = in(freg) 0.0_f32,
            out("t0") _, out("t1") _, out("t2") _,
            out("ft0") _, out("ft1") _, out("ft2") _, out("ft3") _, out("ft4") _, out("ft5") _,
            out("v0") _, out("v8") _, out("v9") _, out("v10") _, out("v11") _, out("v12") _,
            options(nostack),
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_and_l2_normalize_modes_and_matryoshka() {
        // 3 tokens x 4 dims
        let seq = [
            3.0_f32, 4.0, 10.0, 20.0, // row 0 (CLS)
            1.0, 2.0, -4.0, 6.0, // row 1
            -1.0, 6.0, 9.0, -5.0, // row 2 (LastToken)
        ];

        // Mean over all 4 dims: mean = [1.0, 4.0, 5.0, 7.0]
        let mut mean_out = [0.0_f32; 4];
        pool_and_l2_normalize_f32(&mut mean_out, &seq, 3, 4, EMBED_POOL_MEAN, None).unwrap();
        let norm_mean = (1.0_f32 + 16.0 + 25.0 + 49.0).sqrt();
        for (i, &exp) in [1.0_f32, 4.0, 5.0, 7.0].iter().enumerate() {
            assert!((mean_out[i] - exp / norm_mean).abs() < 1e-5);
        }
        let sq_sum: f32 = mean_out.iter().map(|v| v * v).sum();
        assert!((sq_sum - 1.0_f32).abs() < 1e-5);

        // CLS with Matryoshka truncation to first 2 dims: [3.0, 4.0] -> [0.6, 0.8]
        let mut cls_m = [0.0_f32; 2];
        pool_and_l2_normalize_f32(&mut cls_m, &seq, 3, 4, EMBED_POOL_CLS, Some(2)).unwrap();
        assert!((cls_m[0] - 0.6_f32).abs() < 1e-5);
        assert!((cls_m[1] - 0.8_f32).abs() < 1e-5);

        // LastToken full 4 dims
        let mut last_out = [0.0_f32; 4];
        pool_and_l2_normalize_f32(&mut last_out, &seq, 3, 4, EMBED_POOL_LAST_TOKEN, None).unwrap();
        let norm_last = (1.0_f32 + 36.0 + 81.0 + 25.0).sqrt();
        for (i, &exp) in [-1.0_f32, 6.0, 9.0, -5.0].iter().enumerate() {
            assert!((last_out[i] - exp / norm_last).abs() < 1e-5);
        }

        // Zero-norm vector stays exact zero without NaN, and extreme scales normalize cleanly
        let zeros = [0.0_f32; 8];
        let mut zero_out = [9.0_f32; 4];
        pool_and_l2_normalize_f32(&mut zero_out, &zeros, 2, 4, EMBED_POOL_MEAN, None).unwrap();
        assert_eq!(zero_out, [0.0_f32; 4]);

        let huge = [1e25_f32, 0.0, 0.0, 0.0];
        let mut huge_out = [0.0_f32; 4];
        pool_and_l2_normalize_f32(&mut huge_out, &huge, 1, 4, EMBED_POOL_CLS, None).unwrap();
        assert!((huge_out[0] - 1.0_f32).abs() < 1e-5);
        assert_eq!(huge_out[1..], [0.0_f32; 3]);
    }

    #[test]
    fn test_pool_and_l2_normalize_fail_closed() {
        let seq = [1.0_f32; 6];
        let mut out = [7.0_f32; 3];
        // Invalid mode codes
        for bad_mode in [0_u8, 4, 255] {
            assert!(matches!(
                pool_and_l2_normalize_f32(&mut out, &seq, 2, 3, bad_mode, None),
                Err(K3Error::TensorBoundsError { .. })
            ));
        }
        // Invalid matryoshka bounds
        assert!(matches!(
            pool_and_l2_normalize_f32(&mut [], &seq, 2, 3, EMBED_POOL_CLS, Some(0)),
            Err(K3Error::TensorBoundsError { .. })
        ));
        assert!(matches!(
            pool_and_l2_normalize_f32(&mut out, &seq, 2, 3, EMBED_POOL_CLS, Some(4)),
            Err(K3Error::TensorBoundsError { .. })
        ));
        // Zero / overflow / mismatched lengths
        assert!(matches!(
            pool_and_l2_normalize_f32(&mut out, &seq, 0, 3, EMBED_POOL_MEAN, None),
            Err(K3Error::TensorBoundsError { .. })
        ));
        assert!(matches!(
            pool_and_l2_normalize_f32(&mut out, &seq, usize::MAX, 2, EMBED_POOL_MEAN, None),
            Err(K3Error::TensorBoundsError { .. })
        ));
        assert!(matches!(
            pool_and_l2_normalize_f32(&mut out[..2], &seq, 2, 3, EMBED_POOL_MEAN, None),
            Err(K3Error::TensorBoundsError { .. })
        ));
        // Non-finite input leaves output untouched
        let bad_seq = [1.0_f32, 2.0, 3.0, 4.0, f32::NAN, 6.0];
        assert!(matches!(
            pool_and_l2_normalize_f32(&mut out, &bad_seq, 2, 3, EMBED_POOL_CLS, None),
            Err(K3Error::TensorBoundsError { .. })
        ));
        assert_eq!(out, [7.0_f32; 3]);
    }

    #[test]
    fn test_cosine_similarity_batch_parity_and_edge_cases() {
        let dim = 37;
        let num_docs = 5;
        let query: Vec<f32> = (0..dim)
            .map(|i| {
                #[expect(clippy::cast_precision_loss, reason = "small test index")]
                {
                    (i as f32 - 18.0) * 0.125
                }
            })
            .collect();
        let mut docs = vec![0.0_f32; num_docs * dim];
        // doc 0: identical to query -> 1.0
        docs[..dim].copy_from_slice(&query);
        // doc 1: negated query -> -1.0
        for (j, &q) in query.iter().enumerate() {
            docs[dim + j] = -2.5 * q;
        }
        // doc 2: all zero -> 0.0 without NaN
        // doc 3: mixed values
        for j in 0..dim {
            #[expect(clippy::cast_precision_loss, reason = "small test index")]
            {
                docs[3 * dim + j] = ((j * 7) % 13) as f32 - 6.0;
            }
        }
        // doc 4: huge scale avoiding overflow
        docs[4 * dim] = 1e25_f32;

        let mut scores = [0.0_f32; 5];
        let mut scores_scalar = [0.0_f32; 5];
        cosine_similarity_batch_f32(&mut scores, &query, &docs, num_docs, dim).unwrap();
        cosine_similarity_batch_f32_scalar(&mut scores_scalar, &query, &docs, num_docs, dim)
            .unwrap();

        for i in 0..num_docs {
            assert!(scores[i].is_finite());
            assert!((scores[i] - scores_scalar[i]).abs() < 1e-5);
        }
        assert!((scores[0] - 1.0_f32).abs() < 1e-5);
        assert!((scores[1] + 1.0_f32).abs() < 1e-5);
        assert_eq!(scores[2], 0.0_f32);

        // Zero query returns all zeros
        let zero_q = vec![0.0_f32; dim];
        let mut z_scores = [9.0_f32; 5];
        cosine_similarity_batch_f32(&mut z_scores, &zero_q, &docs, num_docs, dim).unwrap();
        assert_eq!(z_scores, [0.0_f32; 5]);

        // Bounds & NaN rejection
        assert!(matches!(
            cosine_similarity_batch_f32(&mut z_scores, &[f32::INFINITY; 37], &docs, num_docs, dim),
            Err(K3Error::TensorBoundsError { .. })
        ));
        assert!(matches!(
            cosine_similarity_batch_f32(&mut z_scores[..4], &query, &docs, num_docs, dim),
            Err(K3Error::TensorBoundsError { .. })
        ));
    }

    #[test]
    fn test_rerank_sigmoid_scores_extremes_and_validation() {
        let logits = [-1000.0_f32, -88.0, -5.0, 0.0, 2.5, 88.0, 1000.0];
        let mut out = [0.0_f32; 7];
        let mut out_scalar = [0.0_f32; 7];
        rerank_sigmoid_scores_f32(&mut out, &logits, 1.5).unwrap();
        rerank_sigmoid_scores_f32_scalar(&mut out_scalar, &logits, 1.5).unwrap();

        for i in 0..logits.len() {
            assert!(out[i].is_finite());
            assert!((0.0_f32..=1.0_f32).contains(&out[i]));
            assert!((out[i] - out_scalar[i]).abs() < 1e-5);
        }
        assert_eq!(out[0], 0.0_f32);
        assert!((out[3] - 0.5_f32).abs() < 1e-6);
        assert_eq!(out[6], 1.0_f32);

        // Invalid scale, empty slice, mismatch, and NaN fail closed
        assert!(matches!(
            rerank_sigmoid_scores_f32(&mut out, &logits, 0.0),
            Err(K3Error::TensorBoundsError { .. })
        ));
        assert!(matches!(
            rerank_sigmoid_scores_f32(&mut out, &logits, f32::NAN),
            Err(K3Error::TensorBoundsError { .. })
        ));
        assert!(matches!(
            rerank_sigmoid_scores_f32(&mut [], &[], 1.0),
            Err(K3Error::TensorBoundsError { .. })
        ));
        assert!(matches!(
            rerank_sigmoid_scores_f32(&mut out, &[f32::NAN; 7], 1.0),
            Err(K3Error::TensorBoundsError { .. })
        ));
    }
}
