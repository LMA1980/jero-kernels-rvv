//! Grouped-Query Attention (`GQA`) score calculation and weighted value accumulation.
//!
//! Supports Multi-Head Attention (`n_heads_q == n_heads_kv`), Multi-Query Attention
//! (`n_heads_kv == 1`), and Grouped-Query Attention (`n_heads_q % n_heads_kv == 0`)
//! with numerically stable softmax and token-interleaved KV cache indexing.

use jero_core::K3Error;

#[cfg(all(target_arch = "riscv64", not(miri)))]
use crate::dot_product_f32_rvv;
#[cfg(any(not(target_arch = "riscv64"), miri))]
use crate::dot_product_f32_scalar;

#[inline]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
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
fn axpy_f32(out: &mut [f32], x: &[f32], alpha: f32) {
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

/// Computes Grouped-Query Attention (`GQA`) for a single query token across a sequence of cached keys and values.
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
#[expect(
    clippy::too_many_lines,
    reason = "GQA score calculation and softmax accumulation loop"
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
    if n_heads_q == 0 || n_heads_kv == 0 || head_dim == 0 || seq_len == 0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: "gqa".to_owned(),
            reason: format!(
                "GQA dimensions must be non-zero (n_heads_q={n_heads_q}, n_heads_kv={n_heads_kv}, head_dim={head_dim}, seq_len={seq_len})"
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

    let expected_q_len =
        n_heads_q
            .checked_mul(head_dim)
            .ok_or_else(|| K3Error::TensorBoundsError {
                tensor_name: "gqa_q".to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: usize::MAX,
                reason: "q dimension overflow".to_owned(),
            })?;

    if q.len() != expected_q_len {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "gqa_q".to_owned(),
            offset: 0,
            byte_len: size_of_val(q),
            buffer_len: expected_q_len * size_of::<f32>(),
            reason: format!("query length {} != expected {expected_q_len}", q.len()),
        });
    }

    if out.len() != expected_q_len {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "gqa_out".to_owned(),
            offset: 0,
            byte_len: size_of_val(out),
            buffer_len: expected_q_len * size_of::<f32>(),
            reason: format!("output length {} != expected {expected_q_len}", out.len()),
        });
    }

    let expected_kv_len = seq_len
        .checked_mul(n_heads_kv)
        .and_then(|v| v.checked_mul(head_dim))
        .ok_or_else(|| K3Error::TensorBoundsError {
            tensor_name: "gqa_kv".to_owned(),
            offset: 0,
            byte_len: 0,
            buffer_len: usize::MAX,
            reason: "kv dimension overflow".to_owned(),
        })?;

    if k_cache.len() < expected_kv_len {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "gqa_k_cache".to_owned(),
            offset: 0,
            byte_len: size_of_val(k_cache),
            buffer_len: expected_kv_len * size_of::<f32>(),
            reason: format!(
                "k_cache length {} < expected {expected_kv_len}",
                k_cache.len()
            ),
        });
    }

    if v_cache.len() < expected_kv_len {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "gqa_v_cache".to_owned(),
            offset: 0,
            byte_len: size_of_val(v_cache),
            buffer_len: expected_kv_len * size_of::<f32>(),
            reason: format!(
                "v_cache length {} < expected {expected_kv_len}",
                v_cache.len()
            ),
        });
    }

    let group_size = n_heads_q / n_heads_kv;
    let mut scores = vec![0.0_f32; seq_len];

    for h_q in 0..n_heads_q {
        let h_kv = h_q / group_size;
        let q_head = &q[h_q * head_dim..(h_q + 1) * head_dim];

        // 1. Scaled dot product score across cached tokens
        let mut max_score = f32::NEG_INFINITY;
        for (s, score_slot) in scores.iter_mut().enumerate().take(seq_len) {
            let kv_offset = (s * n_heads_kv + h_kv) * head_dim;
            let k_slice = &k_cache[kv_offset..kv_offset + head_dim];
            let raw_dot = dot_f32(q_head, k_slice);
            let sc = raw_dot * scale;
            *score_slot = sc;
            if sc > max_score {
                max_score = sc;
            }
        }

        // 2. Numerically stable softmax
        let mut sum_exp = 0.0_f32;
        for score in &mut scores[..seq_len] {
            let exp_val = (*score - max_score).exp();
            *score = exp_val;
            sum_exp += exp_val;
        }

        #[expect(
            clippy::cast_precision_loss,
            reason = "seq_len fits comfortably in f32"
        )]
        let inv_sum = if sum_exp > 0.0 && sum_exp.is_finite() {
            1.0_f32 / sum_exp
        } else {
            1.0_f32 / (seq_len as f32)
        };

        for score in &mut scores[..seq_len] {
            *score *= inv_sum;
        }

        // 3. Weighted accumulation into output head
        let out_head = &mut out[h_q * head_dim..(h_q + 1) * head_dim];
        out_head.fill(0.0);

        for (s, &weight) in scores.iter().enumerate().take(seq_len) {
            if weight == 0.0 {
                continue;
            }
            let kv_offset = (s * n_heads_kv + h_kv) * head_dim;
            let v_slice = &v_cache[kv_offset..kv_offset + head_dim];
            axpy_f32(out_head, v_slice, weight);
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
}
