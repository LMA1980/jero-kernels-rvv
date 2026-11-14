//! Wave 3 integration verification for zero-spill fused RVV/IME GEMM (`#85`, `#86`),
//! compact nibble storage (`#88`), lock-free parallel repacking (`#89`), and multi-token
//! GQA prefill with SWA, softcap, and KV-head parallelism (`#92`, `#93`).

use std::time::Instant;

use jero_core::{QuantType, TensorView};
use jero_kernels_rvv::{
    Aligned128, AttentionMaskConfig, PackedNibbleMatrix, QuantizedActQ8, RepackedWeight,
    gemm_q4_0_ime, gemm_q4_k_ime, gemm_repacked, gemm_repacked_rows, gemm_repacked_vla_sim,
    gqa_causal_attention_prefill_f32, gqa_causal_attention_prefill_kv_heads_f32,
    gqa_causal_attention_prefill_strided_kv_f32, gqa_causal_attention_with_scratch_f32,
    gqa_kv_head_partition, has_ime, repack_into_tile_range, repack_nibble_into_tile_range,
    repack_q4_k_m_parallel_slices, vlen_bytes,
};

fn small_f32(val: usize, modulo: usize, offset: f32, scale: f32) -> f32 {
    let rem = u16::try_from(val % modulo).expect("modulo fits in u16");
    (f32::from(rem) - offset) * scale
}

fn fill_pseudo_quant_tensor(quant: QuantType, rows: usize, cols: usize) -> Vec<u8> {
    let block_elems = quant.block_size();
    let block_bytes = quant.block_bytes();
    let n_blocks = (rows * cols) / block_elems;
    let mut raw = vec![0_u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let off = b * block_bytes;
        let blk = &mut raw[off..off + block_bytes];
        match quant {
            QuantType::Q8_0 => {
                blk[0..2].copy_from_slice(&0x3400_u16.to_le_bytes()); // 0.25
                for (k, slot) in blk[2..].iter_mut().enumerate() {
                    let signed = i8::try_from((b * 7 + k * 5) % 21).expect("i8") - 10;
                    *slot = signed.to_le_bytes()[0];
                }
            }
            QuantType::Q4_0 | QuantType::IQ4_NL => {
                blk[0..2].copy_from_slice(&0x3800_u16.to_le_bytes()); // 0.5
                for (k, slot) in blk[2..].iter_mut().enumerate() {
                    let lo = u8::try_from((b + k * 3) & 0x0F).expect("nibble");
                    let hi = u8::try_from((b * 3 + k * 5 + 1) & 0x0F).expect("nibble");
                    *slot = lo | (hi << 4);
                }
            }
            QuantType::Q4_1 => {
                blk[0..2].copy_from_slice(&0x3800_u16.to_le_bytes()); // 0.5
                blk[2..4].copy_from_slice(&0x3400_u16.to_le_bytes()); // 0.25
                for (k, slot) in blk[4..].iter_mut().enumerate() {
                    let lo = u8::try_from((b + k * 3) & 0x0F).expect("nibble");
                    let hi = u8::try_from((b * 5 + k + 2) & 0x0F).expect("nibble");
                    *slot = lo | (hi << 4);
                }
            }
            QuantType::Q4_K_M => {
                blk[0..2].copy_from_slice(&0x3400_u16.to_le_bytes());
                blk[2..4].copy_from_slice(&0x3000_u16.to_le_bytes());
                for (k, slot) in blk[4..16].iter_mut().enumerate() {
                    *slot = u8::try_from((b * 5 + k * 3 + 1) & 0x3F).expect("6-bit");
                }
                for (k, slot) in blk[16..].iter_mut().enumerate() {
                    *slot = u8::try_from((b * 7 + k * 11 + 3) & 0xFF).expect("byte");
                }
            }
            QuantType::Q5_K => {
                blk[0..2].copy_from_slice(&0x3400_u16.to_le_bytes());
                blk[2..4].copy_from_slice(&0x3000_u16.to_le_bytes());
                for (k, slot) in blk[4..16].iter_mut().enumerate() {
                    *slot = u8::try_from((b * 3 + k * 5 + 1) & 0x3F).expect("6-bit");
                }
                for (k, slot) in blk[16..].iter_mut().enumerate() {
                    *slot = u8::try_from((b * 9 + k * 7 + 5) & 0xFF).expect("byte");
                }
            }
            QuantType::Q6_K => {
                for (k, slot) in blk[..192].iter_mut().enumerate() {
                    *slot = u8::try_from((b * 11 + k * 7 + 3) & 0xFF).expect("byte");
                }
                for (k, slot) in blk[192..208].iter_mut().enumerate() {
                    let signed = i8::try_from((b * 3 + k * 5) % 15).expect("i8") - 7;
                    *slot = signed.to_le_bytes()[0];
                }
                blk[208..210].copy_from_slice(&0x3000_u16.to_le_bytes());
            }
            QuantType::TQ2_0 => {
                for (k, slot) in blk[..64].iter_mut().enumerate() {
                    let t0 = (b + k) % 3;
                    let t1 = (b + k * 2 + 1) % 3;
                    let t2 = (b * 2 + k + 2) % 3;
                    let t3 = (b * 3 + k) % 3;
                    *slot = u8::try_from(t0 | (t1 << 2) | (t2 << 4) | (t3 << 6)).expect("trits");
                }
                blk[64..66].copy_from_slice(&0x3800_u16.to_le_bytes());
            }
            _ => unreachable!("unsupported test format"),
        }
    }
    raw
}

fn naive_f32_gemm_rows(
    weight: &RepackedWeight,
    x: &[f32],
    m_tokens: usize,
    row_range: std::ops::Range<usize>,
) -> Vec<f32> {
    let cols = weight.cols();
    let sub_n = row_range.len();
    let mut row_w = vec![0.0_f32; cols];
    let mut out = vec![0.0_f32; m_tokens * sub_n];
    for (local_n, r) in row_range.enumerate() {
        weight.dequantize_row(r, &mut row_w).unwrap();
        for tok in 0..m_tokens {
            let x_row = &x[tok * cols..(tok + 1) * cols];
            let acc: f32 = x_row.iter().zip(&row_w).map(|(a, w)| a * w).sum();
            out[tok * sub_n + local_n] = acc;
        }
    }
    out
}

fn check_gemm_parity(
    quant: QuantType,
    m_tokens: usize,
    rows: usize,
    cols: usize,
    row_strip: Option<std::ops::Range<usize>>,
) {
    let raw = fill_pseudo_quant_tensor(quant, rows, cols);
    let dims = [cols, rows];
    let view = TensorView::from_bytes("w", quant, &dims, 0, &raw).unwrap();
    let weight = RepackedWeight::from_tensor(&view, rows, cols).unwrap();

    // Construct `x` on the Q8 quantization lattice (`amax = 127 * step` per 32-element block)
    // so `max_rel_err < 1e-3` measures pure fused RVV/IME kernel vs f32 GEMM fidelity.
    let step = 0.031_25_f32;
    let x: Vec<f32> = (0..m_tokens * cols)
        .map(|i| {
            if i % 32 == 0 {
                127.0_f32 * step
            } else {
                let q = i16::try_from((i * 7 + i / cols * 13) % 121).expect("< 121") - 60;
                f32::from(q) * step
            }
        })
        .collect();
    let act = QuantizedActQ8::quantize(&x, m_tokens, cols).unwrap();

    let range = row_strip.unwrap_or(0..rows);
    let mut got = vec![0.0_f32; m_tokens * range.len()];
    gemm_repacked_rows(&mut got, &weight, &act, range.clone()).unwrap();
    let expected = naive_f32_gemm_rows(&weight, &x, m_tokens, range);

    let mut max_rel = 0.0_f32;
    for (g, e) in got.iter().zip(&expected) {
        let rel = (g - e).abs() / e.abs().max(1.0);
        if rel > max_rel {
            max_rel = rel;
        }
    }
    assert!(
        max_rel < 1e-3,
        "{quant:?} M={m_tokens} K={cols} N={rows}: max_rel={max_rel}"
    );
}

#[test]
fn wave3_fused_gemm_parity_across_block_scaled_quants_and_large_shapes() {
    let _ = (vlen_bytes(), has_ime());
    for quant in [
        QuantType::Q8_0,
        QuantType::Q4_0,
        QuantType::Q4_K_M,
        QuantType::Q5_K,
        QuantType::Q6_K,
        QuantType::TQ2_0,
    ] {
        check_gemm_parity(quant, 8, 256, 256, None);
        check_gemm_parity(quant, 17, 256, 256, None);
    }

    // Exercise large K in {2048, 4096}, N in {1024, 4096}, M in {64, 256} across edge/mid/tail tiles.
    check_gemm_parity(QuantType::Q4_K_M, 64, 1024, 2048, Some(0..16));
    check_gemm_parity(QuantType::Q4_K_M, 64, 1024, 2048, Some(1008..1024));
    check_gemm_parity(QuantType::Q8_0, 256, 4096, 4096, Some(0..8));
    check_gemm_parity(QuantType::Q4_K_M, 256, 4096, 4096, Some(2040..2048));
    check_gemm_parity(QuantType::Q6_K, 17, 4096, 4096, Some(4080..4096));
}

#[test]
fn wave3_rvv_4x8_and_vla_sim_parity_plus_diagnostic_timing() {
    let (rows, cols) = (64, 256);
    let raw = fill_pseudo_quant_tensor(QuantType::Q4_K_M, rows, cols);
    let dims = [cols, rows];
    let view = TensorView::from_bytes("w_bench", QuantType::Q4_K_M, &dims, 0, &raw).unwrap();
    let weight = RepackedWeight::from_tensor(&view, rows, cols).unwrap();

    for m_tokens in [8_usize, 32, 128, 512] {
        let x: Vec<f32> = (0..m_tokens * cols)
            .map(|i| 0.5_f32 + small_f32(i, 19, 8.0, 0.125))
            .collect();
        let act = QuantizedActQ8::quantize(&x, m_tokens, cols).unwrap();

        let mut out_hw = vec![0.0_f32; m_tokens * rows];
        let mut out_256 = vec![0.0_f32; m_tokens * rows];
        let mut out_1024 = vec![0.0_f32; m_tokens * rows];

        let t0 = Instant::now();
        gemm_repacked(&mut out_hw, &weight, &act).unwrap();
        let dt_hw = t0.elapsed();

        gemm_repacked_vla_sim(&mut out_256, &weight, &act, 256).unwrap();
        gemm_repacked_vla_sim(&mut out_1024, &weight, &act, 1024).unwrap();

        for (a, b) in out_hw.iter().zip(&out_256) {
            assert!((a - b).abs() < 1e-4, "M={m_tokens}: {a} vs {b}");
        }
        for (a, b) in out_256.iter().zip(&out_1024) {
            assert!(
                (a - b).abs() < 1e-4,
                "VLA 256 vs 1024 M={m_tokens}: {a} vs {b}"
            );
        }
        eprintln!(
            "wave3_gemm_diag M={m_tokens} K={cols} N={rows} elapsed_us={}",
            dt_hw.as_micros()
        );
    }
}

#[test]
fn wave3_compact_nibble_footprint_and_gemm_parity_all_formats() {
    for quant in [
        QuantType::Q4_0,
        QuantType::Q4_1,
        QuantType::IQ4_NL,
        QuantType::Q4_K_M,
    ] {
        let (rows, cols) = (256, 512);
        let raw = fill_pseudo_quant_tensor(quant, rows, cols);
        let dims = [cols, rows];
        let view = TensorView::from_bytes("nib", quant, &dims, 0, &raw).unwrap();
        let nib = PackedNibbleMatrix::repack(&view).unwrap();
        let repacked = RepackedWeight::from_tensor(&view, rows, cols).unwrap();

        assert_eq!(nib.packed_bytes(), nib.raw_tensor_bytes(), "{quant:?}");
        assert!(
            nib.packed_bytes() * 100 <= nib.raw_tensor_bytes() * 105,
            "{quant:?} exceeded 1.05x"
        );

        let x: Vec<f32> = (0..17 * cols)
            .map(|i| 0.5_f32 + small_f32(i, 13, 5.0, 0.25))
            .collect();
        let act = QuantizedActQ8::quantize(&x, 17, cols).unwrap();
        let mut out_nib = vec![0.0_f32; 17 * rows];
        let mut out_ref = vec![0.0_f32; 17 * rows];
        match quant {
            QuantType::Q4_0 => gemm_q4_0_ime(&mut out_nib, &nib, &act).unwrap(),
            QuantType::Q4_K_M => gemm_q4_k_ime(&mut out_nib, &nib, &act).unwrap(),
            _ => nib.gemm(&mut out_nib, &act).unwrap(),
        }
        gemm_repacked(&mut out_ref, &repacked, &act).unwrap();
        for (a, b) in out_nib.iter().zip(&out_ref) {
            assert!((a - b).abs() < 1e-4, "{quant:?}: {a} vs {b}");
        }
    }

    // Verify large unaligned row count still comfortably beats 1.05x raw footprint (`203 -> 208`, +2.46%).
    let (rows_unaligned, cols_large) = (203, 512);
    let raw_large = fill_pseudo_quant_tensor(QuantType::Q4_K_M, rows_unaligned, cols_large);
    let dims = [cols_large, rows_unaligned];
    let view =
        TensorView::from_bytes("nib_large", QuantType::Q4_K_M, &dims, 0, &raw_large).unwrap();
    let nib_large = PackedNibbleMatrix::from_tensor(&view, rows_unaligned, cols_large).unwrap();
    assert!(nib_large.packed_bytes() * 100 <= nib_large.raw_tensor_bytes() * 105);
}

#[test]
fn wave3_parallel_repack_is_bit_exact_across_4_threads() {
    for quant in [
        QuantType::Q4_0,
        QuantType::Q4_K_M,
        QuantType::Q5_K,
        QuantType::Q6_K,
        QuantType::Q8_0,
    ] {
        let (rows, cols) = (67, 512);
        let raw = fill_pseudo_quant_tensor(quant, rows, cols);
        let dims = [cols, rows];
        let view = TensorView::from_bytes("w_par", quant, &dims, 0, &raw).unwrap();
        let seq = RepackedWeight::from_tensor(&view, rows, cols).unwrap();
        let layout = RepackedWeight::block_scaled_layout(&view, rows, cols).unwrap();

        let mut par_tiles = vec![Aligned128::zeroed(); layout.total_tiles()];
        let mut par_scales = vec![0.0_f32; layout.total_scales()];
        let mut par_mins = layout
            .has_mins()
            .then(|| vec![0.0_f32; layout.total_mins()]);

        let n_tiles = layout.n_tiles();
        let workers = 4_usize;
        let per = n_tiles.div_ceil(workers);
        std::thread::scope(|s| {
            let mut tile_rem = &mut par_tiles[..];
            let mut scale_rem = &mut par_scales[..];
            let mut min_rem = par_mins.as_deref_mut();
            let mut start = 0_usize;
            while start < n_tiles {
                let end = (start + per).min(n_tiles);
                let count = end - start;
                let (t_slice, t_rest) = tile_rem.split_at_mut(count * layout.k_chunks());
                tile_rem = t_rest;
                let (s_slice, s_rest) = scale_rem.split_at_mut(layout.scales_for_tiles(count));
                scale_rem = s_rest;
                let m_slice = min_rem.take().map(|buf| {
                    let (m_cur, m_rest) = buf.split_at_mut(layout.scales_for_tiles(count));
                    min_rem = Some(m_rest);
                    m_cur
                });
                let view_ref = &view;
                let use_q4k_i8 = quant == QuantType::Q4_K_M && start > 0;
                s.spawn(move || {
                    if use_q4k_i8 {
                        repack_q4_k_m_parallel_slices(
                            view_ref,
                            rows,
                            cols,
                            start..end,
                            Aligned128::slice_as_i8_mut(t_slice),
                            s_slice,
                            m_slice.expect("Q4_K_M mins"),
                        )
                        .unwrap();
                    } else {
                        repack_into_tile_range(
                            view_ref,
                            rows,
                            cols,
                            start..end,
                            t_slice,
                            s_slice,
                            m_slice,
                        )
                        .unwrap();
                    }
                });
                start = end;
            }
        });

        let par =
            RepackedWeight::from_tile_buffers(layout, par_tiles, par_scales, par_mins).unwrap();
        let mut row_seq = vec![0.0_f32; cols];
        let mut row_par = vec![0.0_f32; cols];
        for r in 0..rows {
            seq.dequantize_row(r, &mut row_seq).unwrap();
            par.dequantize_row(r, &mut row_par).unwrap();
            assert_eq!(row_seq, row_par, "{quant:?} row={r}");
        }
    }
}

#[test]
fn wave3_parallel_nibble_repack_is_bit_exact_across_4_threads() {
    let (rows, cols) = (67, 512);
    let raw = fill_pseudo_quant_tensor(QuantType::Q4_K_M, rows, cols);
    let dims = [cols, rows];
    let view = TensorView::from_bytes("nib_par", QuantType::Q4_K_M, &dims, 0, &raw).unwrap();
    let seq = PackedNibbleMatrix::repack(&view).unwrap();
    let n_tiles = rows.div_ceil(8);
    let k_blocks32 = cols / 32;
    let n_sblks = cols / 256;
    let mut par_tiles = vec![Aligned128::zeroed(); n_tiles * k_blocks32];
    let mut par_hdrs = vec![[0_u8; 16]; n_tiles * n_sblks * 8];
    let per = n_tiles.div_ceil(4);
    std::thread::scope(|s| {
        let mut t_rem = &mut par_tiles[..];
        let mut h_rem = &mut par_hdrs[..];
        let mut start = 0_usize;
        while start < n_tiles {
            let end = (start + per).min(n_tiles);
            let count = end - start;
            let (ts, tr) = t_rem.split_at_mut(count * k_blocks32);
            t_rem = tr;
            let (hs, hr) = h_rem.split_at_mut(count * n_sblks * 8);
            h_rem = hr;
            let view_ref = &view;
            s.spawn(move || {
                repack_nibble_into_tile_range(
                    view_ref,
                    rows,
                    cols,
                    start..end,
                    ts,
                    &mut [],
                    None,
                    Some(hs),
                )
                .unwrap();
            });
            start = end;
        }
    });
    let par = PackedNibbleMatrix::from_q4_k_buffers(rows, cols, par_tiles, par_hdrs).unwrap();
    let mut row_seq = vec![0.0_f32; cols];
    let mut row_par = vec![0.0_f32; cols];
    for r in 0..rows {
        seq.dequantize_row(r, &mut row_seq).unwrap();
        par.dequantize_row(r, &mut row_par).unwrap();
        assert_eq!(row_seq, row_par);
    }
}

fn naive_gqa_prefill_ref(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    dims: [usize; 5],
    cfg: AttentionMaskConfig,
) -> Vec<f32> {
    let [m_tokens, n_q, n_kv, head_dim, total_seq_len] = dims;
    let group_size = n_q / n_kv;
    let mut out = vec![0.0_f32; m_tokens * n_q * head_dim];
    for tok in 0..m_tokens {
        let q_pos = cfg.start_pos + tok;
        for h_q in 0..n_q {
            let h_kv = h_q / group_size;
            let q_vec = &q[(tok * n_q + h_q) * head_dim..][..head_dim];
            let mut scores = Vec::with_capacity(total_seq_len);
            let mut max_s = f32::NEG_INFINITY;
            for kv_pos in 0..total_seq_len {
                let causal_ok = kv_pos <= q_pos;
                let swa_ok = cfg.sliding_window.is_none_or(|w| kv_pos + w > q_pos);
                if causal_ok && swa_ok {
                    let k_off = (kv_pos * n_kv + h_kv) * head_dim;
                    let dot: f32 = q_vec
                        .iter()
                        .zip(&k_cache[k_off..k_off + head_dim])
                        .map(|(a, b)| a * b)
                        .sum();
                    let mut sc = dot * cfg.scale;
                    if let Some(cap) = cfg.logit_softcap {
                        sc = cap * (sc / cap).tanh();
                    }
                    if sc > max_s {
                        max_s = sc;
                    }
                    scores.push(Some(sc));
                } else {
                    scores.push(None);
                }
            }
            let mut sum = 0.0_f32;
            let mut probs = vec![0.0_f32; total_seq_len];
            for (idx, opt_s) in scores.into_iter().enumerate() {
                if let Some(sc) = opt_s {
                    let e = (sc - max_s).exp();
                    probs[idx] = e;
                    sum += e;
                }
            }
            let out_slice = &mut out[(tok * n_q + h_q) * head_dim..][..head_dim];
            for (kv_pos, prob) in probs.into_iter().enumerate() {
                let weight = prob / sum;
                if weight > 0.0 {
                    let v_off = (kv_pos * n_kv + h_kv) * head_dim;
                    for d_idx in 0..head_dim {
                        out_slice[d_idx] += weight * v_cache[v_off + d_idx];
                    }
                }
            }
        }
    }
    out
}

#[test]
fn wave3_gqa_prefill_swa_softcap_and_swept_dimensions() {
    // 1. Exact M=16 prefill == 16x M=1 decode, plus SWA and softcap reference parity.
    let (m_tokens, start_pos, n_q, n_kv, head_dim) = (16, 17, 8, 2, 64);
    let total_seq = start_pos + m_tokens;
    let scale = 1.0_f32 / 8.0_f32;
    let q: Vec<f32> = (0..m_tokens * n_q * head_dim)
        .map(|i| small_f32(i, 23, 11.0, 0.1))
        .collect();
    let k: Vec<f32> = (0..total_seq * n_kv * head_dim)
        .map(|i| small_f32(i * 3, 29, 14.0, 0.1))
        .collect();
    let v: Vec<f32> = (0..total_seq * n_kv * head_dim)
        .map(|i| small_f32(i * 5, 31, 15.0, 0.1))
        .collect();

    let mut scratch = vec![0.0_f32; total_seq];
    let mut out_prefill = vec![0.0_f32; m_tokens * n_q * head_dim];
    gqa_causal_attention_prefill_f32(
        &mut out_prefill,
        &q,
        &k,
        &v,
        m_tokens,
        n_q,
        n_kv,
        head_dim,
        total_seq,
        AttentionMaskConfig::causal(start_pos, scale),
        &mut scratch,
    )
    .unwrap();
    let mut out_step = vec![0.0_f32; n_q * head_dim];
    for tok in 0..m_tokens {
        let seq = start_pos + tok + 1;
        gqa_causal_attention_with_scratch_f32(
            &mut out_step,
            &q[tok * n_q * head_dim..(tok + 1) * n_q * head_dim],
            &k[..seq * n_kv * head_dim],
            &v[..seq * n_kv * head_dim],
            n_q,
            n_kv,
            head_dim,
            seq,
            scale,
            &mut scratch,
        )
        .unwrap();
        for (a, b) in out_prefill[tok * n_q * head_dim..(tok + 1) * n_q * head_dim]
            .iter()
            .zip(&out_step)
        {
            assert!((a - b).abs() < 1e-6);
        }
    }

    let cfg_swa_cap = AttentionMaskConfig::causal(5, scale)
        .with_sliding_window(Some(4))
        .with_logit_softcap(Some(1.5));
    let m_swa = 7;
    let total_swa = 16;
    let q_swa = &q[..m_swa * n_q * head_dim];
    let k_swa = &k[..total_swa * n_kv * head_dim];
    let v_swa = &v[..total_swa * n_kv * head_dim];
    let mut out_swa = vec![0.0_f32; m_swa * n_q * head_dim];
    gqa_causal_attention_prefill_f32(
        &mut out_swa,
        q_swa,
        k_swa,
        v_swa,
        m_swa,
        n_q,
        n_kv,
        head_dim,
        total_swa,
        cfg_swa_cap,
        &mut scratch,
    )
    .unwrap();
    let ref_swa = naive_gqa_prefill_ref(
        q_swa,
        k_swa,
        v_swa,
        [m_swa, n_q, n_kv, head_dim, total_swa],
        cfg_swa_cap,
    );
    let max_err = out_swa
        .iter()
        .zip(&ref_swa)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(max_err < 1e-5, "SWA + softcap max_err={max_err}");
}

#[test]
fn wave3_gqa_prefill_swept_ratios_dims_and_lengths() {
    // Sweep gqa_ratio in {1, 4, 7, 8}, head_dim in {64, 96, 128, 256}, seq_len in {1, 33, 512, 2048}.
    let sweep = [
        (1_usize, 1_usize, 64_usize, 1_usize),
        (4, 2, 96, 33),
        (7, 1, 128, 512),
        (8, 2, 256, 33),
        (4, 1, 64, 2048),
    ];
    for (gqa_ratio, kv_heads, hd, total_len) in sweep {
        let q_heads = gqa_ratio * kv_heads;
        let m_prefill = 1_usize.min(total_len);
        let q_s: Vec<f32> = (0..m_prefill * q_heads * hd)
            .map(|i| small_f32(i, 17, 8.0, 0.1))
            .collect();
        let k_s: Vec<f32> = (0..total_len * kv_heads * hd)
            .map(|i| small_f32(i * 3, 19, 9.0, 0.1))
            .collect();
        let v_s: Vec<f32> = (0..total_len * kv_heads * hd)
            .map(|i| small_f32(i * 5, 23, 11.0, 0.1))
            .collect();
        let cfg = AttentionMaskConfig::causal(total_len - m_prefill, 0.125);
        let mut sc = vec![0.0_f32; total_len];
        let mut out = vec![0.0_f32; m_prefill * q_heads * hd];
        gqa_causal_attention_prefill_f32(
            &mut out, &q_s, &k_s, &v_s, m_prefill, q_heads, kv_heads, hd, total_len, cfg, &mut sc,
        )
        .unwrap();
        let expected = naive_gqa_prefill_ref(
            &q_s,
            &k_s,
            &v_s,
            [m_prefill, q_heads, kv_heads, hd, total_len],
            cfg,
        );
        let err = out
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            err < 1e-5,
            "gqa_ratio={gqa_ratio} hd={hd} seq_len={total_len}: err={err}"
        );
    }
}

#[test]
fn wave3_gqa_prefill_multithread_and_strided_kv() {
    let (m_par, n_q_par, n_kv_par, hd_par, seq_par) = (5, 16, 4, 64, 48);
    let q_p: Vec<f32> = (0..m_par * n_q_par * hd_par)
        .map(|i| small_f32(i, 19, 9.0, 0.1))
        .collect();
    let stride = (n_kv_par + 1) * hd_par;
    let mut k_strided = vec![0.0_f32; seq_par * stride];
    let mut v_strided = vec![0.0_f32; seq_par * stride];
    for s_idx in 0..seq_par {
        for kv_idx in 0..n_kv_par {
            for d_idx in 0..hd_par {
                let off = s_idx * stride + kv_idx * hd_par + d_idx;
                k_strided[off] = small_f32(off * 3, 29, 14.0, 0.1);
                v_strided[off] = small_f32(off * 7, 31, 15.0, 0.1);
            }
        }
    }
    let k_dense: Vec<f32> = k_strided
        .chunks(stride)
        .flat_map(|row| row[..n_kv_par * hd_par].iter().copied())
        .collect();
    let v_dense: Vec<f32> = v_strided
        .chunks(stride)
        .flat_map(|row| row[..n_kv_par * hd_par].iter().copied())
        .collect();
    let cfg_par = AttentionMaskConfig::causal(seq_par - m_par, 0.125).with_sliding_window(Some(16));
    let mut out_single = vec![0.0_f32; m_par * n_q_par * hd_par];
    let mut scratch_single = vec![0.0_f32; seq_par];
    gqa_causal_attention_prefill_strided_kv_f32(
        &mut out_single,
        &q_p,
        &k_strided,
        &v_strided,
        m_par,
        n_q_par,
        n_kv_par,
        hd_par,
        seq_par,
        stride,
        0..n_kv_par,
        cfg_par,
        &mut scratch_single,
    )
    .unwrap();

    let mut worker_results: Vec<(std::ops::Range<usize>, Vec<f32>)> = Vec::new();
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..4)
            .filter_map(|wid| {
                let (kv_r, q_r) = gqa_kv_head_partition(n_kv_par, n_q_par, wid, 4)?;
                let q_ref = &q_p;
                let k_ref = &k_dense;
                let v_ref = &v_dense;
                Some(s.spawn(move || {
                    let mut local = vec![0.0_f32; m_par * q_r.len() * hd_par];
                    let mut local_scratch = vec![0.0_f32; seq_par];
                    gqa_causal_attention_prefill_kv_heads_f32(
                        &mut local,
                        q_ref,
                        k_ref,
                        v_ref,
                        m_par,
                        n_q_par,
                        n_kv_par,
                        hd_par,
                        seq_par,
                        kv_r,
                        cfg_par,
                        &mut local_scratch,
                    )
                    .unwrap();
                    (q_r, local)
                }))
            })
            .collect();
        for handle in handles {
            worker_results.push(handle.join().unwrap());
        }
    });
    assert_eq!(worker_results.len(), 4);
    for (q_range, local) in worker_results {
        let local_q = q_range.len();
        for tok in 0..m_par {
            assert_eq!(
                &local[tok * local_q * hd_par..(tok + 1) * local_q * hd_par],
                &out_single[tok * n_q_par * hd_par + q_range.start * hd_par
                    ..tok * n_q_par * hd_par + q_range.end * hd_par]
            );
        }
    }
}
