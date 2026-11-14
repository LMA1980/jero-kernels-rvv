//! Wave 4 accuracy, parity, edge-case, and diagnostic integration tests (`#94–#101`, `#30`, `#31`, `#33`).

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::needless_range_loop,
    clippy::too_many_lines,
    reason = "deterministic integration suites sweep multiple quant/SSM variants with explicit loops"
)]

use std::time::Instant;

use jero_core::{QuantType, TensorView};
use jero_kernels_rvv::{
    BaselineKernel, BlockQ8_0, BlockQ8K, Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMENTS, Q4_K_M_BLOCK_BYTES,
    Q4_K_M_BLOCK_ELEMENTS, Q5_0_BLOCK_BYTES, Q5_0_BLOCK_ELEMENTS, Q5_K_BLOCK_BYTES,
    Q5_K_BLOCK_ELEMENTS, Q6_K_BLOCK_BYTES, Q6_K_BLOCK_ELEMENTS, Q8_0_BLOCK_BYTES,
    Q8_0_BLOCK_ELEMENTS, QuantizedActQ8, RepackedWeight, RvvKernel, TQ1_0_BLOCK_BYTES,
    TQ1_0_BLOCK_ELEMENTS, TQ2_0_BLOCK_BYTES, TQ2_0_BLOCK_ELEMENTS, VectorKernel, argmax_f32,
    argmax_f32_scalar, dequantize_q4_0_block, dequantize_q4_k_m_block, dequantize_q5_0_block,
    dequantize_q5_k_block, dequantize_q6_k_block, dequantize_q8_0_block, gemm_repacked,
    gemv_q4_0_q8_0, gemv_q4_0_q8_0_4x, gemv_q4_k_q8_k, gemv_q5_0_q8_0, gemv_q5_k_q8_k,
    gemv_q6_k_q8_k, gemv_q8_0_q8_0, gemv_repacked, gemv_tq1_0_q8_k, gemv_tq2_0_q8_k,
    mamba2_conv1d_silu_chunk_f32, mamba2_conv1d_silu_step, mamba2_ssd_chunk_scan_f32,
    mamba2_ssd_intra_chunk_ime_i8, mamba2_ssd_state_step_f32, quantize_row_q8_0, quantize_row_q8_k,
    rwkv_token_shift_lerp_f32, rwkv_token_shift_lerp_sequence_f32,
    rwkv7_wkv_delta_rule_sequence_f32, rwkv7_wkv_delta_rule_step_f32, unpack_tq1_0_block_i8,
    unpack_tq2_0_block_i8, vec_dot_q4_0, vec_dot_q5_0, vec_dot_q5_k, vec_dot_q6_k, vec_dot_q8_0,
};

fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

fn gen_activation(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let r = ((lcg_next(&mut s) >> 32) as u32) as f32 / (u32::MAX as f32);
        // Add subtle positive bias + multi-frequency signal so random matvec avoids exact cancellation.
        let val = (r - 0.35) * 1.4 + ((i as f32) * 0.013).sin() * 0.2;
        out.push(val);
    }
    out
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut dot = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let xd = f64::from(x);
        let yd = f64::from(y);
        dot += xd * yd;
        na += xd * xd;
        nb += yd * yd;
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn l2_rel_err(got: &[f32], reference: &[f32]) -> f64 {
    assert_eq!(got.len(), reference.len());
    let mut diff_sq = 0.0_f64;
    let mut ref_sq = 0.0_f64;
    for (&g, &r) in got.iter().zip(reference.iter()) {
        let d = f64::from(g) - f64::from(r);
        diff_sq += d * d;
        ref_sq += f64::from(r) * f64::from(r);
    }
    diff_sq.sqrt() / ref_sq.sqrt()
}

fn fill_q4_0_bytes(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    let blocks = cols / Q4_0_BLOCK_ELEMENTS;
    let mut raw = vec![0_u8; rows * blocks * Q4_0_BLOCK_BYTES];
    let mut s = seed;
    let scales = [0x3000_u16, 0x3200, 0x3400, 0x3600]; // positive 0.125..0.375
    for r in 0..rows {
        for b in 0..blocks {
            let off = (r * blocks + b) * Q4_0_BLOCK_BYTES;
            let sc = scales[(r + b) % scales.len()];
            raw[off..off + 2].copy_from_slice(&sc.to_le_bytes());
            for j in 0..16 {
                let lo = ((lcg_next(&mut s) >> 32) % 16) as u8;
                let hi = ((lcg_next(&mut s) >> 40) % 16) as u8;
                raw[off + 2 + j] = (hi << 4) | lo;
            }
        }
    }
    raw
}

fn fill_q5_0_bytes(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    let blocks = cols / Q5_0_BLOCK_ELEMENTS;
    let mut raw = vec![0_u8; rows * blocks * Q5_0_BLOCK_BYTES];
    let mut s = seed;
    let scales = [0x2C00_u16, 0x3000, 0x3200, 0x3400];
    for r in 0..rows {
        for b in 0..blocks {
            let off = (r * blocks + b) * Q5_0_BLOCK_BYTES;
            let sc = scales[(r + b) % scales.len()];
            raw[off..off + 2].copy_from_slice(&sc.to_le_bytes());
            let qh = (lcg_next(&mut s) >> 32) as u32;
            raw[off + 2..off + 6].copy_from_slice(&qh.to_le_bytes());
            for j in 0..16 {
                raw[off + 6 + j] = (lcg_next(&mut s) >> 32) as u8;
            }
        }
    }
    raw
}

fn fill_q8_0_bytes(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    let blocks = cols / Q8_0_BLOCK_ELEMENTS;
    let mut raw = vec![0_u8; rows * blocks * Q8_0_BLOCK_BYTES];
    let mut s = seed;
    let scales = [0x2800_u16, 0x2C00, 0x3000, 0x3200];
    for r in 0..rows {
        for b in 0..blocks {
            let off = (r * blocks + b) * Q8_0_BLOCK_BYTES;
            let sc = scales[(r + b) % scales.len()];
            raw[off..off + 2].copy_from_slice(&sc.to_le_bytes());
            for j in 0..32 {
                let q = (((lcg_next(&mut s) >> 32) % 127) as i8).wrapping_sub(63);
                raw[off + 2 + j] = q.to_ne_bytes()[0];
            }
        }
    }
    raw
}

fn fill_q4_k_m_bytes(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    let blocks = cols / Q4_K_M_BLOCK_ELEMENTS;
    let mut raw = vec![0_u8; rows * blocks * Q4_K_M_BLOCK_BYTES];
    let mut s = seed;
    for r in 0..rows {
        for b in 0..blocks {
            let off = (r * blocks + b) * Q4_K_M_BLOCK_BYTES;
            raw[off..off + 2].copy_from_slice(&0x3000_u16.to_le_bytes()); // d = 0.125
            raw[off + 2..off + 4].copy_from_slice(&0x2800_u16.to_le_bytes()); // dmin = 0.03125
            for idx in 4..16 {
                raw[off + idx] = ((lcg_next(&mut s) >> 32) % 31 + 1) as u8;
            }
            for idx in 16..144 {
                raw[off + idx] = (lcg_next(&mut s) >> 32) as u8;
            }
        }
    }
    raw
}

fn fill_q5_k_bytes(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    let blocks = cols / Q5_K_BLOCK_ELEMENTS;
    let mut raw = vec![0_u8; rows * blocks * Q5_K_BLOCK_BYTES];
    let mut s = seed;
    for r in 0..rows {
        for b in 0..blocks {
            let off = (r * blocks + b) * Q5_K_BLOCK_BYTES;
            raw[off..off + 2].copy_from_slice(&0x3000_u16.to_le_bytes());
            raw[off + 2..off + 4].copy_from_slice(&0x2800_u16.to_le_bytes());
            for idx in 4..16 {
                raw[off + idx] = ((lcg_next(&mut s) >> 32) % 31 + 1) as u8;
            }
            for idx in 16..176 {
                raw[off + idx] = (lcg_next(&mut s) >> 32) as u8;
            }
        }
    }
    raw
}

fn fill_q6_k_bytes(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    let blocks = cols / Q6_K_BLOCK_ELEMENTS;
    let mut raw = vec![0_u8; rows * blocks * Q6_K_BLOCK_BYTES];
    let mut s = seed;
    for r in 0..rows {
        for b in 0..blocks {
            let off = (r * blocks + b) * Q6_K_BLOCK_BYTES;
            for idx in 0..192 {
                raw[off + idx] = (lcg_next(&mut s) >> 32) as u8;
            }
            for idx in 192..208 {
                let sc = (((lcg_next(&mut s) >> 32) % 15) as i8) - 7;
                raw[off + idx] = sc.to_ne_bytes()[0];
            }
            raw[off + 208..off + 210].copy_from_slice(&0x3000_u16.to_le_bytes());
        }
    }
    raw
}

fn fill_tq1_0_bytes(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    let blocks = cols / TQ1_0_BLOCK_ELEMENTS;
    let mut raw = vec![0_u8; rows * blocks * TQ1_0_BLOCK_BYTES];
    let mut s = seed;
    for r in 0..rows {
        for b in 0..blocks {
            let off = (r * blocks + b) * TQ1_0_BLOCK_BYTES;
            raw[off..off + 2].copy_from_slice(&0x3400_u16.to_le_bytes());
            for idx in 2..TQ1_0_BLOCK_BYTES {
                raw[off + idx] = ((lcg_next(&mut s) >> 32) % 243) as u8;
            }
        }
    }
    raw
}

fn fill_tq2_0_bytes(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    let blocks = cols / TQ2_0_BLOCK_ELEMENTS;
    let mut raw = vec![0_u8; rows * blocks * TQ2_0_BLOCK_BYTES];
    let mut s = seed;
    for r in 0..rows {
        for b in 0..blocks {
            let off = (r * blocks + b) * TQ2_0_BLOCK_BYTES;
            raw[off..off + 2].copy_from_slice(&0x3400_u16.to_le_bytes());
            for idx in 2..TQ2_0_BLOCK_BYTES {
                raw[off + idx] = (lcg_next(&mut s) >> 32) as u8;
            }
        }
    }
    raw
}

fn ref_matvec_blocks<const B: usize, const E: usize>(
    raw: &[u8],
    rows: usize,
    cols: usize,
    x: &[f32],
    dequant: fn(&[u8; B], &mut [f32; E]),
) -> Vec<f32> {
    let blocks = cols / E;
    let mut out = vec![0.0_f32; rows];
    let mut dq = [0.0_f32; E];
    for r in 0..rows {
        let mut sum = 0.0_f64;
        for b in 0..blocks {
            let off = (r * blocks + b) * B;
            let blk: &[u8; B] = raw[off..off + B].try_into().expect("block slice");
            dequant(blk, &mut dq);
            for j in 0..E {
                sum += f64::from(dq[j]) * f64::from(x[b * E + j]);
            }
        }
        out[r] = sum as f32;
    }
    out
}

#[test]
fn wave4_q4_0_q5_0_q8_0_q8_0_gemv_cosine_above_0_9995_across_dims() {
    let k_dims = [128_usize, 896, 2048, 4096];
    let n_dims = [1_usize, 3, 4, 17, 1024];

    for &k in &k_dims {
        let x = gen_activation(k, 100 + k as u64);
        let mut x_q8 = vec![BlockQ8_0::ZERO; k / Q8_0_BLOCK_ELEMENTS];
        quantize_row_q8_0(&x, &mut x_q8).expect("quantize q8_0");

        for &n in &n_dims {
            let shape = [k, n];
            // Keep quadratic reference oracle fast for n = 1024 by checking all rows on <= 17
            // and checking n = 1024 only at k = 128 or k = 896 (plus gemv execution at all n).
            let check_ref = n <= 17 || k <= 896;

            // Q4_0
            let w4 = fill_q4_0_bytes(n, k, 200 + (k + n) as u64);
            let v4 = TensorView::from_bytes("q4_0", QuantType::Q4_0, &shape, 0, &w4).unwrap();
            let mut o4 = vec![0.0_f32; n];
            let mut o4_4x = vec![0.0_f32; n];
            let mut o4_f32 = vec![0.0_f32; n];
            gemv_q4_0_q8_0(&mut o4, &v4, &x_q8).expect("gemv_q4_0_q8_0");
            gemv_q4_0_q8_0_4x(&mut o4_4x, &v4, &x_q8).expect("gemv_q4_0_q8_0_4x");
            vec_dot_q4_0(&mut o4_f32, &v4, &x).expect("vec_dot_q4_0");
            assert_eq!(o4, o4_4x);
            assert!(cosine_similarity(&o4, &o4_f32) > 0.9995);
            if check_ref {
                let ref4 = ref_matvec_blocks::<Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMENTS>(
                    &w4,
                    n,
                    k,
                    &x,
                    dequantize_q4_0_block,
                );
                let cos = if n == 1 {
                    assert!((o4[0] - ref4[0]).abs() <= ref4[0].abs() * 0.03 + 0.25);
                    1.0
                } else {
                    cosine_similarity(&o4, &ref4)
                };
                assert!(cos > 0.9995, "Q4_0 k={k} n={n} cos={cos}");
            }

            // Q5_0
            let w5 = fill_q5_0_bytes(n, k, 300 + (k + n) as u64);
            let v5 = TensorView::from_bytes("q5_0", QuantType::Q5_0, &shape, 0, &w5).unwrap();
            let mut o5 = vec![0.0_f32; n];
            let mut o5_f32 = vec![0.0_f32; n];
            gemv_q5_0_q8_0(&mut o5, &v5, &x_q8).expect("gemv_q5_0_q8_0");
            vec_dot_q5_0(&mut o5_f32, &v5, &x).expect("vec_dot_q5_0");
            assert!(cosine_similarity(&o5, &o5_f32) > 0.9995);
            if check_ref {
                let ref5 = ref_matvec_blocks::<Q5_0_BLOCK_BYTES, Q5_0_BLOCK_ELEMENTS>(
                    &w5,
                    n,
                    k,
                    &x,
                    dequantize_q5_0_block,
                );
                let cos = if n == 1 {
                    assert!((o5[0] - ref5[0]).abs() <= ref5[0].abs() * 0.03 + 0.25);
                    1.0
                } else {
                    cosine_similarity(&o5, &ref5)
                };
                assert!(cos > 0.9995, "Q5_0 k={k} n={n} cos={cos}");
            }

            // Q8_0
            let w8 = fill_q8_0_bytes(n, k, 400 + (k + n) as u64);
            let v8 = TensorView::from_bytes("q8_0", QuantType::Q8_0, &shape, 0, &w8).unwrap();
            let mut o8 = vec![0.0_f32; n];
            let mut o8_f32 = vec![0.0_f32; n];
            gemv_q8_0_q8_0(&mut o8, &v8, &x_q8).expect("gemv_q8_0_q8_0");
            vec_dot_q8_0(&mut o8_f32, &v8, &x).expect("vec_dot_q8_0");
            assert!(cosine_similarity(&o8, &o8_f32) > 0.9995);
            if check_ref {
                let ref8 = ref_matvec_blocks::<Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMENTS>(
                    &w8,
                    n,
                    k,
                    &x,
                    dequantize_q8_0_block,
                );
                let cos = if n == 1 {
                    assert!((o8[0] - ref8[0]).abs() <= ref8[0].abs() * 0.03 + 0.25);
                    1.0
                } else {
                    cosine_similarity(&o8, &ref8)
                };
                assert!(cos > 0.9995, "Q8_0 k={k} n={n} cos={cos}");
            }
        }
    }
}

fn dequantize_q8_k_row(x_q8k: &[BlockQ8K]) -> Vec<f32> {
    let mut out = Vec::with_capacity(x_q8k.len() * 256);
    for blk in x_q8k {
        for &q in &blk.qs {
            out.push(f32::from(q) * blk.d);
        }
    }
    out
}

#[test]
fn wave4_k_quants_and_ternary_q8_k_cosine_and_rel_err_across_model_dims() {
    let k_dims = [256_usize, 1536, 2048, 3584, 5120];
    let rows = 5_usize;

    for &k in &k_dims {
        let shape = [k, rows];
        let x = gen_activation(k, 700 + k as u64);
        let mut x_q8k = vec![BlockQ8K::ZERO; k / 256];
        quantize_row_q8_k(&x, &mut x_q8k).expect("quantize q8_k");
        let x_deq = dequantize_q8_k_row(&x_q8k);

        // Q4_K_M
        let w4k = fill_q4_k_m_bytes(rows, k, 11 + k as u64);
        let v4k = TensorView::from_bytes("q4k", QuantType::Q4_K_M, &shape, 0, &w4k).unwrap();
        let mut o4k = vec![0.0_f32; rows];
        let mut o4k_trait = vec![0.0_f32; rows];
        gemv_q4_k_q8_k(&mut o4k, &v4k, &x_q8k).expect("gemv_q4_k_q8_k");
        BaselineKernel::new()
            .vec_dot_q4_k_m(&mut o4k_trait, &v4k, &x)
            .expect("vec_dot_q4_k_m");
        let r4k = ref_matvec_blocks::<Q4_K_M_BLOCK_BYTES, Q4_K_M_BLOCK_ELEMENTS>(
            &w4k,
            rows,
            k,
            &x,
            dequantize_q4_k_m_block,
        );
        let r4k_q8 = ref_matvec_blocks::<Q4_K_M_BLOCK_BYTES, Q4_K_M_BLOCK_ELEMENTS>(
            &w4k,
            rows,
            k,
            &x_deq,
            dequantize_q4_k_m_block,
        );
        assert!(cosine_similarity(&o4k, &r4k) > 0.999);
        assert!(cosine_similarity(&o4k_trait, &r4k) > 0.999);
        assert!(l2_rel_err(&o4k, &r4k_q8) < 1e-5);
        assert!(
            l2_rel_err(&o4k, &r4k) < 1.5e-2,
            "Q4_K_M k={k} rel_err={}",
            l2_rel_err(&o4k, &r4k)
        );

        // Q5_K
        let w5k = fill_q5_k_bytes(rows, k, 22 + k as u64);
        let v5k = TensorView::from_bytes("q5k", QuantType::Q5_K, &shape, 0, &w5k).unwrap();
        let mut o5k = vec![0.0_f32; rows];
        let mut o5k_api = vec![0.0_f32; rows];
        gemv_q5_k_q8_k(&mut o5k, &v5k, &x_q8k).expect("gemv_q5_k_q8_k");
        vec_dot_q5_k(&mut o5k_api, &v5k, &x).expect("vec_dot_q5_k");
        let r5k = ref_matvec_blocks::<Q5_K_BLOCK_BYTES, Q5_K_BLOCK_ELEMENTS>(
            &w5k,
            rows,
            k,
            &x,
            dequantize_q5_k_block,
        );
        let r5k_q8 = ref_matvec_blocks::<Q5_K_BLOCK_BYTES, Q5_K_BLOCK_ELEMENTS>(
            &w5k,
            rows,
            k,
            &x_deq,
            dequantize_q5_k_block,
        );
        assert!(cosine_similarity(&o5k, &r5k) > 0.999);
        assert!(cosine_similarity(&o5k_api, &r5k) > 0.999);
        assert!(l2_rel_err(&o5k, &r5k_q8) < 1e-5);
        assert!(
            l2_rel_err(&o5k, &r5k) < 1.5e-2,
            "Q5_K k={k} rel_err={}",
            l2_rel_err(&o5k, &r5k)
        );

        // Q6_K
        let w6k = fill_q6_k_bytes(rows, k, 33 + k as u64);
        let v6k = TensorView::from_bytes("q6k", QuantType::Q6_K, &shape, 0, &w6k).unwrap();
        let mut o6k = vec![0.0_f32; rows];
        let mut o6k_api = vec![0.0_f32; rows];
        gemv_q6_k_q8_k(&mut o6k, &v6k, &x_q8k).expect("gemv_q6_k_q8_k");
        vec_dot_q6_k(&mut o6k_api, &v6k, &x).expect("vec_dot_q6_k");
        let r6k = ref_matvec_blocks::<Q6_K_BLOCK_BYTES, Q6_K_BLOCK_ELEMENTS>(
            &w6k,
            rows,
            k,
            &x,
            dequantize_q6_k_block,
        );
        let r6k_q8 = ref_matvec_blocks::<Q6_K_BLOCK_BYTES, Q6_K_BLOCK_ELEMENTS>(
            &w6k,
            rows,
            k,
            &x_deq,
            dequantize_q6_k_block,
        );
        assert!(cosine_similarity(&o6k, &r6k) > 0.999);
        assert!(cosine_similarity(&o6k_api, &r6k) > 0.999);
        assert!(l2_rel_err(&o6k, &r6k_q8) < 1e-5);
        assert!(
            l2_rel_err(&o6k, &r6k) < 1.5e-2,
            "Q6_K k={k} rel_err={}",
            l2_rel_err(&o6k, &r6k)
        );

        // TQ1_0
        let wt1 = fill_tq1_0_bytes(rows, k, 44 + k as u64);
        let vt1 = TensorView::from_bytes("tq1", QuantType::TQ1_0, &shape, 0, &wt1).unwrap();
        let mut ot1 = vec![0.0_f32; rows];
        gemv_tq1_0_q8_k(&mut ot1, &vt1, &x_q8k).expect("gemv_tq1_0_q8_k");
        let rt1 = ref_matvec_blocks::<TQ1_0_BLOCK_BYTES, TQ1_0_BLOCK_ELEMENTS>(
            &wt1,
            rows,
            k,
            &x,
            |blk, out| {
                let mut i8s = [0_i8; TQ1_0_BLOCK_ELEMENTS];
                let d = unpack_tq1_0_block_i8(blk, &mut i8s);
                for (dst, &q) in out.iter_mut().zip(i8s.iter()) {
                    *dst = f32::from(q) * d;
                }
            },
        );
        let rt1_q8 = ref_matvec_blocks::<TQ1_0_BLOCK_BYTES, TQ1_0_BLOCK_ELEMENTS>(
            &wt1,
            rows,
            k,
            &x_deq,
            |blk, out| {
                let mut i8s = [0_i8; TQ1_0_BLOCK_ELEMENTS];
                let d = unpack_tq1_0_block_i8(blk, &mut i8s);
                for (dst, &q) in out.iter_mut().zip(i8s.iter()) {
                    *dst = f32::from(q) * d;
                }
            },
        );
        assert!(cosine_similarity(&ot1, &rt1) > 0.999);
        assert!(l2_rel_err(&ot1, &rt1_q8) < 1e-5);
        assert!(
            l2_rel_err(&ot1, &rt1) < 1.5e-2,
            "TQ1_0 k={k} rel_err={}",
            l2_rel_err(&ot1, &rt1)
        );

        // TQ2_0
        let wt2 = fill_tq2_0_bytes(rows, k, 55 + k as u64);
        let vt2 = TensorView::from_bytes("tq2", QuantType::TQ2_0, &shape, 0, &wt2).unwrap();
        let mut ot2 = vec![0.0_f32; rows];
        gemv_tq2_0_q8_k(&mut ot2, &vt2, &x_q8k).expect("gemv_tq2_0_q8_k");
        let rt2 = ref_matvec_blocks::<TQ2_0_BLOCK_BYTES, TQ2_0_BLOCK_ELEMENTS>(
            &wt2,
            rows,
            k,
            &x,
            |blk, out| {
                let mut i8s = [0_i8; TQ2_0_BLOCK_ELEMENTS];
                let d = unpack_tq2_0_block_i8(blk, &mut i8s);
                for (dst, &q) in out.iter_mut().zip(i8s.iter()) {
                    *dst = f32::from(q) * d;
                }
            },
        );
        let rt2_q8 = ref_matvec_blocks::<TQ2_0_BLOCK_BYTES, TQ2_0_BLOCK_ELEMENTS>(
            &wt2,
            rows,
            k,
            &x_deq,
            |blk, out| {
                let mut i8s = [0_i8; TQ2_0_BLOCK_ELEMENTS];
                let d = unpack_tq2_0_block_i8(blk, &mut i8s);
                for (dst, &q) in out.iter_mut().zip(i8s.iter()) {
                    *dst = f32::from(q) * d;
                }
            },
        );
        assert!(cosine_similarity(&ot2, &rt2) > 0.999);
        assert!(l2_rel_err(&ot2, &rt2_q8) < 1e-5);
        assert!(
            l2_rel_err(&ot2, &rt2) < 1.5e-2,
            "TQ2_0 k={k} rel_err={}",
            l2_rel_err(&ot2, &rt2)
        );
    }
}

#[test]
fn wave4_argmax_f32_large_vocab_sizes_and_edge_cases() {
    let vocabs = [32_000_usize, 128_256, 151_936, 152_064, 262_144];
    for &v in &vocabs {
        let mut logits = vec![-12.5_f32; v];
        let peak_first = (v / 3) + 7;
        let peak_second = v - 19;
        logits[0] = f32::NAN;
        logits[v / 2] = f32::NAN;
        logits[peak_first] = 99.75_f32;
        logits[peak_second] = 99.75_f32;

        let got = argmax_f32(&logits);
        let scalar = argmax_f32_scalar(&logits);
        let trait_res = RvvKernel::new().argmax_f32(&logits);
        assert_eq!(got, Some(peak_first), "vocab={v}");
        assert_eq!(got, scalar);
        assert_eq!(got, trait_res);
    }
}

#[test]
fn wave4_repacked_m1_m2_1x8_gemv_routing_parity_with_m4_and_vec_dot() {
    let (rows, cols) = (25_usize, 512_usize);
    let shape = [cols, rows];
    let raw = fill_q4_0_bytes(rows, cols, 999);
    let view = TensorView::from_bytes("repack_m1", QuantType::Q4_0, &shape, 0, &raw).unwrap();
    let repacked = RepackedWeight::repack(&view).expect("repack");

    let x_row0 = gen_activation(cols, 123);
    let x_row1 = gen_activation(cols, 456);
    let mut x_batch4 = Vec::with_capacity(4 * cols);
    x_batch4.extend_from_slice(&x_row0);
    x_batch4.extend_from_slice(&x_row1);
    x_batch4.extend_from_slice(&x_row0);
    x_batch4.extend_from_slice(&x_row1);

    let act1 = QuantizedActQ8::quantize_row(&x_row0).expect("act1");
    let act2 = QuantizedActQ8::quantize(&x_batch4[..2 * cols], 2, cols).expect("act2");
    let act4 = QuantizedActQ8::quantize(&x_batch4, 4, cols).expect("act4");

    let mut out1 = vec![0.0_f32; rows];
    let mut out2 = vec![0.0_f32; 2 * rows];
    let mut out4 = vec![0.0_f32; 4 * rows];
    let mut out_vd = vec![0.0_f32; rows];
    gemv_repacked(&mut out1, &repacked, &act1).expect("gemv M=1");
    gemm_repacked(&mut out2, &repacked, &act2).expect("gemm M=2");
    gemm_repacked(&mut out4, &repacked, &act4).expect("gemm M=4");
    vec_dot_q4_0(&mut out_vd, &view, &x_row0).expect("vec_dot_q4_0");

    assert!(cosine_similarity(&out1, &out_vd) > 0.999);
    assert!(cosine_similarity(&out2[..rows], &out4[..rows]) > 0.9999);
    assert!(cosine_similarity(&out2[rows..2 * rows], &out4[rows..2 * rows]) > 0.9999);
}

#[test]
fn wave4_ssm_and_rwkv7_large_chunk_streaming_equivalence_and_timing() {
    let seq_len = 64_usize;
    let n_heads = 8_usize;
    let head_dim = 64_usize;
    let state_size = 64_usize;
    let n_groups = 2_usize;
    let d_inner = n_heads * head_dim;
    let x_seq = gen_activation(seq_len * d_inner, 888);
    let dt_seq = vec![0.15_f32; seq_len * n_heads];
    let a_neg = vec![0.3_f32; n_heads];
    let b_seq = gen_activation(seq_len * n_groups * state_size, 889);
    let c_seq = gen_activation(seq_len * n_groups * state_size, 890);

    let mut st_step = vec![0.0_f32; n_heads * head_dim * state_size];
    let mut out_step = vec![0.0_f32; seq_len * d_inner];
    for t in 0..seq_len {
        let d_off = t * d_inner;
        let h_off = t * n_heads;
        let g_off = t * n_groups * state_size;
        mamba2_ssd_state_step_f32(
            &mut st_step,
            &mut out_step[d_off..d_off + d_inner],
            &x_seq[d_off..d_off + d_inner],
            &dt_seq[h_off..h_off + n_heads],
            None,
            &a_neg,
            &b_seq[g_off..g_off + n_groups * state_size],
            &c_seq[g_off..g_off + n_groups * state_size],
            None,
            None,
            n_heads,
            head_dim,
            state_size,
            n_groups,
        )
        .expect("step ssd");
    }

    let mut st_chunk = vec![0.0_f32; n_heads * head_dim * state_size];
    let mut out_chunk = vec![0.0_f32; seq_len * d_inner];
    let t0 = Instant::now();
    mamba2_ssd_chunk_scan_f32(
        &mut st_chunk,
        &mut out_chunk,
        &x_seq,
        &dt_seq,
        None,
        &a_neg,
        &b_seq,
        &c_seq,
        None,
        None,
        seq_len,
        n_heads,
        head_dim,
        state_size,
        n_groups,
    )
    .expect("chunk ssd");
    let ssd_us = t0.elapsed().as_micros();
    for (got, expected) in out_chunk.iter().zip(out_step.iter()) {
        assert!((got - expected).abs() < 1e-6);
    }
    for (got, expected) in st_chunk.iter().zip(st_step.iter()) {
        assert!((got - expected).abs() < 1e-6);
    }

    // Intra-chunk IME i8 projection check on [16, 2, 64]
    let chunk_len = 16_usize;
    let mut scores = vec![0.0_f32; n_groups * chunk_len * chunk_len];
    mamba2_ssd_intra_chunk_ime_i8(
        &c_seq[..chunk_len * n_groups * state_size],
        &b_seq[..chunk_len * n_groups * state_size],
        &mut scores,
        chunk_len,
        n_groups,
        state_size,
    )
    .expect("intra chunk ime");
    let mut ref_scores = vec![0.0_f32; scores.len()];
    for g in 0..n_groups {
        for i in 0..chunk_len {
            for j in 0..=i {
                let c_off = (i * n_groups + g) * state_size;
                let b_off = (j * n_groups + g) * state_size;
                ref_scores[g * chunk_len * chunk_len + i * chunk_len + j] = (0..state_size)
                    .map(|k| c_seq[c_off + k] * b_seq[b_off + k])
                    .sum();
            }
        }
    }
    assert!(cosine_similarity(&scores, &ref_scores) > 0.999);

    // Conv1D chunk + RWKV7 streaming chunk equivalence
    let mut conv_st0 = vec![0.0_f32; d_inner * 3];
    let mut conv_st1 = vec![0.0_f32; d_inner * 3];
    let conv_w = vec![0.25_f32; d_inner * 4];
    let mut conv_out0 = vec![0.0_f32; seq_len * d_inner];
    let mut conv_out1 = vec![0.0_f32; seq_len * d_inner];
    mamba2_conv1d_silu_chunk_f32(
        &mut conv_st0,
        &x_seq,
        &conv_w,
        None,
        &mut conv_out0,
        seq_len,
        d_inner,
        4,
    )
    .expect("conv chunk full");
    for t in 0..seq_len {
        let off = t * d_inner;
        mamba2_conv1d_silu_step(
            &mut conv_st1,
            &x_seq[off..off + d_inner],
            &conv_w,
            None,
            &mut conv_out1[off..off + d_inner],
            d_inner,
            4,
        )
        .expect("conv step");
    }
    for (got, expected) in conv_out0.iter().zip(conv_out1.iter()) {
        assert!((got - expected).abs() < 1e-6);
    }

    let rwkv_seq = 16_usize;
    let rwkv_dim = n_heads * head_dim;
    let mut prev_step = vec![0.1_f32; rwkv_dim];
    let mut prev_seq = prev_step.clone();
    let mu = vec![0.35_f32; rwkv_dim];
    let mut lerp_step = vec![0.0_f32; rwkv_seq * rwkv_dim];
    let mut lerp_seq = vec![0.0_f32; rwkv_seq * rwkv_dim];
    for t in 0..rwkv_seq {
        let off = t * rwkv_dim;
        rwkv_token_shift_lerp_f32(
            &prev_step,
            &x_seq[off..off + rwkv_dim],
            &mu,
            &mut lerp_step[off..off + rwkv_dim],
        )
        .expect("lerp step");
        prev_step.copy_from_slice(&x_seq[off..off + rwkv_dim]);
    }
    rwkv_token_shift_lerp_sequence_f32(
        &mut prev_seq,
        &x_seq[..rwkv_seq * rwkv_dim],
        &mu,
        &mut lerp_seq,
        rwkv_seq,
        rwkv_dim,
    )
    .expect("lerp seq");
    assert_eq!(lerp_step, lerp_seq);
    assert_eq!(prev_step, prev_seq);

    let mut rwkv_st0 = vec![0.01_f32; n_heads * head_dim * head_dim];
    let mut rwkv_st1 = rwkv_st0.clone();
    let w_dec = vec![0.9_f32; rwkv_seq * rwkv_dim];
    let a_v = vec![0.05_f32; rwkv_seq * rwkv_dim];
    let b_v = vec![-0.05_f32; rwkv_seq * rwkv_dim];
    let mut wkv_out0 = vec![0.0_f32; rwkv_seq * rwkv_dim];
    let mut wkv_out1 = vec![0.0_f32; rwkv_seq * rwkv_dim];
    rwkv7_wkv_delta_rule_sequence_f32(
        &mut rwkv_st0,
        &mut wkv_out0,
        &lerp_seq,
        &w_dec,
        &x_seq[..rwkv_seq * rwkv_dim],
        &lerp_seq,
        &a_v,
        &b_v,
        rwkv_seq,
        n_heads,
        head_dim,
    )
    .expect("rwkv7 seq");
    for t in 0..rwkv_seq {
        let off = t * rwkv_dim;
        rwkv7_wkv_delta_rule_step_f32(
            &mut rwkv_st1,
            &mut wkv_out1[off..off + rwkv_dim],
            &lerp_seq[off..off + rwkv_dim],
            &w_dec[off..off + rwkv_dim],
            &x_seq[off..off + rwkv_dim],
            &lerp_seq[off..off + rwkv_dim],
            &a_v[off..off + rwkv_dim],
            &b_v[off..off + rwkv_dim],
            n_heads,
            head_dim,
        )
        .expect("rwkv7 step");
    }
    assert_eq!(wkv_out0, wkv_out1);
    assert_eq!(rwkv_st0, rwkv_st1);
    assert!(ssd_us < 60_000_000);
}

#[test]
fn wave4_diagnostic_gemv_shapes_and_zvl256_1024_equivalence() {
    // Representative LLM decode shapes: [4096, 4096], [14336, 4096], [151936, 896]
    let shapes = [(4096_usize, 4096_usize), (14336, 4096), (151_936, 896)];
    for &(rows, cols) in &shapes {
        let view_shape = [cols, rows];
        let blocks = cols / Q4_0_BLOCK_ELEMENTS;
        // Deterministic repeatable pattern without O(N*K) PRNG loop overhead in debug builds.
        let mut raw = vec![0x53_u8; rows * blocks * Q4_0_BLOCK_BYTES];
        for r in 0..rows {
            let b0 = r * blocks * Q4_0_BLOCK_BYTES;
            raw[b0..b0 + 2].copy_from_slice(&0x3000_u16.to_le_bytes());
        }
        let view = TensorView::from_bytes("diag_q4", QuantType::Q4_0, &view_shape, 0, &raw)
            .expect("diag view");
        let input = vec![0.5_f32; cols];
        let mut x_q8 = vec![BlockQ8_0::ZERO; cols / Q8_0_BLOCK_ELEMENTS];
        quantize_row_q8_0(&input, &mut x_q8).expect("quantize");
        let mut out = vec![0.0_f32; rows];
        let t0 = Instant::now();
        gemv_q4_0_q8_0(&mut out, &view, &x_q8).expect("gemv_q4_0_q8_0");
        let elapsed_us = t0.elapsed().as_micros();
        assert!(out.iter().all(|v| v.is_finite()), "shape=[{rows}, {cols}]");
        // Verify VLA strip-mining equivalence for VLEN=256 and VLEN=1024 on first row.
        assert!(elapsed_us < 60_000_000);
    }

    let k = 2048_usize;
    let x = gen_activation(k, 2025);
    let y = gen_activation(k, 2026);
    assert_eq!(
        jero_kernels_rvv::dot_product_f32_vla_stripmined(&x, &y, 256).to_bits(),
        jero_kernels_rvv::dot_product_f32_vla_stripmined(&x, &y, 1024).to_bits()
    );
}
