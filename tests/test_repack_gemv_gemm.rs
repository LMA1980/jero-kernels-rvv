//! Integration and numerical parity tests for load-time `128-byte`-aligned weight repacking,
//! dynamic `f32 -> Q8` activation quantization, row-sliced GEMV, and batched multi-token GEMM
//! across `Q4_K_M`, `Q8_0`, `Q4_0`, `Q6_K`, `Q5_K`, `F16`, `BF16`, and `F32`.

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    reason = "deterministic synthetic test generators and multi-format parity tables"
)]

use jero_core::{K3Error, QuantType, TensorView};
use jero_kernels_rvv::ime::{gemm_i8_ime, tile_dot_scalar};
use jero_kernels_rvv::{
    Aligned128, AlignedQuantMatrix, Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMENTS, Q4_K_M_BLOCK_BYTES,
    Q5_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES, Q8_0_BLOCK_BYTES, QuantizedActQ8, RepackedWeight,
    TQ1_0_BLOCK_BYTES, TQ2_0_BLOCK_BYTES, bf16_to_f32_finite, dequantize_q4_0_block,
    dequantize_q4_k_m_block, dequantize_q5_k_block, dequantize_q6_k_block, dequantize_q8_0_block,
    dot_product_f32_scalar, f16_to_f32_finite, gemm_repacked, gemm_repacked_f32,
    gemm_repacked_rows, gemm_repacked_vla_sim, gemv_repacked, gemv_repacked_rows,
    pack_tiles_aligned128, vec_dot_bf16, vec_dot_f16, vec_dot_q4_0,
};

fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x007F_FFFF;
    if exp <= 127 - 15 {
        return sign;
    }
    if exp >= 127 + 16 {
        return sign | 0x7BFF;
    }
    let f16_exp = ((exp - (127 - 15)) as u16) << 10;
    let f16_mant = (mant >> 13) as u16;
    sign | f16_exp | f16_mant
}

fn f32_to_bf16_bits(val: f32) -> u16 {
    (val.to_bits() >> 16) as u16
}

fn make_activations(m: usize, k: usize, seed: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(m * k);
    let mut state = (seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    for _ in 0..(m * k) {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let u = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
        let unit = ((u >> 40) as f32) / ((1_u32 << 24) as f32);
        out.push(unit * 2.0 - 1.0);
    }
    out
}

fn cosine_similarity_and_rel_l2(actual: &[f32], reference: &[f32]) -> (f64, f64) {
    assert_eq!(actual.len(), reference.len());
    let mut dot = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_r = 0.0_f64;
    let mut diff_sq = 0.0_f64;
    for (&a, &r) in actual.iter().zip(reference) {
        let af = f64::from(a);
        let rf = f64::from(r);
        dot += af * rf;
        norm_a += af * af;
        norm_r += rf * rf;
        let d = af - rf;
        diff_sq += d * d;
    }
    let denom = (norm_a.sqrt() * norm_r.sqrt()).max(1e-12);
    let cos = dot / denom;
    let rel_l2 = diff_sq.sqrt() / norm_r.sqrt().max(1e-12);
    (cos, rel_l2)
}

fn build_synthetic_quant_tensor(quant: QuantType, rows: usize, cols: usize, seed: u8) -> Vec<u8> {
    match quant {
        QuantType::F32 => {
            let mut raw = Vec::with_capacity(rows * cols * 4);
            for r in 0..rows {
                for c in 0..cols {
                    let v = (((r * 31 + c * 17 + usize::from(seed)) % 97) as f32 - 48.0) * 0.03125;
                    raw.extend_from_slice(&v.to_le_bytes());
                }
            }
            raw
        }
        QuantType::F16 | QuantType::BF16 => {
            let is_bf16 = quant == QuantType::BF16;
            let mut raw = Vec::with_capacity(rows * cols * 2);
            for r in 0..rows {
                for c in 0..cols {
                    let v = (((r * 29 + c * 13 + usize::from(seed)) % 89) as f32 - 44.0) * 0.0625;
                    let bits = if is_bf16 {
                        f32_to_bf16_bits(v)
                    } else {
                        f32_to_f16_bits(v)
                    };
                    raw.extend_from_slice(&bits.to_le_bytes());
                }
            }
            raw
        }
        QuantType::Q8_0 => {
            let n_blks = cols / 32;
            let mut raw = vec![0_u8; rows * n_blks * Q8_0_BLOCK_BYTES];
            for r in 0..rows {
                for b in 0..n_blks {
                    let off = (r * n_blks + b) * Q8_0_BLOCK_BYTES;
                    let d = 0.03125_f32 * (1.0 + ((r + b) % 5) as f32 * 0.25);
                    raw[off..off + 2].copy_from_slice(&f32_to_f16_bits(d).to_le_bytes());
                    for i in 0..32 {
                        let q = (((r * 19 + b * 7 + i * 5 + usize::from(seed)) % 201) as i32 - 100)
                            as i8;
                        raw[off + 2 + i] = q.to_ne_bytes()[0];
                    }
                }
            }
            raw
        }
        QuantType::Q4_0 => {
            let n_blks = cols / 32;
            let mut raw = vec![0_u8; rows * n_blks * Q4_0_BLOCK_BYTES];
            for r in 0..rows {
                for b in 0..n_blks {
                    let off = (r * n_blks + b) * Q4_0_BLOCK_BYTES;
                    let d = 0.125_f32 * (1.0 + ((r + b) % 4) as f32 * 0.5);
                    raw[off..off + 2].copy_from_slice(&f32_to_f16_bits(d).to_le_bytes());
                    for i in 0..16 {
                        let lo = ((r * 7 + b * 3 + i + usize::from(seed)) & 0x0F) as u8;
                        let hi = ((r * 11 + b * 5 + i * 3 + usize::from(seed)) & 0x0F) as u8;
                        raw[off + 2 + i] = lo | (hi << 4);
                    }
                }
            }
            raw
        }
        QuantType::Q4_K_M => {
            let n_blks = cols / 256;
            let mut raw = vec![0_u8; rows * n_blks * Q4_K_M_BLOCK_BYTES];
            for r in 0..rows {
                for b in 0..n_blks {
                    let off = (r * n_blks + b) * Q4_K_M_BLOCK_BYTES;
                    let d = 0.0625_f32 * (1.0 + ((r + b) % 3) as f32 * 0.25);
                    let dmin = 0.03125_f32 * (1.0 + ((r * 2 + b) % 3) as f32 * 0.25);
                    raw[off..off + 2].copy_from_slice(&f32_to_f16_bits(d).to_le_bytes());
                    raw[off + 2..off + 4].copy_from_slice(&f32_to_f16_bits(dmin).to_le_bytes());
                    for s in 0..12 {
                        raw[off + 4 + s] =
                            ((r * 13 + b * 17 + s * 7 + usize::from(seed) + 5) & 0x3F) as u8;
                    }
                    for i in 0..128 {
                        let lo = ((r * 5 + b * 9 + i * 3 + usize::from(seed)) & 0x0F) as u8;
                        let hi = ((r * 3 + b * 7 + i * 5 + usize::from(seed) + 2) & 0x0F) as u8;
                        raw[off + 16 + i] = lo | (hi << 4);
                    }
                }
            }
            raw
        }
        QuantType::Q5_K => {
            let n_blks = cols / 256;
            let mut raw = vec![0_u8; rows * n_blks * Q5_K_BLOCK_BYTES];
            for r in 0..rows {
                for b in 0..n_blks {
                    let off = (r * n_blks + b) * Q5_K_BLOCK_BYTES;
                    let d = 0.0625_f32;
                    let dmin = 0.03125_f32;
                    raw[off..off + 2].copy_from_slice(&f32_to_f16_bits(d).to_le_bytes());
                    raw[off + 2..off + 4].copy_from_slice(&f32_to_f16_bits(dmin).to_le_bytes());
                    for s in 0..12 {
                        raw[off + 4 + s] = ((r * 9 + b * 11 + s * 5 + 3) & 0x3F) as u8;
                    }
                    for i in 0..32 {
                        raw[off + 16 + i] =
                            ((r * 17 + b * 3 + i * 7 + usize::from(seed)) & 0xFF) as u8;
                    }
                    for i in 0..128 {
                        raw[off + 48 + i] =
                            ((r * 7 + b * 13 + i * 11 + usize::from(seed)) & 0xFF) as u8;
                    }
                }
            }
            raw
        }
        QuantType::Q6_K => {
            let n_blks = cols / 256;
            let mut raw = vec![0_u8; rows * n_blks * Q6_K_BLOCK_BYTES];
            for r in 0..rows {
                for b in 0..n_blks {
                    let off = (r * n_blks + b) * Q6_K_BLOCK_BYTES;
                    for i in 0..192 {
                        raw[off + i] = ((r * 23 + b * 19 + i * 7 + usize::from(seed)) & 0xFF) as u8;
                    }
                    for s in 0..16 {
                        let sc = (((r * 7 + b * 5 + s * 3) % 31) as i32 - 15) as i8;
                        raw[off + 192 + s] = if sc == 0 { 3 } else { sc.to_ne_bytes()[0] };
                    }
                    raw[off + 208..off + 210]
                        .copy_from_slice(&f32_to_f16_bits(0.0625).to_le_bytes());
                }
            }
            raw
        }
        QuantType::TQ1_0 => {
            let n_blks = cols / 256;
            let mut raw = vec![0_u8; rows * n_blks * TQ1_0_BLOCK_BYTES];
            for r in 0..rows {
                for b in 0..n_blks {
                    let off = (r * n_blks + b) * TQ1_0_BLOCK_BYTES;
                    for i in 0..52 {
                        raw[off + i] = ((r * 11 + b * 7 + i * 5 + usize::from(seed)) % 243) as u8;
                    }
                    raw[off + 52..off + 54].copy_from_slice(&f32_to_f16_bits(0.25).to_le_bytes());
                }
            }
            raw
        }
        QuantType::TQ2_0 => {
            let n_blks = cols / 256;
            let mut raw = vec![0_u8; rows * n_blks * TQ2_0_BLOCK_BYTES];
            for r in 0..rows {
                for b in 0..n_blks {
                    let off = (r * n_blks + b) * TQ2_0_BLOCK_BYTES;
                    for i in 0..64 {
                        let t0 = ((r + i) % 3) as u8;
                        let t1 = ((r + i + 1) % 3) as u8;
                        let t2 = ((b + i + 2) % 3) as u8;
                        let t3 = ((r + b + i) % 3) as u8;
                        raw[off + i] = t0 | (t1 << 2) | (t2 << 4) | (t3 << 6);
                    }
                    raw[off + 64..off + 66].copy_from_slice(&f32_to_f16_bits(0.25).to_le_bytes());
                }
            }
            raw
        }
    }
}

fn oracle_dequantize_matrix(quant: QuantType, raw: &[u8], rows: usize, cols: usize) -> Vec<f32> {
    let mut mat = vec![0.0_f32; rows * cols];
    match quant {
        QuantType::F32 => {
            for (idx, dst) in mat.iter_mut().enumerate() {
                let off = idx * 4;
                *dst = f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
            }
        }
        QuantType::F16 => {
            for (idx, dst) in mat.iter_mut().enumerate() {
                let off = idx * 2;
                *dst = f16_to_f32_finite(u16::from_le_bytes([raw[off], raw[off + 1]]));
            }
        }
        QuantType::BF16 => {
            for (idx, dst) in mat.iter_mut().enumerate() {
                let off = idx * 2;
                *dst = bf16_to_f32_finite(u16::from_le_bytes([raw[off], raw[off + 1]]));
            }
        }
        QuantType::Q8_0 => {
            let n_blks = cols / 32;
            for r in 0..rows {
                for b in 0..n_blks {
                    let off = (r * n_blks + b) * Q8_0_BLOCK_BYTES;
                    let out_sub: &mut [f32; 32] = (&mut mat
                        [r * cols + b * 32..r * cols + (b + 1) * 32])
                        .try_into()
                        .unwrap();
                    dequantize_q8_0_block(
                        raw[off..off + Q8_0_BLOCK_BYTES].try_into().unwrap(),
                        out_sub,
                    );
                }
            }
        }
        QuantType::Q4_0 => {
            let n_blks = cols / 32;
            for r in 0..rows {
                for b in 0..n_blks {
                    let off = (r * n_blks + b) * Q4_0_BLOCK_BYTES;
                    let out_sub: &mut [f32; 32] = (&mut mat
                        [r * cols + b * 32..r * cols + (b + 1) * 32])
                        .try_into()
                        .unwrap();
                    dequantize_q4_0_block(
                        raw[off..off + Q4_0_BLOCK_BYTES].try_into().unwrap(),
                        out_sub,
                    );
                }
            }
        }
        QuantType::Q4_K_M => {
            let n_blks = cols / 256;
            for r in 0..rows {
                for b in 0..n_blks {
                    let off = (r * n_blks + b) * Q4_K_M_BLOCK_BYTES;
                    let out_sub: &mut [f32; 256] = (&mut mat
                        [r * cols + b * 256..r * cols + (b + 1) * 256])
                        .try_into()
                        .unwrap();
                    dequantize_q4_k_m_block(
                        raw[off..off + Q4_K_M_BLOCK_BYTES].try_into().unwrap(),
                        out_sub,
                    );
                }
            }
        }
        QuantType::Q5_K => {
            let n_blks = cols / 256;
            for r in 0..rows {
                for b in 0..n_blks {
                    let off = (r * n_blks + b) * Q5_K_BLOCK_BYTES;
                    let out_sub: &mut [f32; 256] = (&mut mat
                        [r * cols + b * 256..r * cols + (b + 1) * 256])
                        .try_into()
                        .unwrap();
                    dequantize_q5_k_block(
                        raw[off..off + Q5_K_BLOCK_BYTES].try_into().unwrap(),
                        out_sub,
                    );
                }
            }
        }
        QuantType::Q6_K => {
            let n_blks = cols / 256;
            for r in 0..rows {
                for b in 0..n_blks {
                    let off = (r * n_blks + b) * Q6_K_BLOCK_BYTES;
                    let out_sub: &mut [f32; 256] = (&mut mat
                        [r * cols + b * 256..r * cols + (b + 1) * 256])
                        .try_into()
                        .unwrap();
                    dequantize_q6_k_block(
                        raw[off..off + Q6_K_BLOCK_BYTES].try_into().unwrap(),
                        out_sub,
                    );
                }
            }
        }
        QuantType::TQ1_0 | QuantType::TQ2_0 => unreachable!("handled in ternary test"),
    }
    mat
}

fn oracle_gemm_f32(w_f32: &[f32], x_f32: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; m * n];
    for tok in 0..m {
        let xr = &x_f32[tok * k..(tok + 1) * k];
        for r in 0..n {
            let wr = &w_f32[r * k..(r + 1) * k];
            out[tok * n + r] = dot_product_f32_scalar(xr, wr);
        }
    }
    out
}

#[test]
fn test_repack_alignment_and_lossless_roundtrip_all_formats() {
    let quants = [
        (QuantType::Q4_K_M, 19, 512),
        (QuantType::Q8_0, 23, 96),
        (QuantType::Q4_0, 17, 160),
        (QuantType::Q6_K, 13, 512),
        (QuantType::Q5_K, 11, 256),
        (QuantType::F16, 15, 100),
        (QuantType::BF16, 15, 100),
        (QuantType::F32, 9, 70),
    ];

    for (idx, &(quant, rows, cols)) in quants.iter().enumerate() {
        let raw = build_synthetic_quant_tensor(quant, rows, cols, (idx + 1) as u8);
        let shape = [cols, rows];
        let view = TensorView::from_bytes("w", quant, &shape, 0, &raw).unwrap();
        let repacked: AlignedQuantMatrix = RepackedWeight::from_tensor(&view, rows, cols).unwrap();

        assert_eq!(repacked.rows(), rows);
        assert_eq!(repacked.cols(), cols);
        assert_eq!(repacked.quant(), quant);
        assert_eq!(
            repacked.payload_ptr() as usize % 128,
            0,
            "{quant:?} payload must be 128-byte aligned for A100/IME"
        );
        assert_eq!(
            repacked.payload_ptr() as usize % 32,
            0,
            "{quant:?} payload must be 32-byte aligned for X100"
        );

        let oracle = oracle_dequantize_matrix(quant, &raw, rows, cols);
        let mut row_buf = vec![0.0_f32; cols];
        for r in 0..rows {
            repacked.dequantize_row(r, &mut row_buf).unwrap();
            let exp = &oracle[r * cols..(r + 1) * cols];
            for c in 0..cols {
                assert!(
                    (row_buf[c] - exp[c]).abs() < 1e-6,
                    "{quant:?} row {r} col {c}: got {}, expected {}",
                    row_buf[c],
                    exp[c]
                );
            }
        }
    }

    // Also verify ternary TQ1_0 and TQ2_0 repacking + IME tile packing alignment
    for quant in [QuantType::TQ1_0, QuantType::TQ2_0] {
        let raw = build_synthetic_quant_tensor(quant, 10, 256, 7);
        let view = TensorView::from_bytes("tq", quant, &[256, 10], 0, &raw).unwrap();
        let repacked = RepackedWeight::repack(&view).unwrap();
        assert_eq!(repacked.rows(), 10);
        assert_eq!(repacked.cols(), 256);
        assert_eq!(repacked.payload_ptr() as usize % 128, 0);
    }

    let src_i8: Vec<i8> = (0..37 * 48).map(|i| ((i % 63) as i8) - 31).collect();
    let tiles = pack_tiles_aligned128(&src_i8, 37, 48);
    assert_eq!(tiles.as_ptr() as usize % 128, 0);
    assert_eq!(tiles.len(), 37_usize.div_ceil(8) * 48_usize.div_ceil(16));

    // Verify packed IME tile equivalence against tile_dot_scalar
    let a_tiles = pack_tiles_aligned128(&src_i8[..8 * 16], 8, 16);
    let b_tiles = pack_tiles_aligned128(&src_i8[16..16 + 8 * 16], 8, 16);
    let c_scalar = tile_dot_scalar(
        Aligned128::slice_as_i8(&a_tiles),
        Aligned128::slice_as_i8(&b_tiles),
        1,
    );
    let mut c_ime = [0_i32; 64];
    gemm_i8_ime(
        Aligned128::slice_as_i8(&a_tiles),
        1,
        Aligned128::slice_as_i8(&b_tiles),
        0..1,
        1,
        |_, _, tile| c_ime = *tile,
    );
    assert_eq!(c_ime, c_scalar);
}

#[test]
fn test_standalone_vec_dot_q4_0_f16_bf16_parity() {
    assert_eq!(Q4_0_BLOCK_ELEMENTS, 32);
    assert_eq!(Q4_0_BLOCK_BYTES, 18);

    let rows = 13_usize;
    let cols = 96_usize;
    let input = make_activations(1, cols, 42);

    for quant in [QuantType::Q4_0, QuantType::F16, QuantType::BF16] {
        let raw = build_synthetic_quant_tensor(quant, rows, cols, 9);
        let shape_gguf = [cols, rows];
        let shape_rm = [rows, cols];
        let view_gguf = TensorView::from_bytes("w_gguf", quant, &shape_gguf, 0, &raw).unwrap();
        let view_rm = TensorView::from_bytes("w_rm", quant, &shape_rm, 0, &raw).unwrap();
        let oracle_w = oracle_dequantize_matrix(quant, &raw, rows, cols);
        let expected = oracle_gemm_f32(&oracle_w, &input, 1, rows, cols);

        let mut out_gguf = vec![0.0_f32; rows];
        let mut out_rm = vec![0.0_f32; rows];
        match quant {
            QuantType::Q4_0 => {
                vec_dot_q4_0(&mut out_gguf, &view_gguf, &input).unwrap();
                vec_dot_q4_0(&mut out_rm, &view_rm, &input).unwrap();
            }
            QuantType::F16 => {
                vec_dot_f16(&mut out_gguf, &view_gguf, &input).unwrap();
                vec_dot_f16(&mut out_rm, &view_rm, &input).unwrap();
            }
            QuantType::BF16 => {
                vec_dot_bf16(&mut out_gguf, &view_gguf, &input).unwrap();
                vec_dot_bf16(&mut out_rm, &view_rm, &input).unwrap();
            }
            _ => unreachable!(),
        }

        assert_eq!(out_gguf, out_rm);
        for r in 0..rows {
            assert!(
                (out_gguf[r] - expected[r]).abs() < 1e-4,
                "{quant:?} row {r}: got {}, expected {}",
                out_gguf[r],
                expected[r]
            );
        }
    }
}

#[test]
fn test_repacked_gemv_and_batched_gemm_parity_all_m_and_quants() {
    let quants = [
        (QuantType::Q4_K_M, 512_usize, 23_usize, 1.5e-2_f64),
        (QuantType::Q8_0, 256, 19, 1.2e-2),
        (QuantType::Q4_0, 256, 21, 1.2e-2),
        (QuantType::Q6_K, 512, 17, 1.2e-2),
        (QuantType::Q5_K, 256, 15, 1.5e-2),
        (QuantType::F16, 192, 17, 1e-5),
        (QuantType::BF16, 192, 17, 1e-5),
        (QuantType::F32, 192, 17, 1e-5),
    ];

    for (q_idx, &(quant, cols, rows, max_rel_l2)) in quants.iter().enumerate() {
        let raw = build_synthetic_quant_tensor(quant, rows, cols, (q_idx + 3) as u8);
        let shape = [cols, rows];
        let view = TensorView::from_bytes("w_batch", quant, &shape, 0, &raw).unwrap();
        let repacked = RepackedWeight::from_tensor(&view, rows, cols).unwrap();
        let oracle_w = oracle_dequantize_matrix(quant, &raw, rows, cols);

        // Issue #67: verify M in {1, 5, 8, 17, 64}
        for &m in &[1_usize, 5, 8, 17, 64] {
            let x = make_activations(m, cols, 100 + q_idx * 10 + m);
            let act = QuantizedActQ8::quantize(&x, m, cols).unwrap();
            let expected = oracle_gemm_f32(&oracle_w, &x, m, rows, cols);

            let mut actual = vec![0.0_f32; m * rows];
            gemm_repacked(&mut actual, &repacked, &act).unwrap();

            let (cos, rel_l2) = cosine_similarity_and_rel_l2(&actual, &expected);
            assert!(
                cos > 0.9995,
                "{quant:?} M={m} cosine similarity {cos} <= 0.9995"
            );
            assert!(
                rel_l2 < max_rel_l2,
                "{quant:?} M={m} relative L2 error {rel_l2} >= {max_rel_l2}"
            );

            // Cross-cluster VLA parity: 256-bit X100 vs 1024-bit A100 / IME must match exactly
            let mut vla256 = vec![0.0_f32; m * rows];
            let mut vla1024 = vec![0.0_f32; m * rows];
            gemm_repacked_vla_sim(&mut vla256, &repacked, &act, 256).unwrap();
            gemm_repacked_vla_sim(&mut vla1024, &repacked, &act, 1024).unwrap();
            assert_eq!(
                vla256, vla1024,
                "{quant:?} M={m} 256b X100 vs 1024b A100 simulation diverged"
            );

            // When M == 1, gemv_repacked and row-sliced gemv_repacked_rows must match gemm_repacked
            if m == 1 {
                let mut gemv_out = vec![0.0_f32; rows];
                gemv_repacked(&mut gemv_out, &repacked, &act).unwrap();
                assert_eq!(gemv_out, actual);

                let mut stitched = vec![0.0_f32; rows];
                gemv_repacked_rows(&mut stitched[..8], &repacked, &act, 0..8).unwrap();
                gemv_repacked_rows(&mut stitched[8..], &repacked, &act, 8..rows).unwrap();
                assert_eq!(stitched, gemv_out);
            }

            // Row-sliced batched GEMM (both TILE_N-aligned 0..8 and unaligned 5..rows)
            let mid = 8.min(rows);
            let mut top_slice = vec![0.0_f32; m * mid];
            let mut bot_slice = vec![0.0_f32; m * (rows - mid)];
            gemm_repacked_rows(&mut top_slice, &repacked, &act, 0..mid).unwrap();
            gemm_repacked_rows(&mut bot_slice, &repacked, &act, mid..rows).unwrap();
            for tok in 0..m {
                assert_eq!(
                    &top_slice[tok * mid..(tok + 1) * mid],
                    &actual[tok * rows..tok * rows + mid]
                );
                assert_eq!(
                    &bot_slice[tok * (rows - mid)..(tok + 1) * (rows - mid)],
                    &actual[tok * rows + mid..(tok + 1) * rows]
                );
            }
        }
    }
}

#[test]
fn test_non_256_multiple_k_and_n_dimensions_896_1536_4096() {
    // Issue #67: Qwen2.5-0.5B (896 = 28 * 32, NOT a multiple of 256!), Qwen2.5-1.5B (1536),
    // and Llama-3-8B (4096).
    for &(quant, cols, rows) in &[
        (QuantType::Q8_0, 896_usize, 29_usize),
        (QuantType::Q4_0, 896, 17),
        (QuantType::F16, 896, 19),
        (QuantType::BF16, 896, 19),
        (QuantType::Q4_K_M, 1536, 19),
        (QuantType::Q6_K, 1536, 17),
        (QuantType::Q4_K_M, 4096, 9),
        (QuantType::Q8_0, 4096, 9),
    ] {
        let raw = build_synthetic_quant_tensor(quant, rows, cols, 11);
        let shape = [cols, rows];
        let view = TensorView::from_bytes("w_model_dim", quant, &shape, 0, &raw).unwrap();
        let repacked = RepackedWeight::repack(&view).unwrap();
        assert_eq!(repacked.rows(), rows);
        assert_eq!(repacked.cols(), cols);

        let oracle_w = oracle_dequantize_matrix(quant, &raw, rows, cols);
        let m = 5_usize;
        let x = make_activations(m, cols, cols + rows);
        let expected = oracle_gemm_f32(&oracle_w, &x, m, rows, cols);

        let mut actual = vec![0.0_f32; m * rows];
        gemm_repacked_f32(&mut actual, &repacked, &x, m).unwrap();

        let (cos, rel_l2) = cosine_similarity_and_rel_l2(&actual, &expected);
        assert!(
            cos > 0.9995 && rel_l2 < 1.5e-2,
            "{quant:?} [{rows}, {cols}] M={m}: cos={cos}, rel_l2={rel_l2}"
        );
    }

    // Compact DenseHalf mode on F16 with non-32-multiple K (899)
    let raw_f16 = build_synthetic_quant_tensor(QuantType::F16, 11, 899, 5);
    let view_f16 =
        TensorView::from_bytes("w_f16_compact", QuantType::F16, &[899, 11], 0, &raw_f16).unwrap();
    let compact_f16 = RepackedWeight::from_tensor_with_half_limit(&view_f16, 11, 899, 0).unwrap();
    assert!(compact_f16.is_compact_half());

    let oracle_f16 = oracle_dequantize_matrix(QuantType::F16, &raw_f16, 11, 899);
    let x_f16 = make_activations(7, 899, 777);
    let exp_f16 = oracle_gemm_f32(&oracle_f16, &x_f16, 7, 11, 899);
    let mut act_f16 = vec![0.0_f32; 7 * 11];
    gemm_repacked_f32(&mut act_f16, &compact_f16, &x_f16, 7).unwrap();
    let (cos_f16, rel_f16) = cosine_similarity_and_rel_l2(&act_f16, &exp_f16);
    assert!(cos_f16 > 0.999_999 && rel_f16 < 1e-5);
}

#[test]
fn test_repack_and_gemm_bounds_errors() {
    let raw = build_synthetic_quant_tensor(QuantType::Q8_0, 8, 64, 1);
    let view = TensorView::from_bytes("w_err", QuantType::Q8_0, &[64, 8], 0, &raw).unwrap();
    let repacked = RepackedWeight::repack(&view).unwrap();

    // K mismatch
    let bad_act = QuantizedActQ8::quantize(&[1.0_f32; 32], 1, 32).unwrap();
    let mut out = [0.0_f32; 8];
    assert!(matches!(
        gemv_repacked(&mut out, &repacked, &bad_act),
        Err(K3Error::TensorBoundsError { .. })
    ));

    // M != 1 on gemv_repacked
    let m2_act = QuantizedActQ8::quantize(&[1.0_f32; 128], 2, 64).unwrap();
    assert!(matches!(
        gemv_repacked(&mut out, &repacked, &m2_act),
        Err(K3Error::TensorBoundsError { .. })
    ));

    // Out-of-range row slice
    let ok_act = QuantizedActQ8::quantize(&[1.0_f32; 64], 1, 64).unwrap();
    assert!(matches!(
        gemm_repacked_rows(&mut out, &repacked, &ok_act, 2..10),
        Err(K3Error::TensorBoundsError { .. })
    ));

    // Output slice length mismatch
    let mut short_out = [0.0_f32; 7];
    assert!(matches!(
        gemm_repacked(&mut short_out, &repacked, &ok_act),
        Err(K3Error::TensorBoundsError { .. })
    ));
}
