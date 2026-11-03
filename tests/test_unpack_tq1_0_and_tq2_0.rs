//! Integration tests for Milestone 3: TQ Unpacking, `BitLinear`, `SubLN`, and `ReLU^2`.
#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "test constants and assertions"
)]

use k3_core::{K3Error, QuantType, TensorView};
use kernels_rvv::ime::{
    TQ1_0_BLOCK_BYTES, TQ1_0_BLOCK_ELEMENTS, TQ2_0_BLOCK_BYTES, TQ2_0_BLOCK_ELEMENTS,
    bitlinear_i8_ime, bitlinear_i8_rvv, unpack_tq1_0_block_i8, unpack_tq2_0_block_i8,
};
use kernels_rvv::{relu_squared_gate_f32, relu_squared_gate_scalar, subln_quantize_absmax_i8};

// ============================================================================
// 1. test_unpack_tq1_0_and_tq2_0
// ============================================================================
#[test]
fn test_unpack_tq1_0_and_tq2_0() {
    // ------------------------------------------------------------------------
    // TQ2_0: 66 bytes per block = 2B FP16 scale + 64B packed 2-bit digits
    // 2-bit mapping: 0 -> -1, 1 -> 0, 2 -> +1
    // ------------------------------------------------------------------------
    let mut block_tq2 = [0u8; TQ2_0_BLOCK_BYTES];
    // Scale = 1.5 in FP16: 0x3E00
    block_tq2[0..2].copy_from_slice(&0x3E00_u16.to_le_bytes());

    // Byte pattern: 0b10_01_00_10 (elem 0: 2->+1, elem 1: 0->-1, elem 2: 1->0, elem 3: 2->+1)
    // 0x92 = 0b10010010
    for b in &mut block_tq2[2..66] {
        *b = 0x92;
    }

    let mut out_tq2 = [0_i8; TQ2_0_BLOCK_ELEMENTS];
    let scale_tq2 = unpack_tq2_0_block_i8(&block_tq2, &mut out_tq2);
    assert!((scale_tq2 - 1.5).abs() < 1e-5, "TQ2_0 scale mismatch");

    for chunk in 0..64 {
        assert_eq!(out_tq2[chunk * 4], 1, "TQ2 element 0 must be +1");
        assert_eq!(out_tq2[chunk * 4 + 1], -1, "TQ2 element 1 must be -1");
        assert_eq!(out_tq2[chunk * 4 + 2], 0, "TQ2 element 2 must be 0");
        assert_eq!(out_tq2[chunk * 4 + 3], 1, "TQ2 element 3 must be +1");
    }

    // ------------------------------------------------------------------------
    // TQ1_0: 54 bytes per block = 2B FP16 scale + 52B packed base-3 digits
    // Base-3 digits: (val % 3) - 1 in {-1, 0, +1}
    // 51 bytes * 5 = 255 digits + 1 byte with 1 digit = 256 elements
    // ------------------------------------------------------------------------
    let mut block_tq1 = [0u8; TQ1_0_BLOCK_BYTES];
    // Scale = 2.0 in FP16: 0x4000
    block_tq1[0..2].copy_from_slice(&0x4000_u16.to_le_bytes());

    // Encode base-3 pattern: d0=+1 (2), d1=-1 (0), d2=0 (1), d3=+1 (2), d4=-1 (0)
    // byte = 2 + 0*3 + 1*9 + 2*27 + 0*81 = 2 + 9 + 54 = 65
    for b in &mut block_tq1[2..53] {
        *b = 65;
    }
    // 52nd byte (index 53): 1 digit: d0 = +1 (2)
    block_tq1[53] = 2;

    let mut out_tq1 = [0_i8; TQ1_0_BLOCK_ELEMENTS];
    let scale_tq1 = unpack_tq1_0_block_i8(&block_tq1, &mut out_tq1);
    assert!((scale_tq1 - 2.0).abs() < 1e-5, "TQ1_0 scale mismatch");

    for chunk in 0..51 {
        assert_eq!(out_tq1[chunk * 5], 1, "TQ1 element 0 must be +1");
        assert_eq!(out_tq1[chunk * 5 + 1], -1, "TQ1 element 1 must be -1");
        assert_eq!(out_tq1[chunk * 5 + 2], 0, "TQ1 element 2 must be 0");
        assert_eq!(out_tq1[chunk * 5 + 3], 1, "TQ1 element 3 must be +1");
        assert_eq!(out_tq1[chunk * 5 + 4], -1, "TQ1 element 4 must be -1");
    }
    assert_eq!(out_tq1[255], 1, "TQ1 256th element must be +1");

    // Scale edge cases
    block_tq2[0..2].copy_from_slice(&0x0000_u16.to_le_bytes());
    assert_eq!(unpack_tq2_0_block_i8(&block_tq2, &mut out_tq2), 0.0);

    block_tq2[0..2].copy_from_slice(&0x7C00_u16.to_le_bytes()); // +Inf clamped
    assert_eq!(unpack_tq2_0_block_i8(&block_tq2, &mut out_tq2), 0.0625);
}

// ============================================================================
// 2. test_bitlinear_ime_rvv_bit_identical
// ============================================================================
#[test]
fn test_bitlinear_ime_rvv_bit_identical() {
    let rows = 16;
    let cols = 256;
    let mut raw_bytes = vec![0u8; rows * TQ2_0_BLOCK_BYTES]; // 16 blocks of TQ2_0

    for r in 0..rows {
        let base = r * TQ2_0_BLOCK_BYTES;
        // FP16 scale = 0.5 (0x3800)
        raw_bytes[base..base + 2].copy_from_slice(&0x3800_u16.to_le_bytes());
        for i in 0..64 {
            raw_bytes[base + 2 + i] = ((r * 13 + i * 7) % 256) as u8;
        }
    }

    let shape = [rows, cols];
    let weights = TensorView::from_bytes("bitlinear_w", QuantType::TQ2_0, &shape, 0, &raw_bytes)
        .expect("valid tensor view");

    let act_i8: Vec<i8> = (0..cols).map(|i| ((i * 11) % 255) as i8).collect();
    let act_scale = 0.007_812_5_f32; // 1.0 / 128.0

    let mut out_ime = vec![0.0_f32; rows];
    let mut out_rvv = vec![0.0_f32; rows];

    bitlinear_i8_ime(&mut out_ime, &weights, &act_i8, act_scale).expect("ime bitlinear succeeds");
    bitlinear_i8_rvv(&mut out_rvv, &weights, &act_i8, act_scale).expect("rvv bitlinear succeeds");

    for r in 0..rows {
        assert_eq!(
            out_ime[r].to_bits(),
            out_rvv[r].to_bits(),
            "Bit-identical intermediate i32 accumulation mismatch at row {r}: IME={} RVV={}",
            out_ime[r],
            out_rvv[r]
        );
    }
}

// ============================================================================
// 3. test_bitlinear_tq1_tq2_ime_vs_rvv
// ============================================================================
#[test]
fn test_bitlinear_tq1_tq2_ime_vs_rvv() {
    // Multi-block rows: 512 cols = 2 blocks per row
    let rows = 8;
    let cols = 512;

    // Test TQ1_0 multi-block
    let mut raw_tq1 = vec![0x55_u8; rows * (2 * TQ1_0_BLOCK_BYTES)];
    for blk in 0..(rows * 2) {
        let base = blk * TQ1_0_BLOCK_BYTES;
        raw_tq1[base..base + 2].copy_from_slice(&0x3C00_u16.to_le_bytes()); // 1.0
    }
    let shape_tq1 = [rows, cols];
    let weights_tq1 =
        TensorView::from_bytes("w_tq1_multi", QuantType::TQ1_0, &shape_tq1, 0, &raw_tq1)
            .expect("tq1 view");

    // Test TQ2_0 multi-block
    let mut raw_tq2 = vec![0xAA_u8; rows * (2 * TQ2_0_BLOCK_BYTES)];
    for blk in 0..(rows * 2) {
        let base = blk * TQ2_0_BLOCK_BYTES;
        raw_tq2[base..base + 2].copy_from_slice(&0x3C00_u16.to_le_bytes()); // 1.0
    }
    let shape_tq2 = [rows, cols];
    let weights_tq2 =
        TensorView::from_bytes("w_tq2_multi", QuantType::TQ2_0, &shape_tq2, 0, &raw_tq2)
            .expect("tq2 view");

    let act_i8: Vec<i8> = (0..cols).map(|i| (i % 17) as i8 - 8).collect();
    let act_scale = 0.05_f32;

    let mut out_ime_tq1 = vec![0.0_f32; rows];
    let mut out_rvv_tq1 = vec![0.0_f32; rows];
    bitlinear_i8_ime(&mut out_ime_tq1, &weights_tq1, &act_i8, act_scale).expect("tq1 ime");
    bitlinear_i8_rvv(&mut out_rvv_tq1, &weights_tq1, &act_i8, act_scale).expect("tq1 rvv");
    assert_eq!(out_ime_tq1, out_rvv_tq1);

    let mut out_ime_tq2 = vec![0.0_f32; rows];
    let mut out_rvv_tq2 = vec![0.0_f32; rows];
    bitlinear_i8_ime(&mut out_ime_tq2, &weights_tq2, &act_i8, act_scale).expect("tq2 ime");
    bitlinear_i8_rvv(&mut out_rvv_tq2, &weights_tq2, &act_i8, act_scale).expect("tq2 rvv");
    assert_eq!(out_ime_tq2, out_rvv_tq2);

    // Test unaligned row counts (e.g. rows = 7, not multiple of 8)
    let unaligned_rows = 7;
    let mut raw_unaligned = vec![0x64_u8; unaligned_rows * TQ2_0_BLOCK_BYTES];
    for r in 0..unaligned_rows {
        let base = r * TQ2_0_BLOCK_BYTES;
        raw_unaligned[base..base + 2].copy_from_slice(&0x3800_u16.to_le_bytes());
    }
    let shape_unaligned = [unaligned_rows, 256];
    let weights_unaligned = TensorView::from_bytes(
        "w_unaligned",
        QuantType::TQ2_0,
        &shape_unaligned,
        0,
        &raw_unaligned,
    )
    .expect("unaligned view");

    let act_256: Vec<i8> = (0..256).map(|i| (i % 19) as i8 - 9).collect();
    let mut out_ime_unaligned = vec![0.0_f32; unaligned_rows];
    let mut out_rvv_unaligned = vec![0.0_f32; unaligned_rows];
    bitlinear_i8_ime(
        &mut out_ime_unaligned,
        &weights_unaligned,
        &act_256,
        act_scale,
    )
    .expect("unaligned ime");
    bitlinear_i8_rvv(
        &mut out_rvv_unaligned,
        &weights_unaligned,
        &act_256,
        act_scale,
    )
    .expect("unaligned rvv");
    assert_eq!(out_ime_unaligned, out_rvv_unaligned);
}

// ============================================================================
// 4. test_tq1_0_tq2_0_bitlinear_equivalence
// ============================================================================
#[test]
fn test_tq1_0_tq2_0_bitlinear_equivalence() {
    let rows = 4;
    let cols = 256;

    // Define identical ternary values across all rows: all +1
    // Scale = 1.0 in FP16 (0x3C00)
    let mut raw_tq2 = vec![0u8; rows * TQ2_0_BLOCK_BYTES];
    for r in 0..rows {
        let base = r * TQ2_0_BLOCK_BYTES;
        raw_tq2[base..base + 2].copy_from_slice(&0x3C00_u16.to_le_bytes());
        // All +1: 2-bit value 2 -> 0b10101010 = 0xAA
        for b in &mut raw_tq2[base + 2..base + 66] {
            *b = 0xAA;
        }
    }

    let mut raw_tq1 = vec![0u8; rows * TQ1_0_BLOCK_BYTES];
    for r in 0..rows {
        let base = r * TQ1_0_BLOCK_BYTES;
        raw_tq1[base..base + 2].copy_from_slice(&0x3C00_u16.to_le_bytes());
        // All +1: base-3 digit 2 -> 2 + 6 + 18 + 54 + 162 = 242 (0xF2)
        for b in &mut raw_tq1[base + 2..base + 53] {
            *b = 242;
        }
        raw_tq1[base + 53] = 2; // 256th element
    }

    let shape = [rows, cols];
    let view_tq2 =
        TensorView::from_bytes("eq_tq2", QuantType::TQ2_0, &shape, 0, &raw_tq2).expect("view tq2");
    let view_tq1 =
        TensorView::from_bytes("eq_tq1", QuantType::TQ1_0, &shape, 0, &raw_tq1).expect("view tq1");

    let act_i8 = vec![2_i8; cols];
    let act_scale = 0.125_f32;

    let mut out_tq2 = vec![0.0_f32; rows];
    let mut out_tq1 = vec![0.0_f32; rows];

    bitlinear_i8_rvv(&mut out_tq2, &view_tq2, &act_i8, act_scale).expect("tq2 forward");
    bitlinear_i8_rvv(&mut out_tq1, &view_tq1, &act_i8, act_scale).expect("tq1 forward");

    // Expected: 256 * (+1 * 2) * (1.0 * 0.125) = 512 * 0.125 = 64.0
    for r in 0..rows {
        assert_eq!(
            out_tq1[r].to_bits(),
            out_tq2[r].to_bits(),
            "TQ1_0 and TQ2_0 outputs must be bit-identical for equivalent ternary weights"
        );
        assert!((out_tq1[r] - 64.0).abs() < 1e-4);
    }
}

// ============================================================================
// 5. test_subln_quantize_absmax_i8
// ============================================================================
#[test]
fn test_subln_quantize_absmax_i8() {
    // 5.1 Analytical test
    let input = [3.0_f32, -3.0, 3.0, -3.0];
    let weight = [1.0_f32, 1.0, 1.0, 1.0];
    let mut out_i8 = [0_i8; 4];
    let eps = 1e-5_f32;

    let scale = subln_quantize_absmax_i8(&mut out_i8, &input, &weight, eps)
        .expect("subln analytical succeeds");

    // RMS = sqrt(36 / 4) = 3.0 -> x_norm = [1.0, -1.0, 1.0, -1.0]
    // max_val = 1.0 -> scale = 1.0 / 127.0
    assert!((scale - (1.0 / 127.0)).abs() < 1e-5);
    assert_eq!(out_i8, [127, -127, 127, -127]);

    // 5.2 All-Zero safe handling
    let zero_input = [0.0_f32; 128];
    let zero_weight = [1.0_f32; 128];
    let mut zero_out = [42_i8; 128];
    let safe_scale = subln_quantize_absmax_i8(&mut zero_out, &zero_input, &zero_weight, eps)
        .expect("all-zero must succeed safely");

    assert!(safe_scale > 0.0 && safe_scale.is_finite());
    assert!(!safe_scale.is_nan());
    assert!(zero_out.iter().all(|&v| v == 0));

    // 5.3 Near-zero safe handling
    let near_zero_input = [1e-15_f32; 64];
    let mut near_zero_out = [99_i8; 64];
    let nz_scale = subln_quantize_absmax_i8(
        &mut near_zero_out,
        &near_zero_input,
        &zero_weight[..64],
        eps,
    )
    .expect("near-zero succeeds");
    assert!(nz_scale > 0.0);
    assert!(near_zero_out.iter().all(|&v| v == 0));

    // 5.4 Dimension mismatch error check
    let mismatch_in = [1.0_f32; 4];
    let mismatch_w = [1.0_f32; 3];
    let mut mismatch_out = [0_i8; 4];
    assert!(matches!(
        subln_quantize_absmax_i8(&mut mismatch_out, &mismatch_in, &mismatch_w, eps),
        Err(K3Error::TensorBoundsError { .. })
    ));
}

// ============================================================================
// 6. test_relu_squared_gate_f32
// ============================================================================
#[test]
fn test_relu_squared_gate_f32() {
    let gate = [-10.0_f32, -0.5, 0.0, 0.5, 2.0, 10.0];
    let up = [5.0_f32, 2.0, 3.0, 2.0, 3.0, 0.5];
    let mut out = [0.0_f32; 6];

    relu_squared_gate_f32(&mut out, &gate, &up).expect("relu_squared succeeds");

    // Negative / zero -> 0.0
    assert_eq!(out[0], 0.0);
    assert_eq!(out[1], 0.0);
    assert_eq!(out[2], 0.0);

    // 0.5^2 * 2.0 = 0.25 * 2.0 = 0.5
    assert!((out[3] - 0.5).abs() < 1e-6);

    // 2.0^2 * 3.0 = 4.0 * 3.0 = 12.0
    assert!((out[4] - 12.0).abs() < 1e-6);

    // 10.0^2 * 0.5 = 100.0 * 0.5 = 50.0
    assert!((out[5] - 50.0).abs() < 1e-6);

    // Equivalence between scalar and dispatch
    let mut out_scalar = [0.0_f32; 6];
    relu_squared_gate_scalar(&mut out_scalar, &gate, &up);
    assert_eq!(out, out_scalar);

    // Dimension mismatch
    let bad_gate = [1.0_f32; 3];
    assert!(matches!(
        relu_squared_gate_f32(&mut out, &bad_gate, &up),
        Err(K3Error::TensorBoundsError { .. })
    ));
}

// ============================================================================
// 7. test_subln_absmax_and_relu2
// ============================================================================
#[test]
fn test_subln_absmax_and_relu2() {
    // Simulates an end-to-end BitNet FFN forward pass:
    // x -> SubLN -> BitLinear Gate & Up -> ReLU^2 -> SubLN -> BitLinear Down
    let hidden_dim = 256;
    let intermediate_dim = 512;
    let eps = 1e-5_f32;

    let x_in: Vec<f32> = (0..hidden_dim).map(|i| (i as f32) * 0.01 - 1.28).collect();
    let ffn_sub_norm = vec![1.0_f32; hidden_dim];

    // 1. SubLN before gate and up projections
    let mut act_gate_i8 = vec![0_i8; hidden_dim];
    let mut act_up_i8 = vec![0_i8; hidden_dim];
    let scale_gate =
        subln_quantize_absmax_i8(&mut act_gate_i8, &x_in, &ffn_sub_norm, eps).expect("subln gate");
    let scale_up =
        subln_quantize_absmax_i8(&mut act_up_i8, &x_in, &ffn_sub_norm, eps).expect("subln up");
    assert_eq!(scale_gate, scale_up);
    assert_eq!(act_gate_i8, act_up_i8);

    // 2. Synthesize TQ2_0 weights for Gate and Up
    let mut raw_w = vec![0xAA_u8; intermediate_dim * TQ2_0_BLOCK_BYTES];
    for r in 0..intermediate_dim {
        let base = r * TQ2_0_BLOCK_BYTES;
        raw_w[base..base + 2].copy_from_slice(&0x3C00_u16.to_le_bytes()); // scale 1.0
    }
    let shape_gate_up = [intermediate_dim, hidden_dim];
    let w_gate = TensorView::from_bytes(
        "ffn_gate.weight",
        QuantType::TQ2_0,
        &shape_gate_up,
        0,
        &raw_w,
    )
    .expect("w_gate view");
    let w_up = TensorView::from_bytes("ffn_up.weight", QuantType::TQ2_0, &shape_gate_up, 0, &raw_w)
        .expect("w_up view");

    let mut gate_f32 = vec![0.0_f32; intermediate_dim];
    let mut up_f32 = vec![0.0_f32; intermediate_dim];
    bitlinear_i8_rvv(&mut gate_f32, &w_gate, &act_gate_i8, scale_gate).expect("bitlinear gate");
    bitlinear_i8_rvv(&mut up_f32, &w_up, &act_up_i8, scale_up).expect("bitlinear up");

    // 3. ReLU^2 activation
    let mut ffn_inter = vec![0.0_f32; intermediate_dim];
    relu_squared_gate_f32(&mut ffn_inter, &gate_f32, &up_f32).expect("relu^2 forward");

    // Verify non-negativity and finite bounds
    for &val in &ffn_inter {
        assert!(val >= 0.0);
        assert!(val.is_finite());
    }

    // 4. SubLN before down projection
    let ffn_down_sub_norm = vec![1.0_f32; intermediate_dim];
    let mut act_down_i8 = vec![0_i8; intermediate_dim];
    let scale_down =
        subln_quantize_absmax_i8(&mut act_down_i8, &ffn_inter, &ffn_down_sub_norm, eps)
            .expect("subln down");

    assert!(scale_down > 0.0 && scale_down.is_finite());

    // 5. BitLinear down projection
    let mut raw_w_down = vec![0x55_u8; hidden_dim * (2 * TQ2_0_BLOCK_BYTES)];
    for blk in 0..(hidden_dim * 2) {
        let base = blk * TQ2_0_BLOCK_BYTES;
        raw_w_down[base..base + 2].copy_from_slice(&0x3C00_u16.to_le_bytes()); // scale 1.0
    }
    let shape_down = [hidden_dim, intermediate_dim];
    let w_down = TensorView::from_bytes(
        "ffn_down.weight",
        QuantType::TQ2_0,
        &shape_down,
        0,
        &raw_w_down,
    )
    .expect("w_down view");

    let mut ffn_out = vec![0.0_f32; hidden_dim];
    bitlinear_i8_rvv(&mut ffn_out, &w_down, &act_down_i8, scale_down).expect("bitlinear down");

    assert_eq!(ffn_out.len(), hidden_dim);
    for &val in &ffn_out {
        assert!(val.is_finite());
    }
}
