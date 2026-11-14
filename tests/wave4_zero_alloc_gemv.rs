//! Integration test enforcing zero heap allocations in Wave 4 GEMV, `argmax_f32`,
//! `M <= 2` repacked `1x8` GEMV, and `Mamba-2`/`RWKV-7` chunked sequence scan kernels.

#![expect(
    clippy::cast_precision_loss,
    reason = "deterministic test buffer setup outside allocation measurement windows"
)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use jero_core::{QuantType, TensorView};
use jero_kernels_rvv::{
    BaselineKernel, BlockQ8_0, BlockQ8K, Q4_0_BLOCK_BYTES, Q4_K_M_BLOCK_BYTES, Q5_0_BLOCK_BYTES,
    Q5_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES, Q8_0_BLOCK_BYTES, QuantizedActQ8, RepackedWeight,
    TQ1_0_BLOCK_BYTES, TQ2_0_BLOCK_BYTES, VectorKernel, argmax_f32, gemm_repacked, gemv_q4_0_q8_0,
    gemv_q4_0_q8_0_4x, gemv_q4_k_q8_k, gemv_q5_0_q8_0, gemv_q5_0_q8_0_4x, gemv_q5_k_q8_k,
    gemv_q6_k_q8_k, gemv_q8_0_q8_0, gemv_q8_0_q8_0_4x, gemv_repacked, gemv_tq1_0_q8_k,
    gemv_tq2_0_q8_k, mamba2_conv1d_silu_chunk_f32, mamba2_ssd_chunk_scan_f32,
    mamba2_ssd_intra_chunk_ime_i8, quantize_row_q8_0, quantize_row_q8_0_f32, quantize_row_q8_k,
    rwkv_token_shift_lerp_sequence_f32, rwkv7_wkv_delta_rule_sequence_f32, vec_dot_q4_0,
    vec_dot_q4_0_q8_0, vec_dot_q4_k_q8_k, vec_dot_q5_0, vec_dot_q5_0_q8_0_blocks, vec_dot_q5_k,
    vec_dot_q5_k_q8_k, vec_dot_q6_k, vec_dot_q6_k_q8_k, vec_dot_q8_0, vec_dot_q8_0_q8_0,
    vec_dot_tq1_0_q8_k, vec_dot_tq2_0_q8_k,
};

struct CountingAllocator;

thread_local! {
    static COUNTING_ENABLED: Cell<bool> = const { Cell::new(false) };
    static ALLOCATED_BYTES: Cell<usize> = const { Cell::new(0) };
}

// SAFETY: Delegates directly to `std::alloc::System` while accounting allocations on opt-in threads.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        COUNTING_ENABLED.with(|enabled| {
            if enabled.get() {
                ALLOCATED_BYTES.with(|bytes| bytes.set(bytes.get().saturating_add(layout.size())));
            }
        });
        // SAFETY: Caller satisfies `GlobalAlloc::alloc` layout invariants; delegated to `System`.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        COUNTING_ENABLED.with(|enabled| {
            if enabled.get() {
                ALLOCATED_BYTES.with(|bytes| bytes.set(bytes.get().saturating_add(layout.size())));
            }
        });
        // SAFETY: Caller satisfies `GlobalAlloc::alloc_zeroed` layout invariants; delegated to `System`.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: Caller guarantees `ptr` was allocated by this allocator with `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        COUNTING_ENABLED.with(|enabled| {
            if enabled.get() && new_size > layout.size() {
                ALLOCATED_BYTES
                    .with(|bytes| bytes.set(bytes.get().saturating_add(new_size - layout.size())));
            }
        });
        // SAFETY: Caller satisfies `GlobalAlloc::realloc` invariants; delegated to `System`.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn measure_alloc_bytes<T>(f: impl FnOnce() -> T) -> (T, usize) {
    COUNTING_ENABLED.with(|flag| flag.set(false));
    ALLOCATED_BYTES.with(|bytes| bytes.set(0));
    COUNTING_ENABLED.with(|flag| flag.set(true));
    let result = f();
    COUNTING_ENABLED.with(|flag| flag.set(false));
    (result, ALLOCATED_BYTES.with(Cell::get))
}

#[test]
fn wave4_quant_q8_gemv_argmax_and_repack_m1_perform_zero_heap_allocations() {
    let rows = 13_usize;
    let cols = 512_usize;
    let blocks_32 = cols / 32;
    let blocks_256 = cols / 256;

    let input: Vec<f32> = (0..cols)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.125)
        .collect();
    let mut q8_0 = vec![BlockQ8_0::ZERO; blocks_32];
    let mut q8_k = vec![BlockQ8K::ZERO; blocks_256];
    let mut out = vec![0.0_f32; rows];
    let mut out2 = vec![0.0_f32; 2 * rows];

    let w_q4_0 = vec![0x3C_u8; rows * blocks_32 * Q4_0_BLOCK_BYTES];
    let w_q5_0 = vec![0x3C_u8; rows * blocks_32 * Q5_0_BLOCK_BYTES];
    let w_q8_0 = vec![0x3C_u8; rows * blocks_32 * Q8_0_BLOCK_BYTES];
    let w_q4_k = vec![0x3C_u8; rows * blocks_256 * Q4_K_M_BLOCK_BYTES];
    let w_q5_k = vec![0x3C_u8; rows * blocks_256 * Q5_K_BLOCK_BYTES];
    let w_q6_k = vec![0x3C_u8; rows * blocks_256 * Q6_K_BLOCK_BYTES];
    let w_tq1 = vec![0x3C_u8; rows * blocks_256 * TQ1_0_BLOCK_BYTES];
    let w_tq2 = vec![0x3C_u8; rows * blocks_256 * TQ2_0_BLOCK_BYTES];

    let shape = [cols, rows];
    let v_q4_0 = TensorView::from_bytes("q4_0", QuantType::Q4_0, &shape, 0, &w_q4_0).unwrap();
    let v_q5_0 = TensorView::from_bytes("q5_0", QuantType::Q5_0, &shape, 0, &w_q5_0).unwrap();
    let v_q8_0 = TensorView::from_bytes("q8_0", QuantType::Q8_0, &shape, 0, &w_q8_0).unwrap();
    let v_q4_k = TensorView::from_bytes("q4_k", QuantType::Q4_K_M, &shape, 0, &w_q4_k).unwrap();
    let v_q5_k = TensorView::from_bytes("q5_k", QuantType::Q5_K, &shape, 0, &w_q5_k).unwrap();
    let v_q6_k = TensorView::from_bytes("q6_k", QuantType::Q6_K, &shape, 0, &w_q6_k).unwrap();
    let v_tq1 = TensorView::from_bytes("tq1", QuantType::TQ1_0, &shape, 0, &w_tq1).unwrap();
    let v_tq2 = TensorView::from_bytes("tq2", QuantType::TQ2_0, &shape, 0, &w_tq2).unwrap();

    let repacked = RepackedWeight::repack(&v_q4_0).expect("repack");
    let act1 = QuantizedActQ8::quantize_row(&input).expect("act1");
    let input2: Vec<f32> = input.iter().chain(input.iter()).copied().collect();
    let act2 = QuantizedActQ8::quantize(&input2, 2, cols).expect("act2");
    let logits = vec![0.5_f32; 32_000];
    let kernel = BaselineKernel::new();

    let ((), bytes) = measure_alloc_bytes(|| {
        quantize_row_q8_0(&input, &mut q8_0).expect("quantize q8_0");
        quantize_row_q8_0_f32(&input, &mut q8_0).expect("quantize q8_0_f32");
        quantize_row_q8_k(&input, &mut q8_k).expect("quantize q8_k");

        let _ = vec_dot_q4_0_q8_0(&w_q4_0[..blocks_32 * Q4_0_BLOCK_BYTES], &q8_0).unwrap();
        let _ = vec_dot_q5_0_q8_0_blocks(&w_q5_0[..blocks_32 * Q5_0_BLOCK_BYTES], &q8_0).unwrap();
        let _ = vec_dot_q8_0_q8_0(&w_q8_0[..blocks_32 * Q8_0_BLOCK_BYTES], &q8_0).unwrap();
        let _ = vec_dot_q4_k_q8_k(&w_q4_k[..blocks_256 * Q4_K_M_BLOCK_BYTES], &q8_k).unwrap();
        let _ = vec_dot_q5_k_q8_k(&w_q5_k[..blocks_256 * Q5_K_BLOCK_BYTES], &q8_k).unwrap();
        let _ = vec_dot_q6_k_q8_k(&w_q6_k[..blocks_256 * Q6_K_BLOCK_BYTES], &q8_k).unwrap();
        let _ = vec_dot_tq1_0_q8_k(&w_tq1[..blocks_256 * TQ1_0_BLOCK_BYTES], &q8_k).unwrap();
        let _ = vec_dot_tq2_0_q8_k(&w_tq2[..blocks_256 * TQ2_0_BLOCK_BYTES], &q8_k).unwrap();

        gemv_q4_0_q8_0(&mut out, &v_q4_0, &q8_0).unwrap();
        gemv_q4_0_q8_0_4x(&mut out, &v_q4_0, &q8_0).unwrap();
        gemv_q5_0_q8_0(&mut out, &v_q5_0, &q8_0).unwrap();
        gemv_q5_0_q8_0_4x(&mut out, &v_q5_0, &q8_0).unwrap();
        gemv_q8_0_q8_0(&mut out, &v_q8_0, &q8_0).unwrap();
        gemv_q8_0_q8_0_4x(&mut out, &v_q8_0, &q8_0).unwrap();
        gemv_q4_k_q8_k(&mut out, &v_q4_k, &q8_k).unwrap();
        gemv_q5_k_q8_k(&mut out, &v_q5_k, &q8_k).unwrap();
        gemv_q6_k_q8_k(&mut out, &v_q6_k, &q8_k).unwrap();
        gemv_tq1_0_q8_k(&mut out, &v_tq1, &q8_k).unwrap();
        gemv_tq2_0_q8_k(&mut out, &v_tq2, &q8_k).unwrap();

        vec_dot_q4_0(&mut out, &v_q4_0, &input).unwrap();
        vec_dot_q5_0(&mut out, &v_q5_0, &input).unwrap();
        vec_dot_q8_0(&mut out, &v_q8_0, &input).unwrap();
        kernel.vec_dot_q4_k_m(&mut out, &v_q4_k, &input).unwrap();
        vec_dot_q5_k(&mut out, &v_q5_k, &input).unwrap();
        vec_dot_q6_k(&mut out, &v_q6_k, &input).unwrap();

        gemv_repacked(&mut out, &repacked, &act1).unwrap();
        gemm_repacked(&mut out2, &repacked, &act2).unwrap();
        let _ = argmax_f32(&logits);
    });
    assert_eq!(
        bytes, 0,
        "expected 0 heap allocations in GEMV/argmax hot paths"
    );
}

#[test]
fn wave4_ssm_and_rwkv7_chunk_scans_perform_zero_heap_allocations() {
    let seq_len = 8_usize;
    let n_heads = 4_usize;
    let head_dim = 16_usize;
    let state_size = 16_usize;
    let n_groups = 2_usize;
    let d_inner = n_heads * head_dim;

    let mut conv_state = vec![0.0_f32; d_inner * 3];
    let x_seq = vec![0.25_f32; seq_len * d_inner];
    let conv_w = vec![0.5_f32; d_inner * 4];
    let mut conv_out = vec![0.0_f32; seq_len * d_inner];

    let mut ssd_state = vec![0.0_f32; n_heads * head_dim * state_size];
    let mut ssd_out = vec![0.0_f32; seq_len * d_inner];
    let dt_seq = vec![0.1_f32; seq_len * n_heads];
    let a_neg = vec![0.2_f32; n_heads];
    let b_seq = vec![0.3_f32; seq_len * n_groups * state_size];
    let c_seq = vec![0.4_f32; seq_len * n_groups * state_size];
    let mut intra_scores = vec![0.0_f32; n_groups * seq_len * seq_len];

    let mut prev_x = vec![0.1_f32; d_inner];
    let mu = vec![0.5_f32; d_inner];
    let mut lerp_out = vec![0.0_f32; seq_len * d_inner];
    let mut rwkv_state = vec![0.0_f32; n_heads * head_dim * head_dim];
    let mut rwkv_out = vec![0.0_f32; seq_len * d_inner];
    let w_dec = vec![0.9_f32; seq_len * d_inner];

    let ((), bytes) = measure_alloc_bytes(|| {
        mamba2_conv1d_silu_chunk_f32(
            &mut conv_state,
            &x_seq,
            &conv_w,
            None,
            &mut conv_out,
            seq_len,
            d_inner,
            4,
        )
        .unwrap();
        mamba2_ssd_chunk_scan_f32(
            &mut ssd_state,
            &mut ssd_out,
            &conv_out,
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
        .unwrap();
        mamba2_ssd_intra_chunk_ime_i8(
            &c_seq,
            &b_seq,
            &mut intra_scores,
            seq_len,
            n_groups,
            state_size,
        )
        .unwrap();
        rwkv_token_shift_lerp_sequence_f32(
            &mut prev_x,
            &x_seq,
            &mu,
            &mut lerp_out,
            seq_len,
            d_inner,
        )
        .unwrap();
        rwkv7_wkv_delta_rule_sequence_f32(
            &mut rwkv_state,
            &mut rwkv_out,
            &lerp_out,
            &w_dec,
            &x_seq,
            &lerp_out,
            &x_seq,
            &x_seq,
            seq_len,
            n_heads,
            head_dim,
        )
        .unwrap();
    });
    assert_eq!(
        bytes, 0,
        "expected 0 heap allocations in SSM/RWKV7 chunk kernels"
    );
}
