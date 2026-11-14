//! Integration test enforcing zero heap allocations inside steady-state GQA decode & prefill kernels (`#92`, `#93`).

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use jero_kernels_rvv::{
    AttentionMaskConfig, gqa_causal_attention_f32, gqa_causal_attention_prefill_f32,
    gqa_causal_attention_prefill_kv_heads_f32, gqa_causal_attention_with_scratch_f32,
    gqa_kv_head_partition,
};

struct CountingAllocator;

thread_local! {
    static COUNTING_ENABLED: Cell<bool> = const { Cell::new(false) };
    static ALLOCATED_BYTES: Cell<usize> = const { Cell::new(0) };
}

// SAFETY: Delegates directly to `std::alloc::System` while counting bytes when enabled on the calling thread.
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
    // Prime TLS slots outside measurement window.
    COUNTING_ENABLED.with(|flag| flag.set(false));
    ALLOCATED_BYTES.with(|bytes| bytes.set(0));

    COUNTING_ENABLED.with(|flag| flag.set(true));
    let result = f();
    COUNTING_ENABLED.with(|flag| flag.set(false));

    let measured = ALLOCATED_BYTES.with(Cell::get);
    (result, measured)
}

#[test]
fn gqa_decode_and_prefill_with_scratch_perform_zero_heap_allocations() {
    let m_tokens = 16;
    let total_seq = 512;
    let n_heads_q = 8;
    let n_heads_kv = 2;
    let head_dim = 64;
    let scale = 1.0_f32 / 8.0_f32;

    let q_decode = vec![0.125_f32; n_heads_q * head_dim];
    let q_prefill = vec![0.125_f32; m_tokens * n_heads_q * head_dim];
    let k_cache = vec![0.25_f32; total_seq * n_heads_kv * head_dim];
    let v_cache = vec![0.5_f32; total_seq * n_heads_kv * head_dim];
    let mut out_decode = vec![0.0_f32; n_heads_q * head_dim];
    let mut out_prefill = vec![0.0_f32; m_tokens * n_heads_q * head_dim];
    let mut scratch = vec![0.0_f32; total_seq];

    let (decode_scratch_res, decode_scratch_bytes) = measure_alloc_bytes(|| {
        gqa_causal_attention_with_scratch_f32(
            &mut out_decode,
            &q_decode,
            &k_cache,
            &v_cache,
            n_heads_q,
            n_heads_kv,
            head_dim,
            total_seq,
            scale,
            &mut scratch,
        )
    });
    decode_scratch_res.expect("decode with scratch succeeds");
    assert_eq!(decode_scratch_bytes, 0);

    // Stack-scratch path in legacy entrypoint also performs zero heap allocations for seq_len <= 2048.
    let mut out_decode_legacy = vec![0.0_f32; n_heads_q * head_dim];
    let (legacy_res, legacy_bytes) = measure_alloc_bytes(|| {
        gqa_causal_attention_f32(
            &mut out_decode_legacy,
            &q_decode,
            &k_cache,
            &v_cache,
            n_heads_q,
            n_heads_kv,
            head_dim,
            total_seq,
            scale,
        )
    });
    legacy_res.expect("legacy decode within stack scratch bound succeeds");
    assert_eq!(legacy_bytes, 0);
    assert_eq!(out_decode, out_decode_legacy);

    let cfg = AttentionMaskConfig::causal(total_seq - m_tokens, scale)
        .with_sliding_window(Some(64))
        .with_logit_softcap(Some(30.0));
    let (prefill_res, prefill_bytes) = measure_alloc_bytes(|| {
        gqa_causal_attention_prefill_f32(
            &mut out_prefill,
            &q_prefill,
            &k_cache,
            &v_cache,
            m_tokens,
            n_heads_q,
            n_heads_kv,
            head_dim,
            total_seq,
            cfg,
            &mut scratch,
        )
    });
    prefill_res.expect("prefill with scratch succeeds");
    assert_eq!(prefill_bytes, 0);

    let (kv_range, q_range) =
        gqa_kv_head_partition(n_heads_kv, n_heads_q, 1, 2).expect("partition");
    let mut out_part = vec![0.0_f32; m_tokens * q_range.len() * head_dim];
    let (part_res, part_bytes) = measure_alloc_bytes(|| {
        gqa_causal_attention_prefill_kv_heads_f32(
            &mut out_part,
            &q_prefill,
            &k_cache,
            &v_cache,
            m_tokens,
            n_heads_q,
            n_heads_kv,
            head_dim,
            total_seq,
            kv_range,
            cfg,
            &mut scratch,
        )
    });
    part_res.expect("worker-sliced prefill succeeds");
    assert_eq!(part_bytes, 0);
    assert!(out_prefill.iter().all(|v| v.is_finite()));
}
