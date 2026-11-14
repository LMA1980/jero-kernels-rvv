//! Wave 1 integration tests covering `#69`–`#76`:
//! - Zero-allocation verification (`#[global_allocator]` thread-local counter) for `RoPE` APIs up to `head_dim = 256`
//! - Dual-layout (`RopeType::Norm` and `RopeType::NeoX`) + partial `rotary_dim <= head_dim` golden vector parity
//! - Exhaustive 65,536-value IEEE-754 vs zero-safe finite checks for `f16` and `bf16` conversions
//! - `SwiGLU` `RvvKernel` / `BaselineKernel` parity on extreme logits

use core::alloc::{GlobalAlloc, Layout};
use core::cell::Cell;
use std::alloc::System;

use jero_kernels_rvv::{
    BaselineKernel, RopeScalingConfig, RopeType, RvvKernel, VectorKernel, apply_rope_inplace,
    apply_rope_inplace_with_layout, apply_rope_with_cos_sin, apply_rope_with_inv_freq,
    apply_rope_with_inv_freq_and_layout, bf16_to_f32_finite, bf16_to_f32_ieee,
    compute_rope_cos_sin, f16_to_f32_finite, f16_to_f32_ieee, precompute_rope_inv_freq, swiglu_f32,
    swiglu_f32_scalar,
};

struct ThreadCountingAllocator;

std::thread_local! {
    static TRACK_DEPTH: Cell<usize> = const { Cell::new(0) };
    static ALLOC_COUNT: Cell<usize> = const { Cell::new(0) };
    static REALLOC_COUNT: Cell<usize> = const { Cell::new(0) };
}

// SAFETY: Delegates directly to `std::alloc::System` while incrementing const-initialized
// lock-free/no-drop thread-local counters when `TRACK_DEPTH > 0`.
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

#[test]
fn test_rope_zero_heap_allocations_up_to_head_dim_256() {
    let num_heads = 8usize;
    let kernel = RvvKernel::new();

    for &(head_dim, rotary_dim) in &[
        (64usize, 64usize),
        (64, 32),
        (96, 48),
        (128, 128),
        (128, 64),
        (256, 256),
        (256, 96),
    ] {
        let half = rotary_dim / 2;
        let inv_freq = precompute_rope_inv_freq(rotary_dim, 500_000.0, None, 4096)
            .expect("precompute inv_freq");
        let mut cos = vec![0.0_f32; half];
        let mut sin = vec![0.0_f32; half];
        let mut qk = vec![0.25_f32; num_heads * head_dim];

        // 1. compute_rope_cos_sin (inv_freq = None and Some) at pos = 0 and pos = 42
        for pos in [0usize, 42] {
            assert_zero_heap_allocations("compute_rope_cos_sin(None)", || {
                compute_rope_cos_sin(pos, 500_000.0, 1.0, None, &mut cos, &mut sin)
            })
            .expect("compute_rope_cos_sin(None) succeeds");

            assert_zero_heap_allocations("compute_rope_cos_sin(Some)", || {
                compute_rope_cos_sin(pos, 500_000.0, 1.0, Some(&inv_freq), &mut cos, &mut sin)
            })
            .expect("compute_rope_cos_sin(Some) succeeds");
        }

        // 2. apply_rope_with_cos_sin for both Norm and NeoX
        for rope_type in [RopeType::Norm, RopeType::NeoX] {
            assert_zero_heap_allocations("apply_rope_with_cos_sin", || {
                apply_rope_with_cos_sin(
                    &mut qk, num_heads, head_dim, rotary_dim, rope_type, &cos, &sin,
                )
            })
            .expect("apply_rope_with_cos_sin succeeds");

            assert_zero_heap_allocations("BaselineKernel::apply_rope_with_cos_sin", || {
                kernel.apply_rope_with_cos_sin(
                    &mut qk, num_heads, head_dim, rotary_dim, rope_type, &cos, &sin,
                )
            })
            .expect("VectorKernel apply_rope_with_cos_sin succeeds");
        }

        // 3. apply_rope_with_inv_freq_and_layout (stack cos/sin scratch)
        for rope_type in [RopeType::Norm, RopeType::NeoX] {
            assert_zero_heap_allocations("apply_rope_with_inv_freq_and_layout", || {
                apply_rope_with_inv_freq_and_layout(
                    &mut qk, head_dim, rotary_dim, rope_type, 17, &inv_freq,
                )
            })
            .expect("apply_rope_with_inv_freq_and_layout succeeds");
        }

        // 4. apply_rope_inplace_with_layout (stack inv_freq + stack cos/sin scratch)
        let smooth = RopeScalingConfig::Llama3Smooth {
            factor: 8.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position_embeddings: 8192,
        };
        assert_zero_heap_allocations("apply_rope_inplace_with_layout", || {
            apply_rope_inplace_with_layout(
                &mut qk,
                head_dim,
                rotary_dim,
                RopeType::Norm,
                19,
                16384,
                500_000.0,
                Some(&smooth),
            )
        })
        .expect("apply_rope_inplace_with_layout succeeds");

        // 5. Legacy full-dim wrappers when rotary_dim == head_dim
        if rotary_dim == head_dim {
            assert_zero_heap_allocations("apply_rope_with_inv_freq", || {
                apply_rope_with_inv_freq(&mut qk, head_dim, 11, &inv_freq)
            })
            .expect("apply_rope_with_inv_freq succeeds");

            assert_zero_heap_allocations("apply_rope_inplace", || {
                apply_rope_inplace(&mut qk, head_dim, 11, 4096, 500_000.0, None)
            })
            .expect("apply_rope_inplace succeeds");
        }
    }
}

#[test]
fn test_f16_and_bf16_exhaustive_65536_ieee_and_zero_safe_finite() {
    for raw in 0..=u16::MAX {
        // FP16 checks
        let f16_exp = (raw >> 10) & 0x1F;
        let f16_mant = raw & 0x03FF;
        let f16_ieee = f16_to_f32_ieee(raw);
        let f16_fin = f16_to_f32_finite(raw);

        if f16_exp == 0x1F {
            if f16_mant == 0 {
                assert!(f16_ieee.is_infinite(), "f16 {raw:#06x} must be Inf");
                assert_eq!(f16_ieee.is_sign_negative(), (raw >> 15) == 1);
            } else {
                assert!(f16_ieee.is_nan(), "f16 {raw:#06x} must be NaN");
            }
            assert_eq!(
                f16_fin.to_bits(),
                0.0_f32.to_bits(),
                "f16 non-finite {raw:#06x} must clamp to +0.0"
            );
        } else {
            assert!(f16_fin.is_finite(), "f16 finite {raw:#06x} must be finite");
            assert_eq!(
                f16_fin.to_bits(),
                f16_ieee.to_bits(),
                "f16 finite {raw:#06x} must match strict IEEE bit-for-bit"
            );
        }

        // BF16 checks
        let bf16_exp = (raw >> 7) & 0xFF;
        let bf16_mant = raw & 0x007F;
        let bf16_ieee = bf16_to_f32_ieee(raw);
        let bf16_fin = bf16_to_f32_finite(raw);

        if bf16_exp == 0xFF {
            if bf16_mant == 0 {
                assert!(bf16_ieee.is_infinite(), "bf16 {raw:#06x} must be Inf");
                assert_eq!(bf16_ieee.is_sign_negative(), (raw >> 15) == 1);
            } else {
                assert!(bf16_ieee.is_nan(), "bf16 {raw:#06x} must be NaN");
            }
            assert_eq!(
                bf16_fin.to_bits(),
                0.0_f32.to_bits(),
                "bf16 non-finite {raw:#06x} must clamp to +0.0"
            );
        } else {
            assert!(
                bf16_fin.is_finite(),
                "bf16 finite {raw:#06x} must be finite"
            );
            assert_eq!(
                bf16_fin.to_bits(),
                bf16_ieee.to_bits(),
                "bf16 finite {raw:#06x} must match strict IEEE bit-for-bit"
            );
        }
    }
}

#[test]
fn test_swiglu_kernel_dispatch_and_extreme_logits() {
    let kernel = BaselineKernel::new();
    for &n in &[1usize, 7, 31, 137, 4864, 14336] {
        let mut gate = Vec::with_capacity(n);
        let mut up = Vec::with_capacity(n);
        for i in 0..n {
            let g = match i % 7 {
                0 => -100.0_f32,
                1 => 100.0_f32,
                2 => 0.0_f32,
                3 => f32::MIN_POSITIVE,
                4 => -f32::MIN_POSITIVE,
                _ => {
                    #[expect(
                        clippy::cast_precision_loss,
                        reason = "small modulo value fits exact f32"
                    )]
                    let v = ((i % 41) as f32 - 20.0_f32) * 0.45_f32;
                    v
                }
            };
            gate.push(g);
            up.push(1.25_f32 - f32::from(u8::try_from(i % 5).expect("mod 5 fits u8")) * 0.5_f32);
        }

        let mut out_scalar = vec![0.0_f32; n];
        let mut out_kernel = vec![0.0_f32; n];
        let mut out_fn = vec![0.0_f32; n];

        swiglu_f32_scalar(&mut out_scalar, &gate, &up).expect("scalar swiglu succeeds");
        kernel
            .swiglu_f32(&mut out_kernel, &gate, &up)
            .expect("kernel swiglu succeeds");
        swiglu_f32(&mut out_fn, &gate, &up).expect("free swiglu succeeds");

        #[cfg(all(target_arch = "riscv64", not(miri)))]
        {
            let mut out_rvv = vec![0.0_f32; n];
            // SAFETY: `gate`, `up`, and `out_rvv` all have identical length `n`.
            unsafe { jero_kernels_rvv::swiglu_f32_rvv(&gate, &up, &mut out_rvv) };
            for i in 0..n {
                assert!(out_rvv[i].is_finite(), "rvv non-finite at n={n} i={i}");
                let tol = 1e-6_f32.max(1e-5_f32 * out_scalar[i].abs());
                assert!((out_rvv[i] - out_scalar[i]).abs() <= tol);
            }
        }

        for i in 0..n {
            assert!(out_kernel[i].is_finite());
            let tol = 1e-6_f32.max(1e-5_f32 * out_scalar[i].abs());
            assert!((out_kernel[i] - out_scalar[i]).abs() <= tol);
            assert!((out_fn[i] - out_scalar[i]).abs() <= tol);
        }
    }
}
