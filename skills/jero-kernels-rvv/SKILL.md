---
name: jero-kernels-rvv
description: "Write and verify VLA RVV 1.0 vector kernels, SpacemiT K3 A100 XSMTVDotII matrix instructions (smt.vpack.vv / smt.vmadot), and portable scalar fallbacks in jero-kernels-rvv."
---

# `jero-kernels-rvv` VLA RVV 1.0 & `XSMTVDotII` Matrix Kernel Invariants

## 1. Variable-Length Agnostic (VLA) RVV 1.0 Strip-Mining

- All RVV 1.0 kernels (`dot_product_f32_rvv`, `rmsnorm_rvv`, `vec_dot_q4_k_m_rvv`) must use VLA strip-mining (`vsetvli`) so the exact same binary executes across Cluster 0 (`X100`, `VLEN = 256`, `vlenb = 32`) and Cluster 1 (`A100`, `VLEN = 1024`, `vlenb = 128`).
- Always maintain bit-accurate or tolerance-verified portable scalar fallbacks (`BaselineKernel`) for non-RISC-V hosts and Miri verification.

## 2. SpacemiT `XSMTVDotII` (`ime`) Hardware Rules

- `XSMTVDotII` instructions (`smt.vpack.vv` encoded as `.word 0x6704102b` and `smt.vmadot` encoded as `.word 0xe304302b`, computing `8x16 i8 * 8x16 i8 -> 8x8 i32` in `v0:v1`) are **strictly exclusive to Cluster 1 (`A100`, cores `8–15`, mask `0xFF00`)**.
- Never dispatch `ime` kernels on Cluster 0 (`X100`, cores `0–7`) or without verifying `A100` affinity and runtime capability probe passage, as Cluster 0 traps with `SIGILL`.
- **Single-Token `M <= 2` GEMV Routing**: `execute_gemm_repacked_rows` and `gemv_repacked*` route `M <= 2` away from `8x8` `smt.vmadot` tiles (which waste 7/8 rows on padding) to the dedicated `1x8` RVV 1.0 micro-kernel (`fused_tile_gemv_rvv_1x8`).

## 3. Direct Integer `Q*_x_Q8` Decode GEMV (`quant_q8`)

- **Activation Formats**: `BlockQ8_0` (`32` elements, `f32` scale + `[i8; 32]`) and `BlockQ8K` (`256` elements, `f32` scale + `[i8; 256]` + `[i16; 16]` sub-block sums). Quantize once per decode token using `quantize_row_q8_0`, `quantize_row_q8_0_f32`, or `quantize_row_q8_k` (non-finite inputs clamp to `0`).
- **4-Row Multi-Row Kernels**: `gemv_q4_0_q8_0`, `gemv_q5_0_q8_0`, `gemv_q8_0_q8_0`, `gemv_q4_k_q8_k`, `gemv_q5_k_q8_k`, `gemv_q6_k_q8_k`, `gemv_tq1_0_q8_k`, and `gemv_tq2_0_q8_k` amortize activation loads across 4 weight rows in registers (`v16..v19`) with `O(1)` stack-chunked `vec_dot_*` drivers (`vec_dot_q4_0_rvv`, `vec_dot_q5_0`, `vec_dot_q8_0_rvv`, `vec_dot_q4_k_m_rvv`, `vec_dot_q5_k_rvv`, `vec_dot_q6_k_rvv`) performing **zero heap allocations**.
- **Register-Safe Widening**: All `Q*_x_Q8` RVV paths adhere to `LMUL <= m2` for `vwmul.vv`/`vwmacc.vv` widening and keep `vl = 16` at `e8, m1` -> `e16, m2` -> `vwredsum.vs e32, m1`, safe on both `VLEN = 128`, `VLEN = 256`, and `VLEN = 1024`.

## 4. `argmax_f32`, `Mamba-2` SSD & `RWKV-7` Chunked Sequence Kernels

- **Greedy Logit Selection (`argmax_f32`)**: Vectorized two-pass VLA RVV 1.0 `vfredmax.vs` (masked against `NaN` with `vmfeq.vv`) + `vmfeq.vf` + `vfirst.m` exposed standalone and via `VectorKernel::argmax_f32`, deterministically returning the lowest index on exact ties and `None` on empty or all-`NaN` slices.
- **Zero-Allocation Chunk/Sequence Scans (`ssm`)**: `mamba2_conv1d_silu_chunk_f32`, `mamba2_ssd_chunk_scan_f32` (`F32` exponential decay with multi-group `n_groups` sharing), `mamba2_ssd_intra_chunk_ime_i8` (causal `C * B^T` stack-tiled `i8` dot product), `rwkv_token_shift_lerp_sequence_f32`, and `rwkv7_wkv_delta_rule_sequence_f32` preserve exact step-by-step recurrence equivalence without heap allocations.
