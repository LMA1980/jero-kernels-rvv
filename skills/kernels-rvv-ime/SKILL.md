---
name: kernels-rvv-ime
description: "Write and verify VLA RVV 1.0 vector kernels, SpacemiT K3 A100 XSMTVDotII matrix instructions (smt.vpack.vv / smt.vmadot), and portable scalar fallbacks in kernels-rvv."
---

# `kernels-rvv` VLA RVV 1.0 & `XSMTVDotII` Matrix Kernel Invariants

## 1. Variable-Length Agnostic (VLA) RVV 1.0 Strip-Mining

- All RVV 1.0 kernels (`dot_product_f32_rvv`, `rmsnorm_rvv`, `vec_dot_q4_k_m_rvv`) must use VLA strip-mining (`vsetvli`) so the exact same binary executes across Cluster 0 (`X100`, `VLEN = 256`, `vlenb = 32`) and Cluster 1 (`A100`, `VLEN = 1024`, `vlenb = 128`).
- Always maintain bit-accurate or tolerance-verified portable scalar fallbacks (`BaselineKernel`) for non-RISC-V hosts and Miri verification.

## 2. SpacemiT `XSMTVDotII` (`ime`) Hardware Rules

- `XSMTVDotII` instructions (`smt.vpack.vv` encoded as `.word 0x6704102b` and `smt.vmadot` encoded as `.word 0xe304302b`, computing `8x16 i8 * 8x16 i8 -> 8x8 i32` in `v0:v1`) are **strictly exclusive to Cluster 1 (`A100`, cores `8–15`, mask `0xFF00`)**.
- Never dispatch `ime` kernels on Cluster 0 (`X100`, cores `0–7`) or without verifying `A100` affinity and runtime capability probe passage, as Cluster 0 traps with `SIGILL`.
