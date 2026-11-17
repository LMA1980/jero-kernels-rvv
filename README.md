# jero-kernels-rvv

<p align="center">
  <a href="https://hook.lma1980.duckdns.org/jobs">
    <img src="https://hook.lma1980.duckdns.org/cards/LMA1980/jero-kernels-rvv/card.svg" alt="jero-kernels-rvv Factory Scorecard" width="520"/>
  </a>
</p>

<p align="center">
  <a href="https://hook.lma1980.duckdns.org/jobs"><img src="https://hook.lma1980.duckdns.org/badges/LMA1980/jero-kernels-rvv/status.svg" alt="CI Status"/></a>
  <a href="https://hook.lma1980.duckdns.org/jobs"><img src="https://hook.lma1980.duckdns.org/badges/LMA1980/jero-kernels-rvv/score.svg" alt="Quality Score"/></a>
  <a href="https://hook.lma1980.duckdns.org/jobs"><img src="https://hook.lma1980.duckdns.org/badges/LMA1980/jero-kernels-rvv/coverage.svg" alt="Coverage"/></a>
  <a href="https://hook.lma1980.duckdns.org/jobs"><img src="https://hook.lma1980.duckdns.org/badges/LMA1980/jero-kernels-rvv/benchmark.svg" alt="Benchmark"/></a>
</p>

______________________________________________________________________

VLA RVV 1.0 compute kernels with portable scalar fallbacks for the SpacemiT K3 platform.

## Overview

`jero-kernels-rvv` provides vector-length agnostic (`VLA`) RISC-V RVV 1.0 compute kernels compatible with both `zvl256b` (X100) and `zvl1024b` (A100), alongside bit-equivalent portable scalar fallbacks:

- **`attention`**: Causal/bidirectional GQA, MLA absorbed attention, DeBERTa disentangled attention, and gated DeltaNet steps.
- **`ime`**: A100 `XSMTVDotII` matrix dot primitives (`bitlinear_i8_ime`, `tile_dot_rvv`, 128-byte tile packing).
- **`norm`**: `RMSNorm`, L2 normalization, and in-place QK normalization.
- **`quant` & `repack`**: Block dequantization, Q8 activation quantization, 128-byte aligned tile repacking, and `vec_dot`/`gemv`/`gemm` across `F16`, `BF16`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `IQ4_NL`, `TQ1_0`, `TQ2_0`, and `I2_S`.
- **`residual`, `rope`, `ssm`, & `swiglu`**: nGPT residual LERP, `Norm`/`NeoX` RoPE rotations, Mamba-2 SSD / LFM2 shortconv / RWKV-7 WKV delta-rule state updates, and SwiGLU activations.

## Ecosystem & Documentation

- **Public Documentation Portal**: [`lma1980.github.io/model-runner`](https://github.com/LMA1980/lma1980.github.io/tree/main/model-runner) ([Reference Manual](https://github.com/LMA1980/lma1980.github.io/blob/main/model-runner/manual.md), [Crate Docs](https://github.com/LMA1980/lma1980.github.io/tree/main/model-runner/docs/crates))
- **Upstream Dependency**: [`jero-core`](https://github.com/LMA1980/jero-core)
- **Downstream Consumers**: [`jero-engine`](https://github.com/LMA1980/jero-engine), [`jero-manager`](https://github.com/LMA1980/jero-manager), and the architecture recipe crates ([`jero-arch-synthetic`](https://github.com/LMA1980/jero-arch-synthetic), [`jero-arch-llama`](https://github.com/LMA1980/jero-arch-llama), [`jero-arch-mistral`](https://github.com/LMA1980/jero-arch-mistral), [`jero-arch-phi3`](https://github.com/LMA1980/jero-arch-phi3), [`jero-arch-bitnet`](https://github.com/LMA1980/jero-arch-bitnet), [`jero-arch-dory`](https://github.com/LMA1980/jero-arch-dory))

## License

Licensed under the GNU Lesser General Public License v3.0 with the LGPL-3.0 Linking Exception (`LGPL-3.0-only WITH LGPL-3.0-linking-exception`). See [LICENSE](LICENSE) for details.
