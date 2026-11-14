//! `SwiGLU` activation and finite BF16 conversion helpers.
//!
//! Provides numerically stable `SiLU(gate) * up` gated activations and non-poisoning
//! BF16-to-F32 conversions that guarantee deterministic execution across transformer forward passes.

use jero_core::K3Error;

/// Converts a 16-bit brain floating-point (`BF16`) bit pattern to strict IEEE-754 `f32`.
///
/// Preserves all normal values, subnormals, signed zeroes, infinities, and `NaN` bit patterns.
#[must_use]
pub const fn bf16_to_f32_ieee(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Converts a 16-bit brain floating-point (`BF16`) bit pattern to single-precision `f32`.
///
/// Non-finite values (`NaN`, positive/negative `Inf`) clamp safely to `0.0_f32` so invalid
/// block scales collapse to zero outputs rather than fabricating non-zero weights or poisoning
/// activations with `NaN`/`Inf`. Finite normals, subnormals, and signed zeroes are preserved
/// bit-for-bit.
#[must_use]
pub const fn bf16_to_f32_finite(bits: u16) -> f32 {
    let exp = (bits >> 7) & 0xFF;
    if exp == 0xFF {
        0.0_f32
    } else {
        bf16_to_f32_ieee(bits)
    }
}

fn validate_gated_activation_args(
    op_name: &str,
    out: &[f32],
    gate: &[f32],
    up: &[f32],
) -> Result<usize, K3Error> {
    let len = gate.len();
    if len == 0 || out.len() != len || up.len() != len {
        return Err(K3Error::TensorBoundsError {
            tensor_name: op_name.to_owned(),
            offset: 0,
            byte_len: size_of_val(out),
            buffer_len: size_of_val(gate),
            reason: format!(
                "dimension mismatch: gate={len}, up={}, out={}",
                up.len(),
                out.len()
            ),
        });
    }
    Ok(len)
}

/// Computes the gated `SwiGLU` activation $y = \text{SiLU}(gate) \odot up = \frac{gate}{1 + e^{-gate}} \odot up$.
///
/// Dispatches to [`swiglu_f32_rvv`] on `riscv64` targets and [`swiglu_f32_scalar`] elsewhere.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `out`, `gate`, and `up` lengths do not match or are empty.
pub fn swiglu_f32(out: &mut [f32], gate: &[f32], up: &[f32]) -> Result<(), K3Error> {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        let _ = validate_gated_activation_args("swiglu", out, gate, up)?;
        // SAFETY: `validate_gated_activation_args` just verified `out`, `gate`, and `up` have identical non-zero length.
        unsafe { swiglu_f32_rvv(gate, up, out) };
        Ok(())
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        swiglu_f32_scalar(out, gate, up)
    }
}

/// Computes the gated `SwiGLU` activation using portable scalar arithmetic.
///
/// # Numerical Stability
///
/// For large negative gate values ($gate[i] < -20.0$), $\text{SiLU}(x) \to 0.0$, producing clean
/// zeroes without floating-point overflow or exponent underflow.
/// For large positive gate values ($gate[i] > 20.0$), $\text{SiLU}(x) \approx x$, evaluating
/// directly to $gate[i] \cdot up[i]$.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `out`, `gate`, and `up` lengths do not match or are empty.
pub fn swiglu_f32_scalar(out: &mut [f32], gate: &[f32], up: &[f32]) -> Result<(), K3Error> {
    let len = validate_gated_activation_args("swiglu", out, gate, up)?;

    for i in 0..len {
        let g = gate[i];
        let u = up[i];

        let silu = if g < -20.0 {
            0.0_f32
        } else if g > 20.0 {
            g
        } else {
            g / (1.0_f32 + (-g).exp())
        };

        out[i] = silu * u;
    }

    Ok(())
}

/// Computes the gated `SwiGLU` activation using VLA RVV 1.0 (`e32, m4, ta, ma`) inline assembly.
///
/// Evaluates $\text{SiLU}(gate) \odot up = \frac{gate}{1 + e^{-gate}} \odot up$ with range-reduced
/// vectorized $e^{-x}$ clamped to $[-88.0, 88.0]$ before Cody-Waite reduction and degree-6 Taylor
/// approximation so extreme logits never overflow intermediate exponentials to `NaN` or `Inf`.
///
/// # Safety
///
/// Caller must ensure `gate.len() == up.len()` and `gate.len() == out.len()`, and that the target
/// CPU supports the RISC-V Vector (`V`) 1.0 extension.
///
/// # Panics
///
/// Panics if `gate.len()`, `up.len()`, and `out.len()` do not all match.
#[cfg(all(target_arch = "riscv64", not(miri)))]
#[allow(unused_assignments)]
pub unsafe fn swiglu_f32_rvv(gate: &[f32], up: &[f32], out: &mut [f32]) {
    assert_eq!(gate.len(), up.len(), "swiglu_f32_rvv: gate/up len mismatch");
    assert_eq!(
        gate.len(),
        out.len(),
        "swiglu_f32_rvv: gate/out len mismatch"
    );

    let mut rem = gate.len();
    if rem == 0 {
        return;
    }

    let neg88: f32 = -88.0_f32;
    let pos88: f32 = 88.0_f32;
    let inv_ln2: f32 = std::f32::consts::LOG2_E;
    let ln2_hi: f32 = 0.693_145_75;
    let ln2_lo: f32 = 1.428_606_8e-6;
    let c6: f32 = 1.0_f32 / 720.0_f32;
    let c5: f32 = 1.0_f32 / 120.0_f32;
    let c4: f32 = 1.0_f32 / 24.0_f32;
    let c3: f32 = 1.0_f32 / 6.0_f32;
    let c2: f32 = 0.5_f32;
    let one: f32 = 1.0_f32;
    let bias127: i32 = 127;
    let kmin: i32 = -126;
    let kmax: i32 = 127;

    let mut pgate = gate.as_ptr();
    let mut pup = up.as_ptr();
    let mut pout = out.as_mut_ptr();

    // SAFETY: Caller guarantees equal slice lengths and valid RVV 1.0 support; pointers advance by `vl * 4` bytes within slice bounds.
    unsafe {
        core::arch::asm!(
            "2:",
            "vsetvli t0, {rem}, e32, m4, ta, ma",
            "vle32.v v0, ({pgate})",
            // u = clamp(-g, -88.0, 88.0)
            "vfsgnjn.vv v4, v0, v0",
            "vfmax.vf v4, v4, {neg88}",
            "vfmin.vf v4, v4, {pos88}",
            // k = round(u / ln2)
            "vfmul.vf v8, v4, {inv_ln2}",
            "vfcvt.x.f.v v16, v8",
            "vfcvt.f.x.v v8, v16",
            // r = (u - k * ln2_hi) - k * ln2_lo
            "vfmul.vf v20, v8, {ln2_hi}",
            "vfsub.vv v12, v4, v20",
            "vfmul.vf v20, v8, {ln2_lo}",
            "vfsub.vv v12, v12, v20",
            // Clamp k to [-126, 127] and build 2^k IEEE bit pattern in v16
            "vmax.vx v16, v16, {kmin}",
            "vmin.vx v16, v16, {kmax}",
            "vadd.vx v16, v16, {bias127}",
            "vsll.vi v16, v16, 23",
            // Degree-6 Taylor polynomial for exp(r) on [-ln2/2, ln2/2]
            "vfmv.v.f v20, {c6}",
            "vfmul.vv v20, v20, v12",
            "vfadd.vf v20, v20, {c5}",
            "vfmul.vv v20, v20, v12",
            "vfadd.vf v20, v20, {c4}",
            "vfmul.vv v20, v20, v12",
            "vfadd.vf v20, v20, {c3}",
            "vfmul.vv v20, v20, v12",
            "vfadd.vf v20, v20, {c2}",
            "vfmul.vv v20, v20, v12",
            "vfadd.vf v20, v20, {one}",
            "vfmul.vv v20, v20, v12",
            "vfadd.vf v20, v20, {one}",
            // exp(u) = poly(r) * 2^k; silu = g / (1.0 + exp(u))
            "vfmul.vv v20, v20, v16",
            "vfadd.vf v20, v20, {one}",
            "vfdiv.vv v24, v0, v20",
            "vle32.v v12, ({pup})",
            "vfmul.vv v24, v24, v12",
            "vse32.v v24, ({pout})",
            "slli t1, t0, 2",
            "add {pgate}, {pgate}, t1",
            "add {pup}, {pup}, t1",
            "add {pout}, {pout}, t1",
            "sub {rem}, {rem}, t0",
            "bnez {rem}, 2b",
            pgate = inout(reg) pgate,
            pup = inout(reg) pup,
            pout = inout(reg) pout,
            rem = inout(reg) rem,
            bias127 = in(reg) bias127,
            kmin = in(reg) kmin,
            kmax = in(reg) kmax,
            neg88 = in(freg) neg88,
            pos88 = in(freg) pos88,
            inv_ln2 = in(freg) inv_ln2,
            ln2_hi = in(freg) ln2_hi,
            ln2_lo = in(freg) ln2_lo,
            c6 = in(freg) c6,
            c5 = in(freg) c5,
            c4 = in(freg) c4,
            c3 = in(freg) c3,
            c2 = in(freg) c2,
            one = in(freg) one,
            out("t0") _, out("t1") _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _, out("v5") _,
            out("v6") _, out("v7") _, out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _, out("v16") _, out("v17") _,
            out("v18") _, out("v19") _, out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            out("v24") _, out("v25") _, out("v26") _, out("v27") _,
            options(nostack),
        );
    }
}

/// Computes Gemma `GeGLU` gated activation using the `tanh` GELU approximation:
/// $out[i] = \text{GELU}_{\text{tanh}}(gate[i]) \cdot up[i]$.
///
/// # Numerical Stability
///
/// Clamps extreme gate inputs beyond $[-15.0, 15.0]$ where $\tanh$ saturates to $\pm 1.0$
/// so $gate^3$ never overflows to `NaN` or infinity.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `out`, `gate`, and `up` lengths do not match or are empty.
pub fn geglu_f32(out: &mut [f32], gate: &[f32], up: &[f32]) -> Result<(), K3Error> {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const GELU_CUBIC_COEF: f32 = 0.044_715;

    let len = validate_gated_activation_args("geglu", out, gate, up)?;

    for i in 0..len {
        let g = gate[i];
        let gelu = if !g.is_finite() || g <= -15.0_f32 {
            0.0_f32
        } else if g >= 15.0_f32 {
            g
        } else {
            let inner = SQRT_2_OVER_PI * (g + GELU_CUBIC_COEF * g * g * g);
            0.5_f32 * g * (1.0_f32 + inner.tanh())
        };
        out[i] = gelu * up[i];
    }

    Ok(())
}

/// Applies Gemma 2 logit softcapping in-place: `x = cap * tanh(x / cap)`.
///
/// Values are skipped (no-op) when `cap <= 0.0` or `!cap.is_finite()`.
pub fn logit_softcap_tanh_inplace(logits: &mut [f32], cap: f32) {
    if !cap.is_finite() || cap <= 0.0_f32 {
        return;
    }
    let inv_cap = 1.0_f32 / cap;
    for logit in logits.iter_mut() {
        *logit = cap * (*logit * inv_cap).tanh();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bf16_to_f32_finite_normal_and_subnormal() {
        // 1.0 in BF16: sign=0, exp=127 (0x7F), mant=0 -> 0x3F80
        let one = bf16_to_f32_finite(0x3F80);
        assert!((one - 1.0).abs() < 1e-6);

        // -2.0 in BF16: sign=1, exp=128 (0x80), mant=0 -> 0xC000
        let neg_two = bf16_to_f32_finite(0xC000);
        assert!((neg_two - (-2.0)).abs() < 1e-6);

        // 0.0 in BF16: 0x0000
        let zero = bf16_to_f32_finite(0x0000);
        assert_eq!(zero, 0.0);
    }

    #[test]
    fn test_bf16_to_f32_finite_inf_nan_clamping() {
        // +Inf, -Inf, and all NaN bit patterns in BF16 collapse cleanly to 0.0.
        assert_eq!(bf16_to_f32_finite(0x7F80).to_bits(), 0.0_f32.to_bits());
        assert_eq!(bf16_to_f32_finite(0xFF80).to_bits(), 0.0_f32.to_bits());
        assert_eq!(bf16_to_f32_finite(0x7FC0).to_bits(), 0.0_f32.to_bits());
        assert_eq!(bf16_to_f32_finite(0x7FFF).to_bits(), 0.0_f32.to_bits());
        assert_eq!(bf16_to_f32_finite(0xFFC0).to_bits(), 0.0_f32.to_bits());
        assert_eq!(bf16_to_f32_finite(0xFFFF).to_bits(), 0.0_f32.to_bits());

        // Signed zeroes, subnormals, and normals remain bit-preserving.
        assert_eq!(bf16_to_f32_finite(0x8000).to_bits(), 0x8000_0000);
        assert_eq!(bf16_to_f32_finite(0x0001), f32::from_bits(0x0001_0000));
        assert_eq!(bf16_to_f32_finite(0x807F), f32::from_bits(0x807F_0000));
    }

    #[test]
    fn test_bf16_and_f16_ieee_preserve_non_finite() {
        assert!(bf16_to_f32_ieee(0x7F80).is_infinite());
        assert!(bf16_to_f32_ieee(0x7F80).is_sign_positive());
        assert!(bf16_to_f32_ieee(0xFF80).is_infinite());
        assert!(bf16_to_f32_ieee(0xFF80).is_sign_negative());
        assert!(bf16_to_f32_ieee(0x7FC0).is_nan());
        assert!(bf16_to_f32_ieee(0xFFFF).is_nan());

        assert!(crate::f16_to_f32_ieee(0x7C00).is_infinite());
        assert!(crate::f16_to_f32_ieee(0x7C00).is_sign_positive());
        assert!(crate::f16_to_f32_ieee(0xFC00).is_infinite());
        assert!(crate::f16_to_f32_ieee(0xFC00).is_sign_negative());
        assert!(crate::f16_to_f32_ieee(0x7E00).is_nan());
        assert!(crate::f16_to_f32_ieee(0xFFFF).is_nan());
        assert_eq!(
            crate::f16_to_f32_finite(0x7C00).to_bits(),
            0.0_f32.to_bits()
        );
        assert_eq!(
            crate::f16_to_f32_finite(0xFC00).to_bits(),
            0.0_f32.to_bits()
        );
        assert_eq!(
            crate::f16_to_f32_finite(0x7E00).to_bits(),
            0.0_f32.to_bits()
        );
        assert_eq!(crate::f16_to_f32_finite(0x8000).to_bits(), 0x8000_0000);
    }

    /// Reference strip-mined VLA simulator matching `swiglu_f32_rvv` range reduction and
    /// degree-6 Taylor polynomial semantics for arbitrary `VL` (`8` for `zvl256b`, `32` for `zvl1024b`).
    fn swiglu_f32_vla_sim(out: &mut [f32], gate: &[f32], up: &[f32], vl: usize) {
        const INV_LN2: f32 = std::f32::consts::LOG2_E;
        const LN2_HI: f32 = 0.693_145_75;
        const LN2_LO: f32 = 1.428_606_8e-6;
        const C6: f32 = 1.0_f32 / 720.0_f32;
        const C5: f32 = 1.0_f32 / 120.0_f32;
        const C4: f32 = 1.0_f32 / 24.0_f32;
        const C3: f32 = 1.0_f32 / 6.0_f32;
        const C2: f32 = 0.5_f32;

        let mut offset = 0usize;
        while offset < gate.len() {
            let chunk = (gate.len() - offset).min(vl.max(1));
            for i in 0..chunk {
                let g = gate[offset + i];
                let u = (-g).clamp(-88.0_f32, 88.0_f32);
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "vfcvt.x.f.v rounds finite clamped [-127, 127] values to i32"
                )]
                let k = (u * INV_LN2).round_ties_even() as i32;
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "vfcvt.f.x.v converts small integer k in [-127, 127] losslessly to f32"
                )]
                let kf = k as f32;
                let r = (u - kf * LN2_HI) - kf * LN2_LO;
                let k_clamped = k.clamp(-126, 127);
                #[expect(
                    clippy::cast_sign_loss,
                    reason = "k_clamped + 127 is strictly in 1..=254"
                )]
                let pow2k = f32::from_bits(((k_clamped + 127) as u32) << 23);
                let poly =
                    (((((C6 * r + C5) * r + C4) * r + C3) * r + C2) * r + 1.0_f32) * r + 1.0_f32;
                let exp_u = poly * pow2k;
                let silu = g / (1.0_f32 + exp_u);
                out[offset + i] = silu * up[offset + i];
            }
            offset += chunk;
        }
    }

    #[test]
    fn test_swiglu_f32_vla_parity_and_extreme_logits() {
        let test_lengths: &[usize] = if cfg!(miri) {
            &[1, 7, 31, 137]
        } else {
            &[1, 7, 31, 137, 4864, 14336]
        };

        let edge_patterns = [
            0.0_f32,
            -0.0_f32,
            1e-40_f32,
            -1e-40_f32,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            -100.0_f32,
            100.0_f32,
            -88.0_f32,
            88.0_f32,
            -20.0_f32,
            20.0_f32,
            -1.25_f32,
            1.25_f32,
        ];

        for &n in test_lengths {
            let mut gate = Vec::with_capacity(n);
            let mut up = Vec::with_capacity(n);
            for idx in 0..n {
                let g = if idx < edge_patterns.len() {
                    edge_patterns[idx]
                } else {
                    #[expect(
                        clippy::cast_precision_loss,
                        reason = "synthetic test index comfortably within exact f32 mantissa range"
                    )]
                    {
                        ((idx % 200) as f32 - 100.0_f32) * 0.25_f32
                    }
                };
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "synthetic test index comfortably within exact f32 mantissa range"
                )]
                let u = 0.5_f32 + ((idx % 19) as f32 - 9.0_f32) * 0.125_f32;
                gate.push(g);
                up.push(u);
            }

            let mut out_scalar = vec![0.0_f32; n];
            let mut out_vl8 = vec![0.0_f32; n];
            let mut out_vl32 = vec![0.0_f32; n];

            swiglu_f32_scalar(&mut out_scalar, &gate, &up).expect("scalar swiglu succeeds");
            swiglu_f32_vla_sim(&mut out_vl8, &gate, &up, 8);
            swiglu_f32_vla_sim(&mut out_vl32, &gate, &up, 32);

            for i in 0..n {
                assert!(
                    out_scalar[i].is_finite(),
                    "scalar non-finite at n={n} i={i}"
                );
                assert!(out_vl8[i].is_finite(), "vl8 non-finite at n={n} i={i}");
                assert!(out_vl32[i].is_finite(), "vl32 non-finite at n={n} i={i}");

                let tol = 1e-6_f32.max(1e-5_f32 * out_scalar[i].abs());
                assert!(
                    (out_vl8[i] - out_scalar[i]).abs() <= tol,
                    "vl8 parity failed at n={n} i={i}: g={}, scalar={}, vl8={}",
                    gate[i],
                    out_scalar[i],
                    out_vl8[i]
                );
                assert_eq!(out_vl8[i], out_vl32[i], "vl8/vl32 mismatch at n={n} i={i}");
            }

            let mut out_dispatch = vec![0.0_f32; n];
            swiglu_f32(&mut out_dispatch, &gate, &up).expect("swiglu_f32 dispatch succeeds");
            for i in 0..n {
                let tol = 1e-6_f32.max(1e-5_f32 * out_scalar[i].abs());
                assert!((out_dispatch[i] - out_scalar[i]).abs() <= tol);
            }
        }
    }

    #[test]
    fn test_swiglu_f32_computation() {
        let gate = [0.0_f32, 2.0, -25.0, 30.0];
        let up = [5.0_f32, 10.0, 2.0, 0.5];
        let mut out = [0.0_f32; 4];

        swiglu_f32(&mut out, &gate, &up).expect("swiglu succeeds");

        // SiLU(0.0) = 0.0 -> out[0] = 0.0 * 5.0 = 0.0
        assert_eq!(out[0], 0.0);

        // SiLU(2.0) = 2.0 / (1 + exp(-2)) ≈ 1.761594 -> out[1] ≈ 17.61594
        let expected_1 = (2.0 / (1.0 + (-2.0_f32).exp())) * 10.0;
        assert!((out[1] - expected_1).abs() < 1e-4);

        // Large negative (-25.0) -> SiLU ≈ 0.0 -> out[2] = 0.0
        assert_eq!(out[2], 0.0);

        // Large positive (30.0) -> SiLU ≈ 30.0 -> out[3] ≈ 30.0 * 0.5 = 15.0
        assert!((out[3] - 15.0).abs() < 1e-4);
    }

    #[test]
    fn test_swiglu_f32_dimension_mismatch() {
        let gate = [1.0_f32; 4];
        let up = [1.0_f32; 3];
        let mut out = [0.0_f32; 4];
        assert!(matches!(
            swiglu_f32(&mut out, &gate, &up),
            Err(K3Error::TensorBoundsError { .. })
        ));
    }

    #[test]
    fn test_geglu_and_logit_softcap() {
        let gate = [0.0_f32, 1.0, -100.0, 100.0];
        let up = [4.0_f32, 2.0, 3.0, 0.5];
        let mut out = [0.0_f32; 4];
        geglu_f32(&mut out, &gate, &up).expect("geglu_f32 succeeds");

        assert_eq!(out[0], 0.0);
        let expected_gelu_one = f32::midpoint(
            1.0_f32,
            (0.797_884_6_f32 * (1.0_f32 + 0.044_715_f32)).tanh(),
        );
        assert!((out[1] - expected_gelu_one * 2.0).abs() < 1e-4);
        assert_eq!(out[2], 0.0);
        assert!((out[3] - 50.0).abs() < 1e-4);
        assert!(out.iter().all(|v| v.is_finite()));

        let mut logits = [-200.0_f32, -30.0, 0.0, 30.0, 200.0];
        logit_softcap_tanh_inplace(&mut logits, 30.0);
        assert!((logits[2] - 0.0).abs() < 1e-6);
        assert!((logits[3] - 30.0 * 1.0_f32.tanh()).abs() < 1e-4);
        assert!((logits[1] - (-30.0 * 1.0_f32.tanh())).abs() < 1e-4);
        assert!(logits[0] > -30.0001 && logits[0] <= -29.99);
        assert!(logits[4] < 30.0001 && logits[4] >= 29.99);

        // Non-positive cap is a safe no-op.
        let before = logits;
        logit_softcap_tanh_inplace(&mut logits, 0.0);
        assert_eq!(logits, before);
    }
}
