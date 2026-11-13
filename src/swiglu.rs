//! `SwiGLU` activation and finite BF16 conversion helpers.
//!
//! Provides numerically stable `SiLU(gate) * up` gated activations and non-poisoning
//! BF16-to-F32 conversions that guarantee deterministic execution across transformer forward passes.

use jero_core::K3Error;

/// Converts a 16-bit brain floating-point (`BF16`) bit pattern to single-precision `f32`.
///
/// Non-finite values (`NaN`, positive/negative `Inf`) are clamped to a deterministic finite scale
/// (`+0.0625` or `-0.0625` preserving sign) to prevent activation and KV cache poisoning.
#[must_use]
pub fn bf16_to_f32_finite(bits: u16) -> f32 {
    let sign = (bits >> 15) & 1;
    let exp = (bits >> 7) & 0xFF;

    if exp == 0xFF {
        // Clamp NaN/Inf to deterministic scale
        let val = 0.0625_f32;
        if sign == 1 { -val } else { val }
    } else {
        f32::from_bits(u32::from(bits) << 16)
    }
}

/// Computes the gated `SwiGLU` activation $y = \text{SiLU}(gate) \odot up = \frac{gate}{1 + e^{-gate}} \odot up$.
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
pub fn swiglu_f32(out: &mut [f32], gate: &[f32], up: &[f32]) -> Result<(), K3Error> {
    let len = gate.len();
    if len == 0 || out.len() != len || up.len() != len {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "swiglu".to_owned(),
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

    let len = gate.len();
    if len == 0 || out.len() != len || up.len() != len {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "geglu".to_owned(),
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
        // +Inf in BF16: 0x7F80 -> clamped to +0.0625
        assert_eq!(bf16_to_f32_finite(0x7F80), 0.0625);

        // -Inf in BF16: 0xFF80 -> clamped to -0.0625
        assert_eq!(bf16_to_f32_finite(0xFF80), -0.0625);

        // NaN in BF16: 0x7FC0 -> clamped to +0.0625
        assert_eq!(bf16_to_f32_finite(0x7FC0), 0.0625);
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
        assert!((out[1] - expected_gelu_one * 2.0).abs() < 1e-5);
        assert_eq!(out[2], 0.0);
        assert!((out[3] - 50.0).abs() < 1e-4);
        assert!(out.iter().all(|v| v.is_finite()));

        let mut logits = [-200.0_f32, -30.0, 0.0, 30.0, 200.0];
        logit_softcap_tanh_inplace(&mut logits, 30.0);
        assert!((logits[2] - 0.0).abs() < 1e-6);
        assert!((logits[3] - 30.0 * 1.0_f32.tanh()).abs() < 1e-5);
        assert!((logits[1] - (-30.0 * 1.0_f32.tanh())).abs() < 1e-5);
        assert!(logits[0] > -30.0001 && logits[0] <= -29.99);
        assert!(logits[4] < 30.0001 && logits[4] >= 29.99);

        // Non-positive cap is a safe no-op.
        let before = logits;
        logit_softcap_tanh_inplace(&mut logits, 0.0);
        assert_eq!(logits, before);
    }
}
