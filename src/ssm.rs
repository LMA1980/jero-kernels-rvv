//! Linear-time State-Space (`Mamba2` SSD), `RWKV-7` WKV delta-rule, and `LFM2` short-conv kernels.
//!
//! All recurrence primitives operate directly on caller-provided mutable state slices in `O(1)`
//! working memory per sequence step, using `RISC-V` `RVV 1.0` (`VLA`) vector strip-mining on
//! `riscv64` targets and portable scalar fallbacks elsewhere and under Miri.

use jero_core::K3Error;

use crate::attention::dot_f32;

/// Computes numerically stable softplus: `ln(1 + exp(x))`.
#[must_use]
#[inline]
pub fn softplus_f32(x: f32) -> f32 {
    if x > 20.0_f32 {
        x
    } else if x < -20.0_f32 {
        x.exp()
    } else {
        x.exp().ln_1p()
    }
}

/// Computes `SiLU` activation: `x / (1 + exp(-x))`.
#[must_use]
#[inline]
pub fn silu_scalar_f32(x: f32) -> f32 {
    x / (1.0_f32 + (-x).exp())
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[inline]
#[allow(unused_assignments)]
fn scale_and_axpy_f32_rvv(dst: &mut [f32], scale: f32, rhs: &[f32], coeff: f32) {
    let len = dst.len().min(rhs.len());
    if len == 0 {
        return;
    }
    let mut pd = dst.as_mut_ptr();
    let mut pr = rhs.as_ptr();
    let mut rem = len;

    // SAFETY: `pd` and `pr` point to disjoint slices of `rem` elements (`dst` is `&mut`).
    unsafe {
        core::arch::asm!(
            "2:",
            "vsetvli t0, {rem}, e32, m4, ta, ma",
            "vle32.v v8, ({pd})",
            "vle32.v v16, ({pr})",
            "vfmul.vf v8, v8, {scale}",
            "vfmacc.vf v8, {coeff}, v16",
            "vse32.v v8, ({pd})",
            "slli t1, t0, 2",
            "add {pd}, {pd}, t1",
            "add {pr}, {pr}, t1",
            "sub {rem}, {rem}, t0",
            "bnez {rem}, 2b",
            pd = inout(reg) pd,
            pr = inout(reg) pr,
            rem = inout(reg) rem,
            scale = in(freg) scale,
            coeff = in(freg) coeff,
            out("t0") _,
            out("t1") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            options(nostack),
        );
    }
}

#[inline]
fn scale_and_axpy_f32(dst: &mut [f32], scale: f32, rhs: &[f32], coeff: f32) {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        scale_and_axpy_f32_rvv(dst, scale, rhs, coeff);
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        for (d, &r) in dst.iter_mut().zip(rhs) {
            *d = (*d * scale) + (coeff * r);
        }
    }
}

fn check_len(
    tensor_name: &str,
    actual: usize,
    expected: usize,
    context: &str,
) -> Result<(), K3Error> {
    if actual != expected {
        return Err(K3Error::TensorBoundsError {
            tensor_name: tensor_name.to_owned(),
            offset: 0,
            byte_len: actual.saturating_mul(size_of::<f32>()),
            buffer_len: expected.saturating_mul(size_of::<f32>()),
            reason: format!("{context}: '{tensor_name}' length {actual} != expected {expected}"),
        });
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "internal helper shared by Mamba2 conv1d and LFM2 shortconv steps"
)]
fn causal_depthwise_conv1d_step(
    architecture: &str,
    conv_state: &mut [f32],
    x_in: &[f32],
    conv_weight: &[f32],
    conv_bias: Option<&[f32]>,
    out: &mut [f32],
    channels: usize,
    kernel_size: usize,
) -> Result<(), K3Error> {
    if channels == 0 || kernel_size == 0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: architecture.to_owned(),
            reason: format!(
                "causal depthwise conv1d requires non-zero channels={channels} and kernel_size={kernel_size}"
            ),
        });
    }
    let hist_len = kernel_size - 1;
    let expected_state =
        channels
            .checked_mul(hist_len)
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: architecture.to_owned(),
                reason: "conv1d history state length overflowed usize".to_owned(),
            })?;
    let expected_weight =
        channels
            .checked_mul(kernel_size)
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: architecture.to_owned(),
                reason: "conv1d weight length overflowed usize".to_owned(),
            })?;

    check_len("conv_state", conv_state.len(), expected_state, architecture)?;
    check_len("x_in", x_in.len(), channels, architecture)?;
    check_len(
        "conv_weight",
        conv_weight.len(),
        expected_weight,
        architecture,
    )?;
    check_len("out", out.len(), channels, architecture)?;
    if let Some(bias) = conv_bias {
        check_len("conv_bias", bias.len(), channels, architecture)?;
    }

    for ch in 0..channels {
        let w_ch = &conv_weight[ch * kernel_size..(ch + 1) * kernel_size];
        let mut acc = conv_bias.map_or(0.0_f32, |b| b[ch]);
        if hist_len > 0 {
            let state_ch = &mut conv_state[ch * hist_len..(ch + 1) * hist_len];
            acc += dot_f32(state_ch, &w_ch[..hist_len]);
            acc += x_in[ch] * w_ch[hist_len];
            state_ch.copy_within(1.., 0);
            state_ch[hist_len - 1] = x_in[ch];
        } else {
            acc += x_in[ch] * w_ch[0];
        }
        out[ch] = acc;
    }

    Ok(())
}

/// Executes a single autoregressive causal depthwise `Conv1D + SiLU` step for `Mamba2`.
///
/// `conv_state` stores the trailing `kernel_size - 1` inputs per channel in `[channels, kernel_size - 1]`
/// row-major layout and is shifted in place to record `x_in`.
///
/// # Errors
///
/// - Returns [`K3Error::CapabilityMismatch`] if `channels == 0` or `kernel_size == 0` or lengths overflow.
/// - Returns [`K3Error::TensorBoundsError`] if slice lengths mismatch the configured dimensions.
pub fn mamba2_conv1d_silu_step(
    conv_state: &mut [f32],
    x_in: &[f32],
    conv_weight: &[f32],
    conv_bias: Option<&[f32]>,
    out: &mut [f32],
    channels: usize,
    kernel_size: usize,
) -> Result<(), K3Error> {
    causal_depthwise_conv1d_step(
        "mamba2",
        conv_state,
        x_in,
        conv_weight,
        conv_bias,
        out,
        channels,
        kernel_size,
    )?;
    for val in out.iter_mut() {
        *val = silu_scalar_f32(*val);
    }
    Ok(())
}

/// Executes one autoregressive step of the `Mamba2` Structured State-Space Duality (`SSD`) recurrence.
///
/// Updates per-head recurrent state `state: [n_heads, head_dim, state_size]` in place using
/// discrete zero-order-hold decay `exp(-|A_h| * softplus(dt_h))` and writes gated output to `out`:
/// - `S_{h,p} = decay_h * S_{h,p} + (dt_h * x_{h,p}) * B_{g}`
/// - `y_{h,p} = dot(S_{h,p}, C_{g}) + D_h * x_{h,p}`
/// - `out_{h,p} = y_{h,p} * SiLU(z_{h,p})` (when `z_gate` is provided)
///
/// # Errors
///
/// - Returns [`K3Error::CapabilityMismatch`] if any dimension is zero or `n_heads % n_groups != 0`.
/// - Returns [`K3Error::TensorBoundsError`] if any slice length mismatches its expected shape.
#[expect(
    clippy::too_many_arguments,
    reason = "Mamba2 SSD state update takes explicit state, projections, gates, and geometry"
)]
pub fn mamba2_ssd_state_step_f32(
    state: &mut [f32],
    out: &mut [f32],
    x: &[f32],
    dt_raw: &[f32],
    dt_bias: Option<&[f32]>,
    a_log_or_neg: &[f32],
    b: &[f32],
    c: &[f32],
    d_skip: Option<&[f32]>,
    z_gate: Option<&[f32]>,
    n_heads: usize,
    head_dim: usize,
    state_size: usize,
    n_groups: usize,
) -> Result<(), K3Error> {
    if n_heads == 0 || head_dim == 0 || state_size == 0 || n_groups == 0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: "mamba2".to_owned(),
            reason: format!(
                "mamba2_ssd_state_step_f32 requires non-zero geometry (heads={n_heads}, head_dim={head_dim}, state={state_size}, groups={n_groups})"
            ),
        });
    }
    if n_heads % n_groups != 0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: "mamba2".to_owned(),
            reason: format!(
                "mamba2_ssd_state_step_f32 requires n_heads ({n_heads}) divisible by n_groups ({n_groups})"
            ),
        });
    }

    let inner_dim = n_heads
        .checked_mul(head_dim)
        .ok_or_else(|| K3Error::CapabilityMismatch {
            architecture: "mamba2".to_owned(),
            reason: "mamba2 inner_dim overflowed usize".to_owned(),
        })?;
    let expected_state =
        inner_dim
            .checked_mul(state_size)
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: "mamba2".to_owned(),
                reason: "mamba2 state tensor size overflowed usize".to_owned(),
            })?;
    let expected_bc =
        n_groups
            .checked_mul(state_size)
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: "mamba2".to_owned(),
                reason: "mamba2 group B/C length overflowed usize".to_owned(),
            })?;

    check_len("state", state.len(), expected_state, "mamba2_ssd")?;
    check_len("out", out.len(), inner_dim, "mamba2_ssd")?;
    check_len("x", x.len(), inner_dim, "mamba2_ssd")?;
    check_len("dt_raw", dt_raw.len(), n_heads, "mamba2_ssd")?;
    check_len("a_log_or_neg", a_log_or_neg.len(), n_heads, "mamba2_ssd")?;
    check_len("b", b.len(), expected_bc, "mamba2_ssd")?;
    check_len("c", c.len(), expected_bc, "mamba2_ssd")?;
    if let Some(db) = dt_bias {
        check_len("dt_bias", db.len(), n_heads, "mamba2_ssd")?;
    }
    if let Some(d) = d_skip {
        check_len("d_skip", d.len(), n_heads, "mamba2_ssd")?;
    }
    if let Some(z) = z_gate {
        check_len("z_gate", z.len(), inner_dim, "mamba2_ssd")?;
    }

    let heads_per_group = n_heads / n_groups;
    let head_state_stride = head_dim * state_size;

    for head_idx in 0..n_heads {
        let group_idx = head_idx / heads_per_group;
        let b_g = &b[group_idx * state_size..(group_idx + 1) * state_size];
        let c_g = &c[group_idx * state_size..(group_idx + 1) * state_size];

        let dt_arg = dt_raw[head_idx] + dt_bias.map_or(0.0_f32, |db| db[head_idx]);
        let dt = softplus_f32(dt_arg);
        let a_neg = -a_log_or_neg[head_idx].abs();
        let decay = (dt * a_neg).exp();
        let d_h = d_skip.map_or(0.0_f32, |d| d[head_idx]);

        let s_head = &mut state[head_idx * head_state_stride..(head_idx + 1) * head_state_stride];
        let x_head = &x[head_idx * head_dim..(head_idx + 1) * head_dim];
        let out_head = &mut out[head_idx * head_dim..(head_idx + 1) * head_dim];

        for p in 0..head_dim {
            let x_val = x_head[p];
            let s_row = &mut s_head[p * state_size..(p + 1) * state_size];
            scale_and_axpy_f32(s_row, decay, b_g, dt * x_val);
            let mut y = dot_f32(s_row, c_g) + d_h * x_val;
            if let Some(z) = z_gate {
                y *= silu_scalar_f32(z[head_idx * head_dim + p]);
            }
            out_head[p] = y;
        }
    }

    Ok(())
}

/// Computes `RWKV` token-shift linear interpolation: `out[i] = cur_x[i] + mu[i] * (prev_x[i] - cur_x[i])`.
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if `prev_x` is empty or slice lengths mismatch.
pub fn rwkv_token_shift_lerp_f32(
    prev_x: &[f32],
    cur_x: &[f32],
    mu: &[f32],
    out: &mut [f32],
) -> Result<(), K3Error> {
    let len = prev_x.len();
    if len == 0 {
        return Err(K3Error::TensorBoundsError {
            tensor_name: "prev_x".to_owned(),
            offset: 0,
            byte_len: 0,
            buffer_len: 0,
            reason: "rwkv_token_shift_lerp_f32 requires non-empty input vectors".to_owned(),
        });
    }
    check_len("cur_x", cur_x.len(), len, "rwkv_lerp")?;
    check_len("mu", mu.len(), len, "rwkv_lerp")?;
    check_len("out", out.len(), len, "rwkv_lerp")?;

    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        rwkv_token_shift_lerp_f32_rvv(prev_x, cur_x, mu, out);
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        for (((&p, &c), &m), o) in prev_x.iter().zip(cur_x).zip(mu).zip(out.iter_mut()) {
            *o = c + m * (p - c);
        }
    }

    Ok(())
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[allow(unused_assignments)]
fn rwkv_token_shift_lerp_f32_rvv(prev_x: &[f32], cur_x: &[f32], mu: &[f32], out: &mut [f32]) {
    let mut pp = prev_x.as_ptr();
    let mut pc = cur_x.as_ptr();
    let mut pm = mu.as_ptr();
    let mut po = out.as_mut_ptr();
    let mut rem = prev_x.len();

    // SAFETY: All pointers reference valid contiguous slices of length `rem`.
    unsafe {
        core::arch::asm!(
            "2:",
            "vsetvli t0, {rem}, e32, m4, ta, ma",
            "vle32.v v8, ({pp})",
            "vle32.v v12, ({pc})",
            "vle32.v v16, ({pm})",
            "vfsub.vv v8, v8, v12",
            "vfmacc.vv v12, v16, v8",
            "vse32.v v12, ({po})",
            "slli t1, t0, 2",
            "add {pp}, {pp}, t1",
            "add {pc}, {pc}, t1",
            "add {pm}, {pm}, t1",
            "add {po}, {po}, t1",
            "sub {rem}, {rem}, t0",
            "bnez {rem}, 2b",
            pp = inout(reg) pp,
            pc = inout(reg) pc,
            pm = inout(reg) pm,
            po = inout(reg) po,
            rem = inout(reg) rem,
            out("t0") _,
            out("t1") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v12") _, out("v13") _, out("v14") _, out("v15") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            options(nostack),
        );
    }
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[inline]
#[allow(unused_assignments)]
fn rwkv7_state_row_update_rvv(
    s_row: &mut [f32],
    w_h: &[f32],
    b_h: &[f32],
    k_h: &[f32],
    sa_i: f32,
    v_i: f32,
) {
    let mut ps = s_row.as_mut_ptr();
    let mut pw = w_h.as_ptr();
    let mut pb = b_h.as_ptr();
    let mut pk = k_h.as_ptr();
    let mut rem = s_row.len();
    let zero = 0.0_f32;
    let one = 1.0_f32;

    // SAFETY: All pointers cover `rem` elements; `s_row` is exclusively borrowed.
    unsafe {
        core::arch::asm!(
            "2:",
            "vsetvli t0, {rem}, e32, m2, ta, ma",
            "vle32.v v4, ({ps})",
            "vle32.v v6, ({pw})",
            "vle32.v v8, ({pb})",
            "vle32.v v10, ({pk})",
            "vfmax.vf v6, v6, {zero}",
            "vfmin.vf v6, v6, {one}",
            "vfmul.vv v4, v4, v6",
            "vfmacc.vf v4, {sa}, v8",
            "vfmacc.vf v4, {vi}, v10",
            "vse32.v v4, ({ps})",
            "slli t1, t0, 2",
            "add {ps}, {ps}, t1",
            "add {pw}, {pw}, t1",
            "add {pb}, {pb}, t1",
            "add {pk}, {pk}, t1",
            "sub {rem}, {rem}, t0",
            "bnez {rem}, 2b",
            ps = inout(reg) ps,
            pw = inout(reg) pw,
            pb = inout(reg) pb,
            pk = inout(reg) pk,
            rem = inout(reg) rem,
            zero = in(freg) zero,
            one = in(freg) one,
            sa = in(freg) sa_i,
            vi = in(freg) v_i,
            out("t0") _,
            out("t1") _,
            out("v4") _, out("v5") _,
            out("v6") _, out("v7") _,
            out("v8") _, out("v9") _,
            out("v10") _, out("v11") _,
            options(nostack),
        );
    }
}

#[inline]
fn rwkv7_state_row_update(
    s_row: &mut [f32],
    w_h: &[f32],
    b_h: &[f32],
    k_h: &[f32],
    sa_i: f32,
    v_i: f32,
) {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        rwkv7_state_row_update_rvv(s_row, w_h, b_h, k_h, sa_i, v_i);
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        for (((s_el, &w_el), &b_el), &k_el) in s_row.iter_mut().zip(w_h).zip(b_h).zip(k_h) {
            let w_clamped = w_el.clamp(0.0_f32, 1.0_f32);
            *s_el = (*s_el * w_clamped) + (sa_i * b_el) + (v_i * k_el);
        }
    }
}

/// Executes one autoregressive `RWKV-7` (`Goose`) generalized matrix-valued WKV delta-rule step.
///
/// Updates each head state matrix `S_h: [head_size, head_size]` in place:
/// `S_h' = S_h * diag(clamp(w_h, 0, 1)) + (S_h * a_h) * b_h^T + v_h * k_h^T`,
/// then outputs `out_h = S_h' * r_h`.
///
/// # Errors
///
/// - Returns [`K3Error::CapabilityMismatch`] if `n_heads == 0` or `head_size == 0` or lengths overflow.
/// - Returns [`K3Error::TensorBoundsError`] if slice lengths mismatch `[n_heads, head_size]` / `[n_heads, head_size, head_size]`.
#[expect(
    clippy::too_many_arguments,
    reason = "RWKV7 WKV update takes explicit r, w_decay, k, v, a, b, and head geometry"
)]
pub fn rwkv7_wkv_delta_rule_step_f32(
    state: &mut [f32],
    out: &mut [f32],
    r_vec: &[f32],
    w_decay: &[f32],
    k_vec: &[f32],
    v_vec: &[f32],
    a_vec: &[f32],
    b_vec: &[f32],
    n_heads: usize,
    head_size: usize,
) -> Result<(), K3Error> {
    if n_heads == 0 || head_size == 0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: "rwkv7".to_owned(),
            reason: format!(
                "rwkv7_wkv_delta_rule_step_f32 requires non-zero n_heads={n_heads} and head_size={head_size}"
            ),
        });
    }

    let expected_vec =
        n_heads
            .checked_mul(head_size)
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: "rwkv7".to_owned(),
                reason: "rwkv7 vector size overflowed usize".to_owned(),
            })?;
    let expected_state =
        expected_vec
            .checked_mul(head_size)
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: "rwkv7".to_owned(),
                reason: "rwkv7 state matrix size overflowed usize".to_owned(),
            })?;

    for (name, actual, expected) in [
        ("state", state.len(), expected_state),
        ("out", out.len(), expected_vec),
        ("r", r_vec.len(), expected_vec),
        ("w_decay", w_decay.len(), expected_vec),
        ("k", k_vec.len(), expected_vec),
        ("v", v_vec.len(), expected_vec),
        ("a", a_vec.len(), expected_vec),
        ("b", b_vec.len(), expected_vec),
    ] {
        check_len(name, actual, expected, "rwkv7_wkv")?;
    }

    let head_state_len = head_size * head_size;
    for head_idx in 0..n_heads {
        let base = head_idx * head_size;
        let r_h = &r_vec[base..base + head_size];
        let w_h = &w_decay[base..base + head_size];
        let k_h = &k_vec[base..base + head_size];
        let v_h = &v_vec[base..base + head_size];
        let a_h = &a_vec[base..base + head_size];
        let b_h = &b_vec[base..base + head_size];
        let out_h = &mut out[base..base + head_size];
        let s_h = &mut state[head_idx * head_state_len..(head_idx + 1) * head_state_len];

        for row_idx in 0..head_size {
            let s_row = &mut s_h[row_idx * head_size..(row_idx + 1) * head_size];
            let sa_i = dot_f32(s_row, a_h);
            rwkv7_state_row_update(s_row, w_h, b_h, k_h, sa_i, v_h[row_idx]);
            out_h[row_idx] = dot_f32(s_row, r_h);
        }
    }

    Ok(())
}

/// Executes one autoregressive `LFM2` short-convolution step (`dim` depthwise causal conv over `l_cache` taps).
///
/// Computes `u = b_gate * x_proj`, updates causal history buffer `conv_state: [dim, l_cache - 1]`,
/// applies causal depthwise 1D convolution with `conv_weight: [dim, l_cache]` + optional `conv_bias`,
/// and gates output as `out = c_gate * conv_out`.
///
/// # Errors
///
/// - Returns [`K3Error::CapabilityMismatch`] if `dim == 0` or `l_cache == 0` or sizes overflow.
/// - Returns [`K3Error::TensorBoundsError`] if slice lengths mismatch the configured dimensions.
#[expect(
    clippy::too_many_arguments,
    reason = "LFM2 shortconv step takes explicit gates, projection, conv state/weights, and geometry"
)]
pub fn lfm2_shortconv_step_f32(
    conv_state: &mut [f32],
    b_gate: &[f32],
    c_gate: &[f32],
    x_proj: &[f32],
    conv_weight: &[f32],
    conv_bias: Option<&[f32]>,
    out: &mut [f32],
    dim: usize,
    l_cache: usize,
) -> Result<(), K3Error> {
    if dim == 0 || l_cache == 0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: "lfm2".to_owned(),
            reason: format!(
                "lfm2_shortconv_step_f32 requires non-zero dim={dim} and l_cache={l_cache}"
            ),
        });
    }
    check_len("b_gate", b_gate.len(), dim, "lfm2_shortconv")?;
    check_len("c_gate", c_gate.len(), dim, "lfm2_shortconv")?;
    check_len("x_proj", x_proj.len(), dim, "lfm2_shortconv")?;
    check_len("out", out.len(), dim, "lfm2_shortconv")?;

    let pre_gated: Vec<f32> = b_gate.iter().zip(x_proj).map(|(&b, &x)| b * x).collect();
    causal_depthwise_conv1d_step(
        "lfm2",
        conv_state,
        &pre_gated,
        conv_weight,
        conv_bias,
        out,
        dim,
        l_cache,
    )?;
    for (o, &c) in out.iter_mut().zip(c_gate) {
        *o *= c;
    }
    Ok(())
}

#[inline]
fn checked_ssm_dim(op: &str, dim: usize, name: &str) -> Result<usize, K3Error> {
    if dim == 0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: "mamba2/rwkv7".to_owned(),
            reason: format!("{op} requires non-zero {name}"),
        });
    }
    Ok(dim)
}

#[inline]
fn checked_ssm_mul(op: &str, a: usize, b: usize) -> Result<usize, K3Error> {
    a.checked_mul(b).ok_or_else(|| K3Error::CapabilityMismatch {
        architecture: "mamba2/rwkv7".to_owned(),
        reason: format!("{op}: {a} * {b} overflowed usize"),
    })
}

#[inline]
fn check_ssm_slice_len(op: &str, got: usize, expected: usize) -> Result<(), K3Error> {
    if got != expected {
        return Err(K3Error::TensorBoundsError {
            tensor_name: op.to_owned(),
            offset: 0,
            byte_len: got,
            buffer_len: expected,
            reason: format!("{op} expected {expected} elements, got {got}"),
        });
    }
    Ok(())
}

#[inline]
fn row_max_abs_f32(slice: &[f32]) -> f32 {
    let mut max_abs = 0.0_f32;
    for &v in slice {
        let a = v.abs();
        if a > max_abs {
            max_abs = a;
        }
    }
    max_abs
}

#[inline]
#[allow(clippy::cast_possible_truncation)]
fn quantize_f32_to_i8_tile(src: &[f32], inv_scale: f32, dst: &mut [i8]) {
    for (d, &v) in dst.iter_mut().zip(src.iter()) {
        let q = (v * inv_scale).round().clamp(-127.0_f32, 127.0_f32);
        *d = q as i8;
    }
}

/// Executes a prefill chunk of `Mamba2` causal depthwise `Conv1D + SiLU` over `[seq_len, channels]`.
///
/// Updates `conv_state` (`[channels, kernel_size - 1]`) in place across all `seq_len` steps with
/// zero heap allocations.
///
/// # Errors
///
/// - Returns [`K3Error::CapabilityMismatch`] if `seq_len`, `channels`, or `kernel_size` is zero or overflows.
/// - Returns [`K3Error::TensorBoundsError`] if slice lengths mismatch the configured dimensions.
#[allow(clippy::too_many_arguments)]
pub fn mamba2_conv1d_silu_chunk_f32(
    conv_state: &mut [f32],
    x_seq: &[f32],
    conv_weight: &[f32],
    conv_bias: Option<&[f32]>,
    out_seq: &mut [f32],
    seq_len: usize,
    channels: usize,
    kernel_size: usize,
) -> Result<(), K3Error> {
    const OP: &str = "mamba2_conv1d_silu_chunk_f32";
    let seq_len = checked_ssm_dim(OP, seq_len, "seq_len")?;
    let channels = checked_ssm_dim(OP, channels, "channels")?;
    let kernel_size = checked_ssm_dim(OP, kernel_size, "kernel_size")?;
    let state_len = checked_ssm_mul(OP, channels, kernel_size - 1)?;
    let weight_len = checked_ssm_mul(OP, channels, kernel_size)?;
    let seq_elems = checked_ssm_mul(OP, seq_len, channels)?;

    check_ssm_slice_len(
        "mamba2_conv1d_silu_chunk_f32:conv_state",
        conv_state.len(),
        state_len,
    )?;
    check_ssm_slice_len("mamba2_conv1d_silu_chunk_f32:x_seq", x_seq.len(), seq_elems)?;
    check_ssm_slice_len(
        "mamba2_conv1d_silu_chunk_f32:conv_weight",
        conv_weight.len(),
        weight_len,
    )?;
    check_ssm_slice_len(
        "mamba2_conv1d_silu_chunk_f32:out_seq",
        out_seq.len(),
        seq_elems,
    )?;
    if let Some(bias) = conv_bias {
        check_ssm_slice_len(
            "mamba2_conv1d_silu_chunk_f32:conv_bias",
            bias.len(),
            channels,
        )?;
    }

    for t in 0..seq_len {
        let off = t * channels;
        mamba2_conv1d_silu_step(
            conv_state,
            &x_seq[off..off + channels],
            conv_weight,
            conv_bias,
            &mut out_seq[off..off + channels],
            channels,
            kernel_size,
        )?;
    }
    Ok(())
}

/// Computes causal lower-triangular `Mamba2` intra-chunk `C · B^T` projection scores using `i8` dot products.
///
/// For each group `g` in `0..n_groups` and positions `0 <= i, j < chunk_len`:
/// - `out_scores[g, i, j] = dot(C[i, g, :], B[j, g, :])` when `j <= i` (quantized per row to `i8` and
///   evaluated via [`crate::ime::dot_i8_vector`] in stack-allocated `[i8; 256]` tiles).
/// - `out_scores[g, i, j] = 0.0` when `j > i` (strictly causal).
///
/// Operates with zero heap allocations.
///
/// # Errors
///
/// - Returns [`K3Error::CapabilityMismatch`] if `chunk_len`, `n_groups`, or `state_size` is zero or overflows.
/// - Returns [`K3Error::TensorBoundsError`] if `c_chunk`, `b_chunk`, or `out_scores` lengths mismatch.
pub fn mamba2_ssd_intra_chunk_ime_i8(
    c_chunk: &[f32],
    b_chunk: &[f32],
    out_scores: &mut [f32],
    chunk_len: usize,
    n_groups: usize,
    state_size: usize,
) -> Result<(), K3Error> {
    const OP: &str = "mamba2_ssd_intra_chunk_ime_i8";
    const K_TILE: usize = 256;

    let chunk_len = checked_ssm_dim(OP, chunk_len, "chunk_len")?;
    let n_groups = checked_ssm_dim(OP, n_groups, "n_groups")?;
    let state_size = checked_ssm_dim(OP, state_size, "state_size")?;

    let group_step_elems = checked_ssm_mul(OP, n_groups, state_size)?;
    let cb_elems = checked_ssm_mul(OP, chunk_len, group_step_elems)?;
    let sq_elems = checked_ssm_mul(OP, chunk_len, chunk_len)?;
    let out_elems = checked_ssm_mul(OP, n_groups, sq_elems)?;

    check_ssm_slice_len(
        "mamba2_ssd_intra_chunk_ime_i8:c_chunk",
        c_chunk.len(),
        cb_elems,
    )?;
    check_ssm_slice_len(
        "mamba2_ssd_intra_chunk_ime_i8:b_chunk",
        b_chunk.len(),
        cb_elems,
    )?;
    check_ssm_slice_len(
        "mamba2_ssd_intra_chunk_ime_i8:out_scores",
        out_scores.len(),
        out_elems,
    )?;

    out_scores.fill(0.0_f32);
    let mut c_tile = [0_i8; K_TILE];
    let mut b_tile = [0_i8; K_TILE];

    for g in 0..n_groups {
        let out_g = &mut out_scores[g * sq_elems..(g + 1) * sq_elems];
        for i in 0..chunk_len {
            let c_off = i * group_step_elems + g * state_size;
            let c_row = &c_chunk[c_off..c_off + state_size];
            let c_max = row_max_abs_f32(c_row);
            if c_max == 0.0_f32 {
                continue;
            }
            let c_scale = c_max / 127.0_f32;
            let c_inv = 127.0_f32 / c_max;

            for j in 0..=i {
                let b_off = j * group_step_elems + g * state_size;
                let b_row = &b_chunk[b_off..b_off + state_size];
                let b_max = row_max_abs_f32(b_row);
                if b_max == 0.0_f32 {
                    continue;
                }
                let b_scale = b_max / 127.0_f32;
                let b_inv = 127.0_f32 / b_max;

                let mut acc_i64 = 0_i64;
                let mut k = 0_usize;
                while k < state_size {
                    let len = (state_size - k).min(K_TILE);
                    quantize_f32_to_i8_tile(&c_row[k..k + len], c_inv, &mut c_tile[..len]);
                    quantize_f32_to_i8_tile(&b_row[k..k + len], b_inv, &mut b_tile[..len]);
                    acc_i64 += i64::from(crate::ime::dot_i8_vector(&c_tile[..len], &b_tile[..len]));
                    k += len;
                }
                #[allow(clippy::cast_precision_loss)]
                {
                    out_g[i * chunk_len + j] = (acc_i64 as f32) * (c_scale * b_scale);
                }
            }
        }
    }

    Ok(())
}

/// Executes a prefill chunk scan of `Mamba2` Structured State-Space Duality (`SSD`) over `seq_len` steps.
///
/// Updates `state` (`[n_heads, head_dim, state_size]`) in place using `F32` exponential decay and
/// RVV/scalar vector accumulation via [`mamba2_ssd_state_step_f32`] with zero heap allocations.
///
/// # Errors
///
/// - Returns [`K3Error::CapabilityMismatch`] if any dimension is zero, `n_heads` is not divisible by
///   `n_groups`, or lengths overflow.
/// - Returns [`K3Error::TensorBoundsError`] if any slice length mismatches the configured dimensions.
#[allow(clippy::too_many_arguments)]
pub fn mamba2_ssd_chunk_scan_f32(
    state: &mut [f32],
    out_seq: &mut [f32],
    x_seq: &[f32],
    dt_raw_seq: &[f32],
    dt_bias: Option<&[f32]>,
    a_log_or_neg: &[f32],
    b_seq: &[f32],
    c_seq: &[f32],
    d_skip: Option<&[f32]>,
    z_gate_seq: Option<&[f32]>,
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
    state_size: usize,
    n_groups: usize,
) -> Result<(), K3Error> {
    const OP: &str = "mamba2_ssd_chunk_scan_f32";
    let seq_len = checked_ssm_dim(OP, seq_len, "seq_len")?;
    let n_heads = checked_ssm_dim(OP, n_heads, "n_heads")?;
    let head_dim = checked_ssm_dim(OP, head_dim, "head_dim")?;
    let state_size = checked_ssm_dim(OP, state_size, "state_size")?;
    let n_groups = checked_ssm_dim(OP, n_groups, "n_groups")?;
    if n_heads % n_groups != 0 {
        return Err(K3Error::CapabilityMismatch {
            architecture: "mamba2".to_owned(),
            reason: format!("{OP}: n_heads ({n_heads}) must be divisible by n_groups ({n_groups})"),
        });
    }

    let d_inner = checked_ssm_mul(OP, n_heads, head_dim)?;
    let state_len = checked_ssm_mul(OP, d_inner, state_size)?;
    let group_state = checked_ssm_mul(OP, n_groups, state_size)?;
    let seq_d_inner = checked_ssm_mul(OP, seq_len, d_inner)?;
    let seq_heads = checked_ssm_mul(OP, seq_len, n_heads)?;
    let seq_group_state = checked_ssm_mul(OP, seq_len, group_state)?;

    check_ssm_slice_len("mamba2_ssd_chunk_scan_f32:state", state.len(), state_len)?;
    check_ssm_slice_len(
        "mamba2_ssd_chunk_scan_f32:out_seq",
        out_seq.len(),
        seq_d_inner,
    )?;
    check_ssm_slice_len("mamba2_ssd_chunk_scan_f32:x_seq", x_seq.len(), seq_d_inner)?;
    check_ssm_slice_len(
        "mamba2_ssd_chunk_scan_f32:dt_raw_seq",
        dt_raw_seq.len(),
        seq_heads,
    )?;
    check_ssm_slice_len(
        "mamba2_ssd_chunk_scan_f32:a_log_or_neg",
        a_log_or_neg.len(),
        n_heads,
    )?;
    check_ssm_slice_len(
        "mamba2_ssd_chunk_scan_f32:b_seq",
        b_seq.len(),
        seq_group_state,
    )?;
    check_ssm_slice_len(
        "mamba2_ssd_chunk_scan_f32:c_seq",
        c_seq.len(),
        seq_group_state,
    )?;
    if let Some(bias) = dt_bias {
        check_ssm_slice_len("mamba2_ssd_chunk_scan_f32:dt_bias", bias.len(), n_heads)?;
    }
    if let Some(skip) = d_skip {
        check_ssm_slice_len("mamba2_ssd_chunk_scan_f32:d_skip", skip.len(), n_heads)?;
    }
    if let Some(z_seq) = z_gate_seq {
        check_ssm_slice_len(
            "mamba2_ssd_chunk_scan_f32:z_gate_seq",
            z_seq.len(),
            seq_d_inner,
        )?;
    }

    for t in 0..seq_len {
        let d_off = t * d_inner;
        let h_off = t * n_heads;
        let g_off = t * group_state;
        let z_step = z_gate_seq.map(|z| &z[d_off..d_off + d_inner]);
        mamba2_ssd_state_step_f32(
            state,
            &mut out_seq[d_off..d_off + d_inner],
            &x_seq[d_off..d_off + d_inner],
            &dt_raw_seq[h_off..h_off + n_heads],
            dt_bias,
            a_log_or_neg,
            &b_seq[g_off..g_off + group_state],
            &c_seq[g_off..g_off + group_state],
            d_skip,
            z_step,
            n_heads,
            head_dim,
            state_size,
            n_groups,
        )?;
    }
    Ok(())
}

/// Computes `RWKV-7` token-shift linear interpolation (`x_shifted = cur_x + mu * (prev_x - cur_x)`)
/// across a sequence `[seq_len, dim]` and updates `prev_x` to the final sequence token.
///
/// Operates with zero heap allocations.
///
/// # Errors
///
/// - Returns [`K3Error::CapabilityMismatch`] if `seq_len` or `dim` is zero or overflows.
/// - Returns [`K3Error::TensorBoundsError`] if slice lengths mismatch the configured dimensions.
pub fn rwkv_token_shift_lerp_sequence_f32(
    prev_x: &mut [f32],
    x_seq: &[f32],
    mu: &[f32],
    out_seq: &mut [f32],
    seq_len: usize,
    dim: usize,
) -> Result<(), K3Error> {
    const OP: &str = "rwkv_token_shift_lerp_sequence_f32";
    let seq_len = checked_ssm_dim(OP, seq_len, "seq_len")?;
    let dim = checked_ssm_dim(OP, dim, "dim")?;
    let total = checked_ssm_mul(OP, seq_len, dim)?;

    check_ssm_slice_len(
        "rwkv_token_shift_lerp_sequence_f32:prev_x",
        prev_x.len(),
        dim,
    )?;
    check_ssm_slice_len(
        "rwkv_token_shift_lerp_sequence_f32:x_seq",
        x_seq.len(),
        total,
    )?;
    check_ssm_slice_len("rwkv_token_shift_lerp_sequence_f32:mu", mu.len(), dim)?;
    check_ssm_slice_len(
        "rwkv_token_shift_lerp_sequence_f32:out_seq",
        out_seq.len(),
        total,
    )?;

    rwkv_token_shift_lerp_f32(prev_x, &x_seq[..dim], mu, &mut out_seq[..dim])?;
    for t in 1..seq_len {
        let prev_off = (t - 1) * dim;
        let cur_off = t * dim;
        rwkv_token_shift_lerp_f32(
            &x_seq[prev_off..prev_off + dim],
            &x_seq[cur_off..cur_off + dim],
            mu,
            &mut out_seq[cur_off..cur_off + dim],
        )?;
    }
    let last_off = (seq_len - 1) * dim;
    prev_x.copy_from_slice(&x_seq[last_off..last_off + dim]);
    Ok(())
}

/// Executes a sequence scan of the `RWKV-7` (`Goose`) generalized delta-rule `WKV` update.
///
/// Updates `state` (`[n_heads, head_size, head_size]`) in place across `seq_len` steps via
/// [`rwkv7_wkv_delta_rule_step_f32`] with zero heap allocations.
///
/// # Errors
///
/// - Returns [`K3Error::CapabilityMismatch`] if `seq_len`, `n_heads`, or `head_size` is zero or overflows.
/// - Returns [`K3Error::TensorBoundsError`] if slice lengths mismatch the configured dimensions.
#[allow(clippy::too_many_arguments)]
pub fn rwkv7_wkv_delta_rule_sequence_f32(
    state: &mut [f32],
    out_seq: &mut [f32],
    r_seq: &[f32],
    w_decay_seq: &[f32],
    k_seq: &[f32],
    v_seq: &[f32],
    a_seq: &[f32],
    b_seq: &[f32],
    seq_len: usize,
    n_heads: usize,
    head_size: usize,
) -> Result<(), K3Error> {
    const OP: &str = "rwkv7_wkv_delta_rule_sequence_f32";
    let seq_len = checked_ssm_dim(OP, seq_len, "seq_len")?;
    let n_heads = checked_ssm_dim(OP, n_heads, "n_heads")?;
    let head_size = checked_ssm_dim(OP, head_size, "head_size")?;
    let step_dim = checked_ssm_mul(OP, n_heads, head_size)?;
    let head_sq = checked_ssm_mul(OP, head_size, head_size)?;
    let state_len = checked_ssm_mul(OP, n_heads, head_sq)?;
    let seq_elems = checked_ssm_mul(OP, seq_len, step_dim)?;

    check_ssm_slice_len(
        "rwkv7_wkv_delta_rule_sequence_f32:state",
        state.len(),
        state_len,
    )?;
    check_ssm_slice_len(
        "rwkv7_wkv_delta_rule_sequence_f32:out_seq",
        out_seq.len(),
        seq_elems,
    )?;
    check_ssm_slice_len(
        "rwkv7_wkv_delta_rule_sequence_f32:r_seq",
        r_seq.len(),
        seq_elems,
    )?;
    check_ssm_slice_len(
        "rwkv7_wkv_delta_rule_sequence_f32:w_decay_seq",
        w_decay_seq.len(),
        seq_elems,
    )?;
    check_ssm_slice_len(
        "rwkv7_wkv_delta_rule_sequence_f32:k_seq",
        k_seq.len(),
        seq_elems,
    )?;
    check_ssm_slice_len(
        "rwkv7_wkv_delta_rule_sequence_f32:v_seq",
        v_seq.len(),
        seq_elems,
    )?;
    check_ssm_slice_len(
        "rwkv7_wkv_delta_rule_sequence_f32:a_seq",
        a_seq.len(),
        seq_elems,
    )?;
    check_ssm_slice_len(
        "rwkv7_wkv_delta_rule_sequence_f32:b_seq",
        b_seq.len(),
        seq_elems,
    )?;

    for t in 0..seq_len {
        let off = t * step_dim;
        rwkv7_wkv_delta_rule_step_f32(
            state,
            &mut out_seq[off..off + step_dim],
            &r_seq[off..off + step_dim],
            &w_decay_seq[off..off + step_dim],
            &k_seq[off..off + step_dim],
            &v_seq[off..off + step_dim],
            &a_seq[off..off + step_dim],
            &b_seq[off..off + step_dim],
            n_heads,
            head_size,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mamba2_conv1d_and_ssd_step() {
        // 2 channels, causal kernel_size = 3 => conv_state len = 2 * 2 = 4
        let mut conv_state = vec![1.0_f32, 2.0, -1.0, 0.5];
        let x_in = vec![0.5_f32, -0.5];
        // weights [ch0: (0.25, 0.5, 1.0), ch1: (1.0, -0.5, 0.5)]
        let conv_weight = vec![0.25_f32, 0.5, 1.0, 1.0, -0.5, 0.5];
        let conv_bias = vec![0.1_f32, -0.2];
        let mut out = vec![0.0_f32; 2];

        mamba2_conv1d_silu_step(
            &mut conv_state,
            &x_in,
            &conv_weight,
            Some(&conv_bias),
            &mut out,
            2,
            3,
        )
        .expect("conv1d step");

        // ch0 raw = 1*0.25 + 2*0.5 + 0.5*1.0 + 0.1 = 1.85 -> silu(1.85)
        let exp0 = silu_scalar_f32(1.85);
        // ch1 raw = -1*1.0 + 0.5*-0.5 + -0.5*0.5 - 0.2 = -1.7 -> silu(-1.7)
        let exp1 = silu_scalar_f32(-1.7);
        assert!((out[0] - exp0).abs() < 1e-5);
        assert!((out[1] - exp1).abs() < 1e-5);
        // conv_state shifted window: [2.0, 0.5, 0.5, -0.5]
        assert_eq!(conv_state, vec![2.0, 0.5, 0.5, -0.5]);

        // Now test SSD recurrence step
        let mut ssm_state = vec![0.0_f32; 4]; // 1 head, head_dim=2, state_size=2
        let mut ssd_out = vec![0.0_f32; 2];
        let dt_raw = vec![0.0_f32];
        let a_neg = vec![1.0_f32];
        let b = vec![1.0_f32, 0.5];
        let c = vec![1.0_f32, 1.0];
        mamba2_ssd_state_step_f32(
            &mut ssm_state,
            &mut ssd_out,
            &out,
            &dt_raw,
            None,
            &a_neg,
            &b,
            &c,
            None,
            None,
            1,
            2,
            2,
            1,
        )
        .expect("ssd step");
        assert!(ssd_out.iter().all(|v| v.is_finite()));
        assert!(ssd_out[0] != 0.0);
    }

    #[test]
    fn test_rwkv7_wkv_delta_rule_stability() {
        let n_heads = 2;
        let head_size = 4;
        let mut state = vec![0.0_f32; n_heads * head_size * head_size];
        let mut out = vec![0.0_f32; n_heads * head_size];

        let r_vec = vec![0.5_f32; n_heads * head_size];
        let w_decay = vec![0.9_f32; n_heads * head_size];
        let k_vec = vec![0.25_f32, -0.15, 0.3, -0.1, 0.2, 0.1, -0.25, 0.3];
        let v_vec = vec![0.4_f32, -0.2, 0.15, 0.3, -0.35, 0.2, 0.1, -0.2];
        let a_vec = vec![0.1_f32; n_heads * head_size];
        let b_vec = vec![-0.1_f32; n_heads * head_size];

        let mut first_norm = 0.0_f32;
        for step in 0..300 {
            rwkv7_wkv_delta_rule_step_f32(
                &mut state, &mut out, &r_vec, &w_decay, &k_vec, &v_vec, &a_vec, &b_vec, n_heads,
                head_size,
            )
            .expect("rwkv7 step");
            assert!(out.iter().all(|x| x.is_finite()));
            let max_abs = out.iter().fold(0.0_f32, |m, x| m.max(x.abs()));
            assert!(max_abs < 100.0, "step {step} diverged: {max_abs}");
            if step == 0 {
                first_norm = out.iter().map(|x| x * x).sum::<f32>().sqrt();
            }
        }
        let final_norm = out.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(first_norm > 0.0 && final_norm > 0.0);
        assert!((final_norm - first_norm).abs() > 1e-4);
    }

    #[test]
    fn test_mamba2_ssd_and_rwkv7_wkv_kernels() {
        // Test multi-group head sharing (4 heads, 2 groups, head_dim=2, state_size=3)
        let mut state = vec![0.1_f32; 4 * 2 * 3];
        let mut out = vec![0.0_f32; 4 * 2];
        let x = vec![1.0_f32; 4 * 2];
        let dt_raw = vec![0.2_f32; 4];
        let dt_bias = vec![0.1_f32; 4];
        let a_neg = vec![0.5_f32; 4];
        let b = vec![1.0_f32, 0.5, 0.25, -0.5, 0.5, 1.0]; // 2 groups * 3
        let c = vec![0.5_f32, 1.0, 0.5, 1.0, -0.5, 0.5];
        let d_skip = vec![0.25_f32; 4];
        let z_gate = vec![0.8_f32; 4 * 2];

        mamba2_ssd_state_step_f32(
            &mut state,
            &mut out,
            &x,
            &dt_raw,
            Some(&dt_bias),
            &a_neg,
            &b,
            &c,
            Some(&d_skip),
            Some(&z_gate),
            4,
            2,
            3,
            2,
        )
        .expect("grouped mamba2 ssd");
        assert!(out.iter().all(|v| v.is_finite()));
        // Heads 0..2 use group 0 while heads 2..4 use group 1, so outputs differ cleanly
        assert!((out[0] - out[4]).abs() > 1e-3);

        // Verify divisibility rejection
        let bad_group = mamba2_ssd_state_step_f32(
            &mut state, &mut out, &x, &dt_raw, None, &a_neg, &b, &c, None, None, 3, 2, 3, 2,
        );
        assert!(bad_group.is_err());

        // Single-step analytic check for 1 head, 2x2 RWKV7 state
        let mut st2 = vec![0.0_f32; 4];
        let mut o2 = vec![0.0_f32; 2];
        rwkv7_wkv_delta_rule_step_f32(
            &mut st2,
            &mut o2,
            &[1.0, 2.0],
            &[0.5, 0.5],
            &[1.0, -1.0],
            &[2.0, 3.0],
            &[0.0, 0.0],
            &[0.0, 0.0],
            1,
            2,
        )
        .expect("analytic rwkv7");
        // S_1 = v * k^T = [[2, -2], [3, -3]]; o = S_1 * [1, 2]^T = [-2, -3]
        assert!((o2[0] - (-2.0)).abs() < 1e-6);
        assert!((o2[1] - (-3.0)).abs() < 1e-6);
    }

    #[test]
    fn test_lfm2_shortconv_and_rwkv_lerp_step() {
        let prev_x = vec![1.0_f32, 3.0, -2.0, 4.0];
        let cur_x = vec![2.0_f32, 1.0, 0.0, 2.0];
        let mu = vec![0.25_f32, 0.5, 0.75, 1.0];
        let mut lerp_out = vec![0.0_f32; 4];
        rwkv_token_shift_lerp_f32(&prev_x, &cur_x, &mu, &mut lerp_out).expect("lerp step");
        // [2 + 0.25*(-1)=1.75, 1 + 0.5*2=2.0, 0 + 0.75*(-2)=-1.5, 2 + 1.0*2=4.0]
        assert!((lerp_out[0] - 1.75).abs() < 1e-6);
        assert!((lerp_out[1] - 2.0).abs() < 1e-6);
        assert!((lerp_out[2] - (-1.5)).abs() < 1e-6);
        assert!((lerp_out[3] - 4.0).abs() < 1e-6);

        // LFM2 shortconv: dim=2, l_cache=3 => conv_state len = 2 * 2 = 4
        let mut sc_state = vec![0.5_f32, 1.0, -0.5, 0.25];
        let b_gate = vec![2.0_f32, -1.0];
        let c_gate = vec![0.5_f32, 2.0];
        let x_proj = vec![0.25_f32, 0.5];
        let conv_weight = vec![1.0_f32, 0.5, 2.0, 1.0, 1.0, -1.0];
        let mut sc_out = vec![0.0_f32; 2];

        lfm2_shortconv_step_f32(
            &mut sc_state,
            &b_gate,
            &c_gate,
            &x_proj,
            &conv_weight,
            None,
            &mut sc_out,
            2,
            3,
        )
        .expect("lfm2 shortconv step");
        // ch0: u=0.5, conv = 0.5*1 + 1*0.5 + 0.5*2 = 2.0, * c_gate(0.5) = 1.0
        // ch1: u=-0.5, conv = -0.5*1 + 0.25*1 + (-0.5)*(-1) = 0.25, * c_gate(2.0) = 0.5
        assert!((sc_out[0] - 1.0).abs() < 1e-6);
        assert!((sc_out[1] - 0.5).abs() < 1e-6);
        assert_eq!(sc_state, vec![1.0, 0.5, 0.25, -0.5]);
    }

    #[test]
    fn test_mamba2_conv1d_chunk_matches_step_by_step() {
        let channels = 4;
        let kernel_size = 3;
        let seq_len = 5;
        let init_conv = vec![0.1_f32, -0.2, 0.3, 0.0, -0.4, 0.25, 0.15, -0.1];
        let x_seq: Vec<f32> = (0..seq_len * channels)
            .map(|i| (f32::from(u8::try_from(i % 13).unwrap_or(0)) - 6.0) * 0.15)
            .collect();
        let w_conv = vec![0.5_f32; channels * kernel_size];
        let b_conv = vec![0.05_f32; channels];

        let mut step_conv_state = init_conv.clone();
        let mut step_conv_out = vec![0.0_f32; seq_len * channels];
        for t in 0..seq_len {
            let off = t * channels;
            mamba2_conv1d_silu_step(
                &mut step_conv_state,
                &x_seq[off..off + channels],
                &w_conv,
                Some(&b_conv),
                &mut step_conv_out[off..off + channels],
                channels,
                kernel_size,
            )
            .expect("conv1d step");
        }
        let mut chunk_conv_state = init_conv;
        let mut chunk_conv_out = vec![0.0_f32; seq_len * channels];
        mamba2_conv1d_silu_chunk_f32(
            &mut chunk_conv_state,
            &x_seq,
            &w_conv,
            Some(&b_conv),
            &mut chunk_conv_out,
            seq_len,
            channels,
            kernel_size,
        )
        .expect("conv1d chunk");
        for (got, expected) in chunk_conv_out.iter().zip(step_conv_out.iter()) {
            assert!((got - expected).abs() < 1e-6);
        }
        assert_eq!(step_conv_state, chunk_conv_state);
    }

    #[test]
    fn test_mamba2_ssd_chunk_scan_matches_step_by_step_and_rejects_bad_groups() {
        let (seq_len, n_heads, head_dim, state_size, n_groups) = (5, 4, 2, 4, 2);
        let init_ssd = vec![0.05_f32; n_heads * head_dim * state_size];
        let dt_seq = vec![0.2_f32; seq_len * n_heads];
        let dt_bias = vec![0.05_f32; n_heads];
        let a_neg = vec![0.4_f32; n_heads];
        let b_seq: Vec<f32> = (0..seq_len * n_groups * state_size)
            .map(|i| (f32::from(u8::try_from(i % 11).unwrap_or(0)) - 5.0) * 0.2)
            .collect();
        let c_seq: Vec<f32> = (0..seq_len * n_groups * state_size)
            .map(|i| (f32::from(u8::try_from((i * 3 + 1) % 11).unwrap_or(0)) - 5.0) * 0.2)
            .collect();
        let d_skip = vec![0.25_f32; n_heads];
        let z_seq: Vec<f32> = (0..seq_len * n_heads * head_dim)
            .map(|i| f32::from(u8::try_from(i % 7).unwrap_or(0)) * 0.1)
            .collect();
        let x_ssd: Vec<f32> = (0..seq_len * n_heads * head_dim)
            .map(|i| (f32::from(u8::try_from(i % 13).unwrap_or(0)) - 6.0) * 0.15)
            .collect();

        let mut step_ssd_state = init_ssd.clone();
        let mut step_ssd_out = vec![0.0_f32; x_ssd.len()];
        for t in 0..seq_len {
            let d_off = t * n_heads * head_dim;
            let h_off = t * n_heads;
            let g_off = t * n_groups * state_size;
            mamba2_ssd_state_step_f32(
                &mut step_ssd_state,
                &mut step_ssd_out[d_off..d_off + n_heads * head_dim],
                &x_ssd[d_off..d_off + n_heads * head_dim],
                &dt_seq[h_off..h_off + n_heads],
                Some(&dt_bias),
                &a_neg,
                &b_seq[g_off..g_off + n_groups * state_size],
                &c_seq[g_off..g_off + n_groups * state_size],
                Some(&d_skip),
                Some(&z_seq[d_off..d_off + n_heads * head_dim]),
                n_heads,
                head_dim,
                state_size,
                n_groups,
            )
            .expect("ssd step");
        }
        let mut chunk_ssd_state = init_ssd;
        let mut chunk_ssd_out = vec![0.0_f32; x_ssd.len()];
        mamba2_ssd_chunk_scan_f32(
            &mut chunk_ssd_state,
            &mut chunk_ssd_out,
            &x_ssd,
            &dt_seq,
            Some(&dt_bias),
            &a_neg,
            &b_seq,
            &c_seq,
            Some(&d_skip),
            Some(&z_seq),
            seq_len,
            n_heads,
            head_dim,
            state_size,
            n_groups,
        )
        .expect("ssd chunk scan");
        for (got, expected) in chunk_ssd_out.iter().zip(step_ssd_out.iter()) {
            assert!((got - expected).abs() < 1e-6);
        }
        for (got, expected) in chunk_ssd_state.iter().zip(step_ssd_state.iter()) {
            assert!((got - expected).abs() < 1e-6);
        }

        let bad_div = mamba2_ssd_chunk_scan_f32(
            &mut chunk_ssd_state,
            &mut chunk_ssd_out,
            &x_ssd,
            &dt_seq,
            None,
            &a_neg,
            &b_seq,
            &c_seq,
            None,
            None,
            seq_len,
            3,
            head_dim,
            state_size,
            2,
        );
        assert!(matches!(bad_div, Err(K3Error::CapabilityMismatch { .. })));
    }

    #[test]
    fn test_mamba2_ssd_intra_chunk_ime_i8_causal_cosine() {
        let (seq_len, state_size, n_groups) = (5, 4, 2);
        let b_seq: Vec<f32> = (0..seq_len * n_groups * state_size)
            .map(|i| (f32::from(u8::try_from(i % 11).unwrap_or(0)) - 5.0) * 0.2)
            .collect();
        let c_seq: Vec<f32> = (0..seq_len * n_groups * state_size)
            .map(|i| (f32::from(u8::try_from((i * 3 + 1) % 11).unwrap_or(0)) - 5.0) * 0.2)
            .collect();
        let mut scores_i8 = vec![-99.0_f32; n_groups * seq_len * seq_len];
        mamba2_ssd_intra_chunk_ime_i8(
            &c_seq,
            &b_seq,
            &mut scores_i8,
            seq_len,
            n_groups,
            state_size,
        )
        .expect("intra chunk ime i8");
        let mut dot_num = 0.0_f64;
        let mut norm_a = 0.0_f64;
        let mut norm_b = 0.0_f64;
        for g in 0..n_groups {
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let got = scores_i8[g * seq_len * seq_len + i * seq_len + j];
                    if j > i {
                        assert_eq!(got.to_bits(), 0.0_f32.to_bits());
                    } else {
                        let c_off = (i * n_groups + g) * state_size;
                        let b_off = (j * n_groups + g) * state_size;
                        let ref_dot: f32 = (0..state_size)
                            .map(|k| c_seq[c_off + k] * b_seq[b_off + k])
                            .sum();
                        dot_num += f64::from(got) * f64::from(ref_dot);
                        norm_a += f64::from(got) * f64::from(got);
                        norm_b += f64::from(ref_dot) * f64::from(ref_dot);
                    }
                }
            }
        }
        let cosine = dot_num / (norm_a.sqrt() * norm_b.sqrt());
        assert!(cosine > 0.999, "intra-chunk cosine {cosine} <= 0.999");
    }

    #[test]
    fn test_rwkv7_lerp_and_wkv_sequences_match_step_by_step() {
        let seq_len = 4;
        let n_heads = 2;
        let head_size = 3;
        let dim = n_heads * head_size;
        let prev_init = vec![0.25_f32, -0.5, 0.75, 1.0, -0.25, 0.5];
        let mu = vec![0.2_f32, 0.4, 0.6, 0.8, 0.1, 0.9];
        let x_seq: Vec<f32> = (0..seq_len * dim)
            .map(|i| f32::from(u8::try_from((i * 5 + 3) % 13).unwrap_or(0)) * 0.2 - 1.0)
            .collect();

        let mut step_prev = prev_init.clone();
        let mut step_lerp = vec![0.0_f32; seq_len * dim];
        for t in 0..seq_len {
            let off = t * dim;
            rwkv_token_shift_lerp_f32(
                &step_prev,
                &x_seq[off..off + dim],
                &mu,
                &mut step_lerp[off..off + dim],
            )
            .expect("lerp step");
            step_prev.copy_from_slice(&x_seq[off..off + dim]);
        }

        let mut seq_prev = prev_init;
        let mut seq_lerp = vec![0.0_f32; seq_len * dim];
        rwkv_token_shift_lerp_sequence_f32(&mut seq_prev, &x_seq, &mu, &mut seq_lerp, seq_len, dim)
            .expect("lerp seq");
        assert_eq!(step_lerp, seq_lerp);
        assert_eq!(step_prev, seq_prev);

        let w_seq = vec![0.85_f32; seq_len * dim];
        let a_seq = vec![0.1_f32; seq_len * dim];
        let b_seq = vec![-0.1_f32; seq_len * dim];
        let k_seq = x_seq.clone();
        let v_seq = seq_lerp.clone();
        let r_seq = vec![0.5_f32; seq_len * dim];
        let init_state = vec![0.02_f32; n_heads * head_size * head_size];

        let mut st_step = init_state.clone();
        let mut out_step = vec![0.0_f32; seq_len * dim];
        for t in 0..seq_len {
            let off = t * dim;
            rwkv7_wkv_delta_rule_step_f32(
                &mut st_step,
                &mut out_step[off..off + dim],
                &r_seq[off..off + dim],
                &w_seq[off..off + dim],
                &k_seq[off..off + dim],
                &v_seq[off..off + dim],
                &a_seq[off..off + dim],
                &b_seq[off..off + dim],
                n_heads,
                head_size,
            )
            .expect("rwkv7 step");
        }

        let mut st_seq = init_state;
        let mut out_seq = vec![0.0_f32; seq_len * dim];
        rwkv7_wkv_delta_rule_sequence_f32(
            &mut st_seq,
            &mut out_seq,
            &r_seq,
            &w_seq,
            &k_seq,
            &v_seq,
            &a_seq,
            &b_seq,
            seq_len,
            n_heads,
            head_size,
        )
        .expect("rwkv7 sequence");
        assert_eq!(out_step, out_seq);
        assert_eq!(st_step, st_seq);
    }
}
