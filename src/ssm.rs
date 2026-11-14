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
#[expect(
    clippy::too_many_arguments,
    reason = "RWKV7 row update passes disjoint state/weight/key slices and scalar multipliers"
)]
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
}
