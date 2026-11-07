//! `SpacemiT` IME (`XSMTVDot`, `smt.vmadot`) int8 matrix kernel plus the baseline RVV 1.0 and scalar
//! int8 references it is compared against (US-03 Scenario 2, ADR-034).
//!
//! All three paths accumulate `i8 x i8` products into exact `i32` sums, so they are numerically
//! bit-identical; only their throughput differs. That makes the IME on/off delta attributable to
//! the instruction path alone.
//!
//! # `vmadot` semantics (measured on the K3 `A100` cluster, `VLEN = 1024`, `XSMTVDotII`)
//!
//! On the `A100` cluster (`VLEN = 1024`, `vlenb = 128`, `+xsmtvdotii`), `smt.vmadot v0, v8, v16`
//! treats `v8` as a row-major `8 x 16` `i8` tile `A` (128 bytes), `v16` as a row-major `8 x 16`
//! `i8` tile `B` (128 bytes, each row is one output column, `k`-contiguous) and accumulates the
//! `8 x 8` `i32` product `C[i][j] += sum_k A[i][k] * B[j][k]` into `v0:v1` (64 `i32` elements,
//! row-major).

use jero_core::{K3Error, QuantType, TensorView};
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::quant::{Q8_0_BLOCK_BYTES, Q8_0_BLOCK_ELEMENTS};

/// Storage size in bytes of a single `TQ1_0` block (2B FP16 scale + 52B packed base-3 quants).
pub const TQ1_0_BLOCK_BYTES: usize = 54;

/// Number of scalar elements in a single `TQ1_0` block.
pub const TQ1_0_BLOCK_ELEMENTS: usize = 256;

/// Storage size in bytes of a single `TQ2_0` block (2B FP16 scale + 64B packed 2-bit quants).
pub const TQ2_0_BLOCK_BYTES: usize = 66;

/// Number of scalar elements in a single `TQ2_0` block.
pub const TQ2_0_BLOCK_ELEMENTS: usize = 256;

/// Rows of the `A` tile (and rows of the `C` tile) on `A100` (`VLEN = 1024`).
pub const TILE_M: usize = 8;
/// Rows of the `B` tile (columns of the `C` tile) on `A100` (`VLEN = 1024`).
pub const TILE_N: usize = 8;
/// Reduction depth consumed by one `smt.vmadot` on `A100` (`VLEN = 1024`).
pub const TILE_K: usize = 16;
/// Bytes in one packed `A` or `B` tile (`8 x 16` `i8` = 128 bytes = `vlenb`).
pub const TILE_BYTES: usize = TILE_M * TILE_K;
/// Number of `i32` values in one `C` tile (`8 x 8` = 64 `i32`).
pub const TILE_C: usize = TILE_M * TILE_N;

/// Whether this build contains the real `smt.vmadot` instruction path (`riscv64` only, not miri).
pub const IME_HARDWARE_PATH: bool = cfg!(all(target_arch = "riscv64", not(miri)));

static IME_TILE_CALLS: AtomicU64 = AtomicU64::new(0);
static VECTOR_DOT_CALLS: AtomicU64 = AtomicU64::new(0);

/// Monotonic count of `smt.vmadot` instructions issued by [`gemm_i8_ime`] in this process.
#[must_use]
pub fn ime_vmadot_count() -> u64 {
    IME_TILE_CALLS.load(Ordering::Relaxed)
}

/// Monotonic count of baseline RVV int8 dot-product kernel invocations in this process.
#[must_use]
pub fn vector_dot_count() -> u64 {
    VECTOR_DOT_CALLS.load(Ordering::Relaxed)
}

/// Number of `TILE_M`/`TILE_N` tiles needed to cover `rows` rows.
#[must_use]
pub const fn tile_count(rows: usize) -> usize {
    rows.div_ceil(TILE_M)
}

/// Packs a row-major `rows x k` `i8` matrix into `[row_tile][k_chunk][4][8]` tiles, zero-padding
/// missing rows. `k` must be a multiple of [`TILE_K`].
///
/// # Panics
///
/// Panics if `k` is not a multiple of [`TILE_K`] or `src.len() != rows * k`.
#[must_use]
pub fn pack_tiles(src: &[i8], rows: usize, k: usize) -> Vec<i8> {
    assert!(k % TILE_K == 0, "k must be a multiple of {TILE_K}");
    assert_eq!(src.len(), rows * k, "source length must equal rows * k");
    let k_chunks = k / TILE_K;
    let mut out = vec![0_i8; tile_count(rows) * k_chunks * TILE_BYTES];
    for row in 0..rows {
        let (tile, r) = (row / TILE_M, row % TILE_M);
        for chunk in 0..k_chunks {
            let dst = (tile * k_chunks + chunk) * TILE_BYTES + r * TILE_K;
            let s = row * k + chunk * TILE_K;
            out[dst..dst + TILE_K].copy_from_slice(&src[s..s + TILE_K]);
        }
    }
    out
}

/// Portable scalar `4 x 4` tile product over `k_chunks` packed chunks.
#[must_use]
pub fn tile_dot_scalar(a: &[i8], b: &[i8], k_chunks: usize) -> [i32; TILE_C] {
    let mut c = [0_i32; TILE_C];
    for chunk in 0..k_chunks {
        let at = &a[chunk * TILE_BYTES..(chunk + 1) * TILE_BYTES];
        let bt = &b[chunk * TILE_BYTES..(chunk + 1) * TILE_BYTES];
        for i in 0..TILE_M {
            for j in 0..TILE_N {
                for k in 0..TILE_K {
                    c[i * TILE_N + j] +=
                        i32::from(at[i * TILE_K + k]) * i32::from(bt[j * TILE_K + k]);
                }
            }
        }
    }
    c
}

/// `8 x 8` tile product via the `smt.vmadot` IME instruction over `k_chunks` packed chunks.
///
/// # Safety
///
/// The CPU executing this must implement the `SpacemiT` `XSMTVDotII` extension with `VLEN = 1024`
/// (`vlenb = 128`, `A100` cluster; see [`ime_probe_exec`]); otherwise it raises `SIGILL` or
/// computes a narrow tile. `a` and `b` must each hold at least `k_chunks * TILE_BYTES` bytes.
#[cfg(all(target_arch = "riscv64", not(miri)))]
#[allow(unused_assignments)]
unsafe fn tile_dot_ime(a: *const i8, b: *const i8, k_chunks: usize, c: *mut i32) {
    let (mut pa, mut pb, mut n) = (a, b, k_chunks);
    // SAFETY: `pa`/`pb` read `k_chunks * 128` bytes (caller contract) and `c` receives 64 `i32`.
    // The raw `.word 0xe304302b` is `smt.vmadot v0, v8, v16, i8` (verified with llvm-mc `+xsmtvdotii`).
    unsafe {
        core::arch::asm!(
            "li t0, 64",
            "vsetvli zero, t0, e32, m2, ta, ma",
            "vmv.v.i v0, 0",
            "li t0, 128",
            "vsetvli zero, t0, e8, m1, ta, ma",
            "2:",
            "vle8.v v8, ({pa})",
            "vle8.v v16, ({pb})",
            ".word 0xe304302b",
            "addi {pa}, {pa}, 128",
            "addi {pb}, {pb}, 128",
            "addi {n}, {n}, -1",
            "bnez {n}, 2b",
            "li t0, 64",
            "vsetvli zero, t0, e32, m2, ta, ma",
            "vse32.v v0, ({pc})",
            pa = inout(reg) pa,
            pb = inout(reg) pb,
            n = inout(reg) n,
            pc = in(reg) c,
            out("t0") _,
            out("v0") _,
            out("v1") _,
            out("v8") _,
            out("v16") _,
            options(nostack),
        );
    }
}

#[cfg(all(target_arch = "riscv64", not(miri)))]
#[inline]
fn is_vlen_1024() -> bool {
    let vlenb: usize;
    // SAFETY: reads the read-only `vlenb` CSR.
    unsafe {
        core::arch::asm!("csrr {0}, vlenb", out(reg) vlenb, options(nomem, nostack));
    }
    vlenb == 128
}

/// Executes an `XSMTVDotII` probe (`smt.vpack.vv` + `smt.vmadot`) on the calling CPU and reports
/// whether it ran with `VLEN = 1024` (`vlenb = 128`, `A100` cluster).
///
/// Only async-signal-safe operations are used so a forked, core-pinned child can call it; a CPU
/// without the extension (`X100`) traps with `SIGILL` or fails the `vlenb == 128` check. On
/// non-`riscv64` builds this returns `false`.
#[must_use]
pub fn ime_probe_exec() -> bool {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        if !is_vlen_1024() {
            return false;
        }
        let a = [1_i8; TILE_BYTES];
        let b = [1_i8; TILE_BYTES];
        let mut c = [0_i32; TILE_C];
        // SAFETY: `.word 0x6704102b` is `smt.vpack.vv v0, v8, v16, 1` (`XSMTVDotII`), which traps
        // with SIGILL on cores without Matrix 2.0 (`X100`), followed by one `8 x 8` `smt.vmadot`.
        unsafe {
            core::arch::asm!(
                "li t0, 128",
                "vsetvli zero, t0, e8, m1, ta, ma",
                ".word 0x6704102b",
                out("t0") _,
                out("v0") _,
                out("v8") _,
                out("v16") _,
                options(nomem, nostack),
            );
            tile_dot_ime(a.as_ptr(), b.as_ptr(), 1, c.as_mut_ptr());
        }
        c.iter().all(|&v| v == 16)
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        false
    }
}

/// Computes `C = A * B^T` tiles with the IME path for the `B` tile rows in `n_tiles`.
///
/// `a_packed` holds `m_tiles` packed `A` tile rows and `b_packed` the packed weight tiles (both
/// from [`pack_tiles`] with `k = k_chunks * TILE_K`). `sink(m_tile, n_tile, tile)` receives every
/// finished `8 x 8` tile. When executed on `A100` (`VLEN = 1024`), this issues hardware
/// `smt.vmadot` instructions; off `A100` or off `riscv64`, the bit-identical scalar tile product
/// stands in for the instruction.
pub fn gemm_i8_ime<F>(
    a_packed: &[i8],
    m_tiles: usize,
    b_packed: &[i8],
    n_tiles: Range<usize>,
    k_chunks: usize,
    mut sink: F,
) where
    F: FnMut(usize, usize, &[i32; TILE_C]),
{
    let stride = k_chunks * TILE_BYTES;
    let mut issued = 0_u64;
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    let hw_ime = is_vlen_1024();
    for nt in n_tiles {
        let b = &b_packed[nt * stride..(nt + 1) * stride];
        for mt in 0..m_tiles {
            let a = &a_packed[mt * stride..(mt + 1) * stride];
            #[cfg(all(target_arch = "riscv64", not(miri)))]
            let mut c = [0_i32; TILE_C];
            #[cfg(all(target_arch = "riscv64", not(miri)))]
            if hw_ime {
                // SAFETY: slices hold `k_chunks` 128-byte tiles and `is_vlen_1024()` confirmed we
                // are running on an A100 core (`vlenb = 128`).
                unsafe {
                    tile_dot_ime(a.as_ptr(), b.as_ptr(), k_chunks, c.as_mut_ptr());
                }
            } else {
                c = tile_dot_scalar(a, b, k_chunks);
            }
            #[cfg(any(not(target_arch = "riscv64"), miri))]
            let c = tile_dot_scalar(a, b, k_chunks);
            issued += k_chunks as u64;
            sink(mt, nt, &c);
        }
    }
    IME_TILE_CALLS.fetch_add(issued, Ordering::Relaxed);
}

/// Portable scalar int8 dot product with exact `i32` accumulation.
#[must_use]
pub fn dot_i8_scalar(a: &[i8], b: &[i8]) -> i32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| i32::from(x) * i32::from(y))
        .sum()
}

/// Baseline RVV 1.0 int8 dot product (`vle8`, `vwmul.vv`, `vwredsum.vs`) with exact `i32` sums.
#[cfg(all(target_arch = "riscv64", not(miri)))]
#[must_use]
#[allow(unused_assignments)]
pub fn dot_i8_rvv(a: &[i8], b: &[i8]) -> i32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0;
    }
    let (mut pa, mut pb, mut rem) = (a.as_ptr(), b.as_ptr(), len);
    let out: i64;
    // SAFETY: `pa`/`pb` are valid for `len` bytes; strip-mined by `vsetvli`, accumulated into
    // `v0[0]` with a widening reduction (exact, order independent for integers).
    unsafe {
        core::arch::asm!(
            "vsetivli zero, 1, e32, m1, ta, ma",
            "vmv.v.i v0, 0",
            "2:",
            "vsetvli t0, {rem}, e8, m1, ta, ma",
            "vle8.v v1, ({pa})",
            "vle8.v v2, ({pb})",
            "vwmul.vv v4, v1, v2",
            "vsetvli zero, t0, e16, m2, ta, ma",
            "vwredsum.vs v0, v4, v0",
            "add {pa}, {pa}, t0",
            "add {pb}, {pb}, t0",
            "sub {rem}, {rem}, t0",
            "bnez {rem}, 2b",
            "vsetivli zero, 1, e32, m1, ta, ma",
            "vmv.x.s {out}, v0",
            pa = inout(reg) pa,
            pb = inout(reg) pb,
            rem = inout(reg) rem,
            out = out(reg) out,
            out("t0") _,
            out("v0") _,
            out("v1") _,
            out("v2") _,
            out("v4") _,
            out("v5") _,
            options(nostack, readonly),
        );
    }
    #[expect(clippy::cast_possible_truncation, reason = "vmv.x.s of an e32 element")]
    {
        out as i32
    }
}

/// Baseline int8 dot product: RVV on `riscv64`, scalar elsewhere.
#[must_use]
pub fn dot_i8_vector(a: &[i8], b: &[i8]) -> i32 {
    #[cfg(all(target_arch = "riscv64", not(miri)))]
    {
        dot_i8_rvv(a, b)
    }
    #[cfg(any(not(target_arch = "riscv64"), miri))]
    {
        dot_i8_scalar(a, b)
    }
}

/// Computes `C = A * B^T` with the baseline (non-IME) vector path for output columns `n_range`.
///
/// `a_rows` is the row-major `m x k` activation matrix and `b_rows` the row-major `n x k` weight
/// matrix. `sink(m, n, value)` receives every output element.
pub fn gemm_i8_vector<F>(
    a_rows: &[i8],
    m: usize,
    b_rows: &[i8],
    n_range: Range<usize>,
    k: usize,
    mut sink: F,
) where
    F: FnMut(usize, usize, i32),
{
    let mut calls = 0_u64;
    for n in n_range {
        let b = &b_rows[n * k..(n + 1) * k];
        for row in 0..m {
            sink(row, n, dot_i8_vector(&a_rows[row * k..(row + 1) * k], b));
            calls += 1;
        }
    }
    VECTOR_DOT_CALLS.fetch_add(calls, Ordering::Relaxed);
}

/// Unpacks a 54-byte `TQ1_0` block into 256 `i8` values in `{-1, 0, +1}`.
///
/// # Block Layout
/// - `block[0..2]`: IEEE 754 half-precision (`f16`) little-endian block scale factor.
/// - `block[2..53]`: 51 bytes containing 255 packed base-3 ternary values (5 values per byte, $3^5 = 243$).
/// - `block[53]`: 1 byte containing the 256th ternary value.
///
/// Each base-3 trit $t \in \{0, 1, 2\}$ is mapped to $t - 1 \in \{-1, 0, +1\}$.
///
/// Returns the finite dequantization scale factor `f32`.
#[must_use]
pub fn unpack_tq1_0_block_i8(
    block: &[u8; TQ1_0_BLOCK_BYTES],
    out: &mut [i8; TQ1_0_BLOCK_ELEMENTS],
) -> f32 {
    let d_bits = u16::from_le_bytes([block[0], block[1]]);
    let scale = crate::f16_to_f32_finite(d_bits);

    for i in 0..51 {
        let mut b = block[2 + i];
        let base = i * 5;
        #[expect(
            clippy::cast_possible_wrap,
            reason = "modulo 3 minus 1 yields {-1, 0, 1}"
        )]
        for j in 0..5 {
            out[base + j] = ((b % 3) as i8) - 1;
            b /= 3;
        }
    }

    #[expect(
        clippy::cast_possible_wrap,
        reason = "modulo 3 minus 1 yields {-1, 0, 1}"
    )]
    {
        out[255] = ((block[53] % 3) as i8) - 1;
    }

    scale
}

/// Unpacks a 66-byte `TQ2_0` block into 256 `i8` values in `{-1, 0, +1}`.
///
/// # Block Layout
/// - `block[0..2]`: IEEE 754 half-precision (`f16`) little-endian block scale factor.
/// - `block[2..66]`: 64 bytes containing 256 packed 2-bit values (4 values per byte).
///
/// Each 2-bit value $q \in \{0, 1, 2\}$ is mapped to $q - 1 \in \{-1, 0, +1\}$.
///
/// Returns the finite dequantization scale factor `f32`.
#[must_use]
pub fn unpack_tq2_0_block_i8(
    block: &[u8; TQ2_0_BLOCK_BYTES],
    out: &mut [i8; TQ2_0_BLOCK_ELEMENTS],
) -> f32 {
    let d_bits = u16::from_le_bytes([block[0], block[1]]);
    let scale = crate::f16_to_f32_finite(d_bits);

    for (i, &byte) in block[2..66].iter().enumerate() {
        let base = i * 4;
        #[expect(
            clippy::cast_possible_wrap,
            reason = "quant shift-mask 0..=3 minus 1 fits in i8"
        )]
        {
            out[base] = (((byte) & 0x03) as i8) - 1;
            out[base + 1] = (((byte >> 2) & 0x03) as i8) - 1;
            out[base + 2] = (((byte >> 4) & 0x03) as i8) - 1;
            out[base + 3] = (((byte >> 6) & 0x03) as i8) - 1;
        }
    }

    scale
}

fn validate_bitlinear_dimensions(
    output_len: usize,
    weights: &TensorView<'_>,
    input_len: usize,
) -> Result<(usize, usize, usize, usize), K3Error> {
    let quant = weights.quant();
    let (block_elems, block_bytes) = match quant {
        QuantType::TQ1_0 => (TQ1_0_BLOCK_ELEMENTS, TQ1_0_BLOCK_BYTES),
        QuantType::TQ2_0 => (TQ2_0_BLOCK_ELEMENTS, TQ2_0_BLOCK_BYTES),
        QuantType::Q8_0 => (Q8_0_BLOCK_ELEMENTS, Q8_0_BLOCK_BYTES),
        _ => {
            return Err(K3Error::TensorBoundsError {
                tensor_name: weights.name().to_owned(),
                offset: weights.offset(),
                byte_len: weights.byte_len(),
                buffer_len: weights.as_bytes().len(),
                reason: format!(
                    "unsupported quantization format for BitLinear: {quant:?}; expected TQ1_0, TQ2_0, or Q8_0"
                ),
            });
        }
    };

    let shape = weights.shape();
    if shape.is_empty() {
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: "empty tensor shape".to_owned(),
        });
    }

    if shape.len() == 2 {
        let dim0 = shape[0];
        let dim1 = shape[1];
        if (output_len == dim0 && input_len == dim1 && dim1 % block_elems == 0)
            || (output_len == dim1 && input_len == dim0 && dim0 % block_elems == 0)
        {
            let required_blocks = (output_len * input_len) / block_elems;
            let required_bytes = required_blocks * block_bytes;
            if weights.as_bytes().len() < required_bytes {
                return Err(K3Error::TensorBoundsError {
                    tensor_name: weights.name().to_owned(),
                    offset: weights.offset(),
                    byte_len: weights.byte_len(),
                    buffer_len: weights.as_bytes().len(),
                    reason: format!(
                        "backing buffer length {} is smaller than required {} bytes",
                        weights.as_bytes().len(),
                        required_bytes
                    ),
                });
            }
            return Ok((output_len, input_len, block_elems, block_bytes));
        }
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: format!(
                "matrix shape [{dim0}, {dim1}] does not match output len {output_len} and block-aligned input len {input_len}"
            ),
        });
    }

    let total_elems = weights.num_elements();
    if input_len == 0
        || output_len == 0
        || input_len % block_elems != 0
        || output_len * input_len != total_elems
    {
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: format!(
                "tensor elements {total_elems} do not match output len {output_len} * input len {input_len}"
            ),
        });
    }

    let required_blocks = (output_len * input_len) / block_elems;
    let required_bytes = required_blocks * block_bytes;
    if weights.as_bytes().len() < required_bytes {
        return Err(K3Error::TensorBoundsError {
            tensor_name: weights.name().to_owned(),
            offset: weights.offset(),
            byte_len: weights.byte_len(),
            buffer_len: weights.as_bytes().len(),
            reason: format!(
                "backing buffer length {} is smaller than required {} bytes",
                weights.as_bytes().len(),
                required_bytes
            ),
        });
    }

    Ok((output_len, input_len, block_elems, block_bytes))
}

/// Computes matrix-vector multiplication for `BitLinear` layers using standard RVV integer vector instructions.
///
/// Supports `QuantType::TQ1_0`, `QuantType::TQ2_0`, and `QuantType::Q8_0` weights with on-the-fly unpacking.
/// Intermediate dot-product accumulation in `i32` is bit-identical to [`bitlinear_i8_ime`].
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if quantization is unsupported or dimensions mismatch.
pub fn bitlinear_i8_rvv(
    out: &mut [f32],
    weights: &TensorView<'_>,
    act_i8: &[i8],
    act_scale: f32,
) -> Result<(), K3Error> {
    let (rows, cols, block_elems, block_bytes) =
        validate_bitlinear_dimensions(out.len(), weights, act_i8.len())?;
    let blocks_per_row = cols / block_elems;
    let raw_bytes = weights.as_bytes();
    let quant = weights.quant();

    let mut w_buf = [0_i8; 256];
    let mut dot_calls = 0_u64;

    for (r, out_val) in out.iter_mut().enumerate().take(rows) {
        let mut row_sum = 0.0_f32;
        for blk in 0..blocks_per_row {
            let blk_idx = r * blocks_per_row + blk;
            let byte_start = blk_idx * block_bytes;
            let byte_end = byte_start + block_bytes;
            let block_raw = &raw_bytes[byte_start..byte_end];

            let (w_slice, block_scale): (&[i8], f32) = match quant {
                QuantType::TQ1_0 => {
                    let block: &[u8; 54] =
                        block_raw
                            .try_into()
                            .map_err(|_| K3Error::TensorBoundsError {
                                tensor_name: weights.name().to_owned(),
                                offset: weights.offset() + byte_start,
                                byte_len: block_bytes,
                                buffer_len: raw_bytes.len(),
                                reason: "invalid TQ1_0 block slice".to_owned(),
                            })?;
                    let scale = unpack_tq1_0_block_i8(block, &mut w_buf);
                    (&w_buf[..256], scale)
                }
                QuantType::TQ2_0 => {
                    let block: &[u8; 66] =
                        block_raw
                            .try_into()
                            .map_err(|_| K3Error::TensorBoundsError {
                                tensor_name: weights.name().to_owned(),
                                offset: weights.offset() + byte_start,
                                byte_len: block_bytes,
                                buffer_len: raw_bytes.len(),
                                reason: "invalid TQ2_0 block slice".to_owned(),
                            })?;
                    let scale = unpack_tq2_0_block_i8(block, &mut w_buf);
                    (&w_buf[..256], scale)
                }
                QuantType::Q8_0 => {
                    let block: &[u8; 34] =
                        block_raw
                            .try_into()
                            .map_err(|_| K3Error::TensorBoundsError {
                                tensor_name: weights.name().to_owned(),
                                offset: weights.offset() + byte_start,
                                byte_len: block_bytes,
                                buffer_len: raw_bytes.len(),
                                reason: "invalid Q8_0 block slice".to_owned(),
                            })?;
                    let d_bits = u16::from_le_bytes([block[0], block[1]]);
                    let scale = crate::f16_to_f32_finite(d_bits);
                    #[expect(
                        clippy::cast_possible_wrap,
                        reason = "reinterpreting u8 quant as signed i8"
                    )]
                    for i in 0..32 {
                        w_buf[i] = block[2 + i] as i8;
                    }
                    (&w_buf[..32], scale)
                }
                _ => unreachable!(),
            };

            let act_sub = &act_i8[blk * block_elems..(blk + 1) * block_elems];
            let dot_i32 = dot_i8_vector(w_slice, act_sub);
            dot_calls += 1;

            let effective_scale = block_scale * act_scale;
            #[expect(
                clippy::cast_precision_loss,
                reason = "intermediate dot_i32 fits within 24-bit mantissa precision"
            )]
            {
                row_sum += (dot_i32 as f32) * effective_scale;
            }
        }
        *out_val = row_sum;
    }

    VECTOR_DOT_CALLS.fetch_add(dot_calls, Ordering::Relaxed);
    Ok(())
}

/// Computes matrix-vector multiplication for `BitLinear` layers using A100 IME (`smt.vmadot`) acceleration.
///
/// Supports `QuantType::TQ1_0`, `QuantType::TQ2_0`, and `QuantType::Q8_0` weights with on-the-fly unpacking.
/// Intermediate dot-product accumulation in `i32` is bit-identical to [`bitlinear_i8_rvv`].
///
/// # Errors
///
/// Returns [`K3Error::TensorBoundsError`] if quantization is unsupported or dimensions mismatch.
#[expect(
    clippy::too_many_lines,
    reason = "matrix-vector tiling loop with on-the-fly unpack and IME dispatch"
)]
pub fn bitlinear_i8_ime(
    out: &mut [f32],
    weights: &TensorView<'_>,
    act_i8: &[i8],
    act_scale: f32,
) -> Result<(), K3Error> {
    let (rows, cols, block_elems, block_bytes) =
        validate_bitlinear_dimensions(out.len(), weights, act_i8.len())?;
    let blocks_per_row = cols / block_elems;
    let raw_bytes = weights.as_bytes();
    let quant = weights.quant();
    let k_chunks = block_elems / TILE_K; // 16 for TQ1/TQ2, 2 for Q8_0

    out.fill(0.0);

    #[cfg(all(target_arch = "riscv64", not(miri)))]
    let hw_ime = is_vlen_1024();

    let mut a_tile = [0_i8; 16 * TILE_BYTES];
    let mut b_tile = [0_i8; 16 * TILE_BYTES];
    let mut w_buf = [0_i8; 256];
    let mut scales = [0.0_f32; TILE_N];
    let mut ime_calls = 0_u64;

    for r_base in (0..rows).step_by(TILE_N) {
        let tile_rows = (rows - r_base).min(TILE_N);

        for blk in 0..blocks_per_row {
            let act_sub = &act_i8[blk * block_elems..(blk + 1) * block_elems];

            // 1. Pack Tile A: broadcast act_sub to all TILE_M=8 rows
            for ch in 0..k_chunks {
                let act_chunk = &act_sub[ch * TILE_K..(ch + 1) * TILE_K];
                let ch_base = ch * TILE_BYTES;
                for i in 0..TILE_M {
                    a_tile[ch_base + i * TILE_K..ch_base + (i + 1) * TILE_K]
                        .copy_from_slice(act_chunk);
                }
            }

            // 2. Pack Tile B: unpack up to 8 output rows
            b_tile[..k_chunks * TILE_BYTES].fill(0);
            for j in 0..tile_rows {
                let r = r_base + j;
                let blk_idx = r * blocks_per_row + blk;
                let byte_start = blk_idx * block_bytes;
                let byte_end = byte_start + block_bytes;
                let block_raw = &raw_bytes[byte_start..byte_end];

                let (w_slice, scale): (&[i8], f32) = match quant {
                    QuantType::TQ1_0 => {
                        let block: &[u8; 54] =
                            block_raw
                                .try_into()
                                .map_err(|_| K3Error::TensorBoundsError {
                                    tensor_name: weights.name().to_owned(),
                                    offset: weights.offset() + byte_start,
                                    byte_len: block_bytes,
                                    buffer_len: raw_bytes.len(),
                                    reason: "invalid TQ1_0 block slice".to_owned(),
                                })?;
                        let s = unpack_tq1_0_block_i8(block, &mut w_buf);
                        (&w_buf[..256], s)
                    }
                    QuantType::TQ2_0 => {
                        let block: &[u8; 66] =
                            block_raw
                                .try_into()
                                .map_err(|_| K3Error::TensorBoundsError {
                                    tensor_name: weights.name().to_owned(),
                                    offset: weights.offset() + byte_start,
                                    byte_len: block_bytes,
                                    buffer_len: raw_bytes.len(),
                                    reason: "invalid TQ2_0 block slice".to_owned(),
                                })?;
                        let s = unpack_tq2_0_block_i8(block, &mut w_buf);
                        (&w_buf[..256], s)
                    }
                    QuantType::Q8_0 => {
                        let block: &[u8; 34] =
                            block_raw
                                .try_into()
                                .map_err(|_| K3Error::TensorBoundsError {
                                    tensor_name: weights.name().to_owned(),
                                    offset: weights.offset() + byte_start,
                                    byte_len: block_bytes,
                                    buffer_len: raw_bytes.len(),
                                    reason: "invalid Q8_0 block slice".to_owned(),
                                })?;
                        let d_bits = u16::from_le_bytes([block[0], block[1]]);
                        let s = crate::f16_to_f32_finite(d_bits);
                        #[expect(
                            clippy::cast_possible_wrap,
                            reason = "reinterpreting u8 quant as signed i8"
                        )]
                        for i in 0..32 {
                            w_buf[i] = block[2 + i] as i8;
                        }
                        (&w_buf[..32], s)
                    }
                    _ => unreachable!(),
                };

                scales[j] = scale;
                for ch in 0..k_chunks {
                    let w_chunk = &w_slice[ch * TILE_K..(ch + 1) * TILE_K];
                    let ch_base = ch * TILE_BYTES;
                    b_tile[ch_base + j * TILE_K..ch_base + (j + 1) * TILE_K]
                        .copy_from_slice(w_chunk);
                }
            }

            // 3. Compute 8x8 Tile Dot Product
            let c = {
                #[cfg(all(target_arch = "riscv64", not(miri)))]
                if hw_ime {
                    let mut c_hw = [0_i32; TILE_C];
                    // SAFETY: `a_tile` and `b_tile` have at least `k_chunks * TILE_BYTES` initialized bytes,
                    // `c` has space for 64 i32 values, and `is_vlen_1024()` confirmed A100 execution.
                    unsafe {
                        tile_dot_ime(
                            a_tile.as_ptr(),
                            b_tile.as_ptr(),
                            k_chunks,
                            c_hw.as_mut_ptr(),
                        );
                    }
                    c_hw
                } else {
                    tile_dot_scalar(
                        &a_tile[..k_chunks * TILE_BYTES],
                        &b_tile[..k_chunks * TILE_BYTES],
                        k_chunks,
                    )
                }
                #[cfg(any(not(target_arch = "riscv64"), miri))]
                {
                    tile_dot_scalar(
                        &a_tile[..k_chunks * TILE_BYTES],
                        &b_tile[..k_chunks * TILE_BYTES],
                        k_chunks,
                    )
                }
            };

            ime_calls += u64::try_from(k_chunks).unwrap_or(0);

            // 4. Accumulate scaled contribution for each row
            for j in 0..tile_rows {
                let dot_i32 = c[j]; // Row 0 of A against Row j of B
                let effective_scale = scales[j] * act_scale;
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "intermediate dot_i32 fits within 24-bit mantissa precision"
                )]
                {
                    out[r_base + j] += (dot_i32 as f32) * effective_scale;
                }
            }
        }
    }

    IME_TILE_CALLS.fetch_add(ime_calls, Ordering::Relaxed);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(all(target_arch = "riscv64", not(miri)))]
    fn pin_to_a100_for_test() {
        // SAFETY: Linux riscv64 `SYS_gettid` (178) and `SYS_sched_setaffinity` (122)
        // on the calling thread after registering with `/proc/set_ai_thread`.
        unsafe {
            let tid: isize;
            core::arch::asm!(
                "li a7, 178",
                "ecall",
                lateout("a0") tid,
                out("a7") _,
                options(nostack),
            );
            let _ = std::fs::write("/proc/set_ai_thread", format!("{tid}"));
            let mask: u64 = 1_u64 << 8;
            let rc: isize;
            core::arch::asm!(
                "li a7, 122",
                "ecall",
                inlateout("a0") 0_isize => rc,
                in("a1") size_of::<u64>(),
                in("a2") core::ptr::addr_of!(mask),
                out("a7") _,
                options(nostack),
            );
            let _ = rc;
        }
    }

    #[expect(clippy::cast_possible_truncation, reason = "bounded modulo 17")]
    fn fill(len: usize, salt: usize) -> Vec<i8> {
        (0..len)
            .map(|i| (((i * 7 + salt * 13) % 17) as i8) - 8)
            .collect()
    }

    #[test]
    fn ime_vector_and_scalar_paths_are_bit_identical() {
        #[cfg(all(target_arch = "riscv64", not(miri)))]
        pin_to_a100_for_test();
        #[cfg(all(target_arch = "riscv64", not(miri)))]
        assert!(ime_probe_exec());
        let (rows_m, cols_n, depth_k) = (6, 12, 64);
        let lhs = fill(rows_m * depth_k, 1);
        let rhs = fill(cols_n * depth_k, 2);
        let (ap, bp) = (
            pack_tiles(&lhs, rows_m, depth_k),
            pack_tiles(&rhs, cols_n, depth_k),
        );
        let mut ime = vec![0_i32; tile_count(rows_m) * TILE_M * tile_count(cols_n) * TILE_N];
        let stride = tile_count(cols_n) * TILE_N;
        gemm_i8_ime(
            &ap,
            tile_count(rows_m),
            &bp,
            0..tile_count(cols_n),
            depth_k / TILE_K,
            |mt, nt, tile| {
                for i in 0..TILE_M {
                    for j in 0..TILE_N {
                        ime[(mt * TILE_M + i) * stride + nt * TILE_N + j] = tile[i * TILE_N + j];
                    }
                }
            },
        );
        let mut vec_out = vec![0_i32; rows_m * cols_n];
        gemm_i8_vector(&lhs, rows_m, &rhs, 0..cols_n, depth_k, |r, col, val| {
            vec_out[r * cols_n + col] = val;
        });
        for r in 0..rows_m {
            for col in 0..cols_n {
                let exact = dot_i8_scalar(
                    &lhs[r * depth_k..(r + 1) * depth_k],
                    &rhs[col * depth_k..(col + 1) * depth_k],
                );
                assert_eq!(vec_out[r * cols_n + col], exact);
                assert_eq!(ime[r * stride + col], exact);
            }
        }
        assert!(ime_vmadot_count() > 0);
        assert!(vector_dot_count() > 0);
    }

    #[test]
    fn pack_tiles_pads_and_orders_rows() {
        let src: Vec<i8> = (0..32).collect();
        let packed = pack_tiles(&src, 2, 16);
        assert_eq!(packed.len(), TILE_BYTES);
        assert_eq!(&packed[..32], &src[..]);
        assert!(packed[32..].iter().all(|&v| v == 0));
    }
}
