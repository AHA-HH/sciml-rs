# Flex's f64 FFT uses an f32 twiddle table (f64 FFTs are only f32-accurate)

**Status:** open, a documented limitation rather than a bug; not filed upstream yet
(2026-11-13). Present on the fork `ddfa9af` and in Burn 0.23.0-pre.2.

## Summary

On the Flex (CPU) backend, an f64 `rfft` multiplies by the shared twiddle table, which
is stored in f32 (`get_twiddles` → `TwiddleRef`, `Vec<f32>`). The source says so:

```rust
// burn-flex-0.23.0-pre.2/src/ops/fft.rs:1153
/// f64 complex FFT using f32 twiddle table (widened in inner loop).
/// Twiddle precision is limited to ~7 digits (f32), so output accuracy
/// is below full f64 precision for large N.
fn fft_f64_inplace(re: &mut [f64], im: &mut [f64], n: usize, tw_re: &[f32], ...
```

So f64 FFTs, and any f64 model with a spectral layer, are no more accurate than f32.

## Environment

Burn 0.23.0-pre.2 (and the fork `ax1s-x1zz/burn@ddfa9af`), Flex backend; Apple M2 Pro,
macOS 26.6.2, rustc 1.99.0.

## Reproduction

`tests/gauss_newton_f64.rs`, `repro_flex_rfft_f64_is_f64_accurate` (`#[ignore]`d;
`cargo test --test gauss_newton_f64 -- --ignored`): a length-16 f64 `rfft` against the
exact DFT.

```text
f64 rfft max error 1.89e-7      (f64 would give ~1e-15)
```

`f64_oracle_and_adjoint_on_flex` in the same file fails for the same reason (vjp about
1e-7 off, finite-difference jvp about 1e-3 at the f64 step).

## Impact

No f64 FNO on the CPU: Gauss–Newton's f64 checks cannot run on Flex, and with cuda f64
also blocked ([../../cubek/f64-matmul-declined-on-cuda/](../../cubek/f64-matmul-declined-on-cuda/README.md))
there is no f64 backend for the FNO on Burn 0.23.

## Suggested change

Build an f64 twiddle table for the f64 paths (`rfft_f64`, `fft_f64_inplace` and the
inverses), computed at runtime in f64.
