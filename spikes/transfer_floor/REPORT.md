# T1 report: transfer floor and the choice of R

Phase 5, task 1 (`docs/phase5/T1-transfer-floor.md`). Run on 2026-10-26.

## Machine and reproduction

- Apple M2 Pro (as Phase 0 T3), macOS (Darwin 25.6.0), rustc 1.99.0, release profile,
  f64, flex build of `sciml-rs` (no GPU, no `datagen`). Base commit `f8476d3`.
- Datasets: the main checkout's `datasets/poisson/`, stage 3 (`poisson_n129_K64_test`)
  and stage 4 (`poisson_n257_K64_test`), 200 test samples each. Nothing regenerated.
- Command, from the repository root (wall time 8.3 s for the run itself, 1 min 15 s
  including the first release build):
  ```sh
  cargo run --release -p transfer-floor-spike -- <main checkout>/datasets/poisson > out.md
  ```
  Every table below is `out.md` verbatim.

## Setup

- **Grids.** n = 129 CGL nodes, x_j = −cos(πj/128); the learning grid is
  `uniform_nodes(s)`, s ∈ {16, 32, 64, 128}, both endpoints included.
- **Chebyshev coefficients.** A closed-form DCT-I matrix C (n × n) maps nodal values to
  the coefficients of the degree-(n − 1) interpolant; E(x)[i, k] = T_k(x_i) evaluates
  them. Built in f64 in the spike, no FFTW.
- **Candidate R** (each a dense `[s, n]` matrix, applied along both axes by
  `chebyshev::transfer::apply`):
  - *unfiltered T_cu:* `cheb_to_uniform(n, s)`, the interpolant sampled at the uniform
    nodes;
  - *cell average:* the exact mean of the interpolant over
    [x̃_i − h/2, x̃_i + h/2] ∩ [−1, 1], h = 2/(s − 1) (endpoint cells are half-cells),
    from the closed-form antiderivative of T_k. It is exact, so it equals any quadrature of
    sufficient degree, Clenshaw–Curtis included;
  - *filtered:* E(x̃) · diag(σ) · C with cut-off J ∈ {s − 2, ⌊(s − 2)π/2⌋ (capped at
    n − 1 = 128, which happens at s = 128), n − 1}. The taper is sharp (σ_k = 1 for k ≤ J,
    else 0) or exponential (σ_k = exp(−36 (k/J)^8) for k ≤ J, else 0). Sharp with J = n − 1
    is σ ≡ 1, the unfiltered R, so it is used only for the assert.
- **Candidate P:** `uniform_to_cheb(s, n, d)`, Floater–Hormann, d ∈ {2, 3, 4}.
- **Errors.** Relative L² with Clenshaw–Curtis weights on the 129 CGL grid
  (`cc_rel_l2_error`). Each cell shows `mean / max` over the 200 test samples.
- **Asserts** (run by `main`, both pass): σ ≡ 1 reproduces `cheb_to_uniform(n, s)` to
  1e-12 at every s, and every R has rows summing to 1 to 1e-12 (constants preserved).

## Results

## Reference check (stage 4 at the even 257 nodes vs stage 3, 200 test samples)

| Field | max relative CC-L² |
| --- | --- |
| u | 7.19e-13 |
| f | 0.00e0 |

Asserts passed: σ ≡ 1 reproduces T_cu, and every R has unit row sums (1e-12).

## Transfer-only error ‖P R u − u‖ / ‖u‖, P = T_uc d = 2 (mean / max)

| R | s = 16 | s = 32 | s = 64 | s = 128 |
| --- | --- | --- | --- | --- |
| unfiltered T_cu | 3.03e-3 / 1.22e-2 | 3.47e-4 / 1.35e-3 | 2.45e-5 / 9.50e-5 | 2.53e-6 / 9.50e-6 |
| cell average | 2.63e-2 / 4.70e-2 | 8.54e-3 / 1.47e-2 | 2.89e-3 / 4.91e-3 | 1.00e-3 / 1.70e-3 |
| filtered sharp, J = s − 2 | 5.21e-3 / 2.21e-2 | 6.06e-4 / 2.38e-3 | 6.63e-5 / 2.58e-4 | 2.53e-6 / 9.50e-6 |
| filtered sharp, J = ⌊(s − 2)π/2⌋ | 2.79e-3 / 1.11e-2 | 3.15e-4 / 1.24e-3 | 2.45e-5 / 9.44e-5 | 2.53e-6 / 9.50e-6 |
| filtered exp, J = s − 2 | 2.23e-2 / 1.07e-1 | 2.45e-3 / 1.10e-2 | 2.83e-4 / 1.11e-3 | 2.79e-5 / 1.15e-4 |
| filtered exp, J = ⌊(s − 2)π/2⌋ | 7.18e-3 / 2.89e-2 | 6.81e-4 / 2.76e-3 | 7.46e-5 / 2.87e-4 | 2.61e-5 / 1.08e-4 |
| filtered exp, J = n − 1 | 3.03e-3 / 1.22e-2 | 3.42e-4 / 1.33e-3 | 3.46e-5 / 1.39e-4 | 2.61e-5 / 1.08e-4 |

## Transfer-only error ‖P R u − u‖ / ‖u‖, P = T_uc d = 3 (mean / max)

| R | s = 16 | s = 32 | s = 64 | s = 128 |
| --- | --- | --- | --- | --- |
| unfiltered T_cu | 3.35e-3 / 1.47e-2 | 3.44e-4 / 1.35e-3 | 1.86e-5 / 6.98e-5 | 6.50e-7 / 2.51e-6 |
| cell average | 2.47e-2 / 4.51e-2 | 8.06e-3 / 1.39e-2 | 2.74e-3 / 4.64e-3 | 9.47e-4 / 1.61e-3 |
| filtered sharp, J = s − 2 | 5.35e-3 / 2.30e-2 | 6.06e-4 / 2.37e-3 | 6.43e-5 / 2.49e-4 | 6.50e-7 / 2.51e-6 |
| filtered sharp, J = ⌊(s − 2)π/2⌋ | 3.07e-3 / 1.34e-2 | 3.11e-4 / 1.24e-3 | 1.85e-5 / 6.93e-5 | 6.50e-7 / 2.51e-6 |
| filtered exp, J = s − 2 | 2.23e-2 / 1.07e-1 | 2.45e-3 / 1.10e-2 | 2.82e-4 / 1.11e-3 | 2.78e-5 / 1.15e-4 |
| filtered exp, J = ⌊(s − 2)π/2⌋ | 7.20e-3 / 2.92e-2 | 6.73e-4 / 2.73e-3 | 7.26e-5 / 2.80e-4 | 2.60e-5 / 1.07e-4 |
| filtered exp, J = n − 1 | 3.34e-3 / 1.47e-2 | 3.39e-4 / 1.34e-3 | 3.04e-5 / 1.23e-4 | 2.60e-5 / 1.07e-4 |

## Transfer-only error ‖P R u − u‖ / ‖u‖, P = T_uc d = 4 (mean / max)

| R | s = 16 | s = 32 | s = 64 | s = 128 |
| --- | --- | --- | --- | --- |
| unfiltered T_cu | 4.17e-3 / 1.90e-2 | 3.98e-4 / 1.54e-3 | 2.30e-5 / 8.71e-5 | 2.35e-7 / 9.15e-7 |
| cell average | 2.38e-2 / 4.43e-2 | 7.77e-3 / 1.34e-2 | 2.64e-3 / 4.47e-3 | 9.12e-4 / 1.55e-3 |
| filtered sharp, J = s − 2 | 5.69e-3 / 2.45e-2 | 6.38e-4 / 2.48e-3 | 6.55e-5 / 2.54e-4 | 2.35e-7 / 9.15e-7 |
| filtered sharp, J = ⌊(s − 2)π/2⌋ | 3.77e-3 / 1.74e-2 | 3.67e-4 / 1.43e-3 | 2.29e-5 / 8.67e-5 | 2.35e-7 / 9.15e-7 |
| filtered exp, J = s − 2 | 2.23e-2 / 1.07e-1 | 2.45e-3 / 1.10e-2 | 2.83e-4 / 1.11e-3 | 2.78e-5 / 1.15e-4 |
| filtered exp, J = ⌊(s − 2)π/2⌋ | 7.25e-3 / 2.97e-2 | 6.83e-4 / 2.76e-3 | 7.34e-5 / 2.84e-4 | 2.60e-5 / 1.07e-4 |
| filtered exp, J = n − 1 | 4.17e-3 / 1.90e-2 | 3.93e-4 / 1.53e-3 | 3.26e-5 / 1.32e-4 | 2.60e-5 / 1.07e-4 |

## P's own error: P applied to u sampled by unfiltered T_cu (mean / max)

| d | s = 16 | s = 32 | s = 64 | s = 128 |
| --- | --- | --- | --- | --- |
| 2 | 3.03e-3 / 1.22e-2 | 3.47e-4 / 1.35e-3 | 2.45e-5 / 9.50e-5 | 2.53e-6 / 9.50e-6 |
| 3 | 3.35e-3 / 1.47e-2 | 3.44e-4 / 1.35e-3 | 1.86e-5 / 6.98e-5 | 6.50e-7 / 2.51e-6 |
| 4 | 4.17e-3 / 1.90e-2 | 3.98e-4 / 1.54e-3 | 2.30e-5 / 8.71e-5 | 2.35e-7 / 9.15e-7 |

## Filtered-out energy of f, ‖f − F f‖² / ‖f‖² on the CGL grid (mean / max)

| Filter | s = 16 | s = 32 | s = 64 | s = 128 |
| --- | --- | --- | --- | --- |
| filtered sharp, J = s − 2 | 4.16e-2 / 9.99e-2 | 8.96e-3 / 1.99e-2 | 1.21e-3 / 2.62e-3 | 1.68e-18 / 3.57e-18 |
| filtered sharp, J = ⌊(s − 2)π/2⌋ | 1.90e-2 / 4.23e-2 | 2.97e-3 / 6.34e-3 | 1.59e-5 / 3.58e-5 | 5.15e-29 / 7.82e-29 |
| filtered exp, J = s − 2 | 1.06e-1 / 2.59e-1 | 2.46e-2 / 5.73e-2 | 4.91e-3 / 1.07e-2 | 3.59e-4 / 7.97e-4 |
| filtered exp, J = ⌊(s − 2)π/2⌋ | 4.95e-2 / 1.14e-1 | 9.46e-3 / 2.08e-2 | 1.29e-3 / 2.77e-3 | 3.23e-4 / 7.16e-4 |
| filtered exp, J = n − 1 | 3.23e-4 / 7.16e-4 | 3.23e-4 / 7.16e-4 | 3.23e-4 / 7.16e-4 | 3.23e-4 / 7.16e-4 |

## Aliasing visible in f

Skipped: the exact sine series needs `pde::poisson::grf`, behind `datagen`.

## Cost of R at n = 129: build / apply per sample (µs)

| R | s = 16 | s = 32 | s = 64 | s = 128 |
| --- | --- | --- | --- | --- |
| unfiltered T_cu | 5.5 / 22 | 9.6 / 37 | 18.4 / 76 | 36.8 / 187 |
| cell average | 128.9 / 21 | 229.6 / 38 | 407.9 / 76 | 762.9 / 192 |
| filtered sharp, J = s − 2 | 56.2 / 21 | 78.9 / 37 | 155.4 / 76 | 269.0 / 187 |
| filtered sharp, J = ⌊(s − 2)π/2⌋ | 45.2 / 22 | 83.8 / 36 | 149.3 / 76 | 278.7 / 186 |
| filtered exp, J = s − 2 | 47.7 / 21 | 79.7 / 36 | 152.7 / 77 | 273.3 / 186 |
| filtered exp, J = ⌊(s − 2)π/2⌋ | 47.9 / 21 | 83.5 / 37 | 157.8 / 75 | 271.5 / 186 |
| filtered exp, J = n − 1 | 46.1 / 21 | 78.4 / 36 | 151.0 / 75 | 278.1 / 185 |

## Dataset generation time (sidecar `wall_time_s`)

| Set | train (s) | test (s) |
| --- | --- | --- |
| n = 129, K = 64 | 1.1 | 0.2 |
| n = 257, K = 64 | 5.2 | 1.1 |

## Rule: lowest mean transfer-only error, within 10% → simpler R

| d | s | lowest mean | chosen R | its mean |
| --- | --- | --- | --- | --- |
| 2 | 32 | 3.154e-4 | unfiltered T_cu | 3.468e-4 |
| 2 | 64 | 2.446e-5 | unfiltered T_cu | 2.455e-5 |
| 3 | 32 | 3.112e-4 | filtered sharp, J = ⌊(s − 2)π/2⌋ | 3.112e-4 |
| 3 | 64 | 1.850e-5 | unfiltered T_cu | 1.859e-5 |
| 4 | 32 | 3.669e-4 | unfiltered T_cu | 3.977e-4 |
| 4 | 64 | 2.294e-5 | unfiltered T_cu | 2.305e-5 |

## Reading the tables

- **Reference check passes.** Stage 4 at the even-indexed 257 nodes matches stage 3 to
  7.2e-13 in u, which is the label-check level. f matches bit for bit, because both stages
  evaluate the same sine series at the same points. The phase continues.
- **P dominates the transfer-only error.** The unfiltered R samples the n = 129
  interpolant exactly, and that interpolant is accurate to round-off on the smooth u. Its
  row is therefore P's own error. Every other R is worse or within a few per cent.
- **Filtering removes real content of u.** The sharp J = s − 2 filter discards modes that
  u still carries (the filtered-out energy of f is 9.0e-3 at s = 32), so it roughly doubles
  the error. The exponential taper also damps the modes below J, which costs about 1e-5
  even at J = n − 1. The sharp J = ⌊(s − 2)π/2⌋ filter keeps almost all of u, and it is
  the lowest R at s = 32 and 64: 9% below unfiltered at s = 32 (d = 2) and 0.4% below at
  s = 64. The exponential J = n − 1 filter is 1.4% below unfiltered at s = 32, but above
  it at s = 64.
- **Cell average is the worst R** at every s ≥ 32: 8.5e-3 at s = 32. It smooths u by
  O(h²), which is far above the interpolation error of P.
- **Aliasing visible in f** is not measured. The exact sine series needs `grf`, which
  sits behind `datagen`, and the author chose to skip the row (2026-10-26). The
  filtered-out energy of f is the proxy: at s = 32, the band k > s − 2 holds 0.9% of f's
  energy on average (2% at worst), and unfiltered T_cu folds it back onto the learning
  grid.
- **Cost** does not separate the candidates. Every R takes under 1 ms to build and
  20–190 µs per sample to apply, against 0.2 s to generate the stage 3 test split. The
  build time of the filtered and cell-average R includes the Chebyshev matrices they are
  built from (C, with E or A), which unfiltered T_cu does not need.

## Decisions

- **The rule** (brief deliverable 6): the lowest mean transfer-only error at s = 32 and 64;
  candidates within 10% of it go to the simpler R (unfiltered T_cu, then cell average, then
  filtered).
- **d = 2 is kept.** P stays d = 2 unless it dominates at small s. At s = 16 and 32, d = 2
  is the best d or within 1% of it (P's own error 3.03e-3 and 3.47e-4). d = 3 and d = 4
  only help at s ≥ 64, where every error is already below 2.5e-5.
- **D1: R = unfiltered T_cu, P = Floater–Hormann d = 2.** At d = 2 the lowest mean is the
  sharp ⌊(s − 2)π/2⌋ filter (3.15e-4 at s = 32, 2.45e-5 at s = 64). Unfiltered T_cu is
  within 10% at both (3.47e-4, +10.0%; 2.45e-5, +0.4%), so the rule picks it. At s = 32 it
  passes by a hair: the ratio is 1.0996.
- **Consequences.** T2 needs no filtered restriction and no DCT matrix in `src/`. It
  generalises `Transfers` to an independent (n, s) with `cheb_to_uniform` and d = 2. The T5
  aliasing control at s = 32 (unfiltered T_cu) is then the main arm, so it is not run (T5
  brief, deliverable 3).
- **Caveat.** The rule scores R on u only. Because the aliasing row is skipped, what
  unfiltered T_cu does to the input f at small s (about 1% of f's energy beyond s − 2 at
  s = 32, 4% at s = 16) is not measured here. With D1 as chosen, T5 has no arm that
  separates it.
