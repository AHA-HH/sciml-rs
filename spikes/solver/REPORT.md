# T4 report: reference-solver spike

Phase 0, task 4 (`docs/phase0/T4-solver-spike.md`). Run on 2026-10-07.

## Decisions

No stop condition fired. A-fd reaches 1.1e-14 (sin) and 1.2e-14 (polynomial) by n = 33
(rule: 1e-10), and A agrees with B to at most 3.4e-14 at n ≤ 33 (rule: 1e-10).

| Item | Decision | Rule outcome |
| --- | --- | --- |
| A's solve route | **A-fd (fast diagonalisation)** | A-syl fails both conditions: it agrees with A-fd only to 1.2e-12 (rule ≤ 1e-12, missed at n = 257), and with reuse it is 9.8× slower per solve at n = 257 (rule ≤ 2×) |
| C's discretisation | **C-q1 (Q1, consistent mass, consistent load)** | Both variants show order 2.00 on all three solutions at n = 65, 129, 257. Tie-break: C-q1 takes fewer CG iterations at n = 257 (4731 against 4752, the maximum over the three solutions at tol 1e-13) |
| C's CG tolerance | **1e-6** | It is the loosest tolerance in {1e-6, 1e-8, 1e-10}. With C-q1 at n = 257 the CG error is at most 1.2e-4 × the discretisation error on all three solutions (rule < 0.1×) |

Notes on the decisions:
- **A-fd is a route design §4.2 allows**, so nothing here contradicts the design. Design
  decision 1 names "Sylvester or fast diagonalisation"; the measured result is fast
  diagonalisation. The Sylvester route is the one ruled out.
- **The C tie-break is marginal.** C-q1 takes fewer iterations than C-fv at n = 257 on
  every solution and tolerance, but by less than 1.5%. Per solve it is 22–31% slower
  (9 non-zeros per row against 5), and its discretisation error is 1.3–3.5× larger. The
  brief's rule counts iterations, so it picks C-q1. If wall time or accuracy should decide
  instead, C-fv wins. That is a call for the reader of this report, not one the rule makes.
- **On GRF forcings A matches the exact series solution.** At n ≥ 129 (K = 64) the
  relative L² error of u_A against the exact sine-series solution is 1e-12 or less. At
  n = 33 with K = 16 it is 2.5e-7, because the top modes are barely resolved. That is
  stage 1, which design §3.3 already exempts.

## Findings about rlst 0.9.0

1. **`solve_sylvester` cannot reuse a factorisation.**
   - It computes the Schur forms of both A and B on every call
     (`src/dense/linalg/lapack/sylvester.rs`, lines 94–95).
   - Reuse is possible only through lower-level public pieces: `schur()` once, then
     `rlst::dense::linalg::lapack::interface::trsyl::Trsyl::trsyl` with four GEMMs per
     solve. That is the A-syl-reuse route measured here.
   - Even with reuse, trsyl's O(m³) back-substitution costs about 10× fast
     diagonalisation's pointwise division plus four GEMMs at n = 257.
2. **`eig(EigMode::RightEigenvectors)` panics on every input.**
   - `eig` passes `ldvl = n` with no left-vector buffer, and the geev wrapper asserts
     `ldvl == 1` ("Require `ldvl` 7 == 1").
   - `EigMode::BothEigenvectors` works and is used here. The left vectors are discarded,
     so the extra cost is in setup only.
   - Phase 1's `pde::poisson::collocation` needs the same workaround, or an upstream fix.
3. The eigenvalues and eigenvectors of the interior D² come back exactly real
   (max |Im| = 0 for every n here), as expected: LAPACK returns real eigenpairs of a real
   matrix with real spectrum.
4. `CgIteration::run()` returns only the residual. The iteration count comes from
   `set_callable`, which runs once per update.

## Machine and reproduction

- Apple M2 Pro, macOS 26.6.2 (Darwin 25.6.0), rustc 1.99.0, release profile, f64, Accelerate
  with its default settings.
- Direct dependencies: rlst =0.9.0 (`default-features = false`), ndarray =0.17.2, and
  blas-src 0.14 / lapack-src 0.13. The BLAS/LAPACK provider is chosen per target as in the
  root `chebyshev` feature: Accelerate on macOS, system OpenBLAS on Linux.
  `transfers-spike` (`../transfers`) supplies T3's GRF sampler, CGL nodes and
  Clenshaw–Curtis weights, so the GRF draws are T3's.
- Commands, run from `spikes/solver/`:
  ```sh
  cargo test --release                       # spike checks (7 tests)
  cargo run --release 2>/dev/null > out.md   # every table below, verbatim (about 60 s)
  /usr/bin/time -l cargo run --release -- --only-n 257   # peak memory at n = 257
  ```
- Seeds: GRF samples 1000..1004 (common-grid comparison) and 1000..1049 (cost). Draws are
  T3's: ξ ~ N(0, 1) on a 128 × 128 block, with truncation K(n) = min((n − 1)/2, 64).

## Setup

- **D.** Trefethen's `cheb` on the ascending CGL nodes, built with the negative-sum
  diagonal. D² = D · D. The interior block D_xx = D_yy has size (n − 2)².
- **A-fd.** At setup, `eig(D_xx) = V Λ V⁻¹`, using the real parts, and V⁻¹ from
  `inverse`. Each solve computes Ĝ = V_x⁻¹ F V_y⁻ᵀ, then Û_ij = Ĝ_ij / −(λ_i + μ_j),
  then U = V_x Û V_yᵀ.
- **A-syl-reuse.** At setup, the real Schur forms of A = −D_xx and B = −D_yyᵀ. Each solve
  computes Y = Q_aᵀ F Q_b, then `trsyl` (A X + X B = scale·C), then U = Q_a Y Q_bᵀ / scale.
- **A-syl-call.** `solve_sylvester` as shipped, called once per solve.
- **B.** The dense (n − 2)² Kronecker-sum system −(I ⊗ D_xx + D_yy ⊗ I) vec U = vec F,
  solved with rlst's LU `solve`, for n ≤ 33.
- **C-fv.** Vertex-centred finite volume:
  - dual cells with widths Δ_i = (x_{i+1} − x_{i−1})/2;
  - 5-point fluxes Δy_j/(x_{i+1} − x_i), and likewise in y;
  - load f_ij Δx_i Δy_j (lumped);
  - symmetric by construction.
- **C-q1.** Bilinear Q1 on the tensor mesh:
  - K = K_x ⊗ M_y + M_x ⊗ K_y, with 1D P1 stiffness and consistent mass (9-point);
  - **consistent load** M_x F M_y of the nodal interpolant of f, including the boundary
    nodal values of f.
- **Solving C.** Both systems use `CsrMatrix::from_aij` and `CgIteration`, with no
  preconditioner and a zero initial guess. The tolerance is CG's relative residual. The
  Dirichlet nodes are eliminated.
- **Errors.** "max" is max |e| / max |u_exact|. "L²" is the Clenshaw–Curtis relative L²
  on all n² CGL nodes (design §3.4).
- **Orders.** The observed order is log2 of the error ratio between successive n, because
  h_max halves exactly. The table also shows the order against h_max.
- **CG error.** The relative L² distance between u_C(tol) and u_C(1e-13).
- **Timing.** Setup times are medians of 3 runs. A's solve and the GRF evaluation are
  means over 50 samples. CG times are single solves, load included.

## Peak memory

`/usr/bin/time -l` measures the process-wide peak, over the `--only-n` run: A setup and
50 solves on each route, plus C-fv and C-q1 assembly and one CG solve each.

| n | maximum resident set size | peak memory footprint |
| --- | --- | --- |
| 129 | 29.9 MB | 25.2 MB |
| 257 | 120.1 MB | 115.6 MB |

## Measured tables

The tables below are `out.md`, pasted unchanged.

## 1. Differentiation matrices

| n | max rel \|D − closed form\| | max rel error of D² on x^k, k < n |
| --- | --- | --- |
| 9 | 2.7e-16 | 5.7e-14 |
| 17 | 9.6e-16 | 1.3e-12 |
| 33 | 1.4e-14 | 8.4e-11 |
| 65 | 1.4e-14 | 7.2e-10 |

## 2. Option A on the manufactured solutions

Errors relative to max |u| (max) and Clenshaw–Curtis relative L² (L²). |syl − fd| is max |U_syl − U_fd| / max |U_fd|.

| n | solution | A-fd max | A-fd L² | A-syl-reuse max | A-syl-reuse L² | A-syl-call max | \|syl − fd\| | trsyl status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9 | sin(πx)sin(πy) | 1.3e-4 | 1.1e-4 | 1.3e-4 | 1.1e-4 | 1.3e-4 | 1.6e-15 | Success / Success |
| 9 | (1−x²)(1−y²)(x+y²) | 2.0e-15 | 1.1e-15 | 1.4e-15 | 9.8e-16 | 1.3e-15 | 2.3e-15 | Success / Success |
| 9 | (1−x²)(1−y²)e^(x+2y) | 3.6e-5 | 2.3e-5 | 3.6e-5 | 2.3e-5 | 3.6e-5 | 1.9e-15 | Success / Success |
| 17 | sin(πx)sin(πy) | 2.8e-12 | 2.1e-12 | 2.8e-12 | 2.1e-12 | 2.8e-12 | 6.4e-15 | Success / Success |
| 17 | (1−x²)(1−y²)(x+y²) | 5.8e-15 | 5.0e-15 | 5.9e-15 | 5.1e-15 | 6.0e-15 | 6.8e-15 | Success / Success |
| 17 | (1−x²)(1−y²)e^(x+2y) | 5.7e-14 | 3.5e-14 | 5.7e-14 | 3.4e-14 | 5.7e-14 | 2.2e-15 | Success / Success |
| 33 | sin(πx)sin(πy) | 1.1e-14 | 9.7e-15 | 1.7e-14 | 9.8e-15 | 1.6e-14 | 1.8e-14 | Success / Success |
| 33 | (1−x²)(1−y²)(x+y²) | 1.2e-14 | 8.5e-15 | 2.7e-14 | 2.2e-14 | 2.7e-14 | 3.0e-14 | Success / Success |
| 33 | (1−x²)(1−y²)e^(x+2y) | 9.5e-15 | 7.2e-15 | 1.2e-14 | 8.2e-15 | 1.2e-14 | 1.9e-14 | Success / Success |
| 65 | sin(πx)sin(πy) | 4.6e-14 | 2.7e-14 | 7.8e-14 | 5.0e-14 | 7.8e-14 | 1.1e-13 | Success / Success |
| 65 | (1−x²)(1−y²)(x+y²) | 5.7e-14 | 4.1e-14 | 5.8e-14 | 4.3e-14 | 5.7e-14 | 8.7e-14 | Success / Success |
| 65 | (1−x²)(1−y²)e^(x+2y) | 9.5e-14 | 1.1e-13 | 1.3e-13 | 1.2e-13 | 1.3e-13 | 8.8e-14 | Success / Success |
| 129 | sin(πx)sin(πy) | 4.6e-13 | 2.9e-13 | 4.2e-13 | 3.0e-13 | 4.2e-13 | 7.0e-13 | Success / Success |
| 129 | (1−x²)(1−y²)(x+y²) | 3.9e-13 | 3.0e-13 | 6.2e-13 | 4.7e-13 | 6.2e-13 | 6.9e-13 | Success / Success |
| 129 | (1−x²)(1−y²)e^(x+2y) | 3.7e-13 | 3.5e-13 | 1.4e-13 | 1.4e-13 | 1.4e-13 | 4.6e-13 | Success / Success |
| 257 | sin(πx)sin(πy) | 5.5e-13 | 4.0e-13 | 8.3e-13 | 7.3e-13 | 8.3e-13 | 1.1e-12 | Success / Success |
| 257 | (1−x²)(1−y²)(x+y²) | 8.7e-13 | 5.9e-13 | 1.0e-12 | 6.2e-13 | 1.0e-12 | 1.2e-12 | Success / Success |
| 257 | (1−x²)(1−y²)e^(x+2y) | 1.2e-12 | 1.3e-12 | 4.6e-13 | 3.3e-13 | 4.6e-13 | 1.2e-12 | Success / Success |

Floor (smallest A-fd max error over n):
- sin(πx)sin(πy): 1.1e-14
- (1−x²)(1−y²)(x+y²): 2.0e-15
- (1−x²)(1−y²)e^(x+2y): 9.5e-15

Eigen-decomposition of the interior D² (A-fd setup), largest imaginary parts:

| n | max \|Im λ\| | max \|Im V\| |
| --- | --- | --- |
| 9 | 0.0e0 | 0.0e0 |
| 17 | 0.0e0 | 0.0e0 |
| 33 | 0.0e0 | 0.0e0 |
| 65 | 0.0e0 | 0.0e0 |
| 129 | 0.0e0 | 0.0e0 |
| 257 | 0.0e0 | 0.0e0 |

## 3. Option A against option B (dense Kronecker LU)

max |U_A − U_B| / max |U_B| on the interior.

| n | field | A-fd vs B | A-syl-reuse vs B |
| --- | --- | --- | --- |
| 9 | sin(πx)sin(πy) | 1.4e-15 | 1.7e-15 |
| 9 | (1−x²)(1−y²)(x+y²) | 3.6e-15 | 1.4e-15 |
| 9 | (1−x²)(1−y²)e^(x+2y) | 2.5e-15 | 1.8e-15 |
| 9 | GRF seed 1000 | 4.1e-15 | 2.0e-15 |
| 17 | sin(πx)sin(πy) | 8.5e-15 | 9.3e-15 |
| 17 | (1−x²)(1−y²)(x+y²) | 1.3e-14 | 1.3e-14 |
| 17 | (1−x²)(1−y²)e^(x+2y) | 1.3e-14 | 1.3e-14 |
| 17 | GRF seed 1000 | 4.4e-15 | 7.5e-15 |
| 33 | sin(πx)sin(πy) | 2.4e-14 | 2.1e-14 |
| 33 | (1−x²)(1−y²)(x+y²) | 2.1e-14 | 3.1e-14 |
| 33 | (1−x²)(1−y²)e^(x+2y) | 2.5e-14 | 3.4e-14 |
| 33 | GRF seed 1000 | 1.7e-14 | 8.2e-15 |

## 4. Option C on the manufactured solutions

CG tolerance 1e-13 (relative residual), zero initial guess, no preconditioner. Order is log2(e(previous n) / e(n)); h_max order uses log(e ratio)/log(h_max ratio).

| variant | solution | n | h_max | max err | L² err | order (L²) | order vs h_max (L²) | CG iters | final residual |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C-fv | sin(πx)sin(πy) | 33 | 9.802e-2 | 5.12e-3 | 4.43e-3 | NaN | NaN | 83 | 7.6e-14 |
| C-fv | sin(πx)sin(πy) | 65 | 4.907e-2 | 1.28e-3 | 1.10e-3 | 2.00 | 2.01 | 246 | 8.4e-14 |
| C-fv | sin(πx)sin(πy) | 129 | 2.454e-2 | 3.20e-4 | 2.76e-4 | 2.00 | 2.00 | 740 | 9.6e-14 |
| C-fv | sin(πx)sin(πy) | 257 | 1.227e-2 | 7.99e-5 | 6.89e-5 | 2.00 | 2.00 | 2209 | 9.3e-14 |
| C-fv | (1−x²)(1−y²)(x+y²) | 33 | 9.802e-2 | 4.52e-3 | 3.57e-3 | NaN | NaN | 161 | 6.3e-14 |
| C-fv | (1−x²)(1−y²)(x+y²) | 65 | 4.907e-2 | 1.13e-3 | 8.89e-4 | 2.00 | 2.01 | 466 | 7.4e-14 |
| C-fv | (1−x²)(1−y²)(x+y²) | 129 | 2.454e-2 | 2.81e-4 | 2.22e-4 | 2.00 | 2.00 | 1406 | 8.9e-14 |
| C-fv | (1−x²)(1−y²)(x+y²) | 257 | 1.227e-2 | 7.03e-5 | 5.56e-5 | 2.00 | 2.00 | 4216 | 9.9e-14 |
| C-fv | (1−x²)(1−y²)e^(x+2y) | 33 | 9.802e-2 | 3.28e-3 | 2.40e-3 | NaN | NaN | 182 | 8.1e-14 |
| C-fv | (1−x²)(1−y²)e^(x+2y) | 65 | 4.907e-2 | 8.22e-4 | 5.98e-4 | 2.00 | 2.01 | 528 | 9.9e-14 |
| C-fv | (1−x²)(1−y²)e^(x+2y) | 129 | 2.454e-2 | 2.05e-4 | 1.49e-4 | 2.00 | 2.00 | 1592 | 9.6e-14 |
| C-fv | (1−x²)(1−y²)e^(x+2y) | 257 | 1.227e-2 | 5.13e-5 | 3.74e-5 | 2.00 | 2.00 | 4752 | 9.7e-14 |
| C-q1 | sin(πx)sin(πy) | 33 | 9.802e-2 | 5.81e-3 | 5.68e-3 | NaN | NaN | 77 | 9.2e-14 |
| C-q1 | sin(πx)sin(πy) | 65 | 4.907e-2 | 1.45e-3 | 1.42e-3 | 2.00 | 2.00 | 237 | 8.8e-14 |
| C-q1 | sin(πx)sin(πy) | 129 | 2.454e-2 | 3.62e-4 | 3.56e-4 | 2.00 | 2.00 | 727 | 9.5e-14 |
| C-q1 | sin(πx)sin(πy) | 257 | 1.227e-2 | 9.05e-5 | 8.91e-5 | 2.00 | 2.00 | 2189 | 9.2e-14 |
| C-q1 | (1−x²)(1−y²)(x+y²) | 33 | 9.802e-2 | 9.77e-3 | 8.93e-3 | NaN | NaN | 149 | 9.3e-14 |
| C-q1 | (1−x²)(1−y²)(x+y²) | 65 | 4.907e-2 | 2.43e-3 | 2.23e-3 | 2.00 | 2.00 | 455 | 8.2e-14 |
| C-q1 | (1−x²)(1−y²)(x+y²) | 129 | 2.454e-2 | 6.08e-4 | 5.59e-4 | 2.00 | 2.00 | 1395 | 9.6e-14 |
| C-q1 | (1−x²)(1−y²)(x+y²) | 257 | 1.227e-2 | 1.52e-4 | 1.40e-4 | 2.00 | 2.00 | 4204 | 9.7e-14 |
| C-q1 | (1−x²)(1−y²)e^(x+2y) | 33 | 9.802e-2 | 7.22e-3 | 8.35e-3 | NaN | NaN | 169 | 7.3e-14 |
| C-q1 | (1−x²)(1−y²)e^(x+2y) | 65 | 4.907e-2 | 1.81e-3 | 2.09e-3 | 2.00 | 2.00 | 516 | 9.8e-14 |
| C-q1 | (1−x²)(1−y²)e^(x+2y) | 129 | 2.454e-2 | 4.52e-4 | 5.22e-4 | 2.00 | 2.00 | 1579 | 9.3e-14 |
| C-q1 | (1−x²)(1−y²)e^(x+2y) | 257 | 1.227e-2 | 1.13e-4 | 1.31e-4 | 2.00 | 2.00 | 4731 | 9.9e-14 |

## 5. CG tolerance

CG error is the CC relative L² distance to the 1e-13 solution; discretisation error is that solution's L² error. Time is one solve, load included.

| variant | solution | n | tol | CG iters | time [ms] | CG error | disc. error | CG / disc. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C-fv | sin(πx)sin(πy) | 33 | 1e-6 | 50 | 0.3 | 3.7e-8 | 4.4e-3 | 8.3e-6 |
| C-fv | sin(πx)sin(πy) | 33 | 1e-8 | 61 | 0.3 | 3.7e-10 | 4.4e-3 | 8.4e-8 |
| C-fv | sin(πx)sin(πy) | 33 | 1e-10 | 71 | 0.4 | 2.6e-12 | 4.4e-3 | 5.9e-10 |
| C-fv | sin(πx)sin(πy) | 33 | 1e-13 | 83 | 0.5 | 0.0e0 | 4.4e-3 | 0.0e0 |
| C-fv | sin(πx)sin(πy) | 65 | 1e-6 | 149 | 3.5 | 1.8e-8 | 1.1e-3 | 1.6e-5 |
| C-fv | sin(πx)sin(πy) | 65 | 1e-8 | 178 | 4.0 | 2.2e-10 | 1.1e-3 | 2.0e-7 |
| C-fv | sin(πx)sin(πy) | 65 | 1e-10 | 208 | 4.7 | 1.4e-12 | 1.1e-3 | 1.3e-9 |
| C-fv | sin(πx)sin(πy) | 65 | 1e-13 | 246 | 5.5 | 0.0e0 | 1.1e-3 | 0.0e0 |
| C-fv | sin(πx)sin(πy) | 129 | 1e-6 | 448 | 40.9 | 1.3e-8 | 2.8e-4 | 4.8e-5 |
| C-fv | sin(πx)sin(πy) | 129 | 1e-8 | 547 | 50.2 | 9.6e-11 | 2.8e-4 | 3.5e-7 |
| C-fv | sin(πx)sin(πy) | 129 | 1e-10 | 622 | 57.0 | 1.2e-12 | 2.8e-4 | 4.3e-9 |
| C-fv | sin(πx)sin(πy) | 129 | 1e-13 | 740 | 67.6 | 0.0e0 | 2.8e-4 | 0.0e0 |
| C-fv | sin(πx)sin(πy) | 257 | 1e-6 | 1347 | 503.3 | 8.6e-9 | 6.9e-5 | 1.2e-4 |
| C-fv | sin(πx)sin(πy) | 257 | 1e-8 | 1626 | 622.9 | 7.0e-11 | 6.9e-5 | 1.0e-6 |
| C-fv | sin(πx)sin(πy) | 257 | 1e-10 | 1872 | 697.3 | 7.3e-13 | 6.9e-5 | 1.1e-8 |
| C-fv | sin(πx)sin(πy) | 257 | 1e-13 | 2209 | 827.8 | 0.0e0 | 6.9e-5 | 0.0e0 |
| C-fv | (1−x²)(1−y²)(x+y²) | 33 | 1e-6 | 103 | 0.6 | 1.1e-7 | 3.6e-3 | 3.1e-5 |
| C-fv | (1−x²)(1−y²)(x+y²) | 33 | 1e-8 | 122 | 0.7 | 6.1e-10 | 3.6e-3 | 1.7e-7 |
| C-fv | (1−x²)(1−y²)(x+y²) | 33 | 1e-10 | 139 | 0.8 | 6.0e-12 | 3.6e-3 | 1.7e-9 |
| C-fv | (1−x²)(1−y²)(x+y²) | 33 | 1e-13 | 161 | 1.0 | 0.0e0 | 3.6e-3 | 0.0e0 |
| C-fv | (1−x²)(1−y²)(x+y²) | 65 | 1e-6 | 301 | 7.0 | 6.2e-8 | 8.9e-4 | 6.9e-5 |
| C-fv | (1−x²)(1−y²)(x+y²) | 65 | 1e-8 | 354 | 8.0 | 4.8e-10 | 8.9e-4 | 5.3e-7 |
| C-fv | (1−x²)(1−y²)(x+y²) | 65 | 1e-10 | 404 | 9.4 | 2.3e-12 | 8.9e-4 | 2.5e-9 |
| C-fv | (1−x²)(1−y²)(x+y²) | 65 | 1e-13 | 466 | 10.7 | 0.0e0 | 8.9e-4 | 0.0e0 |
| C-fv | (1−x²)(1−y²)(x+y²) | 129 | 1e-6 | 914 | 84.2 | 3.8e-8 | 2.2e-4 | 1.7e-4 |
| C-fv | (1−x²)(1−y²)(x+y²) | 129 | 1e-8 | 1076 | 97.8 | 2.7e-10 | 2.2e-4 | 1.2e-6 |
| C-fv | (1−x²)(1−y²)(x+y²) | 129 | 1e-10 | 1214 | 111.9 | 1.8e-12 | 2.2e-4 | 7.9e-9 |
| C-fv | (1−x²)(1−y²)(x+y²) | 129 | 1e-13 | 1406 | 131.3 | 0.0e0 | 2.2e-4 | 0.0e0 |
| C-fv | (1−x²)(1−y²)(x+y²) | 257 | 1e-6 | 2778 | 1120.2 | 1.6e-8 | 5.6e-5 | 3.0e-4 |
| C-fv | (1−x²)(1−y²)(x+y²) | 257 | 1e-8 | 3240 | 1198.3 | 1.6e-10 | 5.6e-5 | 2.9e-6 |
| C-fv | (1−x²)(1−y²)(x+y²) | 257 | 1e-10 | 3642 | 1364.1 | 9.4e-13 | 5.6e-5 | 1.7e-8 |
| C-fv | (1−x²)(1−y²)(x+y²) | 257 | 1e-13 | 4216 | 1560.8 | 0.0e0 | 5.6e-5 | 0.0e0 |
| C-fv | (1−x²)(1−y²)e^(x+2y) | 33 | 1e-6 | 111 | 0.6 | 1.2e-7 | 2.4e-3 | 4.9e-5 |
| C-fv | (1−x²)(1−y²)e^(x+2y) | 33 | 1e-8 | 132 | 0.7 | 8.0e-10 | 2.4e-3 | 3.3e-7 |
| C-fv | (1−x²)(1−y²)e^(x+2y) | 33 | 1e-10 | 154 | 0.8 | 4.9e-12 | 2.4e-3 | 2.0e-9 |
| C-fv | (1−x²)(1−y²)e^(x+2y) | 33 | 1e-13 | 182 | 1.0 | 0.0e0 | 2.4e-3 | 0.0e0 |
| C-fv | (1−x²)(1−y²)e^(x+2y) | 65 | 1e-6 | 323 | 7.3 | 6.3e-8 | 6.0e-4 | 1.1e-4 |
| C-fv | (1−x²)(1−y²)e^(x+2y) | 65 | 1e-8 | 384 | 8.6 | 5.4e-10 | 6.0e-4 | 9.0e-7 |
| C-fv | (1−x²)(1−y²)e^(x+2y) | 65 | 1e-10 | 441 | 10.0 | 3.5e-12 | 6.0e-4 | 5.9e-9 |
| C-fv | (1−x²)(1−y²)e^(x+2y) | 65 | 1e-13 | 528 | 11.8 | 0.0e0 | 6.0e-4 | 0.0e0 |
| C-fv | (1−x²)(1−y²)e^(x+2y) | 129 | 1e-6 | 992 | 90.4 | 3.3e-8 | 1.5e-4 | 2.2e-4 |
| C-fv | (1−x²)(1−y²)e^(x+2y) | 129 | 1e-8 | 1168 | 106.0 | 3.3e-10 | 1.5e-4 | 2.2e-6 |
| C-fv | (1−x²)(1−y²)e^(x+2y) | 129 | 1e-10 | 1336 | 123.0 | 2.1e-12 | 1.5e-4 | 1.4e-8 |
| C-fv | (1−x²)(1−y²)e^(x+2y) | 129 | 1e-13 | 1592 | 146.6 | 0.0e0 | 1.5e-4 | 0.0e0 |
| C-fv | (1−x²)(1−y²)e^(x+2y) | 257 | 1e-6 | 3000 | 1113.3 | 1.7e-8 | 3.7e-5 | 4.5e-4 |
| C-fv | (1−x²)(1−y²)e^(x+2y) | 257 | 1e-8 | 3521 | 1303.8 | 2.0e-10 | 3.7e-5 | 5.2e-6 |
| C-fv | (1−x²)(1−y²)e^(x+2y) | 257 | 1e-10 | 4016 | 1487.2 | 1.3e-12 | 3.7e-5 | 3.4e-8 |
| C-fv | (1−x²)(1−y²)e^(x+2y) | 257 | 1e-13 | 4752 | 1755.1 | 0.0e0 | 3.7e-5 | 0.0e0 |
| C-q1 | sin(πx)sin(πy) | 33 | 1e-6 | 46 | 0.3 | 1.9e-8 | 5.7e-3 | 3.4e-6 |
| C-q1 | sin(πx)sin(πy) | 33 | 1e-8 | 54 | 0.4 | 3.8e-10 | 5.7e-3 | 6.7e-8 |
| C-q1 | sin(πx)sin(πy) | 33 | 1e-10 | 65 | 0.5 | 2.1e-12 | 5.7e-3 | 3.7e-10 |
| C-q1 | sin(πx)sin(πy) | 33 | 1e-13 | 77 | 0.6 | 0.0e0 | 5.7e-3 | 0.0e0 |
| C-q1 | sin(πx)sin(πy) | 65 | 1e-6 | 139 | 4.0 | 2.4e-8 | 1.4e-3 | 1.7e-5 |
| C-q1 | sin(πx)sin(πy) | 65 | 1e-8 | 168 | 4.9 | 2.8e-10 | 1.4e-3 | 2.0e-7 |
| C-q1 | sin(πx)sin(πy) | 65 | 1e-10 | 197 | 5.8 | 2.0e-12 | 1.4e-3 | 1.4e-9 |
| C-q1 | sin(πx)sin(πy) | 65 | 1e-13 | 237 | 6.8 | 0.0e0 | 1.4e-3 | 0.0e0 |
| C-q1 | sin(πx)sin(πy) | 129 | 1e-6 | 437 | 52.4 | 1.3e-8 | 3.6e-4 | 3.7e-5 |
| C-q1 | sin(πx)sin(πy) | 129 | 1e-8 | 529 | 62.7 | 1.4e-10 | 3.6e-4 | 3.9e-7 |
| C-q1 | sin(πx)sin(πy) | 129 | 1e-10 | 611 | 72.6 | 1.2e-12 | 3.6e-4 | 3.3e-9 |
| C-q1 | sin(πx)sin(πy) | 129 | 1e-13 | 727 | 86.3 | 0.0e0 | 3.6e-4 | 0.0e0 |
| C-q1 | sin(πx)sin(πy) | 257 | 1e-6 | 1328 | 656.1 | 9.2e-9 | 8.9e-5 | 1.0e-4 |
| C-q1 | sin(πx)sin(πy) | 257 | 1e-8 | 1613 | 794.8 | 6.2e-11 | 8.9e-5 | 6.9e-7 |
| C-q1 | sin(πx)sin(πy) | 257 | 1e-10 | 1857 | 911.5 | 6.8e-13 | 8.9e-5 | 7.7e-9 |
| C-q1 | sin(πx)sin(πy) | 257 | 1e-13 | 2189 | 1069.7 | 0.0e0 | 8.9e-5 | 0.0e0 |
| C-q1 | (1−x²)(1−y²)(x+y²) | 33 | 1e-6 | 96 | 0.7 | 1.5e-7 | 8.9e-3 | 1.7e-5 |
| C-q1 | (1−x²)(1−y²)(x+y²) | 33 | 1e-8 | 115 | 0.8 | 5.8e-10 | 8.9e-3 | 6.4e-8 |
| C-q1 | (1−x²)(1−y²)(x+y²) | 33 | 1e-10 | 130 | 0.9 | 6.5e-12 | 8.9e-3 | 7.3e-10 |
| C-q1 | (1−x²)(1−y²)(x+y²) | 33 | 1e-13 | 149 | 1.1 | 0.0e0 | 8.9e-3 | 0.0e0 |
| C-q1 | (1−x²)(1−y²)(x+y²) | 65 | 1e-6 | 296 | 8.6 | 5.2e-8 | 2.2e-3 | 2.3e-5 |
| C-q1 | (1−x²)(1−y²)(x+y²) | 65 | 1e-8 | 347 | 10.2 | 4.6e-10 | 2.2e-3 | 2.0e-7 |
| C-q1 | (1−x²)(1−y²)(x+y²) | 65 | 1e-10 | 394 | 11.3 | 2.7e-12 | 2.2e-3 | 1.2e-9 |
| C-q1 | (1−x²)(1−y²)(x+y²) | 65 | 1e-13 | 455 | 13.1 | 0.0e0 | 2.2e-3 | 0.0e0 |
| C-q1 | (1−x²)(1−y²)(x+y²) | 129 | 1e-6 | 908 | 108.7 | 3.6e-8 | 5.6e-4 | 6.4e-5 |
| C-q1 | (1−x²)(1−y²)(x+y²) | 129 | 1e-8 | 1066 | 126.3 | 2.8e-10 | 5.6e-4 | 5.1e-7 |
| C-q1 | (1−x²)(1−y²)(x+y²) | 129 | 1e-10 | 1209 | 144.2 | 1.5e-12 | 5.6e-4 | 2.7e-9 |
| C-q1 | (1−x²)(1−y²)(x+y²) | 129 | 1e-13 | 1395 | 166.5 | 0.0e0 | 5.6e-4 | 0.0e0 |
| C-q1 | (1−x²)(1−y²)(x+y²) | 257 | 1e-6 | 2772 | 1364.9 | 1.5e-8 | 1.4e-4 | 1.1e-4 |
| C-q1 | (1−x²)(1−y²)(x+y²) | 257 | 1e-8 | 3214 | 1579.4 | 1.8e-10 | 1.4e-4 | 1.3e-6 |
| C-q1 | (1−x²)(1−y²)(x+y²) | 257 | 1e-10 | 3622 | 1782.2 | 9.9e-13 | 1.4e-4 | 7.1e-9 |
| C-q1 | (1−x²)(1−y²)(x+y²) | 257 | 1e-13 | 4204 | 2046.8 | 0.0e0 | 1.4e-4 | 0.0e0 |
| C-q1 | (1−x²)(1−y²)e^(x+2y) | 33 | 1e-6 | 105 | 0.7 | 1.3e-7 | 8.4e-3 | 1.5e-5 |
| C-q1 | (1−x²)(1−y²)e^(x+2y) | 33 | 1e-8 | 125 | 0.9 | 8.2e-10 | 8.4e-3 | 9.9e-8 |
| C-q1 | (1−x²)(1−y²)e^(x+2y) | 33 | 1e-10 | 144 | 1.0 | 6.3e-12 | 8.4e-3 | 7.5e-10 |
| C-q1 | (1−x²)(1−y²)e^(x+2y) | 33 | 1e-13 | 169 | 1.2 | 0.0e0 | 8.4e-3 | 0.0e0 |
| C-q1 | (1−x²)(1−y²)e^(x+2y) | 65 | 1e-6 | 316 | 9.2 | 7.1e-8 | 2.1e-3 | 3.4e-5 |
| C-q1 | (1−x²)(1−y²)e^(x+2y) | 65 | 1e-8 | 378 | 11.1 | 5.3e-10 | 2.1e-3 | 2.5e-7 |
| C-q1 | (1−x²)(1−y²)e^(x+2y) | 65 | 1e-10 | 433 | 12.6 | 3.2e-12 | 2.1e-3 | 1.5e-9 |
| C-q1 | (1−x²)(1−y²)e^(x+2y) | 65 | 1e-13 | 516 | 14.9 | 0.0e0 | 2.1e-3 | 0.0e0 |
| C-q1 | (1−x²)(1−y²)e^(x+2y) | 129 | 1e-6 | 987 | 118.1 | 3.1e-8 | 5.2e-4 | 5.9e-5 |
| C-q1 | (1−x²)(1−y²)e^(x+2y) | 129 | 1e-8 | 1166 | 137.3 | 2.8e-10 | 5.2e-4 | 5.3e-7 |
| C-q1 | (1−x²)(1−y²)e^(x+2y) | 129 | 1e-10 | 1331 | 159.7 | 1.9e-12 | 5.2e-4 | 3.6e-9 |
| C-q1 | (1−x²)(1−y²)e^(x+2y) | 129 | 1e-13 | 1579 | 187.3 | 0.0e0 | 5.2e-4 | 0.0e0 |
| C-q1 | (1−x²)(1−y²)e^(x+2y) | 257 | 1e-6 | 2998 | 1461.1 | 1.5e-8 | 1.3e-4 | 1.2e-4 |
| C-q1 | (1−x²)(1−y²)e^(x+2y) | 257 | 1e-8 | 3501 | 1705.3 | 2.0e-10 | 1.3e-4 | 1.6e-6 |
| C-q1 | (1−x²)(1−y²)e^(x+2y) | 257 | 1e-10 | 3998 | 1948.2 | 1.3e-12 | 1.3e-4 | 9.9e-9 |
| C-q1 | (1−x²)(1−y²)e^(x+2y) | 257 | 1e-13 | 4731 | 2305.7 | 0.0e0 | 1.3e-4 | 0.0e0 |

## 6. A against C on GRF samples

Seeds 1000..1004, K(n) = min((n − 1)/2, 64). Cells are mean / max over the samples. u_A is A-fd; u_C at CG tol 1e-13. u_exact is the exact sine-series solution of the truncated forcing.

| n | K | ‖u_A − u_exact‖ L² | ‖u_A − u_fv‖ L² | order | ‖u_A − u_fv‖ max | ‖u_A − u_q1‖ L² | order | ‖u_A − u_q1‖ max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 33 | 16 | 2.5e-7 / 5.2e-7 | 3.24e-3 / 5.99e-3 | NaN | 4.75e-3 / 9.18e-3 | 4.50e-3 / 7.34e-3 | NaN | 6.25e-3 / 1.06e-2 |
| 65 | 32 | 3.3e-10 / 7.4e-10 | 8.09e-4 / 1.50e-3 | 2.00 | 1.31e-3 / 2.79e-3 | 1.15e-3 / 1.89e-3 | 1.97 | 1.72e-3 / 3.11e-3 |
| 129 | 64 | 3.4e-13 / 6.1e-13 | 2.02e-4 / 3.74e-4 | 2.00 | 3.44e-4 / 7.26e-4 | 2.89e-4 / 4.79e-4 | 1.99 | 4.48e-4 / 8.22e-4 |
| 257 | 64 | 1.5e-12 / 1.9e-12 | 5.03e-5 / 9.31e-5 | 2.00 | 8.61e-5 / 1.81e-4 | 7.24e-5 / 1.20e-4 | 2.00 | 1.13e-4 / 2.08e-4 |

## 7. Cost

Setup is the median of 3 runs. Solve and GRF-evaluation times are means over 50 GRF samples (seeds 1000..), forcing evaluated beforehand. A-syl-call recomputes both Schur forms per solve. C assembly is one run.

| n | setup A-fd [ms] | setup A-syl-reuse [ms] | solve A-fd [ms] | solve A-syl-reuse [ms] | solve A-syl-call [ms] | syl-reuse / fd | GRF eval [ms] | assembly C-fv [ms] | assembly C-q1 [ms] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9 | 0.03 | 0.01 | 0.001 | 0.002 | 0.012 | 1.71 | 0.195 | 0.0 | 0.0 |
| 17 | 0.07 | 0.04 | 0.003 | 0.007 | 0.054 | 2.18 | 0.194 | 0.1 | 0.1 |
| 33 | 0.27 | 0.17 | 0.011 | 0.036 | 0.214 | 3.45 | 0.203 | 0.2 | 0.3 |
| 65 | 1.54 | 1.02 | 0.034 | 0.155 | 1.226 | 4.57 | 0.242 | 0.6 | 1.2 |
| 129 | 8.68 | 6.73 | 0.123 | 0.884 | 7.626 | 7.18 | 0.437 | 2.8 | 5.8 |
| 257 | 35.89 | 25.73 | 0.607 | 5.928 | 31.892 | 9.76 | 0.704 | 13.4 | 26.7 |

## 8. Projection: 1200 samples (1000 train + 200 test)

setup + 1200 × (GRF evaluation at K(n) + solve), from the 50-sample means.

| n | route | setup [s] | GRF eval total [s] | solve total [s] | total [s] |
| --- | --- | --- | --- | --- | --- |
| 9 | A-fd | 0.000 | 0.23 | 0.00 | 0.24 |
| 9 | A-syl-reuse | 0.000 | 0.23 | 0.00 | 0.24 |
| 17 | A-fd | 0.000 | 0.23 | 0.00 | 0.24 |
| 17 | A-syl-reuse | 0.000 | 0.23 | 0.01 | 0.24 |
| 33 | A-fd | 0.000 | 0.24 | 0.01 | 0.26 |
| 33 | A-syl-reuse | 0.000 | 0.24 | 0.04 | 0.29 |
| 65 | A-fd | 0.002 | 0.29 | 0.04 | 0.33 |
| 65 | A-syl-reuse | 0.001 | 0.29 | 0.19 | 0.48 |
| 129 | A-fd | 0.009 | 0.52 | 0.15 | 0.68 |
| 129 | A-syl-reuse | 0.007 | 0.52 | 1.06 | 1.59 |
| 257 | A-fd | 0.036 | 0.85 | 0.73 | 1.61 |
| 257 | A-syl-reuse | 0.026 | 0.85 | 7.11 | 7.98 |

## Decisions

- **A's route: A-fd.** A-syl-reuse vs A-fd max relative difference 1.2e-12 (rule ≤ 1e-12: not met); per-solve time at n = 257 A-syl-reuse 5.93 ms vs A-fd 0.61 ms, ratio 9.76 (rule ≤ 2: not met).
- C-fv: orders (L², arriving at n = 65, 129, 257) per solution [2.00, 2.00, 2.00; 2.00, 2.00, 2.00; 2.00, 2.00, 2.00]; qualifies: true; max CG iterations at n = 257 (tol 1e-13): 4752.
- C-q1: orders (L², arriving at n = 65, 129, 257) per solution [2.00, 2.00, 2.00; 2.00, 2.00, 2.00; 2.00, 2.00, 2.00]; qualifies: true; max CG iterations at n = 257 (tol 1e-13): 4731.
- **C's discretisation: C-q1.**
- **C's CG tolerance (C-q1): 1e-6** (loosest of [1e-6, 1e-8, 1e-10] with CG error < 0.1 × discretisation error at n = 257 on all three solutions).
