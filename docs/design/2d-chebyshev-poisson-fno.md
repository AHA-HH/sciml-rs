# Learning the 2D Poisson solution operator on a Chebyshev grid

As of 2026-10-05. Status: **signed off** on 2026-10-05 by the author (AHA-HH). The
decisions are recorded in Section 12. This document is the specification for the phase
plans.

This document designs the first 2D experiment of `sciml-rs` that uses data on a Chebyshev
grid. The experiment learns the solution operator f ↦ u of the Poisson equation on a square
with homogeneous Dirichlet conditions. Reference data come from a solver written in this
repository on RLST, and a Burn FNO is trained on them. `docs/CONVENTIONS.md` wins on any
conflict. This design proposes a new section for it (Section 8) and changes none of its
existing sections.

In one paragraph: forcings f are Gaussian random fields, sampled exactly at
Chebyshev–Gauss–Lobatto nodes from a truncated sine series. Reference solutions u are
computed in f64 by Chebyshev collocation, solved as a dense Sylvester equation with RLST.
A sparse, symmetric second-order discretisation solved by RLST's conjugate gradients serves
as an independent cross-check and as the iterative path towards larger problems. Both fields
are interpolated barycentrically to a uniform grid, and the existing FNO is trained there
unchanged. Predictions are mapped back to the Chebyshev nodes with Floater–Hormann rational
interpolation. Solver, transfer and model errors are reported separately. Resolutions are
staged: 33² for tests, 65² for the first local training run, 129² for the first HPC run,
and 257² for the resolution study. Chebyshev nodes, differentiation matrices and grid
transfers are built in this repository. RLST supplies only its non-FFTW linear algebra and
solvers, behind an optional cargo feature, so the default build and its licence are
unchanged.

## 1. Assessment

### 1.1 Size of today's crate

Measured on `974f8db`. "Code" excludes blank and `//` lines.

| File | Lines | Code | Tests |
| --- | --- | --- | --- |
| `models/fno.rs` | 1562 | 1198 | 36 |
| `layers/spectral_convolution.rs` | 756 | 512 | 19 |
| `training/trainer.rs` | 556 | 402 | 6 |
| `data/transforms/normalizers.rs` | 507 | 371 | 13 |
| `losses/data_losses.rs` | 494 | 349 | 21 |
| `data/grids.rs` | 361 | 266 | 11 |
| `data/loaders/darcy.rs` | 312 | 216 | 6 |
| `data/device_batcher.rs` | 300 | 217 | 6 |
| `data/io/readers/mat.rs` | 226 | 178 | 7 |
| `data/loaders/burgers.rs` | 217 | 155 | 3 |
| other `src/` files | – | – | 20 |
| `tests/` (2 binaries) | 389 | 293 | 5 |
| **Total** | `src/` 6748 | – | 148 unit + 5 integration |

Baseline on this branch: `cargo test` passes all 162 tests (unit, integration and
doctests), and clippy, fmt and doc are clean (verified on `1ee3b69`; later commits changed documentation only).

### 1.2 The 2D pipeline today (Darcy)

**What it does.** `examples/train/darcy.rs` is the 2D entry point. It hardcodes n_train =
1000, n_test = 100 and a subsampling rate of 5 (:32–40). It reads
`datasets/piececonst_r421_N1024_smooth{1,2}.mat` (:42–44). The model is an FNO with modes
[12, 12], width 32, 4 layers and padding `Some(9)` (:57–71). Training runs 500 epochs at
batch 20, with lr 1e-3, weight decay 1e-4 and min_lr 1e-5 (:73–78).

`data/loaders/darcy.rs` reads the fields `coeff` and `sol` (:99–100) with `MatFileReader`,
as `[samples, s, s]`. The raw resolution is pinned to 421 (:37–39, checked at :141). x and
y are normalised pointwise with `UnitGaussianNormalizer`, and `y_test` stays in physical
units (:153–159, CONVENTIONS §10). The loader returns inputs `[n, s, s, 1]` and targets
`[n, s, s]`.

The batcher, `device_loader` and `trainers/darcy.rs` are generic apart from their names.
`trainer.rs` uses Adam with a cosine schedule. It trains and tests on
`LpLoss::rel` with p = 2 and `Reduction::Sum` (:305), so every grid point has equal
weight. Run artifacts go to `runs/<name>_<unix_secs>/`.

**What carries over.** All of the model, training, batching, normalisation and artifact
code. A Poisson loader is the Darcy loader with the field names, the file reader and the
resolution check parameterised.

**What is missing.**
- Anything Chebyshev, non-uniform, interpolating or Poisson. A case-insensitive grep for
  `chebyshev|cheb|non-uniform|interpol|poisson|laplac` over `src/`, `tests/` and
  `examples/` finds nothing.
- Reader dispatch by file format. The `.npz` reader (`data/io/readers/npz.rs`, fields by
  name) works, but no loader calls it.
- Quadrature-weighted errors on a non-uniform grid. `LpLoss::abs` assumes uniform
  spacing (CONVENTIONS §11).
- Non-uniform coordinates in the model. `FNO::grid_cl` always generates a uniform [0, 1]
  grid (CONVENTIONS §2). `data::grids::grid_from_axes` accepts arbitrary axes, but the
  model never sees them.

### 1.3 RLST

Checked on rlst 0.9.0, the latest release on crates.io.
Citations are to its sources.

- **Dense linear algebra.** It always depends on `blas` 0.23 and `lapack` 0.20
  (Cargo.toml:130, 152), but does not choose a provider: the final binary must add one. On
  macOS this is `blas-src`/`lapack-src` with `accelerate`; on Linux, OpenBLAS
  (`src/doc/getting_started.rs:68–87`). A program that uses no BLAS routine links without a
  provider; one that calls LAPACK fails to link (`_dgees_`, `_dtrsyl_` undefined).
- **Solvers for this problem.**
  - `solve_sylvester` solves A X ± X B = scale · C, through Schur forms and `trsyl`
    (`decompositions.rs:268–283`, `dense/linalg/lapack/sylvester.rs:94–99`). Callers
    must check `status()` and `scale()`.
  - LU, Cholesky, symmetric `eigh` and general `eig` are also available.
- **Sparse matrices and iterative solvers.**
  - `CsrMatrix::from_aij` assembles from triplets, summing duplicates
    (`sparse/csr_mat.rs:135–141`).
  - Two Krylov solvers: `CgIteration` (default tol 1e-6 relative, max 1000 iterations;
    `run()` returns the residual, not a converged flag) and `GmresIteration`
    (`operator/algorithms/{conjugate_gradients,gmres}.rs`).
  - There are **no preconditioners**. A custom `OperatorBase` can apply one, matrix-free
    (`traits/abstract_operator.rs:26–53`).
  - A sparse direct solve exists only in the separate, GPL `rlst-suitesparse`.
- **Chebyshev tools.** The `chebychev` module (values ↔ coefficients, derivatives) needs
  the `fftw` feature (`lib.rs:30–31`) and is not used here. The `interpolation` module
  needs no FFTW (`lib.rs:39`). It has `chebychev_points`, **descending** on [−1, 1]
  (`interpolation.rs:90–116`), barycentric weights, and 1D/2D/3D barycentric evaluation.
  This design uses it only as an independent test oracle (Section 11). RLST has no
  differentiation matrices.
- **The `burn` feature** pulls `burn = "0.21"` from crates.io; our fork is
  `0.22.0-pre.4`. Not built, but inferred: the two versions' tensor types are distinct, so
  conversions would not type-check. Data cross the boundary as `Vec`/slices instead.
- **Build.** A program using CSR + CG, `solve_sylvester` and `eigh` on rlst 0.9.0 with
  Accelerate built (cold, 17 s) and ran on the development Mac (arm64). Ubuntu was not
  tried.

## 2. Requirements

| # | Requirement | How the design meets it | Change? |
| --- | --- | --- | --- |
| 1 | Learn f ↦ u for −Δu = f on Ω = [−1, 1]², u = 0 on ∂Ω, with data on a Chebyshev grid | §3, §6 | – |
| 2 | Reference solver in Rust, on RLST | Collocation as a Sylvester / fast-diagonalisation solve on RLST's dense primitives; sparse SPD + `CgIteration` as validation oracle (§4) | new module |
| 3 | No FFTW in the build | Nodes, differentiation matrices and transfers built in this crate; RLST's `fftw` features off (§9) | – |
| 4 | Default build and licence unchanged | RLST and the BLAS provider behind an optional feature `chebyshev` (§9) | `Cargo.toml` |
| 5 | Existing FNO unchanged in the baseline | Interpolate to a uniform grid (§6) | none |
| 6 | Staged resolutions: 33² tests, 65² first local run, 129² first HPC run, 257² resolution study | §5.3 | – |
| 7 | Gaussian random field forcings | §3.3 | – |
| 8 | Solver, transfer and model errors separated | §7 | new evaluation code |
| 9 | Every fast path tested against a trusted slow one | §11 | – |
| 10 | Conventions stated before code relies on them | §8, Phase 0 | `CONVENTIONS.md` |

## 3. Problem

### 3.1 Equation and domain

```math
-\Delta u = f \ \text{in}\ \Omega = [-1, 1]^2, \qquad u = 0 \ \text{on}\ \partial\Omega
```

The square matches the natural interval of Chebyshev polynomials. The model's coordinate
channels on [0, 1] (CONVENTIONS §2) map to it by x = 2ξ − 1.

### 3.2 Chebyshev–Gauss–Lobatto nodes

With n nodes per axis (n = 2^k + 1), stored in **ascending** order:

```math
x_j = -\cos\Big(\frac{\pi j}{n - 1}\Big), \qquad j = 0, \dots, n - 1
```

Ascending order matches the uniform grid's orientation (CONVENTIONS §2), so a field and its
uniform interpolant share index direction. RLST's `chebychev_points` is descending, so the
test oracle reverses it. A field is stored as `F[i, j] = f(x_i, y_j)`: array axis 1 is x and
axis 2 is y, the 'ij' convention of `data::grids`. The boundary nodes j = 0 and j = n − 1
are included and carry u = 0.

### 3.3 Forcing family

Gaussian random fields with covariance (−Δ + τ²)^(−α), as in Li et al.'s Darcy data, are
sampled by their Karhunen–Loève expansion in the Dirichlet eigenfunctions of −Δ on Ω:

```math
f_K(x, y) = \frac{1}{\sqrt{S_K}} \sum_{k, l = 1}^{K} \xi_{kl}\, \lambda_{kl}^{1/2}
\sin\Big(\frac{k\pi(x+1)}{2}\Big) \sin\Big(\frac{l\pi(y+1)}{2}\Big)
```

```math
\lambda_{kl} = \big(\mu_{kl} + \tau^2\big)^{-\alpha}, \quad
\mu_{kl} = \frac{\pi^2}{4}(k^2 + l^2), \quad
S_K = \frac{1}{4}\sum_{k, l = 1}^{K} \lambda_{kl}, \quad \xi_{kl} \sim \mathcal{N}(0, 1)
```

- **Parameters:** τ = 3 and α = 2, Li et al.'s (−Δ + 9I)^−2.
- **Normalisation:** each sine factor has unit mean square on [−1, 1], so the expected
  mean square of the unnormalised sum over Ω is S_K. Dividing by √S_K makes the expected
  RMS of f_K equal to 1 for every K. Changing the truncation therefore does not change the
  amplitude of the samples.
- **Nested draws:** the ξ_kl of a sample are drawn from its seed in a fixed order over the
  largest K in use, and a smaller K keeps the leading block. So f_K at a small K is the
  truncation of the same sample at a larger K, apart from the normalisation factor.
- **Exact evaluation:** the series is evaluated at the nodes directly, with no FFT and
  no interpolation, at a cost of O(K² n²) per sample.
- **Truncation scales with resolution:** K(n) = min((n − 1)/2, 64). That gives K = 16 at
  33², 32 at 65², and 64 at 129² and 257². K = 64 is the maximum production truncation,
  so the 129² and 257² datasets contain the same fields, which is what the resolution
  study needs.
- **Unresolved-mode energy:** the fraction of the full series' expected energy outside
  the truncation, computed by summing λ_kl to convergence (k, l ≤ 3000, plus an integral
  tail):

  | K | 16 | 32 | 64 | 128 |
  | --- | --- | --- | --- | --- |
  | ε_K | 1.9e-2 | 4.9e-3 | 1.3e-3 | 3.2e-4 |

  The documented tolerance is ε_K ≤ 5e-3 for every training and evaluation dataset
  (stages 2–4 of §5.3). Stage 1 (33², K = 16) is for tests and debugging only and is
  exempt; its ε_K is recorded in its sidecar. ε_K measures f; in u the tail is further
  damped by 1/μ_kl. Phase 0 T3 confirms the table and the tolerance.
- **Why sine and not cosine:** with the sine basis f = 0 on ∂Ω. If f is nonzero at a
  corner, u has an r² log r corner singularity, and the convergence of any Chebyshev
  method drops from spectral to algebraic. A cosine (Neumann) KL basis, as Li et al. use,
  would bring that singularity into every sample.

### 3.4 Norms and quadrature

Errors on the Chebyshev grid use Clenshaw–Curtis weights w_i (exact for polynomials of
degree n − 1), in tensor product:

```math
\lVert v \rVert_{L^2(\Omega)}^2 \approx \sum_{i, j} w_i\, w_j\, v(x_i, y_j)^2
```

Errors on the uniform grid use `LpLoss` (CONVENTIONS §11). Training keeps `LpLoss::rel`
with equal weights, as today.

## 4. Reference solver

### 4.1 Options

| Option | Matrix | RLST routine | Accuracy | Cost (n per axis) |
| --- | --- | --- | --- | --- |
| A. Chebyshev collocation, interior equations as a Sylvester equation | dense, non-symmetric, κ = O(n⁴) | `solve_sylvester` | spectral | O(n³) |
| B. Collocation as one Kronecker-sum system | dense (n − 2)² square | LU `solve` | spectral | O(n⁶): only n ≤ 65 |
| C. Second-order finite volume or Q1 FEM on the CGL tensor mesh | sparse, **SPD**, 5 or 9 points | `CsrMatrix` + `CgIteration` | O(h_max²) | O(n² · iterations); no preconditioner, so about n iterations or more |
| D. Collocation solved by GMRES, preconditioned by C (Orszag 1980) | dense apply + sparse preconditioner | `GmresIteration` + custom `OperatorBase` | spectral | needs an inner solve with C at every iteration |
| E. Legendre–Galerkin (Shen 1994) | sparse banded, SPD | `CgIteration` | spectral | needs Legendre ↔ CGL transforms, not available here |

On the interior nodes, option A is the separable equation

```math
-D_{xx}\, U - U\, D_{yy}^{\mathsf T} = F
```

with D_xx and D_yy the interior blocks of the Chebyshev second-derivative matrices along x
and y, and U, F the interior values. They are equal when n_x = n_y, but kept separate so
that rectangular grids work. Boundary values are zero, so they drop out.

### 4.2 Decision

- **Authoritative training labels: A.** Spectral accuracy, f64, O(n³).
  - The equation is solved as a Sylvester equation, or equivalently by fast
    diagonalisation, built on RLST's dense primitives.
  - The design does not assume that RLST provides a public Sylvester routine.
    `solve_sylvester` was found and run in rlst 0.9.0 (§1.3), but Phase 0 T2 re-verifies
    it against the pinned version's public API.
  - Whatever RLST provides, the problem-specific wrapper lives in this crate,
    `pde::poisson::collocation`. It uses `solve_sylvester` if it is public, and otherwise
    fast diagonalisation from RLST's `eig` and dense matrix products.
  - The factorisation of D_xx and D_yy is computed once and reused for every sample. At
    257² this is what keeps thousands of solves cheap.
- **Independent validation oracle and future iterative path: C.**
  - A sparse SPD finite-volume or Q1 finite-element discretisation on the CGL tensor mesh,
    solved with RLST's CG. It runs on selected cases (the manufactured solutions and a
    sample of the GRF forcings), not on every sample.
  - The two solutions are compared on a common grid: C's solution lives on the CGL nodes
    themselves, so no interpolation is needed. The difference must decrease at C's second
    order as n grows.
  - C never produces training labels. It is also the path that extends to 3D and
    distributed grids, where dense solves stop scaling.
- **Not now: B** (only as a test oracle at n ≤ 33), **D** (only if A's cost becomes a
  problem), and **E** (needs transforms we do not have).

### 4.3 Verification

- **Manufactured solutions.** u = sin(πx) sin(πy), with f = 2π² u; the polynomial
  u = (1 − x²)(1 − y²)(x + y²); and a non-symmetric one, u = (1 − x²)(1 − y²) e^(x + 2y).
- **Option A** reaches its f64 floor by n = 33 on the first two. A convergence study
  over n = 9..129 must show spectral decay down to a stated floor.
- **Option C** must show order 2 on the same functions.
- **A against B** agree to round-off at n ≤ 33.
- **D^(2)** is checked against Trefethen's `cheb` formulas, and on polynomials, where it
  is exact.

## 5. Datasets

### 5.1 Generation

A new example, `generate_poisson` (feature `chebyshev`), writes for each resolution:

- one `.npz` per split, with fields `f` and `u`, `[N, n, n]`, f64;
- fields `x` and `y` holding the nodes;
- a JSON sidecar recording (K, τ, α), the seeds, n and the git commit.

Sample i of a split uses seed `base_seed + i`, and the train and test bases differ.
Reading uses the existing `.npz` reader. Writing uses `ndarray-npy` (already a
dependency).

### 5.2 Loader

A Poisson loader follows `loaders/darcy.rs`: inputs `[N, s, s, 1]`, targets `[N, s, s]`,
pointwise `UnitGaussianNormalizer` on x and y fitted on the training split, `y_test` left
raw (CONVENTIONS §10). It reads `.npz` and applies the Chebyshev → uniform transfer
(§6.2) at load time, in f64, before normalisation. Darcy's loader stays as it is.

### 5.3 Resolutions and sizes

| Stage | Chebyshev n | Uniform s (model grid) | Purpose | Where |
| --- | --- | --- | --- | --- |
| 1 | 33 | 32 | unit tests, manufactured solutions, debugging | local |
| 2 | 65 | 64 | first end-to-end FNO training run | local (flex, metal) |
| 3 | 129 | 128 | first production run: accuracy and performance | HPC, GPU |
| 4 | 257 | 256 | resolution generalisation and scaling, after profiling stage 3 | HPC, GPU |

Sizes: 1000 train / 200 test per resolution (as Li et al.), from the same seeds at every
resolution.

**Note on the uniform sizes.** The FFT in this crate accepts any length (Bluestein,
CONVENTIONS §4), and padding already makes the extent non-power-of-two. A uniform grid of
n points (33, 65, …) with both endpoints has spacing 2/(n − 1), so its points are every
second node of a power-of-two dyadic grid. Using it would avoid one interpolation
mismatch. The table keeps the stated sizes; Phase 0 T3 also measures s = n, and the
sizes switch only if s = n is clearly better (decision 5).

## 6. Chebyshev data and the FNO

### 6.1 Options

| Option | Model change | Convention change | Transfer error | Notes |
| --- | --- | --- | --- | --- |
| 1. Interpolate to a uniform grid, unchanged FNO, interpolate back | none | new §12 only | yes, both ways | baseline; Geo-FNO reports learned maps about 2× better than interpolation (Li et al. 2023) |
| 2. θ-map: x = cos θ, so CGL nodes are uniform in θ; even extension then FFT (a DCT-I) | grid channels in θ; extension and crop around the spectral layers | §2, §5 (version bump) | none | 4× the points in 2D; evenness of the output only approximate |
| 3. Chebyshev/DCT spectral layer (SNO, Fanaskov & Oseledets; OPNO, Liu et al. 2022) | new spectral layer, new transform | §4, §5, §6 (version bump) | none | SNO reports 2.2e-2 vs FNO 2.7e-2 on a 2D elliptic problem |
| 4. Mesh-free operators (DeepONet, graph kernels) | new model | new | none | out of scope |

### 6.2 Recommendation and transfers

**Option 1 for every phase up to the HPC runs.** It needs no model change and no change to
CONVENTIONS §1–§6. Its errors are measured separately (§7).

- **Chebyshev → uniform:** barycentric polynomial interpolation, tensor-product, as one
  dense 1D matrix applied along each axis. It is forward stable on Chebyshev nodes
  (Higham 2004; Berrut & Trefethen 2004) and spectrally accurate for smooth fields.
- **Uniform → Chebyshev:** Floater–Hormann rational interpolation of degree d. Polynomial
  interpolation from uniform nodes is ill-conditioned (the Runge phenomenon); FH has no
  real poles and a logarithmically growing Lebesgue constant. d is chosen in Phase 0 from
  measured round-trip errors.
- Both transfers are fixed matrices per (n, s), built once, in f64.

**Option 2 as an optional later comparison (Phase 5).** It does not block completion and
needs its own convention diff.

Decision 2.

## 7. Training and evaluation

- Training on the uniform grid as today: `LpLoss::rel` (p = 2), Adam with cosine
  schedule, FNO modes [12, 12], width 32, 4 layers. Padding `Some(p)` because the problem
  is not periodic; p is chosen in Phase 3 (Darcy uses 9 at s = 85).
- **Errors reported**, each as a relative L² error over the test set (mean and maximum):
  1. **solver:** A against manufactured solutions (§4.3), and A against C as h → 0;
  2. **transfer:** T_uc(T_cu u) − u on the Chebyshev grid, for the test u;
  3. **model:** prediction against T_cu u on the uniform grid (the training metric);
  4. **total:** T_uc(prediction) against u on the Chebyshev grid, with Clenshaw–Curtis
     weights (§3.4).
- **Expectation, to calibrate tests and not as a target:** FNO relative L² errors around
  1e-2 on smooth elliptic problems (Li et al. 2021, Darcy 0.0108 at 85²). Transfer errors
  should sit well below that; Phase 0 confirms it.
- **Resolution study (stage 4):** train at 65 or 129, evaluate at 129 and 257 on the same
  seeds.

## 8. Conventions diff

Append to `docs/CONVENTIONS.md`:

```markdown
## 12. Chebyshev grids and transfers

- Domain Ω = [−1, 1]². Model coordinates ξ ∈ [0, 1] (§2) map to it by x = 2ξ − 1.
- Chebyshev–Gauss–Lobatto nodes, n per axis, ascending: x_j = −cos(πj/(n − 1)),
  j = 0..n − 1, endpoints included.
- Fields are stored `F[i, j] = f(x_i, y_j)`: axis 1 is x, axis 2 is y.
- Chebyshev → uniform transfer: tensor-product barycentric interpolation. Uniform →
  Chebyshev: tensor-product Floater–Hormann of degree d (recorded here after Phase 0).
- Chebyshev-grid norms use tensor-product Clenshaw–Curtis weights.
- Reference data are f64 on the host; the model sees f32 (§8).
```

And extend §9: "§12 fixes how dataset files and transfers are laid out; a change to it bumps
`CONVENTION_VERSION`." The new section adds to and changes nothing in §1–§6, so no saved
checkpoint changes. Recommendation: add §12 at version 1 and bump only on a later change to
it. Decision 4.

## 9. Dependencies and platform

- **Optional feature `chebyshev`.**
  ```toml
  chebyshev = ["dep:rlst", "dep:blas-src", "dep:lapack-src"]
  rlst = { version = "0.9", optional = true, default-features = false }
  ```
  `blas-src`/`lapack-src` use `accelerate` on macOS and `openblas` (system) on Linux,
  selected by target. Without the feature, the crate builds exactly as today.
- **Not enabled:** RLST's `fftw*`, `burn` and `mpi` features, and `rlst-suitesparse`
  (GPL).
- **CI:**
  - the existing job is unchanged;
  - a new job runs `cargo clippy --features chebyshev --all-targets` and
    `cargo test --features chebyshev` on Ubuntu, after
    `apt-get install libopenblas-dev`. The workflow already carries that step,
    commented out.
- **Precision:** the solver, transfers and dataset files are f64 on the host. The model is
  f32 (CONVENTIONS §8).
- **Backends:** the data side is CPU only. Training must hold on flex; metal locally and
  cuda on HPC are run and reported.
- **3D carry-over (noted only):** nodes, differentiation matrices, transfers and
  quadrature are 1D operators applied per axis. The Sylvester form does not extend to 3D
  directly (3D needs fast diagonalisation or option C/D), and option C does extend.

## 10. Phases and tasks

### Phase 0: conventions and feasibility

| Task | Delivers | Modules |
| --- | --- | --- |
| T1 | CONVENTIONS §12 (Section 8), signed off | `docs/CONVENTIONS.md` |
| T2 | Feature `chebyshev`, rlst 0.9 dependency, BLAS selection, CI job; verify which dense primitives (Sylvester, `eig`, LU) and CG are public in the pinned version; a smoke test running CG and the Sylvester or eigen route | `Cargo.toml`, `.github/workflows/`, `tests/` |
| T3 | Transfer spike: barycentric and FH round-trip errors for n ∈ {33, 65, 129, 257} on smooth and GRF fields; choose d; compare s = n − 1 with s = n; confirm the GRF tail table and tolerance (§3.3) | spike, report |
| T4 | Solver spike: A and C at n = 33..257, error and time; factorisation reuse | spike, report |

### Phase 1: verified reference solver

| Task | Delivers |
| --- | --- |
| T1 | `chebyshev::{nodes, diff_matrix, clenshaw_curtis}` and their tests |
| T2 | `poisson::collocation` (option A) and the manufactured-solution convergence tests |
| T3 | `poisson::sparse` (option C) with the CG solve and the order-2 test; the A-vs-C check |

### Phase 2: datasets and transfers

| Task | Delivers |
| --- | --- |
| T1 | `chebyshev::transfer` (barycentric, FH) with oracle tests |
| T2 | GRF sampler (§3.3) and the `generate_poisson` example; `.npz` + JSON output |
| T3 | Poisson loader (§5.2) |

### Phase 3: training and evaluation, local

| Task | Delivers |
| --- | --- |
| T1 | `train_poisson` example and trainer; first run at 65² |
| T2 | `predict_poisson`: the four errors of §7 on the Chebyshev grid |

### Phase 4: HPC

| Task | Delivers |
| --- | --- |
| T1 | Production runs at 129² on GPU |
| T2 | Resolution study at 257² after profiling |

### Phase 5 (optional, non-blocking)

θ-map comparison (option 2), with its own convention diff.

Module names are proposals. New code lives under `src/neural_operators/chebyshev/` and
`src/neural_operators/pde/poisson/` (feature-gated where it uses RLST), and
`src/neural_operators/data/loaders/poisson.rs`.

## 11. Tests

| Component | Trusted reference | Tolerance (f64 unless noted) |
| --- | --- | --- |
| CGL nodes, barycentric weights | RLST `interpolation` (reversed), closed forms | 1e-14 absolute |
| D, D² | Trefethen's `cheb`; exact on polynomials of degree < n | 1e-10 relative to ‖D‖ |
| Clenshaw–Curtis weights | exact integrals of polynomials of degree ≤ n − 1 | 1e-14 |
| Collocation solver (A) | manufactured solutions; option B at n ≤ 33 | stated floor from the convergence study |
| Sparse solver (C) | manufactured solutions: order 2 ± 0.1; residual ≤ CG tolerance | – |
| Chebyshev → uniform | RLST barycentric evaluation; analytic functions | 1e-12 relative |
| Uniform → Chebyshev (FH) | analytic functions; the measured rate for degree d | from Phase 0 |
| GRF sampler | same seed gives the same field at every n (nodes in common); empirical covariance against the formula | statistical, stated in the brief |
| Loader | shapes, normaliser asymmetry as in Darcy's tests | exact |
| Pipeline | 65² training converges below a threshold set from Phase 3 T1 | f32 |

## 12. Questions for sign-off

| # | Question | Recommendation |
| --- | --- | --- |
| 1 | Reference solver | Collocation solved as a Sylvester equation (option A) for the data; sparse SPD finite volume or Q1 with RLST CG (option C) as the cross-check and the iterative path (§4.2) |
| 2 | How Chebyshev data meet the FNO | Interpolate to uniform with the unchanged FNO (option 1); θ-map as an optional Phase 5 (§6) |
| 3 | GRF basis | Sine (Dirichlet) KL basis, so f = 0 on ∂Ω and no corner singularity; τ = 3, α = 2, K = 64 (§3.3) |
| 4 | Conventions | Add §12 at version 1; changes to it bump from then on (§8) |
| 5 | Uniform grid sizes | Keep 32/64/128/256 as stated, and have Phase 0 T3 also measure s = n (33/65/…); switch if it is clearly better (§5.3) |
| 6 | RLST and BLAS | rlst 0.9 behind the optional feature `chebyshev`, `default-features = false`; Accelerate on macOS, OpenBLAS on Linux; no FFTW, burn, MPI or SuiteSparse |
| 7 | FFTW later | Only as an optional, non-default backend, after supervisor approval, confirmed licensing and a measured benefit |

Further choices made in this design. They were not asked individually and stand unless
overruled later:
- Nodes stored ascending, fields 'ij' (§3.2).
- Training loss unchanged (equal weights on the uniform grid); Clenshaw–Curtis only for
  evaluation (§3.4).
- 1000 train / 200 test per resolution, the same seeds across resolutions (§5.3).
- Transfers applied in the loader at load time, not stored in the dataset files (§5.2).
- A separate Poisson loader instead of generalising the Darcy one (§5.2).

### Sign-off procedure

The author signs off alone. The questions above are answered and recorded here, the
affected sections are updated, and the status line is set to "signed off". After that, the
CONVENTIONS §12 change (Phase 0 T1) and the Phase 0 plan can start.

### Recorded decisions

Signed off by the author (AHA-HH) on 2026-10-05, in a Claude Code session.

1. **Reference solver: changed.** Authoritative labels come from Chebyshev collocation,
   solved through the separable form −D_xx U − U D_yyᵀ = F as a Sylvester or
   fast-diagonalisation solve on RLST's dense primitives.
   - No public Sylvester routine is assumed. Phase 0 T2 verifies the pinned version, and
     the problem-specific wrapper lives downstream in this crate.
   - The sparse SPD finite-volume or Q1 system with RLST's CG solves selected cases
     independently, is compared on the common CGL grid, and serves as the validation
     oracle and future scalable path. It never supplies training labels (§4.2).
2. **Chebyshev data and the FNO: accepted.** Interpolate to uniform, FNO unchanged;
   θ-map as optional Phase 5 (§6).
3. **GRF: changed.**
   - Homogeneous-Dirichlet sine basis with τ = 3 and α = 2.
   - K = 64 is the maximum production truncation at 129². K scales with resolution,
     K(n) = min((n − 1)/2, 64).
   - Unresolved-mode energy is verified to be at most 5e-3 for stages 2–4.
   - Samples are normalised to unit expected RMS, so their amplitude does not change with
     K (§3.3).
4. **Conventions: accepted.** §12 is added at version 1 (§8).
5. **Uniform sizes: accepted.** 32/64/128/256; Phase 0 T3 also measures s = n (§5.3).
6. **RLST: accepted.** Optional feature `chebyshev`, rlst 0.9 (latest), non-FFTW only
   (§9).
7. **FFTW: accepted.** Possibly later, as an optional, non-default backend, subject to
   supervisor approval, confirmed licensing and a measured performance benefit.

## References

- Z. Li et al., "Fourier Neural Operator for Parametric Partial Differential Equations",
  ICLR 2021, arXiv:2010.08895.
- Z. Li, D. Z. Huang, B. Liu, A. Anandkumar, "Fourier Neural Operator with Learned
  Deformations for PDEs on General Geometries", JMLR 24 (2023), arXiv:2207.05209.
- V. Fanaskov, I. Oseledets, "Spectral Neural Operators", arXiv:2205.10573.
- X. Liu et al., "Spectral Operator Learning / OPNO", arXiv:2206.12698.
- L. N. Trefethen, *Spectral Methods in MATLAB*, SIAM, 2000, ch. 6–8.
- D. B. Haidvogel, T. Zang, J. Comput. Phys. 30(2), 1979, doi:10.1016/0021-9991(79)90097-6.
- S. A. Orszag, J. Comput. Phys. 37(1), 1980, doi:10.1016/0021-9991(80)90005-4.
- J. Shen, SIAM J. Sci. Comput. 15(6), 1994, doi:10.1137/0915089.
- N. J. Higham, IMA J. Numer. Anal. 24(4), 2004, doi:10.1093/imanum/24.4.547.
- J.-P. Berrut, L. N. Trefethen, SIAM Rev. 46(3), 2004, doi:10.1137/S0036144502417715.
- M. S. Floater, K. Hormann, Numer. Math. 107, 2007, doi:10.1007/s00211-007-0093-y.
- rlst 0.9.0, codeberg.org/linalg-rs/rlst.
