# Learning the 2D Poisson solution operator on a Chebyshev grid with the Burn FNO

As of 2026-10-05. Draft for review; nothing here is signed off.
2026-10-05: numerics review folded in (§2.1, §2.2, §2.4, §2.5 H4, §2.6, §2.8, §3.4, §7, §8.1, §8.2): canonical sin-form nodes, an asymmetric L1 field, the H4 normaliser procedure, and measured tolerances.

> Where this document and `docs/CONVENTIONS.md` differ, the conventions file takes
> precedence.

**Recommendation in one line:**
- Generate forcings and reference solutions on a tensor-product Chebyshev–Gauss–Lobatto grid with a
  Chebyshev spectral collocation solver, built on RLST behind an optional feature.
- Interpolate both fields to a uniform grid, and train the *existing, unchanged* FNO there.
- Map predictions back to the Chebyshev grid with stable Floater–Hormann rational interpolation.
- Report the solver, transfer and model errors separately, on both grids.

| Question | Recommendation | Section |
|---|---|---|
| Domain and boundary condition | Ω = [−1,1]², homogeneous Dirichlet | §2.1 |
| Nodes | Chebyshev–Gauss–Lobatto, n = 2^k + 1, stored ascending | §2.2 |
| Reference solver | Collocation with fast diagonalisation, f64 | §2.4 |
| What the FNO sees | Uniform s × s grid with both endpoints; FNO unchanged | §4 |
| Back to Chebyshev | Floater–Hormann, degree d from a Phase 0 spike | §2.6, §4 |
| RLST | Optional cargo feature `chebyshev-data`, no `fftw`, no `burn` feature | §6 |
| θ-space (cosine map) FNO | Optional Phase 5 comparison; does not block completion | §3.4, §7 |

## 1. Purpose, scope and assumptions

### 1.1 Goal
The first 2D integration test of the group's stack:
- RLST, for the Chebyshev machinery;
- sciml-rs, for the Burn FNO;
- later ND, for distributed grids in 3D.

The problem is learning the solution operator f ↦ u of the Poisson equation with fixed boundary
conditions, for data that lives on a tensor-product Chebyshev grid. It is a correct, reproducible 2D
workflow that a later 3D experiment can extend (§7 Phase 4). It is not a performance or accuracy
claim.

### 1.2 Definition of done
1. **Reference solver.** A solver that is verified by manufactured solutions and a convergence study
   (§8.1 L1–L5).
2. **Transfer operators.** Chebyshev ↔ uniform transfers with measured accuracy, including a round
   trip on reference solutions (§8.1 L6).
3. **Datasets.**
   - A seeded generator writes training, validation, test and held-out sets.
   - Each set carries metadata that is enough to regenerate it bit for bit on the same platform.
4. **Training.**
   - `train_poisson` trains the FNO.
   - An overfit test proves the pipeline is wired correctly.
5. **Evaluation.** `predict_poisson` reports:
   - held-out accuracy on the uniform grid and on the Chebyshev grid;
   - the boundary error;
   - a residual diagnostic;
   - three baselines.
   The acceptance thresholds are calibrated by a pilot run and fixed before the final run (§8.2).
6. **Record and 3D readiness.**
   - Results are recorded in a §11 Outcome section.
   - A 3D readiness note is written.

### 1.3 In scope
- Domain, boundary conditions, forcing family and held-out sets (§2).
- A Chebyshev spectral reference solver in f64.
- Chebyshev → uniform transfer using RLST barycentric interpolation.
- Uniform → Chebyshev transfer using Floater–Hormann rational interpolation.
- Clenshaw–Curtis quadrature for physical L² norms.
- A dataset generator and loader.
- Training and evaluation examples, baselines, and the validation ladder.
- An optional θ-space comparison (Phase 5).

### 1.4 Out of scope
- 3D implementation; only the interfaces are designed to carry over (§5.5).
- MPI and distributed grids.
- Navier–Stokes or any other PDE.
- Feature parity with PyTorch `neuraloperator` (objective `pytorch-parity`, worktree `goldeye`).
- ND finite elements.
- Any change to `FNO` or `SpectralConv`.
- A Chebyshev-basis spectral layer (§3.4, recorded only).
- GPU-specific tuning.

### 1.5 Assumptions
- **A1.** The meeting transcripts leave the exact problem unspecified. The choices in §2 are
  recommendations, and §9.2 lists them for sign-off.
- **A2.** "RST" in the transcript means RLST (codeberg.org/rlst/rlst); the user confirmed this on
  2026-10-05.
- **A3.** The transcript's "stick to power of 2s" for the FFT is a performance preference, not a
  constraint. §6.2 shows any length works; Phase 0 T4 measures the cost.
- **A4.** Training happens on the default `flex` backend. Metal is run where noted. Model tolerances
  are f32, and solver and transfer tolerances are f64.

## 2. Foundations

### 2.1 The problem
Find u on Ω = [−1,1]² such that

```math
-\Delta u = f \quad \text{in } \Omega, \qquad u = 0 \quad \text{on } \partial\Omega .
```

The learning target is the linear solution operator 𝒢 : f ↦ u. Each sample is a different forcing
f drawn from a distribution (§2.5), so the dataset is a family of PDE problems, not repeated samples
of one solution.

**Corner compatibility.** At a corner both edges carry u = 0, so u_xx = u_yy = 0 there. A classical
solution therefore needs f(corner) = 0. If f(corner) ≠ 0, u has a weak r² log r singularity at the
corner, and Chebyshev convergence becomes algebraic instead of exponential (Trefethen 2000, ch. 7;
cited from memory, unverified).

**Zeroing f at the corners does not restore exponential convergence.**
- It removes only the **leading r² log r term**.
- If f has an x² − y² component at a corner (f_xx ≠ f_yy there), an r⁴ log r term remains.
- Convergence is therefore algebraic but fast.

The numerics review on 2026-10-05 measured the CC-relative self-convergence over 10 samples of §2.5
(b):

| Forcing | n = 33 vs 65 | n = 65 vs 129 |
|---|---|---|
| corner-corrected | 6.3e-10 | 4.9e-13 |
| uncorrected | 4.2e-8 | 6.5e-10 |
| f ≡ 1 | — | 1.0e-10 (at n = 65) |

The training family (§2.5) keeps the corner correction because it gains about two orders of
magnitude for free. f ≡ 1 is kept as a labelled hard held-out case.

### 2.2 Chebyshev–Gauss–Lobatto grid
Per axis, take n = 2^k + 1 nodes in **ascending** order:

```math
x_j = \sin\!\left(\frac{\pi\,(2j-(n-1))}{2(n-1)}\right) \;=\; -\cos\!\left(\frac{\pi j}{n-1}\right), \qquad j = 0,\dots,n-1 .
```

**The sin form is canonical.**
- It is computed as `(PI * (2j − (n−1)) as f64 / (2(n−1)) as f64).sin()`.
- With that order of operations, the array is exactly antisymmetric, x_mid = 0 exactly, and the
  endpoints are exactly ±1.
- The nodes nest bit for bit: the n = 65 nodes equal the even-indexed n = 129 nodes, because the
  arguments differ by exact powers of two.

RLST's `chebychev_points(Kind::Second, n)` returns cos(πj/(n−1)), the same set in **descending**
order (`~/Code/rlst/rlst/src/interpolation.rs:95-118`). Reversing it gives the same values only up to
about 3e-16 absolute: the middle node comes out as +6.1e-17, not 0 (numerics review 2026-10-05).
So the RLST nodes are **not** used. Our `gl_nodes` computes the sin form, and RLST supplies only the
barycentric evaluation.

RLST's docstring at `interpolation.rs:90-94` says the opposite of what the code does: it says the
second kind excludes the endpoints and that the function returns n + 1 points. Our tests pin the
behaviour we rely on.

A 2D field is an array `F[i, j] = f(x_i, y_j)` of shape `[n, n]`. Axis 0 is x and axis 1 is y,
i.e. 'ij' indexing, which matches `data::grids::grid_from_axes` (`src/neural_operators/data/grids.rs:26`).

### 2.3 Uniform model grid
The model grid has s points per axis, including both endpoints:

```math
\xi_k = \frac{k}{s-1}, \quad x_k = 2\xi_k - 1, \qquad k = 0,\dots,s-1 .
```

ξ is exactly the coordinate the FNO generates for itself, `arange(s)/(s−1)` (CONVENTIONS §2,
`src/neural_operators/models/fno.rs:197`). The physical grid and the model's grid channels therefore
agree **with no convention change**.

Batches are `[B, s, s, 1]`, channels-last (CONVENTIONS §1), with tensor axis 1 ↔ x and axis 2 ↔ y.
The grid channels the model appends are `[f, ξ(y), ξ(x)]` (reverse axis order, CONVENTIONS §2).
This is a fixed fact, not a choice, and a Phase 1 test pins it with an asymmetric field.

### 2.4 Reference solver: Chebyshev collocation with fast diagonalisation
Let D be the n × n Chebyshev differentiation matrix on the ascending nodes (Trefethen 2000, `cheb`,
with the node order flipped), and let D₂ = D². Restrict D₂ to the interior indices
I = {1, …, n−2} to get A = D₂[I, I], of size m × m with m = n − 2.

With U[i, j] ≈ u(x_i, y_j) on the interior, the collocation equations are

```math
-(A U + U A^{\top}) = F_I .
```

A is **not symmetric**, so `eigh` does not apply. Its eigenvalues are real, negative and distinct
(Gottlieb & Lustman 1983; cited from memory, unverified). The numerics review measured them for
n ∈ {5, …, 257}:
- |Im λ| = 0 exactly;
- the minimum relative gap is 9.4e-5, at n = 129;
- the largest λ ≈ −π²/4;
- with LAPACK's column normalisation, cond(V) is 1.9 at n = 33, 2.3 at n = 65 and 2.7 at n = 129.

Take A = V Λ V⁻¹ once, from RLST `eig` (LAPACK `dgeev`, complex output type). Check that
|Im| ≤ 1e-12·|Re| and keep the **real parts of both Λ and V**. For real eigenvalues the real
`dgeev` path returns eigenvectors whose imaginary part is exactly zero. RLST's conjugate-pair
detection uses `w[k] == conj(w[k+1])`, which could misfire on a repeated real eigenvalue. That is
harmless here, because the eigenvalues are distinct, and the solver asserts it.

Then every solve is

```math
\tilde F = V^{-1} F_I V^{-\top}, \qquad \tilde U_{ij} = -\frac{\tilde F_{ij}}{\lambda_i + \lambda_j}, \qquad U = V \tilde U V^{\top},
```

which is fast diagonalisation (Lynch, Rice & Thomas 1964). λ_i + λ_j < 0, so there is no division
by zero. The boundary rows and columns of the full n × n solution are set to exactly 0.

- **Cost.** O(m³) once, then four m × m matrix products per solve. Microseconds at n = 65.
- **Trusted slow oracle (tests only).** Dense LU of the Kronecker sum −(I ⊗ A + A ⊗ I) acting on
  vec(U), for n ≤ 33 (961 unknowns).
- **Precision.** Everything is computed in f64. V is non-orthogonal, but well conditioned at these
  sizes (above). Fast diagonalisation matched Kronecker LU to ≤ 1.6e-14 for n ≤ 33 in the review,
  so L3 is fixed at 1e-12 (§8.2).

### 2.5 Forcing family, manufactured solutions and held-out sets
These are three distinct uses of the solver.

**(a) Manufactured solutions: solver verification only, never training data.**
- M1: u = (1−x²)(1−y²), so −Δu = 2(1−x²) + 2(1−y²). The solution is a degree-4 polynomial, so the
  solve is exact to rounding for n ≥ 5.
- M2: u = sin(πx) sin(πy), so −Δu = 2π² u. Analytic, exponential convergence.
- M3: u = (1−x²)(1−y²) e^{x+y/2}, so

  ```math
  -\Delta u = -\Big[(-1-4x-x^2)(1-y^2) + \big(-2-2y+\tfrac{1-y^2}{4}\big)(1-x^2)\Big] e^{x+y/2},
  ```

  which is not symmetric in x and y. That catches swapped axes. Phase 1 must re-derive this formula
  independently, symbolically or by finite differences, before using it as an oracle.

**(b) Training family.** f is a random tensor Chebyshev series with decaying coefficients, then
corrected at the corners:

```math
g(x,y) = \sum_{p=0}^{K-1}\sum_{q=0}^{K-1} a_{pq}\, T_p(x)\, T_q(y), \qquad a_{pq} \sim \mathcal N\!\big(0,\; \sigma_{pq}^2\big), \quad \sigma_{pq} = \big(1 + (p^2+q^2)/K_0^2\big)^{-\alpha/2} \ \text{(std; i.i.d.)},
```

```math
f = c\,\big(g - \Pi_1 g\big),
```

where Π₁g is the bilinear interpolant of g's four corner values, so f vanishes at the corners, and c
scales the set so that the sample RMS of f over the training set is 1.
- Provisional values: **K = 16, K₀ = 4, α = 2.5**. They are set by the Phase 2 pilot; see §9.2 Q3.
- f is a polynomial of degree ≤ K − 1 in each variable, so it is represented exactly on any grid
  with n ≥ K.
- T_p(x_j) = cos(p·arccos x_j), evaluated directly.
- This is one reading of the transcript's "f is just [random in the] Chebyshev [coefficients]";
  §9.2 Q2 asks for confirmation.

**(c) Held-out sets**, each generated with its own seed:

| Set | What changes | Purpose |
|---|---|---|
| H0 `test` | nothing: same distribution | in-distribution generalisation |
| H1 `rough` | α = 1.5, K = 24 | more high-frequency content |
| H2 `bumps` | f = sum of 1–4 Gaussian bumps, centres in [−0.7, 0.7]², widths 0.1–0.3, corner-corrected | out of family; not in the span of (b) |
| H3 `const` | f ≡ 1, with no corner correction | corner singularity; reference converges algebraically |
| H4 `fine` | test distribution, s = 128 model grid | zero-shot change of model resolution (procedure below) |

**H4 procedure.** `UnitGaussianNormalizer` statistics are per grid point, with the shape of one
sample, [64, 64] (`data/transforms/normalizers.rs:33-57, 101-106`). They cannot encode [N, 128, 128]
inputs. So for H4:
1. Map the *training* Chebyshev data to s = 128 with T_cu. This is training data only, so nothing
   leaks.
2. Refit the x and y normalisers at 128, and use them to encode H4 and to decode its predictions.
3. Rebuild the model with padding 2p (CONVENTIONS §3 pads a fixed number of cells). This keeps the
   padded fraction of the domain the same, and changes no parameter shape.

H4 therefore measures only the resolution change. Both choices are recorded in the evaluation output.

### 2.6 Grid transfers
- **Chebyshev → uniform, T_cu.** Tensor barycentric interpolation with Chebyshev weights:
  - RLST `tensor_barycentric_evaluate_2d` (`interpolation.rs:285`);
  - weights from `barycentric_chebychev_weights(Kind::Second, n)` (`interpolation.rs:152`).
  - It is the polynomial interpolant on Chebyshev points: stable, with a Lebesgue constant of
    O(log n), and exponentially accurate for analytic data.
  - Points that coincide with a node are handled exactly (`interpolation.rs:246`).
    Uniform endpoints ±1 coincide with x₀ and x_{n−1} **bit for bit** only if x_k is computed as
    `2.0 * (k as f64 / (s − 1) as f64) − 1.0`, not by accumulating a linspace step. This is
    required, because exact boundary transfer and the §4.2 normaliser argument depend on it.
  - Weight ordering is not a risk. For odd n the second-kind weights are their own reverse, and
    for even n reversal flips only a global sign, which the barycentric quotient cancels (review,
    2026-10-05).
  - The risk that matters is nodes and values ordered inconsistently. The asymmetric L1 field
    catches it.
- **Uniform → Chebyshev, T_uc.** Polynomial interpolation of high degree from equispaced samples is
  exponentially ill-conditioned. No method can be both fast-converging and stable from equispaced
  data (Platte, Trefethen & Kuijlaars 2011; cited from memory, unverified).
  - We use **Floater–Hormann (FH) rational interpolation** of blending degree d (Floater & Hormann
    2007). On s equispaced nodes its barycentric weights are

    ```math
    w_k = (-1)^{k-d} \sum_{i=\max(0,k-d)}^{\min(k,\,s-1-d)} \binom{d}{k-i}, \qquad k=0,\dots,s-1 .
    ```

  - The interpolant has no real poles, and its error is O(h^{d+1}).
  - Its Lebesgue constant is **bounded by** 2^{d−1}(2 + ln s) (Bos, De Marchi, Hormann & Klein
    2012; cited from memory, unverified). The review measured it at s = 64: 6.1 for d = 3 and 93
    for d = 8.
  - Measured 1D convergence rates on M2 (review): d = 3: 4.2, d = 4: 4.9, d = 5: 6.3, d = 6: 6.9,
    d = 7: 8.4, d = 8: 8.7.
  - The barycentric form is the same one RLST evaluates, so T_uc reuses
    `tensor_barycentric_evaluate_2d` with FH weights we compute ourselves.
  - The degree d ∈ {3, …, 8} is chosen by the Phase 0 T3 spike.
- **Boundary values.** Both grids contain ±1. Boundary values transfer **exactly** in both directions
  only for zero boundary data (the Poisson u, which is all §4.2 needs) and at the four corners.
  Along an edge with non-zero data, the transfer still interpolates in the other variable, so it is
  exact only to round-off, about 1e-14 (review, 2026-10-05).
- **No autodiff.** Transfers run on the host in f64 and are never in the autodiff graph (§4.2).

### 2.7 Norms and quadrature
- **Clenshaw–Curtis weights.** For n Gauss–Lobatto nodes, w^CC ∈ ℝⁿ as in Trefethen's `clencurt`.
  They are exact for polynomials of degree ≤ n − 1, and positive. RLST has no quadrature; we add it.
- **Physical L² norm on the Chebyshev grid:**
  ```math
  \|v\|_{L^2(\Omega),\,\mathrm{CC}}^2 = \sum_{i,j} w^{\mathrm{CC}}_i w^{\mathrm{CC}}_j\, v_{ij}^2 .
  ```
- **Nodal ℓ² on the uniform grid**, as used by `LpLoss::rel` (CONVENTIONS §8). This is a fair proxy
  for L² because uniform cells have equal weight up to boundary halving. On the Chebyshev grid,
  unweighted nodal ℓ² over-weights the clustered boundary layer, so it is **not** used there.
- **Relative error with a norm floor.** For a test set with median reference norm ν:
  ```math
  e = \frac{\|\hat u - u\|}{\max(\|u\|, \tau)}, \qquad \tau = 10^{-3}\,\nu .
  ```
  Samples whose norm falls below τ are counted and reported. `LpLoss::rel` is undefined for an
  all-zero target (CONVENTIONS §8). Training never meets one, since corner-corrected f ≢ 0 almost
  surely.

### 2.8 Proposed conventions diff
Append to `docs/CONVENTIONS.md` (Phase 0 T1, signed off before Phase 1). This is a new §9 that
touches nothing in §1–§6, so **no `CONVENTION_VERSION` bump**. The preamble requires a bump only for
§1–§6; a PR note is enough.

```diff
+## 9. Chebyshev grids
+
+- Chebyshev–Gauss–Lobatto nodes in ascending order, computed as
+  x_j = sin(π(2j − (n−1)) / (2(n−1))), j = 0..n−1, evaluated in that order so that the
+  array is exactly antisymmetric, x_mid = 0 and x_0, x_{n−1} = ∓1 exactly. (RLST's
+  descending `chebychev_points` is not used for nodes.)
+- Uniform grids with endpoints are x_k = 2·(k/(s−1)) − 1, computed per index (no
+  accumulated step), so that ±1 are exact.
+- Physical domain [−1, 1] per axis; the model coordinate is ξ = (x + 1)/2, which on a
+  uniform grid with both endpoints is exactly §2's `arange(s)/(s − 1)`.
+- 2D fields are `[n_x, n_y]`, 'ij' indexed: axis 0 is x, axis 1 is y.
+- Physical L² norms on a Chebyshev grid use tensor Clenshaw–Curtis weights; unweighted
+  nodal norms are only used on uniform grids.
+- Uniform → Chebyshev transfer uses Floater–Hormann rational interpolation of degree d
+  (recorded in dataset metadata); Chebyshev → uniform uses Chebyshev barycentric
+  interpolation. Transfers run in f64 on the host and are not differentiated.
```

## 3. State of the art: ways to connect Chebyshev data to an FNO

### 3.1 FFT applied directly to Chebyshev-indexed values (rejected)
Running the existing FNO on values indexed by Chebyshev nodes treats the samples as uniformly
spaced in x, which they are not. The grid channels would also be wrong (CONVENTIONS §2).

In θ = arccos x the nodes *are* uniform, but u(cos θ) is a 2π-periodic **even** function only after
reflection. Without reflection, a zero-padded FFT on θ ∈ [0, π] is still a non-periodic transform
with a jump. Either way the result is not "a Fourier transform in physical coordinates". Rejected as
a silent default.

### 3.2 Interpolate to a uniform grid and use the existing FNO (recommended)
- **Idea.** T_cu maps the data to a uniform grid, the unchanged FNO runs there (with padding for the
  non-periodic Dirichlet problem, CONVENTIONS §3), and T_uc maps predictions back.
- **Cost.** Offline transfers of O(n·s·(n+s)) per sample; negligible.
- **Accuracy.** T_cu is spectrally accurate for smooth fields. T_uc is O(h^{d+1}). Both are measured
  (§8.1 L6).
- **Scope.** No model change and no `CONVENTION_VERSION` bump. This is the same footing as the
  published uniform-grid Darcy and Poisson FNO benchmarks (Li et al. 2021).
- **Weakness.** The model never sees the Chebyshev clustering, so boundary layers are resolved only
  as well as s allows.

### 3.3 Uniform → Chebyshev alternatives considered for T_uc
| Method | Accuracy | Stability | Notes |
|---|---|---|---|
| Global polynomial (barycentric on equispaced nodes) | exponential in exact arithmetic | Lebesgue constant ~2^s/(s log s): unusable | rejected |
| Floater–Hormann degree d | O(h^{d+1}) | Lebesgue constant ~2^{d−1} log s | **chosen**; reuses RLST's barycentric evaluation |
| Tensor cubic spline | O(h⁴) | stable | new code; one fixed order |
| Least-squares Chebyshev fit of degree M ≲ √s | spectral up to M | stable if M = O(√s) | needs a QR per transfer; caps resolution at ~8 modes for s = 64 |

### 3.4 FNO in θ-space and Chebyshev spectral layers (Phase 5, optional)
**The identity.** The Gauss–Lobatto nodes are uniform in θ. With θ_j = πj/(n−1), the ascending
storage of §2.2 is x_j = −cos θ_j = cos(π − θ_j). For the even extension of length 2(n−1),

```math
\tilde v = (v_0, v_1, \dots, v_{n-1}, v_{n-2}, \dots, v_1),
```

the DFT X_k is real and equals the DCT-I of v.
- Because the storage is ascending, the Chebyshev coefficients are c_k = (−1)^k X_k/(n−1), with c₀
  and c_{n−1} halved.
- Phase 5 must document these factors and the sign and pin them with a test.

**Consequence.** An FNO run on ṽ (rfft along the even extension) works in a Chebyshev-like basis. A
spectral weight per retained mode W_k ∈ ℂ, however, maps a real, even spectrum to a complex one. The
inverse transform is then neither even nor the reflection of a Chebyshev series.

Real weights alone are **not enough** to keep evenness. On the cfft axes, CONVENTIONS §5 keeps the
low block 0..m−1 and the high block −m..−1, so −m is retained while +m is not. The two corners also
have independent weights. Evenness needs all of:
- real weights;
- tied corners, W_high(−k) = W_low(k);
- a symmetric mode set.

Otherwise the output must be re-symmetrised.

**Reflection padding alone does not make the network a Chebyshev spectral operator.** The pointwise
layers, the grid channels (which would be θ, not x), and the CONVENTIONS §3 padding (zeros at the
end, not reflection) all change.

**Scope if pursued.** A reflect mode, a θ-grid option, and a real-weight constraint. All three touch
CONVENTIONS §2, §3 and §5, so they need a `CONVENTION_VERSION` bump. That is why the variant is
optional and sits after the required milestone.

**Alternative.** A dedicated Chebyshev spectral layer (DCT-I via an rfft of the even extension,
since Burn has no DCT; §6.2) is a distinct operator variant and stays out of scope.

### 3.5 Reference solvers considered
| Solver | Accuracy | Effort | Verdict |
|---|---|---|---|
| Chebyshev collocation with fast diagonalisation (§2.4) | exponential for smooth f | D₂ plus RLST `eig`, `inverse` | **chosen** (user, 2026-10-05) |
| Dense LU of the Kronecker sum | identical discretisation | O(n⁶) | test oracle only |
| ND finite elements (Q1/Q2 on `ndmesh`) | algebraic | assembly not found in ND (§6.4) | not pursued for 2D |
| Chebyshev–Galerkin (Shen 1995) | exponential | new basis code | unnecessary at these sizes |

## 4. Comparison and recommended strategy

### 4.1 Representation options
| Criterion | §3.1 direct FFT | **§3.2 interpolate (A)** | §3.4 θ-space FNO | §3.4 Chebyshev layer |
|---|---|---|---|---|
| Mathematically consistent | no | yes | yes, with constraints | yes |
| FNO code changed | no | **no** | pad mode, θ grid, weight constraint | new layer |
| Conventions | §2 violated | **new §9 only, no bump** | §2/§3/§5, bump | new layer, bump |
| Extra error source | aliasing in x | transfer, measured | none beyond the model | none beyond the model |
| Autodiff through the transfer | n/a | not needed | not needed | rfft (supported) |
| Carries to 3D | n/a | yes (`*_3d` in RLST) | yes | yes |
| Risk to the first milestone | high | **low** | medium | high |

### 4.2 Decision
Option A (user decision, 2026-10-05). The model is the existing `FNO<4>` with no code change; it
operates on uniform s × s data.
- **Training.** The loss is `LpLoss::rel` (CONVENTIONS §8) on uniform-grid targets, decoded as in the
  Darcy trainer (`src/neural_operators/training/trainers/darcy.rs`).
- **Evaluation.** Predictions are decoded, mapped back to the n × n Chebyshev grid with T_uc, and
  scored with CC-weighted L². The uniform-grid nodal error is reported alongside.
- **Transfers stay outside the autodiff graph.** If a Chebyshev-grid loss is ever wanted, T_uc is a
  fixed linear map (n × s per axis). It can be applied in Burn as two matmuls, which are
  differentiable, without touching the FNO. Recorded as a possible extension, not planned.
- **Boundary.** Raw predicted boundary values are scored as they are.
  - The pointwise y-normaliser has std = 0 at the boundary nodes, where u ≡ 0 in every sample. It
    therefore decodes any boundary output to ≈ 10⁻⁵ × output (`UnitGaussianNormalizer`, eps = 1e-5,
    CONVENTIONS §7).
  - So the boundary error is small **by construction**, and §8 reports it as such rather than as a
    learned property.
  - A "boundary-projected" variant, which sets the boundary to exactly 0, is reported separately and
    never substituted silently.

## 5. Architecture

### 5.1 Modules
New code, everything behind the cargo feature `chebyshev-data` unless marked otherwise.

| Path | Contents | RLST? |
|---|---|---|
| `src/neural_operators/chebyshev/mod.rs` | module docs, re-exports | — |
| `chebyshev/nodes.rs` | ascending Gauss–Lobatto nodes (sin form, §2.2), barycentric weights, Clenshaw–Curtis weights | no |
| `chebyshev/diff.rs` | `cheb_diff(n) -> (Array1<f64>, Array2<f64>)`, D₂ | no (pure ndarray) |
| `chebyshev/poisson.rs` | `PoissonSolver2d` (fast diagonalisation), `kron_lu_solve` (test oracle) | yes (`eig`, `inverse`, `lu`) |
| `chebyshev/transfer.rs` | `ChebToUniform`, `UniformToCheb` (FH), `fh_weights` | yes (barycentric evaluation) |
| `chebyshev/forcing.rs` | `ForcingSampler` (seeded series, corner correction), held-out generators | no |
| `chebyshev/eval.rs` | CC L² norms with the floor, boundary error, residual diagnostic, baselines | yes (through `transfer` and `svd`) |
| `data/loaders/poisson.rs` | `load_poisson`: reads `.npy`, splits train/val/test, fits normalisers on train (CONVENTIONS §7) | **no, always compiled** |
| `examples/gen/poisson_cheb.rs` | `gen_poisson_cheb`: writes datasets and `dataset_meta.json` | `required-features` |
| `examples/train/poisson.rs` | `train_poisson` | no |
| `examples/predict/poisson.rs` | `predict_poisson`: evaluation on both grids | `required-features` |
| `tests/poisson_overfit.rs` | tiny-dataset overfit test | feature-gated |

Changes to existing code:
- `src/neural_operators/mod.rs` gets a `#[cfg(feature = "chebyshev-data")] pub mod chebyshev;`.
- `data/loaders/mod.rs` exports `poisson`.
- `Cargo.toml` gets the feature, the optional dependencies, and `[[example]]` entries with
  `required-features`, so that CI's plain `cargo clippy --examples` skips them.
- `FNO`, `SpectralConv`, `LpLoss` and the trainer are unchanged.

### 5.2 Core interfaces
The signatures below are **proposed pseudocode**. Existing APIs are named as they are.

```rust
// chebyshev/nodes.rs
/// Ascending Gauss–Lobatto nodes x_j = sin(π(2j−(n−1))/(2(n−1))), shape [n]; n ≥ 2.
/// Exactly antisymmetric, endpoints exactly ±1 (CONVENTIONS §9 as proposed).
pub fn gl_nodes(n: usize) -> Array1<f64>;
/// Barycentric weights matching `gl_nodes(n)`, shape [n].
pub fn gl_bary_weights(n: usize) -> Array1<f64>;
/// Clenshaw–Curtis weights on `gl_nodes(n)`, shape [n], sum = 2.
pub fn cc_weights(n: usize) -> Array1<f64>;

// chebyshev/poisson.rs
pub struct PoissonSolver2d { n: usize, v: Array2<f64>, v_inv: Array2<f64>, lambda: Array1<f64> }
impl PoissonSolver2d {
    /// Precomputes D₂, its interior eigendecomposition, V⁻¹. Panics if n < 3 or if an
    /// eigenvalue has |Im| > 1e-10·|Re|.
    pub fn new(n: usize) -> Self;
    /// f: [n, n] nodal forcing ('ij'); returns u: [n, n] with u = 0 on the boundary rows/cols.
    pub fn solve(&self, f: ArrayView2<f64>) -> Array2<f64>;
}

// chebyshev/transfer.rs
pub struct ChebToUniform { /* n, s, weights */ }
impl ChebToUniform { pub fn new(n: usize, s: usize) -> Self;
                     pub fn apply(&self, v: ArrayView2<f64>) -> Array2<f64>; } // [n,n] -> [s,s]
pub struct UniformToCheb { /* s, n, d, FH weights */ }
impl UniformToCheb { pub fn new(s: usize, n: usize, d: usize) -> Self;      // panics if d ≥ s
                     pub fn apply(&self, v: ArrayView2<f64>) -> Array2<f64>; } // [s,s] -> [n,n]
pub fn fh_weights(s: usize, d: usize) -> Array1<f64>;

// chebyshev/forcing.rs
pub struct ForcingParams { pub k: usize, pub k0: f64, pub alpha: f64, pub corner_fix: bool }
pub struct ForcingSampler { /* params, StdRng */ }
impl ForcingSampler { pub fn new(p: ForcingParams, seed: u64) -> Self;
                      /// Coefficients a: [K, K]; evaluate on any 1D node set.
                      pub fn sample(&mut self) -> ChebSeries2d; }
impl ChebSeries2d { pub fn eval_tensor(&self, x: &Array1<f64>, y: &Array1<f64>) -> Array2<f64>; }

// data/loaders/poisson.rs  (always compiled)
pub struct PoissonConfig { pub dir: PathBuf, pub n_train: usize, pub n_val: usize, pub s: usize }
pub fn load_poisson<T: HostFloat>(c: &PoissonConfig)
    -> Result<(OperatorDataset<T>, OperatorDataset<T>, PoissonNormalizers), LoadError>;
// inputs [N, s, s, 1] (encoded), targets [N, s, s] (train encoded, val raw), as Darcy
```

Existing APIs reused, unchanged:
- `FNOConfig` / `FNO<4>` (`models/fno.rs:18-52, 249`);
- `UnitGaussianNormalizer` (`data/transforms/normalizers.rs:36`);
- `OperatorDataset::from_f64` (`data/dataset.rs:33`);
- `NpyFileReader` (`data/io/readers/npy.rs:13`);
- `build_training_components` and `training_loop` (`training/trainer.rs:249, 315`);
- `write_run_artifacts` (`metrics/io.rs:70`);
- `LpLoss::rel` (`losses/data_losses.rs:96`).

### 5.3 Data layout and flow
```
ForcingSampler(seed) ──► ChebSeries2d ──eval on GL nodes──► f_cheb [N,n,n] f64
                                     └─eval on uniform───► f_unif_direct (test oracle only)
f_cheb ──PoissonSolver2d──► u_cheb [N,n,n] f64          (boundary exactly 0)
f_cheb, u_cheb ──ChebToUniform──► f_unif, u_unif [N,s,s] f64
                         └─ write .npy (f64) + dataset_meta.json
load_poisson: f_unif → x [N,s,s,1], u_unif → y [N,s,s]; normalisers fit on train (f64, host)
DeviceBatcher (f32, device) ─► FNO<4>.forward [B,s,s,1] → [B,s,s,1] ─► LpLoss::rel (decoded)
predict: decode ─► û_unif [N,s,s] ─UniformToCheb(d)─► û_cheb [N,n,n] ─► CC-L², boundary, residual
```

- **Copies.** On the host, data passes ndarray f64 → `OperatorDataset<f32>` with one copy.
  `DeviceBatcher` uploads it once (`data/device_batcher.rs:87`). The batch axis is 0. The channel
  axis is last at the model boundary and first inside the model (CONVENTIONS §1).
- **Flattening.** Row-major (ndarray default, C order) over `[s, s]` for `flatten_pair`
  (`training/trainer.rs:88`). RLST arrays are column-major (`dense/layout.rs`). The transfer wrappers
  copy into and out of RLST explicitly and are tested on an asymmetric field (§8.1 L1), so no layout
  assumption leaks.
- **Files.** `datasets/poisson_cheb/<set>/{f_cheb,u_cheb,f_unif,u_unif}.npy`, one directory per set:
  `train`, `val`, `test` and H1–H4.
  - About 44 MB per f64 array of 1 300 × 65² values. Not committed (`datasets/README.md`).
- **`dataset_meta.json`** records:
  - n, s, d, K, K₀, α, the corner fix, c, and the seeds per set;
  - set sizes;
  - the sciml-rs commit, the RLST version, and `CONVENTION_VERSION`;
  - SHA-256 of each `.npy`.

### 5.4 Compatibility
- **No changes:**
  - existing configs, checkpoints, examples or public APIs;
  - `FNOConfig` fields;
  - default features (`flex` only).
- **New `PoissonConfig`.** New configs only; there are no old ones to keep loading.
- **Run artefacts.** `train_poisson` writes the usual `runs/<name>_<unix>/` artefacts
  (`metrics/io.rs:70`). It adds `dataset_meta.json` (copied) and the normaliser records, as
  `examples/train/darcy.rs:95-110` does.
- **Default CI unaffected.** `cargo test`, `clippy` and `doc` without the feature do not compile
  RLST. One added CI job installs OpenBLAS and runs the feature (Phase 0 T2).

### 5.5 What carries over to 3D
- `FNO<5>` exists in type form (rank R = D + 2).
- RLST has `tensor_barycentric_evaluate_3d` (`interpolation.rs:380`).
- Fast diagonalisation extends axis by axis (λ_i + λ_j + λ_k).
- CC weights and FH are tensor products.
- The dataset layout gains one axis.
- New for 3D:
  - memory grows as n³ (129³ f64 ≈ 17 MB per sample);
  - distributed grids would come from ND `ndmesh` (`unit_cube_distributed`; docs only, unverified
    in source);
  - the MPI integration is a separate objective.

## 6. Platform and dependency considerations

### 6.1 RLST
**Version and licence.**
- Read at `~/Code/rlst` @ `3153a14` (2026-08-23). The crates.io 0.6.1 release has the same modules
  (`interpolation.rs`, `chebychev.rs`, `lib.rs` feature gates).
- Pin `rlst = "=0.6.1"`. Phase 0 T2 re-reads the line numbers in the pinned version (§9.2 Q5).
- Licence MIT/Apache-2.0, the same as sciml-rs.

**Used:**
- `interpolation` module, compiled without features (`rlst/src/lib.rs:35`):
  - `chebychev_points` (`interpolation.rs:95`), only in the T2 smoke test; our nodes use the sin form (§2.2);
  - `barycentric_chebychev_weights` (`:152`);
  - `barycentric_evaluate_1d` (`:199`);
  - `tensor_barycentric_evaluate_2d` (`:285`), with exact node coincidence at `:246`.
- Dense LAPACK:
  - `eig` (`dense/linalg/lapack/eigenvalue_decomposition.rs:116`, `?geev`, complex eigenvalues and
    right eigenvectors). In 0.6.1 it is at `:84`; the file differs from `3153a14`, so T2 re-reads every line cited here;
  - `inverse` and `lu` (`traits/linalg/decompositions.rs:32, 48`);
  - `svd`, for the PCA baseline.

**Not used:**
- `chebychev.rs`, the DCT-I value↔coefficient transforms and coefficient derivatives. It is gated on
  `feature = "fftw"` (`lib.rs:26`), and FFTW is GPL (RLST README).
- RLST's `burn` feature (`interface/burn.rs`). It pins crates.io Burn 0.20/0.21, which is a different
  crate from our fork, so data crosses via ndarray and `.npy` instead.

**Gaps we fill:**
- the differentiation matrix;
- Clenshaw–Curtis weights;
- FH weights;
- fast diagonalisation;
- the ascending-order wrappers.

**Build.** RLST needs `blas-src`/`lapack-src`: Accelerate on macOS, OpenBLAS on Linux, as in RLST's
own dev-dependencies (`~/Code/rlst/rlst/Cargo.toml:50-56`). CI already has a commented-out
`libopenblas-dev` step (`.github/workflows/run-tests.yml`). Phase 0 T2 proves the build on both.

### 6.2 Burn fork (`ax1s-x1zz/burn` @ `ddfa9af`)
- `rfft`/`irfft` take any length. Powers of two use the radix-2 backend op; other lengths use
  Bluestein, which pads to the next power of two ≥ 2n − 1 (`burn-signal/src/functions/fft.rs:35,
  53, 102, 118, 167-223`). The transcript's power-of-two concern is about speed, not correctness.
  - s = 64 with padding 9 gives a padded length of 73, which goes through Bluestein.
  - s = 64 with padding 0 or 64, or s = 55 with padding 9, gives a power of two.
  - Phase 0 T4 measures the epoch time for these options before s and the padding are fixed.
- No DCT exists in burn-signal (grep, 2026-10-05). The θ-space variant (§3.4) would build one from
  `rfft` of the even extension.
- Autodiff through rfft/irfft is supported (`burn-signal/src/lib.rs:19`,
  `backends/autodiff.rs:13-24, 88-100`). Option A needs no gradient through the transfers.

### 6.3 Precision and backends
- **Precision.** The solver, transfers, quadrature and metrics are f64 on the host. The model is f32
  on the device.
- **Backends.** The model runs on `flex`, the default. Metal is run for timing and parity in Phase 3,
  and the hand-back says whether it was. CUDA and wgpu are not planned.
- **Known Metal quirk.** It concerns channels-first `cat` after a permute (`CLAUDE.md`). Option A adds
  no new layout operation on the model path.

### 6.4 ND
Read on the web at codeberg.org/nd-project/nd @ `f8cbffe` (2026-10-01). There is no local checkout,
and Codeberg blocked listing the example directories.
- **Verified from the raw `Cargo.toml` files:**
  - crates `ndelement`, `ndmesh` and `ndfunctionspace` (0.4.0);
  - all depend on `rlst = "0.6"`;
  - MPI is optional.
- **Verified from the docs:**
  - Lagrange elements (equispaced or GLL) on interval, quadrilateral and hexahedron cells;
  - mesh shapes `unit_square` and `unit_cube`, with `_distributed` variants.
- **Unknown:** whether ND has a Laplace assembly or a Poisson example.
- **Role here.** ND is **not** on the 2D critical path. Its role is the 3D distributed grid (§5.5).

### 6.5 Resource estimates (order of magnitude; not measured)
These are estimates, not results.
- **Generation.** About 1 900 solves at n = 65 is O(1 900 × 4 × 63³ × 2) ≈ 4·10⁹ flops: seconds.
  Writing about 0.3 GB of `.npy` is seconds.
- **Training.** Darcy-sized (1 000 samples at 64², padded, 4 layers, H = 32, 500 epochs) on `flex`
  or Metal. Phase 0 T4 measures the epoch time, so the budget is set from a measurement rather than
  assumed.
- **Inference.** Milliseconds per batch, plus a negligible T_uc.

## 7. Phased implementation plan

Each phase leaves `main` green. Risky items come first: the conventions, the dependency build, and
transfer stability are all in Phase 0. Component IDs are `C<phase>.<k>`; phase READMEs and task
briefs cite them.

Each phase exit checklist keeps three kinds of item separate:
- tasks merged (implementation done);
- reviewer passes (the `reviewer` and `numerics` agents);
- **human sign-off**, ticked by hand.

### Phase 0: conventions and feasibility
- **Objective.** Fix the conventions, prove the RLST build, and measure the two numbers the design
  depends on: the transfer floor and the FFT size cost.
- **Prerequisites.** This document signed off, or at least §2 and §9.2.
- **Deliverables.** The conventions §9 diff landed, the feature skeleton building in CI, and two
  spike reports.

| ID | Component / task | Files | Test or experiment, and acceptance | Depends on |
|---|---|---|---|---|
| C0.1 / T1 | CONVENTIONS §9 (§2.8), no version bump | `docs/CONVENTIONS.md` | Text matches §2.8; human sign-off | — |
| C0.2 / T2 | `chebyshev-data` feature, RLST `=0.6.1` + BLAS/LAPACK, empty `chebyshev` module, CI job | `Cargo.toml`, `src/neural_operators/{mod.rs,chebyshev/mod.rs}`, `.github/workflows/run-tests.yml` | Default `cargo test`/`clippy`/`doc` unchanged; `cargo test --features chebyshev-data` builds on macOS and in CI; one smoke test calls `chebychev_points` and pins the descending order and endpoints | — |
| C0.3 / T3 | Transfer-stability spike | `spikes/cheb-transfer/` (stand-alone crate, not a workspace member) + `SPIKE_REPORT.md` | Tables: T_cu error vs n for M2/M3; T_uc FH error vs d ∈ {3..8} and s ∈ {33, 64, 65, 128}; round trip T_uc∘T_cu on M2, M3 and 20 Poisson solutions of the family (throwaway Kronecker-LU solver in the spike), for s ∈ {64, 128}; recommends the smallest d with floor ≤ 1e-5 (§8.2 L6). Recommends d. | — |
| C0.4 / T4 | FFT size and padding timing spike | `spikes/fft-sizes/` + report | Epoch time on flex (and Metal if available) for s ∈ {55, 63, 64, 65} × padding ∈ {0, 9, s} on random data; recommends s and padding | — |

- **Unresolved decisions:** §9.2 Q1, Q4 and Q5.
- **Exit checklist:**
  - [ ] T1 merged
  - [ ] T2 merged; CI feature job green
  - [ ] T3 report merged
  - [ ] T4 report merged
  - [ ] Reviewer pass on each task
  - [ ] **Human sign-off:** CONVENTIONS §9
  - [ ] **Human sign-off:** FH degree d
  - [ ] **Human sign-off:** s and padding
  - [ ] Design §2.6, §6.2 and §9.2 updated with the decisions

### Phase 1: verified reference solver
- **Objective.** A Poisson solver whose error is known before any learning happens.
- **Prerequisites.** Phase 0 exit.

| ID | Component | Files | Tests (§8.1) and acceptance | Depends on |
|---|---|---|---|---|
| C1.1 | `gl_nodes`, `gl_bary_weights`, `cc_weights` | `chebyshev/nodes.rs` | L1 (order, endpoints, exact antisymmetry, nesting n=65 ⊂ n=129 bit for bit); L7 (CC exact on x^a y^b, a, b ≤ n − 1) | C0.2 |
| C1.2 | `cheb_diff`, D₂ | `chebyshev/diff.rs` | L2 (D exact on polynomials of degree ≤ n − 1 vs analytic derivatives) | C1.1 |
| C1.3 | `PoissonSolver2d`, `kron_lu_solve` oracle | `chebyshev/poisson.rs` | L3 (FD vs Kronecker LU, n ∈ {5, 9, 17, 33}); eigenvalues real and negative; cond(V) recorded | C1.2 |
| C1.4 | Manufactured and convergence suite | `chebyshev/poisson.rs` tests, `#[ignore]` table printer | L4 (M1 exact; M2/M3 convergence); L5 (boundary exactly 0) | C1.3 |

- **Exit checklist:**
  - [ ] C1.1–C1.4 merged
  - [ ] Convergence table recorded in design §8.3
  - [ ] Numerics review
  - [ ] **Human sign-off:** solver accepted as the reference

### Phase 2: dataset generation and grid transfer
- **Objective.** Reproducible datasets on both grids, with a measured transfer floor.
- **Prerequisites.** Phase 1 exit, and d signed off.

| ID | Component | Files | Tests and acceptance | Depends on |
|---|---|---|---|---|
| C2.1 | `ForcingSampler`, held-out generators H1–H3 | `chebyshev/forcing.rs` | Seeded determinism (same seed, same bits); f = 0 at the corners when `corner_fix`; GL evaluation matches T_p(x_j) = (−1)^p cos(pπj/(n−1)) on the ascending nodes and the three-term recurrence elsewhere (independent of the `cos(p arccos x)` implementation) | C1.1 |
| C2.2 | `ChebToUniform`, `UniformToCheb`, `fh_weights` | `chebyshev/transfer.rs` | L6: T_cu exact on polynomials of degree < n; T_cu(f_cheb) vs direct uniform evaluation of the series; FH rate on M2 within 1 of d + 1, fitted only on errors above 1e-12 (round-off floor); round-trip floor on reference u | C1.1, C0.3 |
| C2.3 | `gen_poisson_cheb` example + `dataset_meta.json` | `examples/gen/poisson_cheb.rs`, `Cargo.toml` | Writes all sets, including the training set mapped to s = 128 for H4 normalisers (§2.5); a re-run gives identical SHA-256; self-convergence of labels n = 65 vs 129 on 20 samples (L4b) | C1.3, C2.1, C2.2 |
| C2.4 | `load_poisson` (train/val split, normalisers on train only) | `data/loaders/poisson.rs` | Unit test on a synthetic `.npy` fixture written in the test: shapes, axis order, normaliser fitted only on train, val y left raw | — (no RLST) |
| C2.5 | Pilot for forcing parameters | `SPIKE_REPORT` section in `docs/2d-chebyshev-poisson-fno/phase2/` | Spectrum of f, ‖u‖ distribution, fraction below τ; confirms or revises K, K₀, α | C2.3 |

- **Exit checklist:**
  - [ ] C2.1–C2.5 merged
  - [ ] Transfer floor recorded in §8.3
  - [ ] **Human sign-off:** forcing parameters and dataset sizes

### Phase 3: FNO training and held-out evaluation
- **Objective.** Train the unchanged FNO and evaluate it honestly on both grids.
- **Prerequisites.** Phase 2 exit.

| ID | Component | Files | Tests and acceptance | Depends on |
|---|---|---|---|---|
| C3.1 | Overfit integration test | `tests/poisson_overfit.rs` | L8: 8 samples, s = 32, flex f32; train rel-L2 < 1e-2 within 1 500 steps (provisional; calibrated once, then fixed) | C2.3, C2.4 |
| C3.2 | `train_poisson` example (val set as the trainer's eval set) | `examples/train/poisson.rs` | Runs end to end; writes run artefacts + `dataset_meta.json` | C2.4 |
| C3.3 | Evaluation metrics | `chebyshev/eval.rs` | CC-L² with floor (§2.7); boundary max error, raw and projected; tested on analytic fields with known norms | C1.1, C2.2 |
| C3.4 | Baselines: zero, mean, PCA-ridge | `chebyshev/eval.rs` | PCA-ridge recovers a known linear map on synthetic data to 1e-8 relative (f64) | C3.3 |
| C3.5 | Residual diagnostic | `chebyshev/eval.rs` | D₂ applied directly to u_cheb (no transfer) gives residual ≤ 1e-8 relative to ‖f‖ (f64); the documented limitation (§8.1 L10) applies to T_uc(û), where D₂ amplifies FH error by up to O(n⁴) | C1.2 |
| C3.6 | `predict_poisson` + pilot calibration + final evaluation | `examples/predict/poisson.rs`, design §8.2/§11 | Pilot (200 train, 100 epochs) → thresholds written into §8.2 and signed off → final run against H0–H4 | C3.2–C3.5 |

- **Exit checklist:**
  - [ ] C3.1–C3.6 merged
  - [ ] Thresholds fixed before the final run (commit order shows it)
  - [ ] Metal run or "not run" stated
  - [ ] **Human sign-off:** results

### Phase 4: reproducible example and 3D readiness
- **Objective.** Anyone can reproduce the experiment from the README, and the 3D step is scoped.
- **Deliverables:**
  - C4.1: a README section with the exact commands and runtimes.
  - C4.2: design §11 Outcome, with decisions as taken and measured numbers.
  - C4.3: `docs/2d-chebyshev-poisson-fno/3d-readiness.md` (the §5.5 items, measured memory and time
    extrapolation, ND/MPI questions), as input to a 3D design document.
- **Exit checklist:**
  - [ ] C4.1–C4.3 merged
  - [ ] **Human sign-off:** ready to start the 3D design

### Phase 5 (optional, non-blocking): θ-space comparison
- **Objective.** Compare §3.4 with option A on the same datasets.
- **Components:**
  - C5.1: a convention proposal (θ grid, reflect padding, real-weight constraint; `CONVENTION_VERSION`
    bump).
  - C5.2: implementation behind new `Option<_>` config fields (CLAUDE.md compatibility rule).
  - C5.3: an evenness-preservation test. With real, corner-tied weights and a symmetric mode set
    (§3.4), the output of an even input stays even to 1e-6 relative in f32. A control with untied
    weights must fail it.
  - C5.4: a comparison report.
- **Prerequisite.** Phase 3 exit. Phase 4 may proceed in parallel.

## 8. Validation and benchmarking

### 8.1 Test layers
Each layer names the error it isolates: **S** for solver, **T** for transfer, **M** for model.

| Layer | Check | Oracle | Error isolated | Where |
|---|---|---|---|---|
| L1 | Node order, endpoints, antisymmetry, nesting; axis order of a `[n, n]` field and of `[B, s, s, 1]` batches using f = x + 3y + x·y² (odd in each axis, not swap-symmetric, so x/y flips and swaps are both caught) | closed form | indexing | C1.1, C2.4 |
| L2 | D, D₂ on xᵖ, p ≤ n − 1 | analytic derivatives | S | C1.2 |
| L3 | Fast diagonalisation = Kronecker LU | independent dense solve | S (algorithm) | C1.3 |
| L4 | M1 exact; M2, M3 errors at n ∈ {9, 17, 33, 65} | analytic u | S (discretisation) | C1.4 |
| L4b | Sampled family: ‖u₆₅ − u₁₂₉‖ / ‖u₁₂₉‖ (CC-L², u₁₂₉ restricted to the n = 65 nodes by nesting) | finer solve | S on the actual data | C2.3 |
| L5 | Boundary rows and columns exactly 0 | definition | S | C1.4 |
| L6 | T_cu vs direct evaluation; T_uc rate; round trip T_uc∘T_cu on reference u | analytic or series | T | C2.2 |
| L7 | CC exactness | analytic integrals | quadrature | C1.1 |
| L8 | Overfit 8 samples | loss decreases to threshold | pipeline wiring | C3.1 |
| L9 | Held-out H0–H4: uniform nodal rel-ℓ², Chebyshev CC rel-L² (floored), boundary max | reference u | M (+T, bounded by L6) | C3.6 |
| L10 | Residual −Δ_h û − f on the Chebyshev grid using D₂ applied to T_uc(û), CC-L² relative to ‖f‖ | — | diagnostic only | C3.5 |
| L11 | Baselines: 0, training mean, PCA-ridge (r components on f_unif → u_unif) | — | is the model using f? | C3.4 |

Notes:
- **L4b nesting.** The GL nodes for n = 65 are a subset of those for n = 129, so the comparison
  needs no interpolation.
- **L10 limitation.** Applying D₂ amplifies high-frequency model error by up to O(n⁴). The residual
  can be large while L9 is small, so it is a diagnostic and is never an acceptance criterion.
- **L11 caveat.** 𝒢 is linear, and training f lies in a span of dimension ≤ K² = 256. A linear
  regression from f to u with ≥ 256 samples can therefore match the in-distribution test (H0) almost
  to solver precision.
  - The FNO is **not required** to beat PCA-ridge on H0. The comparison is reported as it is.
  - The FNO's added value is assessed on H2 (outside the span) and H4 (resolution), where the linear
    baseline cannot extrapolate or does not apply.
  - It must clearly beat the zero and mean baselines.

### 8.2 Error metrics and tolerances
**Solver and transfer.** f64; each threshold is fixed now.

| Check | Tolerance | Relative to |
|---|---|---|
| L1 node values | exact: x_mid = 0, x₀ = −1, x_{n−1} = 1, x_j = −x_{n−1−j}; other nodes ≤ 2·eps absolute vs −cos(πj/(n−1)) | closed form, f64 |
| L2 D on xᵖ, n ≤ 33 | 1e-12 · max\|p xᵖ⁻¹\| (measured 5.7e-14) | analytic, f64 |
| L2 D₂ on xᵖ, n ≤ 33 | 1e-9 · max\|p(p−1) xᵖ⁻²\| (measured 1.2e-10; error grows as O(n⁴ eps)) | analytic, f64 |
| L3 FD vs LU, n ≤ 33, random non-symmetric F | 1e-12 relative max-norm (measured ≤ 1.6e-14; a missing transpose gives 0.2–1) | LU solution, f64 |
| L4 M1, 5 ≤ n ≤ 129 | 1e-12 relative max-norm (measured 3.6e-13 at n = 129) | analytic, f64 |
| L4 M2/M3, n = 33 | 1e-10 relative max-norm | analytic, f64 |
| L4b family, n = 65 | ≤ 1e-10 relative CC-L² (measured 4.9e-13 in review; re-confirmed by the Phase 2 pilot) | n = 129 solve |
| L5 boundary | exactly 0.0 | definition |
| L6 T_cu, polynomial of degree < n | 1e-12 relative max | direct evaluation, f64 |
| L6 round-trip floor | reported; must be ≤ 1/100 of the pilot model error | reference u, f64 |
| L7 CC | 1e-13 absolute | exact integral |

**Model.** f32 thresholds are calibrated, not invented.
- **Pilot.** Phase 3 C3.6 runs 200 training samples for 100 epochs.
- **Thresholds.** From the pilot's H0 median error e_p, the final criterion for H0 is set to 1.5 · e_p
  rounded up. The thresholds are written into this section and signed off **before** the full run
  starts.
- **Also required on H0:**
  - FNO error ≤ 0.1 × the mean-baseline error;
  - the Chebyshev-grid CC error is ≤ 1.5 × the uniform-grid error, i.e. the transfer does not
    dominate;
  - the raw boundary max error is reported alongside the projected one.
- **H1–H4.** Reported, with no pass threshold. They are generalisation measurements.

### 8.3 Reference results and performance measurement
- **Reference numbers.** There is no published number for this exact setup. Li et al. (2021) report
  uniform-grid Darcy errors, which are context only, not a target.
- **Recorded here as phases complete:**
  - the Phase 1 convergence table;
  - the Phase 0/2 transfer floor;
  - the Phase 0 T4 timings.
- **Timing.** Wall-clock per epoch on flex (and Metal), median of 3 runs after one warm-up. Data
  generation time and inference time per batch are reported separately.

## 9. Risks, open questions and working with Claude Code

### 9.1 Risks
| Risk | Mitigation | Retired by |
|---|---|---|
| BLAS/LAPACK does not build in CI or for teammates | Optional feature; OpenBLAS CI step; default build untouched | Phase 0 T2 |
| FH transfer floor too high at s = 64 | Spike over d and s before committing; larger s or a spline fallback (§3.3) | Phase 0 T3 |
| Non-power-of-two FFT too slow | Measure; choose s/padding giving 2^k | Phase 0 T4 |
| RLST node order, docstring, and the ~3e-16 mismatch of the reversed cos form | Own sin-form nodes; L1 pins exact antisymmetry, endpoints, nesting | C1.1 |
| Ill-conditioned V in fast diagonalisation | Measure cond(V); LU oracle; fall back to Kronecker LU at n ≤ 33 or to a Schur-based Sylvester solve | C1.3 |
| Corner singularities spoil reference accuracy | Corner-corrected family; L4b self-convergence; H3 kept as a labelled hard case | C2.3 |
| Boundary error trivially small via normaliser | Stated in §4.2; raw and projected both reported | C3.3 |
| Linear baseline matches the FNO on H0 | Expected (§8.1 L11); value judged on H2/H4 | C3.6 |
| RLST's `burn` feature incompatible with the fork | Not used; data via ndarray/`.npy` | by design |
| ND role unclear | Not on the 2D critical path | §6.4 |

### 9.2 Open questions for the user
Each has a recommendation.
1. **Branch name.** The branch is `design/2d-possion-cheb` (typo). *Recommend:* rename to
   `design/2d-chebyshev-poisson-fno` before the first PR. Not blocking.
2. **Forcing family.** Is a "random Chebyshev series, corner-corrected" what the supervisor meant by
   "f is just [Chebyshev]"? *Recommend:* yes. The alternative is a Gaussian random field
   (Li et al.'s Darcy coefficients) evaluated on the GL grid. Blocks Phase 2, not Phase 0.
3. **K, K₀, α and the dataset sizes.** *Recommend:* the provisional values in §2.5, confirmed by
   pilot C2.5.
4. **Should ND play any role in 2D?** *Recommend:* no; record it for 3D. Blocks nothing.
5. **RLST pin.** `=0.6.1` from crates.io or the git rev `3153a14`? *Recommend:* `=0.6.1` for
   reproducible builds, since `Cargo.lock` is not committed. T2 confirms that the APIs cited here
   exist in it. Blocks Phase 0 T2.
6. **Grid sizes.** n = 65 for labels, and s chosen by T4 (64 is the default). *Recommend:* accept
   after T4.

### 9.3 Handing components to Claude Code
- **Parallelism.**
  - Self-contained, parallel: T2, T3 and T4 in Phase 0.
  - C2.4 is parallel to everything; it needs no RLST.
  - C3.3–C3.5 are parallel once Phase 2 is merged.
- **Human sign-off before dependants start:**
  - CONVENTIONS §9 (T1) before Phase 1;
  - the d, s and padding decisions before C2.2 and C3.2;
  - forcing parameters before the final dataset;
  - model thresholds before the C3.6 final run.
- **One session per task brief** (`/do-task`); `reviewer` and `numerics` agents at the end of each.

**Requirement traceability.** Each item from the objective prompt maps to a design section, a task,
and a check.

| Requirement | Design | Task | Validated by |
|---|---|---|---|
| Capability/gap table | §6, §11.1 | — | review |
| Domain, BCs, node family and ordering | §2.1–2.3 | T1, C1.1 | L1 |
| Grid sizes, refinement | §2.4, §8 | C1.4, T4 | L4, L4b |
| Forcing distribution | §2.5 | C2.1, C2.5 | pilot, determinism test |
| Reference solver, BC enforcement | §2.4 | C1.3 | L3–L5 |
| Inputs, outputs, shapes, coordinate channels | §2.3, §5.3 | C2.4 | L1 |
| Dataset sizes, seeds, splits, normalisation on train only | §2.5, §5.3 | C2.3, C2.4 | C2.4 tests, SHA re-run |
| Manufactured vs family vs held-out | §2.5 | C1.4, C2.1, C2.3 | L4, L9 |
| Representation decision | §3–§4 | T3, C2.2 | L6 |
| FFT size restrictions | §6.2 | T4 | timing report |
| Loss, norms, quadrature, near-zero norms | §2.7 | C1.1, C3.3 | L7, C3.3 tests |
| Overfit test | §8.1 | C3.1 | L8 |
| Held-out accuracy, boundary error | §8.2 | C3.6 | L9 |
| Residual diagnostic | §8.1 | C3.5 | L10 |
| Baselines | §8.1 | C3.4 | L11 |
| Resource estimates | §6.5 | T4, C3.6 | §8.3 |
| 3D carry-forward | §5.5 | C4.3 | sign-off |
| θ-space comparison (optional) | §3.4 | Phase 5 | C5.3 |

## 10. References
**Sources read for this document:**
- sciml-rs @ `478c2cb` (this worktree): `docs/CONVENTIONS.md`, `CLAUDE.md`, and the files cited
  above.
- Burn fork `ax1s-x1zz/burn` @ `ddfa9af`: `crates/burn-signal/src/functions/fft.rs`, `lib.rs`,
  `backends/autodiff.rs`.
- RLST, `~/Code/rlst` @ `3153a14` (codeberg.org/rlst/rlst): `rlst/src/lib.rs`,
  `rlst/src/interpolation.rs`, `rlst/src/traits/linalg/decompositions.rs`,
  `rlst/src/dense/linalg/lapack/eigenvalue_decomposition.rs`, `rlst/Cargo.toml`. Also crates.io
  `rlst-0.6.1` (module list, features).
- ND, codeberg.org/nd-project/nd @ `f8cbffe`: raw `Cargo.toml` files and `ndelement` sources; docs
  at bempp.github.io/nd.
- Workflow model: github.com/tbetcke/fmm (`docs/design/`, `docs/phase*/`, `docs/CONVENTIONS.md`).

**Literature,** cited from memory and **unverified**; check before citing in a paper:
- L. N. Trefethen, *Spectral Methods in MATLAB*, SIAM, 2000. `cheb`, `clencurt`, Poisson on a square.
- L. N. Trefethen, *Approximation Theory and Approximation Practice*, SIAM, 2nd ed. 2019.
- R. E. Lynch, J. R. Rice, D. H. Thomas, "Direct solution of partial difference equations by tensor
  product methods", *Numer. Math.* 6 (1964).
- D. Gottlieb, L. Lustman, "The spectrum of the Chebyshev collocation operator for the heat
  equation", *SIAM J. Numer. Anal.* 20 (1983).
- M. S. Floater, K. Hormann, "Barycentric rational interpolation with no poles and high rates of
  approximation", *Numer. Math.* 107 (2007).
- L. Bos, S. De Marchi, K. Hormann, G. Klein, "On the Lebesgue constant of barycentric rational
  interpolation at equidistant nodes", *Numer. Math.* 121 (2012).
- R. B. Platte, L. N. Trefethen, A. B. J. Kuijlaars, "Impossibility of fast stable approximation of
  analytic functions from equispaced samples", *SIAM Review* 53 (2011).
- J. Waldvogel, "Fast construction of the Fejér and Clenshaw–Curtis quadrature rules", *BIT* 46
  (2006).
- J. Shen, "Efficient spectral-Galerkin method II: direct solvers…", *SIAM J. Sci. Comput.* 16
  (1995).
- Z. Li et al., "Fourier Neural Operator for Parametric Partial Differential Equations", ICLR 2021.

## 11. Outcome
To be filled at the end of each phase: decisions as taken, measured numbers, and sign-off dates.
Signed-off sections above are not rewritten.

### 11.1 Capability and gap table (as of 2026-10-05)
| Capability | Status | Where |
|---|---|---|
| Chebyshev GL/Gauss nodes | **verified** (RLST, descending, cos form); not used for nodes, sin form instead (§2.2) | `rlst/src/interpolation.rs:95` |
| Chebyshev barycentric weights, 1D/2D/3D and tensor evaluation | **verified** (RLST, no feature) | `interpolation.rs:152, 199, 285, 380` |
| General barycentric weights | **verified** (RLST) | `interpolation.rs:122` |
| Chebyshev value↔coefficient transforms (DCT-I), coefficient derivatives | **verified, not used** (needs GPL `fftw`) | `rlst/src/chebychev.rs`, `lib.rs:26` |
| Dense `eig`, `eigh`, `lu`, `inverse`, `svd` | **verified** (RLST, LAPACK) | `decompositions.rs`, `lapack/eigenvalue_decomposition.rs:116` |
| Differentiation matrices | **missing**, proposed C1.2 | — |
| Clenshaw–Curtis quadrature | **missing**, proposed C1.1 | — |
| Kronecker / tensor-operator application | **missing** (axis-wise matmuls in C1.3) | — |
| Floater–Hormann weights | **missing**, proposed C2.2 | — |
| Poisson solver | **missing** (RLST, ND, sciml-rs); proposed C1.3 | — |
| ND grids, fields, storage | **partly verified** (crates, Lagrange GLL, `unit_square`/`unit_cube`); assembly **unknown** | §6.4 |
| RLST ↔ Burn bridge | **verified, unusable** (crates.io Burn ≠ fork) | `rlst/src/interface/burn.rs` |
| 2D FNO, any FFT length, autodiff through FFT | **verified** | `models/fno.rs`, `burn-signal/.../fft.rs` |
| Uniform grid channels [0, 1] | **verified** | CONVENTIONS §2, `fno.rs:197` |
| Normalisers, `.npy` IO, trainer, run artefacts | **verified** (Darcy path) | §5.2 |
| Non-uniform quadrature in losses | **missing**; not needed for training (§4.2) | `losses/data_losses.rs:59-100` |
| Validation split in trainer | **verified usable**: trainer takes one eval set; val is passed as it | `training/trainer.rs:249` |
