# Phase 6: information content of the samples

As of 2026-10-27; done, T1 and T2 (decision 30, amended 2026-10-27 with the Phase 6
planning decisions). Design: `docs/design/2d-chebyshev-poisson-fno.md` (§10 Phase 6,
§13.4). `docs/CONVENTIONS.md` wins on any conflict. Runs beside Phase 5 (design §10).

Phase 6 asks whether the forcings f and the solutions u of the stage 3 dataset occupy a
low-dimensional subspace, and how that changes on the Phase 5 learning grids. It answers
with the proper orthogonal decomposition (POD): the SVD of the centred snapshot matrix,
one column per training sample, built separately for f and for u. It reports:
- singular-value decay and cumulative explained energy;
- the rank needed for 90%, 99% and 99.9% of the energy;
- the relative reconstruction error against rank, on the train split (where the basis is
  fitted) and on the test split (held out);
- the same at each learning grid s, with the snapshots restricted by R (T2).

This is **dataset-structure analysis, not network performance** (design §13.4). A low
energy rank does not show that the FNO can learn the operator with that many Fourier
modes, or from that many samples. No model is trained or evaluated in this phase.

## Read first

- Design §3.3 (the sine KL series of f, λ_kl, μ_kl, S_K, the tail table), §3.4 (the
  Clenshaw–Curtis norm), §13.4 (this phase), §13.14 (known limitations), §13.15 (D5),
  §12 decision 30 and its amendments.
- `docs/CONVENTIONS.md` §12 (CGL nodes, the uniform grid, T_cu, the CC weights).
- `docs/phase5/README.md` (the learning grids s ∈ {16, 32, 64, 128}, D1, the transfer-only
  floor per s).
- `CLAUDE.md`: Working agreement, Checks, Build environment (`datagen`).

## Inputs

| Item | Value | Source |
| --- | --- | --- |
| Stage 3 dataset | n = 129, K = 64; 1000 train / 200 test; `datasets/poisson/poisson_n129_K64_{train,test}.npz` (main checkout) | Phase 4 T3 |
| Its label check (exact sine series), max | train 9.17e-13, test 8.97e-13 | Phase 4 T3 |
| The restriction R | unfiltered T_cu, `Transfers::with_s(n, s)` | Phase 5 D1, T2 |
| Transfer-only floor ‖P R u − u‖ / ‖u‖ at s = 16 / 32 / 64 / 128 (mean) | 3.03e-3 / 3.47e-4 / 2.45e-5 / 2.53e-6 | Phase 5 T1 |
| Code in place | `load_poisson_cheb_field` (raw f or u on the CGL grid, ungated); `clencurt` (CC weights, ungated); `chebyshev::transfer` (ungated); RLST 0.9 `svd` (LAPACK, behind `datagen`) | Phases 2–5 |

## Facts verified at `38cd1f2`

- **The population spectrum is known in closed form.** Each sine factor of §3.3 has unit
  mean square on [−1, 1], so the φ_kl are orthonormal in L²(Ω). The covariance of f has
  eigenvalues λ_kl / S_K on φ_kl; since u = (−Δ)⁻¹ f, u's are λ_kl / (S_K μ_kl²) on the
  same modes. Their traces are 4 for f (|Ω| × the unit expected mean square) and
  4.998e-2 for u. The leading values are 1.035, 0.441, 0.441, … for f and 4.25e-2,
  2.90e-3, 2.90e-3, … for u.
- **Degenerate eigenvalues.** λ_kl depends on k² + l² only, so (k, l) and (l, k) share an
  eigenvalue, and sums of two squares in more than one way give larger groups. Empirical
  modes within a group mix, so the closed-form check compares eigenvalues summed over each
  group, not individual modes.
- **Population energy ranks at K = 64 (computed from the closed form).** f: 38 / 418 /
  2175 modes for 90% / 99% / 99.9%. u: 2 / 6 / 17. The top 999 modes carry 99.64% of f's
  energy and all but 1.2e-8 of u's.
- **The sample limits f's rank.** The centred 1000-sample matrix has rank at most 999
  (design §13.4). f's population 99.9% rank (2175) is above that, so the empirical 99.9%
  rank of f measures N, not the field; it is reported with that caveat. f's empirical
  energy fractions are relative to what 999 directions hold. u's ranks are far below N
  and are not affected.
- **Sampling error.** An isolated empirical eigenvalue scatters about the population
  value with relative standard deviation about √(2/N) = 4.5% at N = 1000; trailing
  eigenvalues are biased once the effective dimension approaches N. The closed-form check
  is therefore made on the leading groups and on the trace.
- **Dynamic range.** u's leading and 999th population eigenvalues differ by 2.3e10, so
  its singular values span about 1.5e5. A thin SVD of the snapshot matrix keeps that in
  f64; the N × N Gram matrix (method of snapshots) squares it, so it is not used.
- **No ungated SVD exists.** The crate's dense linear algebra is RLST behind `datagen`.
  RLST 0.9 has `SingularValueDecomposition::svd(SvdMode::Compact)` and `singular_values`
  (LAPACK `gesdd`), and `SymmEig::eigh`. ndarray has none. *Found in T2:* macOS
  Accelerate's `dgesdd` returned wrong singular values (Σ σ² up to 44 times ‖M‖²_F) with
  `info = 0` when computing vectors, on u restricted to s = 16 and 32; `Pod` therefore
  calls LAPACK `gesvd` through RLST's `Gesvd` and checks Σ σ² = ‖M‖²_F on every fit. T1's
  CGL fits agree between the two to 1e-15, so its results stand.
- **Scale.** The weighted snapshot matrix of one field is 16641 × 1000 f64, 133 MB.
  `load_poisson_cheb_field` reads it as `[1000, 129, 129]`.

## Planning decisions

Recorded in design §12, decision 30 (amendment of 2026-10-27). Chosen by the author
(AHA-HH), 2026-10-27.

- **D5, inner product:** Clenshaw–Curtis-weighted on the 129 CGL grid: each row of the
  centred snapshot matrix is scaled by √(w_i w_j), so its singular values squared over
  N − 1 estimate the L² covariance eigenvalues that the closed form gives. Equal weights
  on the uniform learning grids (T2), scaled by the cell area (2/(s − 1))² so values
  compare across s; the energy fractions and ranks do not depend on that scale.
- **Backend:** RLST's thin SVD, so the phase's code sits behind `datagen` (as the
  generator does). No new dependency.
- **Code:** a small library module `metrics/pod` behind `datagen` (fit the POD, energy
  ranks, reconstruction error), with unit tests on synthetic data, and one example,
  `poisson_pod`, that loads the split, runs the analysis and writes the tables and plots.
  T2 extends both with the learning grids.
- **Data:** stage 3 only (the Phase 5 baseline stage). The basis is fitted on the 1000
  train samples; the 200 test samples are only projected (centred with the train mean).
- **Outputs:** `docs/phase6/results.md`, written by `poisson_pod` (Markdown, as `*.csv`
  is git-ignored); gnuplot plots attached to the PRs, not committed (`*.png` is ignored).
- **Fourier spectra (T2):** optional, as the design says; radially averaged power spectra
  on the uniform grids only if they fit in T2 without a new FFT path.

## Tasks

| Task | Brief | Delivers | Depends on |
| --- | --- | --- | --- |
| T1 | [T1-pod.md](T1-pod.md) | `metrics/pod` and `poisson_pod`: POD of f and of u on the 129 CGL grid, singular values, energy, ranks, reconstruction error on train and test; the closed-form check | – |
| T2 | [T2-pod-learning-grids.md](T2-pod-learning-grids.md) | The same at s ∈ {16, 32, 64, 128} after R; rank against learning resolution; optional spectra; phase close-out | T1, Phase 5 T2 (merged) |

One task per branch and PR: `phase6/T<k>-<name>`. Each task is worked from its brief
alone, plus the files it lists under "Read first".

## Module layout

- `metrics/pod.rs` (T1, behind `datagen`): `Pod` (the training mean, the singular values,
  the leading modes), its fit from `[N, a, b]` snapshots with per-axis weights, energy
  ranks, and per-sample reconstruction error against rank. f64 throughout.
- `examples/predict/poisson_pod.rs` (T1; T2 adds the learning grids): registered with
  `required-features = ["datagen"]`, like `generate_poisson`.
- No change to the loaders, the transfers, `runs/` or CONVENTIONS.

## Results

Filled in as tasks merge. Phase 6 is done when every row has a value.

| Item | Value | Source |
| --- | --- | --- |
| Closed-form check: leading eigenvalue groups and trace against λ_kl / S_K (f) and λ_kl / (S_K μ_kl²) (u) | Trace ratio f 0.991 (scatter 0.014), u 0.970 (0.038). Leading 10 groups of equal k² + l²: ratios 0.960–1.071 (f) and 0.934–1.055 (u), all within 2 scatters. Trailing empirical eigenvalues fall below the closed form beyond i ≈ 400 (f) and 200 (u), the bias expected near N | T1, `results.md` |
| Ranks for 90 / 99 / 99.9% energy, f and u, 129 CGL (train) | f 36 / 284 / 743 (population 38 / 418 / 2175; the 99.9% rank is limited by N − 1 = 999, and the 99% rank is low because the sample's energy counts only 999 directions); u 2 / 6 / 17 (population 2 / 6 / 17) | T1 |
| Reconstruction error against rank, train and test, f and u | Mean, train / test: f 0.546 / 0.547 at r = 10, 0.203 / 0.225 at r = 100, 0.063 / 0.122 at r = 500, 6e-15 / 0.099 at r = 999; u 8.7e-2 / 9.0e-2, 4.4e-3 / 4.8e-3, 3.1e-4 / 6.4e-4, 5e-15 / 3.5e-4 | T1 |
| Forcing ranks against solution ranks | u needs 2 modes for 90% where f needs 36, and 17 for 99.9% where f's exceeds N; see below | T1 |
| Ranks and reconstruction error at s = 16, 32, 64, 128 (after R) | 90 / 99 / 99.9% ranks, f: 34 / 143 / 188, 36 / 266 / 595, 36 / 284 / 741, 36 / 284 / 743 (CGL 36 / 284 / 743); u: 2 / 6 / 17 on every grid. At most min((s − 2)², 999) directions: 196 / 900 / 999 / 999. Leading 10 eigenvalues within 0.5% of the CGL fit (f, s = 16), equal to 4 digits for u everywhere; trace ratio f 0.9957 / 0.9996 / 1.0000 / 1.0000, u 1.0000. Mean test error at r = 100: f 0.193 / 0.223 / 0.225 / 0.225, u 4.6e-3 / 4.8e-3 / 4.8e-3 / 4.8e-3 | T2, `results.md` |
| How rank relates to learning resolution, with the Phase 5 transfer-only floor | u keeps its structure from s = 16 up; f's 90% rank is unchanged, its 99% and 99.9% ranks fall below s = 64, capped by the grid's interior points. The Phase 5 floor at small s is P's error, not structure R removes from u; see below | T2 |
| Fourier spectra (optional) | Skipped: ndarray has no FFT, and Burn's `rfft` would add a backend path to a host-only example | T2 |

**Forcings against solutions (T1).** u = (−Δ)⁻¹ f scales mode (k, l) by μ_kl⁻¹ and its
variance by μ_kl⁻², so u's spectrum is f's damped by a factor that grows with k² + l²:
relative to (1, 1) it is (2 / (k² + l²))², already 5.9e-3 at k² + l² = 26. The solutions
therefore sit in a far smaller subspace than the forcings: 2 modes hold 90% of u's energy
against 36 for f, and 17 hold 99.9% where f's population needs 2175, more than the
sample's 999 directions. The leading modes also generalise for u: at r = 100 the held-out
error is 4.8e-3 against 4.4e-3 on the fitting samples. f's held-out error is still 0.099
at r = 999, of the order of the 0.36% of its population energy beyond the top 999 modes
(√0.0036 = 0.06) plus the error in the estimated trailing modes. None of this is a
statement about the FNO (§13.4).

**Rank against learning resolution (T2).** R costs u nothing measurable. On every learning
grid u's ranks are 2 / 6 / 17, its leading ten eigenvalues and its trace equal the CGL
fit's to 4 digits, and its held-out error at r = 100 is 4.6e-3 at s = 16 against 4.8e-3
on the CGL grid: everything that carries u's energy is a low sine mode, which 16 points
resolve. So the Phase 5 transfer-only floor at s = 16 (3.03e-3) does not come from
structure R removes from u; Phase 5 T1 found it is P's own error. f is where resolution
shows. Its 90% rank is 34–36 on every grid, but its 99% and 99.9% ranks fall from
284 / 743 (s ≥ 64) to 266 / 595 at s = 32 and 143 / 188 at s = 16. f and u vanish on ∂Ω
and R keeps the boundary nodes, so the restricted fields live on the (s − 2)² interior
points, 900 at s = 32 and 196 at s = 16; at s = 16 196 modes reconstruct both splits to
2e-15. The energy is folded rather than lost: f's trace at s = 16 is 0.9957 of the CGL
one, although Phase 5 T1 measured 4.2% of f's energy above the s − 2 band, consistent
with an unfiltered R aliasing that band onto the coarse grid. s = 64 and 128 keep f's
structure as the CGL grid holds it. What this cannot show: whether the FNO, whose modes
are fixed Fourier modes truncated at min(12, s/2), uses any of this. A grid that holds
the data's structure is necessary for a model, not sufficient (§13.4).

## Exit checklist

- [ ] T1–T2 merged (T1 #76; T2 on merge).
- [x] The SVD results use training data only (the test split only projected).
- [x] Forcings and solutions analysed separately.
- [x] Rank/error curves generated.
- [x] The relationship between dataset rank and learning-grid resolution discussed.
- [x] The results labelled as dataset-structure analysis, not network performance.
- [x] D5 recorded in design §12 under decision 30.
- [x] Every row of Results filled in.
- [x] Any result that contradicts the design recorded in design §12, dated.
