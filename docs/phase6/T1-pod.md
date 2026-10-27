# T1: POD of f and u on the CGL grid

Phase 6, task 1. Branch `phase6/T1-pod`. Depends on nothing in this phase. Needs the
stage 3 dataset (`datasets/poisson/poisson_n129_K64_{train,test}.npz`) from the main
checkout, by copy or symlink; nothing about that is committed. Builds with `datagen`.

## Read first
- Design §3.3 (λ_kl, μ_kl, S_K), §3.4 (the CC norm), §13.4 (this phase).
- `docs/phase6/README.md` (facts verified, planning decisions, D5).
- `src/neural_operators/data/loaders/poisson/cheb.rs` (`load_poisson_cheb_field`,
  `PoissonField`), `chebyshev/clencurt.rs` (`clencurt`), `metrics/stats.rs`.
- RLST 0.9 `SingularValueDecomposition` (`svd`, `SvdMode::Compact`) and how the crate
  converts to and from RLST arrays today (`pde/poisson/collocation.rs`).

## Goal
Measure how many directions the stage 3 forcings and solutions occupy, on the 129 CGL
grid in the L² inner product, and check the measurement against the closed-form
spectrum of §3.3.

## Deliverables
1. **`metrics/pod.rs`, behind `datagen`:**
   - `Pod::fit(snapshots: [N, a, b], wx: [a], wy: [b], keep: usize)`: subtract the mean
     over the N samples, scale each grid point by √(wx_i wy_j), take RLST's thin SVD of
     the `[a·b, N]` matrix. Keeps the mean, all singular values and the first `keep` left
     singular vectors (the weighted modes).
   - `eigenvalues()`: σ_i² / (N − 1), the covariance eigenvalues in the weighted inner
     product.
   - `energy_rank(fraction)`: the smallest r with Σ_{i ≤ r} σ_i² ≥ fraction · Σ σ_i².
   - `reconstruction_error(samples: [M, a, b], r)`: per sample, ‖v − v_r‖ / ‖v‖ in the
     weighted norm, where v is centred with the fitted mean and v_r its projection on the
     first r modes. On the fitting samples, the squared absolute errors ‖v − v_r‖² summed
     over the samples equal the tail energy Σ_{i > r} σ_i² (a test); on other samples it is the held-out
     error.
   - Every public item documented with tensor shapes and panics.
2. **`examples/predict/poisson_pod.rs`** (`required-features = ["datagen"]`), run as
   `poisson_pod -- <out.md>`:
   - Loads the stage 3 train and test f and u with `load_poisson_cheb_field`; CC weights
     `clencurt(129)` on both axes (D5).
   - Fits one `Pod` for f and one for u on the train split only.
   - Writes Markdown to `<out.md>`: the leading 20 eigenvalues of each field; the ranks
     for 90 / 99 / 99.9% energy (f's 99.9% flagged as limited by N, README facts); mean
     and max reconstruction error at r ∈ {1, 2, 5, 10, 20, 50, 100, 200, 500, 999} on
     train and on test; the closed-form check (below); wall time of each fit.
   - Plots with gnuplot beside `<out.md>` (skipped with a warning if missing):
     singular-value decay (both fields, log scale, with the closed-form √((N − 1)·λ)
     curve), cumulative energy against rank, reconstruction error against rank (train and
     test).
3. **The closed-form check**, in the example: the population eigenvalues of README's
   facts (λ_kl / S_K for f, λ_kl / (S_K μ_kl²) for u, k, l ≤ 64, τ = 3, α = 2), sorted and
   **grouped by k² + l²**. Compare the sum of empirical eigenvalues in each of the
   leading 10 groups with the group's population sum, and the empirical trace with the
   population trace (4 and 4.998e-2). State the expected scatter, about √(2/(N·g)) for a
   group of g modes, and report each ratio. A ratio far outside that scatter is reported
   and investigated before the results are used, not tuned away.
4. **`docs/phase6/results.md`**, the example's output on the real data, committed; and
   the T1 rows of `docs/phase6/README.md` Results filled in, with the ranks of f against
   u discussed in one paragraph (u = (−Δ)⁻¹ f damps mode (k, l) by μ_kl⁻²).

## Tests (in `metrics/pod.rs`, `datagen`)
- A synthetic rank-3 field set (three fixed smooth fields with random coefficients, plus
  the mean) gives exactly 3 nonzero singular values (the rest ≤ 1e-12 relative), energy
  rank 3 at 99.9%, and reconstruction error ≤ 1e-12 at r = 3 on fresh samples of the same
  family.
- Weights: on the CGL grid with CC weights, a field set built from two known L²-orthonormal
  sine modes with variances 4 and 1 gives eigenvalues near those (within the sampling
  tolerance for the chosen N), and the trace matches the sample variance sum to 1e-12.
- `reconstruction_error` at r = 0 is 1 for every centred sample and decreases with r.
- Shape and weight-length mismatches panic with a named message.

## Acceptance
- The POD uses the train split only; the test split is only projected.
- f and u analysed separately; every number in the Markdown comes from the example.
- The closed-form check run and reported, with the trace ratios within the stated
  scatter or the discrepancy explained.

## Checks to run and report
All of `CLAUDE.md`'s CI checks and `cargo clippy --all-targets -- -D warnings`; the
example's run (`cargo run --release --features datagen --example poisson_pod -- …`) with
its wall time. Not backend-sensitive (no Burn tensors): no metal run.

## Do not
- Use the test split in the fit, or the N × N Gram matrix (README: dynamic range).
- Train or evaluate a model, or claim anything about network performance.
- Change the loaders, the transfers or CONVENTIONS.
