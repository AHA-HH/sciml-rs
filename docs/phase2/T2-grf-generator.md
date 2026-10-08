# T2: GRF sampler and `generate_poisson`

Phase 2, task 2. Branch `phase2/T2-grf-generator`. Depends on T1 (transfers, for error 2)
and Phase 1 T2 (solver A).

## Read first
- Design §3.3 (forcing family: KL expansion, normalisation, nested draws, K(n), the
  explicit override, ε_K), §4.3 (A on GRF), §5.1 (generation, label check), §5.3 (stages,
  sizes, the K = 32 evaluation sets), §7 (error 2), §11 (rows for A on GRF, the GRF
  sampler and the label check), §12 decisions 3, 10 and 13.
- `docs/CONVENTIONS.md` §12.
- `spikes/transfers/REPORT.md` §5 and Setup; `spikes/transfers/src/grf.rs` (`lambda`,
  `s_k`, `s_inf`, `eps_k`, `draw_xi`, the sine basis, the series solution).
- `src/neural_operators/pde/poisson/collocation.rs` (`CollocationSolver::new`,
  `solve_full`); `src/neural_operators/chebyshev/` (`nodes`, `clenshaw_curtis`,
  `rel_l2_error`, and T1's `transfer`).
- `src/neural_operators/data/io/readers/npz.rs` (`NpzFileReader`, used to read the files
  back) and `datasets/README.md`.

## Goal
Reproducible Poisson datasets on the Chebyshev grid. Each sample is a GRF forcing f and
its solver-A solution u, every label is checked against the exact sine-series solution,
and each split records its own transfer error.

## Deliverables
1. **`Cargo.toml`.** Add `rand_chacha`, `rand_distr` and `serde_json` as optional
   dependencies, each pinned with `=` to the current release (`Cargo.lock` is
   git-ignored). Add them to the `chebyshev` feature, and update the comment above it.
   Register the example with `required-features = ["chebyshev"]`.
2. **`src/neural_operators/pde/poisson/grf.rs`**, behind `chebyshev` like the rest of
   `pde`:
   - Constants τ = 3, α = 2 and K_MAX = 64, the largest production truncation.
   - `mu(k, l)`, `lambda(k, l)`, `s_k(k)`, and `eps_k(k)` (the unresolved energy).
     S_∞ is a documented constant from the spike (4.984158e-3, k, l ≤ 3000 plus the
     integral tail), with a test that recomputes it.
   - `k_default(n) = min((n − 1)/2, 64)`.
   - The draw: ξ ~ N(0, 1) from `ChaCha8Rng::seed_from_u64(seed)` and `StandardNormal`,
     row-major over a K_MAX × K_MAX block, index `[k − 1, l − 1]`. A truncation K uses
     the leading K × K block, so draws are nested (design §3.3). Document the stream in
     the doc comment; it is part of the dataset format.
   - Forcing coefficients c_kl = ξ_kl λ_kl^½ / √S_K for k, l ≤ K, and series-solution
     coefficients c_kl / μ_kl.
   - Evaluation on given 1D node arrays, as S_x C S_yᵀ with S[i, k] = sin(kπ(x_i + 1)/2):
     O(K n² + K² n), not O(K² n²). The result is `[n_x, n_y]`, 'ij'.
   - A K above K_MAX panics.
3. **Generation**, a library function in `pde::poisson` so the tests can run it at small N:
   - Input: a split spec (n, an optional explicit K, the base seed, the sample count).
   - For each sample i (seed base + i): evaluate f, run `CollocationSolver::solve_full`
     (one solver per split), and evaluate the series solution.
   - **Label check:** relative CC-L² ‖u_A − u_series‖ / ‖u_series‖ per sample, max and
     mean per split. For n ≥ 65 it returns an error naming the sample if any value
     exceeds 1e-8. For n < 65 it records the value without checking it (design §5.1).
   - **Error 2:** relative CC-L² of T_uc(T_cu u_A) − u_A with d = 2 and s = n − 1 (T1),
     max and mean per split. This is recorded only; the transferred fields are not
     stored.
   - Output: `f`, `u` as `[N, n, n]` f64, the nodes, and the two statistics.
4. **`examples/generate_poisson.rs`**:
   - Hyperparameter literals at the top: the dataset list. By default it holds stage 1
     (n = 33) and stage 2 (n = 65), each with 1000 train and 200 test samples, K(n), and
     fixed, distinct train and test base seeds. Stages 3 and 4 and the K = 32 evaluation
     sets 3e and 4e (n = 129 and 257, K = 32, the stage 2 test seeds, 200 samples) are
     commented entries; you enable one by editing the list.
   - Output in `datasets/poisson/` (git-ignored), one `.npz` and one `.json` per split.
     Name each file after n, K and the split, so 3e is distinct from 3.
   - `.npz` fields: `f`, `u` (`[N, n, n]`, f64), and `x`, `y` (the CGL nodes, `[n]`),
     written with `ndarray-npy`.
   - JSON sidecar (design §5.1), written with `serde_json`:
     - n and s;
     - K, and whether it was set explicitly;
     - τ, α and K_MAX;
     - the RNG name;
     - the base seed and the count;
     - ε_K;
     - the label check (max, mean, and whether it was enforced);
     - error 2 (max, mean, d, s);
     - `CONVENTION_VERSION`, the crate version, and the git commit from
       `git rev-parse HEAD` ("unknown" if git is unavailable);
     - the wall time.
   - Prints a one-line summary per split.
5. Document shapes, errors and panics of every public item, citing design §3.3 and §5.1
   and `CONVENTIONS §12`. A Layout line in `CLAUDE.md` for `grf` and the generator; a
   Poisson section in `datasets/README.md` (how to generate the files, their layout and
   sizes).

## Tests (`#[cfg(test)] mod tests`, gated)
- `same_seed_same_field_on_common_nodes`: with a fixed K = 8, the field at n = 17 equals
  the field at n = 33 and 65 on the nodes they share (every 2nd and every 4th), to 1e-14.
- `draws_are_nested`: the K = 32 coefficients are the leading block of K = 64 up to the
  ratio √(S_64 / S_32).
- `eps_k_matches_design_table`: ε_K for K = 16, 32, 64, 128 matches design §3.3 to two
  significant digits. S_∞ is recomputed to 1e-4 relative at a cutoff that keeps the test
  under a few seconds in debug.
- `mean_square_is_one`: Parseval mean square ¼ Σ c_kl² over N = 1000 seeds equals 1
  within 4 standard errors (Phase 0: 0.983 ± 0.013).
- `point_covariance_matches_formula`: the empirical covariance of f at three fixed node
  pairs, at n = 17 and K = 8 with N = 2000, against (1/S_K) Σ λ_kl φ_kl(p) φ_kl(q) (φ the
  sine products). The tolerance is 5 standard errors, estimated from the same samples.
  State the numbers in the PR.
- `collocation_matches_series_on_grf` (design §11, moved from Phase 1 by decision 13):
  A against the exact series, relative CC-L², on five seeds at n = 33, 65, 129, with
  K(n). The bound is ≤ 1e-8 for n ≥ 65; record n = 33 (Phase 0: 2.5e-7) without a bound.
- `label_check_aborts`: a split at n = 65 whose label is corrupted, by a test hook or by
  calling the check function on a perturbed u, returns the label-check error.
- `files_round_trip`: a two-sample split at n = 17 written to a fresh directory under
  `std::env::temp_dir()`. `NpzFileReader` reads back `f`, `u`, `x`, `y` with the shapes
  and values written, and `serde_json` reads back the sidecar fields.

Keep n ≤ 129 and the sample counts small; the tests run in debug inside the existing
budget (109.5 s for the chebyshev suite).

## Acceptance
- All tests pass under `cargo test --features chebyshev` on macOS and in the
  `run-tests-chebyshev` CI job; the default build compiles without them.
- Stage 1 and 2 datasets generated locally with
  `cargo run --release --features chebyshev --example generate_poisson`, with the label
  check enforced and passing at n = 65.
- The T2 rows of Results in `docs/phase2/README.md` are filled in, including error 2 for
  stages 1 and 2.

## Checks to run and report
All of `CLAUDE.md`'s CI checks and `cargo clippy --all-targets -- -D warnings`, then the
same clippy and test commands with `--features chebyshev`, plus
`cargo clippy --no-deps --features chebyshev --examples -- -D warnings`. Give the exact
test filters and check the "running N tests" lines. The generator is CPU-only, so
`--features metal` is not relevant; say so.

## Do not
- Use solver C for labels, or add it to any data path (design §4.2).
- Store transferred (uniform-grid) fields in the dataset files (design §5.2).
- Write the loader (T3) or any training code.
- Use `rand::rngs::StdRng` or an unpinned RNG crate.
- Change solver A or T1's transfers beyond a bug fix stated in the PR.
