# Thermo-nuclear code quality review: Phase 4s T14 (generator on the ungated stack)

- **Repo:** sciml-rs (`AHA-HH/sciml-rs`)
- **Branch:** `phase4s/T14-generator-ungated-stack` (HEAD `f91d64e`, PR #52)
- **Base:** `main` (`a3af111`, T13 merged)
- **Date:** 2026-10-22
- **Diff size:** 16 files, +180 / −117 (in `src` and `examples`: 13 files, +153 / −109)

## Verdict: APPROVE (with nits)

T14 does what its brief asks, and almost nothing else:
- the generator runs on `nodes` / `clencurt` / `cc_rel_l2_error(_weighted)` / `Transfers`;
- the RLST nodes, weights and norm are demoted to `#[cfg(all(test, feature = "chebyshev"))]` oracles, and their re-exports are gone;
- the closed form takes the plain name `nodes`;
- `label_check` loses its weights argument;
- `Split.y` is deleted while the file format stays the same.

No file crosses 1k lines and no branching is added. Two production call sites get shorter (`collocation.rs` and `sparse.rs` test helpers, `generate`). Decision 26 and the Results rows are filled in. The checks below pass.

The review found no structural regression. The one missed simplification is in test-only code: `norm.rs` is a pure-ndarray file that carries six tests of its own and is now an "oracle" for a second implementation of the same formula. Deleting it is the code-judo move. It is not a blocker, because it touches no production path and the brief prescribed demotion, not deletion. It should land in this PR or in T15/T16.

## Presumptive blockers

None.
- No file crosses 1000 lines (largest touched source file: `transfer.rs`, 602).
- No ad-hoc branching was added.
- No helper is duplicated in production code. One test is duplicated (finding 2.2).

## File-size table

| File | Before → after | Note |
| --- | --- | --- |
| `docs/design/2d-chebyshev-poisson-fno.md` | 1292 → 1309 | doc, already over 1k; decision 26 |
| `src/neural_operators/data/loaders/poisson.rs` | 719 → 719 | rename only; `NODE_TOL` widened to `pub(crate)` |
| `src/neural_operators/training/trainer.rs` | 732 → 732 | rename only |
| `src/neural_operators/chebyshev/transfer.rs` | 604 → 602 | |
| `src/neural_operators/pde/poisson/dataset.rs` | 435 → 479 | +2 tests (about +49) |
| `src/neural_operators/pde/poisson/collocation.rs` | 476 → 472 | |
| `src/neural_operators/pde/poisson/sparse.rs` | 440 → 438 | |
| `src/neural_operators/metrics/poisson.rs` | 384 → 384 | |
| `src/neural_operators/chebyshev/clencurt.rs` | 342 → 344 | |
| `src/neural_operators/chebyshev/norm.rs` | 122 → 126 | now test-only |
| `src/neural_operators/chebyshev/quadrature.rs` | 89 → 92 | now test-only |
| `src/neural_operators/chebyshev/points.rs` | 58 → 61 | now test-only |
| `src/neural_operators/chebyshev/mod.rs` | 66 → 62 | |
| All other touched files | < 250 | |

## Findings

### 1. Structural regressions

None. The ownership after T14 is cleaner than before:
- `chebyshev/` exports one set of grid primitives;
- the RLST pieces are visibly oracles;
- `pde/poisson/dataset.rs` imports only ungated names. Its one remaining gated dependency is `CollocationSolver`, which is T15's.

### 2. Missed code-judo simplifications

**2.1 `norm.rs` is not an RLST oracle. Delete it instead of demoting it** (`chebyshev/norm.rs:1-126`, `mod.rs:27-28`).
- `l2_norm` / `rel_l2_error` use ndarray and `check_n` only. Nothing in the file needs RLST or FFTW. It is gated behind `chebyshev` only because it used to sit beside the RLST weights.
- As a test oracle, it is a second, unscaled implementation of the formula that `cc_norm_sq` / `cc_rel_l2_error_weighted` already implement. Its only consumers are:
  - `cc_rel_l2_matches_gated` (`clencurt.rs:314`);
  - `rlst_oracle::fh_error` (`transfer.rs:555-560`);
  - its own six tests (`norm.rs:60-126`), four of which test the panics of a function that is now test-only.

  That makes about 70 lines of tests that test test code.
- The independent part of the oracle is the RLST *weights* (`quadrature::clenshaw_curtis`), not the sum Σ wᵢ wⱼ v². `clencurt_matches_rlst_weights` already pins the weights.

What the scaled implementation needs is an unscaled reference, and that is one expression in the test that uses it:

```rust
// clencurt.rs, cc_rel_l2_matches_gated
let naive = |v: ArrayView2<f64>| v.indexed_iter().map(|((i, j), x)| wx[i] * wy[j] * x * x).sum::<f64>();
let oracle = (naive((&a - &b).view()) / naive(b.view())).sqrt();
```

In `fh_error`, `cc_rel_l2_error_weighted(got, exact, w, w)` with `w = clenshaw_curtis(n)` replaces `rel_l2_error`. The oracle point (RLST weights) is preserved.

What this deletes:
- `norm.rs` (126 lines) and its `mod` line;
- one bullet in `mod.rs`'s docs and in CLAUDE.md "Layout";
- the cross-references in `clencurt.rs:7`, `:86`.

What this changes: `chebyshev --lib` goes 264 → 258 tests, all of them self-tests of the deleted oracle. No production behaviour changes. If the brief's "become oracles" wording is binding, do this in T15 or T16, which edit these lines anyway. T16 would otherwise have to decide whether to ungate a module that should not exist.

**2.2 `generator_nodes_within_node_tol_of_rlst` duplicates `cgl_closed_form_matches_nodes`, and widens two visibilities to do it** (`dataset.rs:423-433`; `transfer.rs:509-516`; `mod.rs:29` `pub(crate) mod points`; `loaders/poisson.rs:43` `pub(crate) const NODE_TOL`).
- The existing oracle test checks `nodes(n)` against `points::nodes(n)` for n ∈ {2, 3, 9, 33, 65, 129, 257} at `1e-14`.
- The new test checks the same pair for n ∈ {33, 65, 129, 257} at `NODE_TOL`, which is `1e-14`, so it is a strict subset.
- To reach the oracle and the constant from `pde/poisson/`, it makes `points` the only `pub(crate)` oracle module (`norm` and `quadrature` stay private) and exports a loader-private constant.

The brief asked for the test, but the brief did not know the transfer test exists. Remedy (pick one):
- Delete the new test and revert both visibilities. In the README Results row, record that `cgl_closed_form_matches_nodes` covers the deliverable. This is the direct option.
- Or keep the coupling to the loader tolerance, which is the part that is actually new. In `transfer.rs`, assert `err <= 1e-14` with a comment naming `NODE_TOL`. Do not reach for `NODE_TOL` itself, because `chebyshev` must not depend on `data::loaders`.

**2.3 `written_split_loads` re-does `files_round_trip`'s setup** (`dataset.rs:436-470` vs `:355-421`). Both tests `generate` a tiny split, `write_split` it to a pid-named temp dir and read it back. The new test adds the loader round trip, which is the useful part. Folding the `load_poisson_cheb` and `load_poisson_uniform` assertions into `files_round_trip` would remove one temp-dir dance and one `generate`. It would cost the second (test) split that `load_poisson_uniform` needs. That is a fair reason to keep the test separate, so this is optional.

### 3. Spaghetti / branching growth

None added.

### 4. Boundary, abstraction and type-contract problems

**4.1 `generate` uses two weight paths in one loop** (`dataset.rs:170`, `:191`, `:193-198`).
- `label_check` now builds `clencurt(n)` internally for each sample, through `cc_rel_l2_error`.
- Two lines later, `err2` uses the `w = clencurt(n)` hoisted at `:170` through `cc_rel_l2_error_weighted`.

Both are correct, and the cost is negligible next to a solve. But a reader sees one quantity obtained two ways. Remedy: call `cc_rel_l2_error(round_trip.view(), u_i.view())` for error 2 too, which deletes `w` and the `_weighted` import. That is one more O(n²) weight build per sample: n = 257 is about 66k flops against an O(n³) solve, and the result is bit-identical, because the weights are deterministic. The alternative, keeping `w` and passing it into `label_check`, is the signature the brief removed.

**4.2 `label_check`'s contract loosened silently** (`dataset.rs:129-139`).
- Before, it panicked unless both fields were `[w.len(), w.len()]`, which is square.
- Now `cc_rel_l2_error` accepts any matching rectangle, and `n = u_a.nrows()` is what gets reported and compared to `LABEL_CHECK_MIN_N`.

The `# Panics` text describes the new behaviour accurately. However, "n is read from their shape" hides that a non-square input would be judged by its row count. Callers only ever pass square fields. Either assert square (`assert_eq!(u_a.nrows(), u_a.ncols(), …)`, one line, restoring the old contract) or say "`[n, n]`; only the row count is read" in the docs. The assert is the better option.

### 5. File size and decomposition

No concerns. `dataset.rs` grows by tests only.

### 6. Modularity

**6.1 The oracle modules are reached in three different ways.**
- `transfer.rs:499-501`, `clencurt.rs:300`, `:315-316` and `norm.rs:58-59` use `super::super::{,super::}…::{norm, points, quadrature}::…`.
- `dataset.rs:425` uses `crate::neural_operators::chebyshev::points::nodes`.

With 2.1 and 2.2 applied, only `quadrature::clenshaw_curtis` and `points::nodes` remain, and only inside `chebyshev/`. The visibility can then go back to private `mod` for all three, with a single gate. No action is needed beyond 2.1 and 2.2.

### 7. Legibility

**7.1 `let nodes = nodes(n)` shadows the function** (`loaders/poisson.rs:145`, `examples/predict/poisson_plot.rs:128`). The rename turned two previously distinct names into a local that hides the function for the rest of the scope. It compiles and reads fine at these two sites, but the loader's closure at `:146-149` then uses `&nodes` meaning the array. Rename the locals to `x` (as `generate` and the tests do) or `cgl`.

**7.2 Unreflowed doc lines from the rename** (`clencurt.rs:87`, `norm.rs:15`, `dataset.rs:228`). Each edited line now runs past the surrounding wrap width (about 90 columns). rustfmt does not wrap comments, so reflow these three by hand.

**7.3 `write_split`'s comment** (`dataset.rs:251`) cites design §5.1 for "the file format keeps both axes". The useful fact for the next reader is that the loader checks `y == x` (`check_nodes`), so the duplicate is validated, not dead. Add a clause such as: "the loader checks both".

## Checks (run for this review)

- `cargo fmt -- --check`: clean.
- `cargo clippy --all-targets -- -D warnings`: clean.
- `cargo clippy --no-deps --features chebyshev --all-targets -- -D warnings`: clean.
- `cargo test --lib`: 209 tests, 208 passed, 1 ignored.
- `cargo test --features chebyshev --lib`: 264 tests, 263 passed, 1 ignored (106 s; the `sparse` tests dominate).
- `cargo doc --no-deps`: no warnings.
- Not run: the dataset acceptance (loading the stored stages, stage-1 regeneration). It needs `datasets/poisson/`, and the PR's README Results row reports it. The integration tests (`tests/`) and `--features metal` were also not run; nothing here is backend-sensitive.

## Next steps

Each step preserves behaviour and can land as its own commit. Order is by complexity deleted.

1. **Delete `norm.rs`** (2.1).
   - Inline the unscaled sum into `cc_rel_l2_matches_gated`.
   - Switch `rlst_oracle::fh_error` to `cc_rel_l2_error_weighted` with `clenshaw_curtis` weights.
   - Drop the `mod norm` line, the `mod.rs` doc bullet's norm half, the CLAUDE.md mention and the cross-references in `clencurt.rs`.
   - Verify: `cargo test --features chebyshev --lib` goes 264 → 258, all six removed tests from `norm::tests`; `cc_rel_l2_matches_gated` and `fh_degree_two_rate` pass.
   - Fits this PR, or T15/T16 if the brief's wording is held to.
2. **Remove the duplicate nodes test** (2.2).
   - Delete `generator_nodes_within_node_tol_of_rlst`.
   - Revert `pub(crate) mod points` → `mod points` and `pub(crate) const NODE_TOL` → `const`.
   - Optionally, comment the `1e-14` in `cgl_closed_form_matches_nodes` with "= the loader's `NODE_TOL`".
   - Update the Results row ("chebyshev 262 → 263").
   - Verify: both clippy runs, and `cargo test --features chebyshev neural_operators::chebyshev::transfer::tests::rlst_oracle::cgl_closed_form_matches_nodes -- --exact` (check that it runs 1 test).
3. **One weight path in `generate`, square assert in `label_check`** (4.1, 4.2).
   - Verify: `files_round_trip` and `label_check_aborts` pass.
   - Generate stage 1 into a scratch directory and diff the sidecar against the T14 scratch sidecar. Error 2 should be bit-identical.
4. **Legibility nits** (7.1–7.3): rename the shadowing locals, reflow the three doc lines and extend the `write_split` comment. Verify: `cargo fmt -- --check`, `cargo doc --no-deps`, `cargo clippy --no-deps --examples -- -D warnings`.

Steps 1 and 2 are independent. Step 3 is independent of both. None of them blocks T15.
