# T8: one owner for the shared Chebyshev pieces

Phase 4s, task 8. Branch `phase4s/T8-chebyshev-shared-pieces`. Depends on T1. Heads the
Chebyshev track (T8 → T14 → T15 → T16).
- Can run in parallel with T2, T3 and T5–T7.
- **Shared files with T4:** `data/loaders/poisson.rs`, `pde/poisson/dataset.rs` (a T4 test
  import) and `examples/predict/{poisson,poisson_plot}.rs`. The conflicts are import-level,
  so whichever merges second rebases.

## Read first
- Review §5 S4, §6 T6-3, T6-4, §7 step C1.
- `docs/CONVENTIONS.md` §12 (CGL grids, the transfers, `TRANSFER_DEGREE`). Design §12
  decisions 13 and 15.
- `chebyshev/transfer.rs` (`cgl_nodes` :51 is `pub(crate)`, re-exported at `mod.rs:39`),
  `chebyshev/clencurt.rs`.
- The four sites that rebuild the transfer recipe (s = n − 1, T_cu / T_uc(d = 2), applied
  on both axes):
  - `pde/poisson/dataset.rs:196-231`
  - `metrics/poisson.rs:109-143` (and the test helper `to_uniform` :258-266)
  - `data/loaders/poisson.rs:162-163` with `transfer_samples` :303-316
  - `examples/predict/poisson_plot.rs:104, 129-132`

## Goal
The feature gate forced copies of the same constant, type and recipe. Give each one a single
ungated owner in `chebyshev/`, and point every user at it.

## Deliverables
1. **`TRANSFER_DEGREE`, defined once:**
   - It lives in `chebyshev::transfer`.
   - Delete the copies at `pde/poisson/dataset.rs:44` and `metrics/poisson.rs:32` (and the
     comment at :29-31 saying it "cannot import").
   - Users: `predict/poisson.rs:30, 194` and `poisson_plot.rs:33, 129`.
2. **`Transfers::new(n)`** in `chebyshev::transfer`:
   - It holds `s = n − 1`, `t_cu` and `t_uc` (degree `TRANSFER_DEGREE`).
   - Methods: `to_uniform(&ArrayView2) -> Array2`, `to_cheb(…)` and `round_trip_error(…)`
     (CC-weighted, as `dataset.rs:221-231` computes it).
   - Its panics are those of `check_s` / `check_n`.
   - Migrate all four sites.
3. **Public nodes:**
   - Make `cgl_nodes` public under a name that won't collide with the gated
     `points::nodes`: `nodes_closed_form` is suggested. T14 picks the final name once the
     gated `nodes` is a test oracle.
   - Delete the private copy in `poisson_plot.rs:47-54`. It returns a `Vec`, so convert with
     `.to_vec()` at :136.
4. **`cc_norm_sq(v, wx, wy)`** in `clencurt.rs`:
   - `cc_rel_l2_error` calls it. It currently recomputes `clencurt(n)` twice per sample
     when the caller already has `w`, so add a variant, or a weights argument, that takes
     `w`.
   - `metrics/poisson.rs::boundary_errors` (:207-231) uses it for its area integral
     (:226-229). The edge line integrals stay.
5. **One `ErrorStats`.** The two copies behave differently:

   | | `pde/poisson/dataset.rs:70-92` | `metrics/poisson.rs:34-58` |
   | --- | --- | --- |
   | Derives | `Copy`, `PartialEq` | `Config` (written into `errors.json`) |
   | Field order | `max`, `mean` | `mean`, `max` |
   | `of` | private | `pub` |
   | Empty input | NaN (test `error_stats_propagate_nan` :344-351) | panics (:49) |

   - Keep **one** ungated type, in `chebyshev` or a small shared module that `pde` and
     `metrics` can both import.
   - Keep the serialised field names `mean` and `max`, so existing `errors.json` files and
     the sidecar still parse.
   - **Empty input:** recommended is to return NaN and document it, since an empty split is
     reported rather than fatal. Keep `error_stats_propagate_nan`, and change the panic test,
     if there is one, to the NaN expectation. State the choice in the PR.
   - Callers: the struct literal at `predict/poisson.rs:197-200`, and
     `predict/poisson_resolution.rs:92-95`.

## Tests
- **Add, ungated:**
  - `transfers_round_trip_matches_manual_recipe`: `Transfers` against the old inline
    recipe for n ∈ {9, 33}, bit-identical.
  - `cc_norm_sq_matches_cc_rel_l2_error`.
  - `public_nodes_match_gated`, gated `all(test, feature = "chebyshev")`, if no test
    already checks this.
- **Existing tests:** `error3_matches_lploss`, the transfer and metrics tests and the
  `dataset.rs` tests all pass.
- The test count does not drop.

## Acceptance
- `predict_poisson` re-run on `runs/poisson_fno_p4_1791922918` prints the same five errors
  as before. This needs the stage 1 datasets; if they aren't there, say so.
- `poisson_plot` produces the same figure for sample 0.

## Checks to run and report
The six from the README **plus the chebyshev pair**. Give the line delta.

## Do not
- Switch the generator onto the ungated nodes, quadrature or norm (T14), or touch the gated
  `points` / `quadrature` / `norm` / `differentiation` modules.
- Change the generated data or the transfer degree.
