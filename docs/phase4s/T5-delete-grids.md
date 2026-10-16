# T5: `data/grids.rs` deleted

Phase 4s, task 5. Branch `phase4s/T5-delete-grids`. Depends on T1.
- Can run fully in parallel with any task except T20, which splits the same `fno.rs` tests
  and must come after this one.
- The `data/mod.rs` line it removes may need a one-line rebase against T11 or T23.

## Read first
- Review §5 S2 ("Model grid" row), §6 T2-5, §7 step A4.
- `docs/CONVENTIONS.md` §2. Its last bullet and the "Verification" list name `data::grids`.
- `src/neural_operators/data/grids.rs` (361 lines) and `models/fno.rs`'s test helpers
  `old_grid_f64` :1234-1238, `with_grid` :1243-1256, `with_old_grid` :1259-1262.

## Goal
`FNO::grid_cl` is the only grid in production. `data/grids.rs` survives only as an f64 test
oracle for it. Move just that oracle into the tests and delete the module.

## Deliverables
1. **A private f64 oracle in `fno.rs`'s test module**, about 25 lines. It reproduces exactly
   what `uniform_grid(&[(0,1); D], spatial)` + `reverse()` + `append_grid(…, AfterData)` build
   today: x_a[j] = j/(s_a − 1), and 0 when s_a = 1 (CONVENTIONS §2). Keep the single
   f64 → f32 rounding through `HostFloat::from_f64`. The bit-for-bit comparisons depend on
   it.
2. **Rewire the test helpers:** `old_grid_f64`, `with_grid` and `with_old_grid` use the
   inlined oracle. Their callers (:580, 663, 926, 985, 1085, 1203, 1302, 1384-1396, 1460,
   1502, 1560) don't change. The `use … grids::{…}` at about :1227 goes.
3. **Delete** `src/neural_operators/data/grids.rs` and `pub mod grids;` (`data/mod.rs:10`).
   Nothing in production calls it: `GridPlacement`, `grid_from_axes`, `uniform_grid`,
   `append_grid` and `linspace_excluding_start` are used only by tests.
4. **CONVENTIONS §2, prose only:**
   - Delete the bullet "Host-side `data::grids` helpers …".
   - Remove `data::grids` tests (`reversed_uniform_grid_2d_matches_numpy_meshgrid`) from
     "Verification".
   - No `CONVENTION_VERSION` bump, because the convention itself doesn't change. Say so in
     the PR.
5. CLAUDE.md "Layout", if it mentions `grids`.

## Tests
- Every `fno.rs` grid test (`grid_2d_order_is_pinned`, `grid_controls_*`,
  `grid_matches_old_loader_grid`, `resolution_change_builds_that_grid`,
  `batch_rows_see_identical_grids`) passes unchanged against the inlined oracle.
- **Deliberate deletions:** the tests inside `grids.rs`. List them in the PR with the count.
- Optional: keep the meshgrid-order test by porting `reversed_uniform_grid_2d_matches_numpy_meshgrid`
  onto the oracle, if it still checks something `grid_2d_order_is_pinned` doesn't.

## Acceptance
`git grep -n 'data::grids\|grids::'` finds nothing in `src/`, `examples/`, `tests/` or
`docs/CONVENTIONS.md`.

## Checks to run and report
The six from the README. The chebyshev pair is not needed. Give the line delta.

## Do not
- Change `FNO::grid_cl` or the grid convention.
- Split or reorganise the `fno.rs` tests beyond the helpers (T20), or retire
  `with_old_grid` (T20 decides).
