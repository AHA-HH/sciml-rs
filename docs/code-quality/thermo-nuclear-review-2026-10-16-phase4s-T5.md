# Thermo-nuclear code quality review: phase4s/T5

| | |
|---|---|
| Repo | sciml-rs (`sciml_rs`), worktree `chiton` |
| Branch | `phase4s/T5` (task brief: `docs/phase4s/T5-delete-grids.md`) |
| Base | `79f9c62` (`git merge-base HEAD main`, the tip of `main`) |
| Date | 2026-10-16 |
| Diff | 4 files, +41 / −381 (uncommitted working tree against base, including the staged deletion of `data/grids.rs`) |

## Verdict: **APPROVE**

The diff is net deletion: 340 lines. It removes a 361-line public module that production
never called (`GridPlacement`, `grid_from_axes`, `uniform_grid`, `append_grid`,
`linspace_excluding_start`) and its 11 tests. In its place is a ~45-line oracle that is private
to the test module and does only what the FNO tests use. The new code adds no production
paths, flags or casts. The oracle copies f64 values exactly as the old module did, so the
bit-for-bit grid tests compare against the same numbers. `fno.rs` was already over 1,000 lines
and grows by 25. That is a real smell, but the brief forbids splitting the tests here and
assigns the split to T20, so it is recorded, not blocked. The remaining findings are small
tidy-ups within the lines this PR already touches.

**Status:** F2 and F3 are applied in this PR. F4 is left as an optional nit, and F1 is
deferred to T20.

## Presumptive blockers

None met.
- No incidental complexity kept where a code-judo move exists. Deleting the module is the
  judo move: it removes a general-purpose API with five items, an enum and its own test
  suite, all kept alive only as a test fixture.
- No file crosses 1,000 lines. `fno.rs` was already above it (1569 → 1594); see F1.
- No ad-hoc branching in shared code. The only new branch (`checked_sub` in `with_grid`)
  is in a test helper and replaces the old `GridPlacement` match.
- No scattered feature checks, no new public abstraction, no casts.
- No duplicated helper. With `grids.rs` gone there is no canonical host grid builder left
  to reuse, and `FNO::grid_cl` must not be the oracle for itself.

## File sizes

| File | Before | After | Note |
|---|---:|---:|---|
| `src/neural_operators/data/grids.rs` | 361 | 0 | deleted |
| `src/neural_operators/models/fno.rs` | 1569 | 1594 | already > 1k; +25 (F1) |
| `docs/CONVENTIONS.md` | 334 | 331 | |
| `src/neural_operators/data/mod.rs` | 14 | 13 | |

## Findings

### 1. Structural code-quality regressions
None.

### 2. Missed opportunities for dramatic simplification
None at the scale of this task. I tested two larger reframings and rejected both:
- **Make `with_grid` = `Tensor::cat([x, grid_from_host(grids)])`.** This would drop the f64
  round-trip of `x` and the data-or-grid branch. But the reference would then depend on
  Burn's `cat`, the same op whose fused-backend bugs shaped `grid_cl`
  (CONVENTIONS §2, `fno.rs:216-219`). A reference built entirely on the host is worth its 10
  lines when the tests run on Metal. Keep it.
- **Represent a grid as per-channel `(axis, Array1)` pairs instead of full `ArrayD`s.** This
  would delete `ij_grid` because nothing would be materialised. But `check_grid_values`
  (`fno.rs:1330-1334`) and the `zeros` control (`:1424-1427`) index full arrays, so every
  caller changes. That breaks the brief's "callers don't change" and moves the complexity
  instead of deleting it. Not worth it.

### 3. Spaghetti / branching complexity
None.

### 4. Boundary / abstraction / type-contract problems

**F2: the "ij grid, then reverse" step is written twice.** At `fno.rs:1248-1251`
(`old_grid_f64`) and `fno.rs:1428-1435` (the `[0, 1)` control). The model's convention is
the *reversed* order, and the second copy repeats that rule by hand. Make the convention the
helper and the controls the exceptions:

```rust
/// `axes` as model-order grid channels: 'ij' grids in reverse axis order (CONVENTIONS §2).
fn xy_grid(axes: &[Array1<f64>]) -> Vec<ArrayD<f64>> {
    let mut grid = ij_grid(axes);
    grid.reverse();
    grid
}

fn old_grid_f64(spatial: &[usize]) -> Vec<ArrayD<f64>> {
    xy_grid(&uniform_axes(spatial))
}
```

The controls then read as a table of their differences from the correct grid:

```rust
let half_open = xy_grid(&spatial.iter()
    .map(|&n| Array1::from_shape_fn(n, |i| i as f64 / n as f64)).collect::<Vec<_>>());
// …
wrong.push(("'ij' order", ij_grid(&uniform_axes(spatial))));
```

This is about −4 lines, every line of it already touched by this diff, and no behaviour
change.

### 5. File-size and decomposition concerns

**F1: `fno.rs` is 1594 lines, and this PR adds 25.** The test module is most of the file. The
oracle replaces a one-line import, so some growth here can't be avoided. The brief rules out
splitting the tests (T20 owns that), so this is not a blocker. When T20 lands, the oracle
(`ij_grid`, `uniform_axes`, `old_grid_f64`, `with_grid`, `with_old_grid`) should move as one
unit into the grid test file. It is self-contained, needing only `HostFloat` and ndarray, so
the move should be mechanical.

### 6. Modularity and abstraction
No issues. Each helper has more than one caller or a clear name for one concept: `ij_grid`
has 3 call sites, `uniform_axes` 2, `old_grid_f64` 4, `with_grid` 2. `HostFloat::from_f64`
is kept on purpose so that the single f64→f32 rounding is visible, as the brief requires.

### 7. Legibility and maintainability

**F3: `with_grid`'s doc comment still says "the way the loaders did it".** That is at
`fno.rs:1254-1256`. After this PR there is no loader code or host module left to point to,
so a reader can't check the claim. Say what the helper does instead: "appended after the
data in f64, then rounded to f32 once (`HostFloat::from_f64`)". Keep the pre-#13 history on
`with_old_grid`, where it explains why that helper exists.

**F4 (docs): the new wording at CONVENTIONS §12 (`:283`) is correct, but it points at the
right section only indirectly.** "as in §2 (x_a varies along axis a)" is accurate: §2 defines
x_a along axis a, before it reverses the channel order. A reader may still mix up §2's
*channel* order ('xy') with §12's *storage* order ('ij'). Optional: "the first spatial axis is
x and the second is y (§2's axis a = 1, 2; §2's channel order is reversed, the storage order
is not)". This is a nit; take it or leave it.

## Next steps

1. **F2: add `xy_grid` and use it in `old_grid_f64` and the `[0, 1)` control** (in this
   PR). Behaviour-preserving: run
   `cargo test neural_operators::models::fno::tests::grid_ -- ` and check that
   `grid_matches_old_loader_grid` and all four `grid_controls_*` still pass, with the same
   rel-err lines printed under `--nocapture`.
2. **F3: rewrite `with_grid`'s doc comment** (in this PR, doc only). Verify with
   `cargo doc --no-deps` and `cargo fmt -- --check`.
3. **F4: optional rewording of CONVENTIONS §12** (in this PR or T24's docs pass). Prose only,
   no `CONVENTION_VERSION` bump.
4. **F1: move the oracle with the grid tests in T20.** This depends on this PR merging
   first, as the T5 brief already says.
