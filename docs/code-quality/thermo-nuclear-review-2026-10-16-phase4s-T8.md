# Thermo-nuclear code quality review: phase4s/T8

- **Repo:** sciml-rs
- **Branch:** `phase4s/T8` (uncommitted working tree)
- **Base:** `79f9c62` (merge-base with `main`)
- **Date:** 2026-10-16
- **Diff:** 12 files, +366 / −195 (prod +86, test +85)
- **Scope:** Phase 4s T8, "one owner for the shared Chebyshev pieces" (`docs/phase4s/T8-chebyshev-shared-pieces.md`)

## Status after fixes (2026-10-16)

- **B1 / 1.1 and 4.1: fixed.**
  - `Transfers::MIN_N = TRANSFER_DEGREE + 2`, and `new` checks it with a single assert.
  - The stored `s` field is gone; `s()` is now `t_cu.nrows()`.
  - The uniform loader checks against `Transfers::MIN_N`. `poisson_errors` dropped its
    own assert and relies on `Transfers::new`.
  - The two panic tests are merged into `transfers_panic_below_min_n`, a deliberate −1
    test.
- **B2 / 1.2: fixed by deleting `Transfers::round_trip_error`.**
  - Metrics computes the round trip from the `u_uni` it already has, so T_cu is applied
    once per sample again.
  - This departs from the T8 brief, which listed the method. A version taking `u_uni`
    would have carried an unchecked contract (`u_uni == to_uniform(u)`).
  - T14 gives the generator the same two-line expression.
- **2.1: deferred to T22.** It is added to `docs/phase4s/T22-poisson-loader-split.md` as
  optional deliverable 6.
- **4.2: not done.** The CGL loader still accepts n = 3. Changing that changes what it
  accepts, which is out of scope here.
- **Verification:**
  - The six README checks and the chebyshev pair pass.
  - Tests: 227 passed by default, 282 with `--features chebyshev`. That is 228 and 283
    minus the merged test.
  - `predict_poisson`: `errors.json` and stdout are byte-identical to the pre-T8
    baseline.
  - `poisson_plot` sample 0: the PNG is byte-identical.

## Verdict (before the fixes): BLOCKED (small fixes)

The direction is right, and it delivers what the brief asked for. Four copies of the
transfer recipe are now one `Transfers`, two `ErrorStats` are now one, the constant is
defined once, and the outputs are bit-identical (same `errors.json`, same PNG). Two
problems the diff itself introduced keep it from approval:

1. **The minimum-n rule now lives in three places with three different messages.**
   `Transfers::new` should own it, but it doesn't.
2. **`Transfers::round_trip_error` fits neither of its natural callers.** Metrics, its
   only caller, now applies T_cu twice per sample. The generator can't use it at all
   until T14.

Both fixes are small (about 30 lines in total) and delete code rather than add it.

## Presumptive blockers

| # | Condition | Evidence |
| --- | --- | --- |
| B1 | Scattered checks in shared code: the rule "n ≥ TRANSFER_DEGREE + 2" is spread over three sites | `metrics/poisson.rs:79-82` (assert), `data/loaders/poisson.rs:154-158` (`Err`), `chebyshev/transfer.rs:203-211` (no explicit check: it relies on `check_n`, a `check_s` that only catches n = 2, and the panic inside `uniform_to_cheb`, whose message names a different function) |
| B2 | An abstraction that doesn't fit its users: `round_trip_error` recomputes T_cu, which its only caller already has | `metrics/poisson.rs:101` computes `u_uni = to_uniform(ui)`; then `metrics/poisson.rs:106` calls `round_trip_error(ui, …)`, which runs `to_uniform(ui)` again (`transfer.rs:249`). The generator skips the method (`dataset.rs:193-195`) and rebuilds the round trip inline |

## File sizes

| File | Before | After | |
| --- | ---: | ---: | --- |
| `examples/predict/poisson.rs` | 271 | 275 | |
| `examples/predict/poisson_plot.rs` | 197 | 186 | |
| `examples/predict/poisson_resolution.rs` | 122 | 125 | |
| `chebyshev/clencurt.rs` | 283 | 342 | |
| `chebyshev/mod.rs` | 62 | 66 | |
| `chebyshev/transfer.rs` | 510 | 624 | largest growth; still well under 1k |
| `data/loaders/poisson.rs` | 748 | 750 | the largest touched file; T22 is meant to split it |
| `metrics/mod.rs` | 8 | 10 | |
| `metrics/poisson.rs` | 445 | 408 | |
| `metrics/stats.rs` | 0 | 71 | new |
| `pde/poisson/dataset.rs` | 476 | 436 | |
| `training/trainers/poisson.rs` | 135 | 135 | |

No file crosses 1000 lines.

## Findings

### 1. Structural regressions

**1.1 `Transfers::new` doesn't own its own invariant (B1).** `transfer.rs:203-211`

The type exists so that callers stop rebuilding the production transfer recipe. But each
caller still has to know the recipe's precondition, n ≥ `TRANSFER_DEGREE + 2`:

- `poisson_errors` asserts it, with its own message.
- The uniform loader turns it into an `Err`, again with its own message.
- `Transfers::new` itself never states it. `check_s` is effectively dead code: for
  n ≥ 2 it only rejects n = 2. n = 3 then panics inside `uniform_to_cheb` with
  "uniform_to_cheb: d must be <= s - 1", which names the wrong function for someone
  who called `Transfers::new`.

The doc comment even explains the bound in prose: "so n must be at least 4". This works,
but it's the kind of scattering the type was supposed to end.

**Remedy:** make the bound part of the type.

```rust
impl Transfers {
    /// The smallest n: T_uc of degree TRANSFER_DEGREE needs s = n − 1 > d.
    pub const MIN_N: usize = TRANSFER_DEGREE + 2;

    /// # Panics
    /// If `n < Transfers::MIN_N`.
    pub fn new(n: usize) -> Self {
        assert!(n >= Self::MIN_N, "Transfers::new: n must be >= {}, got {n}", Self::MIN_N);
        let s = n - 1;
        Self { t_cu: cheb_to_uniform(n, s), t_uc: uniform_to_cheb(s, n, TRANSFER_DEGREE) }
    }
}
```

- The loader then checks `if n < Transfers::MIN_N` and builds its message from the
  constant.
- `poisson_errors` drops its own assert and lets `Transfers::new` panic. Its panic test
  `too_small_n_panics` expects "n must be >= 4", which the new message still contains.
- Also delete `transfers_panic_on_s_below_two`, which tests the dead `check_s` path, and
  keep one panic test for `MIN_N`.

**1.2 `round_trip_error` recomputes work its only caller already did (B2).** `transfer.rs:248-251`, `metrics/poisson.rs:101-106`

Before this diff, metrics applied T_cu once per sample and reused `u_uni` for both error 2
and error 3. Now it applies T_cu twice. The cost is negligible at n = 65, but it's a
design smell: the method is shaped like the generator's call site, which can't use it
yet (T14), not like the one caller that does.

**Remedy (code judo):** let the method take a field that is already transferred, so it
uses what the caller has.

- Change the signature to `round_trip_error(&self, u_uni: ArrayView2, u: ArrayView2, w)`,
  or just have metrics write
  `cc_rel_l2_error_weighted(transfers.to_cheb(u_uni.view()).view(), ui, w, w)` and delete
  the method.
- The brief asked for the method, so the first option keeps the brief and the reuse.
  Deleting the method is the smaller codebase.
- In either case, T14 points the generator at the same expression.
- Behaviour is unchanged bit for bit, because `to_uniform` is deterministic.

### 2. Missed simplification

**2.1 The batched transfer is still written out three times.**

- `data/loaders/poisson.rs:305-318` (`transfer_samples`)
- the `metrics/poisson.rs` test helper `to_uniform` (about line 229)
- the per-sample loop in `poisson_errors` (`metrics/poisson.rs:100`)

The first two are the same loop: "allocate `[N, s, s]`, zip over axis 0, assign
`to_uniform`". The brief removed the duplicated recipe, but this batched loop, which
`Transfers` makes trivial to share, was left copied.

**Remedy:** add `Transfers::to_uniform_all(&self, f: ArrayView3<f64>) -> Array3<f64>`.

- The loader calls it on `field.slice(s![..count, .., ..])` and `transfer_samples`
  disappears.
- The metrics test helper becomes a single call.

This is optional within T8's scope. If it's deferred, note it for T22 (the loader split),
which touches the same lines.

### 3. More branching

None beyond B1. The generator's comment at `dataset.rs:193-194` ("moves onto the ungated
pieces in Phase 4s T14") is debt on purpose, tied to a task number. That's acceptable.

### 4. Boundaries and type contracts

**4.1 `Transfers` stores `s` redundantly.** `transfer.rs:191, 208, 221`

`n()` is derived from `t_cu.ncols()`, but `s()` reads a stored field equal to
`t_cu.nrows()`. Storing one and deriving the other is inconsistent, and the struct ends
up with a field whose only job is to repeat a matrix dimension. Derive both from `t_cu`
and drop the field. This also simplifies `new` (see 1.1).

**4.2 The CGL loader's bound and message no longer match the uniform loader's.**
`data/loaders/poisson.rs:225-228`

`load_poisson_cheb_field` still rejects n < 3 "(the uniform grid s = n - 1 needs both
endpoints)", but it never touches the uniform grid. Its module doc says it is "validated
as `load_poisson_uniform` validates it", and that's now false at n = 3. Pick one:

- Use `Transfers::MIN_N` in both loaders, so "same validation" stays true.
- Or keep 3 and reword the message to say what the CGL loader actually needs.

The first is simpler to explain.

**4.3 `ErrorStats` derives `Copy` and `PartialEq` that nothing in production uses.** `metrics/stats.rs:12`

- `PartialEq` is used by the new round-trip test. That's fine.
- `Copy` isn't needed by anything (`Split` derives `Clone` only). It's harmless on a
  two-`f64` struct.

Keep both if you like; this isn't a blocker.

### 5. File size and decomposition

No concerns. `metrics/stats.rs` is the right size for its job. Putting it under `metrics`
means `pde` now imports from `metrics`. That direction is acceptable: `metrics/stats` has
no dependencies of its own, and the brief offered this option.

### 6. Modularity

**6.1 `cc_rel_l2_error_weighted` now makes two temporary `[n_x, n_y]` arrays per call.** `clencurt.rs:122-123`

This follows from the brief's "`cc_rel_l2_error` calls `cc_norm_sq`". The old version did
one fused `Zip` pass with no allocation. The new version reads better (num and den are
visibly the same norm), and the cost is noise next to the O(n³) transfers. Accept it,
but say so in the PR description so it isn't "optimised" back later.

### 7. Legibility

- `transfer.rs:202`: the `# Panics` text ("If `n < 2` or s < 2, or s ≤ …") becomes one
  line once 1.1 lands.
- `poisson_plot.rs:125`: `nodes_closed_form(n).to_vec()` is fine. The repeat/cycle code
  that builds `xs`/`ys` from it is unchanged and outside T8's scope.

## Next steps

Each step preserves behaviour and can be its own commit on the T8 branch. Steps 1 and 2
are the blockers.

1. **`Transfers::MIN_N`, and derive `s` from the matrices (1.1, 4.1, 4.2).**
   - Replace `check_n`/`check_s` in `new` with one `MIN_N` assert.
   - Drop the `s` field.
   - Use `MIN_N` in the uniform loader, `load_poisson_cheb_field` and `poisson_errors`
     (the last one drops its own assert).
   - Merge the two `Transfers` panic tests into one.
   - **Verify:**
     - `cargo test` and `cargo test --features chebyshev`. The counts should drop by
       exactly one, the merged panic test, and the PR should list it as deliberate.
     - `too_small_n_panics`, `n_below_four_is_invalid_not_a_panic` and the CGL loader's
       n test still pass, with their expected strings updated if needed.
2. **Have `round_trip_error` reuse the caller's `u_uni` (1.2).** Change the signature, or
   delete the method and inline the expression in metrics.
   - **Verify:** `transfers_round_trip_matches_manual_recipe` (bit-identical),
     `exact_prediction_has_only_transfer_error`, and a re-run of `predict_poisson` on
     `runs/poisson_fno_p4_1791922918`, with `cmp` on `errors.json` against the baseline.
   - Independent of step 1.
3. **Optional: `Transfers::to_uniform_all` (2.1).**
   - Delete `transfer_samples` and the metrics test helper's loop.
   - **Verify:** the loader tests (`*_asymmetry*` and the transfer checks), the metrics
     tests, and a re-run of `predict_poisson` with `cmp` on `errors.json`.
   - Best after step 1, since it touches the same loader lines. It can also go to T22.
4. **Add a PR-description note on 6.1** (no code change).

Then re-run the full check set: the six README checks, `cargo clippy --all-targets
--features chebyshev -- -D warnings`, and `poisson_plot` sample 0 with `cmp` on the PNG.
