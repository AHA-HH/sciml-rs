# T14: generator on the ungated Chebyshev stack ⚑

Phase 4s, task 14. Branch `phase4s/T14-generator-ungated-stack`. Depends on T6 and T8.
- Can run in parallel with Group 3 (the run/training track).
- Do not run it beside T4 or T22, which edit `pde/poisson/dataset.rs` or the catalogue it
  imports.
- T15 comes after.

> **⚑ Decision needed before any code changes.** Record the answer in design §12 (the next
> free decision number, 26 or later) and in the README's Results row.
>
> **Refresh at start:** re-read T8's `Transfers`, `cc_norm_sq`, the shared `ErrorStats` and
> the public nodes name on `main`. Check whether T4 changed the reader API that
> `dataset.rs`'s tests use.

## Read first
- Review §5 S2 ("CGL nodes / CC weights / rel-L²" row), §6 T2-8, T4-9, §7 step C2.
- Design §12 decision 12 (RLST toolkit; the closed forms became test oracles) and decision
  13 (dataset reproducibility). CONVENTIONS §12.
- `pde/poisson/dataset.rs`:
  - the gated imports :30-32
  - `nodes` :196, `clenshaw_curtis` :197, `rel_l2_error` :164 and :226
  - `label_check` :157-163, `Split` :96-117 (`y` filled as `x.clone()` at :239-240)
- `data/loaders/poisson.rs`: `NODE_TOL = 1e-14` (:41) and `check_nodes` :244-272.
- `chebyshev/{points,quadrature,norm}.rs`, and the gated cross-check tests in `clencurt.rs`
  (`clencurt_matches_rlst_weights` :239, `cc_rel_l2_matches_gated` :254).
- `pde/poisson/manufactured.rs` (test-only, uses gated `nodes` :8, :92) and
  `pde/poisson/sparse.rs` (test-only after T6, uses gated `nodes`).

## Decision: regenerate the datasets, or keep them?
With the closed-form nodes (−cos(πj/(n − 1)), computed as a sine), the generated `x` / `y`
move by at most about 1 ulp. That is within the loader's `NODE_TOL = 1e-14`, so old files
still load and every stored `f` / `u` stays valid.

| Option | Consequence |
| --- | --- |
| **Keep** (recommended) | No data churn, and every existing run and Results number stays reproducible. New datasets written after T14 differ from old ones only in the last bit of the node coordinates. |
| Regenerate | Byte-exact agreement between new and old files from the same seed. Costs about 1 GB+ of regeneration and invites drift questions on every Phase 4 number. |

**Recommendation: keep them.** T15 changes labels at round-off and decides about
regeneration anyway, so regenerating here would be wasted work.

## Deliverables
1. **Switch the generator** (`dataset.rs`) to the ungated stack:
   - the public closed-form nodes (T8's name)
   - `clencurt` instead of `clenshaw_curtis`
   - `cc_rel_l2_error` / `cc_norm_sq` instead of `rel_l2_error`
   - T8's `Transfers` for the round trip
2. **Settle the nodes name.**
   - The gated `points::nodes` becomes a test oracle, so the closed-form function can take
     the plain name `nodes`.
   - Rename T8's interim name everywhere it is used, and update CLAUDE.md "Layout".
3. **Demote the gated modules.** `points.rs`, `quadrature.rs` and `norm.rs` become
   `#[cfg(all(test, feature = "chebyshev"))]` oracles. Their re-exports in
   `chebyshev/mod.rs` (:31-37) go with them.
   - `manufactured.rs` and `sparse.rs` (both test-only) switch to the ungated nodes, or
     use the oracle explicitly. Prefer ungated.
   - `differentiation.rs` and `unit_coeffs` stay as they are; T15 handles them.
4. **`label_check`** takes no weights, because it only used them to infer `n`. It gets `n`
   from the array shape and computes `clencurt` itself, or takes `&Transfers`. Update its
   callers and docs.
5. **Delete `Split.y`.**
   - It duplicates `x` by contract.
   - The `.npz` keeps writing a `y` array, a copy of `x`, so the **file format does not
     change**. `write_split` writes `x` twice.
   - The loader's check that `y == x` stays.

## Tests
- `label_check` at 1e-8, the manufactured floor at 1e-12 and `error_stats_propagate_nan`
  all pass.
- `cc_rel_l2_matches_gated` and `clencurt_matches_rlst_weights` still run, as gated oracle
  tests.
- **Add** `generator_nodes_within_node_tol_of_rlst`: closed-form against RLST nodes for
  n ∈ {33, 65, 129, 257}, max diff ≤ `NODE_TOL`.
- **Add** an ungated test: a split written by the generator path still loads with
  `load_poisson_uniform` (round trip through `write_split` on a tiny n).

## Acceptance
- **Keep:** load every existing stage with `load_poisson_uniform` and `load_poisson_cheb`
  (all pass `check_nodes`).
- **Regenerate:** regenerate stage 1 (n = 33) and diff the sidecar errors against the old
  sidecar.
- Either way, generate stage 1 into a scratch directory and compare the label check and
  error 2 with the existing sidecar. Expect agreement to about 1e-15. Fill the Results row.

## Checks to run and report
The six from the README **plus the chebyshev pair**. Give the line delta.

## Do not
- Change `diff_matrix` / `diff2_matrix` or drop RLST/FFTW. That is T15.
- Ungate any module (T16).
- Change the `.npz` keys, the sidecar JSON or the seeds.
