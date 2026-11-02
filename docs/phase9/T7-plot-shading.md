# T7: smooth display in `poisson_plot`

Phase 9, task 7. Branch `phase9/T7-plot-shading`. Depends on nothing; runs at any time
(decision 19). Display only.

**Status: closed 2026-11-02. The author chose option 1, keep flat cells**; recorded in
design §12, decision 19 (amendment of 2026-11-02). No code change. The brief below is
kept as history.

## Read first
- `examples/predict/poisson_plot.rs`.
- `src/neural_operators/chebyshev/transfer` (`cheb_to_uniform`, `apply`).
- Design decisions 18 and 19; old Phase 5 "Supplementary task: smooth display in
  `poisson_plot`".

## Goal
Remove the block pattern that flat per-node cells give smooth fields, without hiding
anything the plot is for.

## Options (the author picks one before any code changes)
1. **Keep flat cells.** The plot shows exactly where the CGL nodes are and what each
   holds; the blocks are the honest picture. No code change; decision 19 closed.
2. **Every panel smooth:** each field's Chebyshev interpolant on a fine uniform grid
   (`apply` with `cheb_to_uniform(n, 400)`), drawn as an image. Best-looking; the error
   panel then shows the interpolant of the nodal error, which is not the error between
   nodes.
3. **f, u and û smooth, the error panel flat.** Smooth fields, and the error is shown
   only where it was measured. Recommended.

## Deliverables
1. The choice recorded against decision 19 in design §12, dated.
2. If 2 or 3: the change in `poisson_plot`, the colour scale unchanged, and a before
   and after image in the PR (`*.png` is ignored).
3. The T7 Results row.

## Acceptance
- The plotted node values (flat panels) unchanged; the smooth panels equal the
  interpolant at the nodes to round-off.

## Checks to run and report
All of `CLAUDE.md`'s CI checks and `cargo clippy --all-targets -- -D warnings`
(`clippy --examples` covers the example).

## Do not
- Change any error computation, model, transfer or CONVENTIONS.
- Add a plotting dependency; gnuplot stays the only one.
