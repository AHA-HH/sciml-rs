# CL1.3 N-D errors

Started 2026-11-24 on `laplace3d/cl1-3`. Roadmap: "**CL1.3 N-D errors.** `metrics/poisson`
errors 2–5 for any rank, error 5 over six faces (proposal §3.5)". Proposal: §10; §2 (the
2D-only pieces), §3.5 (norms and errors), §6 (generalise, D1), §9 (D2); CONVENTIONS §12
"Three dimensions (version 3)".

## Goal
The per-sample errors 2–5 of `metrics::poisson` for fields of any rank, with rank 2 routed
to the existing `poisson_errors` so every 2D output stays bit for bit.

## Acceptance (roadmap *Done when*, verbatim, and the proposal's matching text)
"3D errors on synthetic fields against direct computation; 2D unchanged by D2's test."
Proposal §3.5: "Error 5 (boundary) in 3D: the sum over the six faces of the face integral
with 2D tensor CC weights, normalised by the total face area 24 against the volume 8. A
node on an edge or corner belongs to every face that contains it and is counted once per
face." §6: "`metrics/poisson` [takes] rank 2 or 3. Rank 2 calls the existing 2D code
unchanged […] Each has an in-process test that the 2D path's output equals the existing
function's bit for bit (D1, D2)." New fields go on new 3D types, not on
`PoissonErrorsRecord`.

## Read first
- `src/neural_operators/metrics/poisson.rs` (`poisson_errors`, `boundary_errors`,
  `rel_l2_equal_weights`).
- CL1.2's `Transfers::{to_uniform_nd, to_cheb_nd, to_uniform_all_nd}`, `cc_norm_sq_nd`,
  `cc_rel_l2_error_weighted_nd` (`chebyshev/`).
- CONVENTIONS §12: the 2D boundary norm and the 3D one (six faces, 24 against 8).

## Changes
Only `metrics/poisson`, which becomes a folder (code review blocker: the file passed 1,000
lines): `metrics/poisson/mod.rs` and its tests, moved unchanged, in `poisson/tests.rs`.
`poisson_errors`, `boundary_errors`, `evaluate` and
`PoissonErrorsRecord` are not edited, except that the private `rel_l2_equal_weights`
becomes generic over the rank (the same body; `ArrayView2` is its rank-2 instance).
- `poisson_errors_nd<D: RemoveAxis>(pred: ArrayView<f64, D>, u: ArrayView<f64, D>,
  transfers) -> PoissonErrors`: axis 0 the samples, `pred: [N, s, …, s]`, `u: [N, n, …, n]`,
  field rank r = 2 or 3 (the ranks CONVENTIONS §12 defines; plan review). Rank 3 arrays (r = 2) call `poisson_errors`. Otherwise per sample:
  error 2 and 4 by `cc_rel_l2_error_weighted_nd` with `clencurt(n)` on every axis, error 3
  equal weights, ‖u‖ by `cc_norm_sq_nd`, error 5 by `boundary_errors_nd`.
- `boundary_errors_nd` (private): the 2r faces `index_axis(a, 0 | n − 1)`, each integrated
  with `cc_norm_sq_nd` and r − 1 weight vectors; the RMS normalised by the total face area
  2r · 2^{r−1} against the volume 2^r (8 against 4 in 2D, 24 against 8 in 3D, CONVENTIONS
  §12); the max ratio over every face node.
- Returns the existing `PoissonErrors` (no new fields; it is not serialised). The 3D
  `evaluate` and record are later tasks' (CL4.x), on new 3D types.
- Conventions: none changed.

## Tests that prove it
In `metrics/poisson.rs`'s `tests`, rank 4 arrays (3D fields), non-trivial n (9, 11) and s
(n − 1 and coarser):
- `nd_errors_match_direct_computation`: noisy pred, u vanishing on ∂Ω plus noise; every
  error against explicit triple loops (û by Σ T_uc T_uc T_uc g, the CC sums with
  w_i w_j w_k, the six faces written out one by one), to 1e-13 relative.
- `nd_exact_prediction_has_only_transfer_error`: pred = R u gives error 3 ≤ 1e-14,
  error 4 = error 2, error 5 ≤ 1e-14 (the floor identity, design §13.3, in 3D).
- `nd_boundary_offset_is_measured`: u = c₀(1 − x²)(1 − y²)(1 − z²), pred = R u + c; û = c
  on ∂Ω, so the ratio is c/c₀ and the RMS c / √(c₀² (16/15)³ / 8), closed form (the 24/8
  normalisation).
- `nd_u_norm_is_the_cc_norm`: (16/15)^{3/2}, closed form.
- `nd_boundary_at_rank_two_matches_2d`: `boundary_errors_nd` on 2D fields against
  `boundary_errors`, to 1e-14 (the generic face code agrees with the trusted 2D path).
- D2, `nd_rank_two_is_bit_identical_to_2d`: `poisson_errors_nd` on `[N, s, s]` /
  `[N, n, n]` (s = n − 1 and s < n − 1, a transposed view) against `poisson_errors`,
  every vector `assert_eq!`.
- Panics: sample count, rank mismatch, non-cubic, grid not the transfers', field rank not
  2 or 3.

## Checks
`/verify` (fmt, the three CI clippy jobs, `cargo clippy --all-targets`, `cargo test`,
`cargo doc`, datagen clippy and tests); the whole 2D suite passes unchanged. No metal (host
ndarray only).

## VM runs
None.

## Constraints and risks
- Constraint 1 (D2): the 2D `poisson_errors` is not edited and rank 2 is routed to it,
  shown in process by the bit-identity test; `predict_poisson` and `poisson_resolution`
  still call `poisson_errors`, so `errors.json` is unchanged. `rel_l2_equal_weights`
  becomes generic with the same body; the D2 test covers it.
- `poisson.rs` would have grown past 1,000 lines (888 before); its tests moved to
  `poisson/tests.rs` in this task (code review, round 1 blocker).

## Pending runs
None.

## Open questions
- CONVENTIONS §12 "Verification" should name the 3D tests of CL1.2 and CL1.3 (CL1.2's
  follow-up). A CONVENTIONS edit, so it is proposed in the PR for sign-off, not made here.
