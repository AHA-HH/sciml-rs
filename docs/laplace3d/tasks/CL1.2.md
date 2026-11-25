# CL1.2 N-D transfers and quadrature

Started 2026-11-24 on `laplace3d/cl1-2`. Roadmap: "**CL1.2 N-D transfers and quadrature.**
`transfer::apply`/`Transfers` and `cc_rel_l2_error` for any rank". Proposal: §10; §2
(the 2D-only pieces), §3.2 ('ijk'), §3.5 (norms), §6 (generalise, D1), §9 (D2);
CONVENTIONS §12 "Three dimensions (version 3)".

## Goal
Grid transfers and the Clenshaw–Curtis norm and relative error for fields of any rank,
with rank 2 routed to the existing 2D functions so every 2D output stays bit for bit.

## Acceptance (roadmap *Done when*, verbatim, and the proposal's matching text)
"3D round-trip and CC-norm tests against closed forms; the 2D outputs bit for bit unchanged
by D2's test."
Proposal §6: "`transfer::apply`/`Transfers`, `cc_norm_sq`, `cc_rel_l2_error` […] take
rank 2 or 3. Rank 2 calls the existing 2D code unchanged; rank 3 uses an
apply-along-an-axis helper. Each has an in-process test that the 2D path's output equals
the existing function's bit for bit (D1, D2)."

## Read first
- `src/neural_operators/chebyshev/{transfer,clencurt,mod}.rs`.
- CONVENTIONS §12, "Three dimensions": 'ijk' storage; T along x, then y, then z; the
  tensor CC norm Σ w_i w_j w_k v².
- Decisions D1 (generalise, rank 2 routed to the existing code) and D2 (numbers only, in
  process, old against new on fixed 2D inputs, plus the whole 2D suite).

## Changes
Only `chebyshev/transfer.rs`, `chebyshev/clencurt.rs` and the re-exports in
`chebyshev/mod.rs`. The existing 2D functions are not edited; new rank-generic siblings
(generic over ndarray's `Dimension`, so `Array3` in gives `Array3` out):
- `apply_nd(ts: &[ArrayView2<f64>], f: ArrayView<f64, D>) -> Array<f64, D>`: one 1D
  transfer per axis. Rank 2 calls `apply`. Otherwise T₀ along axis 0 as one matrix
  product on `f` reshaped `[n₀, rest]`, then `apply_nd` with the remaining transfers on
  each slice, so rank 3 is x, then y, then z, its inner (y, z) step the 2D `apply`.
- `Transfers::to_uniform_nd`, `to_cheb_nd` (every axis) and `to_uniform_all_nd` (axis 0
  the samples, as `to_uniform_all`).
- `cc_norm_sq_nd(v, ws: &[ArrayView1<f64>])`, `cc_rel_l2_error_nd(a, b)`,
  `cc_rel_l2_error_weighted_nd(a, b, ws)`. Rank 2 calls the 2D function. Otherwise `v` is
  read as `[n₀, rest]` with weights w₀ and the Kronecker product of the rest, so the 2D
  function's scaling (no overflow or underflow) serves every rank unchanged.
Conventions: none changed (CL1.1 wrote §12's 3D statements; this implements them).

## Tests that prove it
3D against closed forms (non-cubic grids, so a swapped axis fails):
- 'ijk' layout: a linear field x + 10y + 100z transferred both ways equals its samples
  (1e-12).
- Separable fields: T(g ⊗ h ⊗ k) = (T_x g) ⊗ (T_y h) ⊗ (T_z k) with the 1D transfers
  (1e-14 relative): the axis order and the per-axis matrices.
- Round trip exact on polynomials of degree ≤ 2 per axis: `to_cheb_nd(to_uniform_nd(u))`
  = u (1e-12; T_cu is exact below degree n, FH d = 2 reproduces degree 2).
- Restriction exact on a resolved sine mode at s = 16, 32 (1e-12), as the 2D test.
- CC norm: ∫ 1 = 8 and ∫ x² y⁴ z² = (2/3)(2/5)(2/3), exact at these n (1e-14);
  `cc_rel_l2_error_nd` of (1 + ε) b against b is |ε| (1e-14); extreme magnitudes
  (1e±170) still give 0.25; a permuted view gives the same error.
- `to_uniform_all_nd` on `[N, n, n, n]` equals `to_uniform_nd` per sample (bit for bit).
- Panics: wrong number of transfers or weights, a mismatched axis.
D2 (old against new, in process, `assert_eq!` on the f64 bits), on fixed 2D inputs
including a non-square grid and a transposed view: `apply_nd` = `apply`;
`to_uniform_nd`, `to_cheb_nd`, `to_uniform_all_nd` = `to_uniform`, `to_cheb`,
`to_uniform_all`; `cc_norm_sq_nd`, `cc_rel_l2_error_nd`, `cc_rel_l2_error_weighted_nd` =
the 2D functions. And the whole 2D suite unchanged.

## Checks
`/verify` (fmt, clippy ×3 jobs and `--all-targets`, `cargo test`, `cargo doc`, datagen
clippy and tests, since the `rlst_oracle` tests live in these modules). No backend code:
metal not needed.

## VM runs
None.

## Constraints and risks
- Constraint 1: no 2D function is edited and no 2D caller changes; rank 2 is routed to the
  existing functions, checked bit for bit.
- Stay inside `chebyshev/` (CL3.1 runs in parallel in `pde/poisson/grf`).
- Rank-3 CC norm sums w₀ᵢ (w₁ⱼ w₂ₖ) v² rather than ((w_i w_j) w_k) v²: a round-off
  difference, no 2D effect, and 3D has no stored outputs yet.

## Pending runs
None.

## Open questions
None.
