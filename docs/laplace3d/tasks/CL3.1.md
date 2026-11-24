# CL3.1 3D GRF

Started 2026-11-24 on `laplace3d/cl3-1`. Roadmap: "**CL3.1 3D GRF.** Separable evaluation
(proposal §3.3), nested draws, the 3D S_K; ε_K for K = 8, 16, 32. *Done when:* the
evaluation matches a direct sum on a small case; the ε_K table; D3 re-proposed.
*Needs:* CL1.1. *Sign-off:* D3." Proposal: §10; §3.3; §2 ("Already separable"); §6
("Extend").

## Goal
3D entry points beside the 2D GRF in `pde/poisson/grf.rs`: the 3D spectrum (μ, λ, S_K,
S_∞, ε_K at α = 3), nested 3D draws from a ChaCha8 stream distinct from the 2D sets', the
forcing and solution coefficients and their separable evaluation on a 3D tensor grid.

## Acceptance (roadmap *Done when*, verbatim, and the proposal's matching text)
- "the evaluation matches a direct sum on a small case": `evaluate_3d` against a triple
  loop over (k, l, m) at every node, n = 9 × 7 × 5 (non-cubic), K = 4, max |diff| ≤
  1e-14 × scale.
- "the ε_K table": ε_K at K = 8, 16, 32 (and 64) from S_∞ recomputed in the test, within
  5 % of proposal §3.3's α = 3 row (2.7e-2, 4.3e-3, 5.9e-4, 7.7e-5); written to
  `docs/laplace3d/results/cl3-1.md`.
- "D3 re-proposed": the PR asks the author to confirm D3 (τ = 3, α = 3, K(n) = (n − 1)/2,
  ε_K ≤ 5e-3 on f) on the measured table.
- Proposal §3.3: S_K with mean square 1/8 per mode; nested draws (K_MAX³ normals, row-major
  over (k, l, m), m fastest, K_MAX = 64, a smaller K the leading block); 3D seeds from a
  stream distinct from the 2D sets' (scheme fixed here, recorded as a sidecar string for
  CL3.2); separable evaluation, O(K n³ + K² n² + K³ n); exact solution c / μ.

## Read first
`src/neural_operators/pde/poisson/grf.rs` (the 2D GRF and its tests), proposal §3.3,
`rand_chacha-0.10.0/src/chacha.rs` (`set_stream`).

## Changes
`src/neural_operators/pde/poisson/grf.rs` only (behind `datagen`, as the module is). New
public items, named with a `_3d` suffix beside the 2D ones; no existing item changes:
- `ALPHA_3D = 3.0` (D3; τ is the shared `TAU`), `S_INF_3D` (computed once, recomputed by
  a test), `STREAM_3D = 3` and `RNG_3D` (the sidecar string).
- `mu_3d`, `lambda_3d`, `s_k_3d`, `eps_k_3d`.
- `draw_xi_3d(seed) -> [K_MAX, K_MAX, K_MAX]`: `ChaCha8Rng::seed_from_u64(seed)` with
  `set_stream(STREAM_3D)`. The 2D draws (and Phase 7's subsets) use stream 0, so a 3D seed
  never repeats a 2D sample's normals, while seeds keep the 2D numbering
  (`base_seed + i`).
- `forcing_coeffs_3d`, `solution_coeffs_3d`, `evaluate_3d` (for each k, the 2D product
  S_y C_k S_zᵀ, then S_x over the stacked results: three mode products).
Conventions: none changed; output stored 'ijk' (CONVENTIONS §12, version 3).

## Tests that prove it
In `grf.rs`'s `mod tests` (datagen):
1. `evaluate_3d_matches_direct_sum` (above).
2. `eps_k_3d_matches_design_table` (above), including S_∞ recomputed by the cube sum to
   m = 160 plus the integral tail bound outside radius m + ½, relative to `S_INF_3D`.
3. `draws_3d_are_nested` (c at K = 8 equals the leading block of c at K = 16 times
   √(S_16/S_8)) and `stream_3d_differs_from_2d` (same seed, different normals; the
   3D stream is reproducible).
4. `mean_square_3d_is_one` (Parseval, 1/8 per mode, within 4 s.e.).
5. `field_and_solution_3d_vanish_on_boundary` (all six faces, ≤ 1e-14 × scale).
6. `solution_3d_solves_poisson`: −Δ of the u series equals f, checked through the
   coefficients against a second-difference stencil of `evaluate_3d` at one interior
   point is weaker than needed, so instead −Δ is applied to each sine product exactly:
   μ_3d(k, l, m) · u_klm = c_klm to 1e-15 relative (the closed form).
7. `k_above_k_max_panics_3d`.
Constraint 1: no 2D item changes, so the 2D suite (`cargo test --features datagen`, whose
2D GRF and dataset tests are unchanged) is the bit-identity check.

## Checks
`/verify` (fmt, the clippy jobs, `cargo test`, `cargo doc`, datagen clippy and tests).

## VM runs
None.

## Constraints and risks
Constraint 1: holds by construction (new items only; 2D tests unchanged and run). The 3D
draws cost 64³ normals per sample (2 MB): fine for generation, kept small in tests.

## Pending runs
None.

## Open questions
None yet.
