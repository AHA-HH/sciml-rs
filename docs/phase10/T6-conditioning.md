# T6: PDE conditioning against neural-operator conditioning (draft)

Phase 10, task 6. Branch `phase10/T6-conditioning`. Depends on T4; may run beside T5.
**Draft:** refined when T4 merges.

## Read first
- Design §13.17 ("Conditioning") and §13.4 (the closed-form spectra, μ_kl), §13.13
  (H10b).
- `docs/phase6/results.md` (POD spectra of f and u on the CGL and learning grids).
- T1's `cg` (Lanczos Ritz values) and `vjp`, T3's `FnoResidual`.

## Goal
Answer H10b: is the Gauss–Newton curvature of the FNO resolution-independent, unlike the
PDE's discrete operator, and does its spectrum follow the solution operator's (1/μ_kl)
or the data's POD spectrum?

## Measurements
1. **Curvature spectra across s.** For trained baseline weights (one Adam seed, and the
   best GN arm of T4 or T5) at s = 32, 64, 128 on stage 3 (modes as in Phase 5,
   min(12, s/2)): the top Ritz values of G + λI by Lanczos (CG on a random right-hand
   side, or a plain Lanczos loop on `ParamVec`; at least 50 iterations), on the same 100
   training samples. Report θ_max, the decay of the top values and the λ-relative
   spread.
2. **Single-sample tangent kernel.** For one training sample at s = 64: Kᵢ = JᵢJᵢᵀ
   (4,096 × 4,096) by 4,096 VJPs, in the sine basis (the PDE's eigenfunctions on the
   uniform grid); its diagonal against 1/μ_kl and against the POD spectrum of u.
3. **PDE side, closed form:** κ of the discrete Laplacians (Chebyshev collocation, O(n⁴);
   uniform finite differences, O(s²)) at the same resolutions, for the comparison.

## Deliverables
1. An example (`poisson_conditioning`, registered in `Cargo.toml`) that runs 1–2 on a
   run directory and writes tables and plots; results appended to
   `docs/phase10/results.md` under "Conditioning".
2. H10b answered; the T6 Results row filled in.

## Checks
All of `CLAUDE.md`'s checks and `cargo clippy --all-targets -- -D warnings`.

## Do not
- Train new models for this task beyond what T4/T5 provide.
- Add the PDE-induced output-norm arm (a follow-up, §13.17); this task only measures.
