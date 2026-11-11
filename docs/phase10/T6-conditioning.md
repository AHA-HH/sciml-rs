# T6: PDE conditioning against neural-operator conditioning

Phase 10, task 6. Branch `phase10/T6-conditioning`. Its code depends on T3 only; its runs
need T4's go and come at the end of the phase (order amended by the author 2026-11-11,
decision 31). Refined 2026-11-11 (below).

## Decisions (author, 2026-11-11)
- **Grids:** each run's s = 64 weights are measured at s = 32, 64 and 128, with both
  normalizers refit on the run's training set at each s (decision 18, as
  `poisson_resolution`): the resolution-independence of one trained operator, H10b's claim.
- **Ritz values:** a generic `lanczos` (full reorthogonalisation) in `optim/gauss_newton`,
  not CG's two extremes: no ghost copies of the top values from f32 and finite-difference
  noise.
- **Shift:** G + μI with μ = 1e-4 (the weight decay, G5) for every run and grid, not a
  run's damping λ.
- **Output:** `docs/phase10/conditioning.md`, its own file: `poisson_matrix
  --by-optimizer` rewrites `results.md` in full.
- **Runs measured** (chosen when the runs are made): one Adam seed of the N = 1000
  baseline, T5's best GN arm (the same N), and T4's N = 100 GN run as a second point.

Smaller choices, in the code: the "same 100 training samples" are Phase 7's subset 0 (T4's
training set; a subset run must be that subset); the kernel's sine basis is the (s − 2)² =
3,844 interior modes at s = 64 (the grid has both boundary rows, where every sine
vanishes), so 3,844 VJPs; Lanczos takes 60 steps from a seeded Rademacher vector; one
adjoint error per (run, s) is reported beside the spectrum; the Chebyshev κ (eigenvalues by
LAPACK) is computed only when the example is built with `datagen`.

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
1. **Curvature spectra across s.** For trained weights (the runs above) at s = 32, 64,
   128 on stage 3 (modes 12, the run's): the top Ritz values of G + μI by `lanczos`
   (60 steps), on the same 100 training samples. Report θ_max, the decay of the top values and the λ-relative
   spread.
2. **Single-sample tangent kernel.** For one training sample at s = 64: the diagonal of
   Kᵢ = JᵢJᵢᵀ by 3,844 VJPs, in the sine basis (the PDE's eigenfunctions on the
   uniform grid); its diagonal against 1/μ_kl and against the POD spectrum of u.
3. **PDE side, closed form:** κ of the discrete Laplacians (Chebyshev collocation, O(n⁴);
   uniform finite differences, O(s²)) at the same resolutions, for the comparison.

## Deliverables
1. An example (`poisson_conditioning`, registered in `Cargo.toml`) that runs 1–3 on run
   directories and writes tables and plots:
   `poisson_conditioning -- docs/phase10/conditioning.md runs/<run_dir> ...`
   (cuda on the VM; with `datagen` for the Chebyshev column).
2. H10b answered; the T6 Results row filled in.

## Checks
All of `CLAUDE.md`'s checks and `cargo clippy --all-targets -- -D warnings`.

## Do not
- Train new models for this task beyond what T4/T5 provide.
- Add the PDE-induced output-norm arm (a follow-up, §13.17); this task only measures.
