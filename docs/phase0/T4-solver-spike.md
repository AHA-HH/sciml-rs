# T4: reference-solver spike

Phase 0, task 4. Branch `phase0/T4-solver-spike`. Depends on T2: it uses the API list and
the BLAS setup T2 verified.

## Read first
- Design §4 (options A–E, decision), §3.3 (GRF), §5.3 (resolutions), §12 decision 1.
- T2's PR description: the API findings.

## Goal
Measure option A (collocation, authoritative) and option C (sparse SPD, CG validation
oracle) at n = 33..257. Choose A's solve route, C's discretisation (finite volume or Q1)
and C's CG tolerance.

## Setup
- Standalone package `spikes/solver/`, depending on `rlst` 0.9 with the same BLAS
  providers as T2. f64 throughout.
- Chebyshev second-derivative matrix: Trefethen's `cheb` (D, then D² = D · D),
  interior block. Sanity check: D² is exact on polynomials of degree < n.
- **Option A**, −D_xx U − U D_yyᵀ = F on the interior. Two routes:
  - A-syl: RLST's Sylvester routine, if T2 found it public;
  - A-fd: fast diagonalisation. Diagonalise D_xx once with `eig`; D_xx has real,
    distinct eigenvalues, so use their real parts and report the largest imaginary part
    seen. Then solve in the eigenbasis: two dense products per side and one pointwise
    division.
  - In both routes the factorisation is computed once and reused across samples.
- **Option B**, dense LU of the Kronecker-sum system: n ≤ 33 only, as an oracle for A.
- **Option C**, implemented both ways:
  - C-fv: a finite-volume 5-point stencil on the CGL mesh, scaled by dual-cell areas so
    that it is SPD;
  - C-q1: Q1 finite elements on the tensor mesh, with a lumped or consistent load (say
    which).

  Both are assembled with `CsrMatrix::from_aij` and solved with `CgIteration`, no
  preconditioner, tolerance ∈ {1e-6, 1e-8, 1e-10}.

## Measurements
1. **Manufactured solutions:** u = sin(πx) sin(πy); u = (1 − x²)(1 − y²)(x + y²);
   u = (1 − x²)(1 − y²) e^(x + 2y). Errors are max-norm and Clenshaw–Curtis relative L²
   on the CGL nodes.
   - A (both routes) at n ∈ {9, 17, 33, 65, 129, 257}: spectral decay, and the floor
     reached.
   - C (both variants) at n ∈ {33, 65, 129, 257}: the observed order.
2. A against B at n ∈ {9, 17, 33}.
3. **Common-grid comparison:** on 5 GRF samples (design §3.3), ‖u_A − u_C‖ on the CGL
   nodes at each n, and its observed order.
4. **Cost:**
   - setup time (factorisation or assembly);
   - time per solve for A at each n, averaged over 50 samples;
   - CG iterations and time per solve for C at each tolerance;
   - peak memory at n = 257 if easily measured.
5. **Projection:** the time to generate 1200 samples (1000 train + 200 test) with A at
   each n.

## Decision rules
State these in `REPORT.md` and apply them:
- **A's route:** A-syl if it exists and agrees with A-fd to 1e-12 relative, and is not
  slower than A-fd by more than 2× at n = 257 with reuse. Otherwise A-fd.
- **C's discretisation:** the variant that shows order 2 (between 1.8 and 2.2) on all
  three manufactured solutions at n ∈ {65, 129, 257}. If both do, the one with fewer
  CG iterations at n = 257.
- **C's CG tolerance:** the loosest tolerance at which the CG error stays below a tenth
  of C's discretisation error at n = 257.
- **Stop and report**, rather than choosing, if A does not reach 1e-10 on the first two
  manufactured solutions by n = 33, or if A disagrees with B by more than 1e-10.

## Deliverables
- `spikes/solver/` and `spikes/solver/REPORT.md`: the tables, the reproduction commands,
  the machine and the decisions.
- In `docs/phase0/README.md`, the Results rows for A's route, FV or Q1, and the CG
  tolerance.

## Acceptance
- `REPORT.md` holds every table above and states the three decisions or the stop
  condition that fired.
- The crate's own checks are unaffected.

## Do not
- Add anything to `src/` (Phase 1 writes the production solvers).
- Generate or commit datasets.
