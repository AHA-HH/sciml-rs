# T1: CONVENTIONS §12, Chebyshev grids and transfers

Phase 0, task 1, done **last** in the phase. Branch `phase0/T1-conventions`. Depends on
T3 and T4 being merged.

## Read first
- `docs/CONVENTIONS.md`, the whole file.
- Design §8 (the diff to apply), §3.2–§3.4, §12 decision 4.
- `spikes/transfers/REPORT.md` (d, s) and `spikes/solver/REPORT.md`.
- `docs/phase0/README.md`, Results.

## Goal
Add §12 to `docs/CONVENTIONS.md` as the design's §8 gives it, with T3's measured degree d
and uniform size rule filled in, and extend §9 to cover §12.

## Deliverables
1. **`docs/CONVENTIONS.md`.**
   - Append "## 12. Chebyshev grids and transfers" with:
     - domain and map: Ω = [−1, 1]², x = 2ξ − 1;
     - CGL nodes, ascending: x_j = −cos(πj/(n − 1)), endpoints included;
     - 'ij' storage: `F[i, j] = f(x_i, y_j)`;
     - the uniform size rule from T3, and the endpoints included;
     - Chebyshev → uniform by tensor-product barycentric interpolation;
     - uniform → Chebyshev by tensor-product Floater–Hormann of degree d = <T3's value>;
     - Clenshaw–Curtis norms on the Chebyshev grid;
     - reference data in f64 on the host.
   - In the style of the file, §12 states each item with a formula in a `math` block
     where there is one, and ends with a `### Verification` list. Name the tests that
     will confirm it, all in Phase 1 T1 and Phase 2 T1; mark them "planned" until they
     exist.
   - Extend §9: "§12 fixes how dataset files and grid transfers are laid out; a change to
     it bumps `CONVENTION_VERSION`." Also update the opening paragraph's list of which
     sections bump the version.
   - Keep `CONVENTION_VERSION = 1`: §12 is additive and changes nothing in §1–§6
     (design decision 4).
2. **`docs/phase0/README.md`.** Fill the "CONVENTIONS §12 merged" row and tick the
   checklist items this task completes.

## Acceptance
- `git diff --stat main` shows only `docs/CONVENTIONS.md` and `docs/phase0/README.md`.
- `grep -n "CONVENTION_VERSION" docs/CONVENTIONS.md` still shows `= 1`.
- Every value in §12 matches the design §3.2–§3.4, or T3's REPORT where the design left
  it open. Any mismatch is resolved in the design's "Recorded decisions" first.

## Do not
- Change §1–§11 beyond the two §9 and opening-paragraph sentences above.
- Add code.
