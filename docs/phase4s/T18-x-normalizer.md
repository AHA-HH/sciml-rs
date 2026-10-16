# T18: `x_normalizer.json`, read or stop writing ⚑

Phase 4s, task 18. Branch `phase4s/T18-x-normalizer`. Depends on T9 and T10. Can run in
parallel with T17 and T19, with small overlaps in the predict examples, so rebase.

> **⚑ Decision needed before any code changes.** Record it in design §12 and fill the README
> Results row "Decisions taken".
>
> **Refresh at start:** re-read `RunDir::normalizers` (T10), `loaders::common` (T9) and the
> Darcy/Poisson predict examples on `main`.

## Read first
- Review §5 S1 (Problems, second bullet), §7 step D2.
- Design §12 decision 18: `poisson_resolution` refits the normalisers on the train split
  at the new n, on purpose.
- Today:
  - `x_normalizer.json` is written by `train/darcy.rs:99` and `train/poisson.rs:143`, and
    never read.
  - Every predict path (`predict_darcy`, `darcy_plot`, `predict_poisson`, `poisson_plot`)
    re-reads and, for Poisson, re-transfers the **whole train split** to refit x.
  - Burgers runs have no normalisers.

## Decision

| Option | Consequence |
| --- | --- |
| **Read it** (recommended) | Predict loads x (and y) from the run, and loads only the test split. That is faster (no train split read, no T_cu over every training sample) and more correct: it uses exactly the statistics the model was trained with, even if the train file on disk changes. It needs a loader entry point that takes fitted normalisers, e.g. `load_*_test(path, &Normalizers)`. |
| Stop writing it | Simpler. Predict keeps refitting, and the file disappears from new runs. Old runs keep a harmless extra file. |

**Recommendation: read it.** Under both options, **`poisson_resolution` keeps refitting** at
the new n (decision 18), because a pointwise normaliser fitted at one s has the wrong shape
at another.

## Deliverables (if "read it")
1. **A test-split loader** taking fitted `Normalizers`, for Darcy and Poisson:
   - `load_darcy_test` / `load_poisson_uniform_test` (names to fit T9's module).
   - It shares the encode step with `normalize_split_pair`.
2. **Predict examples:**
   - `predict_darcy`, `darcy_plot`, `predict_poisson` and `poisson_plot` use
     `RunDir::normalizers()` and the test-split loader.
   - If a run has no `x_normalizer.json`, fail with a clear message naming the run and the
     file. Phase 3+ runs all have it; check that against `runs/`.
3. **`poisson_resolution`** is unchanged. Add a comment citing decision 18 next to its
   refit.

## Deliverables (if "stop writing")
1. `RunDir::save_normalizers` writes only y. The docs say why.
2. Nothing else changes.

## Tests
- **Read it:**
  - `test_loader_with_saved_normalizers_matches_refit`: on a fixture, the test split
    encoded with the saved normalisers equals the refit path bit for bit (same training
    data → same statistics).
  - `RunDir` round trip of both normalisers.
- **Stop writing:** a `RunDir` test that only y is written.

## Acceptance
- `predict_poisson` on `runs/poisson_fno_p4_1791922918` prints the same five errors, and
  `predict_darcy` the same L².
- Report the wall-clock time of `predict_poisson` before and after; the "read it" option
  should be faster.

## Checks to run and report
The six from the README. The chebyshev pair is not needed.

## Do not
- Change `poisson_resolution`'s refit, or interpolate normalisers to a new resolution.
- Change the normaliser JSON format.
