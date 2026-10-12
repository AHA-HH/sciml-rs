# T4: resolution study

Phase 4, task 4. Branch `phase4/T4-resolution-study`. Depends on T3 (a 129² run and the
stage table).

## Read first
- Design §5.3 (stage 4, sets 3e and 4e), §7 (the resolution study, same seeds), §12
  decisions 10 and 18.
- `examples/predict/burgers_resolutions.rs` (`burgers_resolution`), the pattern to
  follow; `examples/predict/poisson.rs` and `metrics::poisson::poisson_errors`.
- The T2 and T3 Results rows of `docs/phase4/README.md`.

## Goal
How the trained FNOs behave on finer grids than they were trained on, with the forcing
fields held fixed so only the grid changes (design §7).

## Deliverables
1. **Datasets:** sets 3e (n = 129, K = 32) and 4e (n = 257, K = 32), the stage 2 test
   seeds; their K = 32 train splits at n = 129 and 257 on the stage 2 train seeds
   (0–999), for the normalisers only (about 270 MB and 1 GB); and the stage 4 (n = 257,
   K = 64) train and test splits, the train split again for the normalisers only. Label
   check and error 2 recorded for each.
2. **`examples/predict/poisson_resolution.rs`**, registered as `poisson_resolution`, no
   required features: `-- runs/<dir> <test file> ...`. For each test file it loads the
   uniform inputs (with the run's x normaliser) and raw u at that file's n, predicts
   without retraining, and reports errors 3 and 4 (mean / max) via `poisson_errors`.
3. **The normalisers at a new s** (decision 18). The x and y normalisers are pointwise,
   fitted at the training s, so they have the wrong shape at 128 or 256. They are refit
   with `load_poisson_uniform` on the same training fields (the same seeds and K)
   generated at the new n: a fixed function of the training distribution, evaluated on
   the finer grid. The model weights are never refit. `burgers_resolution` has no
   normaliser, so it is no guide here.
4. **Part 1:** the 65² model (T2's chosen run) at 65, 129 (3e) and 257 (4e).
5. **Part 2:** the 129² model (T3's run) at 129 and 257 (stage 4).
6. The `poisson_resolution` command in `CLAUDE.md`.

## Tests
If any helper moves into the library, it gets unit tests there, ungated and synthetic.

## Acceptance
- Both T4 rows of `docs/phase4/README.md` Results filled in, on the same seeds.
- The Phase 4 exit checklist is ready to tick, apart from the CLAUDE.md pointer.

## Checks to run and report
All of `CLAUDE.md`'s CI checks and `cargo clippy --all-targets -- -D warnings`, and the
backend each evaluation ran on.

## Do not
- Retrain or fine-tune at the evaluation resolution, or interpolate the fitted
  normalisers instead of refitting them.
- Change modes or the model to suit a resolution.
- Start the θ-map comparison (Phase 5).
