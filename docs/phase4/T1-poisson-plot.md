# T1: `poisson_plot`

Phase 4, task 1. Branch `phase4/T1-poisson-plot`. Depends on Phase 3.

## Read first
- Design §3.2 (CGL nodes), §7 (errors 4 and 5), §12 decisions 17 and 18.
- `docs/CONVENTIONS.md` §10 (the `y_test` asymmetry) and §12 (grids, 'ij' storage).
- `examples/predict/darcy_plot.rs`, the model to follow, and `examples/predict/poisson.rs`
  (loading a Poisson run, decoding, T_uc).
- `src/neural_operators/data/loaders/poisson.rs` (`load_poisson_cheb`, `read_samples`,
  `check_nodes`); `chebyshev::{uniform_to_cheb, apply}`.
- The gnuplot crate at 0.0.46: `Axes2D::points` and `ColorType::VariablePaletteColor`.

## Goal
A picture of what the FNO learned: for one test sample of a saved Poisson run, the forcing
f, the reference u, the prediction û and the error û − u, all on the CGL grid.

## Deliverables
1. **`load_poisson_cheb_field(path, field, n, count) -> Result<Array3<f64>, LoadError>`**
   in `data/loaders/poisson.rs`: the first `count` samples of `field` (`"f"` or `"u"`),
   `[count, n, n]`, with the validation `load_poisson_cheb` has today.
   `load_poisson_cheb` becomes a one-line call to it with `"u"`; its behaviour, docs and
   tests do not change.
2. **`examples/predict/poisson_plot.rs`**, registered as `poisson_plot`, no required
   features: `cargo run --release --example poisson_plot -- runs/<dir> [<sample>]`
   (sample index into the test split, default 0).
   - Loads the model, `data_cfg.json` and `y_normalizer.json` as `predict_poisson` does,
     runs the one sample on `Device::default()` and decodes it to f64.
   - û = T_uc(prediction) with `TRANSFER_DEGREE` (as error 4); f and u from
     `load_poisson_cheb_field`.
   - A 2 × 2 multiplot: f, u, û, û − u. Each panel draws the n² CGL nodes at their (x, y)
     coordinates as filled square points coloured by value (`VariablePaletteColor`),
     with equal axes on [−1, 1]². u and û share one colour range; the error panel uses a
     symmetric range about 0. The title names the run, the sample and that sample's
     error 4.
   - Writes `prediction.png` into the run directory. If gnuplot is missing, it warns
     and exits 0, as `darcy_plot` does.
3. **A p = 0 run at 65² on metal**, regenerated with unchanged code, kept in the main
   checkout's `runs/` (decision 18). Its train / test relative L² is recorded next to
   Phase 3's 3.45e-3 / 1.116e-2, and the figure is made from it.
4. Docs: the `poisson_plot` command in `CLAUDE.md`, and the Layout line for
   `load_poisson_cheb_field`.

## Tests
In `data/loaders/poisson.rs`: `cheb_field_loader_reads_f` (shape and values of `f`
against the fixture) and `cheb_field_loader_rejects_unknown_field` (a `LoadError`). The
existing `cheb_loader_*` tests pass unchanged.

The example has no unit tests (as `darcy_plot`); it is checked by running it.

## Acceptance
- All tests pass under `cargo test` and `cargo test --features chebyshev`; nothing new
  needs `chebyshev`.
- `poisson_plot` run on the regenerated p = 0 run, for sample 0 and one other sample,
  with `--features metal`. The figure is inspected: û resembles u, the error map is
  small and roughly zero on ∂Ω, and the nodes cluster towards the edges.
- The T1 Results row of `docs/phase4/README.md` filled in.

## Checks to run and report
All of `CLAUDE.md`'s CI checks and `cargo clippy --all-targets -- -D warnings`, then
clippy and tests with `--features chebyshev`. Give the exact test filters
(`neural_operators::data::loaders::poisson::tests::<name>`) and check the "running N
tests" lines. Say whether the example was run with `--features metal`, and whether
gnuplot was present.

## Do not
- Change `load_poisson_uniform`, `load_poisson_cheb`'s behaviour, the model or the
  training code.
- Plot on the node index or interpolate onto a uniform grid for display.
- Commit `prediction.png` or the run.
