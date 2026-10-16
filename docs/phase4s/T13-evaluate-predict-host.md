# T13: public `evaluate` and a rank-generic `predict_host`

Phase 4s, task 13. Branch `phase4s/T13-evaluate-predict-host`. Depends on T11. T17 waits for
it.

> **Refresh at start:** re-read `eval_epoch`, the four Burgers/Darcy predict examples after
> T10/T11, and `models/run.rs::predict_host`.

## Read first
- Review §5 S3 (third bullet), §6 T4-1, T4-2, T2-11, §7 step B4.
- `training/trainer.rs::eval_epoch` (:112-138 before T3). It returns a **sum** that the
  caller must divide, and each of its 4 example callers builds `LpLoss::new(d, 2, Sum)`
  with a hard-coded `d`.
- `models/run.rs::predict_host` (:45-73). It is fixed to `FNO<4>` and `Array3`.
- `examples/predict/burgers_resolutions.rs` (registered as `burgers_resolution`),
  `burgers_superresolution.rs`, `darcy_plot.rs` (a hand-rolled forward pass at :61-75),
  and `poisson_resolution.rs` (the argv pattern to follow).

## Goal
Each predict example re-implements "evaluate the mean relative L²", and two Burgers examples
are about 85% the same file. Give evaluation one entry point and inference one rank-generic
function.

## Deliverables
1. **`pub fn evaluate::<const R, const RM1>(model, loader, decoder: <the current hook type>)
   -> f32`**:
   - It returns the per-sample mean, and builds `LpLoss::new(R - 2, 2, Sum)` itself.
   - `eval_epoch` (the sum) becomes private to `training_loop`.
   - Document that the result is a per-sample mean and give the loader's batch shapes.
   - The hook type stays whatever T3 left (`Postprocess`). T17 simplifies it.
2. **`predict_host::<const R>(&FNO<R>, ArrayViewD<f32>, batch, &Device) -> ArrayD<f64>`**:
   - It replaces the `FNO<4>` / `Array3` version.
   - Its Poisson callers (`predict/{poisson:145, poisson_plot:119, poisson_resolution:85}`)
     add `.into_dimensionality::<Ix3>()`, or a thin typed wrapper if that reads better.
   - Keep the docs on output shape and dtype.
3. **One `burgers_resolution` example:**
   - Merge `burgers_resolutions.rs` and `burgers_superresolution.rs` into
     `examples/predict/burgers_resolution.rs`. The file name now matches the registered
     example name.
   - Rates come from argv: `-- runs/<dir> <rate> [<rate> ...]`. The default rates
     reproduce what each old example printed.
   - Remove the `burgers_superresolution` `[[example]]` entry in `Cargo.toml`, and update
     CLAUDE.md "Commands".
4. **Migrate the callers:**
   - `predict_burgers`, `predict_darcy` and `burgers_resolution` call `evaluate`.
   - `darcy_plot` calls `predict_host::<4>` instead of its manual forward pass.

## Tests
- `evaluate_is_mean_of_eval_epoch`: on a tiny model, `evaluate` equals the old sum divided
  by n.
- `predict_host` gets a rank-3 (1D) test next to the existing rank-4 tests in `run.rs`.
- The `eval_epoch` tests stay, since the function is now private.

## Acceptance
Each example reproduces its previous numbers on an existing run, recorded before starting:
- `predict_burgers` and `predict_darcy`: test L².
- `burgers_resolution`: every rate printed by both old examples.
- `darcy_plot`: the same figure.
- `predict_poisson` on `runs/poisson_fno_p4_1791922918`: the same five errors.

Fill the Results rows.

## Checks to run and report
The six from the README. The chebyshev pair is not needed.

## Do not
- Change the train/eval decoding asymmetry. That is T17, which builds on this task.
- Change `LpLoss` (T24).
