# T11: one batching path

Phase 4s, task 11. Branch `phase4s/T11-one-batching-path`. Depends on T10.
- Can run in parallel with T12. Both edit `training/trainer.rs` (T11 its tests, T12
  `training_loop`), so whichever merges second rebases.
- T13 comes after.

> **Refresh at start:** re-read the four predict examples after T10's migration, and check
> `data/mod.rs` against T5 and T23.

## Read first
- Review §5 S2 ("Batching" row), §6 T2-4, §7 step B2.
- `data/device_batcher.rs` (`device_loader` :144-158 and its bit-identity tests),
  `data/batcher.rs` (`OperatorBatcher` :28-114, `Batch` :13-16), `data/dataitem.rs`
  (`DataItem` :41-44, `HostFloat` :22), `data/dataset.rs` (the `Dataset` impl :48-67).
- CLAUDE.md "Burn": an `OperatorBatcher` loader without `set_device` puts batches on
  `Device::default()`.

## Goal
`device_loader` replaced the host batcher, and training already uses only it. Only four
predict examples keep `OperatorBatcher` alive. Move them over, and keep the old batcher
only as the oracle that the bit-identity tests compare against.

## Deliverables
1. **Migrate the predict examples to `device_loader`:** `predict/burgers.rs:53`,
   `darcy.rs:57`, `burgers_resolutions.rs:72` and `burgers_superresolution.rs:79`.
   - Evaluate on `device.inner()`, as the trainer does (CLAUDE.md "Burn").
2. **`OperatorBatcher` becomes test-only.** Move it into `device_batcher.rs`'s test module
   (or `#[cfg(test)]` in `batcher.rs`).
   - It stays as the oracle for the existing bit-identity tests (`host_loader` :183) and
     the trainer tests (:371, :450, :501, :506).
   - Wherever `Batch` ends up (:13-16, used by trainer.rs, learner.rs and
     device_batcher.rs), it stays public.
3. **Delete** `DataItem` and `impl Dataset for OperatorDataset`.
   - `darcy_plot.rs:54` uses `Dataset::get(0)`. Replace that with the `inputs()` accessor
     and a slice. T13 later moves `darcy_plot` onto `predict_host`.
   - `.len()` callers (trainer.rs:256, device_batcher.rs:157) use an inherent `len`.
4. **Rename** `data/dataitem.rs` to `data/host_float.rs`, since only `HostFloat` is left.
   Fix every `dataitem::HostFloat` import (the loaders, trainer, device_batcher, and the
   fno.rs tests at about :1226).
5. CLAUDE.md: the "Burn" bullet about `OperatorBatcher` changes to describe
   `device_loader`.

## Tests
- `device_batcher`'s bit-identity tests and the trainer tests pass, now with the
  test-only oracle.
- Tests for the deleted `Dataset` impl are listed as deliberate deletions; expect few or
  none.

## Acceptance
- `predict_burgers` and `predict_darcy` print the same test L² as recorded before T10.
  Fill the Results row.
- If evaluation moves from host to device batching and the result differs in the last f32
  digit, report the difference and explain it. Don't hide it.

## Checks to run and report
The six from the README. For Metal, run `predict_burgers --features metal` once and say
whether you did. The chebyshev pair is not needed.

## Do not
- Change `eval_epoch`'s contract or build `evaluate` (T13).
- Touch the Poisson examples, which already use `predict_host`.
