# CO0.4 Run provenance: the commit at the start, one directory per run

Started 2026-11-27 on `optimisers/co0-4`. Roadmap: "CO0.4 Run provenance: the commit at
the start, one directory per run" (two faults CO0.1 hit; follow-ups of CO0.1, the author,
2026-11-26, outside O0's gate). Proposal: §1.4, §2 (constraint 1), §10.

## Goal
A run records the commit it started from, and two runs of one name started in the same
second get two directories.

## Acceptance (roadmap *Done when*, verbatim)
"both examples read the commit before training; a test creates two run directories of one
name in the same second and gets two directories."

## Read first
`src/neural_operators/runs/mod.rs` (`RunDir::create`); `examples/train/common.rs`
(`Measured`, `run_costs`); `examples/train/{burgers,darcy,poisson}.rs` (where `Measured`
is built); `tracking/builder.rs::start_ms` (parses the directory's last `_` segment).

## Changes
- **`runs/mod.rs`**: `RunDir::create` creates `root` with `create_dir_all`, then
  `<name>_<unix_secs>` with `create_dir`; on `AlreadyExists` it tries
  `<name>_<unix_secs>_<k>`, k = 1, 2, …. The body moves to a private
  `create_at(root, name, stamp)` so a test can fix the second. Public API unchanged.
- **`examples/train/common.rs`**: `Measured` gains `git_commit`, which `run_costs` writes
  instead of calling `git_commit()` at the end.
- **`examples/train/{burgers,darcy,poisson}.rs`**: `let git_commit = git_commit();` before
  the session starts (new run or `--resume`), passed in `Measured`. Darcy shares
  `Measured`, so it gets the fix too. `poisson.rs`: two lines (CO2.1 edits it in parallel).
- Conventions: none.

## Tests that prove it
- `runs::tests::create_twice_in_one_second_gives_two_directories`: `create_at` twice with
  one stamp gives `<name>_<stamp>` and `<name>_<stamp>_1`, both directories, distinct; a
  third gives `_2`.
- The commit: an example change, no unit test (examples have none); checked by reading
  the code (read before `session`/training) and by a flex smoke run whose `costs.json`
  carries the commit.

## Checks
`/verify` (fmt, the clippy runs, `cargo test`, `cargo doc`, datagen clippy and tests).
Constraint 1: `train_burgers` (Adam) two epochs on flex before (`origin/main`) and after:
`metrics.csv`, configs and `costs.json` equal, `costs.json`'s `git_commit` aside.

## VM runs
None.

## Constraints and risks
- Constraint 1: no model output moves; directory names change only on a collision.
- `tracking::builder::start_ms` reads the last `_` segment as the start second, so a
  collided run (`_<k>`) exports with start time k s after 1970. Outside this task's
  modules: a roadmap follow-up.
- `train_poisson3d` and the profile/predict examples also call `git_commit()` at the end;
  `poisson3d.rs` belongs to laplace3d (CL4.2 runs it now): a follow-up there, not here.

## Pending runs
None.

## Open questions
None.
