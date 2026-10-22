# Thermo-nuclear code quality review: Phase 4s T2 (Learner showcase)

| | |
| --- | --- |
| Repo | sciml-rs (`bichir` worktree) |
| Branch | `phase4s/T2` (uncommitted working tree; no commits ahead of base) |
| Base | `0d71c3b` (merge-base with `main`) |
| Date | 2026-10-19 |
| Diff | 8 files, +143 / −78 |
| Brief | `docs/phase4s/T2-learner-showcase.md` |

Checks run during the review: `cargo test --lib neural_operators::training` (12 passed),
`cargo clippy --all-targets -- -D warnings` (clean), `cargo fmt -- --check` (clean). Not run:
`cargo doc`, the `--features metal` runs, and the brief's acceptance run
(`train_burgers_learner` → `predict_burgers`). The README Results row for T2 is still empty.

## Verdict: APPROVE (with follow-ups)

The change does what the brief asks, and it deletes more than it moves. There is now one loss
definition, `forward_loss`, and every training, evaluation and Learner path calls it. The
`loss_fn: &LpLoss` parameter is gone from `eval_epoch` and `train_epoch`, as are the
`TrainingComponents::loss_fn` field and the four `LpLoss::new(d, …)` literals with a
hard-coded `d` in the predict examples. That was the right thing to delete: as long as
callers passed their own `LpLoss`, the loss was still defined in more than one place. The
Learner example now writes a run directory that `predict_burgers` can read, and
`burgers_learner.rs` gets no new branches.

There is one visible duplication: the head of `write_run_artifacts`, copied into the
example. The brief allows it explicitly ("write the three configs directly"), and T10 is
scheduled to remove it, so it is not a blocker. One type contract is weak: `forward_loss`
returns an anonymous triple holding two tensors of the same type. Fix that before T13 and
T17 add more callers.

## Presumptive blockers

None met.

- **Canonical-helper duplication** (`examples/train/burgers_learner.rs:88-107` repeats
  `metrics/io.rs:77-89`). Justified: the brief allows it, the only clean fix (splitting
  `write_run_artifacts`) is outside this task's modules and would get ahead of T10's
  `RunDir`, and T10's brief already says "Migrate `burgers_learner` too".
- No file crosses 1,000 lines. No new branching is added to shared flows.

## File sizes

| File | Before | After | Note |
| --- | ---: | ---: | --- |
| `src/neural_operators/training/trainer.rs` | 798 | 850 | Under 1k; about half is tests (see 5.1) |
| `src/neural_operators/training/learner.rs` | 182 | 183 | |
| `examples/train/burgers_learner.rs` | 107 | 124 | |
| `examples/predict/burgers.rs` | 68 | 66 | |
| `examples/predict/burgers_resolutions.rs` | 85 | 82 | |
| `examples/predict/burgers_superresolution.rs` | 100 | 98 | |
| `examples/predict/darcy.rs` | 77 | 75 | |
| `CLAUDE.md` | 167 | 171 | |

## Findings

### 1. Structural regressions

None. The reach of the change is net positive. Removing `loss_fn` goes slightly beyond the
brief's literal deliverable, but it follows from "the loss is defined once" and does part of
T13 early (see 6.2).

### 2. Missed simplifications

**2.1 The head of `write_run_artifacts` is copied into the example.**
`examples/train/burgers_learner.rs:88-107` vs `src/neural_operators/metrics/io.rs:77-89`.
The two copies of the stamp, the `runs/<name>_<secs>` join and the three config saves are
identical. The behaviour differs in one deliberate way: the example saves the configs
*before* training, so a run stopped with `q` is still described. That is an improvement
`write_run_artifacts` lacks. Within this task the copy is the right call. The longer-term fix
is to split `write_run_artifacts` into `create(name, cfgs…) -> dir` and
`write_metrics(dir, metrics)`, which is T10's `RunDir::create` / `save_metrics` under another
name. **Remedy for T10:** have `RunDir::create` save the configs at creation time, which
the Learner already does. `train_burgers` and the Learner then use the same call, and the
"configs survive an early stop" property applies to every trainer. Add a line to the T10
brief so the ordering is kept on purpose rather than lost by accident.

### 3. Spaghetti / branching

None added. `train_epoch` actually gets shorter. The forward → flatten → post → loss block
is now one call (`trainer.rs:195`), and MSE becomes a single line on `out.inner()` /
`target.inner()` instead of two named clones (`trainer.rs:197`).

### 4. Boundary and type contracts

**4.1 `forward_loss` returns a positional `(Tensor<1>, Tensor<2>, Tensor<2>)`**
(`trainer.rs:126-136`). Two of the three fields have the same type, so a caller that writes
`let (l2, target, out)` compiles and silently swaps prediction and target. The MSE metric in
`train_epoch` would not notice, because the difference is squared. The Learner would notice,
since `RegressionOutput { output, targets }` would be swapped in what Burn reports. This
function will gain callers in T13 (`evaluate`) and be reshaped in T17 (`Option<&FlatDecoder>`),
so now is the cheap moment to name the contract:

```rust
/// One batch through the model, flattened and post-processed (shapes `[b, n_points]`).
pub struct StepLoss {
    /// Relative L2 summed over the batch, `[1]`.
    pub l2_sum: Tensor<1>,
    pub out: Tensor<2>,
    pub target: Tensor<2>,
}
```

The call sites then read `forward_loss(..).l2_sum` (eval),
`let StepLoss { l2_sum, out, target } = …` (train), and
`RegressionOutput { loss, output: s.out, targets: s.target }` (Learner). Burn's
`RegressionOutput` is *not* a good substitute: its `loss` is per-sample in `learner.rs`, so
reusing it here would give the same field two meanings.

**4.2 `forward_loss` is `pub`, but every caller is in the crate** (`trainer.rs:126`). The
brief says "`pub fn`", but `trainer`'s public surface is already wide (`flatten_pair`,
`identity`, `Postprocess`, `eval_epoch`, `train_epoch`, `training_loop`,
`build_training_components`, `train_fno`), and T13 makes `eval_epoch` private. Making the
new step `pub(crate)` from the start avoids adding a public item that T13 or T17 would then
have to deprecate or reshape. Rustdoc links from `learner.rs` still resolve. Low cost, but
it is the author's call against the brief's wording.

### 5. File size and decomposition

**5.1 `trainer.rs` is 850 lines, about 420 of them tests.** It is not near the limit, but
the test module now holds four unrelated groups: `forward_loss`, schedule/divisors,
device-vs-host loaders, and `train_fno` plus the Poisson smoke test that T3 moved here. Each
group builds its own tiny fixtures (`tiny_dataset`, `tiny_dataset_2d`, `to_batch`, the
inline `probe` at `:638-645`). This is not for T2. Flag it for T12 (epoch observer), which
will add more to this file: split the tests out as `training/trainer/tests.rs` before that
lands.

### 6. Modularity and canonical helpers

**6.1 `to_batch` is a bespoke uploader** (`trainer.rs:480-492`). It converts an
`OperatorDataset<f64>` to an f32 `Batch` on a device. That is what
`device_loader::<R, RM1, T>(&data, device, data.len(), None)` already does, followed by
`.iter().next()`. The same f64→f32 `TensorData` closure also appears inline at `:640-643`.
**Remedy:** take the single batch from `device_loader` in the new test. That deletes
`to_batch` and runs the test on the same upload path as training. Low severity, since it is
test code.

**6.2 The briefs are now stale (T13, T24).** This change already does part of T13: callers
no longer build `LpLoss`, and the loss is owned by the trainer. T13's "Read first" still says
"each of its 4 example callers builds `LpLoss::new(d, 2, Sum)` with a hard-coded `d`", and
T24's `LpLoss::new` caller list still names the examples. Both briefs carry a "refresh at
start" note, so this is not a defect. Still, mention it in the PR description so the T13
author does not look for code that no longer exists.

### 7. Legibility and maintainability

**7.1 The new test only checks the composition with `identity`**
(`trainer.rs:458-507`). The oracle repeats the implementation's three lines, which the brief
asked for. But the one ordering decision `forward_loss` actually makes, *post before
loss, and the returned tensors are post-processed*, is never tested with a non-identity
hook. `train_fno_some_matches_decoder_path` cannot catch a regression there, because both of
its sides now go through `forward_loss`. **Remedy:** add one case with
`post = |o, t| (o * 2.0, t)` and assert `l2` equals `rel(2·o, t)` and that `out` is the
doubled tensor. That is about five lines, and it pins the contract T17 is about to change.
(Because `LpLoss::rel` ignores `d`, see `data_losses.rs:96` and T24, the `R - 2` argument is
not exercised by the test either. That is acceptable but worth knowing.)

**7.2 Two comments in `burgers_learner.rs` disagree.** The module doc (`:9-10`) says the
loss is "comparable to `train_l2` / `test_l2` printed by `train_burgers`". The new comment
(`:73-74`) says "its test L2 is not comparable with a `train_burgers` run". Both are true in
different senses (same metric, different training length), but a reader sees a
contradiction. Suggested wording: "the same metric as `train_burgers`, but after 100
epochs instead of 500, so the values differ."

**7.3 `predict_burgers` points to a file Learner runs don't have**
(`examples/predict/burgers.rs:63`). It prints "(should match the last test_l2 in
…/metrics.csv)". This PR makes Learner runs a supported input to `predict_burgers`, and
those runs have no `metrics.csv`. Either make the hint conditional on the file existing
(one `if dir.join("metrics.csv").exists()`), or leave it and have T10/T13's reworked predict
path handle it. Low severity.

**7.4 CLAUDE.md now mentions `train_burgers_learner` in two adjacent bullets**
(`CLAUDE.md:152-156`). The new bullet is correct, but the line above it still carries the
"needs a real terminal" parenthetical. Fold the two into one bullet that says the
example is a showcase, writes a run that `predict_burgers` reads, and needs a TTY.

## Next steps

Ordered by leverage. Each step preserves behaviour and is small enough to land in this PR
or as a follow-up commit.

1. **Name the `forward_loss` result** (4.1), and make it `pub(crate)` if the author agrees
   (4.2). This touches `trainer.rs` and `learner.rs` only.
   Verify: `cargo test --lib neural_operators::training` (12 tests, the count unchanged),
   `cargo clippy --all-targets -- -D warnings`, `cargo doc --no-deps`.
2. **Test the post hook in `forward_loss`** (7.1), and drop `to_batch` in favour of
   `device_loader` (6.1). Verify: the new test fails if `post` is moved after the loss
   (mutate locally to confirm), and the test count is +0 (new assertion in the existing test)
   or +1.
3. **Comment and doc fixes** (7.2, 7.4, optionally 7.3). Verify: `cargo clippy --examples
   -- -D warnings`.
4. **Acceptance run (outstanding, not a code change):** run `train_burgers_learner` in a
   real terminal, then `predict_burgers -- runs/burgers_fno_learner_<secs>`, and fill the
   README Results row. If no TTY is available, say so in the PR as the brief requires.
5. **Carry into later briefs** (no code in T2):
   - T10: `RunDir::create` saves configs at creation time (2.1).
   - T12: split `trainer.rs` tests out first (5.1).
   - T13/T24: refresh notes on `LpLoss` callers (6.2).
