# T24: documentation pass

Phase 4s, task 24. Branch `phase4s/T24-docs-pass`. Depends on T1–T23. It goes last because
it touches many files. Once it merges, the phase can close (README exit checklist).

> **Refresh at start:** every location below is from `6eb3212`. Re-grep each one on `main`;
> many will have moved or disappeared in earlier tasks.

## Read first
- Review §5 S5, §6 T4-3, T4-8, T7-1, T7-2, T7-3, §7 Wave E5.
- CLAUDE.md "Working rules": document every public item, with tensor shapes and panics.
- The README's exit checklist.

## Goal
After 23 structural PRs, bring the documentation up to the code: no dead citations, every
public item documented, crate docs that describe the crate, and CLAUDE.md that matches the
code.

## Deliverables
1. **`REVIEW.md` / `REVIEW1.md` citations:**
   - 19 code citations at `6eb3212`; neither file exists:
     - `tests/run_artifacts_without_gnuplot.rs:1`
     - `losses/data_losses.rs:374`
     - `layers/spectral_convolution.rs:734`
     - `training/trainer.rs:289, 384, 463`
     - `models/fno.rs:149, 466, 518`
     - `data/batcher.rs:165`
     - `data/device_batcher.rs:215`
     - `data/io/readers/mat.rs:41, 117, 153`
     - `data/transforms/normalizers.rs:319, 397, 436`
     - `data/loaders/darcy.rs:213, 247`
   - Replace each with the CONVENTIONS §n or PR number it stands for. Find the PR with
     `git log -S` on the cited text.
   - Leave the mentions in `docs/` (the review and this README) as history.
2. **Rustdoc for every surviving public item.** In particular, check:
   - `FNOConfig`: a `//` comment instead of a doc comment, and its fields
   - `FNO` and `FNOConfig::init`, with its panics
   - `training_loop`, `TrainingComponents`, the fields of `EpochSums` / `TrainingConfig`
   - `LpLoss::new`

   Use `cargo doc --no-deps` plus a temporary `#![warn(missing_docs)]` locally to find
   gaps (don't commit the lint unless the author wants it).
3. **Crate docs:**
   - `lib.rs:1-18` claims an `FNO<5>`, lists only four modules and mentions only the
     Burgers/Darcy examples. Rewrite it for the post-4s layout: the modules, the feature
     (T16's name) and the Poisson examples.
   - `models/mod.rs:3-4` mentions a "shared model trait" that doesn't exist.
4. **`fno.rs` convention prose** (`fno/mod.rs` after T20), at the old :34-51 and :176-251.
   Replace restated CONVENTIONS §2/§3 prose with "CONVENTIONS §n". Keep only the notes
   specific to the implementation: the Metal channels-last reason and the 1-ulp contract.
5. **Remove `FNOConfig::init`'s `out_channels == 1` assert** (fno.rs:75-78). It enforces a
   trainer restriction that `flatten_pair` already checks. Update the docs on both sides,
   and check that `flatten_pair`'s doctest (`compile_fail,E0080`) still says so.
6. **`LpLoss` cleanup** (`losses/data_losses.rs`):
   - `forward` :104 only calls `rel` and has no production caller. Delete it.
   - `d` is taken by `new` but ignored by `rel`. Make it an argument of `abs` and drop it
     from the struct.
   - `abs`, `Mean` and `None` have no production callers. The minimum is the two changes
     above. Pruning further is the author's call, because `LpLoss` is a documented port
     (CONVENTIONS §11). Ask at the start.
   - Callers of `LpLoss::new(d, …)`: trainer, `evaluate`, `learner.rs`, `tests/gpu_parity.rs`,
     `metrics/poisson.rs` test.
   - CONVENTIONS §11 prose follows; it is outside the version-bumping set.
7. **CLAUDE.md:**
   - Update "Layout", "Commands", "Checks" and "Build environment" for the post-4s code.
   - The "Current phase" line names what comes next (Phases 5 and 6).
   - Line by line, check that each listed module, function and example exists.

## Tests
- No behaviour change except removing the assert.
- **Add** a test that `FNOConfig::init` with `out_channels = 2` builds. Training such a
  model still fails, in `flatten_pair`.
- The `LpLoss` tests are updated for the new `abs(d, …)` signature.

## Acceptance
- `git grep -n 'REVIEW1\?\.md' -- src tests examples` finds nothing.
- `cargo doc --no-deps` gives no warnings. A local `missing_docs` run gives no warnings, or
  only the ones listed in the PR.
- README exit checklist: "No blanket `#[allow]`, no `REVIEW.md` citations, every public
  item documented" and "CLAUDE.md … matches the code" can be ticked.
- Fill the Results row "Repository LOC at `c63eb32` against the phase's last merge", using
  the review's method (in-scope files only).

## Checks to run and report
The six from the README **plus the feature pair** (T16's name). Give the line delta.

## Do not
- Make structural refactors. If you find one, note it for the re-review.
- Change any CONVENTIONS rule (prose updates only).
