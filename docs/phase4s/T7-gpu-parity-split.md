# T7: `tests/metal_diag.rs` split

Phase 4s, task 7. Branch `phase4s/T7-gpu-parity-split`. Depends on T1. Can run fully in
parallel with any task.

## Read first
- Review §6 T2-12, §7 step A6.
- `tests/metal_diag.rs` (1,674 lines, gated `#![cfg(any(feature = "metal", feature =
  "cuda"))]` at :11), especially the helpers `rel` :38 and `cmp` :51-70.
- `upstream/burn/cfft-kernel/README.md` and
  `upstream/burn/fused-cat-reused-at-new-shape/README.md`. They cite test names, not the file.
- The memory notes on the Metal cfft and fusion bugs, and `spikes/solver/Cargo.toml` (how a
  spike package is set up).

## Goal
Of 33 tests, 7 assert. The other 26 print "<-- BAD" and never fail, and CI never compiles
any of them. Keep the 7 regression tests where CI can run them. Move the bisection notebook
out of `tests/`.

## Deliverables
1. **`tests/gpu_parity.rs`** (about 250 lines), holding the asserting tests with their
   helpers:
   - `repro_fused_cat_broadcast` :821, `repro_cfft_bluestein` :835
   - `repro_fused_cat_broadcast_flex` :863, `repro_cfft_bluestein_flex` :877
   - `repro_fusion_cache_new_shape` :1403, `repro_fused_cat_reused_at_new_shape` :1524
   - `replay_2d_poisson` :1595

   Keep the test names: the upstream READMEs cite them.
2. **Run the `*_flex` twins in CI.**
   - Put them outside the GPU gate: a GPU `#[cfg]` on the GPU tests only, not
     `#![cfg]` on the file.
   - Check whether the twins compare Flex against itself or against a reference. If it's
     against itself, say so in the PR. They still guard against Flex regressions.
3. **`replay_2d_poisson`** reads the `PADS` environment variable (:1597). Give it a fixed
   default so it asserts the same thing with no environment set, and keep `PADS` only as an
   override. Write the default into its doc comment.
4. **The other 26 tests** move to `spikes/metal_diag/`, a standalone package like
   `spikes/solver`, with a `path` dependency on the crate and the `metal` feature. Its
   README explains how to run it and that CI never builds it.
   - Alternatively, delete them and leave them to git history: give the last commit that
     had them in the PR.
   - Prefer the spike if they still compile cheaply; otherwise delete them.
5. **Delete `tests/metal_diag.rs`.**
6. **Docs:**
   - Update any `metal_diag` mention in `upstream/burn/*` (expected: none), CLAUDE.md and
     `docs/`, which mention it in Phase 3/4 READMEs. Change only live instructions, not
     historical results.
   - Add a CLAUDE.md "Checks" line: `cargo test --features metal --test gpu_parity` for
     backend-sensitive changes.

## Tests
- `cargo test` now also runs the two `*_flex` twins (+2 tests) and anything else that
  became ungated.
- **Deliberate deletions or moves:** the 26 diagnostic tests.

## Acceptance
- `cargo test --features metal --test gpu_parity`: run it on this Mac and report the result.
  **Say whether Metal was run.** The expected outcome comes from the current Burn pin. If a
  repro is expected to fail until upstream fixes the bug, check how it is handled today
  (`#[ignore]`, inverted assert) and keep that.
- `wc -l tests/*.rs` shows `gpu_parity.rs` at about 250 lines.

## Checks to run and report
The six from the README, plus the Metal run above. The chebyshev pair is not needed.

## Do not
- Change Burn's pin or the fixes in `FNO::grid_cl`.
- Rename the cited repro tests.
