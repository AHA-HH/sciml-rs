# CL0.2 Rank-5 backend parity

Started 2026-11-24 on `laplace3d/cl0-2`. Roadmap: "**CL0.2 Rank-5 backend parity.**
`tests/gpu_parity.rs` cases for the 3D spectral layer and the 3D grid channels on metal
(the 2D cat and fusion bugs)". Proposal: §10; §8 R3; §7 (model sizes); §2 (rank-generic
`FNO<R>`, `SpectralConv<R>`).

## Goal
Show, before any 3D training on metal, that the rank-5 FNO pieces give the same numbers on
the GPU as on Flex, or record the Burn bug and a workaround.

## Acceptance (roadmap *Done when*, verbatim, and the proposal's matching text)
"the cases pass, or fail with a recorded Burn bug and a workaround."
Proposal §8 R3: "Burn backend bugs at rank 5 (the Metal cat and fusion bugs seen in 2D);
`tests/gpu_parity.rs` gains 3D cases (CL0.2) before 3D trains on metal."

## Read first
- `tests/gpu_parity.rs` (the 2D repros and `replay_2d_poisson`).
- `FNO::grid_cl` and `FNO::forward` (`src/neural_operators/models/fno/mod.rs`): the
  channels-last grid, built because a channels-first `cat` after a permute zeroed grid
  entries on Metal; the single-channel early return for the fused-cat-at-new-shape bug.
- `SpectralConv::forward` (`layers/spectral_convolution.rs`): rfft on the last axis, cfft
  on the others (Bluestein for non-powers of two), 2^(D-1) = 4 mode corners at rank 5.

## Changes
`tests/gpu_parity.rs` only, three new GPU tests (behind `metal` or `cuda`, as the existing
GPU tests), each comparing the GPU to Flex on the same weights and inputs. After the
code-quality review (round 1), `replay_2d_poisson`'s body moved into a rank-generic
`replay<R>` that both replays call; the 2D schedule, input, loss and tolerance are
unchanged (the 2D test computes the same values: its per-call errors at padding 4, 8 and
16 are identical before and after on Metal).

1. `grid_3d`: `FNO<5>` with `n_layers = 0` (lifting, grid channels, permutes and
   projection only, so the grid is isolated), evaluated at a sequence of non-cubic shapes,
   [2, 6, 5, 4, 1], [3, 4, 7, 5, 1], then [2, 6, 5, 4, 1] again (the chain replayed at a
   new shape and back). Every axis has its own extent, so a swapped or zeroed grid channel
   shows.
2. `spectral_conv_3d`: `SpectralConv<5>` alone, forward and the input gradient of
   mean(y²) through backward, at [2, 3, 14, 13, 12] (non-powers of two: Bluestein on both
   cfft axes), [2, 3, 16, 16, 16], the first shape again, then [2, 3, 13, 12, 14], whose
   rfft half-width is even (8; the 2D cfft bug appeared only at even trailing widths, a
   point of the plan review); modes [6, 6, 6].
3. `replay_3d_poisson`: `replay_2d_poisson`'s schedule at rank 5. `FNO<5>`, width 32,
   4 layers, modes [6, 6, 6], `SymmetricUniform`, padding 0 and 4 (0 keeps 16³, a power of
   two; 4 gives 24³, which goes through Bluestein); train
   batches 20, 20, 7 at s = 16, eval batches 20 and 3, an eval at the non-cubic
   [2, 14, 13, 12, 1] (resolution change), then train 20 and 7 and eval 20 again.

No public API, no conventions, no dependency changes. s = 16 and modes 6 (not D4's
s = 32, modes 8 or 12) keep the Flex reference fast on the Mac: the cases test the op
patterns at rank 5, which do not depend on size; CL0.1 measures the sizes.

## Tests that prove it
Each case against Flex, the trusted path (its rank-5 grid is pinned by
`grid_controls_3d`, its rank-5 spectral layer by `forward_3d_matches_numpy_reference`):
relative L² error ≤ 1e-5 (the 2D replay's tolerance) over the whole output, and for
the three 3D cases also max |GPU − Flex| / max |Flex| ≤ 1e-5, so a few zeroed entries
cannot hide in an L² norm. The 2D replay prints the max-entry error but asserts L² only,
so the 2D gate keeps its criterion from `main`.

## Checks
`/verify`; `cargo test --release --features metal --test gpu_parity` (all cases, the 2D
ones included, on the Mac's Metal GPU). cuda is not run (Mac only for this lane).

## VM runs
None. The cuda run of these cases is left for a later VM session (a follow-up, not a
condition of this task: the roadmap names metal).

## Constraints and risks
- Constraint 1: tests only; no 2D code changes, so every 2D output is unchanged by
  construction. The 2D replay test is restructured, not changed in what it checks.
- Risk: Metal at rank 5 differs from Flex by more than 1e-5 without a bug (deeper FFT
  chains in f32). If so, the error is reported with its size and the tolerance is a
  question for the author, not raised silently.

## Pending runs
None.

## Open questions
None yet.
