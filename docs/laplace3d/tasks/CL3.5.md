# CL3.5 Exact 2D generator oracle

Started 2026-11-27 on `laplace3d/cl3-5`. Roadmap: "**CL3.5 Exact 2D generator oracle.** L3
gate review finding 4: a test that pins the 2D `generate()`'s `f` and `u` bit for bit, in
process (D2), on a small fixed split, beside the exact-bytes `files_round_trip`.
Test-only." *Needs:* CL4.2, waived by the author 2026-11-27 (CL4.2's s = 64 runs are on the
VM until about 14:30 UTC 2026-11-28, and CL3.5 touches none of CL4.2's files or runs).
Proposal: §10; §1.4 and §2 (constraint 1), §9 (D2).

## Goal
A test in `pde/poisson/dataset.rs` that fails if any later change (CL3.6 above all) alters
one bit of what the 2D `generate()` returns.

## Acceptance (roadmap *Done when*, verbatim, and the proposal's matching text)
- Roadmap: "the test passes on `main`'s code and fails on a one-ulp change to a label."
- Proposal §2, constraint 1 (D2): the 2D numerical outputs stay bit for bit identical,
  checked in process, old against new, on fixed 2D inputs.

## Read first
`pde/poisson/dataset.rs` (`generate`, `files_round_trip`); the D2 precedent
`metrics/poisson/tests.rs::poisson_errors_frozen` (CL4.1); L3 gate report finding 4
(`docs/laplace3d/gates/L3.md`).

## Changes
Test-only, in `dataset.rs`'s `mod tests`:
- `generate_frozen`: `generate()`'s body as on `main` at 94ef52e, verbatim but for the
  wall time, calling `label_check_frozen` (`label_check`'s body at 94ef52e, since CL3.6
  merges the 2D and 3D label checks and no rank-2 test pins it). Do not edit. They keep
  calling the primitives `draw_xi`, `forcing_coeffs`, `evaluate`, `solution_coeffs`,
  `CollocationSolver::solve_full`, which CL3.6 does not own, and `cc_rel_l2_error` and
  `Transfers`, which have their own rank-2 bit-identity tests; `f` and `u` depend on
  neither of the last two.
- `split_bit_difference`: the first difference between two `Split`s, every field but the
  wall time by `to_bits` (so −0.0 ≠ +0.0), naming the field and flat index; `Split` is
  destructured exhaustively, so a field CL3.6 adds does not compile until it is compared
  (code-quality review F1).
- `generate_is_bit_identical_to_frozen`: two fixed splits, n = 17 at the default K (8,
  label check recorded) and n = 65 at an explicit K = 16 (label check enforced). A control
  in the same test: the split with one `u` entry moved one ulp is caught, at that index.
No 2D function body edited; no public API, dependency or convention change.

## Tests that prove it
- The new test passes on `main`'s code.
- The failing control, twice: in the test (one ulp on a clone of the split), and
  once by hand: `generate()` temporarily moves one label entry one ulp
  (`f64::next_up`), the test fails, the edit is reverted; the failing output goes in the PR.
- `files_round_trip` and the whole datagen suite unchanged.

## Checks
`/verify` with `CARGO_TARGET_DIR=~/sciml-rs-target/cl3-5`: fmt, the CI clippy jobs,
`cargo clippy --all-targets`, datagen clippy, `cargo test --features datagen`. Metal: not
relevant (host code only).

## VM runs
None.

## Constraints and risks
Constraint 1: nothing in the library changes; the test only adds a guard. The oracle is
in process, so it holds on Accelerate and OpenBLAS alike; it does not pin values across
platforms (no fixture), which D2 does not ask.

## Pending runs
None.

## Open questions
None.
