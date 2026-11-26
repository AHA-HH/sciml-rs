# CL3.4 One sidecar constructor for 2D and 3D

Started 2026-11-26 on `laplace3d/cl3-4`. Roadmap: "**CL3.4 One sidecar constructor for 2D
and 3D.** CL3.2 review F1: extract the `SidecarRecord` construction into
`pde/poisson/dataset.rs`, used by `dataset3d.rs`." *Needs:* CL3.2 (met). Proposal: §10;
§2 and §1.4 (constraint 1), §6 (2D modules gain 3D siblings, reuse over copies).

The author's go to edit the 2D sidecar-building function body (`write_split` in
`pde/poisson/dataset.rs`), which the roadmap line asks for, was given 2026-11-26 (relayed
by the coordinating session), with the constraint that the 2D exact-bytes test
`files_round_trip` passes unedited, and the whole datagen suite; F2 (a shared label-check
rule) stays not taken.

## Goal
One function in `dataset.rs` builds and writes the JSON sidecar of a generated split, 2D or
3D; `write_split` and `write_split_3d` both call it.

## Acceptance (roadmap *Done when*, verbatim, and the proposal's matching text)
- Roadmap: "`dataset3d.rs` builds no sidecar field by field; the 2D exact-bytes sidecar
  test (`files_round_trip`) passes unedited (constraint 1)."
- Proposal §1.4 / §2, constraint 1: every 2D output stays bit for bit identical.

## Read first
`pde/poisson/dataset.rs` (`write_split`, `files_round_trip`), `pde/poisson/dataset3d.rs`
(`write_split_3d`, `files_round_trip_3d`), `data/loaders/poisson/mod.rs`
(`SidecarRecord`).

## Changes
- `dataset.rs`: new `pub(super) struct SidecarFacts<'a>` (what differs per split and per
  dimension: the spec, K in effect, the label check and whether it was enforced, error 2,
  the wall time, and the GRF's α, RNG and ε_K, the last as its function, which the writer
  evaluates at K: review F2) and `pub(super) fn write_sidecar(path, name,
  git_commit, facts)`, which holds the one `SidecarRecord { … }` literal, the derived
  fields (s = n − 1, d, the tolerance, τ, K_MAX, the versions) and the JSON write.
  `write_split`'s body calls it with the 2D α, RNG and ε_K.
- `dataset3d.rs`: `write_split_3d` calls it with the 3D α, RNG and ε_K; the imports of
  `SidecarError2`, `SidecarLabelCheck`, `CONVENTION_VERSION`, `K_MAX`, `TAU` go where
  unused. Its module doc names the shared writer.
- No public API change, no new dependency, no convention change. The 2D `Split` and
  `Split3d` structs stay as they are (a generic `Split<D>` would edit the 2D struct, beyond
  the go).

## Tests that prove it
- `dataset::tests::files_round_trip`, unedited: the 2D sidecar's exact bytes against the
  untyped writer of phase 4s T19 (bit identity).
- `dataset3d::tests::files_round_trip_3d`: every field of the 3D record; it gains the two
  fields it did not check (`crate_version`, `wall_time_s` > 0).
- The whole `cargo test --features datagen` suite.
A refactor has no test that fails first; the guards pass before and after.

## Checks
`/verify` (fmt, the CI clippy jobs, `cargo clippy --all-targets`, datagen clippy,
`cargo test`, `cargo test --features datagen`, `cargo doc`), with
`CARGO_TARGET_DIR=~/sciml-rs-target/cl3-4`. Metal: not relevant (host code only).

## VM runs
None.

## Constraints and risks
Constraint 1: the 2D sidecar bytes are guarded by `files_round_trip`; the `.npz` writing
is not touched. The 3D sidecars of CL3.2 already on disk stay valid: the record type and
field values do not change.

## Pending runs
None.

## Open questions
None.

## Outcome
Done 2026-11-26. Plan review PASS; code quality APPROVE, A− (F2, F3 taken; F1, a shared
stats struct inside the 2D `Split`, left: it edits the 2D struct, beyond the go).
