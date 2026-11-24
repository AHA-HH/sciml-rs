# CL1.1 CONVENTIONS 3D additions

Started 2026-11-24 on `laplace3d/cl1-1`. Roadmap: "CL1.1 CONVENTIONS 3D additions.
Proposal §4, with a `CONVENTION_VERSION` bump. *Done when:* signed off by the author.
*Needs:* –. *Sign-off:* the additions." Proposal: §10; §3.2, §3.5, §4, §9.

## Goal
Write proposal §4 into `docs/CONVENTIONS.md` §12 as additions, with `CONVENTION_VERSION`
2 → 3, and put them to the author for sign-off.

## Acceptance (roadmap *Done when*, verbatim, and the proposal's matching text)
- Roadmap: "signed off by the author."
- Proposal §4: "§12: the 'ijk' storage; 3D transfers as the 1D T applied along x, then y,
  then z; the 3D Clenshaw–Curtis norm (tensor weights over three axes); the 3D boundary
  norm of Section 3.5. `CONVENTION_VERSION` 2 → 3, since §12 changes. Every 2D dataset
  file stays valid, as at version 2; the version a 2D sidecar records is metadata, outside
  D2's bit identity."

## Read first
`docs/CONVENTIONS.md` §2, §9, §12; proposal §3.2, §3.5, §4, §9 (D2); `src/lib.rs`
(`CONVENTION_VERSION` and its test against the CONVENTIONS header);
`chebyshev::transfer::apply` (2D order: x, then y); `chebyshev::clencurt::cc_norm_sq`;
`metrics::poisson::boundary_errors` (2D error 5).

## Changes
- `docs/CONVENTIONS.md`: a "Three dimensions (version 3)" block at the end of §12, before
  its Verification: Ω = [−1, 1]³, 'ijk' storage, transfers along x, then y, then z, the 3D
  CC norm, the 3D relative boundary RMS (six faces, area 24 against volume 8, edge and
  corner nodes once per face). The version line in the header and in §9 to 3; a version 3
  line in §9's History. Every existing statement unchanged.
- `src/lib.rs`: `CONVENTION_VERSION` 2 → 3 (its test requires the constant and the
  header to agree). No other code.
- Conventions: this task is the proposal; it lands only on the author's sign-off.

## Tests that prove it
- `convention_version_matches_conventions_md` (header and constant agree at 3).
- `dataset.rs` sidecar tests compare against the constant, not a literal; the loaders do
  not check a sidecar's version (CONVENTIONS §9 History), so every 2D dataset still loads.
- Constraint 1: no numerical code changes; the whole 2D suite (`cargo test`, `cargo test
  --features datagen`) passes unchanged. The only 2D effect is the `convention_version`
  a newly generated 2D sidecar records (3), metadata outside D2.
- No 3D tests: the conventions land before their code (§9); CL1.2 and CL1.3 add the tests
  to §12's Verification.

## Checks
`/verify` (fmt, clippy, `cargo test`, `cargo doc`, datagen).

## VM runs
None.

## Constraints and risks
Constraint 1 (proposal §1.4, D2): additions only. Risk: a 3D statement read as changing a
2D one; each is scoped to rank 3 explicitly. CL0.1 runs in parallel in code and examples;
this task touches only `docs/CONVENTIONS.md`, `src/lib.rs` and `docs/laplace3d/`.

## Pending runs
None.

## Open questions
See the PR: the 2D boundary norm's absence from CONVENTIONS, and the branch name.
