# Optimiser benchmarks progress log

Newest entries at the bottom. One entry per task, in that task's pull request, using this
template:

```
## YYYY-MM-DD: <ID> <title>
- Done:
- Tests:
- Checks: ran …; pending …
- Decisions: none, or D… signed off; questions for sign-off …
- Review: plan review (blocking items fixed); code quality APPROVE or BLOCKED → APPROVE (rounds, blockers fixed, findings left as follow-ups)
- Adam and Gauss–Newton: unchanged bit for bit (how shown), or not touched
- Follow-ups:
```

## 2026-11-24: Planning
- Done: reading notes on SOAP, Shampoo and Khodakarami et al. 2026
  (`docs/optimisers/references.md`); the proposal, draft 0.1, in `docs/design/optimisers.md`;
  its Section 10 as `docs/optimisers/roadmap.md`.
- Tests: none.
- Checks: none run; the proposal's cost figures are estimates for O0–O2 to measure.
- Decisions: none yet; D0 (the proposal as the plan) comes first.
- Adam and Gauss–Newton: not touched.
- Follow-ups: review the draft; D0, D2; then CO0.1, CO1.1 and CO2.1, which can run in
  parallel.

## 2026-11-24: Draft 0.2 (the Burgers replication)
- Done: the proposal's draft 0.2. A new milestone O1 replicates Zampini et al. 2024
  (PETScML, arXiv 2403.12188, §4.1.1): their FNO Burgers setting as a 2 × 2 of {AdamW,
  Gauss–Newton} × {weight decay 0, 1e-4}, then Darcy as an extension (proposal §3.1). The
  former O1–O4 are now O2–O5, and their tasks CO1.x–CO4.x are CO2.x–CO5.x; the entry above
  uses the old numbers. New decisions D10–D14 (D13: weight decay as a factor in O4; D14:
  Burgers in O4); risks R7–R8. The library objective's CM0.3 recorded as a need of CO1.1
  and CO2.1 (`training/` changes serial, the author's sequencing of 2026-11-24).
- Tests: none.
- Checks: none run (documents only).
- Decisions: none yet; D0 now covers draft 0.2.
- Adam and Gauss–Newton: not touched.
- Follow-ups: the author's review of draft 0.2; D0, D2, D10–D12; then CO0.1, CO0.2 and CO3.1
  at once, CO3.2 after CO3.1, and CO1.1 after CM0.3 (CO1.1 before CO2.1).

## 2026-11-24: Draft 0.2 corrections
- Done: draft 0.2 checked by an independent agent against PETScML (arXiv 2403.12188), the
  code and Burn 0.23; nine corrections applied (proposal Appendix A, "Draft 0.2,
  corrections").
- Tests: none.
- Checks: none (docs).
- Decisions: none signed off; D0 on draft 0.2 next.
- Adam and Gauss–Newton: not touched.
- Follow-ups: D0, D2; then CO0.1, CO0.2 and CO3.1.

## 2026-11-24: D0 and D2
- Done: D0 signed off by the author: draft 0.2 of `docs/design/optimisers.md`, with the
  corrections of #128, as the plan of record. D2 signed off as recommended: seeds 42–46;
  the equal-time budget (CO0.1's Adam mean) and 5,400 s, every arm at both.
- Tests: none.
- Checks: none (docs).
- Decisions: D0 and D2 signed off 2026-11-24.
- Adam and Gauss–Newton: not touched.
- Follow-ups: CO0.1 and CO0.2 (the VM, at the author's go per run) and CO3.1; CO3.2 after
  CO3.1; the tasks that change `training/` after the library's CM0.3.

## 2026-11-25: D2 amended
- Done: D2 amended by the author: seeds 42–44 (three) at the 5,400 s budget, 42–46 (five)
  at the equal-time budget, for every arm. CO0.1, CO1.3, CO4.1 and the O1 and O4 gates
  updated; proposal Sections 3, 3.1, 7, 12 and Appendix A.
- Tests: none.
- Checks: none (docs).
- Decisions: D2 amended 2026-11-25.
- Adam and Gauss–Newton: not touched.
- Follow-ups: CO0.1 at about 13 GPU-hours instead of 19; its plan checks that two timed runs
  side by side keep a lone run's epoch times.

## 2026-11-24: CO3.1 Burn linear algebra for SOAP
- Done: Burn's `linalg` feature enabled in the workspace `burn` dependency (pin unchanged).
  Spike `spikes/linalg/` (`REPORT.md`): `linalg_check` measures `qr` (device) and `svd`
  (host) at n = 12, 32, 128, 144 in f32 against f64 host references; `fingerprint` hashes a
  seeded FNO forward, Adam and hybrid Gauss–Newton runs. Both correct on flex and metal;
  device `qr` 18–23× slower than a host QR (round trip included) on metal at n ≥ 128 and
  95–240× on cuda (2026-11-25, A100).
- Tests: no new crate test (spike only); the existing suite unedited with the feature on.
- Checks: ran fmt, the three clippy lines and `--all-targets`, `cargo test`, `cargo doc`,
  datagen clippy and tests, metal `gpu_parity`, the spike's clippy (flex, metal) and runs
  (flex, metal; cuda on the VM 2026-11-25), the fingerprint on this branch and on
  `origin/main`.
- Decisions: D9 signed off 2026-11-25: host `svd` for the first basis and host QR for the
  refreshes, on every backend.
- Review: plan review PASS (wording fixes applied); code quality APPROVE, no blockers
  (plan amended for the svd metrics, like-for-like host timing, comments).
- Adam and Gauss–Newton: unchanged bit for bit: equal fingerprints with `linalg` off
  (`origin/main` 4d681d0) and on, flex; the feature adds no code the crate uses (REPORT).
- Follow-ups: CO3.2 (needs D8).
