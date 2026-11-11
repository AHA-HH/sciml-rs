# 3D Poisson and Laplace progress log

Newest entries at the bottom. One entry per task, in that task's pull request, using this
template:

```
## YYYY-MM-DD: <ID> <title>
- Done:
- Tests:
- Checks: ran …; pending …
- Decisions: none, or D… signed off; questions for sign-off …
- Review: plan review (blocking items fixed); code quality APPROVE or BLOCKED → APPROVE (rounds, blockers fixed, findings left as follow-ups)
- 2D: unchanged bit for bit (how shown), or not touched
- Follow-ups:
```

## 2026-11-11: Planning
- Done: the proposal, draft 0.1, in `docs/design/3d-chebyshev-laplace-fno.md`; its
  Section 10 as `docs/laplace3d/roadmap.md`; the docs reorganised by objective (the 2D
  phases moved to `docs/poisson2d/`, index `docs/README.md`).
- Tests: none.
- Checks: none run for the proposal; its cost figures are estimates for CL0.1 to measure.
- Decisions: none yet; D0 (the proposal as the plan) comes first.
- 2D: docs moved and their paths rewritten; no code changed except path strings in
  comments.
- Follow-ups: review the draft; D0; then CL0.1, CL1.1.

## 2026-11-11: Draft 0.2
- Done: an independent review of draft 0.1 against the code and CONVENTIONS (verdict:
  revise first); draft 0.2 answers it (proposal Appendix A). Blocking items: the 3D GRF at
  α = 2 misses ε_K ≤ 5e-3 by 20× at K = 32 (ε_K ~ 1/K in 3D), so α = 3 is proposed (D3);
  D2 now names the 2D outputs, excludes metadata and requires in-process checks. Also:
  error 5 in 3D, the solver's sign, the Laplace discrete lifting with face sine series, the
  §4 grid-order conflict dropped, activation memory and the VM's A100 80 GB in §7, 3D
  dataset names, missing roadmap *Needs*.
- Tests: none (docs). The ε_K table was recomputed independently (lattice sums to K = 160
  plus an integral tail) and agrees with the review's.
- Checks: none (no code changed).
- Decisions: none signed off; D0 is asked again on draft 0.2, with D1–D3 recommendations
  revised.
- Review: the review of 0.1 is the input; 0.2 not yet reviewed by the author.
- 2D: not touched.
- Follow-ups: D0 on draft 0.2; then CL0.1 (the zip64 check added to it), CL0.2, CL1.1.
