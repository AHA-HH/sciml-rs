# 3D Poisson and Laplace progress log

Newest entries at the bottom. One entry per task, in that task's pull request, using this
template:

```
## YYYY-MM-DD: <ID> <title>
- Done:
- Tests:
- Checks: ran …; pending …
- Decisions: none, or D… signed off; questions for sign-off …
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
