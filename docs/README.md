# Documentation

The repository is organised by **objective**. Each objective has a design document in
`design/` and a folder of its own for its execution (phases or milestones, task briefs,
results). Phase or milestone numbers start at 0 within each objective, so always name the
objective with the number ("2D Phase 10 T3", "3D L2").

`CONVENTIONS.md` is shared by every objective and wins over any design document.

| Objective | Design | Execution | Status |
| --- | --- | --- | --- |
| 2D Poisson on a Chebyshev grid | [`design/2d-chebyshev-poisson-fno.md`](design/2d-chebyshev-poisson-fno.md) | [`poisson2d/`](poisson2d/): `phase0/` to `phase10/`, `phase4s/`, `optional/`; roadmap in the design's §10 | Phase 10 (Gauss–Newton) open: its VM runs |
| 3D Poisson, then 3D Laplace | [`design/3d-chebyshev-laplace-fno.md`](design/3d-chebyshev-laplace-fno.md) (draft 0.2) | [`laplace3d/`](laplace3d/): `roadmap.md`, `progress.md`, later `tasks/` and `gates/` | draft 0.2 for review, nothing signed off |
| Neural operator library (`neuraloperator` 2.0.0 and beyond) | [`design/neural-operator-library.md`](design/neural-operator-library.md) (draft 0.1) | [`operators/`](operators/): `roadmap.md`, `progress.md`, `feature-matrix.md`, later `tasks/` and `gates/` | proposal in draft, D1 signed off |

Other folders: `code-quality/` holds repository-wide reviews.

## Layout of an objective

- The 2D objective keeps the layout it was built with: one design document whose §10 is
  the roadmap and whose §12 holds the decisions, and a folder per phase with a `README.md`
  and briefs `T<k>-<name>.md`.
- New objectives follow the layout of the Helmholtz work in `tbetcke/fmm`: the design
  document is a proposal (versioned drafts, decisions in its §12, phased plan in its §10);
  `roadmap.md` carries the milestones, tasks (with *Done when* and *Needs*) and the
  decision table with their sign-offs; `progress.md` is the log, one entry per task;
  `tasks/<ID>.md` is each task's plan, written when it starts; `gates/<milestone>.md` is
  each milestone's close-out report.

## Running an objective

The files above are the whole state of an objective; nothing is tracked outside the
repository. `/next-task` (`.claude/skills/next-task/`) reads the roadmap and the last
progress entries, picks the next runnable task, writes its plan in `tasks/`, implements
and verifies it, ticks the roadmap and logs it in the same pull request. `/milestone-gate`
writes `gates/<milestone>.md` and closes the milestone on the author's sign-off. `/verify`
runs the checks. The 2D objective has no `roadmap.md` and is outside this loop.
