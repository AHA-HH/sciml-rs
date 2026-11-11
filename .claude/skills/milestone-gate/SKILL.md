---
name: milestone-gate
description: Checks the gate of a milestone of an objective with a roadmap (docs/<objective>/roadmap.md, e.g. laplace3d L0-L5, operators M0-M4) against the roadmap and the proposal, writes the gate report, and closes the milestone on the author's sign-off. Use when /next-task says a milestone is complete, or when the user asks to close one.
---

# Milestone gate

Adapted from the `/milestone-gate` skill of `tbetcke/fmm` (MIT or Apache-2.0), without its
GitHub issue sync.

## Permissions

As `/next-task`: invoking this skill is the request to create the branch
`<objective>/<milestone>-gate` (lower case, e.g. `laplace3d/l0-gate`; from `origin/main` with
`git switch --no-track -c`, never checking out `main`, as `/next-task` step 4), commit to it, push it
and open its PR. Merging and pushing to `main` need a yes in the current message. No Claude
attribution in commits or PRs.

## 1. Which milestone

The one named (with its objective, or found from its roadmap), else the first whose tasks
are all ticked on `main` while its gate is open. A gate is open while the milestone's
`**Gate:**` line in the roadmap has not become `**Gate (closed <YYYY-MM-DD>):**`. If a task
is still open, list it and stop.

## 2. Evidence

- Per task: its merged PR (`gh pr list --state merged --search "<objective> <ID>"`), its
  report, its progress entry, and no pending run left.
- `main`'s latest CI runs green: `gh run list --branch main --limit 5`.
- The milestone's gate in the roadmap and the matching parts of the proposal's §10 met,
  with numbers.
- The decisions the milestone prepares (the roadmap's "Prepared by" column) put to the
  author, and signed off or open.
- The proposal's validation items (§9 or §5) that apply; constraint 1 kept (the "2D" or
  "FNO" line of each progress entry).
- Measured numbers set against the proposal's expectations (for example its cost model),
  and the corrections they imply.
- **Code quality across the milestone.** A fresh `general-purpose` subagent reads and
  follows `~/.claude/skills/thermo-nuclear-code-quality-review/SKILL.md` over the
  milestone's whole diff: the base is the `main` commit just before the milestone's first
  task PR was merged (its merge commit's first parent). It looks for what only shows across
  tasks (layer leaks built up over several PRs, near-duplicate helpers from different
  tasks, a file grown past 1,000 lines in steps). Report, git-ignored, in the main
  checkout (first entry of `git worktree list --porcelain`):
  `docs/code-quality/thermo-nuclear-review-<YYYY-MM-DD>-<objective>-<milestone>-gate.md`.
  The gate PR has no code, so its blockers and findings become questions for sign-off or
  roadmap tasks, not fixes.
- **The objective's last milestone only: the whole repository.** A fresh subagent reads and
  follows `~/.claude/skills/thermo-nuclear-repo-review/SKILL.md` (user-invoked only, so
  followed from its file; the author asked for it at the end of each design document,
  2026-11-11), its report in the same folder. Its refactor roadmap is put to the author as
  the objective's follow-ups or a next objective, not applied.

## 3. The report

Write `docs/<objective>/gates/<milestone>.md`: the date; each gate condition with its
evidence (PRs, reports, CI runs); the code-quality verdicts (the milestone's and, at the
last milestone, the repository's), with their blockers, top findings and local report
paths; what failed or remains; the measured numbers against the
proposal; proposed corrections to the proposal, as questions for sign-off rather than edits;
recommendations for the next milestones; and a short retrospective on the loop itself (what
to change in `/next-task`, `/verify` or the roadmap).

## 4. Close

Ask the author to sign off the gate and the proposed corrections. On their answer, on the
branch: change the milestone's line to `**Gate (closed <YYYY-MM-DD>):**`; record the
sign-offs in the decision table in the author's words; apply the signed-off corrections to
the proposal with a line in its Appendix A (a new draft number if the plan changes); update
the roadmap's status line; append a progress entry. Commit as
`docs(<objective>): <milestone> gate`, push, check the remote head, and open the PR.
