---
name: next-task
description: The default development loop of the objectives with a roadmap (docs/<objective>/roadmap.md, e.g. laplace3d, operators). Syncs the open task PRs, picks the next runnable task, plans it, implements it tests first, verifies it, logs it and pushes it as a PR. Use whenever the user says "continue", "next" or "keep going", names a roadmap ID such as CL1.2 or CM0.3, names an objective, or asks what to do next.
---

# Next task

One pass takes one roadmap task as far as it can go here and leaves it as a pull request:
ready for the author's merge, or a draft that says what it waits for. The author's part is
to answer the questions this loop asks and to merge. Everything the loop needs is in the
repository: the objective's proposal (`docs/design/`), its roadmap, its progress log, its
task plans and gate reports, `docs/CONVENTIONS.md` and root `CLAUDE.md`.

Adapted from the `/next-task` skill of `tbetcke/fmm` (MIT or Apache-2.0), without its
GitHub issue sync.

## Permissions

Invoking `/next-task` is the author's request to create the task's branch, commit to it,
push it, and open or update its PR (root `CLAUDE.md`, "Working agreement"). Merging,
pushing to `main`, force-pushing, rewriting history, deleting a branch and closing a PR
each need a yes in the current message. Never add a Claude attribution (no Co-Authored-By
trailer, no "Generated with" footer) to commits or PRs.

## Objectives

An objective is a folder `docs/<objective>/` with a `roadmap.md` (`docs/README.md` lists
them). Its task IDs have a prefix: `CL` is `laplace3d` (proposal
`docs/design/3d-chebyshev-laplace-fno.md`), `CM` is `operators` (proposal
`docs/design/neural-operator-library.md`); a new objective's roadmap states its own. The 2D
objective (`docs/poisson2d/`) has no roadmap file and is outside this loop.

Which objective: the one the user names, or the one the named ID belongs to. Otherwise give
the next runnable task of each objective in one line each and ask which to take (one
question, the objective with the earliest blocking dependency for the others first).

## 0. Orient

- Read root `CLAUDE.md`, `docs/README.md`, the objective's `roadmap.md` (its rules, the
  decisions, the tasks) and the last three entries of its `progress.md`. Keep its proposal
  at hand: §1.4 and §2 (constraints) always, §10 and the roadmap line for the task's
  acceptance, and the sections the task cites.
- `git fetch origin`. If the working tree has changes: on a branch `<objective>/<id>` this
  is an unfinished task, so resume it where its plan (`docs/<objective>/tasks/<ID>.md`) and
  the commits show it stopped; on any other branch stop and ask (never stash or discard
  work).
- Otherwise `git switch main` and `git pull --ff-only`.

## 1. Sync the open task PRs

First `main`'s CI: `gh run list --branch main --limit 5`. Report every job that failed or
was cancelled since the last pass (`gh run view <id> --json jobs`) and propose a separate
small fix before the next task is taken.

Then `gh pr list --state open --author "@me" --json number,title,headRefName,isDraft,url`,
the branches `<objective>/*` of every objective with a roadmap. For each, in roadmap order:

1. **CI failing** (`gh pr checks <n>`): fixing it is this run's task. Switch to its branch,
   fix, `/verify`, push, update the PR, and go to step 10.
2. **Review comments from the author not yet answered** (`gh pr view <n> --comments` and
   `gh api repos/{owner}/{repo}/pulls/<n>/comments`): addressing them is this run's task,
   in the same way.
3. **Draft waiting for VM runs** (its plan's "Pending runs"): check each on the VM
   (`ssh cricket 'tmux ls; tail -n 20 ~/sciml-rs/logs/<log>'`). Finished: fetch the output
   (`scp`), write the results into the task's report, check the acceptance again, update
   the plan and the progress entry, push, and if nothing else is pending `gh pr ready <n>`.
   Still running: note it and go on.
4. **Draft waiting for a sign-off**: if the roadmap's decision table or the author's message
   now has the answer, record it (step 2's wording) on that branch, push and mark it ready.
5. **Ready and green**: it waits for the author's merge.

Merged PRs need no bookkeeping: their roadmap tick and progress entry were part of them. If
item 1 or 2 applied, that was this run's task: stop at step 10 after it.

## 2. Pick

- If the user named a task, take it; if its *Needs* are not met, say which and ask.
- Otherwise take the first task in roadmap order that is runnable (the roadmap's "How to
  read this file"): open, no branch or open PR, every task in *Needs* ticked on `main`,
  every decision in *Needs* signed off, and D0 signed off.
- If the first task in order waits only for decisions, ask the author for them now: one
  question per decision, the proposal's recommendation first, then the alternatives its §12
  names. Record each answer in the decision table as
  `signed off <YYYY-MM-DD>: <the answer as given>`, never stronger than what the author
  said; commit it on the task's branch (step 4). Then take that task. D0 (the proposal as
  the plan of record) is asked this way before the objective's first task.
- If nothing is runnable, say which PRs to merge, which decisions to take or which runs to
  wait for, and stop.
- **Size.** A task that will not fit one reviewable PR (as a rule of thumb more than about
  1,000 changed lines, more than three modules, or new code plus VM measurements) is split
  in the roadmap into lettered sub-tasks, each with *Done when* and *Needs*; commit the
  split on the branch and take the first.

Say in one line which task you take and why it is next.

## 3. Plan

Write `docs/<objective>/tasks/<ID>.md`:

```
# <ID> <title>

Started <YYYY-MM-DD> on `<objective>/<id>`. Roadmap: <the line>. Proposal: §10; §…

## Goal
<one sentence>

## Acceptance (roadmap *Done when*, verbatim, and the proposal's matching text)
<the criterion>

## Read first
<files, sections, code>

## Changes
<modules and files; public API; conventions (none, or a proposal for sign-off)>

## Tests that prove it
<each against its trusted path (closed forms, a slower reference, fixtures, the 2D
outputs on `main`), with the tolerance or bit identity>

## Checks
<`/verify`, and this task's own commands>

## VM runs
<command, expected duration, what it decides; or none>

## Constraints and risks
<the constraints of the proposal's §1.4 and §2 that bind; the effect on the 2D objective>

## Pending runs
<filled in step 6>

## Open questions
<filled when they arise>
```

Stop and ask, with the plan, before going on if the plan:
- changes code on a path whose outputs constraint 1 protects (for `laplace3d`: any 2D
  output; for `operators`: any existing FNO output) without a bit-identity test planned;
- changes `docs/CONVENTIONS.md` (only as a proposal for sign-off, with the
  `CONVENTION_VERSION` bump);
- adds or moves a dependency or a pin (the Burn fork above all), or Python outside
  `tools/parity/`;
- departs from the roadmap's *Done when*, the proposal or a signed-off decision.

Otherwise go on without asking.

## 4. Branch

`git switch -c <objective>/<id>` from `main`, the ID lower-cased with dots as dashes
(`laplace3d/cl1-2`, `operators/cm1-3a`). Commit the plan, and any decision or split, as
`docs(<objective>): <ID> plan`.

## 5. Implement

- Tests first: the plan's tests, failing for the right reason; then the code until they
  pass. Never weaken, skip or delete a test to get green; if one seems wrong, say why and
  ask.
- Root `CLAUDE.md`'s working rules: `cargo fmt` after every Rust edit; every public item
  documented in the existing rustdoc style (tensor shapes, panics); `CONVENTIONS §n`
  citations; examples registered in `Cargo.toml`; the CLAUDE.md "Layout" entry updated for
  new modules.
- Small steps that build, each committed as `<type>(<objective>): <ID> <what>` (`feat`,
  `fix`, `test`, `docs`, `refactor`).
- Spikes go in `spikes/<name>/` with a `REPORT.md`, outside the default build.
- Measurements: release builds; the backend and machine named; no timing asserted in a
  test; the report at the path the plan names (`docs/<objective>/results/<id>.md` unless
  the task says otherwise).

## 6. Verify

Run `/verify` with this task's plan. VM runs: say what the run is and how long it takes,
and wait for the author's go; then on the VM, in tmux with a `tee` log under `logs/`
(`ssh cricket`, `cd ~/sciml-rs`, `git fetch && git switch <branch> && git pull`,
`source ~/sciml-gpu.sh`). A run that has not finished when the rest is done goes under
"Pending runs" with its command, tmux session, log path and what it decides. Never report
an unrun check as passing.

## 7. Self-review

Give the `reviewer` agent the plan file, the roadmap line, the proposal's §1.4, §2 and the
sections the plan cites, and the branch name and its base commit, not your summary. It
checks that every acceptance item has a test or a recorded run that shows it, that the
change stays in scope, that constraint 1 holds, and the rules of root `CLAUDE.md`. Fix what
blocks; list the rest in the PR text.

## 8. Record

On the branch:
- tick the task in the roadmap, adding the report's path;
- update the roadmap's status line: the date, the task just done, what comes next;
- give every follow-up that a sign-off or the work creates a roadmap line in the same PR: a
  new task with *Done when* and *Needs*, or a sentence in the task it belongs to. The
  progress entry's follow-ups name that line;
- for `operators`, set the task's rows in `feature-matrix.md`;
- append to the objective's `progress.md`, using its template;
- where the work shows the proposal wrong, do not edit it silently: put the correction in
  the PR as a question for sign-off.

Commit as `docs(<objective>): <ID> roadmap and progress`.

## 9. Push and the PR

`git push -u origin <branch>`, then check `git ls-remote origin <branch>` shows the
branch's head. Write `target/pr-<id>.md` (ignored by git): the title
`<objective> <ID>: <title>`; a summary; the acceptance items, each with the test or run
behind it, ticked; the checks with their commands and results; the pending runs; the
questions for sign-off, numbered; the effect on the protected outputs (constraint 1); the
files and modules touched. `gh pr create --base main --title "…" --body-file
target/pr-<id>.md`, with `--draft` while a run is pending or a sign-off or open question
blocks. An existing PR is updated with `gh pr edit <n> --body-file …`.

## 10. Report and next

A few lines: the task and its PR (ready, or draft and why); checks run and pending;
questions waiting for the author; the open task PRs and their states; the next runnable
task. If every task of a milestone is ticked on `main` and its gate is open, suggest
`/milestone-gate <milestone>`.
