# CL3.2 3D dataset generation

Started 2026-11-25 on `laplace3d/cl3-2`. Roadmap: "CL3.2 3D dataset generation. Generation,
label check against the closed form, catalogue names carrying the dimension; the n = 65 set
generated on the VM. *Needs:* CL1.2, CL2.1, CL3.1, D3, D5. *Machines:* the VM (n = 65)."
Proposal: §10; §3.2 (storage), §3.3 (forcings, exact solution), §5 (A3), §6 (extend the
dataset writer and `data/catalogue`), §9 (D2); decisions D3, D5 (signed off), D8 (open).

## Goal
Write the 3D Poisson splits of D5, each sample a 3D GRF forcing (CL3.1) and its A3 label
(CL2.1), every label checked against the closed-form series, under names that carry the
dimension.

## Acceptance (roadmap *Done when*, verbatim, and the proposal's matching text)
- Roadmap: "Generation, label check against the closed form, catalogue names carrying the
  dimension; the n = 65 set generated on the VM." L3 gate: "the 3D datasets of D5 written,
  label-checked, loadable" (loadable as `.npz`; the loader is CL3.3).
- D5: n = 33 (tests and Mac runs), 65 (training); 1000 train / 200 test;
  `poisson3d_n{n}_K{K}_{split}`; f64, one `.npz` per split.
- D3: τ = 3, α = 3, K(n) = (n − 1)/2 (16 at n = 33, 32 at n = 65).
- Proposal §6: "the dataset writer and `data/catalogue` get 3D set names carrying the
  dimension (`poisson3d_n{n}_K{K}_{split}`, so no 3D set takes a 2D name)".

## Read first
`pde/poisson/dataset.rs` (the 2D writer, copied in shape), `grf3d.rs`, `collocation.rs`
(`CollocationSolver3d`), `chebyshev/transfer.rs` (`Transfers::to_*_nd`),
`chebyshev/clencurt.rs` (`cc_rel_l2_error_nd`), `data/catalogue.rs`,
`data/loaders/poisson/mod.rs` (`SidecarRecord`), `examples/generate/poisson.rs`,
`docs/laplace3d/results/cl2-1.md` (A3 against the series: 1.9e-13 at n = 33, K = 16).

## Changes
- `data/catalogue.rs`: `Poisson3dSet { n, k }` with `stem` / `npz` / `json`
  (`poisson3d_n{n}_K{K}_{split}`), and `GENERATED_3D = [{33, 16}, {65, 32}]`. `PoissonSet`
  and the 2D constants unchanged.
- `pde/poisson/dataset3d.rs` (new, behind `datagen` with the rest of `pde/`):
  `Split3d` (f, u `[N, n, n, n]` 'ijk', the nodes, label check, error 2, wall time),
  `generate_3d(&SplitSpec)` (the 2D `SplitSpec`, reused: n, optional K, base seed, count),
  `label_check_3d` (relative CC-L² by `cc_rel_l2_error_nd`; the 2D `LabelCheckError`),
  `write_split_3d` (`.npz` with `f`, `u`, `x`, `y`, `z`; the 2D `SidecarRecord` with the 3D
  α, RNG and ε_K, so CL3.3 reads it with the same type; no new JSON field).
- Label check: the 2D rule, `LABEL_CHECK_TOL` = 1e-8 enforced for n ≥ `LABEL_CHECK_MIN_N`
  (65), recorded below. (The first draft of this plan enforced it from n = 33, citing
  CL2.1's 1.9e-13; that figure was at K = (n − 1)/4. At D3's K(n) = (n − 1)/2 the
  measured check is about 4e-7 at n = 33 and 4e-10 at n = 65, 20 seeds, as in 2D.)
- Error 2 per sample, d = `TRANSFER_DEGREE`, s = n − 1, by `Transfers::to_uniform_nd` /
  `to_cheb_nd`, recorded in the sidecar as in 2D.
- `examples/generate/poisson3d.rs` (`generate_poisson3d`, `required-features = ["datagen"]`):
  the sets of `GENERATED_3D`, or those whose n are given as arguments; seeds and counts as
  the 2D generator (train 0 + i, N = 1000; test 1 000 000 + i, N = 200; the 3D draws are
  on their own ChaCha8 stream, CL3.1). Writes `datasets/poisson/` beside the 2D files
  (the names cannot collide).
- `datasets/README.md`: the 3D files. Root `CLAUDE.md` "Layout": the new module.
- Conventions: none (CONVENTIONS version 3 already defines 'ijk', CL1.1).

## Tests that prove it
1. `generate_3d_matches_its_parts` (n = 17, K = 8, 2 samples): f equals
   `grf3d::evaluate` of the seed's coefficients bit for bit; u equals
   `CollocationSolver3d::solve_full(f)` bit for bit and is zero on the six faces; the
   label check not enforced at n = 17.
2. `label_check_3d_at_n65` (n = 65, K = 32, 2 samples): enforced, max ≤ 1e-8 against the
   closed form; n = 33 not enforced.
3. `label_check_3d_aborts`: a corrupted label at n = 65 fails with the sample and seed in
   the error; at n = 33 the same corruption is recorded.
4. `files_round_trip_3d` (n = 9, 2 samples): `poisson3d_n9_K4_train.{npz,json}`; f, u
   `[2, 9, 9, 9]` and x, y, z read back exactly; the sidecar's α = 3, `grf3d::RNG`,
   `grf3d::eps_k(4)`, K, n, s, count, label check, error 2.
5. `poisson3d_names` (catalogue): stems of `Poisson3dSet`; `GENERATED_3D` at K(n); no 3D
   file name equals a 2D one of `GENERATED`.
6. Constraint 1: `generator_stems_unchanged` and the whole 2D suite pass unchanged; no 2D
   function body edited.

## Checks
`/verify` (fmt, the CI clippy jobs, `cargo clippy --all-targets`, `cargo test`, datagen
clippy and tests, `cargo doc`). Then on the Mac:
`cargo run --release --features datagen --example generate_poisson3d -- 33` (the n = 33
sets; their sidecars' label check and error 2 into the report
`docs/laplace3d/results/cl3-2.md`). Metal and gpu_parity: not relevant (host ndarray only).

## VM runs
D8 (open) proposes n = 65 generation on the A100 VM, beside the training it feeds. Only
with the author's go, and **not while a timed run is in tmux** (optimisers CO0.1, session
`co0`). Expected: about under a minute of compute (20 samples at n = 65 take 0.4 s in release
on the Mac) plus writing 4.4 GB (train) and 0.9 GB (test); peak memory about 5 GB
(f and u of the train split in memory).

```sh
ssh cricket
tmux ls                       # no co0 (or other timed run) may be running
cd ~/sciml-rs && git fetch origin
git worktree add --detach ~/sciml-rs-wt/cl3-2 origin/laplace3d/cl3-2
cd ~/sciml-rs-wt/cl3-2
mkdir -p ~/sciml-rs/datasets/poisson   # so the link below exists and the files outlive the worktree
find ~/sciml-rs/datasets -mindepth 1 -maxdepth 1 ! -name README.md -exec ln -s {} datasets/ \;
source ~/sciml-gpu.sh
export CARGO_TARGET_DIR=~/sciml-rs-target/laplace3d
tmux new -d -s cl3-2 "cargo run --release --features datagen --example generate_poisson3d -- 65 2>&1 | tee ~/sciml-rs/logs/cl3-2.log"
```

Afterwards: `tail ~/sciml-rs/logs/cl3-2.log`, `ls -l ~/sciml-rs/datasets/poisson/poisson3d_n65_*`,
the sidecars' label check and error 2 into the report;
`git -C ~/sciml-rs worktree remove ~/sciml-rs-wt/cl3-2`. It decides the L3 gate's "written"
for the training stage. If `datagen` (OpenBLAS, FFTW) does not build on the VM, the
fallback is generating on the Mac and copying the two n = 65 splits (5.3 GB) to the VM.

## Constraints and risks
- Constraint 1 (D2): additions only. `dataset.rs`, `grf.rs`, `PoissonSet`, `STAGES`,
  `GENERATED` and `generate_poisson` are not edited; the 2D files and their names unchanged.
- Memory at n = 65: the train split holds 2 × 2.2 GB; fine on the VM, tight but possible
  on the Mac.
- The `.npz` at n = 65 is 4.4 GB, above 4 GiB: zip64, verified with this writer on the Mac
  and the VM in CL0.1.

## Pending runs
<filled in step 6>

## Open questions
- D8 (open): n = 65 generation on the VM, as above. Asked in the PR.
