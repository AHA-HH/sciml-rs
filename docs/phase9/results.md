# Phase 9 results

Systems measurements. **No number here enters a conclusion of Phases 5–7** (design
§13.11).

## T3: one against two A100s (2026-11-09)

**Machine.** The 2× A100 VM (`cricket`): 2× NVIDIA A100 80 GB PCIe, topology `NODE`
(PCIe within one NUMA node, no NVLink), 80 aarch64 cores (Neoverse-N1), RHEL 9.6;
driver 610.57.04, CUDA 12.6.3, NCCL 2.29.7+cuda12.6 (`env/vm-a100/`); cuda backend,
release build at `271dbd9`. **GPUs unshared:** `nvidia-smi` before every run and after
every run but the 1-GPU s = 128 one (its "after" entry was skipped with its stop; its
`dmon` log ends before the next run starts) showed both GPUs at the idle baseline
(146 / 1 MiB, the 146 being the desktop on GPU 0), so no other process held them;
another user was logged in but not on the GPUs.

**Runs.** The baseline (stage 3, n = 129, p = 0, N = 1000, seed 42, batch 20, 12 modes)
at two loads, s = 64 and s = 128, each stopped after 51 epochs (51 rows of
`metrics.csv`; 1 GPU at s = 128 ran on to 381 epochs because the stop script missed its
directory name, and only its epochs 5–50 are used). s/epoch is the median over epochs
5–50, after warm-up; an epoch is 50 training steps and the 200-sample test pass.

| Load | GPUs | s/epoch (median) | min–max | Speedup | Efficiency | SM busy (`dmon`, steady) | Peak memory per GPU |
| --- | --- | --- | --- | --- | --- | --- | --- |
| s = 64 | 1 | 1.718 | 1.697–1.766 | 1 | 1 | ~50% | 971 MiB |
| s = 64 | 2 | 3.932 | 3.803–4.159 | **0.44** | **0.22** | ~22% / ~15% | 1095 / 967 MiB |
| s = 128 | 1 | 2.450 | 2.440–2.462 | 1 | 1 | ~100% | 2251 MiB |
| s = 128 | 2 | 4.007 | 3.933–4.236 | **0.61** | **0.31** | ~40% / ~35% | 2055 / 3367 MiB |

Run directories (git-ignored): `runs/poisson_fno_n129_s64_p0_1794240387` (1 GPU,
s = 64), `…_s64_p0_1794240487` (2 GPUs, s = 64), `runs/poisson_fno_n129_p0_1794240709`
(1 GPU, s = 128), `…_p0_1794241676` (2 GPUs, s = 128). Peak memory is `dmon`'s maximum
framebuffer use less the 146 MiB the desktop holds on GPU 0. The 1-GPU s = 64 figures
match Phase 5 (1.69 s/epoch, 970 MiB) and the 2-GPU s = 64 one T2's run (median 3.857
s/epoch).

**Where the time goes** (nsys 2024.5.1, `-t cuda,nvtx,osrt`, 20 s of steady training
after a 60 s delay, s = 64):

| | 1 GPU | 2 GPUs |
| --- | --- | --- |
| Kernels in 20 s | 893,221 | 413,542 + 391,000 |
| GPU busy (kernel time / window) | 49.0% | 22.5% / 15.2% |
| Kernel launches per training step and replica | ~1,500 | ~1,500 |
| Host time per launched operation | ~23 µs (~6 µs in `cuLaunchKernel`) | same |
| Step period | 34.4 ms (1.718 s / 50) | 73.8 ms (median, from the all-reduce bursts) |
| All-reduces per step | – | **30 per GPU, one per parameter tensor**; NCCL kernel time per step (median) 6.3 ms on GPU 0 (193 µs each, 8.5% of the step), 1.6 ms on GPU 1 (53 µs, 2.1%) |
| Host threads launching | 1 | 2 (one per device), launching in the same millisecond in 14.8% of active milliseconds |
| Memory copies | < 1 ms | < 1 ms |

**Findings.**

1. **The ~31 ms batch of the profiling finding is host dispatch.** A training step
   launches about 1,500 kernels, mostly a few microseconds long (`slice` 6.9 µs, the
   FFTs ~73 µs), and the host spends about 23 µs on each (Burn's autodiff and fusion,
   cubecl's dispatch, the launch itself). The GPU waits for the host half the time at
   s = 64. At s = 128 the kernels are long enough to keep one GPU busy (~100%).
2. **Two GPUs double the host work.** `train_epoch_parallel` runs each replica's
   forward and backward pass and its optimiser step in turn on the training thread
   (`summed_grads`: a sequential `map` over the replicas). Each replica issues the same
   ~1,500 operations as a one-GPU step, so a 2-GPU step costs about two one-GPU steps
   of host time (2 × 34 ms) plus the all-reduces and their stream synchronisation:
   ~74 ms, the measured step period. The device threads mostly take turns (14.8% overlap). This is why the 2-GPU
   epoch is ~4 s at both loads.
3. **The all-reduce is not the bottleneck.** 30 NCCL calls per step, unbucketed, take
   6.3 ms of kernel time on GPU 0 and 1.6 ms on GPU 1 (8.5% and 2.1% of a step); GPU 0's
   longer kernels most likely include waiting inside the collective for GPU 1, whose
   replica the host reaches second. The 9.5 MB of gradients is small for PCIe; bucketing
   would save little.
4. **What would make a second GPU pay** (not done here: T3 measures only, and the brief
   forbids tuning for 2 GPUs): replicas driven concurrently, one host thread each; or
   fewer, larger operations per step (more fusion, CUDA graphs), which also speeds up
   one GPU. Larger per-GPU work alone helps (efficiency 0.22 → 0.31 from s = 64 to 128)
   but does not reach 1× while the host is serial.

**Artefacts** (git-ignored, in `logs/t3/` on the VM and the author's Mac): the driver
script `run.sh`, `gpu_state.log` (`nvidia-smi` before and after each run), `dmon_*.txt`,
the two traces `nsys_P1_1gpu_s64.nsys-rep` (80 MB) and `nsys_P2_2gpu_s64.nsys-rep`
(88 MB), and the plots `plots/t3_epoch_times.png` and `plots/t3_gpu_utilisation.png`.
The traces are too large for a PR attachment, so the PR carries their summaries.
