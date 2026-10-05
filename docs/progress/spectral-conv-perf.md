# spectral-conv-perf: progress log

A running record of what each task of the
[spectral-conv-perf objective](../design/spectral-conv-perf.md) delivered and what it
measured. Plans live in `docs/spectral-conv-perf/phase<N>/`; this file records what each
task produced.

The official phase 0 baseline is recorded by T3 in the
[phase 0 README](../spectral-conv-perf/phase0/README.md) and design §7/§8.2. Where this
log and those documents disagree, those documents win.

---

## Phase 0 / T2: `bench_spectral_conv` benchmark (C0.1)

Brief: [`docs/spectral-conv-perf/phase0/T2-benchmark.md`](../spectral-conv-perf/phase0/T2-benchmark.md)

### Why it was needed

The objective's definition of done (design §1) is that, on Metal, an FNO training step is
*measurably* faster than the phase 0 baseline. Here, measurably means that across at
least 20 timed steps the median is lower and the inter-quartile ranges do not overlap. The
objective needs a fixed, repeatable benchmark to make that claim. Phase 0 runs it to set
the baseline, and phases 1 to 3 rerun it unchanged to show whether their changes helped.
The benchmark times two things:
- **the layer alone:** `SpectralConv` forward + backward. This isolates the code the
  objective changes;
- **a full training step:** forward, loss, backward and Adam. This is the quantity the
  §1 gate is stated in, and it shows how much of a step the spectral layers take.

### What was implemented

- `examples/bench/spectral_conv.rs`, run with
  `cargo run --release --example bench_spectral_conv [--features metal] [-- <filter>]`.
  - **Settings.** These are the design §1 / §8.3 settings, as `const` literals:
    - B = 20;
    - layer 1D: I = O = 64, s ∈ {256, 1024}, modes [16];
    - layer 2D: I = O = 32, 94 × 94, modes [12, 12];
    - FNO step: the `FNOConfig` of `train_burgers` at s = 256 and 1024, and of
      `train_darcy` at s = 85 with `padding: Some(9)`, so its spectral layers run at
      94 × 94.
  - **Training step.** It matches `trainer.rs` exactly:
    - `LpLoss::new(R - 2, 2, Reduction::Sum)`;
    - Adam with `epsilon = 1e-8` and weight decay 1e-4;
    - a fixed learning rate of 1e-3;
    - `flatten_pair`, reused from `training::trainer`.

    Inputs are random, so no dataset is needed.
  - **Timing.**
    - 5 warm-up steps, then 20 timed steps.
    - Every timed step is bracketed by `device.sync()`, so queued GPU work is counted.
    - The returned gradients are dropped only after the closing sync.
  - **Statistics.** Nearest-rank quartiles: with 20 samples, q1, median and q3 are
    x(5), x(10) and x(15).
  - **Output.** One table on stdout; no files are written.
- `Cargo.toml`: the `[[example]]` entry.
- No library code changed.

### Results

- **Machine:** Apple M2 Pro, 16 GB, macOS 26.6.2, rustc 1.99.0.
- **Code:** base commit `ae48703`, release build.
- **Runs:** one benchmark invocation per backend, on 2026-10-05.
- **Units:** milliseconds per step; IQR is q3 − q1.

| Case | flex median | flex IQR | Metal median | Metal IQR | flex / Metal |
| --- | ---: | ---: | ---: | ---: | ---: |
| layer 1D I=O=64 s=256 m=[16] | 6.081 | 0.466 | 1.769 | 0.436 | 3.4× |
| layer 1D I=O=64 s=1024 m=[16] | 17.081 | 1.267 | 3.324 | 0.474 | 5.1× |
| layer 2D I=O=32 94×94 m=[12,12] | 3087.872 | 111.514 | 578.798 ¹ | 1.418 | 5.3× |
| FNO step Burgers s=256 | 29.226 | 0.975 | 6.588 | 1.097 | 4.4× |
| FNO step Burgers s=1024 | 92.337 | 6.707 | 15.536 | 1.614 | 5.9× |
| FNO step Darcy s=85 (94×94 padded) | 12636.604 | 805.391 | 2289.209 ¹ | 3.512 | 5.5× |

¹ These Metal rows compute **wrong values**. This is the known Metal `cfft` bug on even,
non-power-of-two non-last axes (phase 0 README "Known issue", design §9.1 R8). The
timings show the cost of today's code path, not of a correct computation.

**Sync check.** This confirms that the timings include queued GPU work. The 2D layer case
was rerun with the closing `device.sync()` temporarily removed:

| Backend | with closing sync | without closing sync |
| --- | ---: | ---: |
| Metal | 584.793 (IQR 2.314) | 271.181 (IQR 134.625) |
| flex | 3087.872 | 3067.996 |

- On Metal, the median halves without the sync, and the spread grows about 60×. The clock
  stopped before the GPU had finished.
- On flex, which executes synchronously, the two runs are the same within noise.

The `-- darcy` filter printed only the Darcy row.

### What the results show

- **The 2D case dominates everything.**
  - The 94 × 94 layer carries about 4.3× the data of the 1D s = 1024 layer (8836 points
    × 32 channels against 1024 × 64).
  - It is about 180× slower on flex and about 175× slower on Metal.
  - 94 is not a power of two, so its FFTs take the Bluestein path. That is consistent
    with the motivation for the objective, but the benchmark does not by itself isolate
    the cause. Phases 1 and 2 will show which part of the cost they remove.
- **On Darcy, the training step is almost all spectral layers.**
  - The model has four spectral layers. Four times the 2D layer cost is about 2315 ms on
    Metal against a 2289 ms step, and about 12350 ms on flex against a 12637 ms step.
  - So speeding up `SpectralConv` translates almost one for one into Darcy training time.
  - This is a rough estimate. The layer case is not exactly one FNO layer: its input and
    loss differ.
- **The layer share is smaller for Burgers.**
  - On flex, four layers at s = 256 are about 24 ms of a 29 ms step (about 83 %).
  - On Metal the same estimate (about 7.1 ms) exceeds the measured step (6.6 ms). At these
    small sizes, per-op launch overhead and queue overlap make layer costs non-additive,
    so the Metal Burgers gains of later phases may be smaller than the layer benchmark
    suggests.
- **Metal is 3.4× to 5.9× faster than flex.** The ratio grows with problem size.
- **Metal timings are stable on the large cases.**
  - The 2D layer and Darcy step IQRs are 0.25 % and 0.15 % of their medians.
  - Against non-overlapping IQRs, the design §1 gate should therefore detect even small
    improvements on Darcy.
  - The small 1D cases are relatively noisier: the s = 256 layer's IQR is about 25 % of
    its median. Gains there must be large to clear the gate.
  - Flex is noisier overall: the Darcy step's IQR is about 6 % of its median.

### Status

- Implemented and checked. CI checks pass: fmt, clippy (lib, examples, all targets),
  `cargo test` and `cargo doc`.
- Independent review: pass. One advisory finding, a line citation, was fixed.
- Uncommitted, awaiting PR.
- T3 will rerun the benchmark to record the official baseline.
