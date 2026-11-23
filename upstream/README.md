# Upstream reports

Bugs and limitations found in our dependencies, one folder per project and one per
report (`<project>/<short-name>/README.md`: summary, environment, minimal reproduction,
impact, workaround). Each report opens with its status: open or fixed, and whether it
has been filed upstream. Filing is the author's decision; a report written here is not
yet an upstream issue.

| Report | Project | Status | Affects |
|---|---|---|---|
| [cfft-kernel](burn/cfft-kernel/README.md) | Burn (`burn-cubecl-fusion`) | fixed in 0.23.0-pre.2 | Metal: wrong Bluestein `cfft` |
| [fused-cat-reused-at-new-shape](burn/fused-cat-reused-at-new-shape/README.md) | Burn (`burn-cubecl-fusion`) | fixed in 0.23.0-pre.2 | Metal: fused `cat` replayed at a new shape |
| [flex-f64-fft-f32-twiddles](burn/flex-f64-fft-f32-twiddles/README.md) | Burn (`burn-flex`) | open, not filed | f64 FFTs on the CPU only f32-accurate |
| [f64-matmul-declined-on-cuda](cubek/f64-matmul-declined-on-cuda/README.md) | CubeK (`cubek-matmul`) | open, not filed | every f64 matmul on cuda panics (0.23.0-pre.2) |
| [tf32-default-matmul](cubek/tf32-default-matmul/README.md) | CubeK (`cubek-std`) | by design; `cubek.toml` turns it off | f32 matmuls on cuda 50× less accurate |
| [fft-f64-kernels-launched-as-f32](cubek/fft-f64-kernels-launched-as-f32/README.md) | CubeK (`cubek-fft`) | fixed upstream (#677) | f64 FFTs on cuda returned garbage (fork) |
| [autotune-no-candidate-panics](cubecl/autotune-no-candidate-panics/README.md) | CubeCL (`cubecl-runtime`) | open, not filed | panic and SIGSEGV instead of an error |

The GPU repros of the Burn reports run in `tests/gpu_parity.rs`; the f64 ones in
`tests/gauss_newton_f64.rs` (`#[ignore]`d). `spikes/metal_diag` holds the op-by-op
diagnostics behind the Burn reports.
