# sciml-rs

A Neural Operator library in Rust, built on [Burn](https://burn.dev). 

Tensor rank is a compile-time property, so the spatial dimensionality of a
problem is fixed when the binary is built rather than checked at runtime.
`modes: vec![16]` gives you an 1D FNO
`modes: vec![4, 4]` gives you a 2D FNO.

## Results
| problem    | grid    | Burn    | PyTorch |
|------------|---------|---------|---------|
| 1D Burgers | s = 256 | 0.00188 | 0.00189 |
| 2D Darcy   | s = 16  | 0.009 | 0.009 |

Same hyperparameters across columns. The PyTorch port is a direct
translation of Li et al.'s reference implementation in the official 
PyTorch neural operators toolbox, run locally.

### 2D Poisson on a Chebyshev grid

| grid | padding | test relative L² | backend |
|------|---------|------------------|---------|
| n = 65 (s = 64) | 4 | 0.01069 | cuda |
| n = 129 (s = 128) | 4 | 0.01079 | cuda |

−Δu = f on [−1, 1]² with u = 0 on the boundary, f a Gaussian random field. Reference
solutions come from a Chebyshev collocation solver in this crate (behind the
`datagen` feature). Fields are interpolated to a uniform grid for the unchanged FNO,
and predictions are mapped back to the Chebyshev nodes; that round trip adds at most
about 2e-5. Evaluated on finer grids without retraining, the 65² model's error grows
7× by n = 257 at padding 4 but only 1.4× without padding. Details and all five error
measures are in [`docs/phase4/README.md`](docs/phase4/README.md); the design is
[`docs/design/2d-chebyshev-poisson-fno.md`](docs/design/2d-chebyshev-poisson-fno.md).

The spectral layers support any grid size (non-power-of-two FFTs go through
Bluestein's algorithm), and `examples/train/darcy.rs` now trains at the
paper's s = 85 with 12 modes. The Darcy row above is from an earlier s = 16
configuration; an s = 85 comparison against PyTorch hasn't been run yet.
For non-periodic problems, `FNOConfig::padding` appends that many zero cells to
the end of every spatial axis before the spectral layers and crops them off
afterwards; `train_darcy` uses 9, as in Li et al.'s `fourier_2d.py`. The padded
size needn't be a power of two.

Model inputs carry data channels only (`[batch, spatial.., data_channels]`).
`FNO::forward` generates the uniform `[0, 1]` coordinate channels on the
device for the input's own resolution and appends them after the data:
`[data, x]` in 1D and `[data, x, y]` in 2D (`np.meshgrid` 'xy' order, as in
Li et al.), and in general one channel per spatial axis in reverse axis
order. The loaders therefore no longer store grids, which cuts input storage
and host-to-device transfer by 2x for Burgers and 3x for Darcy. Checkpoints
saved before this change load unchanged (same parameter names and shapes).
Their predictions are bit-identical on the default `flex` backend; on other
backends the generated grid may differ from the old one by up to 1 ulp
(division rounding), so predictions agree only to within that.

## Requirements

- Rust 1.85 or later ([rustup.rs](https://rustup.rs))
- ~2 GB disk for the datasets
- `gnuplot` - required for plots (`brew install gnuplot`, `apt install gnuplot`)

## Getting started

    cargo build --release  # Build target
    cargo test             # Check tests pass
    cargo doc --open       # API documentation

## Datasets

Datasets aren't committed - see [`datasets/README.md`](datasets/README.md)
for download links. The Poisson datasets are generated rather than downloaded
(`generate_poisson`, below).

## Backends

The backend is a cargo feature; the code is the same for all of them.

| feature | hardware |
|---------|----------|
| `flex` (default) | CPU, pure Rust - runs anywhere, slow for training |
| `metal` | Apple GPUs |
| `cuda` | NVIDIA GPUs |
| `rocm` | AMD GPUs |
| `vulkan` | Vulkan GPUs (Linux/Windows; on macOS needs the Vulkan SDK) |
| `wgpu` | any GPU wgpu can find |

Features are additive. At runtime `Device::default()` uses the
highest-priority backend that was compiled in - CUDA, then Metal, ROCm,
Vulkan, wgpu, and Flex last - so enabling a GPU feature on top of the default
is enough. To force one without rebuilding, set 
`BURN_DEVICE` (`cuda`, `rocm`, `metal`, `vulkan`, `wgpu`, `flex`):

    cargo run --release --example train_burgers                      # CPU
    cargo run --release --example train_burgers --features metal     # Apple GPU
    cargo run --release --example train_burgers --no-default-features --features cuda
    BURN_DEVICE=flex cargo run --release --example train_burgers --features metal

Burn's `ndarray` backend is not offered: its FFT extension doesn't support
it, and the FNO needs the FFT.

## Running examples

    cargo run --release --example train_burgers --features metal   # 1D Burgers
    cargo run --release --example train_darcy --features metal     # 2D Darcy flow

Drop `--features metal` to run on the CPU, or swap in another backend above.

The Poisson examples generate their own data first. Generation needs the
`datagen` feature, a system FFTW and, on Linux, OpenBLAS
(`brew install fftw`; `apt install libfftw3-dev libopenblas-dev`):

    cargo run --release --features datagen --example generate_poisson   # writes datasets/poisson/
    cargo run --release --example train_poisson --features metal [-- [<padding>] [<n>]]
    cargo run --release --example predict_poisson -- runs/<run_dir>
    cargo run --release --example poisson_plot -- runs/<run_dir> [<sample>]
    cargo run --release --example poisson_resolution -- runs/<run_dir> <n> [<n> ...]

`train_poisson` trains at n = 65 by default (129 and 257 on request). `predict_poisson`
reports the five errors; `poisson_plot` draws f, u, the prediction and the error on the
Chebyshev nodes; `poisson_resolution` evaluates a run on finer grids.

Each example is self-contained: the hyperparameters are Rust literals at the
top of the file, so the experiment is readable without a separate config
format. Output goes to `runs/<name>_<timestamp>/` - metrics CSV, a loss plot,
the three configs and the trained weights.

To re-evaluate a saved run:

    cargo run --release --example predict_burgers -- runs/burgers_fno_<timestamp>

## Layout

    examples/                 entry points and hyperparameters
    src/neural_operators/
      data/                   loaders, batching, transforms
      layers/                 spectral convolution
      losses/                 data loss
      metrics/                log local files
      models/                 FNO
      chebyshev/              Chebyshev nodes, quadrature, grid transfers
      pde/                    Poisson reference solvers and dataset generation
      training/               training loop, metrics
      utils/                  fft wrappers
    datasets/                 .mat files (not committed)
    runs/                     output (not committed)

## Citations

```
@misc{li2020fourier,
      title={Fourier Neural Operator for Parametric Partial Differential Equations}, 
      author={Zongyi Li and Nikola Kovachki and Kamyar Azizzadenesheli and Burigede Liu and Kaushik Bhattacharya and Andrew Stuart and Anima Anandkumar},
      year={2020},
      eprint={2010.08895},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

@misc{li2020neural,
      title={Neural Operator: Graph Kernel Network for Partial Differential Equations}, 
      author={Zongyi Li and Nikola Kovachki and Kamyar Azizzadenesheli and Burigede Liu and Kaushik Bhattacharya and Andrew Stuart and Anima Anandkumar},
      year={2020},
      eprint={2003.03485},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```