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
| 2D Darcy   | s = 16  | 0.03403 | 0.03451 |

Same hyperparameters across columns. The PyTorch port is a direct
translation of Li et al.'s reference implementation in the official 
PyTorch neural operators toolbox, run locally.

The spectral layers support any grid size (non-power-of-two FFTs go through
Bluestein's algorithm), and `examples/train/darcy.rs` now trains at the
paper's s = 85 with 12 modes. The Darcy row above is from an earlier s = 16
configuration; an s = 85 comparison against PyTorch hasn't been run yet.

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
for download links.

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
is enough. Every example prints the device it picked. To force one without
rebuilding, set `BURN_DEVICE` (`cuda`, `rocm`, `metal`, `vulkan`, `wgpu`,
`flex`):

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