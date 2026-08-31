# sciml-rs

A dimension-generic Fourier Neural Operator in Rust, built on [Burn](https://burn.dev).

Tensor rank is a compile-time property, so the spatial dimensionality of a
problem is fixed when the binary is built rather than checked at runtime.
`modes: vec![16]` gives you an 1D FNO
`modes: vec![4, 4]` gives you a 2D FNO.

## Results


## Requirements

- Rust 1.XX or later ([rustup.rs](https://rustup.rs))
- ~2 GB disk for the datasets

## Getting started

    cargo build --release  # Build target
    cargo test             # Check tests pass
    cargo doc --open       # API documentation

## Datasets

Datasets aren't committed — see [`datasets/README.md`](datasets/README.md)
for download links.

## Running examples

    cargo run --release --example burgers_fno    # 1D Burgers
    cargo run --release --example darcy_fno      # 2D Darcy flow

Each example is self-contained: the hyperparameters are Rust literals at the
top of the file, so the experiment is readable without a separate config
format. Output goes to `runs/<name>_<timestamp>/` — metrics CSV, a loss plot,
the three configs, and the trained weights.

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