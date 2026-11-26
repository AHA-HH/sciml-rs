# Datasets

These files aren't committed. Download them from the
[PDE Datasets](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-)
and place them in this directory.

| example   | download          | extracts to                                                              |
|-----------|-------------------|--------------------------------------------------------------------------|
| `burgers` | `Burgers_R10.zip` | `burgers_data_R10.mat`                                                   |
| `darcy`   | `Darcy_421.zip`   | `piececonst_r421_N1024_smooth1.mat`, `piececonst_r421_N1024_smooth2.mat` |

The examples look for the `.mat` files by exactly those names.

Each file is a MATLAB `.mat` holding named fields. The first index is the
sample; the remaining indices are the spatial discretisation. Both examples
subsample the spatial axes - see `subsample_rate` in each example file.

- `burgers_data_R10.mat` - the Burgers equation. Shape `[2048, 8192]`: 2048
  samples on a grid of 8192. Fields: `a` (initial condition), `u` (solution
  at t = 1).

- `piececonst_r421_N1024_smooth1.mat` and `..._smooth2.mat` - Darcy flow.
  Each `[1024, 421, 421]`: 1024 samples on a 421×421 grid. `smooth1` is used
  for training, `smooth2` for testing. Fields: `coeff` (permeability field),
  `sol` (solution).

Datasets are from Li et al., *Fourier Neural Operator for Parametric Partial
Differential Equations* (2021).

## Poisson (generated)

The Poisson datasets of `docs/design/2d-chebyshev-poisson-fno.md` §5 are generated, not
downloaded. They need the `datagen` feature (system libfftw3, and OpenBLAS on Linux):

```sh
cargo run --release --features datagen --example generate_poisson
```

This writes `datasets/poisson/`: a train and a test split of every set in `GENERATED`
(`src/neural_operators/data/catalogue.rs`), all on by default: stages 1 to 4
(n = 33, 65, 129, 257) and the K = 32 evaluation sets 3e and 4e (n = 129, 257), with their
K = 32 train splits, which the resolution study uses only to refit the normalisers.

Each split is a pair of files named `poisson_n{n}_K{K}_{split}`, e.g.
`poisson_n65_K32_train.npz`:

- `.npz`, uncompressed, f64: `f` (GRF forcing) and `u` (solver-A solution), both
  `[N, n, n]` with `f[i, a, b] = f(x_a, y_b)`, and `x`, `y` (the ascending CGL nodes,
  `[n]`). The fields stay on the Chebyshev grid; the loader transfers them to the uniform
  grid.
- `.json` sidecar: n and s, K and whether it was set explicitly, τ, α, K_max, the RNG, the
  base seed and count (sample i uses seed base + i), ε_K, the label check against the
  exact sine series (max, mean, whether it was enforced, the 1e-8 tolerance), the transfer
  round-trip error 2 (max, mean, d, s), the convention version, the crate version, the git
  commit and the wall time.

Train splits use base seed 0 and test splits 1 000 000, at every n. Sizes (1000 train /
200 test, f + u):

| Set | n | K | train | test |
| --- | --- | --- | --- | --- |
| 1 | 33 | 16 | 17.4 MB | 3.5 MB |
| 2 | 65 | 32 | 67.6 MB | 13.5 MB |
| 3 | 129 | 64 | about 266 MB | about 53 MB |
| 4 | 257 | 64 | about 1.06 GB | about 211 MB |
| 3e, 4e | 129, 257 | 32 | about 266 MB, 1.06 GB | about 53 MB, 211 MB |

## Poisson 3D (generated)

The 3D Poisson datasets of `docs/design/3d-chebyshev-laplace-fno.md` (D3, D5), written
beside the 2D files in `datasets/poisson/`:

```sh
cargo run --release --features datagen --example generate_poisson3d [-- <n> ...]
```

A train and a test split of every set in `GENERATED_3D`, or of the n given: n = 33
(K = 16; tests and Mac runs) and n = 65 (K = 32; training, generated on the VM). Each split
is a pair of files named `poisson3d_n{n}_K{K}_{split}`, laid out as the 2D ones with a third
axis: `f` and `u` (A3 solution) `[N, n, n, n]` with `f[i, a, b, c] = f(x_a, y_b, z_c)`, and
`x`, `y`, `z`; the sidecar has the 2D fields, with the 3D α (3), RNG (ChaCha8 stream 3) and
ε_K. The label check is enforced for n ≥ 65. Seeds and counts as in 2D.

| n | K | train | test |
| --- | --- | --- | --- |
| 33 | 16 | 575 MB | 115 MB |
| 65 | 32 | 4.39 GB (zip64) | 879 MB |
