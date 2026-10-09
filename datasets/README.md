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
downloaded. They need the `chebyshev` feature (system libfftw3, and OpenBLAS on Linux):

```sh
cargo run --release --features chebyshev --example generate_poisson
```

This writes `datasets/poisson/`. Edit `DATASETS` at the top of
`examples/generate/poisson.rs` to choose the sets: stages 1 (n = 33) and 2 (n = 65) are on
by default; stages 3 and 4 (n = 129, 257) and the K = 32 evaluation sets 3e and 4e are
commented out.

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
| 3e, 4e | 129, 257 | 32 | – | about 53 MB, 211 MB |
