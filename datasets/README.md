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
subsample the spatial axes — see `subsample_rate` in each example file.

- `burgers_data_R10.mat` — the Burgers equation. Shape `[2048, 8192]`: 2048
  samples on a grid of 8192. Fields: `a` (initial condition), `u` (solution
  at t = 1).

- `piececonst_r421_N1024_smooth1.mat` and `..._smooth2.mat` — Darcy flow.
  Each `[1024, 421, 421]`: 1024 samples on a 421×421 grid. `smooth1` is used
  for training, `smooth2` for testing. Fields: `coeff` (permeability field),
  `sol` (solution).

Datasets are from Li et al., *Fourier Neural Operator for Parametric Partial
Differential Equations* (2021).