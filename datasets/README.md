# Datasets

These files aren't committed. Download them from the
[PDE Datasets](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-)
and place them in this directory.

| example         | download                           | extracts to                                                              |
|-----------------|------------------------------------|--------------------------------------------------------------------------|
| `burgers`       | `Burgers_R10.zip`                  | `burgers_data_R10.mat`                                                   |
| `darcy`         | `Darcy_421.zip`                    | `piececonst_r421_N1024_smooth1.mat`, `piececonst_r421_N1024_smooth2.mat` |
| `navier_stokes` | see below                          | `ns_V1e-3_N1200_T50.mat`                                                 |

The examples look for the files by exactly those names.

Each file holds named fields. The first index is the sample; the remaining
indices are the spatial (and, for Navier-Stokes, time) discretisation. The
examples subsample the spatial axes - see `subsample_rate` in each example
file.

- `burgers_data_R10.mat` - the Burgers equation. Shape `[2048, 8192]`: 2048
  samples on a grid of 8192. Fields: `a` (initial condition), `u` (solution
  at t = 1).

- `piececonst_r421_N1024_smooth1.mat` and `..._smooth2.mat` - Darcy flow.
  Each `[1024, 421, 421]`: 1024 samples on a 421×421 grid. `smooth1` is used
  for training, `smooth2` for testing. Fields: `coeff` (permeability field),
  `sol` (solution).

- `ns_V1e-3_N1200_T50.mat` - 2D incompressible Navier-Stokes in vorticity
  form on the periodic unit square, viscosity ν = 1e-3. Field `u` of shape
  `[1200, 64, 64, 50]` (single precision): sample, x, y, time - 1200
  trajectories on a 64×64 grid, 50 snapshots each. `train_navier_stokes`
  uses the first 10 snapshots as input and the next 40 as target; train is
  the first `n_train` samples, test the last `n_test`.

  The reader handles MATLAB v5 `.mat` only; a v7.3 (HDF5) file must be
  re-saved as v5 first. Loading reads the whole of `u`, so peak memory is
  roughly 3 GB (the f32 source plus its f64 copy).

Datasets are from Li et al., *Fourier Neural Operator for Parametric Partial
Differential Equations* (2021).
