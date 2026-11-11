# Feature matrix: sciml-rs against neuraloperator 2.0.0

Seeded 2026-11-11 from the file tree of tag `2.0.0` (commit `b307030`) and the class lists
of `embeddings.py`, `data_losses.py`, `meta_losses.py` and `models/__init__.py`. Not yet
verified: CM0.1 completes it (every public item, its options and defaults) and has it
checked independently.

Status: **have** (output parity shown), **partial** (exists, parity not shown or options
missing), **missing**, **not planned**. *Task* is the roadmap task that sets the row.

## Models (`neuralop/models/`)

| Item | sciml-rs | Status | Task |
| --- | --- | --- | --- |
| FNO (`fno.py`) | `FNO<R>`, Li et al.'s architecture | partial | CM1.5 |
| TFNO (`fno.py`, Tucker, rank 0.1) | – | missing | CM2.1 |
| UNO (`uno.py`) | – | missing | CM2.2 |
| LocalNO (`local_no.py`, needs torch_harmonics) | – | missing | CM2.3 |
| UQNO (`uqno.py`) | – | missing | CM2.6 |
| SFNO (`sfno.py`, needs torch_harmonics) | – | missing | CM3.2 |
| GINO (`gino.py`) | – | missing | CM3.4 |
| FNOGNO (`fnogno.py`) | – | missing | CM3.4 |
| CODANO (`codano.py`) | – | missing | CM3.5 |
| `get_model` (`base_model.py`) | – | missing | CM0.4 |

## Layers (`neuralop/layers/`)

| Item | sciml-rs | Status | Task |
| --- | --- | --- | --- |
| `legacy_spectral_convolution` (corners, `backward`) | `SpectralConv` | partial | CM1.1 |
| `spectral_convolution` (centred, `forward`, factorised, separable) | – | missing | CM1.2, CM2.1 |
| `spherical_convolution` | – | missing | CM3.2 |
| `fno_block` | inside `FNO` | partial | CM1.3 |
| `channel_mlp` | – | missing | CM1.3 |
| `skip_connections` (linear, identity, soft-gating) | linear 1×1 | partial | CM1.3 |
| `normalization_layers` (instance, group, AdaIN) | – | missing | CM1.3 |
| `embeddings` (`GridEmbedding2D`, `GridEmbeddingND`, `SinusoidalEmbedding`, `RotaryEmbedding2D`) | grid channels | partial | CM1.4 |
| `padding` | domain padding | partial | CM1.5 |
| `resample` | – | missing | CM1.5, CM2.2 |
| `complex` | – | missing | CM1.5 |
| `fourier_continuation` (`FCLegendre`, `FCGram`) | – | missing | CM2.4 |
| `differential_conv` | – | missing | CM2.3 |
| `discrete_continuous_convolution` | – | missing | CM2.3 |
| `local_no_block` | – | missing | CM2.3 |
| `spectral_projection` (divergence-free) | – | missing | CM2.7 |
| `gno_block`, `gno_weighting_functions`, `integral_transform` | – | missing | CM3.3 |
| `neighbor_search`, `segment_csr` | – | missing | CM3.3 |
| `attention_kernel_integral`, `coda_layer` | – | missing | CM3.5 |
| `einsum_utils`, `base_spectral_conv` | – | (internal) | CM1.2 |

## Losses (`neuralop/losses/`)

| Item | sciml-rs | Status | Task |
| --- | --- | --- | --- |
| `LpLoss` | `LpLoss` | partial | CM1.6 |
| `H1Loss`, `HdivLoss`, `PointwiseQuantileLoss`, `MSELoss` | – | missing | CM1.6 |
| `FieldwiseAggregatorLoss`, `WeightedSumLoss`, `SoftAdapt`, `Relobralo` | – | missing | CM1.6 |
| `differentiation` | Chebyshev D, D2 (`datagen`) | partial | CM1.6 |
| `equation_losses` | – | missing | CM2.5 |

## Training (`neuralop/training/`)

| Item | sciml-rs | Status | Task |
| --- | --- | --- | --- |
| `trainer`, `training_state` | trainer, checkpoint and resume | partial | CM0.3 |
| `incremental` | – | missing | CM1.7 |
| `patching` | – | missing | CM1.7 |
| `adamw` | Burn's Adam | partial | CM1.7 |
| `tensor_galore_projector` | – | missing | CM3.6 |
| `torch_setup` (distributed) | multi-GPU (2D Phase 9 T2) | partial | – |

## Data (`neuralop/data/datasets/`)

| Item | sciml-rs | Status | Task |
| --- | --- | --- | --- |
| Burgers, Darcy | `.mat` loaders | partial | CM1.5 |
| Navier–Stokes, nonlinear Poisson, spherical SWE, car CFD, The Well | – | missing | with the first operator needing each |
| HDF5, Zarr, `.pt` readers | `.mat`, `.npz` | partial | as needed |

## Beyond 2.0.0 (this crate only)

Chebyshev transfers and quadrature, reference Poisson solvers, POD, Gauss–Newton,
per-run directories, MLflow export and live logging, bit-identity and GPU parity tests.
