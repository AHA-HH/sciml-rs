# Feature matrix: sciml-rs against neuraloperator 2.0.0

Read 2026-11-24 (CM0.1) from the source of tag `2.0.0`, commit
`b307030d9b7ee530823d6a05bddb8238ba22a743`, replacing the seed of 2026-11-11. A row is
each public item: a top-level class or function, without a leading underscore, in a
non-test module of `neuralop/`. Options and defaults are in the appendix, extracted from
the signatures with Python's `ast` and checked against the source.

*Source* is the path under `neuralop/` at 2.0.0. *Status*: **have** (output parity shown),
**partial** (exists, parity not shown or options missing), **missing**, **not planned**
(outside proposal §1.2); a parenthetical after a status qualifies it (internal: a helper,
ported with the task that needs it). *Task* is the roadmap task that sets the row; "–" means none yet.
"Exported" means importable from `neuralop` itself or from its subpackage's `__init__.py`
(`neuralop.models`, `neuralop.losses`, `neuralop.training`, `neuralop.data.datasets`); other
items are imported by module path. From `neuralop` itself: `FNO`, `TFNO`, `get_model`,
`Trainer`, `LpLoss`, `H1Loss`, `BurgersEqnLoss`, `ICLoss`, `WeightedSumLoss`, `Aggregator`,
`Relobralo`, `SoftAdapt`, `FourierDiff`, `non_uniform_fd`, `FiniteDiff`. Rows reach **have** as parity is shown (first: CM1.6a, 2026-11-25).

## Models

The registry (`get_model`) holds every subclass of `BaseModel`, by its `name=` or class
name lower-cased: `fno`, `tfno`, `sfno`, `localno`, `uqno`, `fnogno`, `gino`. `UNO` and
`CODANO` subclass `nn.Module` and are not in it.

| Item | Source | sciml-rs | Status | Task |
| --- | --- | --- | --- | --- |
| `FNO` (exported) | `models/fno.py` | `FNO<R>`, Li et al.'s architecture: linear lifting, ReLU, 1×1 convolution skip, projection through 128 channels, corner weights without bias, `backward` FFT, one-sided padding. 2.0.0's defaults differ: GELU, channel-MLP lifting and projection (ratio 2), a channel MLP per block (expansion 0.5, soft-gating skip), centred `SpectralConv` with bias and `forward` FFT, complex-normal init, symmetric padding | partial | CM1.5 |
| `TFNO` (exported; `FNO` with `factorization="Tucker"`, `rank=0.1`) | `models/fno.py` | – | missing | CM2.1 |
| `partialclass` (helper: a subclass with new defaults) | `models/fno.py` | – | not planned (Python idiom) | – |
| `SFNO` (exported, only with torch_harmonics; `FNO` with `factorization="dense"`, `conv_module=SphericalConv`) | `models/sfno.py` | – | missing | CM3.2 |
| `UNO` (exported) | `models/uno.py` | – | missing | CM2.2 |
| `LocalNO` (exported, only with torch_harmonics, which it needs through `EquidistantDiscreteContinuousConv2d`) | `models/local_no.py` | – | missing | CM2.3 |
| `UQNO` (exported; wraps a base and a residual model; returns solution and quantile) | `models/uqno.py` | – | missing | CM2.6 |
| `GINO` (exported) | `models/gino.py` | – | missing | CM3.4 |
| `FNOGNO` (exported) | `models/fnogno.py` | – | missing | CM3.4 |
| `CODANO` (exported) | `models/codano.py` | – | missing | CM3.5 |
| `BaseModel` (registry, saves init arguments) | `models/base_model.py` | `Operator` (`models/operator.rs`): config saved as `model_cfg.json` with its `kind` (`OperatorConfig::KIND`, `models/registry.rs`), `init`, `forward`, `NAME` | have (infrastructure, no output to compare) | CM0.3, CM0.4 |
| `get_model` (exported), `available_models` | `models/base_model.py` | `ModelConfig::from_json` (by `kind`, absent meaning the FNO) and `Model<R>::init` (`models/registry.rs`); `KINDS` | have (infrastructure, no output to compare) | CM0.4 |

## Layers: spectral convolutions

| Item | Source | sciml-rs | Status | Task |
| --- | --- | --- | --- | --- |
| `SpectralConv` (legacy: corner weights, `fft_norm="backward"`, `incremental_n_modes`, `joint_factorization`), `SpectralConv1d`, `SpectralConv2d`, `SpectralConv3d`, `SubConv`, `get_contract_fun` | `layers/legacy_spectral_convolution.py` | `SpectralConv<R>`: corner weights, `backward`, dense only, no bias, Li et al.'s uniform init (`SpectralInit` also offers normal and symmetric uniform). The dense layer (plus a test-only bias) matches 2.0.0's generic legacy `SpectralConv` in 1D–3D within D3 in f32 (CM1.1, `tests/parity_legacy_spectral_conv.rs`); missing: bias, `incremental_n_modes`, `joint_factorization`, factorised and separable weights, `resolution_scaling_factor`; the `SpectralConv1d/2d/3d` subclasses not fixtured | partial | CM1.1 |
| `SpectralConv` (centred weights after `fftshift`, `fft_norm="forward"`, `max_n_modes`, separable, dense / Tucker / CP / TT, bias, `init_std="auto"` = √(2 / (in + out)), complex normal), `get_contract_fun` | `layers/spectral_convolution.py` | – | missing | CM1.2, CM2.1 |
| `BaseSpectralConv` (base class) | `layers/base_spectral_conv.py` | – | missing (internal) | CM1.2 |
| `einsum_complexhalf`, `einsum_complexhalf_two_input` (half-precision complex einsum) | `layers/einsum_utils.py` | – | missing (internal) | CM1.7 |
| `SphericalConv`, `SHT`, `get_contract_fun` | `layers/spherical_convolution.py` | – | missing | CM3.1, CM3.2 |

## Layers: FNO block parts

| Item | Source | sciml-rs | Status | Task |
| --- | --- | --- | --- | --- |
| `FNOBlocks` (norm `None`, `instance_norm`, `group_norm`, `batch_norm`, `ada_in`; `fno_block_precision` `full`, `half`, `mixed`; `stabilizer` `None` or `tanh`; preactivation; `complex_data`), `SubModule` | `layers/fno_block.py` | inside `FNO`: spectral convolution plus 1×1 convolution, ReLU, no norm. Its parts exist in `layers/` (CM1.3a); the options are CM1.3b | partial | CM1.3b |
| `ChannelMLP`, `LinearChannelMLP` | `layers/channel_mlp.py` | `layers::channel_mlp::ChannelMlp` (any rank, any `ActivationConfig`, default GELU) matches `ChannelMLP` within D3 in f32 and f64, 1D and 3D (`tests/parity_block_parts{,_f64}.rs`); missing: dropout (default 0), `LinearChannelMLP` | partial | CM1.3a |
| `skip_connection` (`identity`, `linear`, `soft-gating`), `SoftGating`, `Flattened1dConv` | `layers/skip_connections.py` | `layers::skip::{SkipKind, Skip<R>}`: identity, linear (bias-free 1×1) and soft gating match 2.0.0 within D3 in f32 and f64 (linear 3D, soft gating 2D); soft gating's bias (set by no 2.0.0 caller) left out | have | CM1.3a |
| `InstanceNorm`, `BatchNorm`, `AdaIN` (group norm is torch's `nn.GroupNorm`) | `layers/normalization_layers.py` | `InstanceNorm`: Burn's, `affine = false` (1D); group norm: Burn's `GroupNorm` with one group (3D); `AdaIN`: `layers::norm::AdaIn`, the embedding a `forward` argument (2D); each within D3 in f32 and f64. Missing: `BatchNorm`, AdaIN's custom `mlp` | partial | CM1.3a |
| `GridEmbeddingND` (FNO's default `positional_embedding="grid"`), `GridEmbedding2D`, `regular_grid_nd`, `regular_grid_2d`, `Embedding` (abstract base) | `layers/embeddings.py` | grid channels built by the model, j/(s − 1) on [0, 1] with the endpoint (CONVENTIONS §2); 2.0.0's `regular_grid_nd` is j/s on [0, 1) without it | partial | CM1.4 |
| `SinusoidalEmbedding` (`transformer` or `nerf`), `RotaryEmbedding2D`, `rotate_half`, `apply_rotary_pos_emb` | `layers/embeddings.py` | – | missing | CM1.4 |
| `DomainPadding` (symmetric, fraction of the resolution) | `layers/padding.py` | `FNOConfig::padding`: p zero cells appended at the end of each axis (CONVENTIONS §3) | partial | CM1.5 |
| `resample`, `iterative_resample` (`resample`: linear in 1D, bicubic in 2D, Fourier for 3 or more axes; `iterative_resample` Fourier) | `layers/resample.py` | – | missing | CM1.5, CM2.2 |
| `ComplexValued`, `apply_complex`, `CGELU`, `ctanh` (`complex_data`) | `layers/complex.py` | – | missing | CM1.5 |

## Layers: other operators' parts

| Item | Source | sciml-rs | Status | Task |
| --- | --- | --- | --- | --- |
| `FourierContinuation` (base), `FCLegendre`, `FCGram` (precomputed matrices in `layers/fcgram_matrices/`) | `layers/fourier_continuation.py` | – | missing | CM2.4 |
| `FiniteDifferenceConvolution` | `layers/differential_conv.py` | – | missing | CM2.3 |
| `DiscreteContinuousConv` (base), `DiscreteContinuousConv2d`, `DiscreteContinuousConvTranspose2d`, `EquidistantDiscreteContinuousConv2d`, `EquidistantDiscreteContinuousConvTranspose2d` | `layers/discrete_continuous_convolution.py` | – | missing | CM2.3 |
| `LocalNOBlocks` | `layers/local_no_block.py` | – | missing | CM2.3 |
| `spectral_projection_divergence_free` | `layers/spectral_projection.py` | – | missing | CM2.7 |
| `GNOBlock` | `layers/gno_block.py` | – | missing | CM3.3 |
| `bump_cutoff`, `half_cos_cutoff`, `quadr_cutoff`, `quartic_cutoff`, `octic_cutoff`, `dispatch_weighting_fn` | `layers/gno_weighting_functions.py` | – | missing | CM3.3 |
| `IntegralTransform` | `layers/integral_transform.py` | – | missing | CM3.3 |
| `NeighborSearch`, `native_neighbor_search` (open3d optional) | `layers/neighbor_search.py` | – | missing | CM3.3 |
| `segment_csr` (torch_scatter optional) | `layers/segment_csr.py` | – | missing | CM3.3 |
| `AttentionKernelIntegral` | `layers/attention_kernel_integral.py` | – | missing | CM3.5 |
| `CODALayer` | `layers/coda_layer.py` | – | missing | CM3.5 |

## Losses

| Item | Source | sciml-rs | Status | Task |
| --- | --- | --- | --- | --- |
| `LpLoss` (exported; `d`, `p`, `measure`, quadrature weights, `reduction` `sum` or `mean`, `abs`, `rel`) | `losses/data_losses.py` | `LpLoss`: `abs` (fixed uniform-mesh weight Π h_a^(1/p), h_a = 1/(s_a − 1)) and `rel`, integer p, `Reduction` `Mean`, `Sum`, `None`; no `measure` option or `quadrature` argument (CONVENTIONS §11). `rel` matches 2.0.0's `LpLoss.rel` up to 2.0.0's eps of 1e-8 in the denominator (6e-10 to 1.9e-9 in f64, CM1.6a); `abs` keeps §11's weight | partial | CM1.6a |
| `H1Loss` (exported), `HdivLoss`, `PointwiseQuantileLoss`, `MSELoss` | `losses/data_losses.py` | `SobolevLoss<D>` (`h1`, `hdiv`; `measure`, periodicity, `reduction` `Sum`/`Mean`/`None`, `eps`, `rel`, `abs`), `PointwiseQuantileLoss`, `mse_loss`; parity ≤ 1.7e-7 (f32), ≤ 6e-16 (f64). Not ported: the per-call `quadrature` (equal to `measure = quadrature · s`), `take_root=False`, `MSELoss`'s `dim` | have | CM1.6a |
| `WeightedSumLoss` (exported), `FieldwiseAggregatorLoss` (exported), `Aggregator` (exported), `SoftAdapt` (exported), `Relobralo` (exported) | `losses/meta_losses.py` | – | missing | CM1.6b |
| `FiniteDiff` (exported), `central_diff_1d`, `central_diff_2d`, `central_diff_3d` | `losses/differentiation.py` | `FiniteDiff<D>`: `partial` (orders 1, 2; periodic or one-sided), `dx`/`dy`/`dz`, `partial_sum`, `laplacian`, `gradient`, `divergence`, `curl_2d`, `curl_3d`, any D; the deprecated `central_diff_*` are `partial` per axis; the 1D gradient keeps its component axis; parity as `H1Loss` | have | CM1.6a |
| `FourierDiff` (exported; optional Fourier continuation) | `losses/differentiation.py` | – | missing | CM1.6c |
| `non_uniform_fd` (exported), `get_non_uniform_fd_weights` | `losses/differentiation.py` | Chebyshev `diff_matrix`, `diff2_matrix` (`datagen`): a different, spectral method | missing | CM1.6c |
| `BurgersEqnLoss` (exported), `ICLoss` (exported), `PoissonInteriorLoss`, `PoissonBoundaryLoss`, `PoissonEqnLoss` | `losses/equation_losses.py` | – | missing | CM2.5 |

## Training

| Item | Source | sciml-rs | Status | Task |
| --- | --- | --- | --- | --- |
| `Trainer` (exported; `train` with optimizer, scheduler, regularizer, training and eval losses, `save_every`, `save_best`, `resume_from_dir`, `max_autoregressive_steps`; `mixed_precision`, `eval_interval`, `use_distributed`, `wandb_log`) | `training/trainer.py` | `train_fno`, `training_loop`, `EpochObserver`: Adam with a cosine schedule, or Gauss–Newton / Hybrid (`OptimizerConfig`), per-epoch evaluation, checkpoints every 25 epochs and resume, MLflow logging, data-parallel multi-GPU; generic over any `GridOperator` since CM0.3 (`train`, `build_components`, `train_components`) | partial | CM0.3 |
| `save_training_state`, `load_training_state` (exported) | `training/training_state.py` | `RunDir::save_checkpoint`, `checkpoint`; any operator's since CM0.3 (`checkpoint_of`, `OperatorCheckpoint`) | partial | CM0.3 |
| `IncrementalFNOTrainer` (exported; incremental modes by gradient or loss gap) | `training/incremental.py` | – | missing | CM1.7 |
| `MultigridPatching2D`, `make_patches` | `training/patching.py` | – | missing | CM1.7 |
| `AdamW` (exported; `correct_bias`, GaLore options) | `training/adamw.py` | Burn's Adam with weight decay | partial | CM1.7 |
| `TensorGaLoreProjector` | `training/tensor_galore_projector.py` | – | missing | CM3.6 |
| `setup` (exported; distributed and seeds from a config), `increase_l2_fetch_granularity` | `training/torch_setup.py` | multi-GPU data parallel (2D Phase 9 T2) | partial | – |
| Mixed precision (`Trainer(mixed_precision=True)`, `fno_block_precision`) | `training/trainer.py`, `layers/fno_block.py` | – | missing | CM1.7 |

## Data: datasets

| Item | Source | sciml-rs | Status | Task |
| --- | --- | --- | --- | --- |
| `DarcyDataset` (exported), `load_darcy_flow_small` (exported), `load_darcy_pt` (`.pt` files from Zenodo record 12784353) | `data/datasets/darcy.py` | Darcy loader for Li et al.'s `.mat` files: same problem, different files | partial | CM1.5 |
| `Burgers1dTimeDataset` (exported; space-time), `Burgers1dTimeDataProcessor`, `load_mini_burgers_1dtime` (exported) | `data/datasets/burgers.py` | Burgers loader for Li et al.'s `.mat` files: u(·, 0) → u(·, 1), a different task | missing | CM1.5 |
| `NavierStokesDataset` (exported), `load_navier_stokes_pt` (exported) | `data/datasets/navier_stokes.py` | – | missing | – |
| `NonlinearPoissonDataset`, `load_nonlinear_poisson_pt`, `PoissonGINODataProcessor`, `generate_latent_queries`, `generate_output_queries`, `source` | `data/datasets/nonlinear_poisson.py` | – (the crate's Poisson is linear, on a Chebyshev grid) | missing | – |
| `SphericalSWEDataset`, `load_spherical_swe` (exported with torch_harmonics) | `data/datasets/spherical_swe.py` | – | missing | – |
| `CarCFDDataset` (exported), `load_mini_car` (exported), `MeshDataModule` (exported) | `data/datasets/car_cfd_dataset.py`, `data/datasets/mesh_datamodule.py` | – | missing | – |
| `TheWellDataset`, `ActiveMatterDataset`, `MHD64Dataset` (exported with the_well) | `data/datasets/the_well_dataset.py` | – | missing | – |
| `PTDataset` (exported) | `data/datasets/pt_dataset.py` | – (`.mat` and `.npz` readers) | missing | – |
| `H5pyDataset` | `data/datasets/hdf5_dataset.py` | – | missing | – |
| `ZarrDataset` | `data/datasets/zarr_dataset.py` | – | missing | – |
| `TensorDataset`, `GeneralTensorDataset` | `data/datasets/tensor_dataset.py` | `OperatorDataset` | partial | – |
| `DictDataset` (exported) | `data/datasets/dict_dataset.py` | – | missing | – |
| `download_from_url`, `download_from_zenodo_record`, `calculate_md5`, `check_md5`, `check_integrity` | `data/datasets/web_utils.py` | – | not planned | – |

## Data: transforms

| Item | Source | sciml-rs | Status | Task |
| --- | --- | --- | --- | --- |
| `UnitGaussianNormalizer` (`eps=1e-07`, `dim`, `mask`), `Normalizer`, `DictUnitGaussianNormalizer` | `data/transforms/normalizers.py` | `UnitGaussianNormalizer` (CONVENTIONS §10), default eps 1e-5 against 2.0.0's 1e-7 | partial | – |
| `DataProcessor` (base), `DefaultDataProcessor` (normalise in, decode out) | `data/transforms/data_processors.py` | `Normalizers`, `FlatDecoder` (loader-internal `normalize_split_pair`) | partial | – |
| `IncrementalDataProcessor` (incremental resolution) | `data/transforms/data_processors.py` | – | missing | CM1.7 |
| `MGPatchingDataProcessor` | `data/transforms/data_processors.py` | – | missing | CM1.7 |
| `Transform`, `CompositeTransform`, `DictTransform` | `data/transforms/base_transforms.py` | – | missing | – |
| `MGPatchingTransform`, `RandomMGPatch`, `MGPTensorDataset` | `data/transforms/patching_transforms.py` | – | missing | CM1.7 |
| `TheWellDataProcessor` | `data/transforms/the_well_data_processors.py` | – | missing | – |

## Model parallelism and utilities

| Item | Source | sciml-rs | Status | Task |
| --- | --- | --- | --- | --- |
| `mpu.comm` (`init`, world, rank and group queries, `disable_logging`) | `mpu/comm.py` | – (data parallel only) | not planned | – |
| `mpu.helpers` (`pad_helper`, `truncate_helper`, `split_tensor_along_dim`, `get_memory_format`) | `mpu/helpers.py` | – | not planned | – |
| `mpu.mappings` (copy, reduce, scatter, gather across the model-parallel group) | `mpu/mappings.py` | – | not planned | – |
| `count_model_params`, `count_tensor_params` | `utils.py` | Burn's `Module::num_params`, recorded in `costs.json` | partial | – |
| `spectrum_2d` | `utils.py` | – | missing | – |
| `validate_scaling_factor` (internal) | `utils.py` | – | missing (internal) | CM1.5 |
| `compute_rank`, `compute_stable_rank`, `compute_explained_variance` | `utils.py` | POD energy (`metrics::pod`, behind `datagen`: `cumulative_energy`, `energy_rank_of`), a different method | partial | – |
| `wandb_login`, `set_wandb_api_key`, `get_wandb_api_key`, `get_project_root` | `utils.py` | MLflow export and live logging instead | not planned | – |

## Beyond 2.0.0 (this crate only)

Chebyshev transfers and quadrature, reference Poisson solvers, POD, Gauss–Newton,
per-run directories, MLflow export and live logging, bit-identity and GPU parity tests.

## Appendix: options and defaults at 2.0.0

Each public class's `__init__` or function's signature, `self` dropped, defaults as
written in the source (quotes normalised by `ast.unparse`). Generated with `ast` from the tag; `F` is
`torch.nn.functional`.

### `data/datasets/burgers.py`

- `Burgers1dTimeDataProcessor(*args, **kwargs)`
- `Burgers1dTimeDataset(root_dir, n_train, n_tests, train_resolution=16, test_resolutions=[16], batch_size=32, test_batch_sizes=32, temporal_subsample=None, spatial_subsample=None, pad=0)`
- `load_mini_burgers_1dtime(data_path, n_train, n_test, batch_size, test_batch_size, temporal_subsample=1, spatial_subsample=1)`

### `data/datasets/car_cfd_dataset.py`

- `CarCFDDataset(root_dir, n_train=1, n_test=1, query_res=[32, 32, 32], download=True)`
- `load_mini_car()`

### `data/datasets/darcy.py`

- `DarcyDataset(root_dir, n_train, n_tests, batch_size, test_batch_sizes, train_resolution, test_resolutions=[16, 32], encode_input=False, encode_output=True, encoding='channel-wise', channel_dim=1, subsampling_rate=None, download=True)`
- `load_darcy_flow_small(n_train, n_tests, batch_size, test_batch_sizes, data_root=example_data_root, test_resolutions=[16, 32], encode_input=False, encode_output=True, encoding='channel-wise', channel_dim=1)`
- `load_darcy_pt(n_train, n_tests, batch_size, test_batch_sizes, data_root='./neuralop/data/datasets/data', train_resolution=16, test_resolutions=[16, 32], encode_input=False, encode_output=True, encoding='channel-wise', channel_dim=1, num_workers=1)`

### `data/datasets/dict_dataset.py`

- `DictDataset(data_list, constant=None)`

### `data/datasets/hdf5_dataset.py`

- `H5pyDataset(data_path, resolution=128, transform_x=None, transform_y=None, n_samples=None)`

### `data/datasets/mesh_datamodule.py`

- `MeshDataModule(root_dir, item_dir_name, n_train=None, n_test=None, query_res=None, attributes=None)`

### `data/datasets/navier_stokes.py`

- `NavierStokesDataset(root_dir, n_train, n_tests, batch_size, test_batch_sizes, train_resolution, test_resolutions=[16, 32], encode_input=False, encode_output=True, encoding='channel-wise', channel_dim=1, subsampling_rate=None, download=True)`
- `load_navier_stokes_pt(n_train, n_tests, batch_size, test_batch_sizes, data_root=example_data_root, train_resolution=128, test_resolutions=[128], encode_input=False, encode_output=True, encoding='channel-wise', channel_dim=1, subsampling_rate=None, num_workers=1)`

### `data/datasets/nonlinear_poisson.py`

- `generate_latent_queries(query_res, pad=0.0, domain_lims=[[-1.4, 1.4], [-1.4, 1.4]])`
- `generate_output_queries(grid_res, coefs, domain_lims=[[-1.4, 1.4], [-1.4, 1.4]], tol=1e-07)`
- `source(x, beta, mu_1, mu_2)`
- `NonlinearPoissonDataset(data_path, n_train=1, n_test=1, n_in=6000, n_out=6000, n_bound=1024, n_eval=6000, latent_query_res=48, domain_padding=0, input_min_sample_points=None, input_max_sample_points=None, input_subsample_level=None, output_subsample_level=None, train_out_res=None, provide_output_queries_as_dict=True, download=True)`
- `load_nonlinear_poisson_pt(data_path, query_res=48, domain_padding=0, val_on_same_instance=False, n_train=1, n_test=1, n_in=6000, n_out=6000, n_eval=6000, n_bound=1024, input_min_sample_points=None, input_max_sample_points=None, input_subsample_level=None, output_subsample_level=None, train_out_res=None, return_dict=True)`
- `PoissonGINODataProcessor(device='cpu', in_normalizer=None, out_normalizer=None, positional_encoding=None, input_min=100, input_max=1000, input_sub_level=None, output_sub_level=None, return_dict=True)`

### `data/datasets/pt_dataset.py`

- `PTDataset(root_dir, dataset_name, n_train, n_tests, batch_size, test_batch_sizes, train_resolution, test_resolutions, encode_input=False, encode_output=True, encoding='channel-wise', input_subsampling_rate=None, output_subsampling_rate=None, channel_dim=1, channels_squeezed=True)`

### `data/datasets/spherical_swe.py`

- `load_spherical_swe(n_train, n_tests, batch_size, test_batch_sizes, train_resolution=(256, 512), test_resolutions=[(256, 512)], device=torch.device('cpu'))`
- `SphericalSWEDataset(dt=3600, dims=(256, 512), initial_condition='random', num_examples=32, device=torch.device('cpu'), normalize=True, stream=None)`

### `data/datasets/tensor_dataset.py`

- `TensorDataset(x, y, transform_x=None, transform_y=None)`
- `GeneralTensorDataset(sets, transforms)`

### `data/datasets/the_well_dataset.py`

- `TheWellDataset(root_dir, well_dataset_name, train_task='next_step', eval_tasks=['next_step'], n_steps_input=1, n_steps_output=1, download=True, first_only=True)`
- `ActiveMatterDataset(root_dir, train_task='next_step', eval_tasks=['next_step'], n_steps_input=1, n_steps_output=1, download=True, first_only=True)`
- `MHD64Dataset(root_dir, train_task='next_step', eval_tasks=['next_step'], download=True, first_only=True)`

### `data/datasets/web_utils.py`

- `calculate_md5(fpath, chunk_size=1024 * 1024)`
- `check_md5(fpath, md5, **kwargs)`
- `check_integrity(fpath, md5=None)`
- `download_from_url(url, root, filename=None, md5=None, size=None, chunk_size=256 * 64, extract_tars=True)`
- `download_from_zenodo_record(record_id, root, files_to_download=None)`

### `data/datasets/zarr_dataset.py`

- `ZarrDataset(filename, resolution=128, transform_x=None, transform_y=None, n_samples=None)`

### `data/transforms/base_transforms.py`

- `Transform()`
- `CompositeTransform(transforms)`
- `DictTransform(transform_dict, input_mappings, return_mappings=None)`

### `data/transforms/data_processors.py`

- `DataProcessor()`
- `DefaultDataProcessor(in_normalizer=None, out_normalizer=None)`
- `IncrementalDataProcessor(in_normalizer=None, out_normalizer=None, device='cpu', subsampling_rates=[2, 1], dataset_resolution=16, dataset_indices=[2, 3], epoch_gap=10, verbose=False)`
- `MGPatchingDataProcessor(model, levels, padding_fraction, stitching, device='cpu', use_distributed=False, in_normalizer=None, out_normalizer=None)`

### `data/transforms/normalizers.py`

- `Normalizer(mean, std, eps=1e-06)`
- `UnitGaussianNormalizer(mean=None, std=None, eps=1e-07, dim=None, mask=None)`
- `DictUnitGaussianNormalizer(normalizer_dict, input_mappings, return_mappings)`

### `data/transforms/patching_transforms.py`

- `MGPatchingTransform(model, levels, padding_fraction, stitching)`
- `RandomMGPatch(levels=2)`
- `MGPTensorDataset(x, y, levels=2)`

### `data/transforms/the_well_data_processors.py`

- `TheWellDataProcessor(data_normalizer, const_normalizer=None, n_steps_input=1, n_steps_output=1, time_as_channels=True)`

### `layers/attention_kernel_integral.py`

- `AttentionKernelIntegral(in_channels, out_channels, n_heads, head_n_channels, project_query=True)`

### `layers/base_spectral_conv.py`

- `BaseSpectralConv(device=None, dtype=None)`

### `layers/channel_mlp.py`

- `ChannelMLP(in_channels, out_channels=None, hidden_channels=None, n_layers=2, n_dim=2, non_linearity=F.gelu, dropout=0.0)`
- `LinearChannelMLP(layers, non_linearity=F.gelu, dropout=0.0)`

### `layers/coda_layer.py`

- `CODALayer(n_modes, n_heads=1, token_codimension=1, head_codimension=None, codimension_size=None, per_channel_attention=True, permutation_eq=True, norm='instance_norm', temperature=1.0, nonlinear_attention=False, scale=None, resolution_scaling_factor=None, incremental_n_modes=None, non_linearity=F.gelu, use_channel_mlp=True, channel_mlp_expansion=1.0, fno_skip='linear', channel_mlp_skip='linear', preactivation=False, separable=False, factorization='tucker', rank=1.0, joint_factorization=False, conv_module=SpectralConv, fixed_rank_modes=False, implementation='factorized', decomposition_kwargs=None)`

### `layers/complex.py`

- `CGELU(x)`
- `ctanh(x)`
- `apply_complex(real_func, imag_func, x, dtype=torch.cfloat)`
- `ComplexValued(module)`

### `layers/differential_conv.py`

- `FiniteDifferenceConvolution(in_channels, out_channels, n_dim, kernel_size=3, groups=1, padding='periodic')`

### `layers/discrete_continuous_convolution.py`

- `DiscreteContinuousConv(in_channels, out_channels, kernel_shape, basis_type='piecewise_linear', groups=1, bias=True)`
- `DiscreteContinuousConv2d(in_channels, out_channels, grid_in, grid_out, kernel_shape, basis_type='piecewise_linear', n_in=None, n_out=None, quadrature_weights=None, periodic=False, groups=1, bias=True, radius_cutoff=None)`
- `DiscreteContinuousConvTranspose2d(in_channels, out_channels, grid_in, grid_out, kernel_shape, basis_type='piecewise_linear', n_in=None, n_out=None, quadrature_weights=None, periodic=False, groups=1, bias=True, radius_cutoff=None)`
- `EquidistantDiscreteContinuousConv2d(in_channels, out_channels, in_shape, out_shape, kernel_shape, basis_type='piecewise_linear', domain_length=None, periodic=False, groups=1, bias=True, radius_cutoff=None)`
- `EquidistantDiscreteContinuousConvTranspose2d(in_channels, out_channels, in_shape, out_shape, kernel_shape, basis_type='piecewise_linear', domain_length=None, periodic=False, groups=1, bias=True, radius_cutoff=None)`

### `layers/einsum_utils.py`

- `einsum_complexhalf_two_input(eq, a, b)`
- `einsum_complexhalf(eq, *args)`

### `layers/embeddings.py`

- `Embedding()`
- `GridEmbedding2D(in_channels, grid_boundaries=[[0, 1], [0, 1]])`
- `GridEmbeddingND(in_channels, dim=2, grid_boundaries=[[0, 1], [0, 1]])`
- `SinusoidalEmbedding(in_channels, num_frequencies=None, embedding_type='transformer', max_positions=10000)`
- `RotaryEmbedding2D(dim, min_freq=1 / 64, scale=1.0)`
- `regular_grid_2d(spatial_dims, grid_boundaries=[[0, 1], [0, 1]])`
- `regular_grid_nd(resolutions, grid_boundaries=[[0, 1]] * 2)`
- `rotate_half(x)`
- `apply_rotary_pos_emb(t, freqs)`

### `layers/fno_block.py`

- `FNOBlocks(in_channels, out_channels, n_modes, resolution_scaling_factor=None, n_layers=1, max_n_modes=None, fno_block_precision='full', use_channel_mlp=True, channel_mlp_dropout=0, channel_mlp_expansion=0.5, non_linearity=F.gelu, stabilizer=None, norm=None, ada_in_features=None, preactivation=False, fno_skip='linear', channel_mlp_skip='soft-gating', complex_data=False, separable=False, factorization=None, rank=1.0, conv_module=SpectralConv, fixed_rank_modes=False, implementation='factorized', decomposition_kwargs=dict())`
- `SubModule(main_module, indices)`

### `layers/fourier_continuation.py`

- `FourierContinuation(d=5, n_additional_pts=50)`
- `FCLegendre(d=5, n_additional_pts=50, rcond=1e-15)`
- `FCGram(d=5, n_additional_pts=50, matrices_path=None)`

### `layers/gno_block.py`

- `GNOBlock(in_channels, out_channels, coord_dim, radius, transform_type='linear', weighting_fn=None, reduction='sum', pos_embedding_type='transformer', pos_embedding_channels=32, pos_embedding_max_positions=10000, channel_mlp_layers=[128, 256, 128], channel_mlp_non_linearity=F.gelu, channel_mlp=None, use_torch_scatter_reduce=True, use_open3d_neighbor_search=True)`

### `layers/gno_weighting_functions.py`

- `bump_cutoff(x, radius=1.0, scale=1.0, eps=1e-07)`
- `half_cos_cutoff(x, radius=1.0, scale=1.0)`
- `quadr_cutoff(x, radius=1.0, scale=1.0)`
- `quartic_cutoff(x, radius=1.0, scale=1.0)`
- `octic_cutoff(x, radius=1.0, scale=1.0)`
- `dispatch_weighting_fn(weight_function_name, sq_radius, scale)`

### `layers/integral_transform.py`

- `IntegralTransform(channel_mlp=None, channel_mlp_layers=None, channel_mlp_non_linearity=F.gelu, transform_type='linear', weighting_fn=None, reduction='sum', use_torch_scatter=True)`

### `layers/legacy_spectral_convolution.py`

- `get_contract_fun(weight, implementation='reconstructed', separable=False)`
- `SpectralConv(in_channels, out_channels, n_modes, incremental_n_modes=None, bias=True, n_layers=1, separable=False, resolution_scaling_factor=None, fno_block_precision='full', rank=0.5, factorization=None, implementation='reconstructed', fixed_rank_modes=False, joint_factorization=False, decomposition_kwargs=None, init_std='auto', fft_norm='backward', device=None, dtype=None)`
- `SubConv(main_conv, indices)`
- `SpectralConv1d` (no `__init__` of its own)
- `SpectralConv2d` (no `__init__` of its own)
- `SpectralConv3d` (no `__init__` of its own)

### `layers/local_no_block.py`

- `LocalNOBlocks(in_channels, out_channels, n_modes, default_in_shape, resolution_scaling_factor=None, n_layers=1, disco_layers=True, disco_kernel_shape=[2, 4], radius_cutoff=None, domain_length=[2, 2], disco_groups=1, disco_bias=True, diff_layers=True, conv_padding_mode='periodic', fin_diff_kernel_size=3, mix_derivatives=True, max_n_modes=None, local_no_block_precision='full', use_channel_mlp=False, channel_mlp_dropout=0, channel_mlp_expansion=0.5, non_linearity=F.gelu, stabilizer=None, norm=None, ada_in_features=None, preactivation=False, local_no_skip='linear', channel_mlp_skip='soft-gating', separable=False, factorization=None, rank=1.0, conv_module=SpectralConv, fixed_rank_modes=False, implementation='factorized', decomposition_kwargs=dict(), fft_norm='forward')`

### `layers/neighbor_search.py`

- `NeighborSearch(use_open3d=True, return_norm=False)`
- `native_neighbor_search(data, queries, radius, return_norm=False)`

### `layers/normalization_layers.py`

- `AdaIN(embed_dim, in_channels, mlp=None, eps=1e-05)`
- `InstanceNorm(**kwargs)`
- `BatchNorm(n_dim, num_features, **kwargs)`

### `layers/padding.py`

- `DomainPadding(domain_padding, resolution_scaling_factor=1)`

### `layers/resample.py`

- `resample(x, res_scale, axis, output_shape=None)`
- `iterative_resample(x, res_scale, axis)`

### `layers/segment_csr.py`

- `segment_csr(src, indptr, reduction, use_scatter=True)`

### `layers/skip_connections.py`

- `skip_connection(in_features, out_features, n_dim=2, bias=False, skip_type='soft-gating')`
- `SoftGating(in_features, out_features=None, n_dim=2, bias=False)`
- `Flattened1dConv(in_channels, out_channels, kernel_size, bias=False)`

### `layers/spectral_convolution.py`

- `get_contract_fun(weight, implementation='reconstructed', separable=False)`
- `SpectralConv(in_channels, out_channels, n_modes, complex_data=False, max_n_modes=None, bias=True, separable=False, resolution_scaling_factor=None, fno_block_precision='full', rank=0.5, factorization=None, implementation='reconstructed', fixed_rank_modes=False, decomposition_kwargs=None, init_std='auto', fft_norm='forward', device=None)`

### `layers/spectral_projection.py`

- `spectral_projection_divergence_free(u, domain_size, constraint_modes)`

### `layers/spherical_convolution.py`

- `get_contract_fun(weight, implementation='reconstructed', separable=False)`
- `SHT(dtype=torch.float32, device=None)`
- `SphericalConv(in_channels, out_channels, n_modes, max_n_modes=None, bias=True, separable=False, resolution_scaling_factor=None, fno_block_precision='full', rank=0.5, factorization='cp', implementation='reconstructed', fixed_rank_modes=False, joint_factorization=False, decomposition_kwargs=dict(), init_std='auto', sht_norm='ortho', sht_grids='equiangular', device=None, dtype=torch.float32, complex_data=False)`

### `losses/data_losses.py`

- `LpLoss(d=1, p=2, measure=1.0, reduction='sum', eps=1e-08)`
- `H1Loss(d=1, measure=1.0, reduction='sum', eps=1e-08, periodic_in_x=True, periodic_in_y=True, periodic_in_z=True)`
- `HdivLoss(d=1, measure=1.0, reduction='sum', eps=1e-08, periodic_in_x=True, periodic_in_y=True, periodic_in_z=True)`
- `PointwiseQuantileLoss(alpha, reduction='sum')`
- `MSELoss()`

### `losses/differentiation.py`

- `FiniteDiff(dim, h=1.0, periodic_in_x=True, periodic_in_y=True, periodic_in_z=True)`
- `central_diff_1d(x, h, periodic_in_x=True)`
- `central_diff_2d(x, h, periodic_in_x=True, periodic_in_y=True)`
- `central_diff_3d(x, h, periodic_in_x=True, periodic_in_y=True, periodic_in_z=True)`
- `get_non_uniform_fd_weights(points, num_neighbors=5, derivative_indices=[0], radius=None, regularize_lstsq=False)`
- `non_uniform_fd(points, values, num_neighbors=5, derivative_indices=[0], radius=None, regularize_lstsq=False)`
- `FourierDiff(dim, L=None, use_fc=False, fc_degree=4, fc_n_additional_pts=50, low_pass_filter_ratio=None)`

### `losses/equation_losses.py`

- `BurgersEqnLoss(visc=0.01, method='fdm', loss=F.mse_loss, domain_length=1.0)`
- `ICLoss(loss=F.mse_loss)`
- `PoissonInteriorLoss(method='autograd', loss=F.mse_loss)`
- `PoissonBoundaryLoss(loss=F.mse_loss)`
- `PoissonEqnLoss(boundary_weight, interior_weight, diff_method='autograd', base_loss=F.mse_loss)`

### `losses/meta_losses.py`

- `FieldwiseAggregatorLoss(losses, mappings, logging=False)`
- `WeightedSumLoss(losses, weights=None)`
- `Aggregator(params, num_losses, weights)`
- `SoftAdapt(params, num_losses, eps=1e-08, weights=None)`
- `Relobralo(params, num_losses, alpha=0.95, beta=0.99, tau=1.0, eps=1e-08, weights=None)`

### `models/base_model.py`

- `BaseModel` (no `__init__` of its own)
- `available_models()`
- `get_model(config)`

### `models/codano.py`

- `CODANO(output_variable_codimension=1, lifting_channels=64, hidden_variable_codimension=32, projection_channels=64, use_positional_encoding=False, positional_encoding_dim=8, positional_encoding_modes=None, static_channel_dim=0, variable_ids=None, use_horizontal_skip_connection=False, horizontal_skips_map=None, n_layers=4, n_modes=None, per_layer_scaling_factors=None, n_heads=None, attention_scaling_factors=None, conv_module=SpectralConv, nonlinear_attention=False, non_linearity=F.gelu, attention_token_dim=1, per_channel_attention=False, layer_kwargs={}, domain_padding=0.25, enable_cls_token=False)`

### `models/fno.py`

- `FNO(n_modes, in_channels, out_channels, hidden_channels, n_layers=4, lifting_channel_ratio=2, projection_channel_ratio=2, positional_embedding='grid', non_linearity=F.gelu, norm=None, complex_data=False, use_channel_mlp=True, channel_mlp_dropout=0, channel_mlp_expansion=0.5, channel_mlp_skip='soft-gating', fno_skip='linear', resolution_scaling_factor=None, domain_padding=None, fno_block_precision='full', stabilizer=None, max_n_modes=None, factorization=None, rank=1.0, fixed_rank_modes=False, implementation='factorized', decomposition_kwargs=None, separable=False, preactivation=False, conv_module=SpectralConv)`
- `partialclass(new_name, cls, *args, **kwargs)`
- `TFNO(*args, **kwargs)`

### `models/fnogno.py`

- `FNOGNO(in_channels, out_channels, projection_channel_ratio=4, gno_coord_dim=3, gno_pos_embed_type='transformer', gno_transform_type='linear', fno_n_modes=(16, 16, 16), fno_hidden_channels=64, fno_lifting_channel_ratio=4, fno_n_layers=4, gno_embed_channels=32, gno_embed_max_positions=10000, gno_radius=0.033, gno_weighting_function=None, gno_weight_function_scale=1.0, gno_channel_mlp_hidden_layers=[512, 256], gno_channel_mlp_non_linearity=F.gelu, gno_use_open3d=True, gno_use_torch_scatter=True, gno_batched=False, fno_resolution_scaling_factor=None, fno_block_precision='full', fno_use_channel_mlp=True, fno_channel_mlp_dropout=0, fno_channel_mlp_expansion=0.5, fno_non_linearity=F.gelu, fno_stabilizer=None, fno_norm=None, fno_ada_in_features=None, fno_ada_in_dim=1, fno_preactivation=False, fno_skip='linear', fno_channel_mlp_skip='soft-gating', fno_separable=False, fno_factorization=None, fno_rank=1.0, fno_fixed_rank_modes=False, fno_implementation='factorized', fno_decomposition_kwargs=dict(), fno_conv_module=SpectralConv)`

### `models/gino.py`

- `GINO(in_channels, out_channels, latent_feature_channels=None, projection_channel_ratio=4, gno_coord_dim=3, in_gno_radius=0.033, out_gno_radius=0.033, in_gno_transform_type='linear', out_gno_transform_type='linear', gno_weighting_function=None, gno_weight_function_scale=1, in_gno_pos_embed_type='transformer', out_gno_pos_embed_type='transformer', fno_in_channels=3, fno_n_modes=(16, 16, 16), fno_hidden_channels=64, fno_lifting_channel_ratio=2, fno_n_layers=4, gno_embed_channels=32, gno_embed_max_positions=10000, in_gno_channel_mlp_hidden_layers=[80, 80, 80], out_gno_channel_mlp_hidden_layers=[512, 256], gno_channel_mlp_non_linearity=F.gelu, gno_use_open3d=True, gno_use_torch_scatter=True, out_gno_tanh=None, fno_resolution_scaling_factor=None, fno_block_precision='full', fno_use_channel_mlp=True, fno_channel_mlp_dropout=0, fno_channel_mlp_expansion=0.5, fno_non_linearity=F.gelu, fno_stabilizer=None, fno_norm=None, fno_ada_in_features=4, fno_ada_in_dim=1, fno_preactivation=False, fno_skip='linear', fno_channel_mlp_skip='soft-gating', fno_separable=False, fno_factorization=None, fno_rank=1.0, fno_fixed_rank_modes=False, fno_implementation='factorized', fno_decomposition_kwargs=dict(), fno_conv_module=SpectralConv)`

### `models/local_no.py`

- `LocalNO(n_modes, in_channels, out_channels, hidden_channels, default_in_shape, n_layers=4, disco_layers=True, disco_kernel_shape=[2, 4], radius_cutoff=None, domain_length=[2, 2], disco_groups=1, disco_bias=True, diff_layers=True, conv_padding_mode='periodic', fin_diff_kernel_size=3, mix_derivatives=True, lifting_channel_ratio=2, projection_channel_ratio=2, positional_embedding='grid', non_linearity=F.gelu, norm=None, complex_data=False, use_channel_mlp=False, channel_mlp_dropout=0, channel_mlp_expansion=0.5, channel_mlp_skip='soft-gating', local_no_skip='linear', resolution_scaling_factor=None, domain_padding=None, local_no_block_precision='full', stabilizer=None, max_n_modes=None, factorization=None, rank=1.0, fixed_rank_modes=False, implementation='factorized', decomposition_kwargs=dict(), separable=False, preactivation=False, conv_module=SpectralConv)`

### `models/sfno.py`

- `SFNO = partialclass("SFNO", FNO, factorization="dense", conv_module=SphericalConv)` (FNO's signature with those defaults)

### `models/uno.py`

- `UNO(in_channels, out_channels, hidden_channels, lifting_channels=256, projection_channels=256, positional_embedding='grid', n_layers=4, uno_out_channels=None, uno_n_modes=None, uno_scalings=None, horizontal_skips_map=None, channel_mlp_dropout=0, channel_mlp_expansion=0.5, non_linearity=F.gelu, norm=None, preactivation=False, fno_skip='linear', horizontal_skip='linear', channel_mlp_skip='soft-gating', separable=False, factorization=None, rank=1.0, fixed_rank_modes=False, integral_operator=SpectralConv, operator_block=FNOBlocks, implementation='factorized', decomposition_kwargs=dict(), domain_padding=None, verbose=False)`

### `models/uqno.py`

- `UQNO(base_model, residual_model=None, **kwargs)`

### `mpu/comm.py`

- `disable_logging(level=logging.ERROR)`
- `get_world_size()`
- `get_local_rank()`
- `get_global_rank()`
- `get_data_parallel_size()`
- `get_data_parallel_rank()`
- `get_data_parallel_group()`
- `get_model_parallel_size()`
- `get_model_parallel_rank()`
- `get_model_parallel_group()`
- `init(model_parallel_size=1, verbose=False)`

### `mpu/helpers.py`

- `get_memory_format(tensor)`
- `pad_helper(tensor, dim, new_size, mode='zero')`
- `truncate_helper(tensor, dim, new_size)`
- `split_tensor_along_dim(tensor, dim, num_chunks)`

### `mpu/mappings.py`

- `copy_to_model_parallel_region(input_)`
- `reduce_from_model_parallel_region(input_)`
- `scatter_to_model_parallel_region(input_, dim)`
- `gather_from_model_parallel_region(input_, dim)`

### `training/adamw.py`

- `AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-06, weight_decay=0.0, correct_bias=True, galore_params=None, galore_rank=1.0, galore_update_proj_gap=50, galore_scale=1.0, activation_checkpoint=False, warm_restart=True)`

### `training/incremental.py`

- `IncrementalFNOTrainer(model, n_epochs, wandb_log=False, device='cpu', mixed_precision=False, data_processor=None, eval_interval=1, log_output=False, use_distributed=False, verbose=False, incremental_grad=False, incremental_loss_gap=False, incremental_grad_eps=0.001, incremental_buffer=5, incremental_max_iter=1, incremental_grad_max_iter=10, incremental_loss_eps=0.001)`

### `training/patching.py`

- `MultigridPatching2D(model, levels=0, padding_fraction=0, use_distributed=False, stitching=True)`
- `make_patches(x, n, p=0)`

### `training/tensor_galore_projector.py`

- `TensorGaLoreProjector(rank, update_proj_gap=200, scale=1.0, tucker_n_iter_max=10, warm_restart=False, activation_checkpoint=False)`

### `training/torch_setup.py`

- `setup(config)`
- `increase_l2_fetch_granularity()`

### `training/trainer.py`

- `Trainer(*, model, n_epochs, wandb_log=False, device='cpu', mixed_precision=False, data_processor=None, eval_interval=1, log_output=False, use_distributed=False, verbose=False)`

### `training/training_state.py`

- `load_training_state(save_dir, save_name, model, optimizer=None, scheduler=None, regularizer=None, map_location=None)`
- `save_training_state(save_dir, save_name, model, optimizer=None, scheduler=None, regularizer=None, epoch=None)`

### `utils.py`

- `count_model_params(model)`
- `count_tensor_params(tensor, dims=None)`
- `wandb_login(api_key_file='../config/wandb_api_key.txt', key=None)`
- `set_wandb_api_key(api_key_file='../config/wandb_api_key.txt')`
- `get_wandb_api_key(api_key_file='../config/wandb_api_key.txt')`
- `spectrum_2d(signal, n_observations, normalize=True)`
- `validate_scaling_factor(scaling_factor, n_dim, n_layers=None)`
- `compute_rank(tensor)`
- `compute_stable_rank(tensor)`
- `compute_explained_variance(frequency_max, s)`
- `get_project_root()`
