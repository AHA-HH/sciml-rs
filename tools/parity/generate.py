"""Write output-parity fixtures from neuraloperator 2.0.0 (library objective, CM0.2).

Each fixture is `tests/fixtures/parity/<config>_<dtype>.npz` with a `.json` sidecar:
the weights of a seeded 2.0.0 model, a seeded input and the model's output on the CPU.
See README.md for the format.

    uv run python generate.py            # write every fixture
    uv run python generate.py --check    # regenerate in memory, compare bit for bit
"""

import argparse
import json
import platform
import sys
from pathlib import Path

import neuralop
import numpy as np
import tensorly
import tltorch
import torch
import torch.nn.functional as F
import torch_harmonics
from neuralop.layers.channel_mlp import ChannelMLP
from neuralop.layers.embeddings import GridEmbeddingND, RotaryEmbedding2D, SinusoidalEmbedding
from neuralop.layers.fno_block import FNOBlocks
from neuralop.layers.legacy_spectral_convolution import SpectralConv as LegacySpectralConv
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.layers.normalization_layers import AdaIN, InstanceNorm
from neuralop.layers.skip_connections import skip_connection
from neuralop.models import FNO

OUT = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "parity"
MAX_PARAMS = 10_000  # D8 as recommended: P ≲ 10⁴
MAX_BYTES = 1_000_000  # D8 as recommended: under about 1 MB
WEIGHT_SEED = 0
INPUT_SEED = 1


class AdaINWithEmbedding(AdaIN):
    """2.0.0's `AdaIN` with its embedding drawn at build and kept as a buffer, so the fixture
    carries it (2.0.0 sets it by `set_embedding` before each forward)."""

    def __init__(self, embed_dim, in_channels):
        super().__init__(embed_dim, in_channels)
        del self.embedding
        self.register_buffer("embedding", torch.randn(embed_dim))


def legacy_conv(in_channels, out_channels, n_modes, **_):
    """2.0.0's legacy layer as `FNOBlocks`' `conv_module`, without its bias (the crate's
    `SpectralConv` has none). `FNOBlocks` also passes the new layer's keywords
    (`max_n_modes`, `separable`, …), all at their defaults, which the legacy layer lacks."""
    return LegacySpectralConv(in_channels, out_channels, n_modes, bias=False)


class FNOBlocksRun(torch.nn.Module):
    """2.0.0's `FNOBlocks` over every layer index, as its `FNO.forward` runs them, with the
    legacy convolution; with AdaIN one seeded embedding, a buffer, set on every norm."""

    def __init__(self, ada_in_features=None, **kwargs):
        super().__init__()
        self.fno_blocks = FNOBlocks(conv_module=legacy_conv, ada_in_features=ada_in_features, **kwargs)
        embedding = None if ada_in_features is None else torch.randn(ada_in_features)
        self.register_buffer("embedding", embedding)

    def forward(self, x):
        if self.embedding is not None:
            self.fno_blocks.set_ada_in_embeddings(self.embedding)
        for i in range(self.fno_blocks.n_layers):
            x = self.fno_blocks(x, i)
        return x


# The activations `FNOConfig::activation` offers, in the order of the crate's `Activation`
# (`F.leaky_relu` at its default slope, 0.01).
ACTIVATIONS = [F.relu, F.gelu, torch.tanh, torch.sigmoid, F.silu, F.leaky_relu, F.selu, F.mish, F.hardswish, F.softsign]


class Activations(torch.nn.Module):
    """Each of `ACTIVATIONS` applied to a one-channel input, stacked along the channels."""

    def forward(self, x):
        return torch.cat([f(x) for f in ACTIVATIONS], dim=1)


class Rotary2d(torch.nn.Module):
    """2.0.0's `RotaryEmbedding2D(dim)` applied in 2D: channels 0 and 1 of the input are the x
    and y coordinates, the remaining `2 · dim` the features `t`; the output is
    `apply_2d_rotary_pos_emb(t, rope(x), rope(y))`."""

    def __init__(self, dim):
        super().__init__()
        self.rope = RotaryEmbedding2D(dim)

    def forward(self, x):
        freqs_x, freqs_y = self.rope(x[..., 0]), self.rope(x[..., 1])
        return RotaryEmbedding2D.apply_2d_rotary_pos_emb(x[..., 2:], freqs_x, freqs_y)


BOUNDARIES = [[0.0, 2.0], [-1.0, 1.0]]


def fno_boundaries(in_channels, **kwargs):
    """2.0.0's `FNO` with a `GridEmbeddingND` on `BOUNDARIES` instead of the unit square."""
    grid = GridEmbeddingND(in_channels=in_channels, dim=2, grid_boundaries=BOUNDARIES)
    return FNO(in_channels=in_channels, positional_embedding=grid, **kwargs)


class FNOOutputShape(FNO):
    """2.0.0's `FNO` whose forward passes `output_shape` (a tuple: the last layer's shape)."""

    def __init__(self, output_shape, **kwargs):
        super().__init__(**kwargs)
        self.output_shape = output_shape

    def forward(self, x):
        return super().forward(x, output_shape=self.output_shape)


def blocks(n_modes,channels=4, n_layers=2, **kwargs):
    """`FNOBlocks(channels, channels, n_modes, n_layers, **kwargs)`, 2.0.0's defaults otherwise."""
    return FNOBlocksRun(in_channels=channels, out_channels=channels, n_modes=n_modes, n_layers=n_layers, **kwargs)


# name -> (model class, kwargs, input shape (batch, channels, *grid)).
# 2.0.0's defaults except the sizes.
CONFIGS = {
    "fno1d": (FNO, dict(n_modes=(8,), in_channels=1, out_channels=1, hidden_channels=8), (2, 1, 64)),
    "fno2d": (FNO, dict(n_modes=(4, 4), in_channels=1, out_channels=1, hidden_channels=8), (2, 1, 32, 32)),
    # CM1.1: the legacy corner-layout layer, in ≠ out, odd and non-power-of-two extents.
    "legacy_spectral_conv1d": (LegacySpectralConv, dict(in_channels=3, out_channels=4, n_modes=(12,)), (2, 3, 50)),
    "legacy_spectral_conv2d": (LegacySpectralConv, dict(in_channels=3, out_channels=4, n_modes=(8, 6)), (2, 3, 24, 21)),
    "legacy_spectral_conv3d": (LegacySpectralConv, dict(in_channels=2, out_channels=3, n_modes=(6, 4, 4)), (2, 2, 12, 10, 9)),
    # CM1.2: the centred layer (`fft_norm="forward"`), converted to corners at load.
    "spectral_conv2d": (SpectralConv, dict(in_channels=3, out_channels=4, n_modes=(8, 6)), (2, 3, 24, 21)),
    "spectral_conv2d_backward": (SpectralConv, dict(in_channels=3, out_channels=4, n_modes=(8, 6), fft_norm="backward"), (2, 3, 24, 21)),
    "spectral_conv2d_odd": (SpectralConv, dict(in_channels=3, out_channels=4, n_modes=(7, 6)), (2, 3, 24, 21)),
    "spectral_conv3d": (SpectralConv, dict(in_channels=2, out_channels=3, n_modes=(6, 5, 4)), (2, 2, 12, 10, 9)),
    "spectral_conv2d_separable": (SpectralConv, dict(in_channels=4, out_channels=4, n_modes=(8, 6), separable=True), (2, 4, 24, 21)),
    "spectral_conv2d_max_modes": (SpectralConv, dict(in_channels=3, out_channels=4, n_modes=(6, 6), max_n_modes=(10, 8)), (2, 3, 24, 21)),
    # Odd extent on a shifted axis: 2.0.0 applies `fftshift` again (not `ifftshift`) before the
    # inverse, so its output spectrum moves by one bin there; corners cannot represent this.
    "spectral_conv2d_odd_extent": (SpectralConv, dict(in_channels=3, out_channels=4, n_modes=(8, 6)), (2, 3, 23, 21)),
    # CM1.3a: the block parts of FNOBlocks, 1D to 3D.
    "channel_mlp1d": (ChannelMLP, dict(in_channels=4, hidden_channels=2), (2, 4, 7)),
    "channel_mlp3d": (ChannelMLP, dict(in_channels=3, out_channels=5, hidden_channels=4, n_layers=3, non_linearity=torch.tanh), (2, 3, 5, 4, 3)),
    "linear_skip3d": (skip_connection, dict(in_features=3, out_features=5, n_dim=3, skip_type="linear"), (2, 3, 5, 4, 3)),
    "soft_gating2d": (skip_connection, dict(in_features=4, out_features=4, n_dim=2, skip_type="soft-gating"), (2, 4, 6, 5)),
    "instance_norm1d": (InstanceNorm, dict(), (2, 4, 9)),
    "group_norm3d": (torch.nn.GroupNorm, dict(num_groups=1, num_channels=4), (2, 4, 5, 4, 3)),
    "ada_in2d": (AdaINWithEmbedding, dict(embed_dim=3, in_channels=4), (2, 4, 6, 5)),
    # CM1.3b: FNOBlocks, post- and pre-activation, for each norm.
    "fno_blocks_post1d": (blocks, dict(n_modes=(8,)), (2, 4, 20)),
    "fno_blocks_pre1d": (blocks, dict(n_modes=(8,), preactivation=True), (2, 4, 20)),
    "fno_blocks_instance_post2d": (blocks, dict(n_modes=(6, 4), norm="instance_norm"), (2, 4, 12, 10)),
    "fno_blocks_instance_pre2d": (blocks, dict(n_modes=(6, 4), norm="instance_norm", preactivation=True), (2, 4, 12, 10)),
    "fno_blocks_group_post3d": (blocks, dict(n_modes=(4, 4, 4), norm="group_norm", fno_skip="soft-gating", channel_mlp_skip="linear", channel_mlp_expansion=1.0), (2, 4, 8, 7, 6)),
    "fno_blocks_group_pre3d": (blocks, dict(n_modes=(4, 4, 4), norm="group_norm", fno_skip="identity", channel_mlp_skip="identity", preactivation=True), (2, 4, 8, 7, 6)),
    # AdaIN's MLP is 512 wide: one layer and 2 channels keep P within D8.
    "fno_blocks_ada_in_post2d": (blocks, dict(n_modes=(6, 4), channels=2, n_layers=1, norm="ada_in", ada_in_features=2), (2, 2, 12, 10)),
    "fno_blocks_ada_in_pre2d": (blocks, dict(n_modes=(6, 4), channels=2, n_layers=1, norm="ada_in", ada_in_features=2, preactivation=True), (2, 2, 12, 10)),
    # No channel MLP, no norm: 2.0.0 applies the activation twice per layer (tanh shows it).
    "fno_blocks_plain1d": (blocks, dict(n_modes=(8,), use_channel_mlp=False, fno_skip="soft-gating", non_linearity=torch.tanh), (2, 4, 20)),
    "activations1d": (Activations, dict(), (2, 1, 64)),
    # CM1.4: the embedding, lifting and projection alone (`n_layers=0`: no Fourier layer).
    "fno_lift_project1d": (FNO, dict(n_modes=(8,), in_channels=2, out_channels=1, hidden_channels=6, n_layers=0), (2, 2, 20)),
    # Lifting ratio 0: one layer; projection int(1.5 · 5) = 7 wide.
    "fno_lift_project2d_linear": (FNO, dict(n_modes=(4, 4), in_channels=2, out_channels=3, hidden_channels=5, n_layers=0, lifting_channel_ratio=0, projection_channel_ratio=1.5, non_linearity=torch.tanh), (2, 2, 6, 5)),
    "fno_lift_project3d_no_grid": (FNO, dict(n_modes=(4, 4, 4), in_channels=3, out_channels=2, hidden_channels=4, n_layers=0, positional_embedding=None), (2, 3, 5, 4, 3)),
    "fno_lift_project2d_boundaries": (fno_boundaries, dict(n_modes=(4, 4), in_channels=2, out_channels=1, hidden_channels=4, n_layers=0), (2, 2, 6, 5)),
    # CM1.5a: the whole FNO (centred convolutions with bias) over a sweep of its options.
    "fno3d": (FNO, dict(n_modes=(4, 4, 4), in_channels=2, out_channels=1, hidden_channels=4, n_layers=2), (2, 2, 8, 6, 5)),
    "fno2d_instance_pre": (FNO, dict(n_modes=(6, 4), in_channels=1, out_channels=1, hidden_channels=6, norm="instance_norm", preactivation=True, non_linearity=torch.tanh), (2, 1, 16, 12)),
    "fno1d_group_skips": (FNO, dict(n_modes=(10,), in_channels=2, out_channels=1, hidden_channels=6, norm="group_norm", fno_skip="soft-gating", channel_mlp_skip="linear", channel_mlp_expansion=1.0, lifting_channel_ratio=0, projection_channel_ratio=1.5, positional_embedding=None, non_linearity=F.silu), (2, 2, 40)),
    "fno2d_separable": (FNO, dict(n_modes=(6, 6), max_n_modes=(8, 8), separable=True, in_channels=1, out_channels=1, hidden_channels=6), (2, 1, 16, 16)),
    # An odd centred count (7): the crate's corners carry a zero row for it (CM1.2).
    "fno2d_odd_stabilizer": (FNO, dict(n_modes=(7, 6), in_channels=1, out_channels=1, hidden_channels=6, fno_skip="identity", channel_mlp_skip=None, stabilizer="tanh"), (2, 1, 16, 14)),
    "fno1d_pre_no_mlp": (FNO, dict(n_modes=(8,), in_channels=1, out_channels=1, hidden_channels=6, use_channel_mlp=False, preactivation=True, stabilizer="tanh", non_linearity=F.relu), (2, 1, 32)),
    # CM1.5b: domain padding (symmetric, a fraction per axis) and resolution scaling (the
    # convolution's output size, the skips resampled: linear 1D, bicubic 2D, Fourier 3D).
    "fno1d_padding": (FNO, dict(n_modes=(8,), in_channels=1, out_channels=1, hidden_channels=6, domain_padding=0.25), (2, 1, 40)),
    "fno2d_padding": (FNO, dict(n_modes=(6, 4), in_channels=1, out_channels=1, hidden_channels=6, domain_padding=[0.2, 0.1]), (2, 1, 20, 18)),
    "fno3d_padding": (FNO, dict(n_modes=(4, 4, 4), in_channels=2, out_channels=1, hidden_channels=4, n_layers=2, domain_padding=0.25), (2, 2, 8, 6, 5)),
    "fno1d_scaling": (FNO, dict(n_modes=(8,), in_channels=1, out_channels=1, hidden_channels=6, resolution_scaling_factor=[1.5, 1, 0.5, 1]), (2, 1, 32)),
    "fno2d_scaling": (FNO, dict(n_modes=(6, 4), in_channels=1, out_channels=1, hidden_channels=6, n_layers=2, resolution_scaling_factor=[[2, 1.5], [0.5, 1]]), (2, 1, 12, 10)),
    "fno3d_scaling": (FNO, dict(n_modes=(4, 4, 4), in_channels=2, out_channels=1, hidden_channels=4, n_layers=2, resolution_scaling_factor=[[2, 1, 1.5], [1, 0.5, 1]]), (2, 2, 8, 6, 5)),
    # Padding with scaling runs in 2.0.0 only when one layer scales (its unpad lookup).
    "fno2d_padding_scaling": (FNO, dict(n_modes=(6, 4), in_channels=1, out_channels=1, hidden_channels=6, n_layers=1, domain_padding=0.25, resolution_scaling_factor=2), (2, 1, 12, 10)),
    "fno2d_output_shape": (FNOOutputShape, dict(output_shape=(20, 15), n_modes=(6, 4), in_channels=1, out_channels=1, hidden_channels=6, n_layers=2), (2, 1, 12, 10)),
    "grid_embedding2d": (GridEmbeddingND, dict(in_channels=3, dim=2, grid_boundaries=BOUNDARIES), (2, 3, 6, 5)),
    "sinusoidal_transformer": (SinusoidalEmbedding, dict(in_channels=3, num_frequencies=4), (2, 10, 3)),
    "sinusoidal_nerf": (SinusoidalEmbedding, dict(in_channels=3, num_frequencies=4, embedding_type="nerf"), (2, 10, 3)),
    "rotary2d": (Rotary2d, dict(dim=8), (2, 10, 18)),
}
# Models with parameters whose 2.0.0 init is a constant (ones, zeros): those parameters (and
# only those) are redrawn from a seeded normal, so the fixture tests the map and not only the
# constant.
# CM1.5a's sweeps: the soft-gating and group-norm weights.
SWEEPS = {"fno3d", "fno2d_instance_pre", "fno1d_group_skips", "fno2d_separable", "fno2d_odd_stabilizer", "fno1d_pre_no_mlp"}
# CM1.5b's sweeps: padding, scaling, the output shape.
SWEEPS |= {n for n in CONFIGS if n.endswith(("_padding", "_scaling", "_output_shape"))}
REDRAW = {"soft_gating2d", "group_norm3d"} | {n for n in CONFIGS if n.startswith("fno_blocks")} | SWEEPS
DTYPES = {"f32": torch.float32, "f64": torch.float64}
# f32 only: the legacy layer computes its spectrum in `cfloat` and returns f32 (no bias to
# promote it), so an f64 reference would be f32-accurate, and torch refuses a norm with f64
# parameters on it.
F32_ONLY = {n for n in CONFIGS if n.startswith("fno_blocks")}
# And the activations: D3 needs f32 parity, and an f64 test would need a binary of its own.
F32_ONLY.add("activations1d")
# And the sweeps: f32 parity is D3's test; the f64 defaults (`fno1d`, `fno2d`) record the f64 error.
F32_ONLY |= SWEEPS
PREFIX = "param."


def build(name, dtype):
    """The seeded model and input, in `dtype`, on the CPU.

    Both are drawn in f32 and cast: 2.0.0 creates its spectral weights as `cfloat` whatever
    the default dtype, so an f64 model exists only by casting. The f64 fixture thus has the
    f32 fixture's weights and input, and differs only in the precision of the forward.
    """
    cls, kwargs, shape = CONFIGS[name]
    torch.manual_seed(WEIGHT_SEED)
    model = cls(**kwargs).eval()
    if name in REDRAW:
        with torch.no_grad():
            for p in model.parameters():
                # shortcut: "constant" is read from the values, so a 1-element parameter is
                # never redrawn; name the parameters if a constant scalar init appears. Complex
                # (spectral) weights are drawn at random, never constant.
                if not p.is_complex() and p.numel() > 1 and p.unique().numel() == 1:
                    p.copy_(torch.randn_like(p))
    x = torch.randn(shape, generator=torch.Generator().manual_seed(INPUT_SEED))
    if dtype == torch.float64:
        for t in list(model.parameters()) + list(model.buffers()):
            t.data = t.data.to(torch.complex128 if t.is_complex() else torch.float64)
        x = x.double()
    return model, x


def arrays(model, x, dtype):
    """The fixture's arrays (complex as a trailing (real, imag) axis) and the complex names."""
    with torch.no_grad():
        y = model(x)
    out = {"input": x.numpy(), "output": y.numpy()}
    complex_names = []
    for k, v in model.state_dict().items():
        if not torch.is_tensor(v):
            continue  # `_metadata`: 2.0.0's init kwargs, recorded in the sidecar instead
        if v.is_complex():
            complex_names.append(k)
            v = torch.view_as_real(v)
        out[PREFIX + k] = v.detach().numpy()
    want = np.dtype(np.float32 if dtype == torch.float32 else np.float64)
    bad = {k: a.dtype for k, a in out.items() if a.dtype != want}
    assert not bad, f"arrays not in {want} (NpzFileReader reads f32/f64 only): {bad}"
    return out, complex_names


def reload_output(path, name, dtype, complex_names):
    """The output of a fresh model with the weights read back from `path`."""
    model, _ = build(name, dtype)
    for p in model.parameters():  # same seed, same weights: zero them so only the file's count
        p.data.zero_()
    with np.load(path) as f:
        state = {}
        for k in f.files:
            if not k.startswith(PREFIX):
                continue
            t = torch.from_numpy(f[k])
            key = k[len(PREFIX):]
            state[key] = torch.view_as_complex(t.contiguous()) if key in complex_names else t
        x = torch.from_numpy(f["input"])
    model.load_state_dict(state, strict=True)  # 2.0.0's load_state_dict tolerates no `_metadata`
    with torch.no_grad():
        return model(x).numpy()


def json_text(meta):
    """The sidecar's text; a function argument (`non_linearity`) is named by `__name__`."""
    return json.dumps(meta, indent=2, default=lambda f: f.__name__) + "\n"


def sidecar(name, dtype_name, model, x, complex_names):
    _, kwargs, _ = CONFIGS[name]
    n_params = sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())
    return {
        "config": name,
        "model": type(model).__name__,
        "kwargs": kwargs,
        "dtype": dtype_name,
        "weight_seed": WEIGHT_SEED,
        "input_seed": INPUT_SEED,
        "input_shape": list(x.shape),
        "n_params_real": n_params,
        "param_prefix": PREFIX,
        "complex_params": complex_names,
        "complex_layout": "trailing axis of 2: (real, imag), torch.view_as_real",
        "mode": "eval, torch.no_grad, CPU",
        "versions": {
            "python": platform.python_version(),
            "neuraloperator": neuralop.__version__,
            "torch": torch.__version__,
            "tensorly": tensorly.__version__,
            "tensorly-torch": tltorch.__version__,
            "torch-harmonics": torch_harmonics.__version__,
            "numpy": np.__version__,
        },
        "platform": f"{platform.system()} {platform.machine()}",
    }


def write(path, name, dtype, data, meta):
    """Write one fixture and its sidecar; assert D8's bounds and the reload round trip."""
    np.savez(path, **data)
    path.with_suffix(".json").write_text(json_text(meta))
    size = path.stat().st_size
    assert size <= MAX_BYTES, f"{path.name}: {size} bytes"
    y = reload_output(path, name, dtype, meta["complex_params"])
    assert np.array_equal(y, data["output"]), f"{path.name}: reloaded output differs"
    print(f"{path.name}: P = {meta['n_params_real']}, {size} bytes, round trip bit for bit")


def same_as_committed(path, data, meta):
    """Whether the committed fixture and sidecar equal `data` and `meta`, bit for bit."""
    with np.load(path) as f:
        same = set(f.files) == set(data) and all(
            np.array_equal(f[k], v) and f[k].dtype == v.dtype for k, v in data.items()
        )
    same &= path.with_suffix(".json").read_text() == json_text(meta)
    print(f"{path.name}: {'same' if same else 'DIFFERS'}")
    return same


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="compare with the committed fixtures")
    check = ap.parse_args().check
    torch.use_deterministic_algorithms(True)
    OUT.mkdir(parents=True, exist_ok=True)
    failed = False
    for name in CONFIGS:
        for dtype_name, dtype in DTYPES.items():
            if dtype_name == "f64" and name in F32_ONLY:
                continue
            model, x = build(name, dtype)
            data, complex_names = arrays(model, x, dtype)
            path = OUT / f"{name}_{dtype_name}.npz"
            meta = sidecar(name, dtype_name, model, x, complex_names)
            assert meta["n_params_real"] <= MAX_PARAMS, f"{path.name}: P = {meta['n_params_real']}"
            if check:
                failed |= not same_as_committed(path, data, meta)
            else:
                write(path, name, dtype, data, meta)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
