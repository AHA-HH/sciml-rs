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
import torch_harmonics
from neuralop.models import FNO

OUT = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "parity"
MAX_PARAMS = 10_000  # D8 as recommended: P ≲ 10⁴
MAX_BYTES = 1_000_000  # D8 as recommended: under about 1 MB
WEIGHT_SEED = 0
INPUT_SEED = 1

# name -> (model class, kwargs, input shape (batch, channels, *grid)).
# 2.0.0's defaults except the sizes.
CONFIGS = {
    "fno1d": (FNO, dict(n_modes=(8,), in_channels=1, out_channels=1, hidden_channels=8), (2, 1, 64)),
    "fno2d": (FNO, dict(n_modes=(4, 4), in_channels=1, out_channels=1, hidden_channels=8), (2, 1, 32, 32)),
}
DTYPES = {"f32": torch.float32, "f64": torch.float64}
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
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
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
    same &= path.with_suffix(".json").read_text() == json.dumps(meta, indent=2) + "\n"
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
