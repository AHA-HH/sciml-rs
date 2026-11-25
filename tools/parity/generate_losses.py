"""Write loss and finite-difference parity fixtures from neuraloperator 2.0.0 (library CM1.6a).

Each fixture is `tests/fixtures/parity/losses<D>d_<dtype>.npz` with a `.json` sidecar: a
seeded pair `x`, `y` (2, 2, *grid), a seeded vector field `v` (2, D, *grid), and 2.0.0's
`FiniteDiff`, `H1Loss`, `HdivLoss`, `PointwiseQuantileLoss`, `MSELoss` and `LpLoss.rel` on
them. See README.md.

    uv run python generate_losses.py            # write the fixtures
    uv run python generate_losses.py --check    # regenerate in memory, compare bit for bit
"""

import argparse
import json
import platform
import sys
import warnings

import neuralop
import numpy as np
import torch
from neuralop.losses.data_losses import H1Loss, HdivLoss, LpLoss, MSELoss, PointwiseQuantileLoss
from neuralop.losses.differentiation import FiniteDiff

from generate import DTYPES, MAX_BYTES, OUT, same_as_committed

INPUT_SEED = 2
GRIDS = {1: (32,), 2: (16, 12), 3: (8, 6, 5)}  # non-square, so a swapped axis fails parity
H = (0.1, 0.2, 0.3)  # FiniteDiff steps, per axis
MEASURE = (1.0, 2.0, 0.5)  # H1Loss / HdivLoss domain measure, per axis
ALPHA = 0.1  # PointwiseQuantileLoss
AXES = "xyz"


def boundaries(d):
    """Periodicity per axis: all periodic, none, and (d ≥ 2) mixed: x one-sided, others periodic."""
    out = {"per": (True,) * d, "non": (False,) * d}
    if d >= 2:
        out["mix"] = (False,) + (True,) * (d - 1)
    return out


def periodic_kwargs(flags):
    return {f"periodic_in_{a}": p for a, p in zip(AXES, flags)}


def inputs(d, dtype):
    """`x`, `y` (2, 2, *grid) and `v` (2, d, *grid), drawn in f32 and cast."""
    g = torch.Generator().manual_seed(INPUT_SEED)
    x = torch.randn((2, 2) + GRIDS[d], generator=g)
    y = torch.randn((2, 2) + GRIDS[d], generator=g)
    v = torch.randn((2, d) + GRIDS[d], generator=g)
    return x.to(dtype), y.to(dtype), v.to(dtype)


def arrays(d, dtype):
    x, y, v = inputs(d, dtype)
    out = {"x": x, "y": y, "v": v}
    h = H[:d]
    measure = list(MEASURE[:d])
    for bc, flags in boundaries(d).items():
        fd = FiniteDiff(dim=d, h=h, **periodic_kwargs(flags))
        for a in range(d):
            op = getattr(fd, "d" + AXES[a])
            for order in (1, 2):
                out[f"fd.d{AXES[a]}{order}.{bc}"] = op(x, order=order)
        out[f"fd.laplacian.{bc}"] = fd.laplacian(x)
        out[f"fd.gradient.{bc}"] = fd.gradient(x)
        out[f"fd.divergence.{bc}"] = fd.divergence(v)
        if d >= 2:
            out[f"fd.curl.{bc}"] = fd.curl(v)
        for name, cls in (("h1", H1Loss), ("hdiv", HdivLoss)):
            loss = cls(d=d, measure=measure, reduction="sum", **periodic_kwargs(flags))
            out[f"{name}.rel.{bc}.sum"] = loss.rel(x, y)
            out[f"{name}.abs.{bc}.sum"] = loss.abs(x, y)
    out["h1.rel.per.mean"] = H1Loss(d=d, measure=measure, reduction="mean").rel(x, y)
    for red in ("sum", "mean"):
        out[f"quantile.{red}"] = PointwiseQuantileLoss(alpha=ALPHA, reduction=red)(x, y)
    out["mse"] = MSELoss()(x, y)
    out["lp.rel"] = LpLoss(d=d, p=2, reduction="sum").rel(x, y)
    # Scalars as shape (1,): the Rust losses return [1].
    data = {k: np.atleast_1d(t.detach().numpy()) for k, t in out.items()}
    want = np.dtype(np.float32 if dtype == torch.float32 else np.float64)
    bad = {k: a.dtype for k, a in data.items() if a.dtype != want}
    assert not bad, f"arrays not in {want} (NpzFileReader reads f32/f64 only): {bad}"
    return data


def sidecar(d, dtype_name):
    return {
        "config": f"losses{d}d",
        "dtype": dtype_name,
        "input_seed": INPUT_SEED,
        "grid": list(GRIDS[d]),
        "finite_diff_h": list(H[:d]),
        "measure": list(MEASURE[:d]),
        "quantile_alpha": ALPHA,
        "boundaries": {k: list(v) for k, v in boundaries(d).items()},
        "keys": "fd.d<axis><order>.<bc>, fd.{laplacian,gradient,divergence,curl}.<bc>, "
        "{h1,hdiv}.{rel,abs}.<bc>.sum, h1.rel.per.mean, quantile.{sum,mean}, mse, lp.rel",
        "inputs": "x, y on (2, 2, *grid); v on (2, D, *grid); drawn in f32, then cast",
        "mode": "CPU, deterministic algorithms",
        "versions": {
            "python": platform.python_version(),
            "neuraloperator": neuralop.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "platform": f"{platform.system()} {platform.machine()}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="compare with the committed fixtures")
    check = ap.parse_args().check
    torch.use_deterministic_algorithms(True)
    warnings.simplefilter("ignore", DeprecationWarning)
    OUT.mkdir(parents=True, exist_ok=True)
    failed = False
    for d in GRIDS:
        for dtype_name, dtype in DTYPES.items():
            with torch.no_grad():
                data = arrays(d, dtype)
            meta = sidecar(d, dtype_name)
            path = OUT / f"losses{d}d_{dtype_name}.npz"
            if check:
                failed |= not same_as_committed(path, data, meta)
                continue
            np.savez(path, **data)
            path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
            size = path.stat().st_size
            assert size <= MAX_BYTES, f"{path.name}: {size} bytes"
            print(f"{path.name}: {len(data)} arrays, {size} bytes")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
