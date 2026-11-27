"""Write the Tensor GaLore projector parity fixtures from neuraloperator 2.0.0 (library CM3.6a).

`tests/fixtures/parity/galore_<dtype>.npz` with a `.json` sidecar: for each case of `CASES`,
a structured gradient (a Tucker tensor of the case's ranks plus noise) and 2.0.0's
`TensorGaLoreProjector` on it. Keys, each prefixed `<case>.`: `grad`, `grad2`, `ranks` (the
validated Tucker ranks, as floats), `factor<k>` (mode k's factor), `low`
(`project(grad, 0)`), `back` (`project_back(low)`), `low2` (`project(grad2, gap)`, with the
first factors: 2.0.0 computes them once).

`galore_complex_<dtype>.npz` (library CM3.6b): the same cases on complex gradients, complex
arrays stored as `torch.view_as_real` (a trailing axis of 2). 2.0.0 casts a complex gradient
to `cfloat` and then fails on a `cdouble` one, so the f64 fixture is tensorly's `tucker` on
the `cdouble` gradient with 2.0.0's `transform` / `inverse_transform`. See README.md.

    uv run python generate_galore.py            # write the fixtures
    uv run python generate_galore.py --check    # regenerate in memory, compare bit for bit
"""

import argparse
import sys
import warnings

import numpy as np
import torch
from neuralop.training.tensor_galore_projector import TensorGaLoreProjector
from tensorly import tenalg
from tensorly.decomposition import tucker

from generate import DTYPES, OUT
from generate_losses import emit, finish, provenance

SEED = 7
NOISE = 0.05
# name -> (shape, 2.0.0's rank argument, update_proj_gap, scale, the structure's ranks)
CASES = {
    "matrix": ((12, 7), 3, 5, 1.0, (3, 3)),
    "tensor4": ((6, 5, 4, 3), [3, 2, 3, 2], 200, 0.25, (3, 2, 3, 2)),
    "fraction": ((8, 6, 5), 0.5, 200, 1.0, (5, 3, 3)),
}


def structured(shape, ranks, gen, field=torch.float32):
    """A Tucker tensor of `ranks` with decaying core entries, plus noise, drawn in `field`
    (f32 or cfloat): its leading singular values are separated, so the HOOI stops well inside
    tolerance."""
    decay = torch.tensor([2.0 ** -i for i in range(max(ranks))])
    core = torch.randn(ranks, generator=gen, dtype=field)
    for k, r in enumerate(ranks):
        view = [1] * len(ranks)
        view[k] = r
        core = core * decay[:r].reshape(view)
    factors = [torch.randn(s, r, generator=gen, dtype=field) for s, r in zip(shape, ranks)]
    noise = torch.randn(shape, generator=gen, dtype=field)
    return tenalg.multi_mode_dot(core, factors) + NOISE * noise


class CdoubleProjector(TensorGaLoreProjector):
    """2.0.0's projector without its cast of complex gradients to `cfloat` (which makes it
    fail on `cdouble`): tensorly's `tucker` on the gradient as given."""

    def get_projection_tensor(self, weights, rank):
        return tucker(weights, rank=rank, init="svd")[1]


def arrays(dtype, complex_=False):
    gen = torch.Generator().manual_seed(SEED)
    field = torch.cfloat if complex_ else torch.float32
    cast = {torch.float32: torch.cfloat, torch.float64: torch.cdouble}[dtype] if complex_ else dtype
    projector = CdoubleProjector if cast == torch.cdouble else TensorGaLoreProjector
    out = {}
    for name, (shape, rank, gap, scale, structure) in CASES.items():
        grad = structured(shape, structure, gen, field).to(cast)
        grad2 = structured(shape, structure, gen, field).to(cast)
        p = projector(rank, update_proj_gap=gap, scale=scale)
        low = p.project(grad, 0)
        back = p.project_back(low)
        low2 = p.project(grad2, gap)
        ranks = torch.tensor([f.shape[1] for f in p.proj_tensor], dtype=dtype)
        parts = {"grad": grad, "grad2": grad2, "ranks": ranks, "low": low, "back": back, "low2": low2}
        parts |= {f"factor{k}": f for k, f in enumerate(p.proj_tensor)}
        if complex_:
            parts = {k: v if k == "ranks" else torch.view_as_real(v.resolve_conj()) for k, v in parts.items()}
        out |= {f"{name}.{k}": v for k, v in parts.items()}
    return finish(out, dtype)


def sidecar(dtype_name, complex_=False):
    extra = {
        "complex_layout": "trailing axis of 2: (real, imag), torch.view_as_real",
        "projector": "2.0.0's, cfloat" if dtype_name == "f32" else "tensorly tucker on cdouble with 2.0.0's transforms (2.0.0 fails on cdouble)",
    } if complex_ else {}
    return {
        "config": "galore_complex" if complex_ else "galore",
        "dtype": dtype_name,
        "seed": SEED,
        "noise": NOISE,
        "cases": {
            n: {"shape": list(s), "rank": r, "update_proj_gap": g, "scale": c, "structure": list(t)}
            for n, (s, r, g, c, t) in CASES.items()
        },
        "keys": "<case>.{grad,grad2,ranks,factor<k>,low,back,low2}",
        "inputs": f"drawn in {'cfloat' if complex_ else 'f32'}, then cast; TensorGaLoreProjector defaults otherwise",
        **extra,
        "mode": "CPU, deterministic algorithms",
        **provenance(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="compare with the committed fixtures")
    check = ap.parse_args().check
    torch.use_deterministic_algorithms(True)
    warnings.simplefilter("ignore")  # tensorly warns that one int rank is used for every mode
    OUT.mkdir(parents=True, exist_ok=True)
    failed = False
    for dtype_name, dtype in DTYPES.items():
        for complex_, stem in ((False, "galore"), (True, "galore_complex")):
            with torch.no_grad():
                data = arrays(dtype, complex_)
            failed |= not emit(f"{stem}_{dtype_name}", data, sidecar(dtype_name, complex_), check)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
