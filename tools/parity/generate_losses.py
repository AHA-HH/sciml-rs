"""Write loss and finite-difference parity fixtures from neuraloperator 2.0.0 (library CM1.6a, b; CM2.5).

Each fixture is `tests/fixtures/parity/losses<D>d_<dtype>.npz` with a `.json` sidecar: a
seeded pair `x`, `y` (2, 2, *grid), a seeded vector field `v` (2, D, *grid), and 2.0.0's
`FiniteDiff`, `H1Loss`, `HdivLoss`, `PointwiseQuantileLoss`, `MSELoss` and `LpLoss.rel` on
them. `meta_losses_<dtype>.npz` (CM1.6b): the aggregators `WeightedSumLoss`,
`FieldwiseAggregatorLoss`, `SoftAdapt` and `Relobralo` over seeded losses.
`equation_losses_<dtype>.npz` (CM2.5): `BurgersEqnLoss`, `ICLoss` and the Poisson losses.
See README.md.

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
from neuralop.losses.equation_losses import (
    BurgersEqnLoss,
    ICLoss,
    PoissonBoundaryLoss,
    PoissonEqnLoss,
    PoissonInteriorLoss,
)
from neuralop.losses.meta_losses import FieldwiseAggregatorLoss, Relobralo, SoftAdapt, WeightedSumLoss

from generate import DTYPES, MAX_BYTES, OUT, same_as_committed

INPUT_SEED = 2
GRIDS = {1: (32,), 2: (16, 12), 3: (8, 6, 5)}  # non-square, so a swapped axis fails parity
H = (0.1, 0.2, 0.3)  # FiniteDiff steps, per axis
MEASURE = (1.0, 2.0, 0.5)  # H1Loss / HdivLoss domain measure, per axis
ALPHA = 0.1  # PointwiseQuantileLoss
AXES = "xyz"
META_SEED = 3  # the aggregators' inputs
RHO_SEED = 4  # torch's global RNG for Relobralo's Bernoulli draws
STEPS = 6  # adaptive aggregators: steps of 3 losses
LOSS_KEYS = ("a", "b", "c")
EQN_SEED = 5  # the equation losses' inputs
BURGERS_GRID = (10, 12)  # (n_t, n_x), non-square, so a swapped axis fails parity
BURGERS = {"default": {}, "visc": {"visc": 0.05, "domain_length": [2.0, 3.0]}}
POISSON_POINTS = (6, 20)  # boundary, interior
POISSON_WEIGHTS = {"boundary_weight": 2.0, "interior_weight": 0.5}
# name -> (class, kwargs); weights keyed as 2.0.0's dict, missing keys meaning 1.
ADAPTIVE = {
    "softadapt": (SoftAdapt, {}),
    "softadapt_w": (SoftAdapt, {"weights": {"b": 2.0}}),
    "relobralo": (Relobralo, {}),
    "relobralo_w": (Relobralo, {"alpha": 0.9, "beta": 0.5, "tau": 2.0, "weights": {"c": 0.5}}),
}


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
    return finish(out, dtype)


def adaptive_run(cls, kwargs, table, dtype):
    """Every step of one adaptive aggregator: the aggregate, its weights, d aggregate / d loss,
    and (Relobralo) the Bernoulli draws. Built under torch's default dtype `dtype`, so 2.0.0's
    buffers have the losses' precision."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    # A copy of the weights dict: 2.0.0's weigh_losses adds the missing keys to it.
    kwargs = {k: (dict(v) if isinstance(v, dict) else v) for k, v in kwargs.items()}
    agg = cls([torch.zeros(1)], len(LOSS_KEYS), **kwargs)
    draws = []
    bernoulli = torch.bernoulli

    def recorded(*args, **kw):
        r = bernoulli(*args, **kw)
        draws.append(r.item())
        return r

    torch.manual_seed(RHO_SEED)
    torch.bernoulli = recorded  # meta_losses looks it up at call time
    out, weights, grads = [], [], []
    try:
        for step in range(STEPS):
            leaves = [table[step, i].clone().requires_grad_(True) for i in range(len(LOSS_KEYS))]
            loss, lmbda = agg(dict(zip(LOSS_KEYS, leaves)), step)
            loss.backward()
            out.append(loss.detach().clone())
            weights.append(lmbda.detach().clone())
            grads.append(torch.stack([leaf.grad for leaf in leaves]))
    finally:
        torch.bernoulli = bernoulli
        torch.set_default_dtype(previous)
    return torch.stack(out), torch.stack(weights), torch.stack(grads), torch.tensor(draws, dtype=dtype)


def meta_arrays(dtype):
    g = torch.Generator().manual_seed(META_SEED)
    x = torch.randn((2, 2, 32), generator=g)
    y = torch.randn((2, 2, 32), generator=g)
    pred = torch.randn((16, 3), generator=g)
    truth = torch.randn((16, 3), generator=g)
    table = 0.2 + 1.8 * torch.rand((STEPS, len(LOSS_KEYS)), generator=g)
    x, y, pred, truth, table = (t.to(dtype) for t in (x, y, pred, truth, table))
    out = {"x": x, "y": y, "pred": pred, "truth": truth, "table": table}
    with torch.no_grad():
        pair = [H1Loss(d=1), MSELoss()]
        out["wsum.default"] = WeightedSumLoss(pair)(x, y)
        out["wsum.weighted"] = WeightedSumLoss(pair, weights=[0.3, 0.7])(x, y)
        fieldwise = FieldwiseAggregatorLoss(
            {"u": MSELoss(), "p": PointwiseQuantileLoss(alpha=ALPHA)},
            {"u": (slice(None), slice(0, 1)), "p": (slice(None), slice(2, 3))},
        )
        out["fieldwise"] = fieldwise(pred, truth)
    for name, (cls, kwargs) in ADAPTIVE.items():
        loss, weights, grads, rho = adaptive_run(cls, kwargs, table, dtype)
        out[f"{name}.loss"], out[f"{name}.weights"], out[f"{name}.grad"] = loss, weights, grads
        if cls is Relobralo:
            out[f"{name}.rho"] = rho
    return finish(out, dtype)


def poisson_u(q):
    """The analytic u of the Poisson fixture at query points q (1, n, 2): sin(πx) cos(2y) + x²y,
    as (1, n, 1)."""
    x, y = q[..., 0], q[..., 1]
    return (torch.sin(torch.pi * x) * torch.cos(2 * y) + x * x * y).unsqueeze(-1)


def equation_arrays(dtype):
    """2.0.0's equation losses on seeded inputs. The Poisson ones differentiate u by autograd
    with respect to the query points, so this runs with gradients on."""
    g = torch.Generator().manual_seed(EQN_SEED)
    u = torch.randn((2, 1) + BURGERS_GRID, generator=g)
    x = torch.randn((2, 1) + BURGERS_GRID, generator=g)
    nb, ni = POISSON_POINTS
    q = torch.rand((1, nb + ni, 2), generator=g)
    f = torch.randn((ni,), generator=g)
    y = torch.randn((1, nb + ni, 1), generator=g)
    u, x, q, f, y = (t.to(dtype) for t in (u, x, q, f, y))
    out = {"u": u, "x": x, "poisson.q": q, "poisson.f": f, "poisson.y": y}
    with torch.no_grad():
        for name, kwargs in BURGERS.items():
            out[f"burgers.{name}"] = BurgersEqnLoss(**kwargs)(u)
        out["ic"] = ICLoss()(u, x)

    q.requires_grad_(True)
    up = poisson_u(q)
    num_boundary = torch.tensor(nb)
    kw = {"output_queries": q, "output_source_terms_domain": f, "num_boundary": num_boundary}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # 2.0.0 warns on its own keyword arguments
        out["poisson.interior"] = PoissonInteriorLoss()(up, **kw)
        out["poisson.boundary"] = PoissonBoundaryLoss()(
            up, num_boundary=num_boundary, out_sub_level=1.0, y=y, output_queries=q
        )
        out["poisson.eqn"] = PoissonEqnLoss(**POISSON_WEIGHTS)(up, y, out_sub_level=1.0, **kw)
    # u at every point, and the derivatives 2.0.0 takes inside at the interior points, for the
    # Rust side.
    du = torch.autograd.grad(up.sum(), q, create_graph=True)[0]
    u_xx = torch.autograd.grad(du[..., 0].sum(), q, retain_graph=True)[0][..., 0]
    u_yy = torch.autograd.grad(du[..., 1].sum(), q)[0][..., 1]
    interior = slice(nb, None)
    out["poisson.u"] = up[0, :, 0]
    out["poisson.u_x"] = du[0, interior, 0]
    out["poisson.u_y"] = du[0, interior, 1]
    out["poisson.lap"] = (u_xx + u_yy)[0, interior]
    return finish(out, dtype)


def equation_sidecar(dtype_name):
    return {
        "config": "equation_losses",
        "dtype": dtype_name,
        "input_seed": EQN_SEED,
        "burgers": {k: f"BurgersEqnLoss(**{kw})" for k, kw in BURGERS.items()},
        "burgers_grid": list(BURGERS_GRID),
        "ic": "ICLoss()(u, x)",
        "poisson_points": {"boundary": POISSON_POINTS[0], "interior": POISSON_POINTS[1]},
        "poisson_u": "sin(pi x) cos(2 y) + x^2 y at the query points q, uniform in [0, 1)^2",
        "poisson_eqn": f"PoissonEqnLoss(**{POISSON_WEIGHTS}), out_sub_level 1",
        "keys": "u, x (2, 1, n_t, n_x); burgers.{default,visc}; ic; poisson.{q,f,y}; "
        "poisson.{interior,boundary,eqn}; poisson.u at every point; poisson.{u_x,u_y,lap} at the "
        "interior points (autograd)",
        "inputs": "drawn in f32, then cast",
        **provenance(),
    }


def finish(out, dtype):
    # Scalars as shape (1,): the Rust losses return [1].
    data = {k: np.atleast_1d(t.detach().numpy()) for k, t in out.items()}
    want = np.dtype(np.float32 if dtype == torch.float32 else np.float64)
    bad = {k: a.dtype for k, a in data.items() if a.dtype != want}
    assert not bad, f"arrays not in {want} (NpzFileReader reads f32/f64 only): {bad}"
    return data


def provenance():
    """The sidecars' `versions` and `platform` entries."""
    return {
        "versions": {
            "python": platform.python_version(),
            "neuraloperator": neuralop.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "platform": f"{platform.system()} {platform.machine()}",
    }


def meta_sidecar(dtype_name):
    return {
        "config": "meta_losses",
        "dtype": dtype_name,
        "input_seed": META_SEED,
        "rho_seed": RHO_SEED,
        "steps": STEPS,
        "loss_keys": list(LOSS_KEYS),
        "adaptive": {k: {"class": c.__name__, "kwargs": kw} for k, (c, kw) in ADAPTIVE.items()},
        "weighted_sum": "WeightedSumLoss([H1Loss(d=1), MSELoss()]), weights None and [0.3, 0.7], on x, y (2, 2, 32)",
        "fieldwise": f"u: MSELoss on [:, 0:1], p: PointwiseQuantileLoss(alpha={ALPHA}) on [:, 2:3], on pred, truth (16, 3)",
        "keys": "x, y, pred, truth, table (steps, 3); wsum.{default,weighted}; fieldwise; "
        "<adaptive>.{loss,weights,grad}; relobralo*.rho (the draws of steps 1..)",
        "default_dtype": "torch's default dtype set to the fixture's while the aggregators run",
        **provenance(),
    }


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
        **provenance(),
    }


def emit(name, data, meta, check):
    """Write fixture `name` and its sidecar, or with `check` compare them with the committed
    ones; whether it succeeded."""
    path = OUT / f"{name}.npz"
    if check:
        return same_as_committed(path, data, meta)
    np.savez(path, **data)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    size = path.stat().st_size
    assert size <= MAX_BYTES, f"{path.name}: {size} bytes"
    print(f"{path.name}: {len(data)} arrays, {size} bytes")
    return True


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
            failed |= not emit(f"losses{d}d_{dtype_name}", data, sidecar(d, dtype_name), check)
    for dtype_name, dtype in DTYPES.items():
        data = meta_arrays(dtype)
        failed |= not emit(f"meta_losses_{dtype_name}", data, meta_sidecar(dtype_name), check)
    for dtype_name, dtype in DTYPES.items():
        data = equation_arrays(dtype)
        failed |= not emit(f"equation_losses_{dtype_name}", data, equation_sidecar(dtype_name), check)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
