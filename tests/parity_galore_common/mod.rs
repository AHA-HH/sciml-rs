//! The Tensor GaLore parity checks (library CM3.6a), shared by the f32 and f64 binaries: each
//! case of `tools/parity/generate_galore.py` against 2.0.0's `TensorGaLoreProjector`.

use burn::prelude::*;
use sciml_rs::neural_operators::optim::galore::{GaLoreRank, TensorGaLoreConfig};

use crate::parity_common::{Fixture, assert_parity};

/// The cases' configurations, as `generate_galore.py`'s `CASES`.
pub fn config(case: &str) -> TensorGaLoreConfig {
    let (rank, gap, scale) = match case {
        "matrix" => (GaLoreRank::Uniform(3), 5, 1.0),
        "tensor4" => (GaLoreRank::PerMode(vec![3, 2, 3, 2]), 200, 0.25),
        "fraction" => (GaLoreRank::Fraction(0.5), 200, 1.0),
        _ => panic!("no case {case}"),
    };
    TensorGaLoreConfig {
        rank,
        update_proj_gap: gap,
        scale,
    }
}

/// Case `case` of `fixture` (rank-`D` gradients): the validated ranks, each factor,
/// `project(grad, 0)`, `project_back` of it and `project(grad2, gap)` within D3; prints the
/// largest error.
pub fn run<const D: usize>(fixture: &str, case: &str, device: &Device) {
    let mut fx = Fixture::open(fixture);
    let tol = fx.precision.d3();
    let key = |k: &str| format!("{case}.{k}");
    let config = config(case);
    let grad: Tensor<D> = fx.tensor(&key("grad"), device);

    let ranks: Vec<usize> = fx
        .array(&key("ranks"))
        .iter()
        .map(|&r| r as usize)
        .collect();
    assert_eq!(
        config.rank.ranks(grad.dims()).as_slice(),
        ranks,
        "{case}: ranks"
    );

    let mut projector = config.init::<D>();
    let low = projector.project(grad, 0);
    let mut worst = 0.0_f64;
    for (k, f) in projector
        .factors()
        .expect("factors after project")
        .iter()
        .enumerate()
    {
        let what = format!("{fixture} {case} factor{k}");
        worst = worst.max(assert_parity(
            f.clone(),
            &fx.array(&key(&format!("factor{k}"))),
            tol,
            &what,
        ));
    }
    let back = projector.project_back(low.clone());
    worst = worst.max(assert_parity(low, &fx.array(&key("low")), tol, "low"));
    worst = worst.max(assert_parity(back, &fx.array(&key("back")), tol, "back"));
    let grad2: Tensor<D> = fx.tensor(&key("grad2"), device);
    let low2 = projector.project(grad2, config.update_proj_gap);
    worst = worst.max(assert_parity(low2, &fx.array(&key("low2")), tol, "low2"));
    println!("{fixture} {case}: relative max error {worst:.3e}");
}
