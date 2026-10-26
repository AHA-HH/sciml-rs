//! Phase 5 T1 (`docs/phase5/T1-transfer-floor.md`): the transfer-only error floor and the
//! choice of restriction R, with no model in the loop.
//!
//! Usage: `cargo run --release -p transfer-floor-spike -- <dataset dir>`, where the
//! directory holds the stage 3 and stage 4 test splits (`poisson_n129_K64_test`,
//! `poisson_n257_K64_test`). Prints markdown tables to stdout.

use std::f64::consts::PI;
use std::path::Path;
use std::time::Instant;

use burn::config::Config;
use ndarray::{Array1, Array2, Array3, ArrayView2, Axis, s};
use sciml_rs::neural_operators::chebyshev::{
    apply, cc_rel_l2_error, cheb_to_uniform, nodes, uniform_nodes, uniform_to_cheb,
};
use sciml_rs::neural_operators::data::catalogue::PoissonSet;
use sciml_rs::neural_operators::data::loaders::poisson::{
    PoissonField, SidecarRecord, load_poisson_cheb_field,
};

const STAGE3: PoissonSet = PoissonSet { n: 129, k: 64 };
const STAGE4: PoissonSet = PoissonSet { n: 257, k: 64 };
const COUNT: usize = 200;
const S: [usize; 4] = [16, 32, 64, 128];
const D: [usize; 3] = [2, 3, 4];
/// The rule (brief deliverable 6): the lowest mean at these s; candidates within
/// `RULE_TOL` × that mean go to the simpler R.
const RULE_S: [usize; 2] = [32, 64];
const RULE_TOL: f64 = 1.1;
/// The reference check stops the phase above this (label-check level is about 1e-12).
const REFERENCE_TOL: f64 = 1e-10;

/// T_k(x) = cos(k arccos x), x clamped to [−1, 1].
fn cheb_t(k: usize, x: f64) -> f64 {
    (k as f64 * x.clamp(-1.0, 1.0).acos()).cos()
}

/// E[i, k] = T_k(x_i), shape `[x.len(), n]`.
fn eval_matrix(x: &Array1<f64>, n: usize) -> Array2<f64> {
    Array2::from_shape_fn((x.len(), n), |(i, k)| cheb_t(k, x[i]))
}

/// The closed-form DCT-I `C: [n, n]`, nodal values on the ascending CGL nodes → Chebyshev
/// coefficients: a_k = (2/N)/c_k Σ''_j T_k(x_j) f_j, with T_k(x_j) = (−1)^k cos(πjk/N),
/// N = n − 1, c_0 = c_N = 2, and Σ'' halving j = 0 and j = N.
fn dct1(n: usize) -> Array2<f64> {
    let nn = (n - 1) as f64;
    let end = |i: usize| if i == 0 || i == n - 1 { 0.5 } else { 1.0 };
    Array2::from_shape_fn((n, n), |(k, j)| {
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        2.0 / nn * end(k) * end(j) * sign * (PI * (j * k) as f64 / nn).cos()
    })
}

/// E · diag(σ) · C.
fn filtered(e: &Array2<f64>, sigma: &Array1<f64>, c: &Array2<f64>) -> Array2<f64> {
    (e * &sigma.view().insert_axis(Axis(0))).dot(c)
}

/// A: [s, n], A[i, k] = mean of T_k over the cell [x_i − h/2, x_i + h/2] ∩ [−1, 1],
/// h = 2/(s − 1), by the closed-form antiderivative; A · C is the cell-average R.
fn cell_average(n: usize, s: usize) -> Array2<f64> {
    let x = uniform_nodes(s);
    let h = 2.0 / (s - 1) as f64;
    // ∫T_0 = x, ∫T_1 = x²/2, ∫T_k = T_{k+1}/(2(k+1)) − T_{k−1}/(2(k−1)).
    let anti = |k: usize, x: f64| match k {
        0 => x,
        1 => x * x / 2.0,
        _ => cheb_t(k + 1, x) / (2 * (k + 1)) as f64 - cheb_t(k - 1, x) / (2 * (k - 1)) as f64,
    };
    Array2::from_shape_fn((s, n), |(i, k)| {
        let (a, b) = ((x[i] - h / 2.0).max(-1.0), (x[i] + h / 2.0).min(1.0));
        (anti(k, b) - anti(k, a)) / (b - a)
    })
}

#[derive(Clone, Copy, PartialEq)]
enum Taper {
    Sharp,
    /// σ_k = exp(−36 (k/J)^8): σ_0 = 1, σ_J ≈ machine epsilon.
    Exp,
}

#[derive(Clone, Copy, PartialEq)]
enum Cutoff {
    SMinus2,
    /// ⌊(s − 2)π/2⌋, capped at n − 1.
    SPi,
    NMinus1,
}

/// A Chebyshev-coefficient filter σ with cut-off J: σ_k = 0 above J.
#[derive(Clone, Copy, PartialEq)]
struct Filter {
    taper: Taper,
    cutoff: Cutoff,
}

impl Filter {
    fn name(self) -> String {
        let taper = match self.taper {
            Taper::Sharp => "sharp",
            Taper::Exp => "exp",
        };
        let cut = match self.cutoff {
            Cutoff::SMinus2 => "J = s − 2",
            Cutoff::SPi => "J = ⌊(s − 2)π/2⌋",
            Cutoff::NMinus1 => "J = n − 1",
        };
        format!("filtered {taper}, {cut}")
    }

    /// σ: [n] for the grid sizes n (CGL) and s (uniform).
    fn sigma(self, n: usize, s: usize) -> Array1<f64> {
        let j = match self.cutoff {
            Cutoff::SMinus2 => s - 2,
            Cutoff::SPi => (((s - 2) as f64 * PI / 2.0).floor() as usize).min(n - 1),
            Cutoff::NMinus1 => n - 1,
        };
        Array1::from_shape_fn(n, |k| match (k <= j, self.taper) {
            (false, _) => 0.0,
            (true, Taper::Sharp) => 1.0,
            (true, Taper::Exp) => (-36.0 * (k as f64 / j as f64).powi(8)).exp(),
        })
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Restriction {
    Unfiltered,
    CellAverage,
    Filtered(Filter),
}

const fn filt(taper: Taper, cutoff: Cutoff) -> Restriction {
    Restriction::Filtered(Filter { taper, cutoff })
}

/// In rule order: the simpler R first (brief deliverable 6).
const CANDIDATES: [Restriction; 7] = [
    Restriction::Unfiltered,
    Restriction::CellAverage,
    filt(Taper::Sharp, Cutoff::SMinus2),
    filt(Taper::Sharp, Cutoff::SPi),
    filt(Taper::Exp, Cutoff::SMinus2),
    filt(Taper::Exp, Cutoff::SPi),
    filt(Taper::Exp, Cutoff::NMinus1),
];

impl Restriction {
    fn name(self) -> String {
        match self {
            Restriction::Unfiltered => "unfiltered T_cu".into(),
            Restriction::CellAverage => "cell average".into(),
            Restriction::Filtered(flt) => flt.name(),
        }
    }

    fn filter(self) -> Option<Filter> {
        match self {
            Restriction::Filtered(flt) => Some(flt),
            _ => None,
        }
    }

    /// The `[s, n]` restriction matrix.
    fn matrix(self, n: usize, s: usize, c: &Array2<f64>) -> Array2<f64> {
        match self {
            Restriction::Unfiltered => cheb_to_uniform(n, s),
            Restriction::CellAverage => cell_average(n, s).dot(c),
            Restriction::Filtered(flt) => {
                filtered(&eval_matrix(&uniform_nodes(s), n), &flt.sigma(n, s), c)
            }
        }
    }
}

/// One candidate R at one s: its build and apply times, and transfer-only error per d.
struct Built {
    r: Restriction,
    s: usize,
    build_s: f64,
    /// Applying R along both axes, per sample of f.
    apply_s: f64,
    /// (mean, max) of ‖P R u − u‖ / ‖u‖, indexed like `D`.
    transfer: Vec<(f64, f64)>,
}

fn at(built: &[Built], r: Restriction, s: usize) -> &Built {
    built
        .iter()
        .find(|b| b.r == r && b.s == s)
        .expect("every (R, s) is swept")
}

fn max_abs_diff(a: ArrayView2<f64>, b: ArrayView2<f64>) -> f64 {
    (&a - &b).iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

fn max_row_sum_err(r: &Array2<f64>) -> f64 {
    r.sum_axis(Axis(1))
        .iter()
        .fold(0.0_f64, |m, v| m.max((v - 1.0).abs()))
}

/// (mean, max) of `stat` over the samples of `fields: [samples, n, n]`.
fn mean_max(fields: &Array3<f64>, stat: impl Fn(ArrayView2<f64>) -> f64) -> (f64, f64) {
    let v: Vec<f64> = fields.outer_iter().map(stat).collect();
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    (mean, v.iter().copied().fold(0.0_f64, f64::max))
}

/// ‖f − F f‖² / ‖f‖² over `f`, with F = E diag(σ) C on the CGL grid (`e_cgl` = E).
fn filtered_out_energy(
    flt: Filter,
    s: usize,
    e_cgl: &Array2<f64>,
    c: &Array2<f64>,
    f: &Array3<f64>,
) -> (f64, f64) {
    let filter = filtered(e_cgl, &flt.sigma(e_cgl.nrows(), s), c);
    mean_max(f, |fi| {
        cc_rel_l2_error(apply(filter.view(), filter.view(), fi).view(), fi).powi(2)
    })
}

fn mean_max_str((mean, max): (f64, f64)) -> String {
    format!("{mean:.2e} / {max:.2e}")
}

fn header(first: &str) {
    let cols: Vec<String> = S.iter().map(|s| format!("s = {s}")).collect();
    println!("| {first} | {} |", cols.join(" | "));
    println!("| --- |{}", " --- |".repeat(S.len()));
}

/// One table row: `label`, then `cell(s)` for every s in `S`.
fn row(label: impl std::fmt::Display, cell: impl Fn(usize) -> String) {
    let cells: Vec<String> = S.iter().map(|&s| cell(s)).collect();
    println!("| {label} | {} |", cells.join(" | "));
}

fn load(dir: &Path, set: PoissonSet, field: PoissonField) -> Array3<f64> {
    load_poisson_cheb_field(dir.join(set.npz("test")), field, set.n, COUNT)
        .unwrap_or_else(|e| panic!("loading {} {field}: {e}", set.stem("test")))
}

fn sidecar(dir: &Path, set: PoissonSet, split: &str) -> SidecarRecord {
    let path = dir.join(format!("{}.json", set.stem(split)));
    SidecarRecord::load(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
}

/// Stage 4 at the even-indexed 257 nodes against stage 3 (CONVENTIONS §12), for u and f.
/// Prints the table and returns whether both sit below `REFERENCE_TOL`.
fn reference_check(dir: &Path, u: &Array3<f64>, f: &Array3<f64>) -> bool {
    let (s3, s4) = (sidecar(dir, STAGE3, "test"), sidecar(dir, STAGE4, "test"));
    assert_eq!(s3.base_seed, s4.base_seed, "stage 3 and 4 must share seeds");
    println!(
        "## Reference check (stage 4 at the even 257 nodes vs stage 3, {COUNT} test samples)\n"
    );
    println!("| Field | max relative CC-L² |\n| --- | --- |");
    let mut ok = true;
    for (field, coarse) in [(PoissonField::U, u), (PoissonField::F, f)] {
        let fine = load(dir, STAGE4, field);
        let even = fine.slice(s![.., ..;2, ..;2]);
        let err = coarse
            .outer_iter()
            .zip(even.outer_iter())
            .map(|(c, e)| cc_rel_l2_error(e, c))
            .fold(0.0_f64, f64::max);
        ok &= err <= REFERENCE_TOL;
        println!("| {field} | {err:.2e} |");
    }
    ok
}

/// Builds every R at every s (with the σ ≡ 1 and row-sum asserts, brief Tests) and
/// measures its apply time on `f` and its transfer-only error on `u` for every d.
fn sweep(n: usize, c: &Array2<f64>, u: &Array3<f64>, f: &Array3<f64>) -> Vec<Built> {
    let mut built = Vec::new();
    for s in S {
        // A sharp filter at J = n − 1 is σ ≡ 1.
        let identity = filt(Taper::Sharp, Cutoff::NMinus1).matrix(n, s, c);
        let diff = max_abs_diff(identity.view(), cheb_to_uniform(n, s).view());
        assert!(
            diff < 1e-12,
            "σ ≡ 1 differs from T_cu by {diff:.2e} at s = {s}"
        );
        let ps = D.map(|d| uniform_to_cheb(s, n, d));
        for r in CANDIDATES {
            let t = Instant::now();
            let m = r.matrix(n, s, c);
            let build_s = t.elapsed().as_secs_f64();
            let err = max_row_sum_err(&m);
            assert!(
                err < 1e-12,
                "{}: row sums off by {err:.2e} at s = {s}",
                r.name()
            );
            let t = Instant::now();
            for fi in f.outer_iter() {
                std::hint::black_box(apply(m.view(), m.view(), fi));
            }
            let apply_s = t.elapsed().as_secs_f64() / f.len_of(Axis(0)) as f64;
            // M = P R, applied along both axes.
            let transfer = ps
                .iter()
                .map(|p| {
                    let pr = p.dot(&m);
                    mean_max(u, |ui| {
                        cc_rel_l2_error(apply(pr.view(), pr.view(), ui).view(), ui)
                    })
                })
                .collect();
            built.push(Built {
                r,
                s,
                build_s,
                apply_s,
                transfer,
            });
        }
    }
    built
}

fn main() {
    let dir = std::env::args()
        .nth(1)
        .expect("usage: transfer-floor-spike <dataset dir>");
    let dir = Path::new(&dir);
    let start = Instant::now();
    let n = STAGE3.n;
    let u = load(dir, STAGE3, PoissonField::U);
    let f = load(dir, STAGE3, PoissonField::F);

    if !reference_check(dir, &u, &f) {
        println!("\n**Reference check failed (above {REFERENCE_TOL:.0e}): the phase stops.**");
        std::process::exit(1);
    }

    let c = dct1(n);
    let built = sweep(n, &c, &u, &f);
    println!("\nAsserts passed: σ ≡ 1 reproduces T_cu, and every R has unit row sums (1e-12).");

    for (di, d) in D.iter().enumerate() {
        println!("\n## Transfer-only error ‖P R u − u‖ / ‖u‖, P = T_uc d = {d} (mean / max)\n");
        header("R");
        for r in CANDIDATES {
            row(r.name(), |s| mean_max_str(at(&built, r, s).transfer[di]));
        }
    }

    println!("\n## P's own error: P applied to u sampled by unfiltered T_cu (mean / max)\n");
    header("d");
    for (di, d) in D.iter().enumerate() {
        row(d, |s| {
            mean_max_str(at(&built, Restriction::Unfiltered, s).transfer[di])
        });
    }

    println!("\n## Filtered-out energy of f, ‖f − F f‖² / ‖f‖² on the CGL grid (mean / max)\n");
    header("Filter");
    let e_cgl = eval_matrix(&nodes(n), n);
    for flt in CANDIDATES.iter().filter_map(|r| r.filter()) {
        row(flt.name(), |s| {
            mean_max_str(filtered_out_energy(flt, s, &e_cgl, &c, &f))
        });
    }

    println!(
        "\n## Aliasing visible in f\n\nSkipped: the exact sine series needs `pde::poisson::grf`, behind `datagen`."
    );

    println!("\n## Cost of R at n = {n}: build / apply per sample (µs)\n");
    header("R");
    for r in CANDIDATES {
        row(r.name(), |s| {
            let b = at(&built, r, s);
            format!("{:.1} / {:.0}", b.build_s * 1e6, b.apply_s * 1e6)
        });
    }

    println!("\n## Dataset generation time (sidecar `wall_time_s`)\n");
    println!("| Set | train (s) | test (s) |\n| --- | --- | --- |");
    for set in [STAGE3, STAGE4] {
        let (tr, te) = (sidecar(dir, set, "train"), sidecar(dir, set, "test"));
        println!(
            "| n = {}, K = {} | {:.1} | {:.1} |",
            set.n, set.k, tr.wall_time_s, te.wall_time_s
        );
    }

    println!("\n## Rule: lowest mean transfer-only error, within 10% → simpler R\n");
    println!("| d | s | lowest mean | chosen R | its mean |\n| --- | --- | --- | --- | --- |");
    for (di, d) in D.iter().enumerate() {
        for s in RULE_S {
            let means = CANDIDATES.map(|r| at(&built, r, s).transfer[di].0);
            let best = means.iter().copied().fold(f64::INFINITY, f64::min);
            let chosen = means
                .iter()
                .position(|&m| m <= RULE_TOL * best)
                .expect("best is in range");
            println!(
                "| {d} | {s} | {best:.3e} | {} | {:.3e} |",
                CANDIDATES[chosen].name(),
                means[chosen]
            );
        }
    }

    eprintln!("wall time: {:.1} s", start.elapsed().as_secs_f64());
}
