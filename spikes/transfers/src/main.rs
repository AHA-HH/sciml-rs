//! Runs the T3 measurements and prints markdown tables (pasted into REPORT.md).

use ndarray::Array2;
use std::time::Instant;
use transfers_spike::{grf, grids, interp};

const NS: [usize; 4] = [33, 65, 129, 257];
const D_MAX: usize = 8;
const N_SAMPLES: u64 = 20;
const SEED0: u64 = 1000;
const MS_SAMPLES: u64 = 1000;
const MS_SEED0: u64 = 2000;
const ROUND_TRIP_TOL: f64 = 1e-4;
const LEBESGUE_TOL: f64 = 10.0;
const TAIL_TOL: f64 = 5e-3;
const SET_NAMES: [&str; 6] = [
    "sin(πx)sin(πy)",
    "(1−x²)(1−y²)e^(x+2y)",
    "GRF K(n)",
    "GRF K=64",
    "GRF solution u, K(n)",
    "GRF solution u, K=64",
];
/// Sets before this index are single analytic fields.
const FIRST_GRF: usize = 2;
/// GRF forcing f at K(n): the brief's literal decision set.
const GRF_F: usize = 2;
/// Exact Poisson solution u at K(n): design §7's transfer error, used for the decisions.
const GRF_U: usize = 4;

#[derive(Clone, Copy, Default)]
struct Stats {
    mean: f64,
    max: f64,
}

impl Stats {
    fn of(v: &[f64]) -> Self {
        // f64::max ignores NaN, so a NaN error would otherwise pass the decision rule.
        assert!(v.iter().all(|e| e.is_finite()), "non-finite error: {v:?}");
        Stats {
            mean: v.iter().sum::<f64>() / v.len() as f64,
            max: v.iter().cloned().fold(0.0, f64::max),
        }
    }
    fn cell(&self, single: bool) -> String {
        if single {
            format!("{:.1e}", self.max)
        } else {
            format!("{:.1e} / {:.1e}", self.mean, self.max)
        }
    }
}

struct Row {
    n: usize,
    s: usize,
    /// [set] T_cu error.
    cu: Vec<Stats>,
    /// [set][d] T_uc error.
    uc: Vec<Vec<Stats>>,
    /// [set][d] round-trip error.
    rt: Vec<Vec<Stats>>,
    /// [d] 1D Lebesgue constant of T_uc.
    leb: Vec<f64>,
    /// 1D Lebesgue constant of T_cu.
    leb_cu: f64,
}

fn k_of_n(n: usize) -> usize {
    ((n - 1) / 2).min(64)
}

fn measure(n: usize, s: usize, xis: &[Array2<f64>]) -> Row {
    let xc = grids::cheb_nodes(n);
    let xu = grids::uniform_nodes(s);
    let wc = grids::clenshaw_curtis(n);
    let wu = grids::trapezoid(s);
    let tcu = interp::t_cu(&xc, &xu);
    let tuc: Vec<_> = (0..=D_MAX).map(|d| interp::t_uc(&xu, &xc, d)).collect();

    let pair =
        |f: &dyn Fn(f64, f64) -> f64| (grids::sample(&xc, &xc, f), grids::sample(&xu, &xu, f));
    let grf_pair = |xi: &Array2<f64>, k| (grf::eval(xi, k, &xc, &xc), grf::eval(xi, k, &xu, &xu));
    let sol_pair = |xi: &Array2<f64>, k| {
        (
            grf::eval_solution(xi, k, &xc, &xc),
            grf::eval_solution(xi, k, &xu, &xu),
        )
    };
    let pi = std::f64::consts::PI;
    let sets: Vec<Vec<(Array2<f64>, Array2<f64>)>> = vec![
        vec![pair(&|x, y| (pi * x).sin() * (pi * y).sin())],
        vec![pair(&|x, y| {
            (1.0 - x * x) * (1.0 - y * y) * (x + 2.0 * y).exp()
        })],
        xis.iter().map(|xi| grf_pair(xi, k_of_n(n))).collect(),
        xis.iter().map(|xi| grf_pair(xi, 64)).collect(),
        xis.iter().map(|xi| sol_pair(xi, k_of_n(n))).collect(),
        xis.iter().map(|xi| sol_pair(xi, 64)).collect(),
    ];

    let mut cu = Vec::new();
    let mut uc = Vec::new();
    let mut rt = Vec::new();
    for set in &sets {
        let e: Vec<f64> = set
            .iter()
            .map(|(c, u)| grids::rel_l2(&interp::apply2d(&tcu, c), u, &wu, &wu))
            .collect();
        cu.push(Stats::of(&e));
        let on_uniform: Vec<Array2<f64>> =
            set.iter().map(|(c, _)| interp::apply2d(&tcu, c)).collect();
        let mut uc_d = Vec::new();
        let mut rt_d = Vec::new();
        for t in &tuc {
            let e: Vec<f64> = set
                .iter()
                .map(|(c, u)| grids::rel_l2(&interp::apply2d(t, u), c, &wc, &wc))
                .collect();
            uc_d.push(Stats::of(&e));
            let e: Vec<f64> = set
                .iter()
                .zip(&on_uniform)
                .map(|((c, _), v)| grids::rel_l2(&interp::apply2d(t, v), c, &wc, &wc))
                .collect();
            rt_d.push(Stats::of(&e));
        }
        uc.push(uc_d);
        rt.push(rt_d);
    }
    Row {
        n,
        s,
        cu,
        uc,
        rt,
        leb: tuc.iter().map(interp::lebesgue).collect(),
        leb_cu: interp::lebesgue(&tcu),
    }
}

fn d_header() -> String {
    let mut h = String::from("| n | s |");
    let mut sep = String::from("| --- | --- |");
    for d in 0..=D_MAX {
        h += &format!(" d={d} |");
        sep += " --- |";
    }
    format!("{h}\n{sep}")
}

fn print_d_table(title: &str, rows: &[Row], get: impl Fn(&Row, usize) -> String) {
    println!("\n#### {title}\n\n{}", d_header());
    for r in rows {
        let cells: Vec<String> = (0..=D_MAX).map(|d| get(r, d)).collect();
        println!("| {} | {} | {} |", r.n, r.s, cells.join(" | "));
    }
}

/// Smallest d meeting both rules at n = 65, 129, 257 on field set `set` for the given s.
fn choose_d(rows: &[Row], set: usize, s_is_n: bool) -> Option<usize> {
    let sel: Vec<&Row> = rows
        .iter()
        .filter(|r| r.n >= 65 && (r.s == r.n) == s_is_n)
        .collect();
    (0..=D_MAX).find(|&d| {
        sel.iter()
            .all(|r| r.rt[set][d].max <= ROUND_TRIP_TOL && r.leb[d] < LEBESGUE_TOL)
    })
}

fn median_ms(mut f: impl FnMut(), reps: usize) -> f64 {
    let mut t: Vec<f64> = (0..reps)
        .map(|_| {
            let t0 = Instant::now();
            f();
            t0.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    t.sort_by(f64::total_cmp);
    t[reps / 2]
}

fn main() {
    let xis: Vec<Array2<f64>> = (0..N_SAMPLES).map(|i| grf::draw_xi(SEED0 + i)).collect();

    let mut rows = Vec::new();
    for n in NS {
        for s in [n - 1, n] {
            eprintln!("measuring n={n} s={s}");
            rows.push(measure(n, s, &xis));
        }
    }

    println!("Cells: single field → its error; GRF sets → mean / max over {N_SAMPLES} samples.");
    println!(
        "GRF K(n) = min((n−1)/2, 64); seeds {SEED0}..{}.",
        SEED0 + N_SAMPLES - 1
    );

    println!("\n### 1. T_cu: Chebyshev → uniform (trapezoidal relative L²)\n");
    let mut h = String::from("| n | s |");
    let mut sep = String::from("| --- | --- |");
    for name in SET_NAMES {
        h += &format!(" {name} |");
        sep += " --- |";
    }
    println!("{h} Λ(T_cu) |\n{sep} --- |");
    for r in &rows {
        let cells: Vec<String> =
            r.cu.iter()
                .enumerate()
                .map(|(i, st)| st.cell(i < FIRST_GRF))
                .collect();
        println!(
            "| {} | {} | {} | {:.2} |",
            r.n,
            r.s,
            cells.join(" | "),
            r.leb_cu
        );
    }

    println!("\n### 2. T_uc: uniform → Chebyshev, Floater–Hormann degree d (CC relative L²)");
    for (i, name) in SET_NAMES.iter().enumerate() {
        print_d_table(name, &rows, |r, d| r.uc[i][d].cell(i < FIRST_GRF));
    }

    println!("\n### 3. Round trip T_uc(T_cu u) − u on the Chebyshev grid (CC relative L²)");
    for (i, name) in SET_NAMES.iter().enumerate() {
        print_d_table(name, &rows, |r, d| r.rt[i][d].cell(i < FIRST_GRF));
    }

    println!("\n### 4. Lebesgue constant of T_uc (1D, max row sum; 2D is its square)");
    print_d_table("Λ(T_uc)", &rows, |r, d| format!("{:.2}", r.leb[d]));

    // 5. GRF tail and normalisation.
    eprintln!("GRF tail and mean square");
    let s_inf = grf::s_inf(3000);
    let table = [(16, 1.9e-2), (32, 4.9e-3), (64, 1.3e-3), (128, 3.2e-4)];
    println!("\n### 5. GRF unresolved energy and normalisation\n");
    println!("S_∞ = {s_inf:.6e} (k, l ≤ 3000 plus integral tail).\n");
    let ms_xis: Vec<Array2<f64>> = (0..MS_SAMPLES)
        .map(|i| grf::draw_xi(MS_SEED0 + i))
        .collect();
    let x257 = grids::cheb_nodes(257);
    let w257 = grids::clenshaw_curtis(257);
    println!(
        "| K | ε_K measured | ε_K design §3.3 | mean square, Parseval (mean ± s.e., {MS_SAMPLES} samples) | mean square, CC at n = 257 |"
    );
    println!("| --- | --- | --- | --- | --- |");
    let mut eps32 = f64::NAN;
    for (k, design) in table {
        let eps = grf::eps_k(k, s_inf);
        if k == 32 {
            eps32 = eps;
        }
        let ms: Vec<f64> = ms_xis
            .iter()
            .map(|xi| grf::mean_square_parseval(xi, k))
            .collect();
        let mean = ms.iter().sum::<f64>() / ms.len() as f64;
        let var = ms.iter().map(|m| (m - mean).powi(2)).sum::<f64>() / (ms.len() - 1) as f64;
        let se = (var / ms.len() as f64).sqrt();
        let cc = if k <= 64 {
            let m: f64 = ms_xis
                .iter()
                .map(|xi| grids::wnorm(&grf::eval(xi, k, &x257, &x257), &w257, &w257).powi(2) / 4.0)
                .sum::<f64>()
                / ms_xis.len() as f64;
            format!("{m:.4}")
        } else {
            "–".into()
        };
        println!("| {k} | {eps:.2e} | {design:.1e} | {mean:.4} ± {se:.4} | {cc} |");
    }

    // Decisions.
    println!("\n### Decisions (computed)\n");
    println!(
        "Rule: max over samples ≤ {ROUND_TRIP_TOL:e} and Λ < {LEBESGUE_TOL} at n = 65, 129, 257.\n"
    );
    let mut d_eval = 3;
    for (set, label) in [(GRF_U, "solution u (design §7)"), (GRF_F, "forcing f")] {
        let d_nm1 = choose_d(&rows, set, false);
        let d_n = choose_d(&rows, set, true);
        println!("- GRF {label}, K(n): d for s = n − 1: {d_nm1:?}; d for s = n: {d_n:?}.");
        if set == GRF_U {
            d_eval = d_nm1.unwrap_or_else(|| {
                println!("- No d meets the rule on u; the comparisons below use d = 3.");
                3
            });
        }
    }
    for (set, label) in [(GRF_U, "u"), (GRF_F, "f")] {
        let mut all_10x = true;
        for n in [65, 129, 257] {
            let a = rows.iter().find(|r| r.n == n && r.s == n - 1).unwrap();
            let b = rows.iter().find(|r| r.n == n && r.s == n).unwrap();
            let ratio_mean = a.rt[set][d_eval].mean / b.rt[set][d_eval].mean;
            let ratio_max = a.rt[set][d_eval].max / b.rt[set][d_eval].max;
            all_10x &= ratio_mean >= 10.0 && ratio_max >= 10.0;
            println!(
                "- {label}, n = {n}, d = {d_eval}: round-trip(s = n − 1) / round-trip(s = n) = {ratio_mean:.2} (mean), {ratio_max:.2} (max)."
            );
        }
        println!("- {label}: s = n gives ≥ 10× at every n: {all_10x}.");
    }
    println!(
        "- ε_32 = {eps32:.2e} ≤ {TAIL_TOL:e}: {}.",
        eps32 <= TAIL_TOL
    );

    // Diagnostic: 1D round trip of a single sine mode, by points per wavelength.
    println!("\n### Diagnostic: 1D round trip of sin(kπ(x+1)/2), s = n − 1 (CC relative L²)\n");
    println!("{}", d_header().replacen("| s |", "| k (ppw) |", 1));
    for n in [65, 129, 257] {
        let s = n - 1;
        let xc = grids::cheb_nodes(n);
        let xu = grids::uniform_nodes(s);
        let wc = grids::clenshaw_curtis(n);
        let tcu = interp::t_cu(&xc, &xu);
        let kn = k_of_n(n);
        for k in [kn / 4, kn / 2, kn] {
            let pi = std::f64::consts::PI;
            let u = xc.mapv(|x| (k as f64 * pi * (x + 1.0) / 2.0).sin());
            let v = tcu.dot(&u);
            let cells: Vec<String> = (0..=D_MAX)
                .map(|d| {
                    let e = interp::t_uc(&xu, &xc, d).dot(&v) - &u;
                    let num: f64 = e.iter().zip(&wc).map(|(e, w)| w * e * e).sum();
                    let den: f64 = u.iter().zip(&wc).map(|(u, w)| w * u * u).sum();
                    format!("{:.1e}", (num / den).sqrt())
                })
                .collect();
            let ppw = 2.0 * (s - 1) as f64 / k as f64;
            println!("| {n} | {k} ({ppw:.1}) | {} |", cells.join(" | "));
        }
    }

    // 6. Timing at n = 257.
    println!("\n### 6. Time at n = 257 (median of 21, ms)\n");
    println!("| s | build T_cu | build T_uc (d = {d_eval}) | apply T_cu | apply T_uc |");
    println!("| --- | --- | --- | --- | --- |");
    let n = 257;
    for s in [n - 1, n] {
        let xc = grids::cheb_nodes(n);
        let xu = grids::uniform_nodes(s);
        let tcu = interp::t_cu(&xc, &xu);
        let tuc = interp::t_uc(&xu, &xc, d_eval);
        let fc = grf::eval(&xis[0], 64, &xc, &xc);
        let fu: Array2<f64> = grf::eval(&xis[0], 64, &xu, &xu);
        let b_cu = median_ms(|| drop(std::hint::black_box(interp::t_cu(&xc, &xu))), 21);
        let b_uc = median_ms(
            || drop(std::hint::black_box(interp::t_uc(&xu, &xc, d_eval))),
            21,
        );
        let a_cu = median_ms(
            || drop(std::hint::black_box(interp::apply2d(&tcu, &fc))),
            21,
        );
        let a_uc = median_ms(
            || drop(std::hint::black_box(interp::apply2d(&tuc, &fu))),
            21,
        );
        println!("| {s} | {b_cu:.3} | {b_uc:.3} | {a_cu:.3} | {a_uc:.3} |");
    }
}
