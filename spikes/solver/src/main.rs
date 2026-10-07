//! Runs the T4 measurements and prints markdown tables (pasted into REPORT.md).
//!
//! `cargo run --release` runs everything. `cargo run --release -- --only-n 257` runs only
//! the setup and solve paths at one n, for a peak-memory measurement with
//! `/usr/bin/time -l`.

use ndarray::Array2;
use solver_spike::cheb::{d_closed_form_error, d2_interior, d2_poly_error};
use solver_spike::colloc::{FdSolver, SylReuseSolver, kron_lu, syl_call};
use solver_spike::fem::{System, Variant};
use solver_spike::manufactured::ALL;
use solver_spike::{interior, rel_max, with_boundary};
use std::time::Instant;
use transfers_spike::grf;
use transfers_spike::grids::{cheb_nodes, clenshaw_curtis, rel_l2, sample};

const NS_A: [usize; 6] = [9, 17, 33, 65, 129, 257];
const NS_B: [usize; 3] = [9, 17, 33];
const NS_C: [usize; 4] = [33, 65, 129, 257];
const TOLS: [f64; 3] = [1e-6, 1e-8, 1e-10];
/// CG tolerance for the "converged" C solution (discretisation error only).
const REF_TOL: f64 = 1e-13;
const GRF_SEED0: u64 = 1000;
const N_GRF_COMMON: u64 = 5;
const N_GRF_COST: u64 = 50;
const N_PROJECT: f64 = 1200.0;
const STOP_TOL: f64 = 1e-10;
const SYL_AGREE_TOL: f64 = 1e-12;
const SYL_SLOWDOWN: f64 = 2.0;
const ORDER_RANGE: (f64, f64) = (1.8, 2.2);

fn k_of_n(n: usize) -> usize {
    ((n - 1) / 2).min(64)
}

fn secs(f: impl FnOnce()) -> f64 {
    let t0 = Instant::now();
    f();
    t0.elapsed().as_secs_f64()
}

fn median_secs(mut f: impl FnMut(), reps: usize) -> f64 {
    let mut t: Vec<f64> = (0..reps).map(|_| secs(&mut f)).collect();
    t.sort_by(f64::total_cmp);
    t[reps / 2]
}

/// Field data at one n: nodes, CC weights, and helpers.
struct Grid {
    n: usize,
    x: ndarray::Array1<f64>,
    w: ndarray::Array1<f64>,
    d: Array2<f64>,
}

impl Grid {
    fn new(n: usize) -> Self {
        Grid {
            n,
            x: cheb_nodes(n),
            w: clenshaw_curtis(n),
            d: d2_interior(n),
        }
    }
    fn l2(&self, a: &Array2<f64>, b: &Array2<f64>) -> f64 {
        rel_l2(a, b, &self.w, &self.w)
    }
    fn manufactured(&self, k: usize) -> (Array2<f64>, Array2<f64>) {
        (
            sample(&self.x, &self.x, ALL[k].u),
            sample(&self.x, &self.x, ALL[k].f),
        )
    }
    fn grf(&self, seed: u64) -> (Array2<f64>, Array2<f64>) {
        let xi = grf::draw_xi(seed);
        let k = k_of_n(self.n);
        (
            grf::eval_solution(&xi, k, &self.x, &self.x),
            grf::eval(&xi, k, &self.x, &self.x),
        )
    }
}

fn order(e_coarse: f64, e_fine: f64) -> f64 {
    (e_coarse / e_fine).log2()
}

fn h_max(n: usize) -> f64 {
    let x = cheb_nodes(n);
    (1..n).map(|i| x[i] - x[i - 1]).fold(0.0, f64::max)
}

struct Decisions {
    stop: Vec<String>,
    syl_agree: f64,
    t_fd_257: Option<f64>,
    t_syl_257: Option<f64>,
    /// [variant][solution] orders at n = 65, 129, 257 (CC L²).
    orders: [[Vec<f64>; 3]; 2],
    iters_257: [Option<usize>; 2],
    /// [variant][tol][solution] (CG error, discretisation error) at n = 257.
    cg_257: [[[CgCell; 3]; 3]; 2],
}

/// (CG error, discretisation error), once measured.
type CgCell = Option<(f64, f64)>;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if let Some(pos) = args.iter().position(|a| a == "--only-n") {
        let n: usize = args[pos + 1].parse().expect("--only-n <n>");
        only_n(n);
        return;
    }

    let mut dec = Decisions {
        stop: Vec::new(),
        syl_agree: 0.0,
        t_fd_257: None,
        t_syl_257: None,
        orders: Default::default(),
        iters_257: [None, None],
        cg_257: [[[None; 3]; 3]; 2],
    };

    // 1. D and D².
    println!("## 1. Differentiation matrices\n");
    println!("| n | max rel \\|D − closed form\\| | max rel error of D² on x^k, k < n |");
    println!("| --- | --- | --- |");
    for n in [9, 17, 33, 65] {
        println!(
            "| {n} | {:.1e} | {:.1e} |",
            d_closed_form_error(n),
            d2_poly_error(n)
        );
    }
    println!();

    // 2. Option A convergence.
    println!("## 2. Option A on the manufactured solutions\n");
    println!(
        "Errors relative to max |u| (max) and Clenshaw–Curtis relative L² (L²). \
         |syl − fd| is max |U_syl − U_fd| / max |U_fd|.\n"
    );
    println!(
        "| n | solution | A-fd max | A-fd L² | A-syl-reuse max | A-syl-reuse L² | A-syl-call max | \\|syl − fd\\| | trsyl status |"
    );
    println!("| --- | --- | --- | --- | --- | --- | --- | --- | --- |");
    let mut floors = [f64::INFINITY; 3];
    let mut imag_rows = Vec::new();
    for n in NS_A {
        eprintln!("A convergence n={n}");
        let g = Grid::new(n);
        let fd = FdSolver::new(&g.d, &g.d);
        let syl = SylReuseSolver::new(&g.d, &g.d);
        imag_rows.push((n, fd.max_imag_lambda, fd.max_imag_v));
        for (k, m) in ALL.iter().enumerate() {
            let (u, f) = g.manufactured(k);
            let fi = interior(&f);
            let u_fd = with_boundary(&fd.solve(&fi));
            let (us, st) = syl.solve(&fi);
            let u_syl = with_boundary(&us);
            let (uc, stc) = syl_call(&g.d, &g.d, &fi);
            let u_call = with_boundary(&uc);
            let e_fd = rel_max(&u_fd, &u);
            floors[k] = floors[k].min(e_fd);
            let agree = rel_max(&u_syl, &u_fd);
            dec.syl_agree = dec.syl_agree.max(agree);
            if n == 33 && k < 2 && e_fd > STOP_TOL {
                dec.stop.push(format!(
                    "A-fd error {e_fd:.1e} > 1e-10 on {} at n = 33",
                    m.name
                ));
            }
            println!(
                "| {n} | {} | {:.1e} | {:.1e} | {:.1e} | {:.1e} | {:.1e} | {:.1e} | {:?} / {:?} |",
                m.name,
                e_fd,
                g.l2(&u_fd, &u),
                rel_max(&u_syl, &u),
                g.l2(&u_syl, &u),
                rel_max(&u_call, &u),
                agree,
                st,
                stc
            );
        }
    }
    println!();
    println!("Floor (smallest A-fd max error over n):");
    for (k, m) in ALL.iter().enumerate() {
        println!("- {}: {:.1e}", m.name, floors[k]);
    }
    println!();
    println!("Eigen-decomposition of the interior D² (A-fd setup), largest imaginary parts:\n");
    println!("| n | max \\|Im λ\\| | max \\|Im V\\| |");
    println!("| --- | --- | --- |");
    for (n, il, iv) in imag_rows {
        println!("| {n} | {il:.1e} | {iv:.1e} |");
    }
    println!();

    // 3. A against B.
    println!("## 3. Option A against option B (dense Kronecker LU)\n");
    println!("max |U_A − U_B| / max |U_B| on the interior.\n");
    println!("| n | field | A-fd vs B | A-syl-reuse vs B |");
    println!("| --- | --- | --- | --- |");
    for n in NS_B {
        eprintln!("A vs B n={n}");
        let g = Grid::new(n);
        let fd = FdSolver::new(&g.d, &g.d);
        let syl = SylReuseSolver::new(&g.d, &g.d);
        let mut fields: Vec<(String, Array2<f64>)> = (0..3)
            .map(|k| (ALL[k].name.to_string(), g.manufactured(k).1))
            .collect();
        fields.push((format!("GRF seed {GRF_SEED0}"), g.grf(GRF_SEED0).1));
        for (name, f) in fields {
            let fi = interior(&f);
            let ub = kron_lu(&g.d, &g.d, &fi);
            let e_fd = rel_max(&fd.solve(&fi), &ub);
            let e_syl = rel_max(&syl.solve(&fi).0, &ub);
            for (route, e) in [("A-fd", e_fd), ("A-syl-reuse", e_syl)] {
                if e > STOP_TOL {
                    dec.stop.push(format!(
                        "{route} disagrees with B by {e:.1e} on {name} at n = {n}"
                    ));
                }
            }
            println!("| {n} | {name} | {e_fd:.1e} | {e_syl:.1e} |");
        }
    }
    println!();

    // 4 and 5. Option C convergence and CG tolerance.
    println!("## 4. Option C on the manufactured solutions\n");
    println!(
        "CG tolerance {REF_TOL:.0e} (relative residual), zero initial guess, no preconditioner. \
         Order is log2(e(previous n) / e(n)); h_max order uses log(e ratio)/log(h_max ratio).\n"
    );
    println!(
        "| variant | solution | n | h_max | max err | L² err | order (L²) | order vs h_max (L²) | CG iters | final residual |"
    );
    println!("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |");
    // [variant][solution][n index] -> L² error of the converged C solution.
    let mut c_err: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); 3]; 2];
    let mut tol_rows = Vec::new();
    for (vi, v) in [Variant::Fv, Variant::Q1].into_iter().enumerate() {
        for (k, m) in ALL.iter().enumerate() {
            for (ni, n) in NS_C.into_iter().enumerate() {
                eprintln!("C {} {} n={n}", v.name(), m.name);
                let g = Grid::new(n);
                let sys = System::assemble(v, &g.x);
                let (u, f) = g.manufactured(k);
                let t0 = Instant::now();
                let (uc, it, res) = sys.solve(&f, REF_TOL);
                let t_ref = t0.elapsed().as_secs_f64();
                let el2 = g.l2(&uc, &u);
                let (ord, ord_h) = if ni > 0 {
                    let prev = c_err[vi][k][ni - 1];
                    let nprev = NS_C[ni - 1];
                    (
                        order(prev, el2),
                        (prev / el2).ln() / (h_max(nprev) / h_max(n)).ln(),
                    )
                } else {
                    (f64::NAN, f64::NAN)
                };
                if ni > 0 {
                    dec.orders[vi][k].push(ord);
                }
                if n == 257 {
                    dec.iters_257[vi] = Some(dec.iters_257[vi].unwrap_or(0).max(it));
                }
                println!(
                    "| {} | {} | {n} | {:.3e} | {:.2e} | {:.2e} | {:.2} | {:.2} | {it} | {res:.1e} |",
                    v.name(),
                    m.name,
                    h_max(n),
                    rel_max(&uc, &u),
                    el2,
                    ord,
                    ord_h
                );
                for (ti, tol) in TOLS.into_iter().enumerate() {
                    let t0 = Instant::now();
                    let (ut, itt, _) = sys.solve(&f, tol);
                    let tt = t0.elapsed().as_secs_f64();
                    let cg_err = g.l2(&ut, &uc);
                    if n == 257 {
                        dec.cg_257[vi][ti][k] = Some((cg_err, el2));
                    }
                    tol_rows.push((v.name(), m.name, n, tol, itt, tt, cg_err, el2));
                }
                tol_rows.push((v.name(), m.name, n, REF_TOL, it, t_ref, 0.0, el2));
                c_err[vi][k].push(el2);
            }
        }
    }
    println!();
    println!("## 5. CG tolerance\n");
    println!(
        "CG error is the CC relative L² distance to the {REF_TOL:.0e} solution; \
         discretisation error is that solution's L² error. Time is one solve, load included.\n"
    );
    println!(
        "| variant | solution | n | tol | CG iters | time [ms] | CG error | disc. error | CG / disc. |"
    );
    println!("| --- | --- | --- | --- | --- | --- | --- | --- | --- |");
    for (v, s, n, tol, it, t, ce, de) in &tol_rows {
        println!(
            "| {v} | {s} | {n} | {tol:.0e} | {it} | {:.1} | {ce:.1e} | {de:.1e} | {:.1e} |",
            t * 1e3,
            ce / de
        );
    }
    println!();

    // 6. Common-grid comparison on GRF samples.
    println!("## 6. A against C on GRF samples\n");
    println!(
        "Seeds {GRF_SEED0}..{}, K(n) = min((n − 1)/2, 64). Cells are mean / max over the samples. \
         u_A is A-fd; u_C at CG tol {REF_TOL:.0e}. u_exact is the exact sine-series solution \
         of the truncated forcing.\n",
        GRF_SEED0 + N_GRF_COMMON - 1
    );
    println!(
        "| n | K | ‖u_A − u_exact‖ L² | ‖u_A − u_fv‖ L² | order | ‖u_A − u_fv‖ max | ‖u_A − u_q1‖ L² | order | ‖u_A − u_q1‖ max |"
    );
    println!("| --- | --- | --- | --- | --- | --- | --- | --- | --- |");
    let mut prev = [f64::NAN; 2];
    for n in NS_C {
        eprintln!("common grid n={n}");
        let g = Grid::new(n);
        let fd = FdSolver::new(&g.d, &g.d);
        let systems = [
            System::assemble(Variant::Fv, &g.x),
            System::assemble(Variant::Q1, &g.x),
        ];
        let mut e_exact = Vec::new();
        let mut e_c: [Vec<f64>; 2] = Default::default();
        let mut e_cmax: [Vec<f64>; 2] = Default::default();
        for s in 0..N_GRF_COMMON {
            let (u_ex, f) = g.grf(GRF_SEED0 + s);
            let ua = with_boundary(&fd.solve(&interior(&f)));
            e_exact.push(g.l2(&ua, &u_ex));
            for (vi, sys) in systems.iter().enumerate() {
                let (uc, _, _) = sys.solve(&f, REF_TOL);
                e_c[vi].push(g.l2(&uc, &ua));
                e_cmax[vi].push(rel_max(&uc, &ua));
            }
        }
        let mm = |v: &[f64]| {
            (
                v.iter().sum::<f64>() / v.len() as f64,
                v.iter().cloned().fold(0.0, f64::max),
            )
        };
        let (ex_mean, ex_max) = mm(&e_exact);
        let mut cells = Vec::new();
        for vi in 0..2 {
            let (m_l2, x_l2) = mm(&e_c[vi]);
            let (m_mx, x_mx) = mm(&e_cmax[vi]);
            let ord = order(prev[vi], m_l2);
            prev[vi] = m_l2;
            cells.push(format!(
                "{m_l2:.2e} / {x_l2:.2e} | {ord:.2} | {m_mx:.2e} / {x_mx:.2e}"
            ));
        }
        println!(
            "| {n} | {} | {ex_mean:.1e} / {ex_max:.1e} | {} | {} |",
            k_of_n(n),
            cells[0],
            cells[1]
        );
    }
    println!();

    // 7 and 8. Cost and projection.
    let costs = cost_table(&NS_A, &mut dec);
    println!("## 8. Projection: 1200 samples (1000 train + 200 test)\n");
    println!(
        "setup + 1200 × (GRF evaluation at K(n) + solve), from the {N_GRF_COST}-sample means.\n"
    );
    println!("| n | route | setup [s] | GRF eval total [s] | solve total [s] | total [s] |");
    println!("| --- | --- | --- | --- | --- | --- |");
    for c in &costs {
        for (route, setup, solve) in [
            ("A-fd", c.setup_fd, c.solve_fd),
            ("A-syl-reuse", c.setup_syl, c.solve_syl),
        ] {
            let ev = N_PROJECT * c.eval;
            let so = N_PROJECT * solve;
            println!(
                "| {} | {route} | {setup:.3} | {ev:.2} | {so:.2} | {:.2} |",
                c.n,
                setup + ev + so
            );
        }
    }
    println!();

    decide(&dec);
}

struct Cost {
    n: usize,
    setup_fd: f64,
    setup_syl: f64,
    solve_fd: f64,
    solve_syl: f64,
    eval: f64,
}

fn cost_table(ns: &[usize], dec: &mut Decisions) -> Vec<Cost> {
    println!("## 7. Cost\n");
    println!(
        "Setup is the median of 3 runs. Solve and GRF-evaluation times are means over \
         {N_GRF_COST} GRF samples (seeds {GRF_SEED0}..), forcing evaluated beforehand. \
         A-syl-call recomputes both Schur forms per solve. C assembly is one run.\n"
    );
    println!(
        "| n | setup A-fd [ms] | setup A-syl-reuse [ms] | solve A-fd [ms] | solve A-syl-reuse [ms] | solve A-syl-call [ms] | syl-reuse / fd | GRF eval [ms] | assembly C-fv [ms] | assembly C-q1 [ms] |"
    );
    println!("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |");
    let mut out = Vec::new();
    for &n in ns {
        eprintln!("cost n={n}");
        let c = cost_at(n);
        if n == 257 {
            dec.t_fd_257 = Some(c.solve_fd);
            dec.t_syl_257 = Some(c.solve_syl);
        }
        let g = Grid::new(n);
        let t_fv = secs(|| drop(System::assemble(Variant::Fv, &g.x)));
        let t_q1 = secs(|| drop(System::assemble(Variant::Q1, &g.x)));
        println!(
            "| {n} | {:.2} | {:.2} | {:.3} | {:.3} | {:.3} | {:.2} | {:.3} | {:.1} | {:.1} |",
            c.cost.setup_fd * 1e3,
            c.cost.setup_syl * 1e3,
            c.cost.solve_fd * 1e3,
            c.cost.solve_syl * 1e3,
            c.solve_call * 1e3,
            c.cost.solve_syl / c.cost.solve_fd,
            c.cost.eval * 1e3,
            t_fv * 1e3,
            t_q1 * 1e3
        );
        out.push(c.cost);
    }
    println!();
    out
}

struct CostAt {
    cost: Cost,
    solve_fd: f64,
    solve_syl: f64,
    solve_call: f64,
}

fn cost_at(n: usize) -> CostAt {
    let g = Grid::new(n);
    let setup_fd = median_secs(|| drop(FdSolver::new(&g.d, &g.d)), 3);
    let setup_syl = median_secs(|| drop(SylReuseSolver::new(&g.d, &g.d)), 3);
    let fd = FdSolver::new(&g.d, &g.d);
    let syl = SylReuseSolver::new(&g.d, &g.d);
    let mut fs = Vec::new();
    let eval_total = secs(|| {
        for s in 0..N_GRF_COST {
            let xi = grf::draw_xi(GRF_SEED0 + s);
            fs.push(interior(&grf::eval(&xi, k_of_n(n), &g.x, &g.x)));
        }
    });
    let per = N_GRF_COST as f64;
    let solve_fd = secs(|| fs.iter().for_each(|f| drop(fd.solve(f)))) / per;
    let solve_syl = secs(|| fs.iter().for_each(|f| drop(syl.solve(f)))) / per;
    let solve_call = secs(|| fs.iter().for_each(|f| drop(syl_call(&g.d, &g.d, f)))) / per;
    CostAt {
        cost: Cost {
            n,
            setup_fd,
            setup_syl,
            solve_fd,
            solve_syl,
            eval: eval_total / per,
        },
        solve_fd,
        solve_syl,
        solve_call,
    }
}

/// Setup and solve paths at one n, for peak RSS (`/usr/bin/time -l`).
fn only_n(n: usize) {
    let c = cost_at(n);
    let g = Grid::new(n);
    let (_, f) = g.grf(GRF_SEED0);
    let mut iters = Vec::new();
    for v in [Variant::Fv, Variant::Q1] {
        let sys = System::assemble(v, &g.x);
        iters.push(sys.solve(&f, REF_TOL).1);
    }
    println!(
        "n = {n}: solve A-fd {:.3} ms, A-syl-reuse {:.3} ms, A-syl-call {:.3} ms; CG iters (fv, q1) at {REF_TOL:.0e}: {iters:?}",
        c.solve_fd * 1e3,
        c.solve_syl * 1e3,
        c.solve_call * 1e3
    );
}

fn decide(dec: &Decisions) {
    println!("## Decisions\n");
    if !dec.stop.is_empty() {
        println!("**Stop condition fired:**\n");
        for s in &dec.stop {
            println!("- {s}");
        }
        println!();
    }

    // A route.
    let (tf, ts) = (dec.t_fd_257.unwrap(), dec.t_syl_257.unwrap());
    let agree = dec.syl_agree <= SYL_AGREE_TOL;
    let fast = ts <= SYL_SLOWDOWN * tf;
    let route = if agree && fast { "A-syl" } else { "A-fd" };
    println!(
        "- **A's route: {route}.** A-syl-reuse vs A-fd max relative difference {:.1e} \
         (rule ≤ {SYL_AGREE_TOL:.0e}: {}); per-solve time at n = 257 A-syl-reuse {:.2} ms vs \
         A-fd {:.2} ms, ratio {:.2} (rule ≤ {SYL_SLOWDOWN}: {}).",
        dec.syl_agree,
        if agree { "met" } else { "not met" },
        ts * 1e3,
        tf * 1e3,
        ts / tf,
        if fast { "met" } else { "not met" }
    );

    // C variant.
    let qualifies = |vi: usize| {
        dec.orders[vi]
            .iter()
            .all(|o| o.iter().all(|&x| x >= ORDER_RANGE.0 && x <= ORDER_RANGE.1))
    };
    let names = ["C-fv", "C-q1"];
    for (vi, name) in names.iter().enumerate() {
        let ords: Vec<String> = dec.orders[vi]
            .iter()
            .map(|o| {
                o.iter()
                    .map(|x| format!("{x:.2}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .collect();
        println!(
            "- {name}: orders (L², arriving at n = 65, 129, 257) per solution [{}]; qualifies: {}; \
             max CG iterations at n = 257 (tol {REF_TOL:.0e}): {}.",
            ords.join("; "),
            qualifies(vi),
            dec.iters_257[vi].unwrap()
        );
    }
    let chosen = match (qualifies(0), qualifies(1)) {
        (true, true) => Some(if dec.iters_257[0] <= dec.iters_257[1] {
            0
        } else {
            1
        }),
        (true, false) => Some(0),
        (false, true) => Some(1),
        (false, false) => None,
    };
    match chosen {
        Some(vi) => println!("- **C's discretisation: {}.**", names[vi]),
        None => println!("- **C's discretisation: neither variant meets the order rule.**"),
    }

    // CG tolerance, for the chosen variant (or both if none qualifies).
    let check = |vi: usize| {
        // TOLS is ordered loosest first.
        TOLS.into_iter().enumerate().find_map(|(ti, tol)| {
            dec.cg_257[vi][ti]
                .iter()
                .all(|c| c.is_some_and(|(ce, de)| ce < 0.1 * de))
                .then_some(tol)
        })
    };
    for vi in chosen.map(|v| vec![v]).unwrap_or(vec![0, 1]) {
        match check(vi) {
            Some(t) => println!(
                "- **C's CG tolerance ({}): {t:.0e}** (loosest of {:?} with CG error < 0.1 × discretisation error at n = 257 on all three solutions).",
                names[vi], TOLS
            ),
            None => println!(
                "- **C's CG tolerance ({}): none of {:?} meets the rule.**",
                names[vi], TOLS
            ),
        }
    }
}
