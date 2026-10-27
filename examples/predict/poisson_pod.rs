//! The POD of the stage 3 forcings f and solutions u on the 129 CGL grid (Phase 6 T1) and
//! on the Phase 5 learning grids s ∈ {16, 32, 64, 128} after R (T2), design §13.4:
//! singular values, energy ranks, reconstruction error against rank on the train and test
//! splits, the check against the closed-form spectrum of design §3.3, and the ranks against
//! the learning grid. Dataset-structure analysis: no model is trained or evaluated.
//!
//! Run with: cargo run --release --features datagen --example poisson_pod -- <out.md>
//!
//! Loads both splits' raw f and u with `load_poisson_cheb_field` and fits one
//! `metrics::pod::Pod` per field on the train split only, with Clenshaw–Curtis weights on
//! both axes (D5, docs/phase6/README.md); the test split is only projected. The
//! population covariance has eigenvalues λ_kl / S_K for f and λ_kl / (S_K μ_kl²) for u
//! (k, l ≤ K), both decreasing in k² + l²; degenerate modes mix in the sample, so they are
//! compared summed over each group of equal k² + l², and through the trace. On each
//! learning grid both splits are restricted with `Transfers::with_s(129, s)` (R = T_cu,
//! Phase 5 D1) and the train split fitted with equal weights scaled by the cell area
//! (2/(s − 1))² (D5). Writes the tables as Markdown to `<out.md>` and, if gnuplot is
//! installed, the plots `<stem>_{singular_values,energy,reconstruction,rank_vs_s}.png`
//! beside it.
//! Needs the stage 3 dataset: see datasets/README.md.

use std::fmt::{self, Write as _};
use std::path::Path;
use std::time::Instant;

use burn::config::Config;
use gnuplot::{AxesCommon, Caption, Color, DashType, Figure, LineStyle, LineWidth, PointSymbol};
use ndarray::{Array1, ArrayView3, CowArray, Ix3};
use sciml_rs::neural_operators::{
    chebyshev::{Transfers, clencurt},
    data::{
        catalogue::stage,
        loaders::poisson::{PoissonField, SidecarRecord, load_poisson_cheb_field},
    },
    metrics::{
        pod::{Pod, cumulative_energy, energy_rank_of},
        stats::ErrorStats,
    },
    pde::poisson::grf::{ALPHA, SpectrumMode, covariance_spectrum},
    utils::backend::git_commit,
};

/// The stage: CGL nodes per axis.
const N: usize = 129;
/// The ranks r of the reconstruction-error table; the largest is the number of modes kept.
const RANKS: [usize; 10] = [1, 2, 5, 10, 20, 50, 100, 200, 500, 999];
/// The energy fractions of the rank table.
const FRACTIONS: [f64; 3] = [0.9, 0.99, 0.999];
/// Rows of the leading-eigenvalue table.
const LEADING: usize = 20;
/// Leading groups of equal k² + l² in the closed-form check.
const GROUPS: usize = 10;
/// The Phase 5 learning grids s (design §13.3).
const LEARNING_GRIDS: [usize; 4] = [16, 32, 64, 128];
/// Rows of the learning grids' leading-eigenvalue table.
const LEADING_GRID: usize = 10;

/// The grid a fit lives on.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Grid {
    /// The `N`-point CGL grid, Clenshaw–Curtis weights (T1).
    Cgl,
    /// The uniform learning grid of s points per axis, after R (T2).
    Learning(usize),
}

impl Grid {
    /// Every grid of the report: the learning grids, then the CGL grid.
    fn all() -> impl Iterator<Item = Grid> {
        LEARNING_GRIDS
            .into_iter()
            .map(Grid::Learning)
            .chain([Grid::Cgl])
    }

    /// Points per axis.
    fn points(self) -> usize {
        match self {
            Grid::Cgl => N,
            Grid::Learning(s) => s,
        }
    }

    fn label(self) -> String {
        match self {
            Grid::Cgl => format!("{N} CGL (T1)"),
            Grid::Learning(s) => format!("s = {s}"),
        }
    }

    /// The per-axis weights of D5: Clenshaw–Curtis on the CGL grid; on a learning grid
    /// equal, scaled by the cell width so their product is the cell area (2/(s − 1))².
    fn weights(self) -> Array1<f64> {
        match self {
            Grid::Cgl => clencurt(N),
            Grid::Learning(s) => Array1::from_elem(s, 2.0 / (s - 1) as f64),
        }
    }

    /// `fields: [M, N, N]` on this grid: unchanged on the CGL grid, restricted with
    /// R = T_cu (`Transfers::with_s`, Phase 5 D1) on a learning grid.
    fn restrict(self, fields: ArrayView3<'_, f64>) -> CowArray<'_, f64, Ix3> {
        match self {
            Grid::Cgl => fields.into(),
            Grid::Learning(s) => Transfers::with_s(N, s).to_uniform_all(fields).into(),
        }
    }
}

/// One POD fit on one grid and its reconstruction errors.
struct Fit {
    grid: Grid,
    pod: Pod,
    /// Its eigenvalues, σ_i² / (N − 1).
    eigenvalues: Array1<f64>,
    fit_s: f64,
    /// The entries of `RANKS` the grid supports (r ≤ min(points, N)).
    ranks: Vec<usize>,
    /// Per entry of `ranks`.
    train: Vec<ErrorStats>,
    /// Per entry of `ranks`.
    test: Vec<ErrorStats>,
}

/// One field's fits and its comparison data.
struct FieldReport {
    name: &'static str,
    /// One per grid of [`Grid::all`].
    fits: Vec<Fit>,
    /// Population eigenvalues, descending, in the order of the shared spectrum.
    population: Vec<f64>,
}

impl FieldReport {
    /// The fit on `grid`.
    fn fit(&self, grid: Grid) -> &Fit {
        let found = self.fits.iter().find(|f| f.grid == grid);
        found.unwrap_or_else(|| panic!("no fit on {grid:?}"))
    }

    /// The fit on the CGL grid, the T1 tables' subject.
    fn cgl(&self) -> &Fit {
        self.fit(Grid::Cgl)
    }
}

/// What every table needs besides the fits.
struct Context<'a> {
    reports: &'a [FieldReport],
    /// The population spectrum, shared by f and u.
    spectrum: &'a [SpectrumMode],
    n_train: usize,
    n_test: usize,
    k_max: usize,
}

fn main() {
    let out = std::env::args()
        .nth(1)
        .expect("usage: poisson_pod <out.md>");
    let out = Path::new(&out);
    let set = stage(N).unwrap_or_else(|e| panic!("the stage: {e}"));
    let datasets = Path::new(env!("CARGO_MANIFEST_DIR")).join("datasets/poisson");
    let sidecar = |split| {
        SidecarRecord::load(datasets.join(set.json(split)))
            .unwrap_or_else(|e| panic!("load the {split} sidecar: {e}"))
    };
    let (train_meta, test_meta) = (sidecar("train"), sidecar("test"));
    assert_eq!(train_meta.alpha, ALPHA, "the dataset's α is not grf::ALPHA");
    let spectrum = covariance_spectrum(train_meta.k);

    let reports: Vec<FieldReport> = [(PoissonField::F, "f"), (PoissonField::U, "u")]
        .into_iter()
        .map(|(field, name)| {
            let load = |split, count| {
                load_poisson_cheb_field(datasets.join(set.npz(split)), field, N, count)
                    .unwrap_or_else(|e| panic!("load {name} ({split}): {e}"))
            };
            let train = load("train", train_meta.count);
            let test = load("test", test_meta.count);
            let fits = Grid::all()
                .map(|grid| {
                    let fit = Fit::new(grid, train.view(), test.view());
                    println!("{name}: {} fit in {:.1} s", grid.label(), fit.fit_s);
                    fit
                })
                .collect();
            let population = spectrum
                .iter()
                .map(|m| match field {
                    PoissonField::F => m.f,
                    PoissonField::U => m.u,
                })
                .collect();
            FieldReport {
                name,
                fits,
                population,
            }
        })
        .collect();

    let ctx = Context {
        reports: &reports,
        spectrum: &spectrum,
        n_train: train_meta.count,
        n_test: test_meta.count,
        k_max: train_meta.k,
    };
    let md = markdown(&ctx).expect("formatting into a String cannot fail");
    std::fs::write(out, md).unwrap_or_else(|e| panic!("write {}: {e}", out.display()));
    println!("wrote {}", out.display());
    plots(out, &ctx);
}

impl Fit {
    /// The POD of `train: [N, N_x, N_x]` (CGL fields) on `grid`, with its D5 weights, and the
    /// reconstruction errors of both splits at every rank of `RANKS` the grid supports.
    fn new(grid: Grid, train: ArrayView3<f64>, test: ArrayView3<f64>) -> Self {
        let (train, test) = (grid.restrict(train), grid.restrict(test));
        let (n, a, b) = train.dim();
        let ranks: Vec<usize> = RANKS.into_iter().filter(|&r| r <= n.min(a * b)).collect();
        let w = grid.weights();
        let start = Instant::now();
        let pod = Pod::fit(train.view(), w.view(), w.view(), ranks[ranks.len() - 1]);
        let fit_s = start.elapsed().as_secs_f64();
        let stats = |samples: ArrayView3<f64>| -> Vec<ErrorStats> {
            let errors = pod.reconstruction_errors(samples, &ranks);
            errors.iter().map(|e| ErrorStats::of(&e.to_vec())).collect()
        };
        let (train, test) = (stats(train.view()), stats(test.view()));
        Self {
            grid,
            eigenvalues: pod.eigenvalues(),
            pod,
            fit_s,
            ranks,
            train,
            test,
        }
    }
}

/// The whole results file: the header, then one section per table.
fn markdown(ctx: &Context) -> Result<String, fmt::Error> {
    let mut md = String::new();
    header(&mut md, ctx)?;
    traces(&mut md, ctx)?;
    ranks(&mut md, ctx)?;
    leading(&mut md, ctx)?;
    groups(&mut md, ctx)?;
    reconstruction(&mut md, ctx)?;
    grid_ranks(&mut md, ctx)?;
    grid_leading(&mut md, ctx)?;
    grid_reconstruction(&mut md, ctx)?;
    Ok(md)
}

fn header(w: &mut String, ctx: &Context) -> fmt::Result {
    let (n_train, n_test, k_max) = (ctx.n_train, ctx.n_test, ctx.k_max);
    writeln!(w, "# Phase 6 results\n")?;
    writeln!(
        w,
        "Dataset-structure analysis, not network performance (design §13.4): no model is \
         trained or evaluated. Written by `poisson_pod` at commit `{}`; every number below \
         comes from it.\n",
        git_commit()
    )?;
    writeln!(w, "## T1: POD on the {N} CGL grid\n")?;
    writeln!(
        w,
        "Stage 3: n = {N}, K = {k_max}; the basis fitted on the {n_train} train samples, the \
         {n_test} test samples only projected (centred with the train mean). Inner product: \
         Clenshaw–Curtis weights on both axes (D5). Eigenvalues are σ_i² / (N − 1). \
         Population: λ_kl / S_K for f and λ_kl / (S_K μ_kl²) for u, k, l ≤ {k_max}, τ = 3, \
         α = 2.\n"
    )
}

fn traces(w: &mut String, ctx: &Context) -> fmt::Result {
    let nf = ctx.n_train as f64;
    writeln!(w, "### Fits and traces\n")?;
    writeln!(
        w,
        "The expected scatter of the trace is one standard deviation of the sample \
         variance of a Gaussian field with the population spectrum, √(2 Σ λ_i² / (N − 1)) / \
         Σ λ_i.\n"
    )?;
    writeln!(
        w,
        "| Field | Fit wall time (s) | Trace, empirical | Trace, population | Ratio | Expected scatter |"
    )?;
    writeln!(w, "| --- | --- | --- | --- | --- | --- |")?;
    for r in ctx.reports {
        let emp = r.cgl().eigenvalues.sum();
        let pop: f64 = r.population.iter().sum();
        let scatter =
            (2.0 * r.population.iter().map(|l| l * l).sum::<f64>() / (nf - 1.0)).sqrt() / pop;
        writeln!(
            w,
            "| {} | {:.2} | {emp:.4e} | {pop:.4e} | {:.4} | {scatter:.1e} |",
            r.name,
            r.cgl().fit_s,
            emp / pop
        )?;
    }
    Ok(())
}

fn ranks(w: &mut String, ctx: &Context) -> fmt::Result {
    let directions = ctx.n_train - 1;
    writeln!(w, "\n### Energy ranks\n")?;
    writeln!(
        w,
        "The smallest r whose leading eigenvalues hold the fraction of the energy. The \
         empirical fractions are of the sample's energy, which lies in at most N − 1 = \
         {directions} directions; a population rank above that is reached in no sample of \
         this size, so the empirical rank at that fraction measures N, not the field \
         (marked †).\n"
    )?;
    let header: Vec<String> = FRACTIONS
        .iter()
        .map(|f| format!("{}%", f * 100.0))
        .collect();
    writeln!(
        w,
        "| Field | {} | Population {} |",
        header.join(" | "),
        header.join(" / ")
    )?;
    writeln!(w, "| --- |{} --- |", " --- |".repeat(FRACTIONS.len()))?;
    for r in ctx.reports {
        let pop = FRACTIONS.map(|f| energy_rank_of(&r.population, f));
        let emp: Vec<String> = FRACTIONS
            .iter()
            .zip(pop)
            .map(|(&f, p)| {
                let limited = if p > directions { " †" } else { "" };
                format!("{}{limited}", r.cgl().pod.energy_rank(f))
            })
            .collect();
        let pop: Vec<String> = pop.iter().map(usize::to_string).collect();
        writeln!(
            w,
            "| {} | {} | {} |",
            r.name,
            emp.join(" | "),
            pop.join(" / ")
        )?;
    }
    Ok(())
}

fn leading(w: &mut String, ctx: &Context) -> fmt::Result {
    writeln!(w, "\n### Leading eigenvalues\n")?;
    writeln!(
        w,
        "Individual eigenvalues within a degenerate group mix in the sample; the group check \
         below is the comparison.\n"
    )?;
    writeln!(
        w,
        "| i | f, empirical | f, population | u, empirical | u, population |"
    )?;
    writeln!(w, "| --- | --- | --- | --- | --- |")?;
    for i in 0..LEADING {
        let cells: Vec<String> = ctx
            .reports
            .iter()
            .map(|r| format!("{:.4e} | {:.4e}", r.cgl().eigenvalues[i], r.population[i]))
            .collect();
        writeln!(w, "| {} | {} |", i + 1, cells.join(" | "))?;
    }
    Ok(())
}

fn groups(w: &mut String, ctx: &Context) -> fmt::Result {
    let nf = ctx.n_train as f64;
    writeln!(w, "\n### Closed-form check: groups of equal k² + l²\n")?;
    writeln!(
        w,
        "Empirical eigenvalues are summed over the positions the group takes in the sorted \
         population spectrum. Expected scatter of a group of g modes: about √(2 / (N g)). \
         A ratio more than 3 scatters from 1 is marked ‼.\n"
    )?;
    writeln!(
        w,
        "| k² + l² | g | f, empirical | f, population | f ratio | u, empirical | u, population | u ratio | Scatter |"
    )?;
    writeln!(w, "| --- | --- | --- | --- | --- | --- | --- | --- | --- |")?;
    let mut offset = 0;
    for group in ctx.spectrum.chunk_by(|a, b| a.q == b.q).take(GROUPS) {
        let (q, g) = (group[0].q, group.len());
        let range = offset..offset + g;
        let scatter = (2.0 / (nf * g as f64)).sqrt();
        let cells: Vec<String> = ctx
            .reports
            .iter()
            .map(|r| {
                let emp = r.cgl().eigenvalues.slice(ndarray::s![range.clone()]).sum();
                let pop: f64 = r.population[range.clone()].iter().sum();
                let ratio = emp / pop;
                let flag = if (ratio - 1.0).abs() > 3.0 * scatter {
                    " ‼"
                } else {
                    ""
                };
                format!("{emp:.4e} | {pop:.4e} | {ratio:.3}{flag}")
            })
            .collect();
        writeln!(w, "| {q} | {g} | {} | {scatter:.3} |", cells.join(" | "))?;
        offset += g;
    }
    Ok(())
}

fn reconstruction(w: &mut String, ctx: &Context) -> fmt::Result {
    writeln!(w, "\n### Reconstruction error against rank\n")?;
    writeln!(
        w,
        "Per sample ‖v − v_r‖ / ‖v‖ in the CC norm, v the sample centred with the train \
         mean and v_r its projection on the first r modes; mean [max] over the split.\n"
    )?;
    writeln!(w, "| r | f, train | f, test | u, train | u, test |")?;
    writeln!(w, "| --- | --- | --- | --- | --- |")?;
    for (i, rank) in RANKS.iter().enumerate() {
        let cells: Vec<String> = ctx
            .reports
            .iter()
            .flat_map(|r| [r.cgl().train[i], r.cgl().test[i]])
            .map(|s| format!("{:.3e} [{:.3e}]", s.mean, s.max))
            .collect();
        writeln!(w, "| {rank} | {} |", cells.join(" | "))?;
    }
    Ok(())
}

fn grid_ranks(w: &mut String, ctx: &Context) -> fmt::Result {
    writeln!(w, "\n## T2: POD on the learning grids\n")?;
    writeln!(
        w,
        "R = T_cu (`Transfers::with_s({N}, s)`, Phase 5 D1) restricts both splits to the \
         s × s uniform learning grid. The POD of the restricted train split uses equal \
         weights scaled by the cell area (2/(s − 1))² (D5), so its eigenvalues approximate \
         L² ones and compare with the CGL fit; the test split is only projected. f and u \
         vanish on ∂Ω and R keeps the boundary nodes, so only the (s − 2)² interior points \
         carry data: the centred snapshot matrix has at most min((s − 2)², N − 1) \
         directions.\n"
    )?;
    writeln!(w, "### Energy ranks against the learning grid\n")?;
    writeln!(
        w,
        "| Grid | Directions | f, 90% | f, 99% | f, 99.9% | u, 90% | u, 99% | u, 99.9% |"
    )?;
    writeln!(w, "| --- | --- | --- | --- | --- | --- | --- | --- |")?;
    for grid in Grid::all() {
        let directions = (grid.points() - 2).pow(2).min(ctx.n_train - 1);
        let cells: Vec<String> = ctx
            .reports
            .iter()
            .flat_map(|r| FRACTIONS.map(|f| r.fit(grid).pod.energy_rank(f).to_string()))
            .collect();
        writeln!(
            w,
            "| {} | {directions} | {} |",
            grid.label(),
            cells.join(" | ")
        )?;
    }
    Ok(())
}

fn grid_leading(w: &mut String, ctx: &Context) -> fmt::Result {
    writeln!(w, "\n### Leading eigenvalues against the CGL fit\n")?;
    writeln!(
        w,
        "λ_i on the learning grid over λ_i on the {N} CGL grid (T1), and the same for the \
         trace.\n"
    )?;
    let header: Vec<String> = ctx
        .reports
        .iter()
        .flat_map(|r| LEARNING_GRIDS.map(|s| format!("{}, s = {s}", r.name)))
        .collect();
    writeln!(w, "| i | {} |", header.join(" | "))?;
    writeln!(w, "| --- |{}", " --- |".repeat(header.len()))?;
    // λ_i, then the trace, on each learning grid over the same on the CGL grid
    let rows = (0..LEADING_GRID)
        .map(|i| ((i + 1).to_string(), Some(i)))
        .chain([("trace".to_string(), None)]);
    for (label, i) in rows {
        let value = |f: &Fit| i.map_or_else(|| f.eigenvalues.sum(), |i| f.eigenvalues[i]);
        let cells: Vec<String> = ctx
            .reports
            .iter()
            .flat_map(|r| {
                LEARNING_GRIDS.map(|s| {
                    let ratio = value(r.fit(Grid::Learning(s))) / value(r.cgl());
                    format!("{ratio:.4}")
                })
            })
            .collect();
        writeln!(w, "| {label} | {} |", cells.join(" | "))?;
    }
    Ok(())
}

fn grid_reconstruction(w: &mut String, ctx: &Context) -> fmt::Result {
    writeln!(
        w,
        "\n### Reconstruction error against rank on the learning grids\n"
    )?;
    writeln!(
        w,
        "As in T1, in the area-weighted norm of the learning grid; mean [max] over the \
         split.\n"
    )?;
    writeln!(w, "| s | r | f, train | f, test | u, train | u, test |")?;
    writeln!(w, "| --- | --- | --- | --- | --- | --- |")?;
    for s in LEARNING_GRIDS {
        let grid = Grid::Learning(s);
        for (i, rank) in ctx.reports[0].fit(grid).ranks.iter().enumerate() {
            let cells: Vec<String> = ctx
                .reports
                .iter()
                .flat_map(|r| [r.fit(grid).train[i], r.fit(grid).test[i]])
                .map(|s| format!("{:.3e} [{:.3e}]", s.mean, s.max))
                .collect();
            writeln!(w, "| {s} | {rank} | {} |", cells.join(" | "))?;
        }
    }
    Ok(())
}

/// The four plots beside `out`; a missing gnuplot only warns.
fn plots(out: &Path, ctx: &Context) {
    let reports = ctx.reports;
    let n_train = ctx.n_train;
    let stem = out.with_extension("");
    let path = |name: &str| format!("{}_{name}.png", stem.display());
    let colors = ["blue", "red"];
    let save = |fig: &mut Figure, file: String| match fig.save_to_png(&file, 1000, 700) {
        Ok(()) => println!("plot written to {file}"),
        Err(e) => eprintln!("plot failed ({e}) — is gnuplot installed?"),
    };

    let mut fig = Figure::new();
    let ax = fig
        .axes2d()
        .set_title("Singular values of the centred CC-weighted snapshots", &[])
        .set_x_label("index i", &[])
        .set_y_label("σ_i", &[])
        .set_y_log(Some(10.0));
    for (r, c) in reports.iter().zip(colors) {
        let kept = RANKS[RANKS.len() - 1];
        let i: Vec<usize> = (1..=kept).collect();
        let emp: Vec<f64> = r
            .cgl()
            .pod
            .singular_values()
            .iter()
            .take(kept)
            .copied()
            .collect();
        let pop: Vec<f64> = r.population[..kept]
            .iter()
            .map(|l| ((n_train as f64 - 1.0) * l).sqrt())
            .collect();
        let name = r.name;
        ax.lines(
            &i,
            &emp,
            &[
                Caption(&format!("{name}, empirical")),
                Color(gnuplot::RGBString(c)),
                LineWidth(1.5),
            ],
        )
        .lines(
            &i,
            &pop,
            &[
                Caption(&format!("{name}, closed form √((N−1)λ)")),
                Color(gnuplot::RGBString(c)),
                LineStyle(DashType::Dash),
            ],
        );
    }
    save(&mut fig, path("singular_values"));

    let mut fig = Figure::new();
    let ax = fig
        .axes2d()
        .set_title("Cumulative energy against rank (train)", &[])
        .set_x_label("rank r", &[])
        .set_y_label("energy fraction", &[])
        .set_x_log(Some(10.0));
    for (r, c) in reports.iter().zip(colors) {
        let cum = cumulative_energy(&r.cgl().eigenvalues.to_vec());
        let i: Vec<usize> = (1..=cum.len()).collect();
        ax.lines(
            &i,
            &cum,
            &[
                Caption(r.name),
                Color(gnuplot::RGBString(c)),
                LineWidth(1.5),
            ],
        );
    }
    save(&mut fig, path("energy"));

    let mut fig = Figure::new();
    let ax = fig
        .axes2d()
        .set_title("Mean relative reconstruction error against rank", &[])
        .set_x_label("rank r", &[])
        .set_y_label("‖v − v_r‖ / ‖v‖", &[])
        .set_x_log(Some(10.0))
        .set_y_log(Some(10.0));
    for (r, c) in reports.iter().zip(colors) {
        let train: Vec<f64> = r.cgl().train.iter().map(|s| s.mean).collect();
        let test: Vec<f64> = r.cgl().test.iter().map(|s| s.mean).collect();
        let name = r.name;
        ax.lines_points(
            RANKS,
            &train,
            &[
                Caption(&format!("{name}, train")),
                Color(gnuplot::RGBString(c)),
            ],
        )
        .lines_points(
            RANKS,
            &test,
            &[
                Caption(&format!("{name}, test")),
                Color(gnuplot::RGBString(c)),
                LineStyle(DashType::Dash),
            ],
        );
    }
    save(&mut fig, path("reconstruction"));

    let mut fig = Figure::new();
    let ax = fig
        .axes2d()
        .set_title(
            &format!("Energy rank against the learning grid (CGL {N} dotted)"),
            &[],
        )
        .set_x_label("learning grid s", &[])
        .set_y_label("energy rank", &[])
        .set_x_log(Some(2.0))
        .set_y_log(Some(10.0));
    let span = [LEARNING_GRIDS[0], LEARNING_GRIDS[LEARNING_GRIDS.len() - 1]];
    for (r, c) in reports.iter().zip(colors) {
        for (f, symbol) in FRACTIONS.iter().zip(['O', 'S', 'T']) {
            let ranks = LEARNING_GRIDS.map(|s| r.fit(Grid::Learning(s)).pod.energy_rank(*f));
            let cgl = r.cgl().pod.energy_rank(*f);
            let label = format!("{}, {}%", r.name, f * 100.0);
            ax.lines_points(
                LEARNING_GRIDS,
                ranks,
                &[
                    Caption(&label),
                    Color(gnuplot::RGBString(c)),
                    PointSymbol(symbol),
                ],
            )
            .lines(
                span,
                [cgl, cgl],
                &[
                    Caption(""),
                    Color(gnuplot::RGBString(c)),
                    LineStyle(DashType::Dot),
                ],
            );
        }
    }
    save(&mut fig, path("rank_vs_s"));
}
