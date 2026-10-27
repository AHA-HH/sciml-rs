//! The POD of the stage 3 forcings f and solutions u on the 129 CGL grid (Phase 6 T1,
//! design §13.4): singular values, energy ranks, reconstruction error against rank on the
//! train and test splits, and the check against the closed-form spectrum of design §3.3.
//! Dataset-structure analysis: no model is trained or evaluated.
//!
//! Run with: cargo run --release --features datagen --example poisson_pod -- <out.md>
//!
//! Loads both splits' raw f and u with `load_poisson_cheb_field` and fits one
//! `metrics::pod::Pod` per field on the train split only, with Clenshaw–Curtis weights on
//! both axes (D5, docs/phase6/README.md); the test split is only projected. The
//! population covariance has eigenvalues λ_kl / S_K for f and λ_kl / (S_K μ_kl²) for u
//! (k, l ≤ K), both decreasing in k² + l²; degenerate modes mix in the sample, so they are
//! compared summed over each group of equal k² + l², and through the trace. Writes the
//! tables as Markdown to `<out.md>` and, if gnuplot is installed, the plots
//! `<stem>_{singular_values,energy,reconstruction}.png` beside it.
//! Needs the stage 3 dataset: see datasets/README.md.

use std::fmt::{self, Write as _};
use std::path::Path;
use std::time::Instant;

use burn::config::Config;
use gnuplot::{AxesCommon, Caption, Color, DashType, Figure, LineStyle, LineWidth};
use ndarray::Array1;
use sciml_rs::neural_operators::{
    chebyshev::clencurt,
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

/// One field's fit and its comparison data.
struct FieldReport {
    name: &'static str,
    pod: Pod,
    /// Its eigenvalues, σ_i² / (N − 1).
    eigenvalues: Array1<f64>,
    fit_s: f64,
    /// Per entry of `RANKS`.
    train: Vec<ErrorStats>,
    /// Per entry of `RANKS`.
    test: Vec<ErrorStats>,
    /// Population eigenvalues, descending, in the order of the shared spectrum.
    population: Vec<f64>,
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
    let w = clencurt(N);
    let keep = RANKS[RANKS.len() - 1];

    let reports: Vec<FieldReport> = [(PoissonField::F, "f"), (PoissonField::U, "u")]
        .into_iter()
        .map(|(field, name)| {
            let load = |split, count| {
                load_poisson_cheb_field(datasets.join(set.npz(split)), field, N, count)
                    .unwrap_or_else(|e| panic!("load {name} ({split}): {e}"))
            };
            let train = load("train", train_meta.count);
            let test = load("test", test_meta.count);
            let start = Instant::now();
            let pod = Pod::fit(train.view(), w.view(), w.view(), keep);
            let fit_s = start.elapsed().as_secs_f64();
            println!("{name}: fit {} snapshots in {fit_s:.1} s", train_meta.count);
            let stats = |samples: &ndarray::Array3<f64>| -> Vec<ErrorStats> {
                let errors = pod.reconstruction_errors(samples.view(), &RANKS);
                errors.iter().map(|e| ErrorStats::of(&e.to_vec())).collect()
            };
            let (train, test) = (stats(&train), stats(&test));
            let population = spectrum
                .iter()
                .map(|m| match field {
                    PoissonField::F => m.f,
                    PoissonField::U => m.u,
                })
                .collect();
            FieldReport {
                name,
                eigenvalues: pod.eigenvalues(),
                pod,
                fit_s,
                train,
                test,
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

/// The whole results file: the header, then one section per table.
fn markdown(ctx: &Context) -> Result<String, fmt::Error> {
    let mut md = String::new();
    header(&mut md, ctx)?;
    traces(&mut md, ctx)?;
    ranks(&mut md, ctx)?;
    leading(&mut md, ctx)?;
    groups(&mut md, ctx)?;
    reconstruction(&mut md, ctx)?;
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
        "| Field | SVD wall time (s) | Trace, empirical | Trace, population | Ratio | Expected scatter |"
    )?;
    writeln!(w, "| --- | --- | --- | --- | --- | --- |")?;
    for r in ctx.reports {
        let emp = r.eigenvalues.sum();
        let pop: f64 = r.population.iter().sum();
        let scatter =
            (2.0 * r.population.iter().map(|l| l * l).sum::<f64>() / (nf - 1.0)).sqrt() / pop;
        writeln!(
            w,
            "| {} | {:.2} | {emp:.4e} | {pop:.4e} | {:.4} | {scatter:.1e} |",
            r.name,
            r.fit_s,
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
                format!("{}{limited}", r.pod.energy_rank(f))
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
            .map(|r| format!("{:.4e} | {:.4e}", r.eigenvalues[i], r.population[i]))
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
                let emp = r.eigenvalues.slice(ndarray::s![range.clone()]).sum();
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
            .flat_map(|r| [r.train[i], r.test[i]])
            .map(|s| format!("{:.3e} [{:.3e}]", s.mean, s.max))
            .collect();
        writeln!(w, "| {rank} | {} |", cells.join(" | "))?;
    }
    Ok(())
}

/// The three plots beside `out`; a missing gnuplot only warns.
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
        let emp: Vec<f64> = r.pod.singular_values().iter().take(kept).copied().collect();
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
        let cum = cumulative_energy(&r.eigenvalues.to_vec());
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
        let train: Vec<f64> = r.train.iter().map(|s| s.mean).collect();
        let test: Vec<f64> = r.test.iter().map(|s| s.mean).collect();
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
}
