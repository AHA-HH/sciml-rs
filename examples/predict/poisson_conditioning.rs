//! Phase 10 T6 (docs/poisson2d/phase10/T6-conditioning.md, design §13.17 "Conditioning", H10b):
//! the Gauss–Newton curvature of trained Poisson FNOs across learning grids, the
//! single-sample tangent kernel in the sine basis, and the PDE side's condition numbers.
//!
//! Run with: cargo run --release --example poisson_conditioning -- docs/poisson2d/phase10/conditioning.md runs/<run_dir> ...
//! (`--no-default-features --features cuda` on the VM; add `datagen` for the Chebyshev
//! collocation column of the PDE table).
//!
//! For each run (stage 3, any optimiser; its own weights, never retrained):
//! 1. **Curvature spectra across s.** At each s of [`GRIDS`] the run's weights act on stage 3
//!    transferred to that grid, with both normalizers refit on the run's training set at
//!    that s (decision 18, as `poisson_resolution` does; at the run's own s they are the
//!    run's). On the same [`SAMPLES`] training samples, Phase 7's subset [`SUBSET_SEED`]
//!    (T4's training set), [`lanczos`] gives the Ritz values of G + μI, G = JᵀJ of T3's
//!    residual (`FnoResidual`, J v by finite differences) and μ = [`MU`], the weight
//!    decay, the same for every run (no run's damping λ). Reported: θ_max, the top
//!    [`TOP`] values, their decay θ_TOP / θ_max, θ_max / μ, and the adjoint error of one
//!    random direction.
//! 2. **Tangent kernel at s = [`KERNEL_S`].** For the first of those samples, the diagonal
//!    of Kᵢ = JᵢJᵢᵀ in the orthonormal sine basis of the uniform grid, ‖Jᵢᵀ φ_kl‖² for
//!    k, l = 1..s − 2 (one VJP each; the grid's boundary rows carry no sine), against
//!    1/μ_kl, and sorted by rank for comparison with Phase 6's POD eigenvalues of u.
//!
//! Every run must be a stage 3 run trained at s = 64, on the whole split of 1,000 or on
//! Phase 7's subset 0 of 100, so all are measured on the same samples.
//! 3. **PDE side.** κ of the uniform finite-difference Laplacian at each s (closed form)
//!    and, built with `datagen`, of the Chebyshev collocation Laplacian at n = s + 1.
//!
//! Writes the tables to the first argument and two plots beside it with gnuplot
//! (`<stem>_ritz.png`, `<stem>_kernel.png`; skipped with a warning if it is missing).

use std::f64::consts::PI;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use burn::prelude::*;
use gnuplot::{AxesCommon, Caption, Figure, PointSymbol};
use ndarray::Axis;
use rand_chacha::{
    ChaCha8Rng,
    rand_core::{Rng, SeedableRng},
};
use sciml_rs::neural_operators::{
    chebyshev::uniform_nodes,
    data::{
        batcher::Batch,
        catalogue::stage,
        dataset::OperatorDataset,
        device_batcher::device_loader,
        loaders::poisson::{PoissonConfig, load_poisson_uniform, subset_indices},
        transforms::normalizers::{FlatDecoder, UnitGaussianNormalizer},
    },
    metrics::poisson::log_correlation,
    models::fno::FNO,
    optim::gauss_newton::{ParamVec, adjoint_error, jvp, lanczos, params, vjp},
    runs::RunDir,
    training::gn::FnoResidual,
    utils::backend::backend_features,
};

/// The learning grids of measurement 1 (design §13.17).
const GRIDS: [usize; 3] = [32, 64, 128];
/// The grid of the tangent kernel (measurement 2).
const KERNEL_S: usize = 64;
/// Training samples of the curvature: Phase 7's subset of this size and seed.
const SAMPLES: usize = 100;
/// The subset seed: subset 0 is T4's training set.
const SUBSET_SEED: u64 = 0;
/// The train split the subset is drawn from: stage 3's 1,000 samples (Phase 7).
const POPULATION: usize = 1000;
/// Stage 3, the runs' stage.
const STAGE_N: usize = 129;
/// Ranks of the sorted kernel diagonal tabulated beside Phase 6's POD eigenvalues of u.
const POD_RANKS: usize = 12;
/// Lanczos steps (the brief asks for at least 50).
const LANCZOS_ITERS: usize = 60;
/// Ritz values tabulated per (run, s).
const TOP: usize = 10;
/// The shift of G + μI: the weight decay μ of both optimisers (G5).
const MU: f64 = 1e-4;
/// Seed of the Lanczos start vector and of the adjoint check's directions.
const SEED: u64 = 0;

/// μ_kl = (π²/4)(k² + l²), the eigenvalue of −Δ on [−1, 1]² for sin sin (design §3.3); as
/// `pde::poisson::grf::mu`, which needs `datagen`.
fn mu(k: usize, l: usize) -> f64 {
    PI * PI / 4.0 * ((k * k + l * l) as f64)
}

/// The κ of the 2D uniform finite-difference Laplacian on s points per axis (both
/// endpoints Dirichlet, m = s − 2 interior): sin²(mπ/2(m+1)) / sin²(π/2(m+1)), the 1D ratio,
/// since the 2D eigenvalues are sums of two 1D ones.
fn fd_kappa(s: usize) -> f64 {
    let m = (s - 2) as f64;
    let sin2 = |j: f64| (j * PI / (2.0 * (m + 1.0))).sin().powi(2);
    sin2(m) / sin2(1.0)
}

/// The κ of the 2D Chebyshev collocation Laplacian on n CGL nodes per axis.
#[cfg(feature = "datagen")]
fn cheb_kappa(n: usize) -> Option<f64> {
    let lam = sciml_rs::neural_operators::pde::poisson::collocation::interior_d2_eigenvalues(n);
    let (lo, hi) = lam.iter().fold((f64::INFINITY, 0.0_f64), |(lo, hi), l| {
        (lo.min(l.abs()), hi.max(l.abs()))
    });
    Some(hi / lo)
}

#[cfg(not(feature = "datagen"))]
fn cheb_kappa(_n: usize) -> Option<f64> {
    None
}

/// The orthonormal sine mode φ_kl on the s × s uniform grid, flattened as the residual is
/// (row-major over the two spatial axes): (2/(s − 1)) sin(kπ(x + 1)/2) sin(lπ(y + 1)/2),
/// k, l in 1..s − 2. It vanishes on the boundary rows.
fn sine_mode(s: usize, k: usize, l: usize) -> Vec<f32> {
    let x = uniform_nodes(s);
    let norm = 2.0 / (s - 1) as f64;
    let mut v = Vec::with_capacity(s * s);
    for a in 0..s {
        for b in 0..s {
            let sa = (k as f64 * PI * (x[a] + 1.0) / 2.0).sin();
            let sb = (l as f64 * PI * (x[b] + 1.0) / 2.0).sin();
            v.push((norm * sa * sb) as f32);
        }
    }
    v
}

/// A Rademacher vector with `template`'s structure, from `seed`.
fn rademacher(template: &ParamVec, seed: u64) -> ParamVec {
    let z = rademacher_host(template.numel(), seed);
    template.from_flat_like(Tensor::from_floats(z.as_slice(), &template.device()))
}

/// The run's training data at learning grid `s`, its normalizers refit there on the run's
/// training set; only the rows of Phase 7's subset [`SUBSET_SEED`] of [`SAMPLES`].
///
/// # Panics
/// If the data cannot be loaded, or the run trained on a subset other than that one.
fn samples_at(run: &RunDir, s: usize) -> (OperatorDataset<f32>, UnitGaussianNormalizer) {
    let trained: PoissonConfig = run
        .data_config()
        .unwrap_or_else(|e| panic!("{}: data config: {e}", run.path().display()));
    // every run is one operator on stage 3, trained at s = 64, measured on one sample set
    assert!(
        trained.n == STAGE_N && trained.s() == KERNEL_S,
        "{}: trained at n = {}, s = {}; T6 measures stage 3 runs trained at s = {KERNEL_S}",
        run.path().display(),
        trained.n,
        trained.s()
    );
    let set = stage(trained.n).unwrap_or_else(|e| panic!("{e}"));
    let datasets = Path::new(env!("CARGO_MANIFEST_DIR")).join("datasets/poisson");
    let cfg = trained.clone().with_s(Some(s));
    let (train, _, normalizers) = load_poisson_uniform::<f32>(
        datasets.join(set.npz("train")),
        datasets.join(set.npz("test")),
        &cfg,
    )
    .unwrap_or_else(|e| panic!("could not load stage {} at s = {s}: {e}", trained.n));
    let rows: Vec<usize> = match &trained.subset {
        None => {
            assert_eq!(
                train.len(),
                POPULATION,
                "{}: subset {SUBSET_SEED} is drawn from the whole split of {POPULATION}",
                run.path().display()
            );
            subset_indices(POPULATION, SAMPLES, SUBSET_SEED)
        }
        // the loader already kept the subset's rows, in index order
        Some(sub) => {
            assert!(
                sub.seed == SUBSET_SEED
                    && trained.base.n_train == SAMPLES
                    && sub.population == POPULATION,
                "{}: trained on subset {} of {}; T6 measures on subset {SUBSET_SEED} of {SAMPLES}",
                run.path().display(),
                sub.seed,
                trained.base.n_train
            );
            (0..SAMPLES).collect()
        }
    };
    let pick = |a: &ndarray::ArrayD<f32>| a.select(Axis(0), &rows);
    (
        OperatorDataset::new(pick(train.inputs()), pick(train.targets())),
        normalizers.y,
    )
}

/// `data` as one batch on `device`.
fn one_batch(data: &OperatorDataset<f32>, device: &Device) -> Batch<4, 3> {
    device_loader::<4, 3, f32>(data, device, data.len(), None)
        .iter()
        .next()
        .expect("a non-empty dataset")
        .unwrap_or_else(|e| panic!("could not batch the samples: {e}"))
}

/// Measurement 1 at one grid: the Ritz values, descending, and one adjoint error.
struct Spectrum {
    s: usize,
    ritz: Vec<f64>,
    adjoint_error: f64,
}

/// Measurement 2: ‖Jᵢᵀ φ_kl‖² per (k, l).
struct Kernel {
    diag: Vec<(usize, usize, f64)>,
}

struct RunResult {
    name: String,
    spectra: Vec<Spectrum>,
    kernel: Kernel,
}

fn measure(dir: &Path, device: &Device) -> RunResult {
    let run = RunDir::open(dir).unwrap_or_else(|e| panic!("open {}: {e}", dir.display()));
    let model: FNO<4> = run
        .load_fno(device)
        .unwrap_or_else(|e| panic!("load the model: {e}"));
    let name = dir
        .file_name()
        .map_or_else(|| dir.display().to_string(), |f| f.to_string_lossy().into());
    let mut spectra = Vec::new();
    let mut kernel = None;
    for s in GRIDS {
        let (data, y) = samples_at(&run, s);
        let decoder = FlatDecoder::new(&y, device);
        let problem = FnoResidual::<4, 3>::new(Some(&decoder));
        let batch = one_batch(&data, device);
        let apply = |v: &ParamVec| {
            let jv = jvp(&problem, &model, &batch, v, None);
            vjp(&problem, &model, &batch, &jv).axpy(MU, v)
        };
        let theta = params(&model);
        let mut ritz = lanczos(apply, &rademacher(&theta, SEED), LANCZOS_ITERS);
        ritz.reverse();
        let m = data.len() * s * s;
        let u = Tensor::<1>::from_floats(rademacher_host(m, SEED + 1).as_slice(), &theta.device());
        let adjoint = adjoint_error(
            &problem,
            &model,
            &batch,
            &u,
            &rademacher(&theta, SEED + 2),
            None,
        );
        println!(
            "{name} s = {s}: θ_max {:.4e}, θ_{TOP} {:.4e}, adjoint error {adjoint:.2e}",
            ritz[0],
            ritz[TOP.min(ritz.len()) - 1]
        );
        spectra.push(Spectrum {
            s,
            ritz,
            adjoint_error: adjoint,
        });

        if s == KERNEL_S {
            let first = OperatorDataset::new(
                data.inputs().select(Axis(0), &[0]),
                data.targets().select(Axis(0), &[0]),
            );
            let batch = one_batch(&first, device);
            let mut diag = Vec::with_capacity((s - 2) * (s - 2));
            for k in 1..s - 1 {
                for l in 1..s - 1 {
                    let phi = Tensor::<1>::from_floats(sine_mode(s, k, l).as_slice(), device);
                    let jt_phi = vjp(&problem, &model, &batch, &phi);
                    diag.push((k, l, jt_phi.dot(&jt_phi)));
                }
            }
            kernel = Some(Kernel { diag });
        }
    }
    RunResult {
        name,
        spectra,
        kernel: kernel.expect("KERNEL_S is one of GRIDS"),
    }
}

/// A host Rademacher vector of length `m`, from `seed`.
fn rademacher_host(m: usize, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..m)
        .map(|_| if rng.next_u32() & 1 == 0 { -1.0 } else { 1.0 })
        .collect()
}

fn sci(x: f64) -> String {
    format!("{x:.3e}")
}

fn table(results: &[RunResult]) -> String {
    let mut md = String::new();
    let w = &mut md;
    writeln!(w, "# Phase 10 conditioning (T6)\n").unwrap();
    writeln!(
        w,
        "Generated by `poisson_conditioning` (backend {:?}). Curvature: the Ritz values of \
         G + μI, μ = {MU:e}, by {LANCZOS_ITERS} steps of Lanczos with full \
         reorthogonalisation, on Phase 7's subset {SUBSET_SEED} of {SAMPLES} training samples \
         of stage 3, each run's weights at every s with normalizers refit there. Tangent \
         kernel: the diagonal of JᵢJᵢᵀ in the orthonormal sine basis at s = {KERNEL_S}, for \
         the first of those samples.\n",
        backend_features()
    )
    .unwrap();

    writeln!(w, "## Curvature spectra across s\n").unwrap();
    writeln!(
        w,
        "| run | s | θ_max | θ_{TOP} / θ_max | θ_max / μ | top {TOP} Ritz values | adjoint error |"
    )
    .unwrap();
    writeln!(w, "|{}", " --- |".repeat(7)).unwrap();
    for r in results {
        for sp in &r.spectra {
            let top = &sp.ritz[..TOP.min(sp.ritz.len())];
            writeln!(
                w,
                "| `{}` | {} | {} | {} | {} | {} | {} |",
                r.name,
                sp.s,
                sci(top[0]),
                sci(top[top.len() - 1] / top[0]),
                sci(top[0] / MU),
                top.iter().map(|x| sci(*x)).collect::<Vec<_>>().join(", "),
                sci(sp.adjoint_error),
            )
            .unwrap();
        }
    }

    writeln!(w, "\n## Tangent kernel in the sine basis, s = {KERNEL_S}\n").unwrap();
    writeln!(
        w,
        "Per run: the correlation of log ‖Jᵢᵀφ_kl‖² with log 1/μ_kl over the {} modes, and \
         the diagonal at the lowest modes against 1/μ_kl, both normalised by (1, 1).\n",
        (KERNEL_S - 2).pow(2)
    )
    .unwrap();
    writeln!(
        w,
        "| run | log correlation | (1,1) | (1,2) | (2,2) | (1,4) | (4,4) | (8,8) | (16,16) |"
    )
    .unwrap();
    writeln!(w, "|{}", " --- |".repeat(9)).unwrap();
    let modes = [(1, 1), (1, 2), (2, 2), (1, 4), (4, 4), (8, 8), (16, 16)];
    for r in results {
        let at = |k: usize, l: usize| {
            r.kernel
                .diag
                .iter()
                .find(|(a, b, _)| (*a, *b) == (k, l))
                .map_or(f64::NAN, |d| d.2)
        };
        let (diag, inv_mu): (Vec<f64>, Vec<f64>) = r
            .kernel
            .diag
            .iter()
            .map(|(k, l, d)| (*d, 1.0 / mu(*k, *l)))
            .unzip();
        let cells: Vec<String> = modes
            .iter()
            .map(|&(k, l)| {
                format!(
                    "{} / {}",
                    sci(at(k, l) / at(1, 1)),
                    sci(mu(1, 1) / mu(k, l))
                )
            })
            .collect();
        writeln!(
            w,
            "| `{}` | {:.3} | {} |",
            r.name,
            log_correlation(&diag, &inv_mu),
            cells.join(" | ")
        )
        .unwrap();
    }
    writeln!(
        w,
        "\nEach cell: kernel diagonal / (1/μ_kl), both relative to (1,1)."
    )
    .unwrap();

    writeln!(
        w,
        "\nThe kernel diagonal sorted descending, relative to its largest, at ranks \
         1..{POD_RANKS}: compare with the leading POD eigenvalues of u at s = {KERNEL_S}, \
         relative to the first (`docs/poisson2d/phase6/results.md`, \"Leading eigenvalues against the \
         CGL fit\"), and with 1/μ_kl sorted the same way.\n"
    )
    .unwrap();
    let names: Vec<String> = results.iter().map(|r| format!("`{}`", r.name)).collect();
    writeln!(w, "| rank | 1/μ (sorted) | {} |", names.join(" | ")).unwrap();
    writeln!(w, "|{}", " --- |".repeat(2 + results.len())).unwrap();
    let descending = |mut v: Vec<f64>| {
        v.sort_by(|a, b| b.total_cmp(a));
        v
    };
    let inv_mu = descending(
        (1..KERNEL_S - 1)
            .flat_map(|k| (1..KERNEL_S - 1).map(move |l| 1.0 / mu(k, l)))
            .collect(),
    );
    let sorted: Vec<Vec<f64>> = results
        .iter()
        .map(|r| descending(r.kernel.diag.iter().map(|x| x.2).collect()))
        .collect();
    for i in 0..POD_RANKS {
        let cells: Vec<String> = sorted.iter().map(|d| sci(d[i] / d[0])).collect();
        writeln!(
            w,
            "| {} | {} | {} |",
            i + 1,
            sci(inv_mu[i] / inv_mu[0]),
            cells.join(" | ")
        )
        .unwrap();
    }

    writeln!(w, "\n## PDE side: condition numbers\n").unwrap();
    writeln!(
        w,
        "| s | κ, uniform finite differences | κ, Chebyshev collocation at n = s + 1 |"
    )
    .unwrap();
    writeln!(w, "|{}", " --- |".repeat(3)).unwrap();
    for s in GRIDS {
        writeln!(
            w,
            "| {s} | {} | {} |",
            sci(fd_kappa(s)),
            cheb_kappa(s + 1).map_or("– (needs datagen)".into(), sci)
        )
        .unwrap();
    }
    md
}

fn plots(out: &Path, results: &[RunResult]) -> Result<Vec<PathBuf>, String> {
    let stem = out.file_stem().unwrap_or_default().to_string_lossy();
    let path = |name: &str| out.with_file_name(format!("{stem}_{name}.png"));
    let mut fig = Figure::new();
    let ax = fig
        .axes2d()
        .set_title("Ritz values of G + μI, descending", &[])
        .set_x_label("index", &[])
        .set_y_label("Ritz value", &[])
        .set_y_log(Some(10.0));
    for r in results {
        for sp in &r.spectra {
            let x: Vec<f64> = (1..=sp.ritz.len()).map(|i| i as f64).collect();
            ax.lines_points(
                &x,
                &sp.ritz,
                &[
                    Caption(&format!("{}, s = {}", r.name, sp.s)),
                    PointSymbol('O'),
                ],
            );
        }
    }
    fig.save_to_png(path("ritz"), 1000, 700)
        .map_err(|e| format!("{e} (is gnuplot installed?)"))?;

    let mut fig = Figure::new();
    let ax = fig
        .axes2d()
        .set_title(
            &format!("Tangent kernel diagonal against 1/μ_kl, s = {KERNEL_S}"),
            &[],
        )
        .set_x_label("1/μ_kl", &[])
        .set_y_label("‖Jᵢᵀφ_kl‖²", &[])
        .set_x_log(Some(10.0))
        .set_y_log(Some(10.0));
    for r in results {
        let (x, y): (Vec<f64>, Vec<f64>) = r
            .kernel
            .diag
            .iter()
            .map(|(k, l, d)| (1.0 / mu(*k, *l), *d))
            .unzip();
        ax.points(&x, &y, &[Caption(&r.name), PointSymbol('.')]);
    }
    fig.save_to_png(path("kernel"), 1000, 700)
        .map_err(|e| format!("{e} (is gnuplot installed?)"))?;
    Ok(vec![path("ritz"), path("kernel")])
}

fn main() {
    let mut args = std::env::args().skip(1);
    let usage = "usage: poisson_conditioning <out.md> <run_dir> ...";
    let out = PathBuf::from(args.next().expect(usage));
    let dirs: Vec<PathBuf> = args.map(PathBuf::from).collect();
    assert!(!dirs.is_empty(), "{usage}");
    // VJPs need the autodiff device
    let device = Device::default().autodiff();
    println!(
        "device {device:?}, backend features {:?}",
        backend_features()
    );
    let results: Vec<RunResult> = dirs.iter().map(|d| measure(d, &device)).collect();
    std::fs::write(&out, table(&results)).expect("write the tables");
    println!("wrote {}", out.display());
    match plots(&out, &results) {
        Ok(files) => files.iter().for_each(|f| println!("wrote {}", f.display())),
        Err(e) => eprintln!("plots skipped: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sine modes are orthonormal on the grid and vanish on its boundary rows.
    #[test]
    fn sine_modes_are_orthonormal() {
        let s = 10;
        let dot = |a: &[f32], b: &[f32]| -> f64 {
            a.iter()
                .zip(b)
                .map(|(x, y)| f64::from(*x) * f64::from(*y))
                .sum()
        };
        let (a, b, c) = (sine_mode(s, 1, 1), sine_mode(s, 2, 3), sine_mode(s, 8, 8));
        assert!((dot(&a, &a) - 1.0).abs() < 1e-5 && (dot(&c, &c) - 1.0).abs() < 1e-5);
        assert!(dot(&a, &b).abs() < 1e-6 && dot(&b, &c).abs() < 1e-6);
        assert!((0..s).all(|j| a[j] == 0.0 && a[(s - 1) * s + j].abs() < 1e-6));
    }

    /// κ of the finite-difference Laplacian: 1 on one interior point (s = 3), and about
    /// 4(m + 1)²/π² for large m.
    #[test]
    fn fd_kappa_closed_form() {
        assert!((fd_kappa(3) - 1.0).abs() < 1e-12);
        let m = 126.0;
        assert!((fd_kappa(128) / (4.0 * (m + 1.0_f64).powi(2) / (PI * PI)) - 1.0).abs() < 1e-3);
    }
}
