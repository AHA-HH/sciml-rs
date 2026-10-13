//! Metal-only diagnostics: Flex vs Metal, op by op, behind the Burn bug reports in
//! `upstream/burn/` (the fused-`cat` stride and new-shape replay bugs) and the 2D
//! replay check `replay_2d_poisson`. Prints comparisons; most tests assert nothing.
//! Run with `cargo test [--release] --features metal --test metal_diag -- --nocapture`.
#![cfg(feature = "metal")]

use burn::module::Module;
use burn::nn::{Linear, LinearConfig, PaddingConfig1d, conv::Conv1dConfig};
use burn::prelude::*;
use burn::tensor::{Distribution, ops::PadMode, signal};
use sciml_rs::neural_operators::layers::spectral_convolution::SpectralConv;
use sciml_rs::neural_operators::models::fno::FNOConfig;
use sciml_rs::neural_operators::utils::fft::icfft_full_spectrum;

fn metal() -> Device {
    Device::metal(Default::default())
}

fn v<const R: usize>(t: Tensor<R>) -> Vec<f32> {
    t.into_data().try_to_vec::<f32>().unwrap()
}

fn rel(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>();
    let den: f32 = b.iter().map(|y| y * y).sum::<f32>();
    (num / den.max(1e-30)).sqrt()
}

fn rnd<const R: usize>(shape: [usize; R]) -> Tensor<R> {
    let f = Device::flex();
    Tensor::<R>::random(shape, Distribution::Normal(0.0, 1.0), &f)
}

/// Runs `op` on Flex and twice on Metal from the same host input; prints
/// Metal-vs-Flex and Metal-vs-Metal relative errors.
fn cmp<const R: usize, const S: usize>(
    name: &str,
    x: Tensor<R>,
    op: impl Fn(Tensor<R>) -> Tensor<S>,
) {
    let m = metal();
    let f = v(op(x.clone()));
    let m1 = v(op(x.clone().to_device(&m)));
    let m2 = v(op(x.to_device(&m)));
    let flag = if rel(&m1, &f) > 1e-5 || rel(&m1, &m2) > 1e-6 {
        "  <-- BAD"
    } else {
        ""
    };
    println!(
        "{name:<48} metal/flex {:.2e}  metal/metal {:.2e}{flag}",
        rel(&m1, &f),
        rel(&m1, &m2)
    );
}

fn shapes5() -> Vec<[usize; 5]> {
    vec![
        [2, 6, 4, 6, 5],
        [2, 6, 6, 8, 7],
        [2, 6, 8, 8, 8],
        [2, 6, 4, 4, 4],
    ]
}
fn shapes4() -> Vec<[usize; 4]> {
    vec![[2, 6, 8, 6], [2, 6, 11, 9], [2, 6, 17, 15], [2, 6, 16, 16]]
}

#[test]
fn ffts() {
    for s in shapes5() {
        cmp(&format!("rfft last  {s:?}"), rnd(s), |x| {
            let (r, i) = signal::rfft(x, 4, Some(s[4]));
            Tensor::cat(vec![r, i], 4)
        });
        for axis in 2..4 {
            cmp(&format!("cfft ax{axis}   {s:?}"), rnd(s), move |x| {
                let (r, i) = signal::cfft(x.clone(), x.mul_scalar(0.5), axis, Some(s[axis]));
                Tensor::cat(vec![r, i], 4)
            });
            cmp(&format!("icfft ax{axis}  {s:?}"), rnd(s), move |x| {
                let (r, i) = icfft_full_spectrum(x.clone(), x.mul_scalar(-0.3), axis, s[axis]);
                Tensor::cat(vec![r, i], 4)
            });
        }
        let mut h = s;
        h[4] = s[4] / 2 + 1;
        cmp(&format!("irfft last {h:?}->{}", s[4]), rnd(h), |x| {
            signal::irfft(x.clone(), x.mul_scalar(0.7), 4, Some(s[4]))
        });
    }
    for s in shapes4() {
        cmp(&format!("rfft last  {s:?}"), rnd(s), |x| {
            let (r, i) = signal::rfft(x, 3, Some(s[3]));
            Tensor::cat(vec![r, i], 3)
        });
        cmp(&format!("cfft ax2   {s:?}"), rnd(s), move |x| {
            let (r, i) = signal::cfft(x.clone(), x.mul_scalar(0.5), 2, Some(s[2]));
            Tensor::cat(vec![r, i], 3)
        });
        cmp(&format!("icfft ax2  {s:?}"), rnd(s), move |x| {
            let (r, i) = icfft_full_spectrum(x.clone(), x.mul_scalar(-0.3), 2, s[2]);
            Tensor::cat(vec![r, i], 3)
        });
        let mut h = s;
        h[3] = s[3] / 2 + 1;
        cmp(&format!("irfft last {h:?}->{}", s[3]), rnd(h), |x| {
            signal::irfft(x.clone(), x.mul_scalar(0.7), 3, Some(s[3]))
        });
    }
}

#[test]
fn layout_ops() {
    for s in shapes5() {
        cmp(&format!("pad +2     {s:?}"), rnd(s), |x| {
            x.pad(
                [(0, 0), (0, 0), (0, 2), (0, 2), (0, 2)],
                PadMode::Constant(0.0),
            )
        });
        cmp(&format!("cat zeros  {s:?}"), rnd(s), |x| {
            let mut d = x.dims();
            d[3] = 2;
            let z = Tensor::<5>::zeros(d, &x.device());
            Tensor::cat(vec![x, z], 3)
        });
        cmp(&format!("slice      {s:?}"), rnd(s), |x| {
            x.slice([0..2, 0..6, 0..3, 1..5, 0..4])
        });
        cmp(&format!("slice_assign {s:?}"), rnd(s), |x| {
            let z = Tensor::<5>::zeros(x.dims(), &x.device());
            z.slice_assign(
                [0..2, 0..6, 0..2, 0..2, 0..2],
                x.slice([0..2, 0..6, 0..2, 0..2, 0..2]),
            )
        });
        cmp(&format!("permute+reshape {s:?}"), rnd(s), |x| {
            let [b, c, a, bb, cc] = x.dims();
            x.permute([0, 4, 1, 2, 3]).reshape([b, cc * c, a * bb])
        });
    }
}

#[test]
fn matmuls() {
    let f = Device::flex();
    for (flat, b, i, o) in [(8, 2, 6, 6), (12, 2, 6, 6), (1, 2, 6, 6), (8, 1, 3, 3)] {
        let w = rnd([flat, i, o]);
        cmp(
            &format!("bmm [{flat},{b},{i}]x[{flat},{i},{o}]"),
            rnd([flat, b, i]),
            move |x| x.clone().matmul(w.clone().to_device(&x.device())),
        );
    }
    for n in [120, 336, 255, 16] {
        let w = rnd([6, 6]);
        cmp(
            &format!("pointwise [1,6,6]@[2,6,{n}]"),
            rnd([2, 6, n]),
            move |x| w.clone().to_device(&x.device()).unsqueeze::<3>().matmul(x),
        );
    }
    let lin: Linear = LinearConfig::new(6, 6).init(&f);
    let lin2 = lin.clone().to_device(&metal());
    let x = rnd([2, 6, 8, 7, 6]);
    let a = v(lin.forward(x.clone()));
    let b = v(lin2.forward(x.to_device(&metal())));
    println!(
        "{:<48} metal/flex {:.2e}",
        "Linear [2,6,8,7,6]",
        rel(&b, &a)
    );
    let conv = Conv1dConfig::new(6, 6, 1)
        .with_padding(PaddingConfig1d::Valid)
        .init(&f);
    let conv2 = conv.clone().to_device(&metal());
    let x = rnd([2, 6, 336]);
    let a = v(conv.forward(x.clone()));
    let b = v(conv2.forward(x.to_device(&metal())));
    println!(
        "{:<48} metal/flex {:.2e}",
        "Conv1d k=1 [2,6,336]",
        rel(&b, &a)
    );
}

#[test]
fn spectral_and_fno() {
    let f = Device::flex();
    for (s, modes) in [
        ([2, 6, 4, 6, 5], vec![2, 2, 2]),
        ([2, 6, 6, 8, 7], vec![2, 2, 2]),
        ([2, 6, 8, 8, 8], vec![2, 2, 2]),
    ] {
        let sc = SpectralConv::<5>::new(&f, 6, 6, &modes);
        let sm = sc.clone().to_device(&metal());
        let x = rnd(s);
        let a = v(sc.forward(x.clone()));
        let b1 = v(sm.forward(x.clone().to_device(&metal())));
        let b2 = v(sm.forward(x.to_device(&metal())));
        println!(
            "{:<48} metal/flex {:.2e}  metal/metal {:.2e}",
            format!("SpectralConv<5> {s:?}"),
            rel(&b1, &a),
            rel(&b1, &b2)
        );
    }
    for (s, modes) in [([2, 6, 8, 6], vec![3, 2]), ([2, 6, 17, 15], vec![3, 2])] {
        let sc = SpectralConv::<4>::new(&f, 6, 6, &modes);
        let sm = sc.clone().to_device(&metal());
        let x = rnd(s);
        let a = v(sc.forward(x.clone()));
        let b = v(sm.forward(x.to_device(&metal())));
        println!(
            "{:<48} metal/flex {:.2e}",
            format!("SpectralConv<4> {s:?}"),
            rel(&b, &a)
        );
    }
    for p in [None, Some(2)] {
        let m = FNOConfig::new(vec![2, 2, 2], 3, 1)
            .with_hidden_channels(6)
            .with_n_layers(2)
            .with_padding(p)
            .init::<5>(&f);
        let mm = m.clone().to_device(&metal());
        let x = rnd([2, 4, 6, 5, 3]);
        let a = v(m.forward(x.clone()));
        let b = v(mm.forward(x.to_device(&metal())));
        println!(
            "{:<48} metal/flex {:.2e}",
            format!("FNO<5> pad {p:?}"),
            rel(&b, &a)
        );
    }
    for p in [None, Some(3), Some(9)] {
        let m = FNOConfig::new(vec![3, 2], 1, 1)
            .with_hidden_channels(6)
            .with_n_layers(2)
            .with_padding(p)
            .init::<4>(&f);
        let mm = m.clone().to_device(&metal());
        let x = rnd([2, 8, 6, 1]);
        let a = v(m.forward(x.clone()));
        let b = v(mm.forward(x.to_device(&metal())));
        println!(
            "{:<48} metal/flex {:.2e}",
            format!("FNO<4> pad {p:?}"),
            rel(&b, &a)
        );
    }
}

/// SpectralConv::forward rebuilt stage by stage (2D), each stage compared.
fn stages2(s: [usize; 4], modes: [usize; 2]) {
    let x = rnd(s);
    let w_re = rnd([s[1], s[1], modes[0], modes[1]]);
    let w_im = rnd([s[1], s[1], modes[0], modes[1]]);
    let fwd = |x: Tensor<4>| {
        let (r, i) = signal::rfft(x, 3, Some(s[3]));
        signal::cfft(r, i, 2, Some(s[2]))
    };
    let both = |(r, i): (Tensor<4>, Tensor<4>)| Tensor::cat(vec![r, i], 3);
    cmp(&format!("{s:?} 1 rfft->cfft"), x.clone(), move |x| {
        both(fwd(x))
    });
    cmp(
        &format!("{s:?} 1b rfft->cfft, cloned inputs"),
        x.clone(),
        move |x| {
            let (r, i) = signal::rfft(x, 3, Some(s[3]));
            let (r, i) = (r.clone() + 0.0, i.clone() + 0.0);
            both(signal::cfft(r, i, 2, Some(s[2])))
        },
    );
    let h = s[3] / 2 + 1;
    let spec = [s[0], s[1], s[2], h];
    let inv = move |r: Tensor<4>, i: Tensor<4>| {
        let (r, i) = icfft_full_spectrum(r, i, 2, s[2]);
        signal::irfft(r, i, 3, Some(s[3]))
    };
    cmp(
        &format!("{s:?} 2 icfft->irfft (random spec)"),
        rnd(spec),
        move |x| inv(x.clone(), x.mul_scalar(0.3)),
    );
    cmp(&format!("{s:?} 3 full roundtrip"), x.clone(), move |x| {
        let (r, i) = fwd(x);
        inv(r, i)
    });
    let (wr, wi) = (w_re.clone(), w_im.clone());
    cmp(
        &format!("{s:?} 4 spectralconv rebuilt"),
        x.clone(),
        move |x| {
            let d = x.device();
            let (wr, wi) = (wr.clone().to_device(&d), wi.clone().to_device(&d));
            let (r, i) = fwd(x);
            let mut or = Tensor::<4>::zeros(spec, &d);
            let mut oi = Tensor::<4>::zeros(spec, &d);
            for rows in [0..modes[0], s[2] - modes[0]..s[2]] {
                let rg = [0..s[0], 0..s[1], rows.clone(), 0..modes[1]];
                let xr = r.clone().slice(rg.clone());
                let xi = i.clone().slice(rg.clone());
                let fl = modes[0] * modes[1];
                let p = |t: Tensor<4>, a, b| t.reshape([a, b, fl]).permute([2, 0, 1]);
                let (xr, xi) = (p(xr, s[0], s[1]), p(xi, s[0], s[1]));
                let (ar, ai) = (p(wr.clone(), s[1], s[1]), p(wi.clone(), s[1], s[1]));
                let ac = xr.clone().matmul(ar.clone());
                let bd = xi.clone().matmul(ai.clone());
                let abcd = (xr + xi).matmul(ar + ai);
                let back = |t: Tensor<3>| {
                    t.permute([1, 2, 0])
                        .reshape([s[0], s[1], modes[0], modes[1]])
                };
                or = or.slice_assign(rg.clone(), back(ac.clone() - bd.clone()));
                oi = oi.slice_assign(rg, back(abcd - ac - bd));
            }
            inv(or, oi)
        },
    );
    cmp(
        &format!("{s:?} 5 slice spectrum only + inverse"),
        x,
        move |x| {
            let d = x.device();
            let (r, i) = fwd(x);
            let rg = [0..s[0], 0..s[1], 0..modes[0], 0..modes[1]];
            let or = Tensor::<4>::zeros(spec, &d).slice_assign(rg.clone(), r.slice(rg.clone()));
            let oi = Tensor::<4>::zeros(spec, &d).slice_assign(rg.clone(), i.slice(rg));
            inv(or, oi)
        },
    );
}

#[test]
fn stages() {
    for s in [
        [2, 6, 8, 6],
        [2, 6, 11, 9],
        [2, 6, 17, 15],
        [2, 6, 16, 16],
        [2, 6, 17, 16],
        [2, 6, 16, 15],
        [2, 6, 15, 17],
        [2, 1, 17, 15],
    ] {
        stages2(s, [3, 2]);
    }
}

#[test]
fn cfft_alone() {
    macro_rules! go {
        ($s:expr, $axis:expr) => {{
            let s = $s;
            let ax = $axis;
            cmp(
                &format!("cfft ax{ax} n={} {s:?}", s[ax]),
                rnd(s),
                move |x| {
                    let (r, i) = signal::cfft(x.clone(), x.mul_scalar(0.5), ax, Some(s[ax]));
                    Tensor::cat(vec![r, i], s.len() - 1)
                },
            );
        }};
    }
    go!([17, 8], 0);
    go!([17, 1], 0);
    go!([17, 2], 0);
    go!([17, 4], 0);
    go!([17, 7], 0);
    go!([17, 9], 0);
    go!([17, 16], 0);
    go!([2, 6, 17, 8], 2);
    go!([2, 1, 17, 8], 2);
    go!([2, 1, 17, 15], 2);
    go!([2, 1, 15, 9], 2);
    go!([2, 1, 6, 4], 2);
    go!([2, 1, 6, 8], 2);
    go!([2, 1, 12, 4], 2);
    go!([2, 1, 5, 4], 2);
    go!([2, 1, 7, 4], 2);
    go!([2, 1, 3, 2], 2);
    go!([2, 6, 6, 8, 4], 2);
    go!([2, 6, 6, 8, 4], 3);
    go!([2, 6, 4, 6, 3], 2);
    go!([2, 6, 4, 6, 3], 3);
    go!([2, 6, 8, 8, 5], 2);
    go!([2, 6, 8, 8, 5], 3);
    go!([1, 33, 17], 1);
    go!([1, 41, 21], 1);
    go!([1, 64, 33], 1);
}

#[test]
fn sweep_sizes() {
    let f = Device::flex();
    for n in [64usize, 68, 72, 80, 66] {
        let h = n / 2 + 1;
        cmp(
            &format!("cfft ax2 n={n} [2,8,{n},{h}]"),
            rnd([2, 8, n, h]),
            move |x| {
                let (r, i) = signal::cfft(x.clone(), x.mul_scalar(0.5), 2, Some(n));
                Tensor::cat(vec![r, i], 3)
            },
        );
        let sc = SpectralConv::<4>::new(&f, 8, 8, &[12, 12]);
        let sm = sc.clone().to_device(&metal());
        let x = rnd([2, 8, n, n]);
        let a = v(sc.forward(x.clone()));
        let b = v(sm.forward(x.to_device(&metal())));
        println!(
            "{:<48} metal/flex {:.2e}",
            format!("SpectralConv<4> {n}x{n} modes 12"),
            rel(&b, &a)
        );
    }
    for p in [None, Some(4), Some(8), Some(16), Some(2)] {
        let m = FNOConfig::new(vec![12, 12], 1, 1)
            .with_hidden_channels(8)
            .with_n_layers(2)
            .with_padding(p)
            .init::<4>(&f);
        let mm = m.clone().to_device(&metal());
        let x = rnd([2, 64, 64, 1]);
        let a = v(m.forward(x.clone()));
        let b = v(mm.forward(x.to_device(&metal())));
        println!(
            "{:<48} metal/flex {:.2e}",
            format!("FNO<4> 64x64 pad {p:?}"),
            rel(&b, &a)
        );
    }
}

/// Both backends vs a naive f64 DFT along axis 0 of a [n, w] tensor.
#[test]
fn vs_naive_dft() {
    for (n, w) in [(17usize, 8usize), (17, 7), (6, 4), (16, 8)] {
        let x = rnd([n, w]);
        let xr = v(x.clone());
        let xi: Vec<f32> = xr.iter().map(|a| a * 0.5).collect();
        let mut want = vec![0.0f32; 2 * n * w];
        for k in 0..n {
            for c in 0..w {
                let (mut sr, mut si) = (0.0f64, 0.0f64);
                for j in 0..n {
                    let t = -2.0 * std::f64::consts::PI * (j * k) as f64 / n as f64;
                    let (a, b) = (xr[j * w + c] as f64, xi[j * w + c] as f64);
                    sr += a * t.cos() - b * t.sin();
                    si += a * t.sin() + b * t.cos();
                }
                want[k * 2 * w + c] = sr as f32;
                want[k * 2 * w + w + c] = si as f32;
            }
        }
        let run = |d: &Device| {
            let x = x.clone().to_device(d);
            let (r, i) = signal::cfft(x.clone(), x.mul_scalar(0.5), 0, Some(n));
            v(Tensor::cat(vec![r, i], 1))
        };
        println!(
            "cfft [{n}, {w}] ax0   flex/naive {:.2e}   metal/naive {:.2e}",
            rel(&run(&Device::flex()), &want),
            rel(&run(&metal()), &want)
        );
    }
}

/// burn-signal's bluestein_dft + cfft(pow2) + hermitian_extend, inlined
/// stage by stage for a real [n, w] input along axis 0 (the rfft path).
fn bluestein_stages(x: Tensor<2>, d: &Device) -> Vec<(&'static str, Vec<f32>)> {
    let x = x.to_device(d);
    let n = x.dims()[0];
    let w = x.dims()[1];
    let m = (2 * n - 1).next_power_of_two();
    let mut out = Vec::new();
    let phase = |k: usize| {
        let r = (k as u128 * k as u128 % (2 * n as u128)) as f64;
        let a = std::f64::consts::PI * r / n as f64;
        (a.cos(), -a.sin())
    };
    let col = |vals: Vec<f64>, len: usize| {
        let vals: Vec<f32> = vals.into_iter().map(|v| v as f32).collect();
        Tensor::<1>::from_floats(vals.as_slice(), d).reshape([len, 1])
    };
    let (wr, wi): (Vec<f64>, Vec<f64>) = (0..n).map(phase).unzip();
    let (w_re, w_im) = (col(wr, n), col(wi, n));
    let mut br = vec![0.0; m];
    let mut bi = vec![0.0; m];
    for k in 0..n {
        let (c, s) = phase(k);
        br[k] = c;
        bi[k] = -s;
        if k > 0 {
            br[m - k] = c;
            bi[m - k] = -s;
        }
    }
    let (b_re, b_im) = (col(br, m), col(bi, m));

    let re = x.clone();
    let im = x.zeros_like();
    let a_re = re.clone() * w_re.clone() - im.clone() * w_im.clone();
    let a_im = re * w_im.clone() + im * w_re.clone();
    out.push(("a = x*w", v(a_re.clone())));
    let pad = |t: Tensor<2>| t.pad([(0, m - n), (0, 0)], PadMode::Constant(0.0));
    let (a_re, a_im) = (pad(a_re), pad(a_im));
    out.push(("a padded to m", v(a_re.clone())));

    // cfft(a, m) = rfft(a_re) & rfft(a_im) (backend, pow2) + hermitian_extend
    let (xr, xi) = signal::rfft(a_re.clone(), 0, Some(m));
    out.push(("backend rfft(a_re) re [pad input]", v(xr.clone())));
    out.push(("backend rfft(a_re) im [pad input]", v(xi.clone())));
    let (cr, _) = signal::rfft(a_re.clone() + 0.0, 0, Some(m));
    out.push(("backend rfft(a_re + 0.0)", v(cr)));
    let fresh = Tensor::<2>::from_floats(TensorData::new(v(a_re.clone()), [m, w]), d);
    let (fr, _) = signal::rfft(fresh, 0, Some(m));
    out.push(("backend rfft(a_re from host)", v(fr)));
    let (yr, yi) = signal::rfft(a_im, 0, Some(m));
    let ext = |hr: Tensor<2>, hi: Tensor<2>| {
        let half = hr.dims()[0];
        let ml = m - half;
        let mr = hr.clone().narrow(0, 1, ml).flip([0]);
        let mi = hi.clone().narrow(0, 1, ml).flip([0]).neg();
        (Tensor::cat(vec![hr, mr], 0), Tensor::cat(vec![hi, mi], 0))
    };
    let (xr, xi) = ext(xr, xi);
    out.push(("hermitian_extend(rfft a_re)", v(xr.clone())));
    let (yr, yi) = ext(yr, yi);
    let (fa_re, fa_im) = (xr - yi.clone(), xi + yr);
    out.push(("fa = cfft(a)", v(fa_re.clone())));
    let (fb_re, fb_im) = signal::cfft(b_re, b_im, 0, Some(m));
    out.push(("fb = cfft(b) [m,1]", v(fb_re.clone())));
    let conv_re = fa_re.clone() * fb_re.clone() - fa_im.clone() * fb_im.clone();
    let conv_im = fa_re * fb_im + fa_im * fb_re;
    out.push(("conv = fa*fb", v(conv_re.clone())));
    let (ir, ii) = signal::cfft(conv_re, conv_im.neg(), 0, Some(m));
    out.push(("cfft(conj conv)", v(ir.clone())));
    let s = 1.0 / m as f64;
    let y_re = ir.mul_scalar(s).narrow(0, 0, n);
    let y_im = ii.neg().mul_scalar(s).narrow(0, 0, n);
    let xr = y_re.clone() * w_re.clone() - y_im.clone() * w_im.clone();
    out.push(("X = y*w (bluestein out)", v(xr)));
    let _ = y_im * w_re;
    let (r, _) = signal::rfft(x, 0, Some(n));
    out.push(("signal::rfft(x, ax0, n) real", v(r)));
    out
}

#[test]
fn bluestein_bisect() {
    for (n, w) in [(17usize, 8usize), (17, 7), (6, 4)] {
        println!(
            "--- [{n}, {w}] axis 0, m = {}",
            (2 * n - 1).next_power_of_two()
        );
        let x = rnd([n, w]);
        let f = bluestein_stages(x.clone(), &Device::flex());
        let m = bluestein_stages(x, &metal());
        for ((name, a), (_, b)) in f.iter().zip(&m) {
            let e = rel(b, a);
            println!(
                "  {name:<40} metal/flex {e:.2e}{}",
                if e > 1e-5 { "  <-- BAD" } else { "" }
            );
        }
    }
}

/// Same pipeline, one output read; `cuts` lists the stage boundaries at
/// which the tensor is round-tripped through the host (breaking fusion).
fn bluestein_cut(x: Tensor<2>, d: &Device, cuts: &[u8]) -> Vec<f32> {
    let cut = |t: Tensor<2>, at: u8| {
        if cuts.contains(&at) {
            let dims = t.dims();
            Tensor::<2>::from_floats(TensorData::new(v(t), dims), d)
        } else {
            t
        }
    };
    let x = x.to_device(d);
    let n = x.dims()[0];
    let m = (2 * n - 1).next_power_of_two();
    let phase = |k: usize| {
        let r = (k as u128 * k as u128 % (2 * n as u128)) as f64;
        let a = std::f64::consts::PI * r / n as f64;
        (a.cos(), -a.sin())
    };
    let col = |vals: Vec<f64>, len: usize| {
        let vals: Vec<f32> = vals.into_iter().map(|v| v as f32).collect();
        Tensor::<1>::from_floats(vals.as_slice(), d).reshape([len, 1])
    };
    let (wr, wi): (Vec<f64>, Vec<f64>) = (0..n).map(phase).unzip();
    let (w_re, w_im) = (col(wr, n), col(wi, n));
    let (w_re, w_im) = (cut(w_re, 9), cut(w_im, 9));
    let mut br = vec![0.0; m];
    let mut bi = vec![0.0; m];
    for k in 0..n {
        let (c, s) = phase(k);
        br[k] = c;
        bi[k] = -s;
        if k > 0 {
            br[m - k] = c;
            bi[m - k] = -s;
        }
    }
    let (b_re, b_im) = (col(br, m), col(bi, m));
    let (b_re, b_im) = (cut(b_re, 8), cut(b_im, 8));
    let re = x.clone();
    let im = x.zeros_like();
    let a_re = re.clone() * w_re.clone() - im.clone() * w_im.clone();
    let a_im = re * w_im.clone() + im * w_re.clone();
    let (a_re, a_im) = (cut(a_re, 1), cut(a_im, 1));
    let pad = |t: Tensor<2>| t.pad([(0, m - n), (0, 0)], PadMode::Constant(0.0));
    let (a_re, a_im) = (cut(pad(a_re), 2), cut(pad(a_im), 2));
    let (fa_re, fa_im) = signal::cfft(a_re, a_im, 0, Some(m));
    let (fa_re, fa_im) = (cut(fa_re, 3), cut(fa_im, 3));
    let (fb_re, fb_im) = signal::cfft(b_re, b_im, 0, Some(m));
    let (fb_re, fb_im) = (cut(fb_re, 7), cut(fb_im, 7));
    let conv_re = fa_re.clone() * fb_re.clone() - fa_im.clone() * fb_im.clone();
    let conv_im = fa_re * fb_im + fa_im * fb_re;
    let (conv_re, conv_im) = (cut(conv_re, 4), cut(conv_im, 4));
    let (ir, ii) = signal::cfft(conv_re, conv_im.neg(), 0, Some(m));
    let (ir, ii) = (cut(ir, 5), cut(ii, 5));
    let s = 1.0 / m as f64;
    let y_re = cut(ir.mul_scalar(s).narrow(0, 0, n), 6);
    let y_im = cut(ii.neg().mul_scalar(s).narrow(0, 0, n), 6);
    v(y_re * w_re - y_im * w_im)
}

#[test]
fn bluestein_fusion() {
    for (n, w) in [(17usize, 8usize), (6, 4), (17, 7)] {
        println!("--- [{n}, {w}] axis 0");
        let x = rnd([n, w]);
        let want = bluestein_cut(x.clone(), &Device::flex(), &[]);
        let cases: [(&str, &[u8]); 13] = [
            ("no cuts", &[]),
            ("all cuts", &[1, 2, 3, 4, 5, 6]),
            ("all cuts + 7,8,9", &[1, 2, 3, 4, 5, 6, 7, 8, 9]),
            ("cut 7 (after cfft(b))", &[7]),
            ("cut 8 (b from host, reshaped)", &[8]),
            ("cut 9 (w from host, reshaped)", &[9]),
            ("cut 1 (after a=x*w)", &[1]),
            ("cut 2 (after pad)", &[2]),
            ("cut 3 (after cfft(a))", &[3]),
            ("cut 4 (after conv)", &[4]),
            ("cut 5 (after inverse cfft)", &[5]),
            ("cut 6 (after narrow)", &[6]),
            ("cuts 1+2", &[1, 2]),
        ];
        for (name, cuts) in cases {
            let e = rel(&bluestein_cut(x.clone(), &metal(), cuts), &want);
            println!(
                "  {name:<32} metal/flex {e:.2e}{}",
                if e > 1e-5 { "  <-- BAD" } else { "" }
            );
        }
    }
}

#[test]
fn pow2_non_last() {
    let host = |x: &Tensor<2>, d: &Device| {
        let dims = x.dims();
        Tensor::<2>::from_floats(TensorData::new(v(x.clone()), dims), d)
    };
    for (n, w) in [
        (16usize, 8usize),
        (32, 8),
        (64, 8),
        (64, 7),
        (64, 4),
        (64, 2),
        (64, 1),
        (128, 8),
        (16, 16),
        (64, 16),
    ] {
        let x = rnd([n, w]);
        let y = rnd([n, w]);
        let run_c = |d: &Device| {
            let (r, i) = signal::cfft(host(&x, d), host(&y, d), 0, Some(n));
            v(Tensor::cat(vec![r, i], 1))
        };
        let run_r = |d: &Device| {
            let (r, i) = signal::rfft(host(&x, d), 0, Some(n));
            v(Tensor::cat(vec![r, i], 1))
        };
        let run_r_sync = |d: &Device| {
            let (r, i) = signal::rfft(host(&x, d), 0, Some(n));
            let (r, i) = (host(&r, d), host(&i, d));
            v(Tensor::cat(vec![r, i], 1))
        };
        let ec = rel(&run_c(&metal()), &run_c(&Device::flex()));
        let er = rel(&run_r(&metal()), &run_r(&Device::flex()));
        let es = rel(&run_r_sync(&metal()), &run_r_sync(&Device::flex()));
        let bad = |e: f32| if e > 1e-5 { " BAD" } else { "" };
        println!(
            "[{n:>3}, {w:>2}] ax0   cfft {ec:.2e}{}   rfft {er:.2e}{}   rfft+read {es:.2e}{}",
            bad(ec),
            bad(er),
            bad(es)
        );
    }
}

/// a[n, w] * b[n, 1] (broadcast over w), with b produced by different ops.
#[test]
fn broadcast_mul() {
    let host = |t: &Tensor<2>, d: &Device| {
        let dims = t.dims();
        Tensor::<2>::from_floats(TensorData::new(v(t.clone()), dims), d)
    };
    let n = 64;
    for w in [8usize, 7, 4, 2, 1, 16] {
        let a = rnd([n, w]);
        let b0 = rnd([n, 1]);
        type Make = fn(Tensor<2>) -> Tensor<2>;
        let makers: [(&str, Make); 9] = [
            ("host [n,1]", |b| b),
            ("b + 0", |b| b + 0.0),
            ("flip", |b| b.flip([0])),
            ("narrow of [2n,1]", |b| {
                Tensor::cat(vec![b.clone(), b], 0).narrow(0, 0, 64)
            }),
            ("cat(half, flipped half)", |b| {
                Tensor::cat(
                    vec![b.clone().narrow(0, 0, 32), b.narrow(0, 32, 32).flip([0])],
                    0,
                )
            }),
            ("rfft ax0 re (pow2, [n/2+1,1]) cat", |b| {
                let (r, _) = signal::rfft(b.clone(), 0, Some(64));
                Tensor::cat(vec![r, b.narrow(0, 0, 31)], 0)
            }),
            ("cfft ax0 re", |b| {
                signal::cfft(b.clone(), b.mul_scalar(0.5), 0, Some(64)).0
            }),
            ("1D reshape [n]->[n,1]", |b| {
                b.reshape([64]).reshape([64, 1])
            }),
            ("swap_dims of [1,n]", |b| b.reshape([1, 64]).swap_dims(0, 1)),
        ];
        for (name, make) in makers {
            let run = |d: &Device| {
                let b = make(host(&b0, d));
                v(host(&a, d) * b)
            };
            let run_sync = |d: &Device| {
                let b = host(&make(host(&b0, d)), d);
                v(host(&a, d) * b)
            };
            let e = rel(&run(&metal()), &run(&Device::flex()));
            let es = rel(&run_sync(&metal()), &run_sync(&Device::flex()));
            println!(
                "w={w:<2} b = {name:<36} a*b {e:.2e}{}   (b materialised {es:.2e})",
                if e > 1e-5 { " BAD" } else { "" }
            );
        }
    }
}

#[test]
fn broadcast_cat_min() {
    let host = |t: &Tensor<2>, d: &Device| {
        let dims = t.dims();
        Tensor::<2>::from_floats(TensorData::new(v(t.clone()), dims), d)
    };
    let a = rnd([4, 2]);
    let p = rnd([2, 1]);
    let q = rnd([2, 1]);
    type Op = fn(Tensor<2>, Tensor<2>, Tensor<2>) -> Tensor<2>;
    let ops: [(&str, Op); 7] = [
        ("a * cat(p, q)", |a, p, q| a * Tensor::cat(vec![p, q], 0)),
        ("a + cat(p, q)", |a, p, q| a + Tensor::cat(vec![p, q], 0)),
        ("cat(p, q) * a", |a, p, q| Tensor::cat(vec![p, q], 0) * a),
        ("a * cat(p, q).expand([4,2])", |a, p, q| {
            a * Tensor::cat(vec![p, q], 0).expand([4, 2])
        }),
        ("a * (cat(p, q) + 0)", |a, p, q| {
            a * (Tensor::cat(vec![p, q], 0) + 0.0)
        }),
        ("a * cat(p, q) along ax1 [1,4]^T", |a, p, q| {
            a * Tensor::cat(vec![p.reshape([1, 2]), q.reshape([1, 2])], 1).reshape([4, 1])
        }),
        ("cat(p, q) alone, expand([4,2])", |_, p, q| {
            Tensor::cat(vec![p, q], 0).expand([4, 2])
        }),
    ];
    for (name, op) in ops {
        let run = |d: &Device| v(op(host(&a, d), host(&p, d), host(&q, d)));
        let (m, f) = (run(&metal()), run(&Device::flex()));
        let e = rel(&m, &f);
        println!("{name:<36} {e:.2e}{}", if e > 1e-5 { " BAD" } else { "" });
        if e > 1e-5 {
            println!("    flex  {f:?}\n    metal {m:?}");
        }
    }
    let (pv, qv) = (v(p), v(q));
    println!("    a = {:?}\n    p = {pv:?}  q = {qv:?}", v(a));
}

/// Bug-report repro: fixed values, asserts.
#[test]
fn repro_fused_cat_broadcast() {
    let device = Device::metal(Default::default());
    let a = Tensor::<2>::from_floats([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], &device);
    let p = Tensor::<2>::from_floats([[10.0], [20.0]], &device);
    let q = Tensor::<2>::from_floats([[30.0], [40.0]], &device);

    let out = a * Tensor::cat(vec![p, q], 0); // [4, 2] * [4, 1] -> [4, 2]
    let got = out.into_data().try_to_vec::<f32>().unwrap();
    println!("got {got:?}");
    assert_eq!(got, [10.0, 20.0, 60.0, 80.0, 150.0, 180.0, 280.0, 320.0]);
}

/// Bug-report repro: Bluestein cfft along a non-last axis vs a naive DFT.
#[test]
fn repro_cfft_bluestein() {
    let device = Device::metal(Default::default());
    let (n, w) = (6usize, 4usize);
    let vals: Vec<f32> = (0..n * w).map(|k| ((k * 7 % 11) as f32) - 5.0).collect();
    let re = Tensor::<2>::from_data(TensorData::new(vals.clone(), [n, w]), &device);
    let im = re.zeros_like();
    let (out_re, _) = signal::cfft(re, im, 0, Some(n));
    let got = out_re.into_data().try_to_vec::<f32>().unwrap();
    let mut max_err = 0.0f32;
    for k in 0..n {
        for c in 0..w {
            let want: f64 = (0..n)
                .map(|j| {
                    vals[j * w + c] as f64
                        * (2.0 * std::f64::consts::PI * (j * k) as f64 / n as f64).cos()
                })
                .sum();
            max_err = max_err.max((got[k * w + c] - want as f32).abs());
        }
    }
    println!("max abs err vs naive DFT: {max_err}");
    assert!(
        max_err < 1e-3,
        "cfft [6, 4] axis 0 wrong: max abs err {max_err}"
    );
}
/// Bug-report repro: fixed values, asserts.
#[test]
fn repro_fused_cat_broadcast_flex() {
    let device = Device::flex();
    let a = Tensor::<2>::from_floats([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], &device);
    let p = Tensor::<2>::from_floats([[10.0], [20.0]], &device);
    let q = Tensor::<2>::from_floats([[30.0], [40.0]], &device);

    let out = a * Tensor::cat(vec![p, q], 0); // [4, 2] * [4, 1] -> [4, 2]
    let got = out.into_data().try_to_vec::<f32>().unwrap();
    println!("got {got:?}");
    assert_eq!(got, [10.0, 20.0, 60.0, 80.0, 150.0, 180.0, 280.0, 320.0]);
}

/// Bug-report repro: Bluestein cfft along a non-last axis vs a naive DFT.
#[test]
fn repro_cfft_bluestein_flex() {
    let device = Device::flex();
    let (n, w) = (6usize, 4usize);
    let vals: Vec<f32> = (0..n * w).map(|k| ((k * 7 % 11) as f32) - 5.0).collect();
    let re = Tensor::<2>::from_data(TensorData::new(vals.clone(), [n, w]), &device);
    let im = re.zeros_like();
    let (out_re, _) = signal::cfft(re, im, 0, Some(n));
    let got = out_re.into_data().try_to_vec::<f32>().unwrap();
    let mut max_err = 0.0f32;
    for k in 0..n {
        for c in 0..w {
            let want: f64 = (0..n)
                .map(|j| {
                    vals[j * w + c] as f64
                        * (2.0 * std::f64::consts::PI * (j * k) as f64 / n as f64).cos()
                })
                .sum();
            max_err = max_err.max((got[k * w + c] - want as f32).abs());
        }
    }
    println!("max abs err vs naive DFT: {max_err}");
    assert!(
        max_err < 1e-3,
        "cfft [6, 4] axis 0 wrong: max abs err {max_err}"
    );
}

/// Flex timing: cfft on axis 2 directly (old) vs swap to last + back (new).
#[test]
fn flex_swap_timing() {
    let f = Device::flex();
    for n in [64usize, 72] {
        let h = n / 2 + 1;
        let x = Tensor::<4>::random([20, 32, n, h], Distribution::Normal(0.0, 1.0), &f);
        let y = x.clone().mul_scalar(0.5);
        let time = |new: bool| {
            let mut best = f64::MAX;
            for _ in 0..3 {
                let t = std::time::Instant::now();
                let (r, _) = if new {
                    let (r, i) = signal::cfft(
                        x.clone().swap_dims(2, 3),
                        y.clone().swap_dims(2, 3),
                        3,
                        Some(n),
                    );
                    (r.swap_dims(2, 3), i.swap_dims(2, 3))
                } else {
                    signal::cfft(x.clone(), y.clone(), 2, Some(n))
                };
                let _ = v(r);
                best = best.min(t.elapsed().as_secs_f64());
            }
            best
        };
        let (old, new) = (time(false), time(true));
        println!(
            "flex cfft [20,32,{n},{h}] ax2: old {old:.3}s  new {new:.3}s  ratio {:.2}",
            new / old
        );
    }
}

/// Learner-test triage: tiny 1D FNO forward and LpLoss, Metal vs Flex.
#[test]
fn learner_triage() {
    use sciml_rs::neural_operators::losses::data_losses::{LpLoss, Reduction};
    let f = Device::flex();
    let model = FNOConfig::new(vec![2], 1, 1)
        .with_hidden_channels(4)
        .with_n_layers(1)
        .init::<3>(&f);
    let mm = model.clone().to_device(&metal());
    let (b, s) = (3, 8);
    let xin: Vec<f32> = (0..b * s).map(|i| (i as f32 * 0.37).sin()).collect();
    let tin: Vec<f32> = (0..b * s).map(|i| 2.0 + (i as f32 * 0.7).cos()).collect();
    let x = |d: &Device| Tensor::<3>::from_data(TensorData::new(xin.clone(), [b, s, 1]), d);
    let t = |d: &Device| Tensor::<2>::from_data(TensorData::new(tin.clone(), [b, s]), d);
    let m = metal();

    let of = v(model.forward(x(&f)));
    let o1 = v(mm.forward(x(&m)));
    let o2 = v(mm.forward(x(&m)));
    println!(
        "forward       metal/flex {:.2e}  metal/metal {:.2e}",
        rel(&o1, &of),
        rel(&o1, &o2)
    );

    let loss = LpLoss::new(1, 2, Reduction::Sum);
    let lf = loss
        .rel(model.forward(x(&f)).reshape([b, s]), t(&f))
        .into_scalar::<f32>();
    let l_sep = loss
        .rel(mm.forward(x(&m)).reshape([b, s]), t(&m))
        .into_scalar::<f32>();
    let l_div = loss
        .rel(mm.forward(x(&m)).reshape([b, s]), t(&m))
        .div_scalar(3.0)
        .into_scalar::<f32>()
        * 3.0;
    // Loss on a host-materialised forward output: isolates LpLoss.
    let o_host = Tensor::<2>::from_data(TensorData::new(o1.clone(), [b, s]), &m);
    let l_host = loss.rel(o_host, t(&m)).into_scalar::<f32>();
    let o_host_f = Tensor::<2>::from_data(TensorData::new(of.clone(), [b, s]), &f);
    let l_host_f = loss.rel(o_host_f, t(&f)).into_scalar::<f32>();
    println!("loss flex {lf}  flex(host out) {l_host_f}");
    println!("loss metal fused {l_sep}  fused+div {l_div}  metal(host out) {l_host}");

    // Per-sample norms on metal from host values, vs flex.
    let diff = |d: &Device, o: &[f32]| {
        let o = Tensor::<2>::from_data(TensorData::new(o.to_vec(), [b, s]), d);
        v(o - t(d)).to_vec()
    };
    println!(
        "diff metal/flex {:.2e}",
        rel(&diff(&m, &of), &diff(&f, &of))
    );
    let norms = |d: &Device| {
        let y = t(d);
        v(y.clone().powf_scalar(2.0).sum_dim(1))
    };
    println!(
        "||y||^2 per row  flex {:?}  metal {:?}",
        norms(&f),
        norms(&m)
    );
}

/// Same as the learner test: model initialised on Metal itself.
#[test]
fn learner_triage_init_on_metal() {
    let m = metal();
    for trial in 0..3 {
        m.seed(7);
        let model = FNOConfig::new(vec![2], 1, 1)
            .with_hidden_channels(4)
            .with_n_layers(1)
            .init::<3>(&m);
        let (b, s) = (3, 8);
        let xin: Vec<f32> = (0..b * s).map(|i| (i as f32 * 0.37).sin()).collect();
        let x = || Tensor::<3>::from_data(TensorData::new(xin.clone(), [b, s, 1]), &m);
        let o1 = v(model.forward(x()));
        let o2 = v(model.forward(x()));
        let o3 = v(model.forward(x()));
        // Same weights on Flex.
        let mf = model.clone().to_device(&Device::flex());
        let of = v(mf.forward(Tensor::<3>::from_data(
            TensorData::new(xin.clone(), [b, s, 1]),
            &Device::flex(),
        )));
        println!(
            "trial {trial}: run1/run2 {:.2e}  run2/run3 {:.2e}  run3/flex {:.2e}",
            rel(&o1, &o2),
            rel(&o2, &o3),
            rel(&o3, &of)
        );
    }
}

/// A different-shape 1D FNO forward first, then the tiny learner model.
#[test]
fn learner_triage_after_warmup() {
    let m = metal();
    let f = Device::flex();
    // Warm-up, like padded_forward_matches_reference_1d: FNO<3>, [2, 16, 1], p = 5.
    let warm = FNOConfig::new(vec![4], 1, 1)
        .with_hidden_channels(6)
        .with_n_layers(2)
        .with_padding(Some(5))
        .init::<3>(&m);
    for _ in 0..3 {
        let _ = v(warm.forward(Tensor::<3>::random(
            [2, 16, 1],
            Distribution::Normal(0.0, 1.0),
            &m,
        )));
    }
    m.seed(7);
    let model = FNOConfig::new(vec![2], 1, 1)
        .with_hidden_channels(4)
        .with_n_layers(1)
        .init::<3>(&m);
    let (b, s) = (3, 8);
    let xin: Vec<f32> = (0..b * s).map(|i| (i as f32 * 0.37).sin()).collect();
    let mf = model.clone().to_device(&f);
    let of = v(mf.forward(Tensor::<3>::from_data(
        TensorData::new(xin.clone(), [b, s, 1]),
        &f,
    )));
    for run in 0..5 {
        let o = v(model.forward(Tensor::<3>::from_data(
            TensorData::new(xin.clone(), [b, s, 1]),
            &m,
        )));
        println!("run {run}: metal/flex {:.2e}", rel(&o, &of));
    }
}

/// FNO::forward's grid step: cat([x, arange.reshape.expand], last) -> permute
/// -> pointwise matmul. Metal vs Flex, several runs, grid broadcast or not.
fn grid_chain(d: &Device, expand: bool, b: usize, n: usize) -> Vec<f32> {
    let xin: Vec<f32> = (0..b * n).map(|i| (i as f32 * 0.37).sin()).collect();
    let x = Tensor::<3>::from_data(TensorData::new(xin, [b, n, 1]), d);
    let coords = Tensor::<1, Int>::arange(0..n as i64, d)
        .float()
        .div_scalar((n - 1) as f64);
    let g = coords.reshape([1, n, 1]);
    let g = if expand {
        g.expand([b, n, 1])
    } else {
        let gv = v(g);
        let rep: Vec<f32> = (0..b).flat_map(|_| gv.clone()).collect();
        Tensor::<3>::from_data(TensorData::new(rep, [b, n, 1]), d)
    };
    let c = Tensor::cat(vec![x, g], 2); // [b, n, 2]
    let wv: Vec<f32> = (0..8).map(|i| 0.3 + 0.1 * i as f32).collect();
    let w = Tensor::<2>::from_data(TensorData::new(wv, [4, 2]), d);
    let y = w.unsqueeze::<3>().matmul(c.permute([0, 2, 1])); // [1,4,2] @ [b,2,n]
    v(y)
}

#[test]
fn grid_cat_expand() {
    for (b, n) in [(3usize, 8usize), (2, 16), (1, 8)] {
        for expand in [true, false] {
            let f = grid_chain(&Device::flex(), expand, b, n);
            let errs: Vec<String> = (0..6)
                .map(|_| format!("{:.1e}", rel(&grid_chain(&metal(), expand, b, n), &f)))
                .collect();
            println!(
                "b={b} n={n:<2} expand={expand:<5} metal/flex runs: {}",
                errs.join(" ")
            );
        }
    }
}

/// Does cloning to Flex before any Metal use copy the Metal weights?
#[test]
fn lazy_init_clone_check() {
    use burn::module::Module;
    let m = metal();
    m.seed(7);
    let model = FNOConfig::new(vec![2], 1, 1)
        .with_hidden_channels(4)
        .with_n_layers(1)
        .init::<3>(&m);
    let mf = model.clone().to_device(&Device::flex());
    let flat = |md: &sciml_rs::neural_operators::models::fno::FNO<3>| -> Vec<f32> {
        let rec = md.clone().into_record();
        let _ = rec;
        let mut out = Vec::new();
        md.clone().map(&mut Collect(&mut out));
        out
    };
    struct Collect<'a>(&'a mut Vec<f32>);
    impl burn::module::ModuleMapper for Collect<'_> {
        fn map_float<const D: usize>(
            &mut self,
            param: burn::module::Param<Tensor<D>>,
        ) -> burn::module::Param<Tensor<D>> {
            self.0.extend(v(param.val()));
            param
        }
    }
    let a = flat(&model);
    let b = flat(&mf);
    println!(
        "params metal vs flex-clone: n={} rel {:.2e}",
        a.len(),
        rel(&a, &b)
    );
}

/// After a 1D FNO warm-up on Metal, check the tiny model's building blocks.
#[test]
fn after_warmup_blocks() {
    let m = metal();
    let f = Device::flex();
    let warm = FNOConfig::new(vec![4], 1, 1)
        .with_hidden_channels(6)
        .with_n_layers(2)
        .with_padding(Some(5))
        .init::<3>(&m);
    for _ in 0..3 {
        let _ = v(warm.forward(Tensor::<3>::random(
            [2, 16, 1],
            Distribution::Normal(0.0, 1.0),
            &m,
        )));
    }
    let report = |name: &str, run: &dyn Fn(&Device) -> Vec<f32>| {
        let want = run(&f);
        let errs: Vec<String> = (0..3)
            .map(|_| format!("{:.1e}", rel(&run(&m), &want)))
            .collect();
        println!("{name:<44} {}", errs.join(" "));
    };
    let h = rnd([3, 4, 8]);
    report("rfft [3,4,8] last", &|d| {
        let (r, i) = signal::rfft(h.clone().to_device(d), 2, Some(8));
        v(Tensor::cat(vec![r, i], 2))
    });
    let spec = rnd([3, 4, 5]);
    report("irfft [3,4,5]->8", &|d| {
        let s = spec.clone().to_device(d);
        v(signal::irfft(s.clone(), s.mul_scalar(0.5), 2, Some(8)))
    });
    let sc = SpectralConv::<3>::new(&f, 4, 4, &[2]);
    report("SpectralConv<3> 4->4 modes 2 [3,4,8]", &|d| {
        v(sc.clone().to_device(d).forward(h.clone().to_device(d)))
    });
    let lin: Linear = LinearConfig::new(2, 4).init(&f);
    let x2 = rnd([3, 8, 2]);
    report("Linear 2->4 on [3,8,2]", &|d| {
        v(lin.clone().to_device(d).forward(x2.clone().to_device(d)))
    });
    let conv = Conv1dConfig::new(4, 4, 1).init(&f);
    report("Conv1d k=1 [3,4,8]", &|d| {
        v(conv.clone().to_device(d).forward(h.clone().to_device(d)))
    });
    let w = rnd([4, 4]);
    report("pointwise [1,4,4]@[3,4,8]", &|d| {
        v(w.clone()
            .to_device(d)
            .unsqueeze::<3>()
            .matmul(h.clone().to_device(d)))
    });
    let tiny = FNOConfig::new(vec![2], 1, 1)
        .with_hidden_channels(4)
        .with_n_layers(1)
        .init::<3>(&f);
    let xin = rnd([3, 8, 1]);
    report("tiny FNO<3> forward", &|d| {
        v(tiny.clone().to_device(d).forward(xin.clone().to_device(d)))
    });
}

/// Pure-Burn 1D FNO-like forward: lift, (rfft -> keep modes -> irfft) + 1x1,
/// relu, project. `parts` toggles pieces for bisection.
#[derive(Clone)]
struct Toy {
    w0: Tensor<2>, // [H, C]
    b0: Tensor<1>, // [H]
    ws: Tensor<3>, // [M, H, H] spectral (real only)
    w1: Tensor<2>, // [H, H]
    w2: Tensor<2>, // [1, H]
    modes: usize,
}

fn toy_new(h: usize, c: usize, modes: usize, seed: u64) -> Toy {
    let f = Device::flex();
    f.seed(seed);
    let r = |s: Vec<usize>| {
        Tensor::<1>::random(
            [s.iter().product::<usize>()],
            Distribution::Normal(0.0, 0.5),
            &f,
        )
    };
    Toy {
        w0: r(vec![h, c]).reshape([h, c]),
        b0: r(vec![h]),
        ws: r(vec![modes, h, h]).reshape([modes, h, h]),
        w1: r(vec![h, h]).reshape([h, h]),
        w2: r(vec![1, h]).reshape([1, h]),
        modes,
    }
}

fn toy_to(t: &Toy, d: &Device) -> Toy {
    Toy {
        w0: t.w0.clone().to_device(d),
        b0: t.b0.clone().to_device(d),
        ws: t.ws.clone().to_device(d),
        w1: t.w1.clone().to_device(d),
        w2: t.w2.clone().to_device(d),
        modes: t.modes,
    }
}

/// x: [B, C, N] channels-first.
fn toy_forward(t: &Toy, x: Tensor<3>, parts: &str) -> Tensor<3> {
    let [b, _, n] = x.dims();
    let h = t.w0.dims()[0];
    let d = x.device();
    // lift: [1,H,C] @ [B,C,N] + bias
    let mut y = t.w0.clone().unsqueeze::<3>().matmul(x) + t.b0.clone().reshape([1, h, 1]);
    if parts.contains('s') {
        let (re, im) = signal::rfft(y.clone(), 2, Some(n));
        let k = n / 2 + 1;
        let m = t.modes;
        // per-mode channel mix: [M,B,H] @ [M,H,H]
        let mix = |z: Tensor<3>| {
            let z = z.slice([0..b, 0..h, 0..m]).permute([2, 0, 1]);
            z.matmul(t.ws.clone()).permute([1, 2, 0])
        };
        let (ore, oim) = (mix(re), mix(im));
        let zr = Tensor::<3>::zeros([b, h, k], &d).slice_assign([0..b, 0..h, 0..m], ore);
        let zi = Tensor::<3>::zeros([b, h, k], &d).slice_assign([0..b, 0..h, 0..m], oim);
        let spec = signal::irfft(zr, zi, 2, Some(n));
        let pw = t.w1.clone().unsqueeze::<3>().matmul(y.clone());
        y = spec + pw;
    } else {
        y = t.w1.clone().unsqueeze::<3>().matmul(y);
    }
    if parts.contains('r') {
        y = burn::tensor::activation::relu(y);
    }
    t.w2.clone().unsqueeze::<3>().matmul(y)
}

#[test]
fn toy_repeat() {
    let parts = std::env::var("PARTS").unwrap_or_else(|_| "sr".into());
    let m = metal();
    let warm = toy_to(&toy_new(6, 2, 4, 1), &m);
    for _ in 0..3 {
        let _ = v(toy_forward(&warm, rnd([2, 2, 21]).to_device(&m), &parts));
    }
    let toy = toy_new(4, 2, 2, 2);
    let x = rnd([3, 2, 8]);
    let want = v(toy_forward(&toy, x.clone(), &parts));
    let tm = toy_to(&toy, &m);
    let errs: Vec<String> = (0..5)
        .map(|_| {
            format!(
                "{:.1e}",
                rel(&v(toy_forward(&tm, x.clone().to_device(&m), &parts)), &want)
            )
        })
        .collect();
    println!("PARTS={parts}: {}", errs.join(" "));
}

/// Toy forward with FNO's real entry: channels-last x [B, N, 1], grid built
/// on device (arange/cast/div/reshape/expand), cat on the last axis, permute.
fn toy_forward_grid(t: &Toy, x: Tensor<3>, parts: &str) -> Tensor<3> {
    let [b, n, _] = x.dims();
    let d = x.device();
    let coords = Tensor::<1, Int>::arange(0..n as i64, &d)
        .float()
        .div_scalar((n - 1) as f64)
        .reshape([1, n, 1])
        .expand([b, n, 1]);
    let grid = Tensor::cat(vec![coords], 2);
    let xc = Tensor::cat(vec![x, grid], 2).permute([0, 2, 1]); // [B, 2, N]
    toy_forward(t, xc, parts).permute([0, 2, 1])
}

#[test]
fn toy_repeat_grid() {
    let parts = std::env::var("PARTS").unwrap_or_else(|_| "sr".into());
    let m = metal();
    let warm = toy_to(&toy_new(6, 2, 4, 1), &m);
    for _ in 0..3 {
        let _ = v(toy_forward_grid(
            &warm,
            rnd([2, 21, 1]).to_device(&m),
            &parts,
        ));
    }
    let toy = toy_new(4, 2, 2, 2);
    let x = rnd([3, 8, 1]);
    let want = v(toy_forward_grid(&toy, x.clone(), &parts));
    let tm = toy_to(&toy, &m);
    let errs: Vec<String> = (0..5)
        .map(|_| {
            format!(
                "{:.1e}",
                rel(
                    &v(toy_forward_grid(&tm, x.clone().to_device(&m), &parts)),
                    &want
                )
            )
        })
        .collect();
    println!("grid PARTS={parts}: {}", errs.join(" "));
}

/// Minimal: out = cat([x, grid], 2) with grid = cat([arange/(n-1) reshaped+expanded], 2),
/// optionally followed by permute. Variants via MIN env var.
fn min_chain(x: Tensor<3>, var: &str) -> Tensor<3> {
    let [b, n, _] = x.dims();
    let d = x.device();
    let coords = Tensor::<1, Int>::arange(0..n as i64, &d)
        .float()
        .div_scalar((n - 1) as f64)
        .reshape([1, n, 1])
        .expand([b, n, 1]);
    let grid = if var.contains("nocat1") {
        coords
    } else {
        Tensor::cat(vec![coords], 2)
    };
    let out = Tensor::cat(vec![x, grid], 2);
    if var.contains("perm") {
        out.permute([0, 2, 1])
    } else {
        out
    }
}

#[test]
fn min_repeat() {
    let var = std::env::var("MIN").unwrap_or_default();
    let m = metal();
    if !var.contains("nowarm") {
        for _ in 0..3 {
            let _ = v(min_chain(rnd([2, 21, 1]).to_device(&m), &var));
        }
    }
    let x = rnd([3, 8, 1]);
    let want = v(min_chain(x.clone(), &var));
    let errs: Vec<String> = (0..5)
        .map(|_| {
            format!(
                "{:.1e}",
                rel(&v(min_chain(x.clone().to_device(&m), &var)), &want)
            )
        })
        .collect();
    println!("MIN={var:<20} {}", errs.join(" "));
}

/// Bug-report repro 2: a fused cat chain gives wrong values when re-run at a
/// new shape with a freshly uploaded input.
#[test]
fn repro_fusion_cache_new_shape() {
    fn chain(x: Tensor<3>) -> Tensor<3> {
        let [b, n, _] = x.dims();
        let g = Tensor::<1, Int>::arange(0..n as i64, &x.device())
            .float()
            .reshape([1, n, 1])
            .expand([b, n, 1]);
        let g = Tensor::cat(vec![g], 2); // single-input cat
        Tensor::cat(vec![x, g], 2) // [b, n, 2]: (x, index) pairs
    }
    let (wb, wn, b, n) = match std::env::var("SHAPES").as_deref() {
        Ok("small") => (2, 5, 1, 4),
        Ok("mid") => (2, 5, 2, 4),
        _ => (2, 21, 3, 8),
    };
    let device = Device::metal(Default::default());
    let upload = |b: usize, n: usize| {
        let vals: Vec<f32> = (0..b * n).map(|i| 10.0 * (i + 1) as f32).collect();
        Tensor::<3>::from_data(TensorData::new(vals, [b, n, 1]), &device)
    };
    // 1) Run the chain a few times at one shape.
    let warms = std::env::var("WARMS")
        .ok()
        .and_then(|w| w.parse().ok())
        .unwrap_or(3);
    for _ in 0..warms {
        let _ = chain(upload(wb, wn)).into_data();
    }
    // 2) Run it at another shape, uploading the input afresh each call.
    let want: Vec<f32> = (0..b * n)
        .flat_map(|i| [10.0 * (i + 1) as f32, (i % n) as f32])
        .collect();
    let mut failed = None;
    for call in 0..3 {
        let got = chain(upload(b, n)).into_data().try_to_vec::<f32>().unwrap();
        println!("call {call}: {got:?}");
        if got != want && failed.is_none() {
            failed = Some(call);
        }
    }
    println!("want:   {want:?}");
    assert!(failed.is_none(), "first wrong call: {failed:?}");
}

fn chain_v(x: Tensor<3>, div: bool) -> Tensor<3> {
    let [b, n, _] = x.dims();
    let g = Tensor::<1, Int>::arange(0..n as i64, &x.device()).float();
    let g = if div { g.div_scalar((n - 1) as f64) } else { g };
    let g = Tensor::cat(vec![g.reshape([1, n, 1]).expand([b, n, 1])], 2);
    Tensor::cat(vec![x, g], 2)
}

#[test]
fn min_repeat_grid2() {
    let cfg = std::env::var("CFG").unwrap_or_default(); // e.g. "div,w3,b3n8,wb2n21"
    let div = cfg.contains("div");
    let warms = if cfg.contains("w3") { 3 } else { 1 };
    let (wb, wn) = if cfg.contains("wb2n21") {
        (2, 21)
    } else {
        (2, 5)
    };
    let (b, n) = if cfg.contains("b3n8") { (3, 8) } else { (1, 4) };
    let m = metal();
    for _ in 0..warms {
        let _ = v(chain_v(Tensor::<3>::zeros([wb, wn, 1], &m), div));
    }
    let xv: Vec<f32> = (0..b * n).map(|i| 10.0 * (i + 1) as f32).collect();
    let x = Tensor::<3>::from_data(TensorData::new(xv, [b, n, 1]), &m);
    let want = v(chain_v(x.clone().to_device(&Device::flex()), div));
    let grid_ch = |a: &[f32]| a.iter().skip(1).step_by(2).copied().collect::<Vec<f32>>();
    let errs: Vec<String> = (0..4)
        .map(|_| {
            let got = v(chain_v(x.clone(), div));
            let d = grid_ch(&got)
                .iter()
                .zip(grid_ch(&want))
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            format!("{d:.1e}")
        })
        .collect();
    println!("CFG={cfg:<24} grid max|err| {}", errs.join(" "));
}

/// Like min_repeat_grid2, but x enters by from_data each call (fresh upload)
/// or by to_device from Flex, as in min_repeat.
#[test]
fn min_repeat_grid3() {
    let cfg = std::env::var("CFG").unwrap_or_default();
    let m = metal();
    let f = Device::flex();
    let make = |b: usize, n: usize, d: &Device| -> Tensor<3> {
        let xv: Vec<f32> = (0..b * n).map(|i| 0.1 * (i + 1) as f32).collect();
        if cfg.contains("xfer") {
            Tensor::<3>::from_data(TensorData::new(xv, [b, n, 1]), &f).to_device(d)
        } else {
            Tensor::<3>::from_data(TensorData::new(xv, [b, n, 1]), d)
        }
    };
    let div = cfg.contains("div");
    for _ in 0..3 {
        let _ = v(chain_v(make(2, 21, &m), div));
    }
    let want = v(chain_v(make(3, 8, &f), div));
    let errs: Vec<String> = (0..4)
        .map(|_| {
            let got = v(chain_v(make(3, 8, &m), div));
            let d = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            format!("{d:.1e}")
        })
        .collect();
    println!("CFG={cfg:<12} max|err| {}", errs.join(" "));
}

/// Bug-report repro 2, as published.
#[test]
fn repro_fused_cat_reused_at_new_shape() {
    fn chain(x: Tensor<3>) -> Tensor<3> {
        let [b, n, _] = x.dims();
        let g = Tensor::<1, Int>::arange(0..n as i64, &x.device())
            .float()
            .reshape([1, n, 1])
            .expand([b, n, 1]);
        let g = Tensor::cat(vec![g], 2); // single-input cat
        Tensor::cat(vec![x, g], 2) // [b, n, 2]: (x, index) pairs
    }
    let device = Device::metal(Default::default());
    let upload = |b: usize, n: usize| {
        let vals: Vec<f32> = (0..b * n).map(|i| 10.0 * (i + 1) as f32).collect();
        Tensor::<3>::from_data(TensorData::new(vals, [b, n, 1]), &device)
    };

    // One run at [2, 21, 1].
    let _ = chain(upload(2, 21)).into_data();

    // Then three runs at [3, 8, 1], each with a freshly uploaded input.
    let want: Vec<f32> = (0..24)
        .flat_map(|i| [10.0 * (i + 1) as f32, (i % 8) as f32])
        .collect();
    for call in 0..3 {
        let got = chain(upload(3, 8)).into_data().try_to_vec::<f32>().unwrap();
        assert_eq!(got, want, "call {call}");
    }
}

/// Poisson-trainer-sized step (forward + backward) timing on Metal.
#[test]
fn bench_poisson_step() {
    let d = metal().autodiff();
    for p in [0usize, 8] {
        let model = FNOConfig::new(vec![12, 12], 1, 1)
            .with_hidden_channels(32)
            .with_n_layers(4)
            .with_padding(Some(p))
            .init::<4>(&d);
        let step = || {
            let x = Tensor::<4>::random([20, 64, 64, 1], Distribution::Normal(0.0, 1.0), &d);
            let y = model.forward(x);
            let loss = y.powf_scalar(2.0).mean();
            let g = loss.backward();
            let _ = g;
            d.sync().unwrap();
        };
        for _ in 0..3 {
            step();
        }
        let n = 10;
        let t = std::time::Instant::now();
        for _ in 0..n {
            step();
        }
        let per = t.elapsed().as_secs_f64() / n as f64;
        println!(
            "p={p:<2} step {per:.3}s  (~{:.0} s per 50-step epoch)",
            per * 50.0
        );
    }
}

/// Flex reference outputs, keyed by (call kind, batch size).
type RefCache = std::collections::HashMap<(&'static str, usize), (Vec<f32>, Vec<f32>)>;

/// 2D replay check at the Poisson trainer's configuration: shape changes as in
/// training (full, partial and test batches), each shape called repeatedly,
/// inputs uploaded afresh per call; every Metal call compared to Flex
/// (forward output, and loss + input gradient through backward).
#[test]
fn replay_2d_poisson() {
    use sciml_rs::neural_operators::layers::spectral_convolution::SpectralInit;
    let pads: Vec<usize> = std::env::var("PADS")
        .map(|s| s.split(',').map(|p| p.parse().unwrap()).collect())
        .unwrap_or_else(|_| vec![0, 4, 8, 16]);
    let (m, f) = (metal(), Device::flex());
    let s = 64;
    let input = |b: usize, d: &Device| {
        let vals: Vec<f32> = (0..b * s * s)
            .map(|i| ((i as f32) * 0.0137).sin() + 0.3 * ((i as f32) * 0.0031).cos())
            .collect();
        Tensor::<4>::from_data(TensorData::new(vals, [b, s, s, 1]), d)
    };
    // Training-like order: train batches (20, partial 7), eval batches (20, 3).
    let schedule: [(&str, usize); 8] = [
        ("train", 20),
        ("train", 20),
        ("train", 7),
        ("eval", 20),
        ("eval", 3),
        ("train", 20),
        ("train", 7),
        ("eval", 20),
    ];
    let mut worst = 0.0f32;
    for p in pads {
        f.seed(5);
        let model_f = FNOConfig::new(vec![12, 12], 1, 1)
            .with_hidden_channels(32)
            .with_n_layers(4)
            .with_padding(Some(p))
            .with_spectral_init(Some(SpectralInit::SymmetricUniform))
            .init::<4>(&f);
        let model_m = model_f.clone().to_device(&m);
        let (fa, ma) = (f.clone().autodiff(), m.clone().autodiff());
        let (model_fa, model_ma) = (
            model_f.clone().to_device(&fa),
            model_f.clone().to_device(&ma),
        );

        // Flex references, once per (kind, batch).
        let mut refs: RefCache = Default::default();
        let train_step = |model: &sciml_rs::neural_operators::models::fno::FNO<4>, x: Tensor<4>| {
            let x = x.require_grad();
            let loss = model.forward(x.clone()).powf_scalar(2.0).mean();
            let g = loss.backward();
            let gx = x.grad(&g).unwrap();
            let mut out = v(loss.inner().reshape([1, 1, 1, 1]));
            out.extend(v(gx));
            out
        };
        let mut line = Vec::new();
        for (call, &(kind, b)) in schedule.iter().enumerate() {
            let want = refs
                .entry((kind, b))
                .or_insert_with(|| {
                    let o = match kind {
                        "eval" => v(model_f.forward(input(b, &f))),
                        _ => train_step(&model_fa, input(b, &fa)),
                    };
                    (o, Vec::new())
                })
                .0
                .clone();
            let got = match kind {
                "eval" => v(model_m.forward(input(b, &m))),
                _ => train_step(&model_ma, input(b, &ma)),
            };
            let e = rel(&got, &want);
            worst = worst.max(e);
            line.push(format!(
                "{call}:{kind}{b}={e:.1e}{}",
                if e > 1e-5 { "!" } else { "" }
            ));
        }
        println!("p={p:<2} {}", line.join(" "));
    }
    println!("worst metal/flex rel err: {worst:.2e}");
    assert!(worst <= 1e-5, "Metal differs from Flex: {worst:e}");
}
