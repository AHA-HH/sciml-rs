# T3 report: grid transfers and the GRF tail

Phase 0, task 3 (`docs/phase0/T3-transfer-spike.md`). Run on 2026-10-06.

## Machine and reproduction

- Apple M2 Pro, macOS (Darwin 25.6.0), rustc 1.99.0, release profile, f64.
- Direct dependencies are pinned exactly in `Cargo.toml`: ndarray 0.17.2, rand 0.9.2
  (`StdRng`), rand_distr 0.5.1. Transitive ones (rand_core, rand_chacha) are not: the
  root `.gitignore` excludes `Cargo.lock`, so the `StdRng` stream relies on those crates
  keeping it stable across compatible versions. RLST is not used, because T2 had not landed. The barycentric weights
  are checked against the closed form and against polynomial reproduction instead.
- Commands, run from `spikes/transfers/`:
  ```sh
  cargo test --release                       # acceptance checks (12 tests)
  cargo run --release 2>/dev/null > out.md   # every table below, verbatim
  ```
- Seeds: test samples 1000..1019, mean-square samples 2000..2999. Each draw is
  ξ ~ N(0, 1) on a 128 × 128 block in row-major order, and truncation K takes the leading
  K × K block.

## Setup

- **Nodes.** CGL nodes x_j = −cos(πj/(n − 1)), ascending, n ∈ {33, 65, 129, 257}. The
  uniform grid has s ∈ {n − 1, n} points, both endpoints included.
- **T_cu.** Barycentric interpolation, Chebyshev → uniform, with second-kind weights
  (−1)^j and the endpoints halved.
- **T_uc.** Floater–Hormann of degree d = 0..8, uniform → Chebyshev, using the general
  weight formula of Floater & Hormann (2007).
- **2D application.** Both transfers are dense 1D matrices applied along both axes,
  T F Tᵀ.
- **Errors.** Relative L², with Clenshaw–Curtis weights on the Chebyshev grid and
  trapezoidal weights on the uniform grid.
- **GRF cells.** Each cell shows `mean / max` over the 20 samples.
- **GRF sets.** "GRF K(n)" uses K(n) = min((n − 1)/2, 64). "GRF K=64" evaluates the same
  draws at K = 64 for every n, to separate truncation from transfer error.
- **Solution sets.** "GRF solution u" is the exact solution of −Δu = f, u = 0 on ∂Ω, for
  the same draws: the same sine series with coefficients c_kl / μ_kl, where
  μ_kl = (π²/4)(k² + l²).

## Acceptance checks (`cargo test --release`, all pass)

| Check | Tolerance |
| --- | --- |
| FH with d = s − 1 equals Lagrange interpolation (product formula), s = 2..17 | 1e-10 |
| FH of degree d reproduces T_0..T_d, for d = 0..8 and s ∈ {16, 17, 32, 64, 128, 256} | 1e-12 |
| T_cu reproduces T_0..T_{n−1}, for n ∈ {5, 17, 33, 65, 129, 257} and s ∈ {n − 1, n} | 1e-12 |
| Closed-form CGL weights are proportional to 1/Π(x_k − x_j) | 1e-12 |
| CC weights integrate x^k exactly for k ≤ n − 1 | 1e-14 |
| Nodes ascending, exact ±1, antisymmetric; GRF draws nested (K = 32 inside K = 64) | exact / 1e-13 |
| GRF solution u: 5-point −Δ_h u matches f (K = 8, h = 2/400), and u = 0 on ∂Ω | 1e-3 relative / 1e-14 |

## Decisions

The rules, as stated in the brief:
- **d** is the smallest degree whose round-trip error on the GRF samples is at most 1e-4
  at n = 65, 129 and 257 for the chosen s, and whose Lebesgue constant stays below 10.
  "Error" is taken as the maximum over the 20 samples. The Lebesgue constant is the 1D
  one, the maximum row sum of |T_uc|; the 2D tensor operator's is its square (about 24
  at d = 2), and under that reading no d qualifies at n ≥ 65.
- **s = n** replaces s = n − 1 only if it lowers the round-trip error by at least 10× at
  the chosen d, on the GRF samples at 65, 129 and 257.
- **The tail tolerance** of 5e-3 stays if ε_32 ≤ 5e-3; otherwise, report the K(n) rule
  that meets it.

### Which field the d rule is applied to

The brief says "the GRF samples". The first run of this spike applied the rule to the
forcing f, and **no d ∈ {0, …, 8} met it** for either s. Applied to f:
- at n = 65 and 129, every d gives an error above 1e-4;
- the best maxima for s = n − 1 are 5.9e-4 at n = 65 and 1.7e-4 at n = 129, at d = 8,
  where Λ is 92 and 115.

Design §7 defines the transfer error as T_uc(T_cu u) − u on the test **solution u**, and
T_uc only ever acts on predicted u. The input f only goes through T_cu, which is accurate
to 3.4e-6 or better at n ≥ 65 (table 1). On 2026-10-06 the author therefore decided to
evaluate the d rule on u, the exact Poisson solution of the same 20 draws. The decision
is recorded in design §12. The results on f are kept below as context.

**Why f and u differ.** The diagnostic table shows the FH error depends on points per
wavelength (ppw) on the uniform grid:
- with K(n) = (n − 1)/2, the top modes have about 4 ppw;
- at about 4 ppw, every degree gives errors around 1e-2;
- at about 16 ppw, d ≥ 3 reaches about 5e-5.

In f those top modes carry relative amplitude λ_kl^(1/2) / λ_11^(1/2), which is about
1e-3. In u they are divided again by μ_kl / μ_11, which is about 500–1000 at K = 32. The
under-resolved part of the spectrum is therefore about 3 orders of magnitude weaker in u.

### d = 2

GRF solution u at K(n), round trip, mean / max over 20 samples, s = n − 1:

| d | Λ at n = 65 / 129 / 257 | n = 65 | n = 129 | n = 257 |
| --- | --- | --- | --- | --- |
| 1 | 3.4 / 3.9 / 4.3 | 1.8e-4 / 3.1e-4 | 4.6e-5 / 7.9e-5 | 1.1e-5 / 1.8e-5 |
| **2** | **3.9 / 4.4 / 4.9** | **1.7e-5 / 3.6e-5** | **2.1e-6 / 4.4e-6** | **2.3e-7 / 4.8e-7** |
| 3 | 6.1 / 6.8 / 7.8 | 6.2e-6 / 1.3e-5 | 5.3e-7 / 1.1e-6 | 2.2e-8 / 4.4e-8 |
| 4 | 9.9 / 11.4 / 13.2 | 2.4e-6 / 5.2e-6 | 1.9e-7 / 3.8e-7 | 6.5e-9 / 1.3e-8 |

- **Result.** d = 2 is the smallest degree that meets both conditions at n = 65, 129 and
  257, for both s = n − 1 and s = n. Its worst case is 3.6e-5 at n = 65, a margin of about
  3× under 1e-4. **d = 2.**
- **d = 1** fails at n = 65 (3.1e-4).
- **d = 3** would give about 3× more margin at n = 65, for Λ < 8. That remains a
  documented alternative if Phase 3 shows the transfer error is visible in the total
  error.
- **n = 33.** Stage 1 is exempt from the rule. There d = 2 gives 3.2e-4 (max).

Under the brief's literal reading on f, no d qualifies. The trade-off table for f
(maximum over samples, s = n − 1) is kept for the record:

| d | n = 65 | n = 129 | n = 257 |
| --- | --- | --- | --- |
| 2 | 1.5e-3 | 4.8e-4 | 5.4e-5 |
| 3 | 1.5e-3 | 4.7e-4 | 1.8e-5 |
| 4 | 6.8e-4 | 2.0e-4 | 6.1e-6 |
| 8 | 5.9e-4 | 1.7e-4 | 7.7e-7 |

### s: keep s = n − 1

At d = 2, the ratio of round-trip errors (s = n − 1)/(s = n), mean (max):

| n | u | f |
| --- | --- | --- |
| 65 | 0.99 (1.01) | 1.04 (1.04) |
| 129 | 1.01 (1.05) | 1.03 (1.03) |
| 257 | 0.96 (1.00) | 1.01 (1.01) |

On u and on f alike, this is far below 10×. **The uniform size rule stays s = n − 1**
(32/64/128/256), as design decision 5 states.

### GRF tail: table confirmed, tolerance kept

- **ε_K.** The measured values are 1.86e-2, 4.92e-3, 1.26e-3 and 3.19e-4 for
  K = 16/32/64/128. They agree with design §3.3's table to the two digits it gives.
- **Tolerance.** ε_32 = 4.92e-3 ≤ 5e-3, so **the tolerance of 5e-3 is kept**, along with
  K(n) = min((n − 1)/2, 64). The margin at K = 32 is small (1.6%).
- **Normalisation.** The empirical mean square over 1000 samples is 0.983 ± 0.013 at
  every K. That is consistent with 1 (1.3 standard errors), which confirms the √S_K
  normalisation. Parseval and the CC quadrature at n = 257 agree to four digits, which
  confirms the field evaluation. The four rows reuse the same draws and the low modes
  dominate every sample, so they are nearly identical and not independent checks.

## Measured tables

Below is the verbatim output of `cargo run --release`.

Cells: single field → its error; GRF sets → mean / max over 20 samples.
GRF K(n) = min((n−1)/2, 64); seeds 1000..1019.

### 1. T_cu: Chebyshev → uniform (trapezoidal relative L²)

| n | s | sin(πx)sin(πy) | (1−x²)(1−y²)e^(x+2y) | GRF K(n) | GRF K=64 | GRF solution u, K(n) | GRF solution u, K=64 | Λ(T_cu) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 33 | 32 | 4.3e-16 | 4.1e-16 | 2.5e-4 / 3.4e-4 | 1.2e-1 / 1.6e-1 | 3.7e-6 / 8.0e-6 | 5.5e-4 / 1.1e-3 | 3.02 |
| 33 | 33 | 4.0e-16 | 3.6e-16 | 2.4e-4 / 3.4e-4 | 1.2e-1 / 1.6e-1 | 3.6e-6 / 7.9e-6 | 5.4e-4 / 1.1e-3 | 3.09 |
| 65 | 64 | 5.2e-16 | 5.1e-16 | 2.3e-6 / 3.4e-6 | 4.5e-2 / 6.2e-2 | 8.9e-9 / 2.0e-8 | 6.9e-5 / 1.4e-4 | 3.53 |
| 65 | 65 | 6.5e-16 | 5.4e-16 | 2.3e-6 / 3.5e-6 | 4.5e-2 / 6.2e-2 | 9.0e-9 / 2.0e-8 | 6.9e-5 / 1.4e-4 | 3.59 |
| 129 | 128 | 7.5e-16 | 7.2e-16 | 7.5e-10 / 1.0e-9 | 7.5e-10 / 1.0e-9 | 7.2e-13 / 1.5e-12 | 7.2e-13 / 1.5e-12 | 4.03 |
| 129 | 129 | 7.1e-16 | 6.7e-16 | 7.5e-10 / 1.0e-9 | 7.5e-10 / 1.0e-9 | 7.2e-13 / 1.5e-12 | 7.2e-13 / 1.5e-12 | 4.03 |
| 257 | 256 | 1.0e-15 | 1.0e-15 | 1.6e-15 / 1.8e-15 | 1.6e-15 / 1.8e-15 | 1.4e-15 / 1.5e-15 | 1.4e-15 / 1.5e-15 | 4.49 |
| 257 | 257 | 1.0e-15 | 1.0e-15 | 1.6e-15 / 1.8e-15 | 1.6e-15 / 1.8e-15 | 1.4e-15 / 1.5e-15 | 1.4e-15 / 1.5e-15 | 4.49 |

### 2. T_uc: uniform → Chebyshev, Floater–Hormann degree d (CC relative L²)

#### sin(πx)sin(πy)

| n | s | d=0 | d=1 | d=2 | d=3 | d=4 | d=5 | d=6 | d=7 | d=8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 33 | 32 | 4.0e-2 | 3.6e-4 | 1.2e-4 | 5.9e-6 | 1.3e-6 | 1.0e-7 | 2.5e-8 | 2.1e-9 | 6.4e-10 |
| 33 | 33 | 2.6e-2 | 2.1e-3 | 1.8e-4 | 1.4e-5 | 2.1e-6 | 2.5e-7 | 4.4e-8 | 7.5e-9 | 1.1e-9 |
| 65 | 64 | 1.8e-2 | 7.8e-5 | 1.2e-5 | 3.0e-7 | 2.5e-8 | 1.3e-9 | 1.1e-10 | 7.3e-12 | 6.3e-13 |
| 65 | 65 | 1.4e-2 | 5.3e-4 | 2.3e-5 | 7.1e-7 | 5.8e-8 | 2.3e-9 | 2.9e-10 | 1.5e-11 | 2.0e-12 |
| 129 | 128 | 9.5e-3 | 2.0e-5 | 1.5e-6 | 1.7e-8 | 5.6e-10 | 1.6e-11 | 5.2e-13 | 2.2e-14 | 2.1e-15 |
| 129 | 129 | 6.6e-3 | 1.3e-4 | 2.8e-6 | 3.9e-8 | 1.6e-9 | 2.3e-11 | 1.8e-12 | 3.2e-14 | 3.2e-15 |
| 257 | 256 | 4.2e-3 | 4.3e-6 | 1.6e-7 | 8.8e-10 | 1.3e-11 | 1.8e-13 | 2.9e-15 | 1.0e-15 | 1.1e-15 |
| 257 | 257 | 3.0e-3 | 3.2e-5 | 3.3e-7 | 2.3e-9 | 4.4e-11 | 2.7e-13 | 1.0e-14 | 1.1e-15 | 1.1e-15 |

#### (1−x²)(1−y²)e^(x+2y)

| n | s | d=0 | d=1 | d=2 | d=3 | d=4 | d=5 | d=6 | d=7 | d=8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 33 | 32 | 4.5e-2 | 4.1e-3 | 3.8e-4 | 3.6e-5 | 3.9e-6 | 4.7e-7 | 5.7e-8 | 7.6e-9 | 9.1e-10 |
| 33 | 33 | 3.0e-2 | 3.7e-3 | 3.3e-4 | 3.1e-5 | 3.3e-6 | 3.7e-7 | 4.7e-8 | 5.6e-9 | 7.3e-10 |
| 65 | 64 | 2.0e-2 | 9.1e-4 | 4.2e-5 | 1.9e-6 | 9.6e-8 | 5.6e-9 | 3.4e-10 | 2.3e-11 | 1.4e-12 |
| 65 | 65 | 1.6e-2 | 9.7e-4 | 4.1e-5 | 1.8e-6 | 9.1e-8 | 5.0e-9 | 3.1e-10 | 1.9e-11 | 1.3e-12 |
| 129 | 128 | 1.0e-2 | 2.3e-4 | 5.2e-6 | 1.1e-7 | 2.5e-9 | 6.8e-11 | 2.0e-12 | 6.5e-14 | 2.4e-15 |
| 129 | 129 | 7.5e-3 | 2.4e-4 | 5.0e-6 | 1.1e-7 | 2.5e-9 | 6.4e-11 | 1.9e-12 | 5.9e-14 | 2.1e-15 |
| 257 | 256 | 4.5e-3 | 5.2e-5 | 5.9e-7 | 6.2e-9 | 6.7e-11 | 8.3e-13 | 1.2e-14 | 9.0e-16 | 9.6e-16 |
| 257 | 257 | 3.5e-3 | 5.6e-5 | 5.9e-7 | 6.2e-9 | 6.7e-11 | 8.0e-13 | 1.2e-14 | 8.7e-16 | 9.0e-16 |

#### GRF K(n)

| n | s | d=0 | d=1 | d=2 | d=3 | d=4 | d=5 | d=6 | d=7 | d=8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 33 | 32 | 4.1e-2 / 5.3e-2 | 6.5e-3 / 9.2e-3 | 3.1e-3 / 4.3e-3 | 3.3e-3 / 4.4e-3 | 1.6e-3 / 2.2e-3 | 1.6e-3 / 2.2e-3 | 3.3e-3 / 4.4e-3 | 3.5e-3 / 4.5e-3 | 1.6e-3 / 2.2e-3 |
| 33 | 33 | 2.8e-2 / 3.9e-2 | 5.5e-3 / 7.7e-3 | 2.8e-3 / 3.7e-3 | 2.8e-3 / 3.7e-3 | 1.3e-3 / 1.8e-3 | 1.4e-3 / 1.9e-3 | 2.6e-3 / 3.4e-3 | 2.5e-3 / 3.3e-3 | 1.5e-3 / 2.0e-3 |
| 65 | 64 | 2.0e-2 / 2.5e-2 | 2.2e-3 / 3.1e-3 | 1.1e-3 / 1.5e-3 | 1.0e-3 / 1.5e-3 | 4.7e-4 / 6.8e-4 | 5.3e-4 / 7.5e-4 | 9.9e-4 / 1.4e-3 | 9.2e-4 / 1.3e-3 | 4.0e-4 / 6.0e-4 |
| 65 | 65 | 1.6e-2 / 2.1e-2 | 2.2e-3 / 3.0e-3 | 1.0e-3 / 1.4e-3 | 9.7e-4 / 1.4e-3 | 4.4e-4 / 6.3e-4 | 5.0e-4 / 7.1e-4 | 8.8e-4 / 1.3e-3 | 7.8e-4 / 1.1e-3 | 3.7e-4 / 5.4e-4 |
| 129 | 128 | 1.0e-2 / 1.3e-2 | 7.9e-4 / 1.1e-3 | 3.5e-4 / 4.8e-4 | 3.4e-4 / 4.7e-4 | 1.5e-4 / 2.0e-4 | 1.8e-4 / 2.5e-4 | 3.2e-4 / 4.3e-4 | 2.8e-4 / 3.8e-4 | 1.2e-4 / 1.7e-4 |
| 129 | 129 | 7.8e-3 / 1.1e-2 | 7.6e-4 / 1.0e-3 | 3.4e-4 / 4.7e-4 | 3.3e-4 / 4.5e-4 | 1.4e-4 / 1.9e-4 | 1.7e-4 / 2.4e-4 | 3.0e-4 / 4.1e-4 | 2.6e-4 / 3.5e-4 | 1.1e-4 / 1.6e-4 |
| 257 | 256 | 4.7e-3 / 5.9e-3 | 1.8e-4 / 2.4e-4 | 4.0e-5 / 5.4e-5 | 1.3e-5 / 1.8e-5 | 4.5e-6 / 6.1e-6 | 3.7e-6 / 5.0e-6 | 4.1e-7 / 5.6e-7 | 1.1e-6 / 1.6e-6 | 5.6e-7 / 7.7e-7 |
| 257 | 257 | 3.7e-3 / 5.0e-3 | 1.8e-4 / 2.4e-4 | 3.9e-5 / 5.4e-5 | 1.3e-5 / 1.8e-5 | 4.4e-6 / 6.0e-6 | 3.6e-6 / 4.9e-6 | 4.1e-7 / 5.6e-7 | 1.1e-6 / 1.5e-6 | 5.4e-7 / 7.4e-7 |

#### GRF K=64

| n | s | d=0 | d=1 | d=2 | d=3 | d=4 | d=5 | d=6 | d=7 | d=8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 33 | 32 | 1.0e-1 / 1.4e-1 | 9.5e-2 / 1.3e-1 | 9.6e-2 / 1.3e-1 | 1.0e-1 / 1.4e-1 | 1.1e-1 / 1.5e-1 | 1.3e-1 / 1.9e-1 | 1.9e-1 / 2.7e-1 | 3.3e-1 / 4.8e-1 | 7.0e-1 / 1.2e0 |
| 33 | 33 | 9.0e-2 / 1.2e-1 | 8.6e-2 / 1.2e-1 | 8.8e-2 / 1.2e-1 | 9.2e-2 / 1.3e-1 | 1.0e-1 / 1.4e-1 | 1.2e-1 / 1.7e-1 | 1.8e-1 / 2.5e-1 | 3.0e-1 / 4.8e-1 | 6.2e-1 / 1.1e0 |
| 65 | 64 | 2.3e-2 / 2.8e-2 | 1.1e-2 / 1.5e-2 | 1.1e-2 / 1.4e-2 | 1.2e-2 / 1.7e-2 | 1.6e-2 / 2.4e-2 | 2.5e-2 / 3.7e-2 | 4.2e-2 / 6.4e-2 | 8.0e-2 / 1.2e-1 | 1.7e-1 / 2.9e-1 |
| 65 | 65 | 1.8e-2 / 2.4e-2 | 7.9e-3 / 1.0e-2 | 7.5e-3 / 9.6e-3 | 8.9e-3 / 1.2e-2 | 1.3e-2 / 1.9e-2 | 2.1e-2 / 3.1e-2 | 3.6e-2 / 5.5e-2 | 6.6e-2 / 1.0e-1 | 1.3e-1 / 2.3e-1 |
| 129 | 128 | 1.0e-2 / 1.3e-2 | 7.9e-4 / 1.1e-3 | 3.5e-4 / 4.8e-4 | 3.4e-4 / 4.7e-4 | 1.5e-4 / 2.0e-4 | 1.8e-4 / 2.5e-4 | 3.2e-4 / 4.3e-4 | 2.8e-4 / 3.8e-4 | 1.2e-4 / 1.7e-4 |
| 129 | 129 | 7.8e-3 / 1.1e-2 | 7.6e-4 / 1.0e-3 | 3.4e-4 / 4.7e-4 | 3.3e-4 / 4.5e-4 | 1.4e-4 / 1.9e-4 | 1.7e-4 / 2.4e-4 | 3.0e-4 / 4.1e-4 | 2.6e-4 / 3.5e-4 | 1.1e-4 / 1.6e-4 |
| 257 | 256 | 4.7e-3 / 5.9e-3 | 1.8e-4 / 2.4e-4 | 4.0e-5 / 5.4e-5 | 1.3e-5 / 1.8e-5 | 4.5e-6 / 6.1e-6 | 3.7e-6 / 5.0e-6 | 4.1e-7 / 5.6e-7 | 1.1e-6 / 1.6e-6 | 5.6e-7 / 7.7e-7 |
| 257 | 257 | 3.7e-3 / 5.0e-3 | 1.8e-4 / 2.4e-4 | 3.9e-5 / 5.4e-5 | 1.3e-5 / 1.8e-5 | 4.4e-6 / 6.0e-6 | 3.6e-6 / 4.9e-6 | 4.1e-7 / 5.6e-7 | 1.1e-6 / 1.5e-6 | 5.4e-7 / 7.4e-7 |

#### GRF solution u, K(n)

| n | s | d=0 | d=1 | d=2 | d=3 | d=4 | d=5 | d=6 | d=7 | d=8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 33 | 32 | 3.1e-2 / 3.7e-2 | 7.8e-4 / 1.4e-3 | 1.4e-4 / 3.2e-4 | 7.3e-5 / 1.7e-4 | 3.0e-5 / 6.4e-5 | 3.5e-5 / 8.1e-5 | 5.6e-5 / 1.1e-4 | 5.5e-5 / 1.1e-4 | 3.0e-5 / 6.8e-5 |
| 33 | 33 | 1.7e-2 / 2.3e-2 | 8.2e-4 / 1.9e-3 | 1.3e-4 / 2.8e-4 | 6.2e-5 / 1.4e-4 | 2.4e-5 / 5.5e-5 | 3.0e-5 / 6.5e-5 | 4.4e-5 / 9.0e-5 | 3.9e-5 / 8.0e-5 | 2.6e-5 / 5.4e-5 |
| 65 | 64 | 1.5e-2 / 1.7e-2 | 1.8e-4 / 3.1e-4 | 1.7e-5 / 3.6e-5 | 6.2e-6 / 1.3e-5 | 2.4e-6 / 5.2e-6 | 3.0e-6 / 6.2e-6 | 4.5e-6 / 9.7e-6 | 3.9e-6 / 8.5e-6 | 2.0e-6 / 4.5e-6 |
| 65 | 65 | 9.1e-3 / 1.2e-2 | 2.1e-4 / 4.8e-4 | 1.7e-5 / 3.5e-5 | 5.9e-6 / 1.2e-5 | 2.2e-6 / 4.8e-6 | 2.8e-6 / 5.8e-6 | 4.0e-6 / 8.7e-6 | 3.3e-6 / 7.2e-6 | 1.8e-6 / 4.0e-6 |
| 129 | 128 | 7.5e-3 / 8.7e-3 | 4.6e-5 / 7.9e-5 | 2.1e-6 / 4.4e-6 | 5.3e-7 / 1.1e-6 | 1.9e-7 / 3.8e-7 | 2.5e-7 / 5.1e-7 | 3.7e-7 / 7.4e-7 | 3.0e-7 / 6.0e-7 | 1.6e-7 / 3.2e-7 |
| 129 | 129 | 4.4e-3 / 5.9e-3 | 5.2e-5 / 1.2e-4 | 2.0e-6 / 4.2e-6 | 5.1e-7 / 1.0e-6 | 1.9e-7 / 3.6e-7 | 2.4e-7 / 5.0e-7 | 3.5e-7 / 7.0e-7 | 2.7e-7 / 5.5e-7 | 1.5e-7 / 3.1e-7 |
| 257 | 256 | 3.4e-3 / 3.9e-3 | 1.1e-5 / 1.8e-5 | 2.3e-7 / 4.8e-7 | 2.2e-8 / 4.4e-8 | 6.5e-9 / 1.3e-8 | 4.4e-9 / 8.9e-9 | 5.3e-10 / 1.1e-9 | 1.3e-9 / 2.7e-9 | 5.9e-10 / 1.2e-9 |
| 257 | 257 | 2.0e-3 / 2.7e-3 | 1.3e-5 / 2.9e-5 | 2.3e-7 / 4.8e-7 | 2.1e-8 / 4.3e-8 | 6.4e-9 / 1.3e-8 | 4.3e-9 / 8.7e-9 | 5.2e-10 / 1.1e-9 | 1.3e-9 / 2.6e-9 | 5.7e-10 / 1.2e-9 |

#### GRF solution u, K=64

| n | s | d=0 | d=1 | d=2 | d=3 | d=4 | d=5 | d=6 | d=7 | d=8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 33 | 32 | 3.1e-2 / 3.7e-2 | 8.3e-4 / 1.5e-3 | 3.0e-4 / 6.1e-4 | 3.0e-4 / 6.4e-4 | 3.4e-4 / 7.5e-4 | 4.4e-4 / 9.8e-4 | 6.6e-4 / 1.4e-3 | 1.1e-3 / 2.3e-3 | 2.2e-3 / 4.8e-3 |
| 33 | 33 | 1.7e-2 / 2.3e-2 | 8.6e-4 / 1.9e-3 | 2.6e-4 / 5.2e-4 | 2.6e-4 / 5.3e-4 | 3.0e-4 / 6.6e-4 | 4.0e-4 / 8.9e-4 | 6.0e-4 / 1.3e-3 | 1.0e-3 / 2.2e-3 | 1.9e-3 / 3.8e-3 |
| 65 | 64 | 1.5e-2 / 1.7e-2 | 1.8e-4 / 3.1e-4 | 2.0e-5 / 4.0e-5 | 1.5e-5 / 2.9e-5 | 1.9e-5 / 3.9e-5 | 2.8e-5 / 5.8e-5 | 4.6e-5 / 9.0e-5 | 8.0e-5 / 1.5e-4 | 1.5e-4 / 3.0e-4 |
| 65 | 65 | 9.1e-3 / 1.2e-2 | 2.1e-4 / 4.8e-4 | 1.9e-5 / 3.8e-5 | 1.2e-5 / 2.4e-5 | 1.6e-5 / 3.3e-5 | 2.4e-5 / 5.0e-5 | 3.9e-5 / 8.1e-5 | 6.8e-5 / 1.4e-4 | 1.2e-4 / 2.9e-4 |
| 129 | 128 | 7.5e-3 / 8.7e-3 | 4.6e-5 / 7.9e-5 | 2.1e-6 / 4.4e-6 | 5.3e-7 / 1.1e-6 | 1.9e-7 / 3.8e-7 | 2.5e-7 / 5.1e-7 | 3.7e-7 / 7.4e-7 | 3.0e-7 / 6.0e-7 | 1.6e-7 / 3.2e-7 |
| 129 | 129 | 4.4e-3 / 5.9e-3 | 5.2e-5 / 1.2e-4 | 2.0e-6 / 4.2e-6 | 5.1e-7 / 1.0e-6 | 1.9e-7 / 3.6e-7 | 2.4e-7 / 5.0e-7 | 3.5e-7 / 7.0e-7 | 2.7e-7 / 5.5e-7 | 1.5e-7 / 3.1e-7 |
| 257 | 256 | 3.4e-3 / 3.9e-3 | 1.1e-5 / 1.8e-5 | 2.3e-7 / 4.8e-7 | 2.2e-8 / 4.4e-8 | 6.5e-9 / 1.3e-8 | 4.4e-9 / 8.9e-9 | 5.3e-10 / 1.1e-9 | 1.3e-9 / 2.7e-9 | 5.9e-10 / 1.2e-9 |
| 257 | 257 | 2.0e-3 / 2.7e-3 | 1.3e-5 / 2.9e-5 | 2.3e-7 / 4.8e-7 | 2.1e-8 / 4.3e-8 | 6.4e-9 / 1.3e-8 | 4.3e-9 / 8.7e-9 | 5.2e-10 / 1.1e-9 | 1.3e-9 / 2.6e-9 | 5.7e-10 / 1.2e-9 |

### 3. Round trip T_uc(T_cu u) − u on the Chebyshev grid (CC relative L²)

#### sin(πx)sin(πy)

| n | s | d=0 | d=1 | d=2 | d=3 | d=4 | d=5 | d=6 | d=7 | d=8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 33 | 32 | 4.0e-2 | 3.6e-4 | 1.2e-4 | 5.9e-6 | 1.3e-6 | 1.0e-7 | 2.5e-8 | 2.1e-9 | 6.4e-10 |
| 33 | 33 | 2.6e-2 | 2.1e-3 | 1.8e-4 | 1.4e-5 | 2.1e-6 | 2.5e-7 | 4.4e-8 | 7.5e-9 | 1.1e-9 |
| 65 | 64 | 1.8e-2 | 7.8e-5 | 1.2e-5 | 3.0e-7 | 2.5e-8 | 1.3e-9 | 1.1e-10 | 7.3e-12 | 6.2e-13 |
| 65 | 65 | 1.4e-2 | 5.3e-4 | 2.3e-5 | 7.1e-7 | 5.8e-8 | 2.3e-9 | 2.9e-10 | 1.5e-11 | 2.0e-12 |
| 129 | 128 | 9.5e-3 | 2.0e-5 | 1.5e-6 | 1.7e-8 | 5.6e-10 | 1.6e-11 | 5.2e-13 | 2.2e-14 | 2.2e-15 |
| 129 | 129 | 6.6e-3 | 1.3e-4 | 2.8e-6 | 3.9e-8 | 1.6e-9 | 2.3e-11 | 1.8e-12 | 3.2e-14 | 3.9e-15 |
| 257 | 256 | 4.2e-3 | 4.3e-6 | 1.6e-7 | 8.8e-10 | 1.3e-11 | 1.8e-13 | 3.1e-15 | 1.4e-15 | 1.6e-15 |
| 257 | 257 | 3.0e-3 | 3.2e-5 | 3.3e-7 | 2.3e-9 | 4.4e-11 | 2.7e-13 | 1.1e-14 | 1.4e-15 | 1.5e-15 |

#### (1−x²)(1−y²)e^(x+2y)

| n | s | d=0 | d=1 | d=2 | d=3 | d=4 | d=5 | d=6 | d=7 | d=8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 33 | 32 | 4.5e-2 | 4.1e-3 | 3.8e-4 | 3.6e-5 | 3.9e-6 | 4.7e-7 | 5.7e-8 | 7.6e-9 | 9.1e-10 |
| 33 | 33 | 3.0e-2 | 3.7e-3 | 3.3e-4 | 3.1e-5 | 3.3e-6 | 3.7e-7 | 4.7e-8 | 5.6e-9 | 7.3e-10 |
| 65 | 64 | 2.0e-2 | 9.1e-4 | 4.2e-5 | 1.9e-6 | 9.6e-8 | 5.6e-9 | 3.4e-10 | 2.3e-11 | 1.4e-12 |
| 65 | 65 | 1.6e-2 | 9.7e-4 | 4.1e-5 | 1.8e-6 | 9.1e-8 | 5.0e-9 | 3.1e-10 | 1.9e-11 | 1.3e-12 |
| 129 | 128 | 1.0e-2 | 2.3e-4 | 5.2e-6 | 1.1e-7 | 2.5e-9 | 6.8e-11 | 2.0e-12 | 6.5e-14 | 2.9e-15 |
| 129 | 129 | 7.5e-3 | 2.4e-4 | 5.0e-6 | 1.1e-7 | 2.5e-9 | 6.4e-11 | 1.9e-12 | 5.9e-14 | 2.1e-15 |
| 257 | 256 | 4.5e-3 | 5.2e-5 | 5.9e-7 | 6.2e-9 | 6.7e-11 | 8.3e-13 | 1.2e-14 | 1.4e-15 | 1.5e-15 |
| 257 | 257 | 3.5e-3 | 5.6e-5 | 5.9e-7 | 6.2e-9 | 6.7e-11 | 8.0e-13 | 1.2e-14 | 1.3e-15 | 1.4e-15 |

#### GRF K(n)

| n | s | d=0 | d=1 | d=2 | d=3 | d=4 | d=5 | d=6 | d=7 | d=8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 33 | 32 | 4.1e-2 / 5.3e-2 | 6.5e-3 / 9.2e-3 | 3.1e-3 / 4.3e-3 | 3.2e-3 / 4.4e-3 | 1.6e-3 / 2.1e-3 | 1.7e-3 / 2.3e-3 | 3.5e-3 / 4.5e-3 | 3.7e-3 / 4.8e-3 | 2.0e-3 / 2.8e-3 |
| 33 | 33 | 2.8e-2 / 3.9e-2 | 5.5e-3 / 7.8e-3 | 2.8e-3 / 3.7e-3 | 2.7e-3 / 3.7e-3 | 1.3e-3 / 1.7e-3 | 1.4e-3 / 1.9e-3 | 2.6e-3 / 3.5e-3 | 2.6e-3 / 3.4e-3 | 1.7e-3 / 2.3e-3 |
| 65 | 64 | 2.0e-2 / 2.5e-2 | 2.2e-3 / 3.1e-3 | 1.1e-3 / 1.5e-3 | 1.0e-3 / 1.5e-3 | 4.7e-4 / 6.8e-4 | 5.3e-4 / 7.5e-4 | 9.9e-4 / 1.4e-3 | 9.2e-4 / 1.3e-3 | 4.0e-4 / 5.9e-4 |
| 65 | 65 | 1.6e-2 / 2.1e-2 | 2.2e-3 / 3.0e-3 | 1.0e-3 / 1.4e-3 | 9.7e-4 / 1.4e-3 | 4.4e-4 / 6.3e-4 | 5.0e-4 / 7.1e-4 | 8.8e-4 / 1.3e-3 | 7.8e-4 / 1.1e-3 | 3.6e-4 / 5.3e-4 |
| 129 | 128 | 1.0e-2 / 1.3e-2 | 7.9e-4 / 1.1e-3 | 3.5e-4 / 4.8e-4 | 3.4e-4 / 4.7e-4 | 1.5e-4 / 2.0e-4 | 1.8e-4 / 2.5e-4 | 3.2e-4 / 4.3e-4 | 2.8e-4 / 3.8e-4 | 1.2e-4 / 1.7e-4 |
| 129 | 129 | 7.8e-3 / 1.1e-2 | 7.6e-4 / 1.0e-3 | 3.4e-4 / 4.7e-4 | 3.3e-4 / 4.5e-4 | 1.4e-4 / 1.9e-4 | 1.7e-4 / 2.4e-4 | 3.0e-4 / 4.1e-4 | 2.6e-4 / 3.5e-4 | 1.1e-4 / 1.6e-4 |
| 257 | 256 | 4.7e-3 / 5.9e-3 | 1.8e-4 / 2.4e-4 | 4.0e-5 / 5.4e-5 | 1.3e-5 / 1.8e-5 | 4.5e-6 / 6.1e-6 | 3.7e-6 / 5.0e-6 | 4.1e-7 / 5.6e-7 | 1.1e-6 / 1.6e-6 | 5.6e-7 / 7.7e-7 |
| 257 | 257 | 3.7e-3 / 5.0e-3 | 1.8e-4 / 2.4e-4 | 3.9e-5 / 5.4e-5 | 1.3e-5 / 1.8e-5 | 4.4e-6 / 6.0e-6 | 3.6e-6 / 4.9e-6 | 4.1e-7 / 5.6e-7 | 1.1e-6 / 1.5e-6 | 5.4e-7 / 7.4e-7 |

#### GRF K=64

| n | s | d=0 | d=1 | d=2 | d=3 | d=4 | d=5 | d=6 | d=7 | d=8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 33 | 32 | 5.9e-2 / 7.4e-2 | 4.3e-2 / 6.0e-2 | 4.5e-2 / 6.5e-2 | 5.2e-2 / 7.4e-2 | 6.3e-2 / 9.1e-2 | 8.2e-2 / 1.2e-1 | 1.1e-1 / 1.7e-1 | 1.7e-1 / 2.6e-1 | 2.7e-1 / 4.4e-1 |
| 33 | 33 | 4.6e-2 / 6.0e-2 | 3.8e-2 / 5.3e-2 | 4.1e-2 / 5.8e-2 | 4.8e-2 / 6.8e-2 | 5.9e-2 / 8.6e-2 | 7.8e-2 / 1.2e-1 | 1.1e-1 / 1.7e-1 | 1.6e-1 / 2.6e-1 | 2.6e-1 / 4.4e-1 |
| 65 | 64 | 2.2e-2 / 2.7e-2 | 9.1e-3 / 1.3e-2 | 8.9e-3 / 1.2e-2 | 1.0e-2 / 1.4e-2 | 1.4e-2 / 2.1e-2 | 2.2e-2 / 3.4e-2 | 3.7e-2 / 5.8e-2 | 6.5e-2 / 1.0e-1 | 1.2e-1 / 1.8e-1 |
| 65 | 65 | 1.8e-2 / 2.3e-2 | 8.6e-3 / 1.2e-2 | 8.3e-3 / 1.1e-2 | 9.7e-3 / 1.3e-2 | 1.3e-2 / 2.0e-2 | 2.1e-2 / 3.3e-2 | 3.4e-2 / 5.5e-2 | 6.0e-2 / 9.3e-2 | 1.1e-1 / 1.6e-1 |
| 129 | 128 | 1.0e-2 / 1.3e-2 | 7.9e-4 / 1.1e-3 | 3.5e-4 / 4.8e-4 | 3.4e-4 / 4.7e-4 | 1.5e-4 / 2.0e-4 | 1.8e-4 / 2.5e-4 | 3.2e-4 / 4.3e-4 | 2.8e-4 / 3.8e-4 | 1.2e-4 / 1.7e-4 |
| 129 | 129 | 7.8e-3 / 1.1e-2 | 7.6e-4 / 1.0e-3 | 3.4e-4 / 4.7e-4 | 3.3e-4 / 4.5e-4 | 1.4e-4 / 1.9e-4 | 1.7e-4 / 2.4e-4 | 3.0e-4 / 4.1e-4 | 2.6e-4 / 3.5e-4 | 1.1e-4 / 1.6e-4 |
| 257 | 256 | 4.7e-3 / 5.9e-3 | 1.8e-4 / 2.4e-4 | 4.0e-5 / 5.4e-5 | 1.3e-5 / 1.8e-5 | 4.5e-6 / 6.1e-6 | 3.7e-6 / 5.0e-6 | 4.1e-7 / 5.6e-7 | 1.1e-6 / 1.6e-6 | 5.6e-7 / 7.7e-7 |
| 257 | 257 | 3.7e-3 / 5.0e-3 | 1.8e-4 / 2.4e-4 | 3.9e-5 / 5.4e-5 | 1.3e-5 / 1.8e-5 | 4.4e-6 / 6.0e-6 | 3.6e-6 / 4.9e-6 | 4.1e-7 / 5.6e-7 | 1.1e-6 / 1.5e-6 | 5.4e-7 / 7.4e-7 |

#### GRF solution u, K(n)

| n | s | d=0 | d=1 | d=2 | d=3 | d=4 | d=5 | d=6 | d=7 | d=8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 33 | 32 | 3.1e-2 / 3.7e-2 | 7.8e-4 / 1.4e-3 | 1.4e-4 / 3.2e-4 | 7.3e-5 / 1.7e-4 | 3.0e-5 / 6.3e-5 | 3.6e-5 / 8.1e-5 | 5.8e-5 / 1.2e-4 | 5.9e-5 / 1.2e-4 | 3.5e-5 / 7.9e-5 |
| 33 | 33 | 1.7e-2 / 2.3e-2 | 8.2e-4 / 1.9e-3 | 1.3e-4 / 2.8e-4 | 6.2e-5 / 1.4e-4 | 2.4e-5 / 5.5e-5 | 3.0e-5 / 6.5e-5 | 4.5e-5 / 9.1e-5 | 4.1e-5 / 8.4e-5 | 2.9e-5 / 6.0e-5 |
| 65 | 64 | 1.5e-2 / 1.7e-2 | 1.8e-4 / 3.1e-4 | 1.7e-5 / 3.6e-5 | 6.2e-6 / 1.3e-5 | 2.4e-6 / 5.2e-6 | 3.0e-6 / 6.2e-6 | 4.5e-6 / 9.7e-6 | 3.9e-6 / 8.5e-6 | 2.0e-6 / 4.5e-6 |
| 65 | 65 | 9.1e-3 / 1.2e-2 | 2.1e-4 / 4.8e-4 | 1.7e-5 / 3.5e-5 | 5.9e-6 / 1.2e-5 | 2.2e-6 / 4.8e-6 | 2.8e-6 / 5.8e-6 | 4.0e-6 / 8.7e-6 | 3.3e-6 / 7.2e-6 | 1.8e-6 / 4.0e-6 |
| 129 | 128 | 7.5e-3 / 8.7e-3 | 4.6e-5 / 7.9e-5 | 2.1e-6 / 4.4e-6 | 5.3e-7 / 1.1e-6 | 1.9e-7 / 3.8e-7 | 2.5e-7 / 5.1e-7 | 3.7e-7 / 7.4e-7 | 3.0e-7 / 6.0e-7 | 1.6e-7 / 3.2e-7 |
| 129 | 129 | 4.4e-3 / 5.9e-3 | 5.2e-5 / 1.2e-4 | 2.0e-6 / 4.2e-6 | 5.1e-7 / 1.0e-6 | 1.9e-7 / 3.6e-7 | 2.4e-7 / 5.0e-7 | 3.5e-7 / 7.0e-7 | 2.7e-7 / 5.5e-7 | 1.5e-7 / 3.1e-7 |
| 257 | 256 | 3.4e-3 / 3.9e-3 | 1.1e-5 / 1.8e-5 | 2.3e-7 / 4.8e-7 | 2.2e-8 / 4.4e-8 | 6.5e-9 / 1.3e-8 | 4.4e-9 / 8.9e-9 | 5.3e-10 / 1.1e-9 | 1.3e-9 / 2.7e-9 | 5.9e-10 / 1.2e-9 |
| 257 | 257 | 2.0e-3 / 2.7e-3 | 1.3e-5 / 2.9e-5 | 2.3e-7 / 4.8e-7 | 2.1e-8 / 4.3e-8 | 6.4e-9 / 1.3e-8 | 4.3e-9 / 8.7e-9 | 5.2e-10 / 1.1e-9 | 1.3e-9 / 2.6e-9 | 5.7e-10 / 1.2e-9 |

#### GRF solution u, K=64

| n | s | d=0 | d=1 | d=2 | d=3 | d=4 | d=5 | d=6 | d=7 | d=8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 33 | 32 | 3.1e-2 / 3.7e-2 | 8.0e-4 / 1.5e-3 | 2.2e-4 / 4.2e-4 | 2.1e-4 / 4.2e-4 | 2.5e-4 / 5.4e-4 | 3.3e-4 / 7.4e-4 | 4.7e-4 / 1.0e-3 | 6.7e-4 / 1.5e-3 | 9.8e-4 / 2.1e-3 |
| 33 | 33 | 1.7e-2 / 2.3e-2 | 8.3e-4 / 1.9e-3 | 1.9e-4 / 3.6e-4 | 1.8e-4 / 3.6e-4 | 2.2e-4 / 4.6e-4 | 2.8e-4 / 6.2e-4 | 3.9e-4 / 8.5e-4 | 5.6e-4 / 1.2e-3 | 8.3e-4 / 1.7e-3 |
| 65 | 64 | 1.5e-2 / 1.7e-2 | 1.8e-4 / 3.1e-4 | 2.0e-5 / 4.1e-5 | 1.5e-5 / 3.0e-5 | 1.9e-5 / 4.0e-5 | 2.8e-5 / 5.8e-5 | 4.6e-5 / 9.0e-5 | 7.8e-5 / 1.5e-4 | 1.4e-4 / 2.9e-4 |
| 65 | 65 | 9.1e-3 / 1.2e-2 | 2.1e-4 / 4.8e-4 | 2.0e-5 / 4.0e-5 | 1.4e-5 / 2.8e-5 | 1.8e-5 / 3.8e-5 | 2.7e-5 / 5.5e-5 | 4.4e-5 / 8.7e-5 | 7.5e-5 / 1.5e-4 | 1.3e-4 / 2.7e-4 |
| 129 | 128 | 7.5e-3 / 8.7e-3 | 4.6e-5 / 7.9e-5 | 2.1e-6 / 4.4e-6 | 5.3e-7 / 1.1e-6 | 1.9e-7 / 3.8e-7 | 2.5e-7 / 5.1e-7 | 3.7e-7 / 7.4e-7 | 3.0e-7 / 6.0e-7 | 1.6e-7 / 3.2e-7 |
| 129 | 129 | 4.4e-3 / 5.9e-3 | 5.2e-5 / 1.2e-4 | 2.0e-6 / 4.2e-6 | 5.1e-7 / 1.0e-6 | 1.9e-7 / 3.6e-7 | 2.4e-7 / 5.0e-7 | 3.5e-7 / 7.0e-7 | 2.7e-7 / 5.5e-7 | 1.5e-7 / 3.1e-7 |
| 257 | 256 | 3.4e-3 / 3.9e-3 | 1.1e-5 / 1.8e-5 | 2.3e-7 / 4.8e-7 | 2.2e-8 / 4.4e-8 | 6.5e-9 / 1.3e-8 | 4.4e-9 / 8.9e-9 | 5.3e-10 / 1.1e-9 | 1.3e-9 / 2.7e-9 | 5.9e-10 / 1.2e-9 |
| 257 | 257 | 2.0e-3 / 2.7e-3 | 1.3e-5 / 2.9e-5 | 2.3e-7 / 4.8e-7 | 2.1e-8 / 4.3e-8 | 6.4e-9 / 1.3e-8 | 4.3e-9 / 8.7e-9 | 5.2e-10 / 1.1e-9 | 1.3e-9 / 2.6e-9 | 5.7e-10 / 1.2e-9 |

### 4. Lebesgue constant of T_uc (1D, max row sum; 2D is its square)

#### Λ(T_uc)

| n | s | d=0 | d=1 | d=2 | d=3 | d=4 | d=5 | d=6 | d=7 | d=8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 33 | 32 | 3.08 | 2.99 | 3.32 | 5.08 | 8.11 | 13.41 | 22.75 | 39.38 | 69.17 |
| 33 | 33 | 2.98 | 2.97 | 3.37 | 5.15 | 8.24 | 13.61 | 23.10 | 39.99 | 70.28 |
| 65 | 64 | 3.49 | 3.45 | 3.91 | 6.10 | 9.94 | 16.76 | 29.01 | 51.24 | 91.89 |
| 65 | 65 | 3.41 | 3.40 | 3.93 | 6.13 | 9.98 | 16.83 | 29.11 | 51.39 | 92.13 |
| 129 | 128 | 3.92 | 3.89 | 4.38 | 6.84 | 11.43 | 19.74 | 34.92 | 62.92 | 114.99 |
| 129 | 129 | 3.89 | 3.88 | 4.38 | 6.86 | 11.47 | 19.79 | 35.00 | 63.05 | 115.21 |
| 257 | 256 | 4.35 | 4.34 | 4.87 | 7.83 | 13.17 | 22.85 | 40.59 | 73.45 | 134.73 |
| 257 | 257 | 4.33 | 4.32 | 4.87 | 7.84 | 13.18 | 22.86 | 40.61 | 73.47 | 134.74 |

### 5. GRF unresolved energy and normalisation

S_∞ = 4.984158e-3 (k, l ≤ 3000 plus integral tail).

| K | ε_K measured | ε_K design §3.3 | mean square, Parseval (mean ± s.e., 1000 samples) | mean square, CC at n = 257 |
| --- | --- | --- | --- | --- |
| 16 | 1.86e-2 | 1.9e-2 | 0.9828 ± 0.0134 | 0.9828 |
| 32 | 4.92e-3 | 4.9e-3 | 0.9830 ± 0.0132 | 0.9830 |
| 64 | 1.26e-3 | 1.3e-3 | 0.9830 ± 0.0131 | 0.9830 |
| 128 | 3.19e-4 | 3.2e-4 | 0.9831 ± 0.0131 | – |

### Decisions (computed)

Rule: max over samples ≤ 1e-4 and Λ < 10 at n = 65, 129, 257.

- GRF solution u (design §7), K(n): d for s = n − 1: Some(2); d for s = n: Some(2).
- GRF forcing f, K(n): d for s = n − 1: None; d for s = n: None.
- u, n = 65, d = 2: round-trip(s = n − 1) / round-trip(s = n) = 0.99 (mean), 1.01 (max).
- u, n = 129, d = 2: round-trip(s = n − 1) / round-trip(s = n) = 1.01 (mean), 1.05 (max).
- u, n = 257, d = 2: round-trip(s = n − 1) / round-trip(s = n) = 0.96 (mean), 1.00 (max).
- u: s = n gives ≥ 10× at every n: false.
- f, n = 65, d = 2: round-trip(s = n − 1) / round-trip(s = n) = 1.04 (mean), 1.04 (max).
- f, n = 129, d = 2: round-trip(s = n − 1) / round-trip(s = n) = 1.03 (mean), 1.03 (max).
- f, n = 257, d = 2: round-trip(s = n − 1) / round-trip(s = n) = 1.01 (mean), 1.01 (max).
- f: s = n gives ≥ 10× at every n: false.
- ε_32 = 4.92e-3 ≤ 5e-3: true.

### Diagnostic: 1D round trip of sin(kπ(x+1)/2), s = n − 1 (CC relative L²)

| n | k (ppw) | d=0 | d=1 | d=2 | d=3 | d=4 | d=5 | d=6 | d=7 | d=8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 65 | 8 (15.8) | 2.4e-2 | 1.9e-3 | 4.8e-4 | 5.0e-5 | 2.6e-5 | 3.8e-6 | 2.1e-6 | 5.6e-7 | 1.7e-7 |
| 65 | 16 (7.9) | 3.3e-2 | 6.5e-3 | 3.0e-3 | 1.2e-3 | 4.8e-4 | 4.8e-4 | 3.5e-5 | 1.8e-4 | 9.3e-5 |
| 65 | 32 (3.9) | 5.2e-2 | 2.1e-2 | 1.8e-2 | 2.4e-2 | 1.4e-2 | 1.0e-2 | 3.0e-2 | 3.4e-2 | 1.3e-2 |
| 129 | 16 (15.9) | 1.7e-2 | 1.6e-3 | 3.2e-4 | 5.0e-5 | 1.7e-5 | 3.3e-6 | 1.2e-6 | 4.3e-7 | 8.6e-8 |
| 129 | 32 (7.9) | 2.4e-2 | 4.9e-3 | 2.0e-3 | 8.7e-4 | 3.2e-4 | 3.3e-4 | 3.6e-5 | 1.1e-4 | 6.5e-5 |
| 129 | 64 (4.0) | 3.6e-2 | 1.6e-2 | 1.2e-2 | 1.6e-2 | 9.4e-3 | 6.9e-3 | 2.0e-2 | 2.2e-2 | 8.5e-3 |
| 257 | 16 (31.9) | 8.1e-3 | 3.5e-4 | 3.7e-5 | 2.1e-6 | 4.2e-7 | 2.1e-8 | 8.7e-9 | 5.5e-10 | 2.2e-10 |
| 257 | 32 (15.9) | 1.1e-2 | 1.1e-3 | 2.2e-4 | 3.5e-5 | 1.1e-5 | 2.4e-6 | 7.9e-7 | 3.1e-7 | 4.9e-8 |
| 257 | 64 (8.0) | 1.6e-2 | 3.5e-3 | 1.4e-3 | 6.1e-4 | 2.1e-4 | 2.2e-4 | 2.8e-5 | 7.6e-5 | 4.5e-5 |

### 6. Time at n = 257 (median of 21, ms)

| s | build T_cu | build T_uc (d = 2) | apply T_cu | apply T_uc |
| --- | --- | --- | --- | --- |
| 256 | 0.098 | 0.101 | 1.423 | 1.436 |
| 257 | 0.098 | 0.101 | 1.495 | 1.461 |
