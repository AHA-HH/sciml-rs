# T3: grid-transfer spike

Phase 0, task 3. Branch `phase0/T3-transfer-spike`. Independent of T4. It may use RLST
the way T2 sets it up, but does not need to wait for T2.

## Read first
- Design §3.2 (nodes, 'ij' storage), §3.3 (GRF, K(n), tail table, tolerance), §5.3
  (resolutions), §6.2 (transfers), §7 (error decomposition), §12 decisions 3 and 5.

## Goal
Measure the Chebyshev → uniform and uniform → Chebyshev transfer errors at the four
resolutions. Choose the Floater–Hormann degree d and the uniform size rule (s = n − 1 or
s = n). Confirm the GRF tail table and its tolerance.

## Setup
- Standalone package `spikes/transfers/` (f64, plain Rust). Its dependencies are
  `ndarray`, `rand` and `rand_distr` (seeded), and optionally `rlst` as an oracle.
- Nodes: x_j = −cos(πj/(n − 1)), ascending, n ∈ {33, 65, 129, 257}.
- Uniform grids: s points on [−1, 1] with both endpoints, for s ∈ {n − 1, n}.
- Transfers, each a dense 1D matrix applied along both axes:
  - T_cu: barycentric interpolation, Chebyshev → uniform (second-kind weights, signs
    alternating, endpoints halved);
  - T_uc: Floater–Hormann of degree d ∈ {0, …, 8}, uniform → Chebyshev.
- Test fields:
  1. sin(πx) sin(πy);
  2. (1 − x²)(1 − y²) e^(x + 2y);
  3. 20 GRF samples per resolution (design §3.3: sine basis, τ = 3, α = 2,
     K(n) = min((n − 1)/2, 64), normalised by √S_K, fixed seeds);
  4. the same 20 samples evaluated at K = 64 at every n, to separate truncation from
     transfer error.

## Measurements
Report relative L² errors, Clenshaw–Curtis on the Chebyshev grid and trapezoidal on the
uniform grid, as mean and maximum over the samples:
1. T_cu: the interpolant on the uniform grid against the exact field there.
2. T_uc for each d: the exact field sampled on the uniform grid, interpolated to the
   Chebyshev nodes, against the exact field there.
3. The round trip T_uc(T_cu u) − u on the Chebyshev grid, for each d. This is the
   "transfer" error of design §7.
4. The Lebesgue constant of T_uc for each d and s (maximum row sum of |T_uc|).
5. For the GRF: ε_K for K ∈ {16, 32, 64, 128}, against design §3.3's table. Also the
   empirical mean square over 1000 samples, which should be close to 1 at every K: the
   check of the √S_K normalisation.
6. Time to build T_cu and T_uc at n = 257, and to apply them to one field.

## Decision rules
State these in `REPORT.md` and apply them:
- **d** is the smallest degree whose round-trip error on the GRF samples is at most 1e-4
  at n = 65, 129 and 257 for the chosen s, and whose Lebesgue constant stays below 10.
  If no d meets both, report the best trade-off and stop; do not pick silently.
- **s = n** replaces the stated s = n − 1 only if it lowers the round-trip error by at
  least 10× at the chosen d on the GRF samples at 65, 129 and 257 (design decision 5).
- **The tail tolerance** of 5e-3 stays if the measured ε_K for K = 32 is at most 5e-3;
  otherwise, report the K(n) rule that meets it.

## Deliverables
- `spikes/transfers/` (code and `Cargo.toml`), and `spikes/transfers/REPORT.md` with the
  tables, the reproduction commands, the machine and the decisions.
- In `docs/phase0/README.md`, the Results rows for d, the uniform size rule and the GRF
  tail table.
- One line in the Layout section of `CLAUDE.md`: "`spikes/`: standalone packages for
  Phase 0 measurements; never built by CI."

## Acceptance
- Checks inside the spike pass, with the tolerance named next to each:
  - FH with d = s − 1 equals polynomial interpolation on the uniform nodes, to 1e-10 at
    s ≤ 17;
  - FH of degree d reproduces polynomials of degree ≤ d, to 1e-12;
  - T_cu reproduces polynomials of degree < n, to 1e-12, and agrees with RLST's
    barycentric evaluation (reversed node order) to 1e-13 if RLST is used.
- `REPORT.md` contains every table above and states d, s and the tolerance outcome.
- The crate's own checks are unaffected: `cargo test` and clippy as in `CLAUDE.md`.

## Do not
- Add anything to `src/` (Phase 2 T1 writes the production transfers).
- Edit `docs/CONVENTIONS.md` (T1 does, with these numbers).
