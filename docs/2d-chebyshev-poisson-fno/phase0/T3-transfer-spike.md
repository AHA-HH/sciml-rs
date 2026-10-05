# 2d-chebyshev-poisson-fno phase 0 / T3 - grid-transfer stability spike (C0.3)

Can start at once. It does not need T2, because the spike has its own dependencies.

Read first:
- `docs/design/2d-chebyshev-poisson-fno.md`:
  - §2.2–§2.3 (grids, exact formulas);
  - §2.5 (M2, M3);
  - §2.6 (transfers, FH weights);
  - §2.7 (CC norm);
  - §3.3 (alternatives);
  - §8.2.
- Phase 0 README "Design decisions", including the error measures.
- `~/Code/rlst/rlst/src/interpolation.rs`: `barycentric_chebychev_weights` (:152),
  `barycentric_evaluate_1d` (:199), `tensor_barycentric_evaluate_2d` (:285). Note the exact-node
  handling at :246.

This spike measures how accurately data moves between the Chebyshev grid and the uniform model
grid. It recommends the Floater–Hormann blending degree d for the uniform → Chebyshev transfer
T_uc. The report sets the "transfer floor" that model errors are later judged against (design §8.2),
and Phase 2 (C2.2) implements the library transfers from its findings.

Do:
- Create `spikes/cheb-transfer/`, a stand-alone binary crate:
  - `Cargo.toml` with an empty `[workspace]` table;
  - dependencies `rlst = "=0.6.1"` and `ndarray = "0.17"`;
  - the macOS Accelerate `blas-src`/`lapack-src` dependencies as in the Phase 0 README;
  - it is not built by CI.
- Add `/spikes/*/target/` to `.gitignore`. T4 adds the same line; keep one copy.
- `spikes/cheb-transfer/src/main.rs` contains:
  - **Grids.**
    - `gl_nodes(n) -> Array1<f64>`, the sin form exactly as in the Phase 0 README;
    - `uniform_nodes(s) -> Array1<f64>`, computed per index exactly as in the README.
  - **Weights.**
    - `cheb_weights(n)`: the second-kind barycentric weights (−1)^j·δ_j, with δ_0 = δ_{n−1} = ½,
      else 1. Either call RLST and reverse, or compute directly; record which. The quotient
      ignores any global sign.
    - `fh_weights(s, d)`: the formula in design §2.6.
  - **Transfers.**
    - `t_cu(v: [n,n]) -> [s,s]` and `t_uc(v: [s,s], d) -> [n,n]`.
    - Both use RLST `tensor_barycentric_evaluate_2d`, or two 1D passes with
      `barycentric_evaluate_1d` if the 2D signature is awkward; record which.
    - Copy into and out of RLST arrays element by element through `[i, j]`.
  - **Clenshaw–Curtis.** `cc_weights(n)` per Trefethen's `clencurt`.
    - Check: the weights sum to 2 within 1e-14 absolute.
    - Check: ∫x^{2k} = 2/(2k+1) for 2k ≤ n − 1, within 1e-13 absolute.
  - **Test fields** (f64; M2, M3, A analytic, R_k computed):
    - M2: sin(πx)sin(πy);
    - M3: (1−x²)(1−y²)e^{x+y/2};
    - A: x + 3y + x·y² (the asymmetric axis-order field from design §8.1 L1);
    - R1–R20: 20 **actual Poisson solutions** of the training family (design §2.5 (b)), computed in
      the spike:
      - Forcing:
        - g = Σ_{p,q<16} a_pq T_p(x)T_q(y), with a_pq ~ Normal(mean 0, **std**
          (1+(p²+q²)/16)^{−1.25}), i.e. variance (1+(p²+q²)/16)^{−2.5};
        - corner-corrected, f = g − Π₁g;
        - seeds 0..19.
      - Solver: dense LU of the Kronecker-sum system −(I⊗A + A⊗I)u = f on the 63² interior of the
        n = 65 grid, with A = D₂[1..n−2, 1..n−2] built from Trefethen's `cheb` on the ascending nodes.
        Use RLST `lu`/`Solve`; 3 969 unknowns is a few seconds.
      - Solve check: M2 through the same solver, `rel_max` ≤ 1e-10 against the analytic solution.
      - Use any seeded normal generator, for example `rand` 0.9 + `rand_distr`, and record the crate
        versions.
      - This spike's solver is throwaway. Phase 1 builds the library solver independently.
      - Do not use polynomials with the spectrum of f as stand-ins: they are far rougher than Poisson
        solutions and would distort d (review, 2026-10-05).
- **Measurements, printed as Markdown tables** by `cargo run --release --manifest-path spikes/cheb-transfer/Cargo.toml`:
  1. **T_cu accuracy.**
     - n ∈ {17, 33, 65, 129}, s = 64.
     - `rel_max(t_cu(field on GL grid), field on uniform grid)` for M2, M3 and A.
  2. **T_uc accuracy.**
     - d ∈ {3, …, 8}, s ∈ {33, 64, 65, 128}, n = 65.
     - `rel_max` and `rel_cc` of `t_uc(field on uniform grid, d)` against the field on the GL grid,
       for M2, M3 and A.
     - Fit the convergence rate in s per d, using only errors above 1e-12.
  3. **Round trip.** `rel_cc(t_uc(t_cu(v), d), v)` at n = 65, for s ∈ {64, 128}, d = 3..8 and v ∈ {M2, M3, R1–R20}. Report the max and the median over R1–R20.
  4. **Lebesgue constant of FH at s = 64.**
     - The max over 10 000 equispaced evaluation points of Σ_k |ℓ_k(x)|, for d = 3..8.
     - At points that coincide with a node, set the value to 1 explicitly. The formula there is
       inf/inf = NaN, and `f64::max` silently ignores NaN.
     - Compare it with the bound 2^{d−1}(2 + ln s) (design §2.6).
  5. **Boundary exactness.** At n = 65, s = 64:
     - M3 (zero boundary data): every boundary row and column of T_cu and T_uc output must equal
       the field exactly (`==`, i.e. 0.0).
     - A (non-zero boundary data): exact equality is required **only at the four corners**.
     - Along A's edges the transfer still interpolates in the other variable, so equality holds
       only to round-off. Report the max |diff| as a number; the review measured about 1e-14.
- **`spikes/cheb-transfer/SPIKE_REPORT.md`** contains:
  - the tables;
  - the machine and toolchain;
  - the crate versions;
  - every deviation from this brief;
  - a recommendation: **the smallest d whose max round-trip `rel_cc` at s = 64 is ≤ 1e-5 for M2,
    M3 and every R_k**.
    - The 1e-5 target is design §8.2 L6: the floor must be ≤ 1/100 of the model error, and f32 FNO
      errors are expected to be ≥ 1e-3. The pilot in Phase 3 re-checks it.
    - Report the smallest d meeting the target at s = 128 too. If s = 64 is marginal (within 2× of
      the target; the review estimated d = 5 at 9.95e-6), say so, because it interacts with T4's
      choice of s.
    - If no d meets the target at s = 64, recommend the s = 128 value or a fallback from design §3.3,
      with numbers.

Tests that define done (as `#[test]`s in the spike crate, run with `cargo test --manifest-path spikes/cheb-transfer/Cargo.toml`):
- `gl_nodes_exact`: for n ∈ {5, 65, 129}:
  - x₀ == −1.0, x_{n−1} == 1.0, x_{(n−1)/2} == 0.0, and x_j == −x_{n−1−j} for all j;
  - `gl_nodes(65)[j] == gl_nodes(129)[2j]` for all j (bitwise).
- `fh_weights_match_definition`: for s = 12 and d ∈ {3, 5}, evaluate the FH interpolant via
  `fh_weights` and barycentric evaluation.
  - Data: exp(x)·sin(3x) at the uniform nodes. Not a polynomial of degree ≤ d, which would hide a
    wrong binomial or degree.
  - Evaluation points: 50 seeded uniform random points in (−1, 1), at least 1e-6 away from every
    node, since the naive λ is infinite at a node.
  - Compare with a naive implementation in the test: the blend Σ_i λ_i(x) p_i(x) / Σ_i λ_i(x) of
    local degree-d Lagrange polynomials p_i, as in Floater–Hormann 2007, eq. (2).
  - Tolerance: 1e-12, relative to max |naive|.
- `t_cu_exact_on_polynomials`: n = 17, s = 64, field x^7·y^5 + x^3.
  - `rel_max` ≤ 1e-12 against direct evaluation on the uniform grid.
  - The trusted reference is the closed form.
- `axis_order_preserved`: field A, n = 33, s = 64.
  - `rel_max(t_cu(A_gl), A_unif)` ≤ 1e-12.
  - The same computation with the y axis flipped, or with x and y swapped, gives `rel_max` > 0.1.
    This proves the field detects both mistakes.
- `cc_exact`: the two Clenshaw–Curtis checks above, for n ∈ {5, 17, 65, 129}.

Must pass:

```sh
cargo fmt --manifest-path spikes/cheb-transfer/Cargo.toml -- --check
cargo clippy --manifest-path spikes/cheb-transfer/Cargo.toml -- -D warnings
cargo test --manifest-path spikes/cheb-transfer/Cargo.toml
cargo run --release --manifest-path spikes/cheb-transfer/Cargo.toml
```

Also run the repository's full checks from `CLAUDE.md`, to show the root crate is unaffected:

```sh
cargo fmt -- --check
cargo clippy --no-deps -- -D warnings
cargo clippy --no-deps --examples -- -D warnings
cargo test
cargo doc --no-deps
cargo clippy --all-targets -- -D warnings
```

Do not:
- add anything to `src/` or the root `Cargo.toml`. Library transfers are Phase 2, C2.2;
- choose d by any criterion other than the one above, or change the 1e-6 target. If it seems
  wrong, say so in the report;
- add the spike to CI;
- edit the design document. The measured numbers go into it at the phase exit, after sign-off.
