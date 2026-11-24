# Optimisers: reference notes (SOAP, Shampoo, SOAP on neural operators)

Engineering notes for the "optimisers" objective (SOAP and L-BFGS benchmarked against Adam
and Gauss–Newton on the 2D FNO, later 3D). Read 2026-11-24 from the arXiv HTML and PDF
copies and the authors' public code. Section, algorithm and line numbers refer to those
copies. Anything not checked against a source is marked **not verified**.

| Key | Paper | arXiv | Version read |
|---|---|---|---|
| [SOAP] | Vyas, Morwani, Zhao, Kwun, Shapira, Brandfonbrener, Janson, Kakade, "SOAP: Improving and Stabilizing Shampoo using Adam" | 2409.11321 | v2, 31 Jan 2025 |
| [SOAP-code] | Reference implementation `soap.py`, https://github.com/nikhilvyas/SOAP/tree/main | — | commit `a1e5535` (2024-12-02) |
| [Shampoo] | Gupta, Koren, Singer, "Shampoo: Preconditioned Stochastic Tensor Optimization" (ICML 2018) | 1802.09568 | HTML |
| [Khod] | Khodakarami, Oommen, Ahmadi Daryakenari, Beekenkamp, Karniadakis, "Spectral bias in physics-informed and operator learning: Analysis and mitigation guidelines" | 2602.19265 | v1, 22 Feb 2026 |
| [Khod-code] | https://github.com/SiaK4/PIN_NO_Spectral_Bias | — | HEAD `1b79100` (2026-06-07) |

## 1. SOAP [SOAP]

### 1.1 Idea
Shampoo with exponent 1/2 equals Adafactor run in the eigenbasis of Shampoo's
preconditioner (§4.1, Claim 1, Algorithms 1–2). SOAP replaces Adafactor with Adam in that
basis, updates Adam's second moment every step, and refreshes the basis only every `f`
steps (§4.1, Algorithm 3). Its one extra hyperparameter over AdamW is the preconditioning
frequency `f` (abstract; §1).

### 1.2 Update rule (Algorithm 3, one m × n layer)
State per layer: L (m×m), R (n×n), Q_L, Q_R, M and V (m×n).
1. G ← gradient on the batch.
2. G′ ← Q_Lᵀ G Q_R (rotate the gradient).
3. M ← β₁ M + (1 − β₁) G (in the original space); M′ ← Q_Lᵀ M Q_R.
4. V ← β₂ V + (1 − β₂) G′ ⊙ G′ (Adam's second moment, in the rotated space).
5. N′ ← M′ / (√V̂ + ε); N ← Q_L N′ Q_Rᵀ (back to the original space).
6. W ← W − η N.
7. L ← β₂ L + (1 − β₂) G Gᵀ;  R ← β₂ R + (1 − β₂) Gᵀ G (EMA, with Adam's β₂ in the paper).
8. If t mod f = 0: Q_L ← Eigenvectors(L, Q_L); Q_R ← Eigenvectors(R, Q_R).

Bias correction and decoupled weight decay are used as in AdamW but left out of the box
(§4.1, implementation detail 4). The factors are updated after the step, so the current
gradient never enters its own projection ([SOAP-code] lines 206–210).

### 1.3 Eigenbasis: computation and refresh
- First time: a full eigendecomposition (`torch.linalg.eigh`) (§4.1, detail 2). The code
  adds 1e-30·I, retries in float64 if eigh fails, and flips Q to descending eigenvalue order
  ([SOAP-code] `get_orthogonal_matrix`, lines 337–370).
- Then, every `f` steps: one step of power iteration and a QR, Algorithm 4: S ← P Q,
  Q ← QR(S) (§4.1, detail 2, after Wang et al. 2024). Before the QR the code estimates
  eigenvalues as diag(Qᵀ P Q), sorts them in descending order and permutes Q's columns
  ([SOAP-code] lines 412–417).
- §7.3 and Figure 7 (right): power iteration with QR performs as well as a fresh eigh at every
  frequency they tried.
- Frequency: default 10 (Figure 1, §6.1). They used 80 at a 256k-token batch (Figure 4) and
  kept batch × f constant when shrinking the batch (§6.3). Frequencies 1–100 all beat AdamW,
  and SOAP degrades more slowly with f than Shampoo (§6.2).

### 1.4 Adam's moments when the basis changes
This is in the code, not the paper. M is stored in the current eigenbasis. Every step, M is
projected back to the original space and then into the (possibly new) basis, so M is
rotated exactly ([SOAP-code] `update_preconditioner`, lines 273–274 and 306–307). V is
**not** rotated. Only its indices are permuted along each preconditioned dimension, to follow
the sorted columns of Q (lines 414 and 429). So V is carried over into the new basis
unchanged apart from the permutation. Step 1 only initialises L, R and Q: the parameter is
not updated on that step (line 153).

### 1.5 Tensors of order > 2, large dimensions, 1D parameters
The paper only treats 2D layers. Higher order comes from the code and from [Shampoo]:
- Per-dimension factors: for every dimension k of the gradient,
  GG_k ← lerp(GG_k, contraction of G with itself over all dims except k, 1 − shampoo_beta).
  Projection is a `tensordot` with each Q_k in turn ([SOAP-code] lines 290–298, 240–266).
- `max_precond_dim` (default 10000): a dimension larger than this gets no factor, i.e. Q = I
  on that side (lines 230–234; paper §4.1 detail 3, "for layers with huge dimensions …
  we fix the rotation matrix to be identity"). If both sides are the identity, SOAP is Adam.
- `merge_dims` (default False): reshapes the tensor by multiplying consecutive dimensions
  together while the product stays ≤ `max_precond_dim` (lines 79–106). §5 below works this
  through for our shapes.
- One-sided variant (§7.1): Q = I on the larger side. It does slightly worse than SOAP but
  matches or beats Shampoo (Figure 6). In the code this is reached only through
  `max_precond_dim`.
- 1D parameters (biases) get plain AdamW (§4.1, detail 1). In the code,
  `precondition_1d=False` (lines 221–225).
- A size-1 dimension (e.g. a kernel-size-1 conv) gets a 1×1 factor in the code. That is
  harmless.

### 1.6 Hyperparameters

| Name | Code default ([SOAP-code] lines 50–61) | Paper experiments (App. A) |
|---|---|---|
| lr | 3e-3 | swept {0.1, 0.0316, …, 3.16e-4} |
| betas (β₁, β₂) | (0.95, 0.95) | β₁ 0.95; β₂ 0.95 (2m batch), 0.99 (256k batch) |
| shampoo_beta | −1 → use β₂ | = β₂ (Algorithm 3 lines 13–14) |
| eps | 1e-8 | 1e-8 |
| weight_decay | 0.01 (decoupled) | 1e-4 (decoupled, after Wortsman et al.) |
| precondition_frequency | 10 | 10; 80 at 256k batch |
| max_precond_dim | 10000 | not stated |
| merge_dims | False | not stated |
| precondition_1d | False | AdamW for 1D (§4.1) |
| normalize_grads | False; "helps at f ≈ 100, hurts at f ≈ 10" (docstring) | not stated |
| correct_bias | True | "identical to AdamW" (§4.1, detail 4) |

The paper also used warmup with cosine decay to 0.1× the peak lr, and z-loss (App. A).

### 1.7 Memory and compute (§7.2, §7.3; m×n layer, m > n)
- Memory: 2m² (L, Q_L) + 2n² (R, Q_R) + 3mn (gradient, M, V), against 3mn for AdamW. Same as
  DistributedShampoo (§7.2). The one-sided variant needs 2·min(m,n)² + 3mn (§7.2.1).
- Per-step overhead: m³ + n³ (updating L, R) + 2m²n + 2mn² (projecting and projecting back).
  That is more than Shampoo's m³ + n³ + m²n + mn² (§7.3). The eigenbasis refresh comes on top,
  every f steps.
- The overhead falls with f but levels off above zero because of the per-step matmuls
  (Figure 7 left, §7.3). Headline numbers: at 2m batch, at least 40% fewer iterations and at
  least 35% less wall time than AdamW (Figure 1, §6.1). At 256k batch, 25% fewer iterations
  and at least 15% less wall time (§6.3).

### 1.8 Precision
- The optimizer matmuls run in float32 (§7.3). `torch.linalg.qr` does not support precision
  below float32 (§7.3.1). Lower-precision preconditioners are left to future work (§8).
- Models were trained in bf16 mixed precision (App. A). The paper states no requirement for
  float64 eigendecompositions. The code falls back to float64 only when eigh fails
  (§1.3 above).

## 2. Shampoo [Shampoo]
- Matrix case (Algorithm 1, §1.1): L_t = L_{t−1} + G Gᵀ, R_t = R_{t−1} + Gᵀ G,
  W ← W − η L^{−1/4} G R^{−1/4}, with L₀ = εI, R₀ = εI. Memory m² + n² instead of m²n². Building
  the preconditioners costs O(m³ + n³) instead of O(m³n³) (§1.1).
- Order-k tensor (Algorithm 2, §4.2): one n_i × n_i factor per dimension,
  H^i_t = H^i_{t−1} + G^{(i)}_t, where G^{(i)} = mat_i(G) mat_i(G)ᵀ (contraction over all
  other dims, §4.1). The update is G̃ = G ×₁ (H¹)^{−1/2k} ⋯ ×_k (H^k)^{−1/2k}. The order of
  the mode products does not matter (§4.1).
- Implementation (§5): matrix powers via SVD. Each tensor is preconditioned on its own,
  so the preconditioner is block-diagonal over tensors. A dimension above a threshold (they
  used about 1200) switches to the diagonal variant (Appendix A, Algorithm 3). The choice is
  made per dimension. Experiments include CIFAR-100 ResNets whose parameters are all
  order-4 convolution tensors, the largest (256, 3, 3, 256) (§6.1).
- SOAP's Shampoo uses exponent 1/2, not 1/4, with EMA factors rather than sums ([SOAP] §4.1,
  §2). SOAP never forms a matrix power: it needs only the eigenvectors.

## 3. Khodakarami et al. 2026 [Khod], arXiv 2602.19265
The ID is correct: title "Spectral bias in physics-informed and operator learning: Analysis
and mitigation guidelines", authors as in the table above, submitted 22 Feb 2026.

- **Operators:** DeepONet, DeepOKAN, FNO and CNO (§3.2, Q6 in §1). Two problems: 2D Schlieren
  super-resolution, 128×256 grid, 800/100/100 split (§4.5.1); and 1D earthquake roof
  response, 4000 time steps, 100 training and 44 test records (§4.5.2).
- **FNO is TFNO** (Tucker-factorised, rank 0.10, 4 layers, 256 channels, 100 modes) for the
  earthquake case ([Khod-code] `Earthquake/PyTorch/configs/model_configs.py`, `FNOConfig`). So the
  spectral parameters are a Tucker core and factor matrices, not dense [C, C, m] tensors.
  The Schlieren FNO configuration is not in the repository: **not verified**. FNO has 3.4M
  parameters there (Table 5).
- **SOAP configuration:** the paper gives no SOAP hyperparameters. The code
  ([Khod-code] `configs/training_config.py`) uses the reference defaults: lr 1e-3, betas
  (0.95, 0.95), shampoo_beta 0.95, eps 1e-8, precondition_frequency 10, max_precond_dim 10000,
  merge_dims False, precondition_1d False, normalize_grads False. Also weight decay 1e-4,
  batch 128, 100 epochs, cosine schedule. The Adam lr is also 1e-3.
- **Complex weights:** their `soap.py` is the reference code plus one change. A complex
  gradient is viewed as real with `torch.view_as_real` (a trailing dimension of size 2, so
  re/im get a joint 2×2 factor), and the update is viewed back as complex (lines 100–103,
  174–175). The paper itself does not discuss this.
- **Results, SOAP against Adam** (App. D, Table E.2, earthquake): DeepONet-SIREN field NRMSE
  0.0015 → 0.0008 (2.0×). DeepOKAN does not converge under Adam with any seed; SOAP reaches
  0.0022. "FNO and CNO show minimal optimizer sensitivity in field accuracy". FNO field error
  0.0037 (Adam) against 0.0038 (SOAP). FNO with BSP loss: log-spectral error 0.098 → 0.082 (1.2×).
  Schlieren (App. D, Figure D.2): SOAP keeps more high-k energy than Adam for DeepONet and
  CNO. That comparison is qualitative, and FNO is not in it. Table 5 (all runs SOAP) has no
  Adam column.
- **Second-order methods that failed** (App. D): SS-Broyden, the best optimiser for their
  PINNs, could not be made to work for data-driven neural operators. The reasons they give:
  (i) PINN quasi-Newton runs are full batch on modest collocation sets, while operators are
  trained in mini-batches on large data. (ii) Precision: FP64 prevents L-BFGS stalls in PINNs
  (citing arXiv 2505.10949), while operators usually run in FP32. (iii) Stable estimation
  and inversion of Fisher or Gauss–Newton structure then "requires aggressive approximations
  (subsampling, low-rank structure, truncated solves)", which are "sensitive to noise, damping,
  and numerical conditioning". They did not report L-BFGS or Gauss–Newton runs on operators.
- **Future work** (App. D last paragraph, §5): "reliable second-order variants tailored to
  neural operators", with "improved precision control, better curvature estimators, and
  scalable linear-solve strategies". Also extension to large-scale 3D problems. And: "SOAP
  can use mini-batching but SS-Broyden cannot".
- Their PINN SOAP (`Kdv/Pytorch/Optimizer/soap_double.py`) runs the eigen and QR steps in
  float64. That is for PINNs, not operators.

## 4. SOAP or Shampoo on an FNO elsewhere
One web search ("SOAP optimizer Fourier neural operator Shampoo spectral weights training",
2026-11-24) found no paper that applies SOAP or Shampoo to an FNO's spectral weights. Hits
were general SOAP and Shampoo analyses: arXiv 2509.22938 (SOAP as gradient whitening),
2510.25000, and 2506.03595. The only use found is [Khod] (TFNO, earthquake, §3 above). Not
exhaustive: one search only.

## 5. Design questions for SOAP on our FNO

### 5.1 Our parameters
2,368,001 parameters. 16 spectral tensors (4 layers × 2 corners × {re, im}), each
[in=32, out=32, mx=12, my=12] = 147,456 elements, 2,359,296 in all (99.6%). The other 8,705
(from `models/fno/mod.rs`):

| Tensor | Shape (Burn) | Elements |
|---|---|---|
| fc0 weight / bias | [3, 32] / [32] | 96 / 32 |
| w (Conv1d k=1) ×4 weight / bias | [32, 32, 1] / [32] | 4 × (1024 + 32) |
| fc1 weight / bias | [32, 128] / [128] | 4096 / 128 |
| fc2 weight / bias | [128, 1] / [1] | 128 / 1 |

In 3D, `SpectralConv` keeps 2^(D−1) corners (`spectral_convolution.rs` line 97), so 4
corners per layer and 32 tensors of shape [32, 32, m, m, m] = 1024·m³ elements.

### 5.2 Factor sizes and memory per tensor
"Overhead" = factors GG plus eigenbases Q (2 · Σ dᵢ² floats) relative to the tensor's own
size N. M and V add 2N, as in Adam (§1.7). Shapes after merging follow the code's
`merge_dims` rule (§1.5).

| Option (2D spectral, N = 147,456) | Dims preconditioned | Σ dᵢ² | GG+Q / N | All 16 tensors (f32) |
|---|---|---|---|---|
| A. Plain per-dim (code default, merge_dims=False) | 32, 32, 12, 12 | 2,336 | 3.2% | 0.30 MB |
| B. Merge (mx, my) → 144 | 32, 32, 144 | 22,784 | 31% | 2.9 MB |
| C. Code `merge_dims=True`, max 10000 → [1024, 144] | 1024, 144 | 1,069,312 | 14.5× | 137 MB |
| D. Mode dims left unpreconditioned (Q = I) | 32, 32 | 2,048 | 2.8% | 0.26 MB |
| E. Channels merged only, [1024, 12, 12] | 1024, 12, 12 | 1,048,864 | 14.2× | 134 MB |

| 3D spectral [32, 32, m, m, m] | m = 8 (N = 524,288) | m = 12 (N = 1,769,472) |
|---|---|---|
| A. Per-dim (32, 32, m, m, m) | 0.85% | 0.28% |
| B. Merge modes → m³ (32, 32, m³) | 1.0× | 3.4× |
| C. Code merge, max 10000 | [8192, 64] → 256× | [1024, 1728] → 4.6× |
| D. Modes unpreconditioned | 0.78% | 0.23% |

| Other tensors (option A) | Factors | GG+Q / N |
|---|---|---|
| fc0 [3, 32] | 3², 32² | 21× (2,066 floats) |
| w [32, 32, 1] | 32², 32², 1² | 4.0× |
| fc1 [32, 128] | 32², 128² | 8.5× |
| fc2 [128, 1] | 128², 1² | 256× (32,770 floats) |
| biases | none (AdamW, §1.5) | 0 |

The non-spectral tensors cost about 0.1M floats in total under option A. That is
negligible in absolute terms, despite the ratios.

Compute per step under option A (**not verified** by measurement): each projection of a
tensor costs about 2·N·Σdᵢ flops, with Σdᵢ = 88 for 2D. The reference code does four
projections per step (gradient in, update out, M out and back in), plus Σ 2·N·dᵢ for the
factor contractions. That gives about 880·N ≈ 1.3·10⁸ flops per spectral tensor per step, before
any eigenbasis refresh.

### 5.3 Choices the design must make (baseline: the paper's and code's defaults)
1. **Which dims to precondition.** Baseline = A: every dim ≤ 10000, `merge_dims=False`
   (§1.5, §1.6), which is what [Khod-code] ran. B lets the 144 modes mix (cross-mode
   correlation). D keeps the per-mode diagonal scaling (Adam along the modes, rotation only
   in channel space). C and E are what the code's `merge_dims=True` produces for our shapes.
   They are dominated by a 1024 × 1024 channel-pair factor, and C is impractical in 3D at
   m = 8. A different `max_precond_dim` changes the merge: for example 1000 gives
   [32, 384, 12].
2. **re and im: separately or together.** Baseline as our parameters stand: 16 independent
   real tensors, each with its own factors (block-diagonal over tensors, [Shampoo] §5).
   Alternatives: (a) stack re and im into [32, 32, 12, 12, 2] and add a 2×2 factor. That
   matches [Khod-code]'s `view_as_real` treatment of complex weights (§3). (b) Share one set
   of factors between re and im (GG summed over both). (c) Also stack the 2 corners of a
   layer (an extra size-2 dim). None of these is evaluated in any paper read here.
3. **max_precond_dim / one-sided.** The default 10000 never triggers for our shapes under A,
   B or D. One-sided SOAP (§7.1) would set Q = I on the larger side, for example the 128 side
   of fc1.
4. **Frequency f and normalize_grads.** Default 10. The paper's evidence is for LLM batches
   (§6.2, §6.3). Our batch and epoch structure differs, so f is a sweep variable. The code
   docstring ties `normalize_grads` to large f.
5. **lr, betas, weight decay.** Code defaults lr 3e-3, (0.95, 0.95), wd 0.01. Paper runs:
   wd 1e-4. [Khod-code]: lr 1e-3, wd 1e-4. Our Adam baseline's settings are the natural
   starting point. That is a design decision, not a finding.
6. **Moments on a basis change.** Follow the code (rotate M exactly, permute V) or the
   paper's Algorithm 3 (M kept in the original space and projected every step). These are
   equivalent for M. V's treatment is code-only (§1.4).
7. **Precision.** The paper runs the optimiser in f32, its models in bf16 mixed precision (§1.8). Our runs are f32 on GPU, and f64 has
   known cuda limits in this repo (project memory). Whether the refresh needs f64 for our
   12×12 and 32×32 factors is open: **not verified**.

### 5.4 What Burn needs (Burn 0.23.0-pre.2, `burn-linalg`)
| Operation | Needed by | Burn status (from the crate source) |
|---|---|---|
| Tensor contraction / mode product (tensordot) | all options, every step | reshape/permute + `matmul`. Available |
| EMA (lerp), elementwise sqrt and divide | all, every step | available |
| `argsort` and `select` along a dim (sort Q's columns, permute V) | every refresh | `argsort_descending` (orderable.rs l.189) and `select` (base.rs l.1715). Available |
| QR of a d×d matrix (power iteration refresh, Alg. 4) | every f steps | `burn::linalg::qr(tensor, reduced)`, Householder written in tensor ops, so it runs on the tensor's device. Needs the `linalg` feature, which this crate does not enable today. GPU behaviour **not verified** |
| Symmetric eigendecomposition (initial Q) | once per tensor | **no `eigh`** in burn-linalg. Substitute: `burn::linalg::svd`. For a symmetric PSD L the left singular vectors are eigenvectors (sorted descending, as the code wants). Its default implementation runs on the host (`svd_host.rs`, Golub–Kahan + `dbdsqr`). That is fine for d ≤ 144, but C/E (d = 1024) and 3D B (d = 1728) would be slow on host: **not measured**. Second substitute: start from Q = I and rely on the QR refresh. That departs from the paper, and its effect is **not verified** |

Sizes Burn must handle per option: A and D: d ∈ {32, 12} (2D), {32, m} (3D). B: up to 144
(2D) or m³ (3D). C and E: 1024 and up. Batched QR over the 16 tensors' equal-sized factors
is possible because `qr` accepts `[..., n, n]`.
