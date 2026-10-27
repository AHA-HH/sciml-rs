# Phase 6 results

Dataset-structure analysis, not network performance (design §13.4): no model is trained or evaluated. Written by `poisson_pod` at commit `3992c4d5ba547c461f6c0a99c211c56b67901a5d`; every number below comes from it.

## T1: POD on the 129 CGL grid

Stage 3: n = 129, K = 64; the basis fitted on the 1000 train samples, the 200 test samples only projected (centred with the train mean). Inner product: Clenshaw–Curtis weights on both axes (D5). Eigenvalues are σ_i² / (N − 1). Population: λ_kl / S_K for f and λ_kl / (S_K μ_kl²) for u, k, l ≤ 64, τ = 3, α = 2.

### Fits and traces

The expected scatter of the trace is one standard deviation of the sample variance of a Gaussian field with the population spectrum, √(2 Σ λ_i² / (N − 1)) / Σ λ_i.

| Field | SVD wall time (s) | Trace, empirical | Trace, population | Ratio | Expected scatter |
| --- | --- | --- | --- | --- | --- |
| f | 1.37 | 3.9642e0 | 4.0000e0 | 0.9911 | 1.4e-2 |
| u | 1.24 | 4.8474e-2 | 4.9979e-2 | 0.9699 | 3.8e-2 |

### Energy ranks

The smallest r whose leading eigenvalues hold the fraction of the energy. The empirical fractions are of the sample's energy, which lies in at most N − 1 = 999 directions; a population rank above that is reached in no sample of this size, so the empirical rank at that fraction measures N, not the field (marked †).

| Field | 90% | 99% | 99.9% | Population 90% / 99% / 99.9% |
| --- | --- | --- | --- | --- |
| f | 36 | 284 | 743 † | 38 / 418 / 2175 |
| u | 2 | 6 | 17 | 2 / 6 / 17 |

### Leading eigenvalues

Individual eigenvalues within a degenerate group mix in the sample; the group check below is the comparison.

| i | f, empirical | f, population | u, empirical | u, population |
| --- | --- | --- | --- | --- |
| 1 | 1.0037e0 | 1.0346e0 | 4.1070e-2 | 4.2483e-2 |
| 2 | 4.4344e-1 | 4.4125e-1 | 2.8962e-3 | 2.8991e-3 |
| 3 | 4.2412e-1 | 4.4125e-1 | 2.7707e-3 | 2.8991e-3 |
| 4 | 2.6057e-1 | 2.4322e-1 | 6.5873e-4 | 6.2423e-4 |
| 5 | 1.8920e-1 | 1.7716e-1 | 3.1054e-4 | 2.9100e-4 |
| 6 | 1.7759e-1 | 1.7716e-1 | 2.8840e-4 | 2.9100e-4 |
| 7 | 1.2282e-1 | 1.1906e-1 | 1.1865e-4 | 1.1572e-4 |
| 8 | 1.0826e-1 | 1.1906e-1 | 1.0337e-4 | 1.1572e-4 |
| 9 | 8.2509e-2 | 7.7400e-2 | 4.6496e-5 | 4.3991e-5 |
| 10 | 8.0697e-2 | 7.7400e-2 | 4.4180e-5 | 4.3991e-5 |
| 11 | 6.7877e-2 | 7.0414e-2 | 3.3338e-5 | 3.5697e-5 |
| 12 | 6.0662e-2 | 5.9007e-2 | 2.5773e-5 | 2.4231e-5 |
| 13 | 5.6727e-2 | 5.9007e-2 | 2.3000e-5 | 2.4231e-5 |
| 14 | 4.0996e-2 | 4.0207e-2 | 1.0422e-5 | 1.0567e-5 |
| 15 | 3.8275e-2 | 4.0207e-2 | 9.4405e-6 | 1.0567e-5 |
| 16 | 3.6486e-2 | 3.7540e-2 | 9.1034e-6 | 9.1216e-6 |
| 17 | 3.5592e-2 | 3.7540e-2 | 8.7312e-6 | 9.1216e-6 |
| 18 | 3.1982e-2 | 3.0958e-2 | 6.1453e-6 | 6.0464e-6 |
| 19 | 2.9660e-2 | 3.0958e-2 | 5.7223e-6 | 6.0464e-6 |
| 20 | 2.5925e-2 | 2.5967e-2 | 4.0966e-6 | 4.1652e-6 |

### Closed-form check: groups of equal k² + l²

Empirical eigenvalues are summed over the positions the group takes in the sorted population spectrum. Expected scatter of a group of g modes: about √(2 / (N g)). A ratio more than 3 scatters from 1 is marked ‼.

| k² + l² | g | f, empirical | f, population | f ratio | u, empirical | u, population | u ratio | Scatter |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 1 | 1.0037e0 | 1.0346e0 | 0.970 | 4.1070e-2 | 4.2483e-2 | 0.967 | 0.045 |
| 5 | 2 | 8.6756e-1 | 8.8251e-1 | 0.983 | 5.6669e-3 | 5.7983e-3 | 0.977 | 0.032 |
| 8 | 1 | 2.6057e-1 | 2.4322e-1 | 1.071 | 6.5873e-4 | 6.2423e-4 | 1.055 | 0.045 |
| 10 | 2 | 3.6679e-1 | 3.5432e-1 | 1.035 | 5.9894e-4 | 5.8199e-4 | 1.029 | 0.032 |
| 13 | 2 | 2.3108e-1 | 2.3813e-1 | 0.970 | 2.2202e-4 | 2.3144e-4 | 0.959 | 0.032 |
| 17 | 2 | 1.6321e-1 | 1.5480e-1 | 1.054 | 9.0676e-5 | 8.7982e-5 | 1.031 | 0.032 |
| 18 | 1 | 6.7877e-2 | 7.0414e-2 | 0.964 | 3.3338e-5 | 3.5697e-5 | 0.934 | 0.045 |
| 20 | 2 | 1.1739e-1 | 1.1801e-1 | 0.995 | 4.8772e-5 | 4.8461e-5 | 1.006 | 0.032 |
| 25 | 2 | 7.9271e-2 | 8.0414e-2 | 0.986 | 1.9862e-5 | 2.1134e-5 | 0.940 | 0.032 |
| 26 | 2 | 7.2078e-2 | 7.5081e-2 | 0.960 | 1.7835e-5 | 1.8243e-5 | 0.978 | 0.032 |

### Reconstruction error against rank

Per sample ‖v − v_r‖ / ‖v‖ in the CC norm, v the sample centred with the train mean and v_r its projection on the first r modes; mean [max] over the split.

| r | f, train | f, test | u, train | u, test |
| --- | --- | --- | --- | --- |
| 1 | 8.826e-1 [1.000e0] | 8.790e-1 [1.000e0] | 5.296e-1 [1.000e0] | 5.325e-1 [1.000e0] |
| 2 | 8.188e-1 [9.999e-1] | 8.144e-1 [9.998e-1] | 4.143e-1 [9.980e-1] | 4.181e-1 [9.985e-1] |
| 5 | 6.711e-1 [9.769e-1] | 6.648e-1 [9.627e-1] | 1.841e-1 [9.073e-1] | 1.829e-1 [6.437e-1] |
| 10 | 5.458e-1 [9.099e-1] | 5.465e-1 [8.794e-1] | 8.739e-2 [4.295e-1] | 8.971e-2 [3.936e-1] |
| 20 | 4.258e-1 [6.941e-1] | 4.262e-1 [6.796e-1] | 3.803e-2 [1.526e-1] | 3.858e-2 [1.605e-1] |
| 50 | 2.869e-1 [4.762e-1] | 2.985e-1 [4.565e-1] | 1.162e-2 [5.225e-2] | 1.220e-2 [4.427e-2] |
| 100 | 2.029e-1 [3.633e-1] | 2.250e-1 [3.685e-1] | 4.388e-3 [2.449e-2] | 4.843e-3 [2.312e-2] |
| 200 | 1.352e-1 [2.505e-1] | 1.703e-1 [2.615e-1] | 1.540e-3 [9.212e-3] | 1.953e-3 [7.966e-3] |
| 500 | 6.276e-2 [1.166e-1] | 1.219e-1 [1.908e-1] | 3.107e-4 [1.787e-3] | 6.414e-4 [2.584e-3] |
| 999 | 5.680e-15 [9.791e-15] | 9.925e-2 [1.568e-1] | 4.750e-15 [1.568e-14] | 3.522e-4 [1.343e-3] |
