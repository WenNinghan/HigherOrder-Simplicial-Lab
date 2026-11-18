# HigherOrder-Simplicial-Lab: Hodge-Laplacian Spectral Topology, Helmholtz-Hodge Decomposition, Simplicial Neural Networks & Higher-Order Phase Transitions

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-pytest-blue.svg)](tests/test_simplicial.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-0f766e.svg)](LICENSE)

**HigherOrder-Simplicial-Lab** is a self-contained research laboratory at the intersection of **Algebraic Topology**, **Discrete Exterior Calculus (DEC)**, **Topological / Geometric Deep Learning**, and **Higher-Order Network Dynamics**. Unlike pairwise graph frameworks restricted to 0-simplex node signals and 0-Laplacian ($`L_{0}`$) diffusion, this repository builds an oriented 2-simplicial complex engine from first principles—unifying exact **1-Hodge Laplacian spectral topology**, **orthogonal Helmholtz-Hodge flow decomposition**, **orientation-equivariant Hodge Simplicial Neural Networks (`HodgeSimplicialNN`)**, and **explosive higher-order contagion hysteresis**.

---

## Visual Highlights & Key Empirical Findings

### 1. Exact Orthogonal Helmholtz-Hodge Decomposition on Punctured Manifolds

![Helmholtz-Hodge Decomposition](assets/helmholtz_hodge_decomposition.png)

On our 2-hole planar punctured simplicial complex ($`N_{0} = 103`$, $`N_{1} = 244`$, $`N_{2} = 140`$, Betti numbers $`\beta_{0} = 1, \beta_{1} = 2`$), any edge 1-cochain $`f \in \mathbb{R}^{N_{1}}`$ splits uniquely into curl-free gradient, divergence-free rotational curl, and cavity-circulating harmonic flows with **machine-precision orthogonality** ($`\max \lvert \langle f_{a}, f_{b} \rangle \rvert = 5.33 \times 10^{-16}`$) and **Parseval energy additivity to floating-point precision** ($`\lvert \lVert f \rVert_{2}^{2} - \sum_{k} \lVert f_{k} \rVert_{2}^{2} \rvert = 0.0`$).

### 2. Higher-Order Simplicial Neural Networks vs. Pairwise & Isotropic Baselines

![Hodge-SNN Benchmark](assets/hodge_snn_benchmark.png)

The node-potential baseline reconstructs edge signals through $`B_1^\top \hat{\phi}`$, so its output lies in the gradient subspace. The Hodge network handles gradient, curl and harmonic components separately. At 40% missing edges it achieves $`R^2`$ of 0.9309–0.9767 on the five synthetic flow settings. A linear subspace fit to the same complete training flows recovers the missing entries with $`R^2`$ of **0.99983–0.99993**. The synthetic generator confines each regime to 10–22 independent directions. [Linear control results](results/linear_imputation_control.json)

On the original two-hole mesh, training on tight loops and testing on wider, noisier loops gives **94.44% ± 1.04 percentage points** for the Hodge network. A nearest-centroid classifier using two harmonic coordinates scores **100%** on all three splits. In a separate frozen-model experiment on three unseen meshes, the Hodge network averages **77.50%** across meshes and training seeds. The harmonic classifier recovers all **480 of 480** held-out loops using the known, ordered hole centers to align coordinates. The four classes have distinct signed windings, and the synthetic noise lies in gradient and curl directions removed by the harmonic projection. [Cross-mesh predictions and scores](results/mesh_transfer.json)

### 3. Explosive First-Order Phase Transitions & Bistable Hysteresis in Simplicial Contagion

![Simplicial Phase Transition](assets/simplicial_phase_transition.png)

Incorporating synergistic 3-body 2-simplex interactions ($`\lambda_{2} = 2.8 > 1`$) transforms classical continuous epidemic/reaction onset ($`\lambda_{2} = 0.8 < 1`$) into an **explosive discontinuous first-order tipping point** with a wide bistable hysteresis loop ($`\Delta \rho_{\max} = 0.6139`$ across $`\lambda_{1} \in [0.547, 1.000]`$), separating sub-threshold decay from self-sustaining endemic activation across an unstable separatrix $`\rho_{-}^{*}`$.

### 4. Spectral Topology of the 1-Hodge Laplacian Under Cavity Surgery

![Hodge Spectrum Cavity Filling](assets/hodge_spectrum_cavity_filling.png)

As 2-simplices progressively fill open triangular faces of the 1-skeleton, local cycle modes lift off zero one by one ($`\beta_{1}: 142 \to 2`$), stabilizing at an exact topological plateau $`\beta_{1} = \dim \ker(L_{1}) = 2`$ governed by the two macroscopic obstacles—until targeted topological surgery caps Cavity A ($`\beta_{1}: 2 \to 1`$) and Cavity B ($`\beta_{1}: 1 \to 0`$).

---

## Mathematical Foundations

### 1. Oriented Simplicial Complexes & Exact Chain Boundaries

An abstract 2-simplicial complex $`\mathcal{K} = (\mathcal{V}, \mathcal{E}, \mathcal{T})`$ consists of $`N_{0}`$ vertices (0-simplices), $`N_{1}`$ canonically oriented edges $`[i, j]`$ ($`i < j`$), and $`N_{2}`$ canonically oriented filled triangles $`[i, j, k]`$ ($`i < j < k`$), closed under taking faces. The signed boundary operators $`B_{1} \in \{-1, 0, +1\}^{N_{0} \times N_{1}}`$ and $`B_{2} \in \{-1, 0, +1\}^{N_{1} \times N_{2}}`$ act on canonical chains via:

```math
\partial_{1}[i, j] = [j] - [i], \qquad \partial_{2}[i, j, k] = [j, k] - [i, k] + [i, j]
```

By the fundamental lemma of homological algebra, the boundary of a boundary vanishes identically:

```math
B_{1} B_{2} = 0 \in \mathbb{R}^{N_{0} \times N_{2}}
```

### 2. Hodge Laplacians, Betti Numbers & Helmholtz-Hodge Decomposition

Dualizing $`\partial_{1}, \partial_{2}`$ yields the discrete gradient $`\mathrm{grad} = B_{1}^{\top}`$, discrete curl $`\mathrm{curl} = B_{2}^{\top}`$, and negative divergence $`-\mathrm{div} = B_{1}`$. The 0-Laplacian and **1-Hodge Laplacian** take the form:

```math
L_{0} = B_{1} B_{1}^{\top} \in \mathbb{R}^{N_{0} \times N_{0}}, \qquad L_{1} = \underbrace{B_{1}^{\top} B_{1}}_{L_{1,\mathrm{down}}} + \underbrace{B_{2} B_{2}^{\top}}_{L_{1,\mathrm{up}}} \in \mathbb{R}^{N_{1} \times N_{1}}
```

Because $`B_{1} B_{2} = 0`$, the lower and upper Hodge operators annihilate each other ($`L_{1,\mathrm{down}} L_{1,\mathrm{up}} = 0`$), inducing the **discrete Orthogonal Helmholtz-Hodge Decomposition** on the 1-cochain space $`C^{1}(\mathcal{K}; \mathbb{R}) \cong \mathbb{R}^{N_{1}}`$:

```math
\mathbb{R}^{N_{1}} = \mathrm{im}(B_{1}^{\top}) \oplus \mathrm{im}(B_{2}) \oplus \ker(L_{1}), \qquad f = f_{\mathrm{grad}} + f_{\mathrm{curl}} + f_{\mathrm{harm}}
```

where $`f_{\mathrm{grad}} = B_{1}^{\top} \phi`$ is irrotational ($`B_{2}^{\top} f_{\mathrm{grad}} = 0`$), $`f_{\mathrm{curl}} = B_{2} \psi`$ is divergence-free ($`B_{1} f_{\mathrm{curl}} = 0`$), and $`f_{\mathrm{harm}} \in \ker(B_{1}) \cap \ker(B_{2}^{\top}) = \ker(L_{1})`$ encodes global topological circulation around 1-dimensional holes, with dimension equal to the first Betti number:

```math
\beta_{0} = N_{0} - \mathrm{rank}(B_{1}) = \dim \ker(L_{0}), \qquad \beta_{1} = N_{1} - \mathrm{rank}(B_{1}) - \mathrm{rank}(B_{2}) = \dim \ker(L_{1})
```

### 3. Orientation-Equivariant Hodge Simplicial Neural Networks (`HodgeSimplicialNN`)

To respect cochain physics and decoupled Hodge subspaces, each layer of `HodgeSimplicialNN` applies separate polynomial spectral filters over normalized $`\hat{L}_{1,\mathrm{down}}`$ and $`\hat{L}_{1,\mathrm{up}}`$ alongside bias-free linear maps and odd activation $`\sigma(z) = \tanh(z)`$:

```math
H^{(l+1)} = H^{(l)} + \tanh\!\left( H^{(l)} W_{0}^{(l)} + (M \odot H^{(l)}) W_{\mathrm{obs}}^{(l)} + \sum_{k=1}^{K} \hat{L}_{1,\mathrm{down}}^{k} H^{(l)} W_{\mathrm{down}, k}^{(l)} + \sum_{k=1}^{K} \hat{L}_{1,\mathrm{up}}^{k} H^{(l)} W_{\mathrm{up}, k}^{(l)} \right)
```

The readout includes the harmonic projector. For a diagonal matrix $D$ of independently selected edge signs, transform the input to $Dx$, the boundaries to $B_1D$ and $DB_2$, and keep the observation mask fixed. The edge output transforms by $D$. The benchmark reports both the global sign-reversal check and a random local reorientation. The classifier pools absolute magnitudes, RMS magnitudes and pairings with consistently oriented cavity boundaries; its output is invariant to these choices of edge orientation.

### 4. Higher-Order Simplicial Contagion & Synergistic Bifurcations

Following Iacopini et al. (*Nature Communications* 2019), a node $`i`$ on a 2-simplicial complex acquires activation/infection via pairwise edges at rate $`\beta_{1}`$ and via filled 2-simplices $`(i, j, k)`$ at synergistic rate $`\beta_{2}`$ when both $`j`$ and $`k`$ are simultaneously active, recovering at rate $`\mu`$. With rescaled infectivities $`\lambda_{1} = \beta_{1} \langle k_{1} \rangle / \mu`$ and $`\lambda_{2} = \beta_{2} \langle k_{2} \rangle / \mu`$, the homogeneous mean-field density $`\rho(\tau)`$ ($`\tau = \mu t`$) satisfies:

```math
\frac{d\rho}{d\tau} = -\rho + \lambda_{1} \rho (1 - \rho) + \lambda_{2} \rho^{2} (1 - \rho)
```

Non-trivial equilibria obey $`\lambda_{2} (\rho^{*})^{2} + (\lambda_{1} - \lambda_{2})\rho^{*} + (1 - \lambda_{1}) = 0`$, yielding:

```math
\rho_{\pm}^{*} = \frac{(\lambda_{2} - \lambda_{1}) \pm \sqrt{(\lambda_{1} + \lambda_{2})^{2} - 4\lambda_{2}}}{2\lambda_{2}}, \qquad \lambda_{1,c}^{\mathrm{back}} = 2\sqrt{\lambda_{2}} - \lambda_{2} \quad (\lambda_{2} > 1)
```

---

## Quantitative Benchmark Results

The original same-mesh benchmark uses the four neural backbones, a harmonic classifier and five synthetic flow settings. [Summary](results/benchmark_summary.json) · [Table](results/benchmark_table.csv) · [Training-flow subspace control](results/linear_imputation_control.json)

### Table 1 — Imputation of missing edges and same-mesh loop classification

Imputation uses one train/test draw per flow setting, 150 training signals and 60 test signals. Scores use only missing edges. The loop benchmark uses seeds 42, 101 and 202, with 45 training and 30 test observations per class for each seed. Error bars describe variation across those seeds. All four learned classifiers receive the same boundary-cycle pooling, including the edge MLP. The harmonic control uses training labels only to fit its centroids.

| Backbone or control | Gradient R² | Balanced R² | 75% solenoidal R² | Pure solenoidal R² | 75% solenoidal MSE | Loop accuracy, mean ± SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NodeGCN | 0.8465 | 0.4992 | 0.4067 | 0.4716 | 0.5903 | 23.06% ± 1.71 pp |
| EdgeMLP | -0.0004 | -0.0002 | -0.0001 | 0.0000 | 0.9950 | 55.56% ± 10.42 pp |
| Isotropic-L1-GCN | 0.6686 | 0.4855 | 0.4304 | 0.3864 | 0.5667 | 59.44% ± 8.56 pp |
| HodgeSimplicialNN | 0.9767 | 0.9406 | 0.9309 | 0.9689 | 0.0688 | 94.44% ± 1.04 pp |
| Harmonic centroid | — | — | — | — | — | 100.00% ± 0.00 pp |
| Training-flow subspace | 0.99993 | 0.99983 | 0.99983 | 0.99992 | 0.000167 | — |

The subspace control fits an SVD to the 150 training targets available to all supervised models. Its numerical rank is set by training-data precision. For each test flow it estimates coefficients using only observed entries and scores the held-out entries. No topology or test targets enter the fit.

### Frozen transfer to three unseen meshes

For each of three seeds (42, 101 and 202), we trained on a 13 × 9 mesh with 244 edges. All learned weights and class centroids were then frozen. On each target, we rebuilt its boundary operators, harmonic projector and pooling from its own geometry. The targets vary in grid dimensions, triangulation seed and hole radius. The shifted and fine meshes also move both holes. They share the same planar lattice family and two named cavities.

![Loop accuracy and missing-edge R-squared by source and held-out mesh](assets/mesh_transfer.png)

The orange line separates the source-mesh control from unseen meshes. All four learned classifiers receive the same ordered cavity-boundary pooling. The harmonic centroid uses the ordered target hole centers to orient its two coordinates. Only source labels enter the centroids. Each column uses the same test signals for every method and training seed.

| Metric / method | Source control · 244 edges | Coarse · 212 edges | Shifted · 398 edges | Fine · 474 edges |
| --- | ---: | ---: | ---: | ---: |
| Loop accuracy · harmonic centroid | 100% | 100% | 100% | 100% |
| Loop accuracy · HodgeSNN | 93.96% | 85.42% | 72.08% | 75.00% |
| Loop accuracy · isotropic L1 | 58.75% | 52.08% | 35.21% | 42.29% |
| Loop accuracy · edge MLP | 54.38% | 46.67% | 34.38% | 36.46% |
| Loop accuracy · node GCN | 24.38% | 26.04% | 28.75% | 25.62% |
| Missing-edge R² · HodgeSNN | 0.9256 | 0.9063 | 0.9386 | 0.9397 |
| Missing-edge R² · isotropic L1 | 0.4865 | 0.4465 | 0.5367 | 0.5169 |
| Missing-edge R² · low Hodge modes | 0.6420 | 0.6209 | 0.7007 | 0.7046 |
| Missing-edge R² · known generator modes | 0.999842 | 0.999821 | 0.999907 | 0.999917 |

**Loop classification.** Each seed used 180 source examples for training. Each mesh supplied 160 fresh test trajectories, 40 per class. Across the three held-out meshes, all 480 generated labels agreed with the signed winding after harmonic projection. Across training seeds, HodgeSNN exceeded the other learned backbones on every mesh. Its mean accuracy fell from 93.96% on the source control to 77.50% across the held-out meshes. These results apply to the four winding classes with consistent hole identities across meshes.

**Flow completion.** Each seed used 150 complete source flows for supervision. Each mesh supplied 60 fresh balanced flows with 40% of edges hidden and observation noise at 3% of the signal standard deviation. The low-Hodge control uses the target geometry and fits coefficients to visible edges alone. Its 32 modes minimized error among 8, 16, 32 and 64 modes on 40 independent source validation flows. HodgeSNN exceeds this control on each held-out mesh. A separate control with the simulator's known generating basis (10 gradient, 10 dual-face curl and two harmonic directions) attains R² above 0.9998 on all three. That control quantifies the advantage of knowing how these flows were synthesized. The earlier training-flow SVD lives in the source edge coordinates. [Protocol and runnable code](scripts/evaluate_mesh_transfer.py) · [Predictions, per-signal errors and mesh specifications](results/mesh_transfer.json)

### Table 2: Algebraic Topology, Helmholtz-Hodge Precision & Contagion Bifurcation Diagnostics

| Diagnostic / Invariant | Mathematical Quantity | Measured Value |
| :--- | :--- | :---: |
| Punctured Complex Simplex Counts | $`(N_{0}, N_{1}, N_{2})`$ and Euler $`\chi = N_{0} - N_{1} + N_{2}`$ | $`(103, 244, 140)`$, $`\chi = -1`$ |
| Exact Topological Betti Numbers | $`(\beta_{0}, \beta_{1})`$ via SVD & Hodge spectra | $`(1, 2)`$ |
| Exact Chain Complex Identity | $`\max_{i, p} \lvert (B_{1} B_{2})_{i, p} \rvert`$ | $`0.0`$ (exact) |
| HHD Pairwise Orthogonality Error | $`\max_{a \neq b} \lvert \langle f_{a}, f_{b} \rangle \rvert`$ | $`5.33 \times 10^{-16}`$ |
| HHD Parseval Norm Conservation | $`\lvert \lVert f \rVert_{2}^{2} - (\lVert f_{\mathrm{grad}} \rVert_{2}^{2} + \lVert f_{\mathrm{curl}} \rVert_{2}^{2} + \lVert f_{\mathrm{harm}} \rVert_{2}^{2}) \rvert`$ | $`0.0`$ |
| Discrete Curl-of-Grad / Div-of-Curl | $`\lVert B_{2}^{\top} f_{\mathrm{grad}} \rVert_{\infty}`$ / $`\lVert B_{1} f_{\mathrm{curl}} \rVert_{\infty}`$ | $`1.39 \times 10^{-17}`$ / $`5.55 \times 10^{-17}`$ |
| Orientation Equivariance Error (`float64`) | $`\max \lvert \mathcal{F}_{\theta}(-x) + \mathcal{F}_{\theta}(x) \rvert`$ | $`0.0`$ (exact) |
| Harmonic zero-mode residual | Infinity norm of the 1-Hodge Laplacian applied to the harmonic component | 4.55 × 10⁻¹⁵ |
| Random local edge-orientation flips | Maximum equivariance residual, float64 | 6.11 × 10⁻¹⁶ |
| Filtration Betti Trajectory | 1-Skeleton $`\to`$ Mesh Filled $`\to`$ Cap A $`\to`$ Cap B | $`142 \to 2 \to 1 \to 0`$ |
| Trimolecular Reaction Complex ($`\mathrm{O}_{3} + \mathrm{OH}^{\bullet} + P`$) | $`(N_{0}, N_{1}, N_{2})`$ and $`(\beta_{0}, \beta_{1})`$ | $`(12, 33, 8)`$ and $`(1, 14)`$ |
| Simplicial Contagion Hysteresis ($`\lambda_{2} = 2.8`$) | $`\max_{\lambda_{1}} (\rho_{\mathrm{backward}} - \rho_{\mathrm{forward}})`$ at $`\lambda_{1} = 0.879`$ | $`0.6139`$ (1st-order explosive) |
| Pairwise-Dominated Sweep ($`\lambda_{2} = 0.8`$) | $`\max_{\lambda_{1}} (\rho_{\mathrm{backward}} - \rho_{\mathrm{forward}})`$ | $`0.0336`$ (2nd-order continuous) |

---

## Repository Architecture

```text
HigherOrder-Simplicial-Lab/
├── simplicial_lab/
│   ├── __init__.py          # Public API with PEP-562 lazy PyTorch loading
│   ├── complexes.py         # Oriented SimplicialComplex, exact B1/B2, punctured/RSC/chemical generators
│   ├── hodge.py             # L0/L1 Hodge Laplacians, Betti numbers, harmonic basis, exact HHD & filtration
│   ├── models.py            # HodgeSimplicialNN, NodeGCN, Isotropic-L1-GCN, EdgeMLP & benchmark tasks
│   └── contagion.py         # Simplicial Contagion ODE, stochastic Markovian SIS & hysteresis sweeps
├── scripts/
│   ├── run_benchmarks.py    # Same-mesh metrics and four figures
│   ├── evaluate_linear_imputation.py  # Same-mesh training-flow subspace control
│   └── evaluate_mesh_transfer.py      # Frozen-model cross-mesh benchmark and figure
├── tests/
│   └── test_simplicial.py   # Rigorous algebraic, spectral, equivariance & dynamical pytest suite
├── assets/                  # Generated 300-DPI visual diagnostics
├── results/                 # Same-mesh metrics, linear control and cross-mesh predictions
├── app.py                   # 3-tab interactive Streamlit workbench
├── pyproject.toml           # Modern PEP-621 package configuration
├── requirements.txt         #Pinned minimum scientific & deep learning dependencies
└── LICENSE                  # MIT License
```

---

## Quickstart & Reproducibility

### 1. Installation

```bash
pip install -e ".[dev]"
```

### 2. Run Unit Test Suite

```bash
pytest -v
```

### 3. Reproduce All Benchmarks & 300-DPI Figures

```bash
python scripts/run_benchmarks.py
python scripts/evaluate_linear_imputation.py
python scripts/evaluate_mesh_transfer.py --device cpu
```

The recorded mesh-transfer run used `--device cuda` with PyTorch 2.7.1 and NumPy 2.2.6. Both device settings use the same samples, splits and training budget.

### 4. Launch Interactive Streamlit Workbench

```bash
streamlit run app.py
```

### 5. Minimal Python API Example

```python
from simplicial_lab import (
    planar_punctured_complex,
    compute_betti_numbers,
    synthesize_hodge_flow,
    helmholtz_hodge_decomposition,
    run_hysteresis_sweep,
)

# 1. Build 2-hole planar simplicial complex and verify exact chain complex identity
sc = planar_punctured_complex(seed=42)
assert sc.verify_boundary_identity() == 0.0
beta_0, beta_1 = compute_betti_numbers(sc)  # (1, 2)

# 2. Orthogonal Helmholtz-Hodge Decomposition f = f_grad + f_curl + f_harm
flow = synthesize_hodge_flow(sc, grad_weight=1.15, curl_weight=1.0, harm_weight=1.05)
hhd = helmholtz_hodge_decomposition(sc, flow["f_total"])
print("Max orthogonality error:", hhd.orthogonality_errors["max"])  # ~5e-16 in the recorded run

# 3. Higher-order simplicial contagion hysteresis loop
sweep = run_hysteresis_sweep(lambda_2=2.8)
print("Explosive bistable hysteresis gap:", sweep.max_hysteresis_gap)  # ~0.614
```

---

## Core References

1. Lim, L.-H. (2020). *Hodge Laplacians on Graphs*. SIAM Review, 62(3), 685–715.
2. Schaub, M. T., Benson, A. R., Horn, P., Lippner, G., & Jadbabaie, A. (2020). *Random Walks on Simplicial Complexes and the Normalized Hodge 1-Laplacian*. SIAM Review, 62(2), 353–391.
3. Ebli, S., Defferrard, M., & Spreemann, G. (2020). *Simplicial Neural Networks*. NeurIPS Workshop on Topological Data Analysis and Beyond.
4. Yang, M., Isufi, E., & Leus, G. (2022). *Simplicial Convolutional Neural Networks*. IEEE Transactions on Signal Processing, 70, 959–971.
5. Iacopini, I., Petri, G., Barrat, A., & Latora, V. (2019). *Simplicial models of social contagion*. Nature Communications, 10, 2485.
6. Battiston, F., Cencetti, G., Iacopini, I., Latora, V., Lucas, M., Patania, A., Young, J.-G., & Petri, G. (2020). *Networks beyond pairwise interactions: Structure and dynamics*. Physics Reports, 874, 1–92.
