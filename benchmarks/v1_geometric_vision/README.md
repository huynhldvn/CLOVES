# CLOVES v1.0: Geometric Ravine Damping & Deep Vision

- **Primary Domain**: Ill-Conditioned Optimization, ResNet-18, DenseNet-121

## 1. Overview
CLOVES v1.0 implements first-order directional cleft damping to suppress transversal oscillations across narrow valley walls without requiring a second-moment state vector. It maintains strict 1-state memory parity (4 bytes per parameter), saving 50% state memory relative to AdamW.

### Variants
1. **CLOVES-Bisector**: Computes the unit angular bisector between stochastic gradient $g_t$ and momentum $m_t$.
2. **CLOVES-Perpendicular**: Decomposes update dynamics into longitudinal progression and transversal orthogonal rejection $g_{\perp}$.
3. **CLOVES-Affine**: Continuous affine subspace projection balancing acceleration and wall damping.

---

## 2. Preliminary Results

> [!NOTE]
> **Objectivity & Validation Protocol**: All neural network metrics (ResNet-18, DenseNet-121) are strictly evaluated on held-out **Validation Sets** (`val_loss`, `val_acc`), never on training batches, ensuring objective and unbiased generalization metrics. Synthetic canyon benchmarks report the exact objective loss $f(x)$.
> 
> **Metric Interpretation**: **↑** = Higher is better &nbsp;|&nbsp; **↓** = Lower is better

### A. 2D Ill-Conditioned Ravine ($\kappa = 100, 250$ steps)

<p align="center">
  <img src="../../docs/images/ravine_3_cloves_variants.png" alt="2D Ravine Trajectory Comparison and Loss Decay" width="95%">
</p>

| Optimizer | Final Objective Loss (↓) | Distance to Minimum (↓) | Oscillation Behavior |
| :--- | :---: | :---: | :--- |
| **CLOVES-Bisector** | **$7.11 \times 10^{-5}$** | **0.0119** | Zero oscillation along canyon floor |
| **CLOVES-Affine** | **$1.00 \times 10^{-4}$** | **0.0142** | Stable convergence |
| **CLOVES-Perpendicular** | **$1.40 \times 10^{-4}$** | **0.0167** | Transverse force suppressed |
| AdamW | 20.20 | 6.3556 | Severe transversal bouncing |
| LION | 24.52 | 7.0000 | Trapped in wall oscillation |

### B. ResNet-18 on CIFAR-10
| Optimizer | 10-Epoch Acc (↑) | 20-Epoch Acc (↑) | Final Val Loss (↓) | Peak VRAM (↓) |
| :--- | :---: | :---: | :---: | :---: |
| **CLOVES-Perpendicular** | **89.04%** | **93.07%** | **0.2237** | **696.0 MB** |
| **CLOVES-Affine** | 88.76% | 92.51% | 0.2310 | 696.0 MB |
| **CLOVES-Bisector** | 88.13% | 92.14% | 0.2405 | 696.0 MB |
| AdamW | 88.47% | 92.35% | 0.2512 | 827.8 MB |
| LION | 84.44% | 89.20% | 0.3840 | 696.0 MB |

### C. DenseNet-121 on CIFAR-100 (30 Epochs)
| Optimizer | Top-1 Accuracy (↑) | Top-5 Accuracy (↑) | Final Val Loss (↓) |
| :--- | :---: | :---: | :---: |
| **CLOVES (Proposed)** | **73.98%** | **92.03%** | **1.3993** |
| AdamW | 72.77% | 92.02% | 1.4435 |
| SGD-M | 72.56% | 91.66% | 1.4304 |
