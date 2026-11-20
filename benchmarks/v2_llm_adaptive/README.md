# CLOVES v2.0: LLM Fine-Tuning & Adaptive Cleft Scaling

- **Primary Domain**: Large Language Model Fine-Tuning, LoRA / QLoRA, Multi-GPU Pipelines
- **Scope Note**: These variants are specifically tuned and validated for autoregressive language models (LLaMA-3, Qwen) and LoRA adapters. They have not yet been evaluated on computer vision, speech, or tabular tasks.

## 1. Overview
In Transformer and LoRA parameter optimization, parameter gradient norms are frequently near $\|g\| \sim 10^{-4}$, causing classical scalar first-order updates to suffer from magnitude mismatch. CLOVES v2.0 addresses this via coordinate cleft alignment, adaptive second-moment curvature scaling, continuous soft-cleft tanh damping, and asymmetric LoRA A/B velocity ratios.

### Variants
1. **CLOVES-Adaptive**: Combines coordinate cleft overstep ($c_i \in \{q, \gamma\}$) with running second-moment variance scaling ($\hat{v}_t$).
2. **CLOVES-LLM**: 1-state sign overstep optimizer (4 bytes/param), eliminating LION's loss drift and saving 50% state memory vs AdamW.
3. **CLOVES-Apex-Asymmetric**: Dedicated LoRA architecture engine featuring asymmetric acceleration ratios ($q_A > q_B$), smooth tanh damping, and cosine cleft annealing.
4. **CLOVES-Matrix-Polar**: Riemannian Stiefel manifold optimizer using Newton-Schulz quintic polar decomposition for 2D weight matrices.

---

## 2. Preliminary Results

> [!NOTE]
> **Objectivity & Validation Protocol**: All loss values reported for neural network benchmarks (LLaMA-3, Qwen-1.5B, Qwen-7B) are held-out **Validation Losses** (`val_loss`), strictly evaluated on the validation split under identical batch sizes and seeds to ensure unbiased, objective comparisons. Synthetic ravine benchmarks report the exact mathematical objective loss $f(x)$.
> 
> **Metric Interpretation**: **↑** = Higher is better &nbsp;|&nbsp; **↓** = Lower is better

### A. 1000-Dimensional Non-Convex Ravine ($\kappa = 10,000, 500$ steps)
| Optimizer | State Memory (↓) | Initial Loss | Final Objective Loss (500 Steps) (↓) | Relative Advantage |
| :--- | :---: | :---: | :---: | :--- |
| **CLOVES-Adaptive** | 2 States (8 B) | 543,195.56 | **17.12** | **-92.0% loss vs AdamW** (12.4x lower) |
| AdamW | 2 States (8 B) | 543,195.56 | 213.23 | Baseline |
| **CLOVES-LLM** | 1 State (4 B) | 543,195.56 | **12,297.97** | **-60.2% loss vs LION** (2.5x lower) |
| LION | 1 State (4 B) | 543,195.56 | 30,861.02 | Baseline |

### B. LLM Instruction Tuning Benchmarks (Kaggle Multi-GPU Fleet)
| Benchmark / Model | Baseline Optimizer | Baseline Val Loss (↓) | CLOVES Variant | CLOVES Val Loss (↓) | Outcome |
| :--- | :--- | :---: | :--- | :---: | :---: |
| **LLaMA-3 QLoRA (500 steps)** | AdamW | 5.7269 | **CLOVES-Perp (Corrected)** | **5.7338** | Matched AdamW, beat Muon |
| **LLaMA-3 QLoRA (500 steps)** | Muon | 5.7941 | **CLOVES-Matrix-Polar** | **5.7822** | Outperformed Muon |
| **Qwen-1.5B (1000 steps)** | AdamW | 0.4298 | **CLOVES-Apex (Tuned)** | **0.4295** | Parity with AdamW (-0.0003) |
| **Qwen-7B (500 steps)** | AdamW | 0.5583 | **CLOVES-Apex (Annealed)** | **0.5573** | Parity with AdamW (-0.0010, -3.36 GB VRAM) |
| **Qwen-7B (500 steps)** | LION | 0.6365 | **CLOVES-Sign-Apex** | **0.6249** | Outperformed LION (-0.0116, -3.37 GB VRAM) |

Detailed raw measurements are recorded in [`benchmarks/results_summary.csv`](../results_summary.csv).

---

## 3. References & Baseline Implementations

- **LION (EvoLved Sign Momentum)**:
  - Paper: Chen et al., *Symbolic Discovery of Optimization Algorithms*, NeurIPS 2023. arXiv: [2302.06675](https://arxiv.org/abs/2302.06675).
  - Principle: Discovered via program synthesis; utilizes coordinate-wise `sign` update with EMA momentum (1 state buffer, 4 bytes/param).
  - Repositories: [google-research/lion](https://github.com/google-research/google-research/tree/master/lion) | [lucidrains/lion-pytorch](https://github.com/lucidrains/lion-pytorch)
- **Muon (Momentum Orthogonalized by Newton-schulz)**:
  - Reference: Keller Jordan et al., *Muon: An Optimizer for Hidden Layers in Neural Networks*, 2024.
  - Principle: Orthogonalizes 2D momentum updates using Newton-Schulz quintic polar iterations on Transformer hidden weights.
  - Repository: [KellerJordan/Muon](https://github.com/KellerJordan/Muon)
