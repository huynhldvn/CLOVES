# Empirical Benchmark Suites & Reproducibility Guidelines

This directory contains the experimental logs, execution configurations, and benchmark summaries for the CLOVES optimizer suite.

---

## 📂 Structure

- **[`v1_geometric_vision/`](v1_geometric_vision/)**:
  - Focus: First-order geometric cleft damping on ill-conditioned 2D ravines ($\kappa = 100$) and deep vision architectures (ResNet-18, DenseNet-121).

- **[`v2_llm_adaptive/`](v2_llm_adaptive/)**:
  - Focus: Adaptive second-moment cleft scaling, 1-state sign optimization, and asymmetric LoRA velocity scaling on 1000D manifolds and LLMs (LLaMA-3, Qwen-1.5B, Qwen-7B).

- **[`results_summary.csv`](results_summary.csv)**:
  - Raw harvested benchmark metrics from all 8 parallel cloud GPU test suites, including validation loss, peak VRAM, step latency, and total training time.
