# Interactive Notebooks

This directory contains ready-to-run Jupyter Notebooks for interactive experimentation and visualization.

---

## 📓 Available Notebooks

| Notebook | Topic | Hardware | Target |
| :--- | :--- | :---: | :--- |
| **[`quickstart_interactive_demo.ipynb`](quickstart_interactive_demo.ipynb)** | 1-Click Interactive Demo (Ravine, Vision, LoRA) | CPU / GPU (Colab) | Quick tour of CLOVES: 2D canyon, classification net, and LoRA tuning. |
| **[`01_cifar10_vision_benchmark.ipynb`](01_cifar10_vision_benchmark.ipynb)** | CIFAR-10 & ResNet-18 Benchmark | 1x GPU (T4 / V100 / Colab) | Demonstrates CLOVES on deep vision, comparing convergence against AdamW and LION. |
| **[`02_llm_qlora_tuning_benchmark.ipynb`](02_llm_qlora_tuning_benchmark.ipynb)** | LLM Instruction Tuning (QLoRA) | 1x GPU (T4 16GB) | Demonstrates CLOVES-Apex on parameter-efficient LoRA tuning with asymmetric velocity scaling. |
| **[`03_2d_ravine_simulation.ipynb`](03_2d_ravine_simulation.ipynb)** | 2D Ill-Conditioned Ravine ($\kappa=100$) | CPU / GPU | Visualizes optimizer trajectory, showing how CLOVES suppresses transversal canyon wall oscillation. |

---

## 🚀 Running on Google Colab or Kaggle

1. In any notebook, install the latest package:
   ```bash
   !pip install cloves
   ```
2. Or clone and install locally:
   ```bash
   !git clone https://github.com/huynhldvn/CLOVES.git
   !pip install -e ./CLOVES
   ```
