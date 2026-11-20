"""
2D Ill-Conditioned Ravine Simulation & Trajectory Visualization
Reproduces the empirical contour trajectory and loss decay curves comparing
CLOVES variants against AdamW, LION, and Momentum baselines.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Ensure local cloves package can be imported
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from cloves import CLOVES_Bisector, CLOVES_Perpendicular, CLOVES_Affine, CLOVES_LLM

def run_trajectory(optimizer_fn, steps=250, x0=10.0, y0=1.0, kappa=100.0):
    """
    Simulates optimization on anisotropic quadratic canyon:
    f(x, y) = 0.5 * (x^2 + kappa * y^2)
    Minimum at (0, 0).
    """
    point = torch.tensor([x0, y0], dtype=torch.float32, requires_grad=True)
    optimizer = optimizer_fn(point)
    
    trajectory = [[point[0].item(), point[1].item()]]
    loss_history = []
    
    for _ in range(steps):
        optimizer.zero_grad()
        loss = 0.5 * (point[0] ** 2 + kappa * (point[1] ** 2))
        loss.backward()
        optimizer.step()
        
        trajectory.append([point[0].item(), point[1].item()])
        loss_history.append(loss.item())
        
    return np.array(trajectory), np.array(loss_history)

def main():
    print("[*] Running 2D Ill-Conditioned Ravine Simulation (kappa=100, 250 steps)...")
    
    # Empirically verified configs matching benchmarks
    configs = {
        "CLOVES-Bisector": (
            lambda p: CLOVES_Bisector([p], lr=0.015, beta=0.9, q=1.60, gamma=0.10, weight_decay=0.0),
            "#E74C3C", 2.4
        ),
        "CLOVES-Perpendicular": (
            lambda p: CLOVES_Perpendicular([p], lr=0.020, beta=0.9, q=1.60, gamma=0.10, gamma_perp=0.25, weight_decay=0.0),
            "#8E44AD", 2.4
        ),
        "CLOVES-Affine": (
            lambda p: CLOVES_Affine([p], lr=0.020, beta=0.9, q=1.60, gamma=0.10, lambda_affine=0.20, weight_decay=0.0),
            "#D35400", 2.4
        ),
        "CLOVES-LLM (Sign)": (
            lambda p: CLOVES_LLM([p], lr=0.005, betas=(0.9, 0.99), q=1.20, gamma=0.60, weight_decay=0.0),
            "#2980B9", 2.0
        ),
        "AdamW": (
            lambda p: torch.optim.AdamW([p], lr=0.020, betas=(0.9, 0.999), weight_decay=0.0),
            "#27AE60", 1.5
        ),
        "SGD-M (0.9)": (
            lambda p: torch.optim.SGD([p], lr=0.015, momentum=0.9, weight_decay=0.0),
            "#7F8C8D", 1.2
        ),
    }
    
    trajectories = {}
    losses = {}
    
    for name, (opt_fn, _, _) in configs.items():
        traj, l_hist = run_trajectory(opt_fn, steps=250, x0=10.0, y0=1.0, kappa=100.0)
        trajectories[name] = traj
        losses[name] = l_hist
        print(f"  - {name:<22} Final Objective Loss: {l_hist[-1]:.6e} | Final pos: ({traj[-1, 0]:.3f}, {traj[-1, 1]:.3e})")
        
    # Generate visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left Panel: 2D Contour & Trajectories
    x_grid = np.linspace(-1, 11, 400)
    y_grid = np.linspace(-1.5, 1.5, 400)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = 0.5 * (X ** 2 + 100.0 * (Y ** 2))
    
    ax1.contour(X, Y, Z, levels=np.logspace(-2, 3, 35), cmap="Blues", alpha=0.50)
    
    for name, (opt_fn, color, lw) in configs.items():
        traj = trajectories[name]
        ax1.plot(
            traj[:, 0], traj[:, 1],
            label=f"{name} (Loss: {losses[name][-1]:.2e})",
            color=color, linewidth=lw, marker='o', markersize=3, markevery=25
        )
        
    ax1.plot(0, 0, 'k*', markersize=14, label="Global Minimum (0, 0)")
    ax1.set_title("2D Ravine Trajectory Comparison (CLOVES vs. Baselines)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("x (Slow Ravine Axis, curvature = 1)")
    ax1.set_ylabel("y (Steep Ravine Axis, curvature = 100)")
    ax1.legend(loc="upper right", framealpha=0.92, fontsize=8.5)
    ax1.grid(True, alpha=0.3)
    
    # Right Panel: Loss Decay (Log Scale)
    for name, (opt_fn, color, lw) in configs.items():
        ax2.semilogy(losses[name], label=name, color=color, linewidth=lw)
        
    ax2.set_title("Ravine Loss Decay Convergence (Log Scale)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Optimization Step")
    ax2.set_ylabel("Objective Loss f(x, y)")
    ax2.legend(loc="upper right", framealpha=0.92, fontsize=8.5)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "images"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ravine_3_cloves_variants.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    
    print(f"\n[+] Visualization successfully saved to: {out_path}")

if __name__ == "__main__":
    main()
