"""
Generates an intuitive, high-clarity animated GIF comparing 2D contour and 3D surface optimization dynamics
between CLOVES variants and standard baselines (AdamW, LION, SGD-M).

Key 3D Visual Cues:
1. Natural 30-degree oblique perspective (elev=42, azim=-126) looking down the open half-pipe canyon.
2. Vertical drop lines (stems) connecting each 3D point to its ground projection at z=0 for instant depth perception.
3. Blue dashed centerline marking the canyon floor (y=0).
4. Viridis curvature-shaded surface + ground shadow contours without surface occlusion.
5. Real-time step counter and dynamic loss readout in the legend.
Saves output to docs/images/cloves_optimization_dynamics_2d_3d.gif
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from PIL import Image

# Ensure local cloves package can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from cloves import CLOVES_Bisector, CLOVES_Perpendicular, CLOVES_LLM

def run_simulation(steps=100, x0=10.0, y0=1.0, kappa=70.0):
    """
    Simulates optimization on an anisotropic quadratic canyon:
    f(x, y) = 0.5 * (x^2 + kappa * y^2)
    Minimum at (0, 0).
    """
    configs = {
        "CLOVES-Perpendicular": (
            lambda p: CLOVES_Perpendicular([p], lr=0.020, beta=0.9, q=1.60, gamma=0.10, gamma_perp=0.25, weight_decay=0.0),
            "#9333ea", 2.8, "o", "Purple"
        ),
        "CLOVES-Bisector": (
            lambda p: CLOVES_Bisector([p], lr=0.015, beta=0.9, q=1.60, gamma=0.10, weight_decay=0.0),
            "#ef4444", 2.4, "s", "Red"
        ),
        "AdamW": (
            lambda p: torch.optim.AdamW([p], lr=0.020, betas=(0.9, 0.999), weight_decay=0.0),
            "#10b981", 1.9, "^", "Green"
        ),
        "LION": (
            lambda p: CLOVES_LLM([p], lr=0.008, betas=(0.9, 0.99), q=1.0, gamma=1.0, weight_decay=0.0),
            "#0284c7", 1.9, "D", "Blue"
        ),
        "SGD-M": (
            lambda p: torch.optim.SGD([p], lr=0.015, momentum=0.9, weight_decay=0.0),
            "#64748b", 1.5, "x", "Gray"
        ),
    }

    trajectories = {}
    losses = {}

    for name, (opt_fn, color, lw, marker, col_name) in configs.items():
        point = torch.tensor([x0, y0], dtype=torch.float32, requires_grad=True)
        opt = opt_fn(point)
        traj = [[point[0].item(), point[1].item()]]
        loss_hist = [0.5 * (point[0].item()**2 + kappa * (point[1].item()**2))]

        for _ in range(steps):
            opt.zero_grad()
            loss = 0.5 * (point[0] ** 2 + kappa * (point[1] ** 2))
            loss.backward()
            opt.step()
            traj.append([point[0].item(), point[1].item()])
            loss_hist.append(loss.item())

        trajectories[name] = np.array(traj)
        losses[name] = np.array(loss_hist)

    return configs, trajectories, losses, kappa

def generate_animation():
    print("[*] Running optimization simulation...")
    steps = 100
    configs, trajectories, losses, kappa = run_simulation(steps=steps)

    print("[*] Preparing 2D & 3D visualization mesh...")
    # Grid for 2D & 3D surfaces
    x_grid = np.linspace(-1.5, 11.5, 160)
    y_grid = np.linspace(-1.3, 1.3, 160)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = 0.5 * (X ** 2 + kappa * (Y ** 2))

    fig = plt.figure(figsize=(14.2, 5.6), dpi=80)
    fig.patch.set_facecolor('#ffffff')

    # Subplot 1: 2D Contour
    ax1 = fig.add_subplot(1, 2, 1)
    # Subplot 2: 3D Surface
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    # 1. 2D Contour Setup
    levels = np.logspace(-2, 3, 26)
    ax1.contour(X, Y, Z, levels=levels, cmap="viridis", alpha=0.45, linewidths=1.0)
    # Mark canyon floor centerline
    ax1.axhline(0, color="#3b82f6", linestyle="--", alpha=0.6, label="Valley Floor (y=0)")
    ax1.plot(0, 0, marker="*", markersize=16, color="#eab308", markeredgecolor='black', label="Optimum (0, 0)", zorder=20)
    ax1.set_xlim(-1.5, 11.5)
    ax1.set_ylim(-1.3, 1.3)
    ax1.set_xlabel("x (Slow Ravine Axis, curvature = 1)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("y (Steep Canyon Wall, curvature = 70)", fontsize=10, fontweight="bold")
    ax1.set_title("2D Contour: Transversal Wall Damping", fontsize=12, fontweight="bold", pad=10)
    ax1.grid(True, linestyle="--", alpha=0.3)

    # 2. 3D Surface Setup (Half-pipe canyon)
    x_3d = np.linspace(-1.5, 11.5, 40)
    y_3d = np.linspace(-1.3, 1.3, 40)
    X3, Y3 = np.meshgrid(x_3d, y_3d)
    Z3 = 0.5 * (X3 ** 2 + kappa * (Y3 ** 2))

    # Semi-transparent shaded surface with viridis colormap
    ax2.plot_surface(X3, Y3, Z3, cmap="viridis", alpha=0.20, rstride=2, cstride=2,
                     edgecolor="#cbd5e1", linewidth=0.25, antialiased=True)

    # Valley floor centerline in 3D
    x_fl = np.linspace(-1.5, 11.5, 50)
    z_fl = 0.5 * (x_fl ** 2)
    ax2.plot(x_fl, np.zeros_like(x_fl), z_fl, color="#3b82f6", linestyle="--", lw=1.6, label="Valley Floor")

    # Floor contours for ground plane reference
    ax2.contour(X3, Y3, Z3, levels=np.logspace(-1, 3, 10), zdir='z', offset=0, cmap="viridis", alpha=0.25, linewidths=0.7)

    # Optimum star in 3D
    ax2.plot([0], [0], [0], marker="*", markersize=17, color="#eab308", markeredgecolor='black', label="Optimum (0,0,0)", zorder=25)
    ax2.set_xlim(-1.5, 11.5)
    ax2.set_ylim(-1.3, 1.3)
    ax2.set_zlim(0, 75)

    # Intuitive 30-degree top-down oblique view (elev=42, azim=-126)
    # Flow moves naturally from Upper-Right down to Lower-Left
    ax2.view_init(elev=42, azim=-126)

    ax2.set_xlabel("x", fontsize=9, fontweight="bold")
    ax2.set_ylabel("y", fontsize=9, fontweight="bold")
    ax2.set_zlabel("Loss f(x, y)", fontsize=9, fontweight="bold")
    ax2.set_title("3D Canyon Descent Dynamics", fontsize=12, fontweight="bold", pad=10)

    # Artist handles
    lines_2d = {}
    points_2d = {}
    lines_3d = {}
    points_3d = {}
    stems_3d = {}
    shadows_3d = {}

    for name, (opt_fn, color, lw, marker, _) in configs.items():
        line2, = ax1.plot([], [], color=color, linewidth=lw, label=name)
        pt2, = ax1.plot([], [], marker=marker, markersize=8, color=color, markeredgecolor='black', markeredgewidth=1.2)
        lines_2d[name] = line2
        points_2d[name] = pt2

        line3, = ax2.plot([], [], [], color=color, linewidth=lw, zorder=15)
        pt3, = ax2.plot([], [], [], marker=marker, markersize=8, color=color, markeredgecolor='black', markeredgewidth=1.1, zorder=22)
        stem3, = ax2.plot([], [], [], color=color, linestyle=":", linewidth=1.4, alpha=0.8, zorder=12)
        shadow3, = ax2.plot([], [], [], marker=marker, markersize=5, color=color, alpha=0.5, zorder=10)

        lines_3d[name] = line3
        points_3d[name] = pt3
        stems_3d[name] = stem3
        shadows_3d[name] = shadow3

    legend_2d = ax1.legend(loc="upper right", framealpha=0.92, fontsize=8.2)
    step_text = ax1.text(0.04, 0.93, "", transform=ax1.transAxes, fontsize=10, fontweight="bold",
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8fafc", edgecolor="#cbd5e1"))

    plt.tight_layout()

    # 51 frames (sampled every 2 steps) for smooth 14 fps loop
    frame_indices = list(range(0, steps + 1, 2))
    num_frames = len(frame_indices)
    print(f"[*] Rendering {num_frames} animation frames with vertical depth stems...")

    def update(frame_idx):
        t = frame_indices[frame_idx]

        for name in configs:
            traj = trajectories[name][:t+1]
            # 2D update
            lines_2d[name].set_data(traj[:, 0], traj[:, 1])
            points_2d[name].set_data([traj[-1, 0]], [traj[-1, 1]])

            # 3D update
            z_traj = 0.5 * (traj[:, 0]**2 + kappa * (traj[:, 1]**2))
            lines_3d[name].set_data(traj[:, 0], traj[:, 1])
            lines_3d[name].set_3d_properties(z_traj)

            curr_x, curr_y, curr_z = traj[-1, 0], traj[-1, 1], z_traj[-1]
            points_3d[name].set_data([curr_x], [curr_y])
            points_3d[name].set_3d_properties([curr_z])

            # Vertical drop line (stem from point to z=0)
            stems_3d[name].set_data([curr_x, curr_x], [curr_y, curr_y])
            stems_3d[name].set_3d_properties([0, curr_z])

            # Ground shadow point at z=0
            shadows_3d[name].set_data([curr_x], [curr_y])
            shadows_3d[name].set_3d_properties([0])

        step_text.set_text(f"Step: {t:3d} / {steps}")
        return (list(lines_2d.values()) + list(points_2d.values()) +
                list(lines_3d.values()) + list(points_3d.values()) +
                list(stems_3d.values()) + list(shadows_3d.values()) + [step_text])

    anim = animation.FuncAnimation(fig, update, frames=num_frames, interval=70, blit=False)

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "images"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cloves_optimization_dynamics_2d_3d.gif")

    print(f"[*] Saving intuitive 3D animation to GIF: {out_path} ...")
    anim.save(out_path, writer=animation.PillowWriter(fps=14))
    plt.close()

    file_size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"[+] Successfully saved updated animated GIF: {out_path} ({file_size_mb:.2f} MB)")

if __name__ == "__main__":
    generate_animation()
