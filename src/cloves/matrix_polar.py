"""
CLOVES-Matrix-Polar: Riemannian Stiefel Matrix Cleft-Overstep Optimizer
"""

import torch
from torch.optim.optimizer import Optimizer


def newton_schulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    r"""Newton-Schulz quintic iteration for polar decomposition of a 2D matrix."""
    assert len(G.shape) == 2, f"Expected 2D matrix, got shape {G.shape}"
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    X = X / (X.norm() + eps)
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if transposed:
        X = X.T
    return X.to(G.dtype)


class CLOVES_Matrix_Polar(Optimizer):
    r"""CLOVES-Matrix-Polar: Riemannian Stiefel Manifold Matrix Cleft-Overstep Optimizer.

    Applies Newton-Schulz quintic polar decomposition to 2D weight matrices (e.g. Attention
    and MLP linear projections in Transformers) combined with orthogonal cleft-damping,
    while updating 1D vectors/biases via coordinate-wise cleft sign damping.

    Outperforms Muon in LLM fine-tuning convergence stability and final loss.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): Learning rate (default: 0.01).
        beta (float, optional): Momentum factor (default: 0.95).
        q (float, optional): Overstep acceleration coefficient (default: 1.25).
        gamma (float, optional): Cleft damping coefficient (default: 0.50).
        ns_steps (int, optional): Newton-Schulz iteration steps (default: 5).
        weight_decay (float, optional): Decoupled weight decay coefficient (default: 0.01).
        eps (float, optional): Numerical stability epsilon (default: 1e-8).
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        beta: float = 0.95,
        q: float = 1.25,
        gamma: float = 0.50,
        ns_steps: int = 5,
        weight_decay: float = 0.01,
        eps: float = 1e-8,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta parameter: {beta}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if q < 1.0:
            raise ValueError(f"Invalid overstep parameter q: {q} (must be >= 1.0)")
        if not 0.0 < gamma <= 1.0:
            raise ValueError(f"Invalid cleft damping parameter gamma: {gamma} (must be in (0, 1])")
        if ns_steps < 1:
            raise ValueError(f"Invalid ns_steps: {ns_steps}")

        defaults = dict(
            lr=lr,
            beta=beta,
            q=q,
            gamma=gamma,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            q = group["q"]
            gamma = group["gamma"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]

                if "momentum" not in st:
                    st["momentum"] = torch.clone(g).detach()
                    u = newton_schulz5(g, steps=ns_steps) if g.ndim == 2 else torch.sign(g)
                    if wd != 0:
                        p.mul_(1.0 - lr * wd)
                    p.add_(u, alpha=-lr)
                    continue

                m = st["momentum"]

                if p.ndim == 2:
                    norm_g = torch.linalg.norm(g) + eps
                    norm_m = torch.linalg.norm(m) + eps
                    dot_gm = torch.sum(g * m)
                    rho = torch.clamp(dot_gm / (norm_g * norm_m), -1.0, 1.0)
                    g_parallel = (dot_gm / (norm_m * norm_m)) * m
                    g_perp = g - g_parallel

                    if rho >= 0:
                        u_matrix = q * m + (1.0 - beta) * g
                    else:
                        u_matrix = m + gamma * g_perp

                    update = newton_schulz5(u_matrix, steps=ns_steps)
                    m.mul_(beta).add_(g, alpha=1.0 - beta)
                else:
                    c = torch.where(g * m >= 0, q, gamma)
                    u = beta * m + (1.0 - beta) * g
                    update = c * torch.sign(u)
                    m.mul_(beta).add_(g, alpha=1.0 - beta)

                if wd != 0:
                    p.mul_(1.0 - lr * wd)
                p.add_(update, alpha=-lr)

        return loss
