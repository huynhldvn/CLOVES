"""
CLOVES: Cleft-Overstep Optimization Algorithm (Bisector Variant)
"""

import math
import torch
from torch.optim.optimizer import Optimizer

INV_SQRT2 = 0.7071067811865475  # 1.0 / math.sqrt(2.0)


class CLOVES(Optimizer):
    r"""Cleft-Overstep Optimization Algorithm (CLOVES-Bisector).

    Computes the normalized angular bisector between the stochastic gradient
    and historical momentum vector, damping high-frequency transversal oscillations
    while accelerating longitudinal descent along ill-conditioned ravines.

    Requires only a single momentum buffer (4 bytes/param state memory),
    achieving strict memory parity with SGD-M and saving 50% state memory
    compared to AdamW.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): Learning rate (default: 0.1).
        beta (float, optional): Momentum factor (default: 0.9).
        weight_decay (float, optional): Decoupled weight decay coefficient (default: 1e-4).
        q (float, optional): Overstep acceleration coefficient along ravine floors, q > 1.0 (default: 1.5).
        gamma (float, optional): Cleft damping coefficient for transversal wall bouncing, gamma in (0, 1] (default: 0.10).
        eps (float, optional): Numerical stability epsilon (default: 1e-8).
    """

    def __init__(
        self,
        params,
        lr: float = 0.1,
        beta: float = 0.9,
        weight_decay: float = 1e-4,
        q: float = 1.5,
        gamma: float = 0.10,
        eps: float = 1e-8,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta parameter: {beta}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if q <= 1.0:
            raise ValueError(f"Invalid overstep parameter q: {q} (must be > 1.0)")
        if not 0.0 < gamma <= 1.0:
            raise ValueError(f"Invalid cleft damping parameter gamma: {gamma} (must be in (0, 1])")
        if eps < 0.0:
            raise ValueError(f"Invalid eps value: {eps}")

        defaults = dict(
            lr=lr,
            beta=beta,
            weight_decay=weight_decay,
            q=q,
            gamma=gamma,
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
            wd = group["weight_decay"]
            q = group["q"]
            gamma = group["gamma"]
            eps = group["eps"]

            one_minus_beta = 1.0 - beta
            wd_factor = 1.0 - lr * wd if wd != 0 else 1.0
            base_c = INV_SQRT2
            q_c = (q - 1.0) * INV_SQRT2
            gamma_c = (1.0 - gamma) * INV_SQRT2

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]

                # Lazy state initialization
                if "momentum" not in state:
                    state["momentum"] = torch.clone(g).detach()
                    p.add_(g, alpha=-lr)
                    if wd != 0:
                        p.mul_(wd_factor)
                    continue

                m = state["momentum"]

                # Energy norms & inner product
                norm_g = torch.linalg.norm(g) + eps
                norm_m = torch.linalg.norm(m) + eps
                dot_gm = torch.dot(g.reshape(-1), m.reshape(-1))
                rho = torch.clamp(dot_gm / (norm_g * norm_m), -1.0, 1.0)

                # Analytic bisector amplitude via shared radicand sqrt(1 + rho)
                sqrt_1p = torch.sqrt(torch.clamp(1.0 + rho, min=eps))
                c_rho = q_c if rho >= 0 else gamma_c
                c_g = (base_c + c_rho * rho) / sqrt_1p
                c_m = c_g * (norm_g / norm_m)

                # In-place algebraic update
                p.add_(g, alpha=-lr * c_g)
                p.add_(m, alpha=-lr * c_m)

                # Decoupled weight decay
                if wd != 0:
                    p.mul_(wd_factor)

                # Momentum EMA tracking
                m.lerp_(g, one_minus_beta)

        return loss


# Alias for explicit clarity
CLOVES_Bisector = CLOVES
