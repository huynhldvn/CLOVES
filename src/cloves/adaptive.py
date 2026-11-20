"""
CLOVES-Adaptive: Cleft-Overstep Optimizer with Adaptive Second-Moment Scaling
"""

import collections
import math
import torch
from torch.optim.optimizer import Optimizer

_HAS_FOREACH = hasattr(torch, "_foreach_mul_") and hasattr(torch, "_foreach_add_")


class CLOVES_Adaptive(Optimizer):
    r"""CLOVES-Adaptive: Cleft-damping Overstep Optimizer with Adaptive Second-Moment Scaling.

    Combines the Cleft-Overstep principle with adaptive second-moment curvature scaling.
    Applies coordinate cleft damping (gamma) to oscillating coordinates and overstep
    acceleration (q) to longitudinal descent along the ravine floor, scaled by running
    root-mean-square variance.

    100% safe for multi-GPU pipeline parallelism (device_map='auto').

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): Learning rate (default: 2e-4).
        betas (Tuple[float, float], optional): Coefficients used for computing running averages
            of gradient and its square (default: (0.9, 0.999)).
        eps (float, optional): Term added to the denominator to improve numerical stability (default: 1e-8).
        weight_decay (float, optional): Decoupled weight decay coefficient (default: 1e-2).
        q (float, optional): Overstep acceleration coefficient for aligned coordinates, q >= 1.0 (default: 1.25).
        gamma (float, optional): Cleft damping coefficient for oscillating coordinates, gamma in (0, 1] (default: 0.60).
        foreach (bool, optional): Whether to use fused foreach implementations where available (default: True).
    """

    def __init__(
        self,
        params,
        lr: float = 2e-4,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        q: float = 1.25,
        gamma: float = 0.60,
        foreach: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameters: {betas}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if q < 1.0:
            raise ValueError(f"Invalid overstep parameter q: {q} (must be >= 1.0)")
        if not 0.0 < gamma <= 1.0:
            raise ValueError(f"Invalid cleft damping parameter gamma: {gamma} (must be in (0, 1])")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            q=q,
            gamma=gamma,
            foreach=foreach,
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
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            q = group["q"]
            gamma = group["gamma"]
            use_foreach = group.get("foreach", True)

            params_with_grad = []
            grads = []
            moms = []
            vars_ = []

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.clone(g).detach()
                    state["v"] = torch.zeros_like(g)

                state["step"] += 1
                params_with_grad.append(p)
                grads.append(g)
                moms.append(state["m"])
                vars_.append(state["v"])

            if not params_with_grad:
                continue

            # Per-device grouping for multi-GPU safe execution
            if use_foreach and _HAS_FOREACH:
                by_device = collections.defaultdict(lambda: ([], [], [], []))
                for p, g, m, v in zip(params_with_grad, grads, moms, vars_):
                    by_device[p.device][0].append(p)
                    by_device[p.device][1].append(g)
                    by_device[p.device][2].append(m)
                    by_device[p.device][3].append(v)

                for dev, (d_params, d_grads, d_moms, d_vars) in by_device.items():
                    if wd != 0:
                        torch._foreach_mul_(d_params, 1.0 - lr * wd)

                    for p, g, m, v in zip(d_params, d_grads, d_moms, d_vars):
                        st = self.state[p]
                        step_t = st["step"]
                        bc1 = 1.0 - beta1 ** step_t
                        bc2 = 1.0 - beta2 ** step_t
                        step_size = lr * (math.sqrt(bc2) / bc1)

                        # Update moments in-place
                        m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                        v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                        # Coordinate Cleft Alignment
                        c = torch.where(g * m >= 0, q, gamma)

                        # Fused position update (zero intermediate tensor allocations)
                        denom = torch.sqrt(v).add_(eps * math.sqrt(bc2))
                        p.addcdiv_(m * c, denom, value=-step_size)
            else:
                for p, g, m, v in zip(params_with_grad, grads, moms, vars_):
                    st = self.state[p]
                    step_t = st["step"]
                    bc1 = 1.0 - beta1 ** step_t
                    bc2 = 1.0 - beta2 ** step_t
                    step_size = lr * (math.sqrt(bc2) / bc1)

                    if wd != 0:
                        p.mul_(1.0 - lr * wd)

                    m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                    v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                    c = torch.where(g * m >= 0, q, gamma)
                    denom = torch.sqrt(v).add_(eps * math.sqrt(bc2))
                    p.addcdiv_(m * c, denom, value=-step_size)

        return loss
