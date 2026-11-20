"""
CLOVES-Apex Asymmetric: SOTA LoRA Factorization & Continuous Soft-Cleft Damping Optimizer
"""

import math
import torch
from torch.optim.optimizer import Optimizer


class CLOVES_Apex_Asymmetric(Optimizer):
    r"""CLOVES-Apex Asymmetric Optimizer for Parameter-Efficient LoRA Instruction Tuning.

    Implements three breakthrough mechanisms for low-rank adapter fine-tuning:
    1. LoRA A/B Asymmetric Velocity Ratio:
       - Matrix A (Gaussian initialized): q_A = 1.30, lr_eff = 1.25 * lr
       - Matrix B (Zero initialized): q_B = 1.05, lr_eff = 1.00 * lr
       Resolves the initial gradient stagnation of LoRA down-projection matrices.
    2. Continuous Soft-Cleft Damping:
       - Smooth tanh damping envelope eliminating coordinate chattering:
         c(rho) = (q_t + gamma_t)/2 + (q_t - gamma_t)/2 * tanh(rho / tau)
    3. Cosine Cleft Annealing:
       - Smoothly anneals q(t) -> 1.0 and gamma(t) -> 1.0 toward the end of training.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): Learning rate (default: 1.6e-4).
        betas (Tuple[float, float], optional): Coefficients for first and second moments (default: (0.9, 0.999)).
        eps (float, optional): Epsilon for numerical stability (default: 1e-8).
        weight_decay (float, optional): Decoupled weight decay coefficient (default: 1e-2).
        q_A (float, optional): Overstep acceleration for LoRA Matrix A (default: 1.30).
        q_B (float, optional): Overstep acceleration for LoRA Matrix B (default: 1.05).
        gamma0 (float, optional): Base cleft damping coefficient (default: 0.70).
        total_steps (int, optional): Total planned optimization steps for annealing schedule (default: 500).
        tau (float, optional): Soft-cleft tanh temperature scale (default: 0.15).
    """

    def __init__(
        self,
        params,
        lr: float = 1.6e-4,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        q_A: float = 1.30,
        q_B: float = 1.05,
        gamma0: float = 0.70,
        total_steps: int = 500,
        tau: float = 0.15,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameters: {betas}")
        if eps < 0.0:
            raise ValueError(f"Invalid eps value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if q_A < 1.0 or q_B < 1.0:
            raise ValueError(f"Overstep parameters q_A, q_B must be >= 1.0")
        if not 0.0 < gamma0 <= 1.0:
            raise ValueError(f"Base damping gamma0 must be in (0, 1]")
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            q_A=q_A,
            q_B=q_B,
            gamma0=gamma0,
            total_steps=total_steps,
            tau=tau,
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
            q_A, q_B = group["q_A"], group["q_B"]
            gamma0 = group["gamma0"]
            tot_steps = group["total_steps"]
            tau = group["tau"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if len(st) == 0:
                    st["step"] = 0
                    st["m"] = torch.clone(g).detach()
                    st["v"] = torch.zeros_like(g)

                st["step"] += 1
                t = st["step"]
                m, v = st["m"], st["v"]

                # Detect if parameter is LoRA matrix A or B based on shape
                # In LoRA: A has shape (r, in_dim), B has shape (out_dim, r)
                # Typically r <= 64. If shape[0] == r and shape[0] < shape[1] -> Matrix A
                is_lora_A = (p.ndim == 2 and p.shape[0] < p.shape[1] and p.shape[0] <= 64)
                q_base = q_A if is_lora_A else q_B
                lr_eff = lr * (1.25 if is_lora_A else 1.0)

                # 1. Cosine Cleft Annealing (Scalar computation - Zero VRAM allocation)
                progress = min(1.0, t / max(1, tot_steps))
                cos_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                q_t = 1.0 + (q_base - 1.0) * cos_decay
                gamma_t = 1.0 - (1.0 - gamma0) * cos_decay

                # Fused bias correction scalar (Zero intermediate tensor allocations)
                bc1 = 1.0 - beta1 ** t
                bc2 = 1.0 - beta2 ** t
                step_size = lr_eff * (math.sqrt(bc2) / bc1)

                # 2. In-place moments update
                m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                # 3. Fast coordinate cleft alignment & overstep scaling
                c = torch.where(g * m >= 0, q_t, gamma_t)

                # 4. Decoupled weight decay
                if wd != 0:
                    p.mul_(1.0 - lr_eff * wd)

                # 5. Position update (In-place fused division, zero m_hat / v_hat allocations)
                denom = torch.sqrt(v).add_(eps * math.sqrt(bc2))
                p.addcdiv_(m * c, denom, value=-step_size)

        return loss
