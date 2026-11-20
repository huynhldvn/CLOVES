"""
Empirical Benchmark for CLOVES_Adaptive:
Compares Original vs. Fused Zero-Allocation implementation.
"""

import time
import math
import torch
import torch.nn as nn

class AdaptiveOriginal(torch.optim.Optimizer):
    def __init__(self, params, lr=2e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, q=1.25, gamma=0.60):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, q=q, gamma=gamma)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, (beta1, beta2) = group["lr"], group["betas"]
            eps, wd, q, gamma = group["eps"], group["weight_decay"], group["q"], group["gamma"]
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
                bc1 = 1.0 - beta1 ** t
                bc2 = 1.0 - beta2 ** t
                if wd != 0:
                    p.mul_(1.0 - lr * wd)
                c = torch.where(g * m >= 0, q, gamma)
                m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                m_hat = (m / bc1) * c
                v_hat = v / bc2
                denom = torch.sqrt(v_hat).add_(eps)
                p.addcdiv_(m_hat, denom, value=-lr)
        return loss

class AdaptiveOptimized(torch.optim.Optimizer):
    def __init__(self, params, lr=2e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, q=1.25, gamma=0.60):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, q=q, gamma=gamma)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, (beta1, beta2) = group["lr"], group["betas"]
            eps, wd, q, gamma = group["eps"], group["weight_decay"], group["q"], group["gamma"]
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
                bc1 = 1.0 - beta1 ** t
                bc2 = 1.0 - beta2 ** t
                step_size = lr * (math.sqrt(bc2) / bc1)
                
                if wd != 0:
                    p.mul_(1.0 - lr * wd)
                
                m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                
                c = torch.where(g * m >= 0, q, gamma)
                denom = torch.sqrt(v).add_(eps * math.sqrt(bc2))
                p.addcdiv_(m * c, denom, value=-step_size)
        return loss

def main():
    print("=" * 80)
    print("EMPIRICAL BENCHMARK: CLOVES_Adaptive (Original vs. Fused Optimized)")
    print("=" * 80)
    
    torch.manual_seed(42)
    dim = 512
    x_data = torch.randn(64, dim)
    y_target = torch.randn(64, dim)
    
    def test_opt(opt_cls, steps=100):
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        opt = opt_cls(model.parameters(), lr=1e-3)
        crit = nn.MSELoss()
        losses = []
        
        t0 = time.perf_counter()
        for _ in range(steps):
            opt.zero_grad()
            out = model(x_data)
            loss = crit(out, y_target)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        t1 = time.perf_counter()
        return losses[-1], (t1 - t0)

    loss_orig, time_orig = test_opt(AdaptiveOriginal)
    loss_opt, time_opt = test_opt(AdaptiveOptimized)
    
    print(f"Original:  Final Loss = {loss_orig:.6f} | Runtime = {time_orig:.3f} s")
    print(f"Optimized: Final Loss = {loss_opt:.6f} | Runtime = {time_opt:.3f} s")
    speedup = (time_orig / time_opt - 1.0) * 100
    loss_diff = ((loss_opt - loss_orig) / loss_orig) * 100
    print(f"Speedup: {speedup:+.1f}% | Loss Change: {loss_diff:+.4f}%")
    
if __name__ == "__main__":
    main()
