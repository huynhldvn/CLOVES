"""
Quickstart Example: Training a Transformer / LoRA Architecture with CLOVES-Apex
"""

import torch
import torch.nn as nn
from cloves import CLOVES_Adaptive, CLOVES_Apex_Asymmetric


class ToyLoRALayer(nn.Module):
    r"""Simulates a Low-Rank Adapter (LoRA) layer with matrices A and B."""
    def __init__(self, in_features=256, out_features=256, rank=16):
        super().__init__()
        # Normalized base weight initialization (variance = 1.0)
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features) / (in_features ** 0.5), requires_grad=False)
        # LoRA Matrix A (Gaussian init, shape: r x in)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.05)
        # LoRA Matrix B (Zero init, shape: out x r)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = 2.0

    def forward(self, x):
        base_out = x @ self.base_weight.T
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return base_out + lora_out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    model = ToyLoRALayer().to(device)

    # CLOVES-Apex automatically identifies LoRA Matrix A vs B and applies asymmetric acceleration
    optimizer = CLOVES_Apex_Asymmetric(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3,
        total_steps=10,
    )
    criterion = nn.MSELoss()

    # Fixed instruction adaptation batch
    x = torch.randn(8, 256, device=device)
    target = model(x).detach() + 0.3 * torch.randn(8, 256, device=device)

    print(f"Running CLOVES-Apex LoRA Quickstart on {device}...")
    for step in range(1, 11):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        if step % 2 == 0 or step == 1:
            print(f"Step {step:2d}/10 | LoRA Loss: {loss.item():.4f}")

    print("LoRA training step completed successfully!")


if __name__ == "__main__":
    main()
