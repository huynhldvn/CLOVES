"""
Quickstart Example: Training a Convolutional Vision Model with CLOVES
"""

import torch
import torch.nn as nn
from cloves import CLOVES, CLOVES_Perpendicular


class SimpleConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        feat = self.features(x)
        return self.classifier(feat.flatten(1))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleConvNet().to(device)

    # CLOVES maintains a single momentum state (4 bytes/param), saving 50% state memory vs AdamW
    optimizer = CLOVES_Perpendicular(
        model.parameters(),
        lr=0.05,
        beta=0.9,
        weight_decay=1e-4,
        q=1.5,
        gamma=0.10,
    )
    criterion = nn.CrossEntropyLoss()

    print(f"Running CLOVES Vision Quickstart on {device}...")
    torch.manual_seed(42)
    x = torch.randn(16, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (16,), device=device)

    for step in range(1, 11):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        if step % 2 == 0 or step == 1:
            print(f"Step {step:2d}/10 | Loss: {loss.item():.4f}")

    print("Vision training step completed successfully!")


if __name__ == "__main__":
    main()
