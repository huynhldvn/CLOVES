"""
Comprehensive unit tests for the CLOVES optimizer suite.
"""

import pytest
import torch
import torch.nn as nn
from cloves import (
    CLOVES,
    CLOVES_Bisector,
    CLOVES_Perpendicular,
    CLOVES_Affine,
    CLOVES_Adaptive,
    CLOVES_LLM,
    CLOVES_Apex_Asymmetric,
    CLOVES_Matrix_Polar,
    __version__,
)


def test_version():
    assert __version__ == "2.0.1"


def test_all_optimizers_step():
    optimizers_to_test = [
        (CLOVES, {"lr": 0.05}),
        (CLOVES_Bisector, {"lr": 0.05}),
        (CLOVES_Perpendicular, {"lr": 0.05}),
        (CLOVES_Affine, {"lr": 0.05}),
        (CLOVES_Adaptive, {"lr": 1e-3}),
        (CLOVES_LLM, {"lr": 1e-3}),
        (CLOVES_Apex_Asymmetric, {"lr": 1e-3, "total_steps": 100}),
        (CLOVES_Matrix_Polar, {"lr": 1e-2}),
    ]

    for opt_cls, kwargs in optimizers_to_test:
        model = nn.Sequential(
            nn.Linear(10, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )
        optimizer = opt_cls(model.parameters(), **kwargs)

        x = torch.randn(4, 10)
        y = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        criterion = nn.CrossEntropyLoss()

        initial_loss = None
        for step in range(5):
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        assert final_loss <= initial_loss, f"{opt_cls.__name__} failed to decrease loss: {initial_loss} -> {final_loss}"


def test_state_dict_save_load():
    model = nn.Linear(4, 2)
    optimizer = CLOVES_Adaptive(model.parameters(), lr=1e-3)
    x = torch.randn(2, 4)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()

    sd = optimizer.state_dict()
    new_opt = CLOVES_Adaptive(model.parameters(), lr=1e-3)
    new_opt.load_state_dict(sd)
    assert len(new_opt.state) == len(optimizer.state)


if __name__ == "__main__":
    test_version()
    test_all_optimizers_step()
    test_state_dict_save_load()
    print("ALL TESTS PASSED SUCCESSFULLY!")
