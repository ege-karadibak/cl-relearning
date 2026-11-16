"""Sanity check for the cl-relearning environment."""

import platform
from pathlib import Path

import torch
import torchvision
from torchvision import datasets

DATA = Path(__file__).resolve().parent / "data"

print(f"Python      {platform.python_version()}")
print(f"torch       {torch.__version__}")
print(f"torchvision {torchvision.__version__}")

assert torch.backends.mps.is_available(), "MPS is NOT available"
print("MPS available: True")

a = torch.randn(256, 256, device="mps")
b = torch.randn(256, 256, device="mps")
c = a @ b
torch.mps.synchronize()
print(f"MPS matmul OK: {tuple(c.shape)} on {c.device}")

loaders = {
    "CIFAR-10": lambda train: datasets.CIFAR10(DATA, train=train, download=False),
    "CIFAR-100": lambda train: datasets.CIFAR100(DATA, train=train, download=False),
    "SVHN": lambda train: datasets.SVHN(DATA / "svhn", split="train" if train else "test", download=False),
}
for name, load in loaders.items():
    print(f"{name:<10} train={len(load(True))} test={len(load(False))}")
