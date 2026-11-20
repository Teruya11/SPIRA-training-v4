from typing import NewType

import torch

__all__ = ["PytorchLabel"]

PytorchLabel = NewType("PytorchLabel", torch.Tensor)
