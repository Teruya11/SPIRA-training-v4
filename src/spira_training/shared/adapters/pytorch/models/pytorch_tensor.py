from typing import NewType

import torch

__all__ = ["PytorchTensor"]

PytorchTensor = NewType("PytorchTensor", torch.Tensor)
