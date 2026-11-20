from typing import NewType

import torch

PytorchParameter = NewType("PytorchParameter", torch.nn.Parameter)
