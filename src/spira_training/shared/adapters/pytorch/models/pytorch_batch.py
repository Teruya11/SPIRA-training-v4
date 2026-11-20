from src.spira_training.shared.adapters.pytorch.models.pytorch_label import (
    PytorchLabel,
)
from src.spira_training.shared.adapters.pytorch.models.pytorch_tensor import (
    PytorchTensor,
)


class PytorchBatch:
    def __init__(
        self, features: list[PytorchTensor], labels: list[PytorchLabel]
    ) -> None:
        self.features = features
        self.labels = labels
