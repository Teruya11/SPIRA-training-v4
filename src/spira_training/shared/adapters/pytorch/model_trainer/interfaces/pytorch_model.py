from abc import abstractmethod
from typing import List

from src.spira_training.shared.adapters.pytorch.models.pytorch_label import (
    PytorchLabel,
)
from src.spira_training.shared.adapters.pytorch.models.pytorch_parameter import (
    PytorchParameter,
)
from src.spira_training.shared.adapters.pytorch.models.pytorch_tensor import (
    PytorchTensor,
)
from src.spira_training.shared.core.models.base_model import BaseModel


class PytorchModel(BaseModel):
    @abstractmethod
    def predict(self, feature: PytorchTensor) -> PytorchLabel: ...

    @abstractmethod
    def predict_batch(
        self, features_batch: List[PytorchTensor]
    ) -> List[PytorchLabel]: ...

    @abstractmethod
    def get_parameters(self) -> list[PytorchParameter]: ...
