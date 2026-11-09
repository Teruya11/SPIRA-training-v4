from pathlib import Path

import torch

from src.spira_training.shared.adapters.filesystem_path_validator import (
    FilesystemPathValidator,
)
from src.spira_training.shared.core.models.trainable_model import TrainableModel
from src.spira_training.shared.ports.path_validator import PathValidator
from src.spira_training.shared.ports.trained_models_repository import (
    TrainedModelsRepository,
)

class FilesystemTrainedModelsRepository(TrainedModelsRepository):
    def __init__(
        self,
        model_path: Path,
        path_validator: PathValidator = FilesystemPathValidator(),
    ) -> None:
        self._path_validator = path_validator
        self._model_path = model_path

    async def get_model(self) -> TrainableModel:
        validated_path = self._path_validator.validate_path(self._model_path)
        return torch.load(validated_path.path, map_location="cpu", weights_only=False)

    async def save_model(self, model: TrainableModel) -> None:
        model_path = self._model_path
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model, model_path)
        self._path_validator.validate_path(model_path)
