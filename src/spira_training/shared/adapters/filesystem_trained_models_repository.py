from pathlib import Path

import torch

from src.spira_training.shared.adapters.filesystem_path_validator import (
    FilesystemPathValidator,
)
from src.spira_training.shared.core.models.base_model import BaseModel
from src.spira_training.shared.ports.path_validator import PathValidator
from src.spira_training.shared.ports.trained_models_repository import (
    TrainedModelsRepository,
)

class FilesystemTrainedModelsRepository(TrainedModelsRepository):
    def __init__(
        self,
        base_path: Path | None = None,
        path_validator: PathValidator = FilesystemPathValidator(),
    ) -> None:
        self._path_validator = path_validator
        self._base_path = base_path

    async def get_model(self, path: Path) -> BaseModel:
        model_path = self._resolve(path)
        validated_path = self._path_validator.validate_path(model_path)
        return torch.load(validated_path.path, map_location="cpu", weights_only=False)

    async def save_model(self, model: BaseModel, path: Path) -> None:
        model_path = self._resolve(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model, model_path)
        self._path_validator.validate_path(model_path)

    def _resolve(self, path: Path) -> Path:
        if self._base_path is None or path.is_absolute():
            return path
        return self._base_path / path
