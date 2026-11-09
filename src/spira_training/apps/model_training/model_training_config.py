from pathlib import Path as PathLibPath

from pydantic import BaseModel

from src.spira_training.shared.adapters.filesystem_path_validator import (
    FilesystemPathValidator,
)
from src.spira_training.shared.core.models.path import Path
from src.spira_training.shared.ports.path_validator import PathValidator


class ModelTrainingConfig(BaseModel):
    dataset_path: Path
    trained_model_path: Path


def make_config(
    config_path: str | PathLibPath | None = None,
    path_validator: PathValidator = FilesystemPathValidator(),
) -> ModelTrainingConfig:
    if config_path is None:
        config_path = PathLibPath(__file__).resolve().parents[4] / "config.json"

    config_path = path_validator.validate_path(PathLibPath(config_path)).path

    from src.spira_training.shared.adapters.json_config_loader import JsonConfigLoader

    return JsonConfigLoader(path_validator).load(config_path).model_training_config
