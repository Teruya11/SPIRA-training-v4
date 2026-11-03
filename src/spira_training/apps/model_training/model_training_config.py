from pathlib import Path as PathLibPath

from pydantic import BaseModel

from src.spira_training.shared.core.models.path import Path


class ModelTrainingConfig(BaseModel):
    dataset_base_path: Path
    dataset_file: str
    trained_model_base_path: Path
    trained_model_file: str


def make_config(config_path: str | PathLibPath | None = None) -> ModelTrainingConfig:
    if config_path is None:
        config_path = PathLibPath(__file__).resolve().parents[4] / "config.json"

    config_path = PathLibPath(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")

    from src.spira_training.shared.adapters.json_config_loader import JsonConfigLoader

    return JsonConfigLoader().load(config_path).model_training_config
