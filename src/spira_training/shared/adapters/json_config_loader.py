import json
from pathlib import Path

from src.spira_training.apps.feature_engineering.configs.feature_engineering_config import (
    FeatureEngineeringConfig,
)
from src.spira_training.apps.model_training.model_training_config import (
    ModelTrainingConfig,
)
from src.spira_training.shared.adapters.filesystem_path_validator import (
    FilesystemPathValidator,
)
from src.spira_training.shared.core.models.config import Config
from src.spira_training.shared.ports.config_loader import ConfigLoader
from src.spira_training.shared.ports.path_validator import PathValidator


class JsonConfigLoader(ConfigLoader):
    def __init__(
        self,
        path_validator: PathValidator = FilesystemPathValidator(),
    ) -> None:
        self._path_validator = path_validator

    def load(self, path: str | Path) -> Config:
        validated_path = self._path_validator.validate_path(path)
        with validated_path.path.open() as file:
            config_json = json.load(file)

        return Config(
            feature_engineering_config=FeatureEngineeringConfig.model_validate(
                config_json["feature_engineering_config"]
            ),
            model_training_config=ModelTrainingConfig.model_validate(
                config_json["model_training_config"]
            ),
        )