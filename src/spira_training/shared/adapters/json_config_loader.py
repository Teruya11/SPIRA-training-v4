import json
from pathlib import Path

from src.spira_training.apps.feature_engineering.configs.feature_engineering_config import (
    FeatureEngineeringConfig,
)
from src.spira_training.apps.model_training.model_training_config import (
    ModelTrainingConfig,
)
from src.spira_training.shared.core.models.config import Config
from src.spira_training.shared.ports.config_loader import ConfigLoader


class JsonConfigLoader(ConfigLoader):
    def load(self, path: str | Path) -> Config:
        with Path(path).open() as file:
            config_json = json.load(file)

        return Config(
            feature_engineering_config=FeatureEngineeringConfig.model_validate(
                config_json["feature_engineering_config"]
            ),
            model_training_config=ModelTrainingConfig.model_validate(
                config_json["model_training_config"]
            ),
        )