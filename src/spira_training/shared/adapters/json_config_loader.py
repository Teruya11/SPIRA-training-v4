import json
from pathlib import Path

from src.spira_training.apps.feature_engineering.configs.feature_engineering_config import FeatureEngineeringConfig
from src.spira_training.shared.core.models.config import Config
from src.spira_training.shared.ports.config_loader import ConfigLoader


class JsonConfigLoader(ConfigLoader):
    def load(self, path: str | Path) -> Config:
        with open(path, 'r') as file:
            config_json = json.load(file)
        return Config(**config_json)

    def load_feature_engineering_config(self, path: str | Path) -> FeatureEngineeringConfig:
        with open(path, 'r') as file:
            config_json = json.load(file)

        if 'feature_engineering_config' in config_json:
            return FeatureEngineeringConfig.model_validate(config_json['feature_engineering_config'])

        return FeatureEngineeringConfig.model_validate(config_json)