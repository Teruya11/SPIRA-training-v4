from src.spira_training.apps.feature_engineering.configs.feature_engineering_config import (
    FeatureEngineeringConfig,
)
from src.spira_training.apps.model_training.model_training_config import (
    ModelTrainingConfig,
)


class Config:
    def __init__(
        self,
        feature_engineering_config: FeatureEngineeringConfig,
        model_training_config: ModelTrainingConfig,
    ) -> None:
        self.feature_engineering_config = feature_engineering_config
        self.model_training_config = model_training_config
