from pydantic import BaseModel

from src.spira_training.apps.feature_engineering.configs.audio_config import AudioConfig
from src.spira_training.apps.feature_engineering.configs.audio_processor_config import AudioProcessorConfig
from src.spira_training.apps.feature_engineering.configs.audio_feature_transformer_config import AudioFeatureTransformerConfig
from src.spira_training.shared.core.models.path import Path

class FeatureEngineeringConfig(BaseModel):
    audio: AudioConfig
    audio_processor: AudioProcessorConfig
    audio_feature_transformer: AudioFeatureTransformerConfig
    dataset_base_path: Path = Path("out")
    dataset_file: str = "processed_dataset.pt"