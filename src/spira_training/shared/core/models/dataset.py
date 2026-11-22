from src.spira_training.shared.core.models.audio import Audio
from src.spira_training.shared.core.models.audio_metadata import Label


class Dataset:
    def __init__(self, features: list[Audio], labels: list[Label]):
        self.features = features
        self.labels = labels
