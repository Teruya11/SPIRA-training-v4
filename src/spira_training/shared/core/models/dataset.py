from src.spira_training.shared.core.models.audio import Audio
from src.spira_training.shared.core.models.enum import BaseEnum


class Label(BaseEnum):
    POSITIVE = 1
    NEGATIVE = 0


class Dataset:
    def __init__(self, features: list[Audio], labels: list[Label]):
        self.features = features
        self.labels = labels
