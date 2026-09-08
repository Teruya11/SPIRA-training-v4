from typing import Any, List
from src.spira_training.shared.core.models.enum import BaseEnum


class Label(BaseEnum):
    POSITIVE = 1
    NEGATIVE = 0


class Dataset:
    def __init__(self, features: List[Any], labels: List[Label]):
        self.features = features
        self.labels = labels
