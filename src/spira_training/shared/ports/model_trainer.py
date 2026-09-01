from abc import ABC, abstractmethod

from src.spira_training.shared.core.models.base_model import BaseModel
from src.spira_training.shared.core.models.dataset import Dataset


class ModelTrainer(ABC):
    @abstractmethod
    def train_model(
        self, train_dataset: Dataset, test_dataset: Dataset, epochs: int
    ) -> BaseModel:
        pass
