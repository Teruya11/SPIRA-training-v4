from abc import ABC, abstractmethod

from src.spira_training.shared.core.models.dataset import Dataset
from src.spira_training.shared.core.models.trainable_model import TrainableModel


class ModelTrainer(ABC):
    @abstractmethod
    def train_model(
        self, train_dataset: Dataset, test_dataset: Dataset, epochs: int
    ) -> TrainableModel:
        pass
