from abc import ABC, abstractmethod

from src.spira_training.shared.core.models.trainable_model import TrainableModel


class ModelPublisher(ABC):
    @abstractmethod
    def publish_model(self, model: TrainableModel) -> None:
        pass
