from abc import ABC, abstractmethod

from src.spira_training.shared.core.models.base_model import BaseModel


class ModelPublisher(ABC):
    @abstractmethod
    def publish_model(self, model: BaseModel) -> None:
        pass
