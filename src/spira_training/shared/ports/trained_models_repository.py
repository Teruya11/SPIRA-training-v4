from abc import ABC, abstractmethod
from src.spira_training.shared.core.models.trainable_model import TrainableModel


class TrainedModelsRepository(ABC):
    @abstractmethod
    async def get_model(self) -> TrainableModel:
        pass

    @abstractmethod
    async def save_model(self, model: TrainableModel) -> None:
        pass
