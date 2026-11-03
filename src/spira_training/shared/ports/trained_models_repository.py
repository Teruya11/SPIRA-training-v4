from abc import ABC, abstractmethod
from pathlib import Path

from src.spira_training.shared.core.models.trainable_model import TrainableModel


class TrainedModelsRepository(ABC):
    @abstractmethod
    async def get_model(self, path: Path) -> TrainableModel:
        pass

    @abstractmethod
    async def save_model(self, model: TrainableModel, path: Path) -> None:
        pass
