from abc import ABC, abstractmethod
from pathlib import Path

from src.spira_training.shared.core.models.base_model import BaseModel


class TrainedModelsRepository(ABC):
    @abstractmethod
    async def get_model(self, path: Path) -> BaseModel:
        pass

    @abstractmethod
    async def save_model(self, model: BaseModel, path: Path) -> None:
        pass
