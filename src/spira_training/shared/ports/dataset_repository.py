from abc import ABC, abstractmethod
from src.spira_training.shared.core.models.dataset import Dataset


class DatasetRepository(ABC):
    @abstractmethod
    async def get_dataset(self) -> Dataset:
        pass

    @abstractmethod
    async def save_dataset(self, dataset: Dataset) -> None:
        pass
