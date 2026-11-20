from abc import ABC, abstractmethod

from src.spira_training.shared.core.models.dataset import Dataset

from .pytorch_dataloader import PytorchDataloader


class PytorchDataloaderFactory(ABC):
    @abstractmethod
    def make_dataloader(self, dataset: Dataset) -> PytorchDataloader: ...
