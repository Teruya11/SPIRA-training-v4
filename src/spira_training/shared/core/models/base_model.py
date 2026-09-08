from abc import ABC, abstractmethod
from typing import Any


class BaseModel(ABC):
    @abstractmethod
    def train(self): ...

    @abstractmethod
    def predict(self, feature: Any): ...

    @abstractmethod
    def predict_batch(self, features_batch: list[Any]): ...

    @abstractmethod
    def get_parameters(self) -> list[Any]: ...

    @abstractmethod
    def dump_state(self) -> dict: ...

    @abstractmethod
    def load_state(self, state_dict: dict): ...
