from abc import ABC, abstractmethod
from typing import Any

class RandomGenerator(ABC):
    @abstractmethod
    def apply_random_seed(self) -> None:
        pass

    @abstractmethod
    def get_randint_in_interval(self, first: int, second: int) -> int:
        pass

    @abstractmethod
    def get_random_float_in_interval(self, first: float, second: float) -> float:
        pass

    @abstractmethod
    def choose_n_elements(self, elements: list[Any], num_elements: int) -> list[Any]:
        pass

    @abstractmethod
    def get_probability(self, alpha: float, beta: float) -> float:
        pass
