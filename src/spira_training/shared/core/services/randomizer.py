import numpy as np
from typing import Self

from src.spira_training.shared.core.interfaces.random import Random


class Randomizer(Random):
    def initialize_random(self, seed: int) -> Self:
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        return self