from abc import ABC, abstractmethod

from src.spira_training.shared.core.models.audio import Audio


class AudiosRepository(ABC):
    @abstractmethod
    def get_audios(self, path: str) -> list[Audio]:
        pass

    @abstractmethod
    def get_audio(self, path: str) -> Audio:
        pass
