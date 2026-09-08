from abc import ABC, abstractmethod

from src.spira_training.shared.core.models.audio_metadata import AudioMetadata


class AudioManifestReader(ABC):
    @abstractmethod
    def read(self, path: str) -> list[AudioMetadata]:
        pass