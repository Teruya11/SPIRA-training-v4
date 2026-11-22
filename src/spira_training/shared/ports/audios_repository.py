from abc import ABC, abstractmethod

from src.spira_training.shared.core.models.audio import Audio
from src.spira_training.shared.core.models.audio_metadata import AudioMetadata


class AudiosRepository(ABC):
    @abstractmethod
    def get_audios(self) -> list[tuple[Audio, AudioMetadata]]:
        pass
