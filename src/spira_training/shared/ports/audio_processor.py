from abc import ABC, abstractmethod

from src.spira_training.shared.core.models.audio import Audio
from src.spira_training.shared.core.models.audio_collection import AudioCollection


class AudioProcessor(ABC):
    @abstractmethod
    def process_audio(self, audio: Audio) -> Audio:
        pass

    @abstractmethod
    def process_audios(self, audios: AudioCollection) -> AudioCollection:
        pass
