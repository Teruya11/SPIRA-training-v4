from pathlib import Path
from typing import List

import torchaudio

from src.spira_training.shared.core.models.audio import Audio
from src.spira_training.shared.core.models.wav import Wav
from src.spira_training.shared.adapters.filesystem_path_validator import (
    FilesystemPathValidator,
)
from src.spira_training.shared.ports.audios_repository import AudiosRepository
from src.spira_training.shared.ports.path_validator import PathValidator


class AudiosRepository(AudiosRepository):
    def __init__(
        self,
        sample_rate: int = 16000,
        path_validator: PathValidator = FilesystemPathValidator(),
    ):
        self.sample_rate = sample_rate
        self.path_validator = path_validator

    def get_audio(self, path: str) -> Audio:
        validated_path = self.path_validator.validate_path(path)
        waveform, sr = torchaudio.load(validated_path.path)

        if waveform.dim() == 2:
            waveform = waveform.mean(dim=0)

        return Audio(wav=Wav(waveform), sample_rate=sr)

    def get_audios(self, path: str) -> List[Audio]:
        folder = self.path_validator.validate_path(path).path

        if folder.is_file():
            return [self.get_audio(str(folder))]

        return [
            self.get_audio(str(file))
            for file in sorted(folder.iterdir())
            if file.is_file() and file.suffix.lower() in {".wav", ".mp3", ".flac"}
        ]
