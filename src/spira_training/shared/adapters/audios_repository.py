from pathlib import Path
from typing import List

import torchaudio

from src.spira_training.shared.core.models.audio import Audio
from src.spira_training.shared.core.models.wav import Wav
from src.spira_training.shared.ports.audios_repository import AudiosRepository


class AudiosRepository(AudiosRepository):
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def get_audio(self, path: str) -> Audio:
        waveform, sr = torchaudio.load(path)

        if waveform.dim() == 2:
            waveform = waveform.mean(dim=0)

        return Audio(wav=Wav(waveform), sample_rate=sr)

    def get_audios(self, path: str) -> List[Audio]:
        folder = Path(path)

        if folder.is_file():
            return [self.get_audio(str(folder))]

        return [
            self.get_audio(str(file))
            for file in sorted(folder.iterdir())
            if file.is_file() and file.suffix.lower() in {".wav", ".mp3", ".flac"}
        ]
