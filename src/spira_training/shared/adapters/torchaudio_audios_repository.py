from pathlib import Path
from typing import List

import torchaudio

from src.spira_training.shared.core.models.audio import Audio
from src.spira_training.shared.core.models.wav import Wav
from src.spira_training.shared.ports.audios_repository import AudiosRepository


class TorchAudioRepository(AudiosRepository):
    """Loads audio files using torchaudio."""

    def get_audios(self, path: str) -> List[Audio]:
        """Load multiple audio files from a directory or list."""
        path_obj = Path(path)
        if path_obj.is_dir():
            audio_files = sorted(path_obj.glob("*.wav"))
            return [self.get_audio(str(f)) for f in audio_files]
        raise ValueError(f"Path is not a directory: {path}")

    def get_audio(self, path: str) -> Audio:
        """Load a single audio file."""
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        
        if not path_obj.suffix.lower() == ".wav":
            raise ValueError(f"File must be .wav format: {path}")
        
        try:
            waveform, sample_rate = torchaudio.load(str(path_obj))
            wav = Wav(waveform.squeeze(0))
            return Audio(wav=wav, sample_rate=sample_rate)
        except Exception as e:
            raise RuntimeError(f"Failed to load audio from {path}: {e}")
