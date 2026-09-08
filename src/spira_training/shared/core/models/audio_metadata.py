from dataclasses import dataclass
from pathlib import Path

from src.spira_training.shared.core.models.dataset import Label


@dataclass(frozen=True)
class AudioMetadata:
    audio_path: Path
    label: Label
