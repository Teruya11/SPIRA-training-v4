from dataclasses import dataclass
from pathlib import Path

from src.spira_training.shared.core.models.enum import BaseEnum


class Label(BaseEnum):
    POSITIVE = 1
    NEGATIVE = 0


@dataclass(frozen=True)
class AudioMetadata:
    audio_path: Path
    label: Label
    # Include other attributes from Manifest as you see fit
