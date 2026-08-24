from pathlib import Path

import pandas as pd
import torch

from src.spira_training.shared.adapters.filesystem_path_validator import (
    FilesystemPathValidator,
)
from src.spira_training.shared.core.models.audio import Audio
from src.spira_training.shared.core.models.dataset import Dataset, Label
from src.spira_training.shared.core.models.wav import Wav
from src.spira_training.shared.ports.dataset_repository import DatasetRepository
from src.spira_training.shared.ports.path_validator import PathValidator


class ParquetDatasetRepository(DatasetRepository):
    """Persists datasets using Parquet format via Pandas."""

    def __init__(self, path_validator: PathValidator | None = None) -> None:
        self._path_validator = path_validator or FilesystemPathValidator()

    async def get_dataset(self, path: Path) -> Dataset:
        validated_path = self._path_validator.validate_path(path)

        try:
            dataframe = pd.read_parquet(validated_path)
            features = [
                Audio(wav=Wav(torch.tensor(wav_data)), sample_rate=16000)
                for wav_data in dataframe["features"]
            ]
            labels = [Label(int(label)) for label in dataframe["labels"]]
            return Dataset(features=features, labels=labels)
        except Exception as error:
            raise ValueError(
                f"Failed to load dataset from {validated_path}: {error}"
            ) from error

    async def save_dataset(self, dataset: Dataset, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            dataframe = pd.DataFrame(
                {
                    "features": [audio.wav.tensor.numpy() for audio in dataset.features],
                    "labels": [label.value for label in dataset.labels],
                }
            )
            dataframe.to_parquet(path, compression="snappy")
            self._path_validator.validate_path(path)
        except Exception as error:
            raise IOError(f"Failed to save dataset to {path}: {error}") from error
