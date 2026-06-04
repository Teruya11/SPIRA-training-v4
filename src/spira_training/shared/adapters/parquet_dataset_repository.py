from pathlib import Path
import torch
import pandas as pd

from src.spira_training.shared.core.models.dataset import Dataset, Label
from src.spira_training.shared.core.models.audio import Audio
from src.spira_training.shared.core.models.wav import Wav
from src.spira_training.shared.ports.dataset_repository import DatasetRepository


class ParquetDatasetRepository(DatasetRepository):
    """Persists datasets using Parquet format via Pandas."""

    async def get_dataset(self, path: Path) -> Dataset:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        try:
            df = pd.read_parquet(path)
            features = [
                Audio(wav=Wav(torch.tensor(wav_data)), sample_rate=16000)
                for wav_data in df["features"]
            ]
            labels = [Label(int(label)) for label in df["labels"]]
            return Dataset(features=features, labels=labels)
        except Exception as e:
            raise ValueError(f"Failed to load dataset from {path}: {e}")

    async def save_dataset(self, dataset: Dataset, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            df = pd.DataFrame(
                {
                    "features": [
                        audio.wav.tensor.numpy() for audio in dataset.features
                    ],
                    "labels": [label.value for label in dataset.labels],
                }
            )
            df.to_parquet(path, compression="snappy")
        except Exception as e:
            raise IOError(f"Failed to save dataset to {path}: {e}")
