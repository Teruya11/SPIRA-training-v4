from pathlib import Path
import pickle

from src.spira_training.shared.core.models.audio import Audio
from src.spira_training.shared.core.models.dataset import Dataset, Label
from src.spira_training.shared.core.models.wav import Wav
from src.spira_training.shared.adapters.filesystem_path_validator import (
    FilesystemPathValidator,
)
from src.spira_training.shared.ports.dataset_repository import DatasetRepository
from src.spira_training.shared.ports.path_validator import PathValidator


class PickleDatasetRepository(DatasetRepository):
    """Persists datasets using Python pickle."""

    def __init__(self, path_validator: PathValidator | None = None) -> None:
        self._path_validator = path_validator or FilesystemPathValidator()

    async def get_dataset(self, path: Path) -> Dataset:
        validated_path = self._path_validator.validate_path(path)

        try:
            with validated_path.path.open("rb") as file:
                payload = pickle.load(file)
            return Dataset(
                features=[
                    Audio(wav=Wav(tensor), sample_rate=sample_rate)
                    for tensor, sample_rate in zip(
                        payload["features"], payload["sample_rates"]
                    )
                ],
                labels=[Label(value) for value in payload["labels"]],
            )
        except Exception as error:
            raise ValueError(
                f"Failed to load dataset from {validated_path}: {error}"
            ) from error

    async def save_dataset(self, dataset: Dataset, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with path.open("wb") as file:
                pickle.dump(
                    {
                        "features": [audio.wav.tensor for audio in dataset.features],
                        "sample_rates": [
                            audio.sample_rate for audio in dataset.features
                        ],
                        "labels": [label.value for label in dataset.labels],
                    },
                    file,
                )
            self._path_validator.validate_path(path)
        except Exception as error:
            raise IOError(f"Failed to save dataset to {path}: {error}") from error