from pathlib import Path

import torch

from src.spira_training.shared.core.models.audio import Audio
from src.spira_training.shared.core.models.dataset import Dataset, Label
from src.spira_training.shared.core.models.wav import Wav
from src.spira_training.shared.adapters.filesystem_path_validator import (
    FilesystemPathValidator,
)
from src.spira_training.shared.ports.dataset_repository import DatasetRepository
from src.spira_training.shared.ports.path_validator import PathValidator


class PytorchDatasetRepository(DatasetRepository):
    def __init__(
        self,
        dataset_path: Path,
        path_validator: PathValidator = FilesystemPathValidator(),
    ) -> None:
        self._dataset_path = dataset_path
        self._path_validator = path_validator

    async def get_dataset(self) -> Dataset:
        validated_path = self._path_validator.validate_path(self._dataset_path)

        try:
            with validated_path.path.open("rb") as file:
                payload = torch.load(file, weights_only=True)
            return Dataset(
                features=[
                    Audio(
                        wav=Wav(feature["tensor"]),
                        sample_rate=feature["sample_rate"],
                    )
                    for feature in payload["features"]
                ],
                labels=[Label(label) for label in payload["labels"]],
            )
        except Exception as error:
            raise ValueError(
                f"Failed to load dataset from {validated_path}: {error}"
            ) from error

    async def save_dataset(self, dataset: Dataset) -> None:
        resolved_path = self._dataset_path
        resolved_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            payload = {
                "features": [
                    {
                        "tensor": feature.wav.tensor,
                        "sample_rate": feature.sample_rate,
                    }
                    for feature in dataset.features
                ],
                "labels": [label.value for label in dataset.labels],
            }
            with resolved_path.open("wb") as file:
                torch.save(payload, file)

            self._path_validator.validate_path(resolved_path)
        except Exception as error:
            raise IOError(f"Failed to save dataset to {resolved_path}: {error}") from error

