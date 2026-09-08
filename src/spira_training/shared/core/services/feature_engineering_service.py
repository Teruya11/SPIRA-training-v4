from pathlib import Path

from src.spira_training.apps.feature_engineering.configs.feature_engineering_config import (
    FeatureEngineeringConfig,
)
from src.spira_training.shared.core.audio_processor import AudioProcessor
from src.spira_training.shared.core.models.dataset import Dataset, Label
from src.spira_training.shared.ports.audios_repository import AudiosRepository
from src.spira_training.shared.ports.dataset_repository import DatasetRepository


class FeatureEngineeringService:
    def __init__(
        self,
        config: FeatureEngineeringConfig,
        dataset_repository: DatasetRepository,
        audios_repository: AudiosRepository,
        audio_processor: AudioProcessor,
    ):
        self.config = config
        self.dataset_repository = dataset_repository
        self.audios_repository = audios_repository
        self.audio_processor = audio_processor

    async def execute(self, save_dataset_path: Path) -> None:
        dataset = self._generate_dataset()

        await self.dataset_repository.save_dataset(dataset, save_dataset_path)  # type: ignore

    def _load_data(self):
        patients_inputs = self._load_audio_data(
            self.config.audio.dataset_paths.patients_dir
        )
        controls_inputs = self._load_audio_data(
            self.config.audio.dataset_paths.controls_dir
        )
        noises = self._load_audio_data(self.config.audio.dataset_paths.noises_dir)

        return patients_inputs, controls_inputs, noises

    def _load_audio_data(self, input_path: Path):
        return self.audios_repository.get_audios(str(input_path))

    def _generate_dataset(self):
        patients_inputs, controls_inputs, noises = self._load_data()

        processed_patients = [
            self.audio_processor.process_audio(audio) for audio in patients_inputs
        ]
        processed_controls = [
            self.audio_processor.process_audio(audio) for audio in controls_inputs
        ]
        processed_noises = [
            self.audio_processor.process_audio(audio) for audio in noises
        ]

        all_features = processed_patients + processed_controls + processed_noises
        labels = (
            [Label.POSITIVE] * len(processed_patients)
            + [Label.NEGATIVE] * (len(processed_controls) + len(processed_noises))
        )

        return Dataset(features=all_features, labels=labels)
