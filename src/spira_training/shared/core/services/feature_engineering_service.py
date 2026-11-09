from pathlib import Path

from src.spira_training.shared.core.audio_processor import AudioProcessor
from src.spira_training.shared.core.models.dataset import Dataset
from src.spira_training.shared.ports.audios_repository import AudiosRepository
from src.spira_training.shared.ports.dataset_repository import DatasetRepository
from src.spira_training.shared.ports.audio_manifest_reader import AudioManifestReader


class FeatureEngineeringService:
    def __init__(
        self,
        metadata_path: Path,
        dataset_repository: DatasetRepository,
        audios_repository: AudiosRepository,
        manifest_reader: AudioManifestReader,
        audio_processor: AudioProcessor,
    ):
        self.metadata_path = metadata_path
        self.dataset_repository = dataset_repository
        self.audios_repository = audios_repository
        self.manifest_reader = manifest_reader
        self.audio_processor = audio_processor

    async def execute(self) -> None:
        dataset = self._generate_dataset()

        await self.dataset_repository.save_dataset(dataset)

    def _generate_dataset(self):
        metadata = self.manifest_reader.read(str(self.metadata_path))
        all_features = []
        labels = []

        for record in metadata:
            audio = self.audios_repository.get_audio(str(record.audio_path))
            all_features.append(self.audio_processor.process_audio(audio))
            labels.append(record.label)

        return Dataset(features=all_features, labels=labels)
