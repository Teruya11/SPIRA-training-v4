from src.spira_training.shared.core.models.dataset import Dataset
from src.spira_training.shared.ports.audio_processor import AudioProcessor
from src.spira_training.shared.ports.audios_repository import AudiosRepository
from src.spira_training.shared.ports.dataset_repository import DatasetRepository


class FeatureEngineeringService:
    def __init__(
        self,
        dataset_repository: DatasetRepository,
        audios_repository: AudiosRepository,
        audio_processor: AudioProcessor,
    ):
        self.dataset_repository = dataset_repository
        self.audios_repository = audios_repository
        self.audio_processor = audio_processor

    async def execute(self) -> None:
        dataset = self._generate_dataset()

        await self.dataset_repository.save_dataset(dataset)

    def _generate_dataset(self):
        all_features = []
        labels = []

        for audio, metadata in self.audios_repository.get_audios():
            processed_audio = self.audio_processor.process_audio(audio)

            all_features.append(processed_audio)
            labels.append(metadata.label)

        return Dataset(features=all_features, labels=labels)
