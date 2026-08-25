import asyncio
from pathlib import Path

from src.spira_training.shared.adapters.json_config_loader import JsonConfigLoader
from src.spira_training.shared.core.services.feature_engineering_service import (
    FeatureEngineeringService,
)
from src.spira_training.shared.core.services.randomizer import Randomizer
from src.spira_training.shared.adapters.filesystem_path_validator import (
    FilesystemPathValidator,
)
from src.spira_training.shared.adapters.csv_file_reader import CSVFileReader
from src.spira_training.shared.adapters.parquet_dataset_repository import (
    ParquetDatasetRepository,
)
from src.spira_training.shared.adapters.pytorch.model_trainer.implementations.simple_pytorch_audio_factory import (
    SimplePytorchTensorFactory,
)
from tests.unit.fakes.fake_audios_repository import FakeAudiosRepository


async def main():
    config = JsonConfigLoader().load_feature_engineering_config("path/to/config")

    randomizer = Randomizer(seed=42).initialize_random(seed=42)
    dataset_repository = ParquetDatasetRepository()
    # No production AudiosRepository implementation exists yet.
    audios_repository = FakeAudiosRepository()
    file_reader = CSVFileReader()
    path_validator = FilesystemPathValidator()
    pytorch_audio_factory = SimplePytorchTensorFactory()

    service = FeatureEngineeringService(
        config=config,
        randomizer=randomizer,
        dataset_repository=dataset_repository,
        audios_repository=audios_repository,
        file_reader=file_reader,
        path_validator=path_validator,
        pytorch_audio_factory=pytorch_audio_factory,
    )

    # TODO - Get the bucket name to save the dataset
    save_dataset_path = Path("any_model_storage_path")

    await service.execute(save_dataset_path=save_dataset_path)


if __name__ == "__main__":
    asyncio.run(main())
