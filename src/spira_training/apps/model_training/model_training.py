import asyncio
from pydantic import BaseModel
from src.spira_training.shared.adapters.parquet_dataset_repository import (
    ParquetDatasetRepository,
)
from src.spira_training.shared.adapters.filesystem_trained_models_repository import (
    FilesystemTrainedModelsRepository,
)
from src.spira_training.shared.adapters.sk_dataset_splitter import SkDatasetSplitter
from src.spira_training.shared.adapters.parquet_dataset_repository import (
    ParquetDatasetRepository,
)
from src.spira_training.shared.adapters.filesystem_trained_models_repository import (
    FilesystemTrainedModelsRepository,
)
from src.spira_training.shared.core.models.path import Path
from src.spira_training.shared.core.services.model_training_service import (
    ModelTrainingService,
)
from tests.unit.fakes.fake_model_trainer import FakeModelTrainer


class ModelTrainingConfig(BaseModel):
    dataset_path: Path
    trained_model_path: Path


async def main():
    # TODO load config
    config = ModelTrainingConfig(
        dataset_path=Path("dataset_path"),
        trained_model_path=Path("trained_model_path"),
    )

    # TODO instantiate the dependencies using configs
    dataset_repository = ParquetDatasetRepository()
    dataset_splitter = SkDatasetSplitter()
    model_trainer = FakeModelTrainer()
    trained_models_repository = FilesystemTrainedModelsRepository()

    service = ModelTrainingService(
        dataset_repository=dataset_repository,
        dataset_splitter=dataset_splitter,
        model_trainer=model_trainer,
        trained_models_repository=trained_models_repository,
    )

    await service.execute(
        dataset_path=config.dataset_path, trained_model_path=config.trained_model_path
    )


if __name__ == "__main__":
    asyncio.run(main())
