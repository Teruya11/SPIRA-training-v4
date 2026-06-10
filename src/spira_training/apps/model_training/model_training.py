import asyncio
from pydantic import BaseModel
from src.spira_training.shared.core.models.path import Path
from src.spira_training.shared.core.services.model_training_service import (
    ModelTrainingService,
)
from src.spira_training.shared.adapters.parquet_dataset_repository import ParquetDatasetRepository
from src.spira_training.shared.adapters.sk_dataset_splitter import SkDatasetSplitter
from src.spira_training.shared.adapters.pytorch.model_trainer.pytorch_model_trainer import PytorchModelTrainer


class ModelTrainingConfig(BaseModel):
    dataset_path: Path
    trained_model_path: Path


async def main():
    # TODO load config
    config = ModelTrainingConfig(
        dataset_path=Path("dataset_path"),
        trained_model_path=Path("trained_model_path"),
    )

    # TODO  instantiate the dependencies using configs
    dataset_repository = ParquetDatasetRepository()
    dataset_splitter = SkDatasetSplitter()
    model_trainer = PytorchModelTrainer()
    trained_models_repository = FakeTrainedModelsRepository()

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
