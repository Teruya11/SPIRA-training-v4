import asyncio
from pathlib import Path as PathLibPath

import torch
from pydantic import BaseModel

from src.spira_training.shared.adapters.debug_train_logger import DebugTrainLogger
from src.spira_training.shared.adapters.filesystem_trained_models_repository import (
    FilesystemTrainedModelsRepository,
)
from src.spira_training.shared.adapters.filesystem_path_validator import (
    FilesystemPathValidator,
)
from src.spira_training.shared.adapters.pickle_dataset_repository import (
    PickleDatasetRepository,
)
from src.spira_training.shared.adapters.pytorch.model_trainer.implementations.noam_lr_pytorch_scheduler import (
    NoamLRPytorchScheduler,
)
from src.spira_training.shared.adapters.pytorch.model_trainer.implementations.pytorch_checkpoint_manager.filesystem_pytorch_checkpoint_manager import (
    FilesystemPytorchCheckpointManager,
)
from src.spira_training.shared.adapters.pytorch.model_trainer.implementations.pytorch_checkpoint_manager.filesystem_checkpoint_builder import (
    FileSystemCheckpointBuilder,
)
from src.spira_training.shared.adapters.pytorch.model_trainer.implementations.pytorch_cnn_builder.overlapped_features import (
    OverlappedFeaturesPytorchCnnBuilder,
)
from src.spira_training.shared.adapters.pytorch.model_trainer.implementations.pytorch_loss_calculator.multiple_loss_calculator import (
    AverageMultipleLossCalculator,
)
from src.spira_training.shared.adapters.pytorch.model_trainer.implementations.pytorch_loss_calculator.single_loss_calculator import (
    BCELossCalculator,
)
from src.spira_training.shared.adapters.pytorch.model_trainer.implementations.pytorch_model.inner_torch_model import (
    InnerTorchModel,
)
from src.spira_training.shared.adapters.pytorch.model_trainer.implementations.pytorch_model.simple_pytorch_model import (
    SimplePytorchModel,
)
from src.spira_training.shared.adapters.pytorch.model_trainer.implementations.simple_pytorch_audio_factory import (
    SimplePytorchTensorFactory,
)
from src.spira_training.shared.adapters.pytorch.model_trainer.implementations.simple_pytorch_dataloader_factory import (
    SimplePytorchDataloaderFactory,
)
from src.spira_training.shared.adapters.pytorch.model_trainer.implementations.simple_pytorch_optimizer import (
    SimplePytorchOptimizer,
)
from src.spira_training.shared.adapters.pytorch.model_trainer.interfaces.pytorch_cnn_builder import (
    PytorchCnnConfig,
)
from src.spira_training.shared.adapters.pytorch.model_trainer.pytorch_model_trainer import (
    PytorchModelTrainer,
)
from src.spira_training.shared.adapters.sk_dataset_splitter import SkDatasetSplitter
from src.spira_training.shared.core.models.path import Path
from src.spira_training.shared.core.services.model_training_service import (
    ModelTrainingService,
)


class ModelTrainingConfig(BaseModel):
    dataset_path: Path
    trained_model_path: Path


async def main():
    # TODO load config
    config = ModelTrainingConfig(
        dataset_path=Path(PathLibPath("tmp/feature_engineering_pipeline_dataset.pkl")),
        trained_model_path=Path(PathLibPath("tmp/model_training_pipeline.pt")),
    )

    # TODO instantiate the dependencies using configs
    dataset_repository = PickleDatasetRepository()
    dataset_splitter = SkDatasetSplitter()

    cnn_builder = OverlappedFeaturesPytorchCnnBuilder(
        config=PytorchCnnConfig(
            name="overlapped_features",
            fc1_dim=3,
            fc2_dim=1,
        ),
        num_features=13,
    )
    inner_model = InnerTorchModel(cnn_builder=cnn_builder)
    model = SimplePytorchModel(model=inner_model)

    pytorch_optimizer = SimplePytorchOptimizer(
        torch_optimizer=torch.optim.Adam(inner_model.parameters(), lr=1e-3)
    )
    tensor_factory = SimplePytorchTensorFactory()
    train_dataloader_factory = SimplePytorchDataloaderFactory(
        batch_size=1,
        num_workers=0,
        dataloader_type="train",
        pytorch_tensor_factory=tensor_factory,
    )
    test_dataloader_factory = SimplePytorchDataloaderFactory(
        batch_size=1,
        num_workers=0,
        dataloader_type="test",
        pytorch_tensor_factory=tensor_factory,
    )
    train_loss_calculator = AverageMultipleLossCalculator(
        single_loss_calculator=BCELossCalculator(reduction="none")
    )
    test_loss_calculator = AverageMultipleLossCalculator(
        single_loss_calculator=BCELossCalculator(reduction="sum")
    )
    scheduler = NoamLRPytorchScheduler(
        pytorch_optimizer_wrapper=pytorch_optimizer,
        warmup_steps=10,
    )
    path_validator = FilesystemPathValidator()
    checkpoint_manager = FilesystemPytorchCheckpointManager(
        checkpoint_builder=FileSystemCheckpointBuilder(
            checkpoint_dir=path_validator.validate_path(PathLibPath("tmp")),
            checkpoint_interval=10,
            fs_path_validator=path_validator,
        ),
        model=model,
        optimizer=pytorch_optimizer,
        initial_checkpoint=None,
    )
    model_trainer = PytorchModelTrainer(
        base_model=model,
        optimizer=pytorch_optimizer,
        train_dataloader_factory=train_dataloader_factory,
        test_dataloader_factory=test_dataloader_factory,
        train_loss_calculator=train_loss_calculator,
        test_loss_calculator=test_loss_calculator,
        train_logger=DebugTrainLogger(),
        scheduler=scheduler,
        checkpoint_manager=checkpoint_manager,
    )
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
