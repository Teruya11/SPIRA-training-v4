import asyncio
from pathlib import Path

from src.spira_training.apps.feature_engineering.configs.feature_engineering_config import (
    FeatureEngineeringConfig,
)
from src.spira_training.shared.adapters.audios_repository import AudiosRepository
from src.spira_training.shared.adapters.csv_audio_manifest_reader import (
    CSVAudioManifestReader,
)
from src.spira_training.shared.adapters.filesystem_path_validator import FilesystemPathValidator
from src.spira_training.shared.adapters.pytorch_dataset_repository import (
    PytorchDatasetRepository,
)
from src.spira_training.shared.adapters.pytorch.model_trainer.implementations.simple_pytorch_audio_factory import (
    SimplePytorchTensorFactory,
)
from src.spira_training.shared.core.audio_processor_factory import (
    create_audio_processor,
)
from src.spira_training.shared.core.services.feature_engineering_service import (
    FeatureEngineeringService,
)
from src.spira_training.shared.ports.path_validator import PathValidator


def make_config(
    config_path: str | Path | None = None,
    path_validator: PathValidator = FilesystemPathValidator(),
) -> FeatureEngineeringConfig:
    if config_path is None:
        config_path = Path(__file__).resolve().parents[4] / "config.json"

    config_path = path_validator.validate_path(Path(config_path)).path

    from src.spira_training.shared.adapters.json_config_loader import JsonConfigLoader

    return JsonConfigLoader(path_validator).load(config_path).feature_engineering_config


async def main():
    path_validator = FilesystemPathValidator()
    config = make_config(path_validator=path_validator)

    dataset_repository = PytorchDatasetRepository(
        config.dataset_base_path, path_validator
    )
    audios_repository = AudiosRepository(path_validator=path_validator)
    manifest_reader = CSVAudioManifestReader(path_validator)

    pytorch_audio_factory = SimplePytorchTensorFactory()
    audio_processor = create_audio_processor(
        config.audio_processor, pytorch_audio_factory
    )

    service = FeatureEngineeringService(
        config=config,
        dataset_repository=dataset_repository,
        audios_repository=audios_repository,
        manifest_reader=manifest_reader,
        audio_processor=audio_processor,
    )

    await service.execute(Path(config.dataset_file))


if __name__ == "__main__":
    asyncio.run(main())
