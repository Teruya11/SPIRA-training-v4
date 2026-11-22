import asyncio
from pathlib import Path

from src.spira_training.apps.feature_engineering.configs.feature_engineering_config import (
    FeatureEngineeringConfig,
)
from src.spira_training.shared.adapters.csv_audio_manifest_reader import (
    CSVAudioManifestReader,
)
from src.spira_training.shared.adapters.filesystem_path_validator import (
    FilesystemPathValidator,
)
from src.spira_training.shared.adapters.pytorch.model_trainer.implementations.simple_pytorch_audio_factory import (
    SimplePytorchTensorFactory,
)
from src.spira_training.shared.adapters.pytorch_audios_repository import (
    PytorchAudiosRepository,
)
from src.spira_training.shared.adapters.pytorch_dataset_repository import (
    PytorchDatasetRepository,
)
from src.spira_training.shared.core.audio_processor_factory import (
    create_pytorch_audio_processor,
)
from src.spira_training.shared.core.services.feature_engineering_service import (
    FeatureEngineeringService,
)
from src.spira_training.shared.ports.path_validator import PathValidator


def make_config(
    path_validator: PathValidator,
    config_path: str | Path | None = None,
) -> FeatureEngineeringConfig:
    if config_path is None:
        config_path = Path(__file__).resolve().parents[4] / "config.json"

    config_path = path_validator.validate_path(Path(config_path)).path

    from src.spira_training.shared.adapters.json_config_loader import JsonConfigLoader

    return JsonConfigLoader(path_validator).load(config_path).feature_engineering_config


async def main():
    path_validator = FilesystemPathValidator()

    config = make_config(path_validator=path_validator)
    manifest_reader = CSVAudioManifestReader(config.audio.dataset_paths.metadata_csv, path_validator)

    pytorch_audio_factory = SimplePytorchTensorFactory()

    # Service Adapters
    dataset_repository = PytorchDatasetRepository(config.dataset_path, path_validator)
    audios_repository = PytorchAudiosRepository(manifest_reader, path_validator)
    audio_processor = create_pytorch_audio_processor(
        config.audio_processor, pytorch_audio_factory
    )

    service = FeatureEngineeringService(
        dataset_repository=dataset_repository,
        audios_repository=audios_repository,
        audio_processor=audio_processor,
    )

    await service.execute()


if __name__ == "__main__":
    asyncio.run(main())
