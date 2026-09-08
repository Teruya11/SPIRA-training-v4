import asyncio
from pathlib import Path

from src.spira_training.apps.feature_engineering.configs.audio_config import (
    AudioConfig,
    DatasetPaths,
)
from src.spira_training.apps.feature_engineering.configs.audio_feature_transformer_config import (
    AudioFeatureTransformerConfig,
    AudioFeatureTransformerOptions,
    AudioFeatureTransformersCollection,
    MixedAudioFeatureTransformerConfig,
    NoisyAudioFeatureTransformerConfig,
    OverlappedAudioFeatureTransformerConfig,
)
from src.spira_training.apps.feature_engineering.configs.audio_processor_config import (
    AudioProcessorConfig,
    AudioProcessorType,
    MFCCAudioProcessorConfig,
    MelspectrogramAudioProcessorConfig,
    SpectrogramAudioProcessorConfig,
)
from src.spira_training.apps.feature_engineering.configs.feature_engineering_config import (
    FeatureEngineeringConfig,
)
from src.spira_training.shared.adapters.audios_repository import AudiosRepository
from src.spira_training.shared.adapters.parquet_dataset_repository import (
    ParquetDatasetRepository,
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


def make_config() -> FeatureEngineeringConfig:
    return FeatureEngineeringConfig(
        audio=AudioConfig(
            dataset_paths=DatasetPaths(
                patients_dir=Path("data/patients"),
                controls_dir=Path("data/controls"),
                noises_dir=Path("data/noises"),
            ),
            normalize=True,
        ),
        audio_processor=AudioProcessorConfig(
            feature_type=AudioProcessorType.MFCC,
            hop_length=512,
            mfcc=MFCCAudioProcessorConfig(
                sample_rate=16000,
                num_mels=40,
                num_mfcc=13,
                log_mels=True,
                n_fft=512,
                win_length=400,
            ),
            spectrogram=SpectrogramAudioProcessorConfig(
                sample_rate=16000,
                num_mels=40,
                mel_fmin=0.0,
                mel_fmax=8000.0,
                num_mfcc=13,
                log_mels=True,
                n_fft=512,
                num_freq=257,
                win_length=400,
            ),
            melspectrogram=MelspectrogramAudioProcessorConfig(
                sample_rate=16000,
                num_mels=40,
                mel_fmin=0.0,
                mel_fmax=8000.0,
                num_mfcc=13,
                log_mels=True,
                n_fft=512,
                num_freq=257,
                win_length=400,
            ),
        ),
        audio_feature_transformer=AudioFeatureTransformerConfig(
            options=AudioFeatureTransformerOptions(
                use_noise=True,
                use_overlapping=True,
                use_padding=True,
                use_mixture=True,
            ),
            audio_feature_transformers=AudioFeatureTransformersCollection(
                noisy_audio=NoisyAudioFeatureTransformerConfig(
                    num_noise_control=5,
                    num_noise_patient=5,
                    noise_max_amp=0.5,
                    noise_min_amp=0.1,
                ),
                overlapped_audio=OverlappedAudioFeatureTransformerConfig(
                    window_length=400,
                    step_size=160,
                ),
                mixed_audio=MixedAudioFeatureTransformerConfig(
                    alpha=0.5,
                    beta=0.5,
                ),
            ),
        ),
    )


async def main():
    config = make_config()

    dataset_repository = ParquetDatasetRepository()
    audios_repository = AudiosRepository()
    pytorch_audio_factory = SimplePytorchTensorFactory()
    audio_processor = create_audio_processor(
        config.audio_processor, pytorch_audio_factory
    )

    service = FeatureEngineeringService(
        config=config,
        dataset_repository=dataset_repository,
        audios_repository=audios_repository,
        audio_processor=audio_processor,
    )

    save_dataset_path = Path("tmp/feature_engineering_pipeline_dataset.parquet")
    save_dataset_path.parent.mkdir(parents=True, exist_ok=True)

    await service.execute(save_dataset_path=save_dataset_path)


if __name__ == "__main__":
    asyncio.run(main())
