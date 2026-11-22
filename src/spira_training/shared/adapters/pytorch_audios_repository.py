import torchaudio

from src.spira_training.shared.core.models.audio import Audio
from src.spira_training.shared.core.models.audio_metadata import AudioMetadata
from src.spira_training.shared.core.models.valid_path import ValidPath
from src.spira_training.shared.core.models.wav import Wav
from src.spira_training.shared.ports.audio_manifest_reader import AudioManifestReader
from src.spira_training.shared.ports.audios_repository import AudiosRepository
from src.spira_training.shared.ports.path_validator import PathValidator


class PytorchAudiosRepository(AudiosRepository):
    def __init__(
        self,
        manifest_reader: AudioManifestReader,
        path_validator: PathValidator,
    ):
        self.manifest_reader = manifest_reader
        self.path_validator = path_validator

    def get_audios(self) -> list[tuple[Audio, AudioMetadata]]:
        VALID_EXTENSIONS = {".wav", ".mp3", ".flac"}

        audios = []
        for metadata in self.manifest_reader.read():
            path = self.path_validator.validate_path(metadata.audio_path)

            has_valid_extension = path.path.suffix.lower() in VALID_EXTENSIONS
            if not (path.path.is_file() and has_valid_extension):
                print(f"Invalid audio file: {metadata.audio_path}")
                continue

            wav, sample_rate = self._load_audio_file(path)
            audio_sample = (Audio(wav, sample_rate), metadata)
            audios.append(audio_sample)

        return audios

    def _load_audio_file(self, valid_path: ValidPath) -> tuple[Wav, int]:
        waveform, sr = torchaudio.load(valid_path.path)

        if waveform.dim() == 2:
            waveform = waveform.mean(dim=0)

        return Wav(waveform), sr
