from pathlib import Path

import pandas as pd

from src.spira_training.shared.core.models.audio_metadata import AudioMetadata
from src.spira_training.shared.core.models.dataset import Label
from src.spira_training.shared.ports.audio_manifest_reader import AudioManifestReader


class CSVAudioManifestReader(AudioManifestReader):
    required_columns = {
        "patient_hash",
        "audio_type",
        "collection_site",
        "collection_date",
        "age",
        "gender",
        "education",
        "smoker_status",
        "saturation",
        "other_comorbidities",
        "audio_path",
    }

    def read(self, path: str) -> list[AudioMetadata]:
        dataframe = pd.read_csv(Path(path))
        missing_columns = self.required_columns - set(dataframe.columns)
        if missing_columns:
            raise ValueError(
                f"Metadata CSV is missing columns: {sorted(missing_columns)}"
            )

        records = []
        for row in dataframe.itertuples(index=False):
            audio_path_value = str(row.audio_path).strip()
            saturation = row.saturation
            if not audio_path_value or pd.isna(saturation):
                raise ValueError(
                    "Metadata CSV contains an empty audio_path or saturation"
                )
            audio_path = Path(audio_path_value)
            if not audio_path.is_absolute():
                audio_path = Path(path).parent / audio_path
            try:
                insufficiency = 100 - float(saturation)
                label = (
                    Label.POSITIVE
                    if insufficiency > 5
                    else Label.NEGATIVE
                )
            except (TypeError, ValueError) as error:
                raise ValueError("saturation must be numeric") from error
            records.append(AudioMetadata(audio_path=audio_path, label=label))

        return records