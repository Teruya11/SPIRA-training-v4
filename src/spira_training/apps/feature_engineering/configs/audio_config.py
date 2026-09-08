from pathlib import Path

from pydantic import BaseModel


class DatasetPaths(BaseModel):
    patients_dir: Path
    controls_dir: Path
    noises_dir: Path

    # Backward compatibility for older CSV-based manifests.
    patients_csv: Path | None = None
    controls_csv: Path | None = None
    noises_csv: Path | None = None


class AudioConfig(BaseModel):
    dataset_paths: DatasetPaths
    normalize: bool