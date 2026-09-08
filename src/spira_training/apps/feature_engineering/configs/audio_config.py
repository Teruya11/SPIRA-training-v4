from pathlib import Path

from pydantic import BaseModel


class DatasetPaths(BaseModel):
    metadata_csv: Path


class AudioConfig(BaseModel):
    dataset_paths: DatasetPaths
    normalize: bool