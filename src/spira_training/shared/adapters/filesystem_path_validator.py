from pathlib import Path

from src.spira_training.shared.core.models.valid_path import ValidPath
from src.spira_training.shared.ports.path_validator import PathValidator


class FilesystemPathValidator(PathValidator):
    """Validates that file paths exist on the filesystem."""

    def validate_path(self, path: Path | str) -> ValidPath:
        """Validate and wrap an existing filesystem file path.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If ``path`` exists but is not a regular file.
        """
        path_obj = Path(path)

        if not path_obj.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        if not path_obj.is_file():
            raise ValueError(f"Path is not a file: {path}")

        return ValidPath(path=path_obj)
