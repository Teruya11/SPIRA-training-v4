from pathlib import Path

from src.spira_training.shared.core.models.valid_path import ValidPath
from src.spira_training.shared.ports.path_validator import PathValidator


class FilesystemPathValidator(PathValidator):
    """Validates that filesystem paths exist and may be files or directories."""

    def validate_path(self, path: Path | str) -> ValidPath:
        """Validate and wrap an existing filesystem path.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If ``path`` exists but is neither a file nor a directory.
        """
        path_obj = Path(path)

        if not path_obj.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        if not path_obj.is_file() and not path_obj.is_dir():
            raise ValueError(f"Path is neither a file nor a directory: {path}")

        return ValidPath(path=path_obj)
