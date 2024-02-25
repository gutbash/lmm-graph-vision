from utils.logger import Logger
from pathlib import Path

logger = Logger(__name__)

def check_extension(filename: str, extension: str) -> str:
    if filename is not None and not filename.endswith(extension):
        logger.warning(f"Filename {filename} does not end with {extension}. Appending extension...")
        filename += extension
    return filename

def check_path_exists(path: Path) -> Path:
    if not path.exists():
        logger.warning(f"Path {path} does not exist. Creating it...")
        path.mkdir(parents=True, exist_ok=True)
    return path

def validate_path(path: Path, filename: str, extension: str) -> Path:
    filename = check_extension(filename, extension)
    path = check_path_exists(path)
    return path / filename

def has_negative_value(values: list) -> bool:
    if any(value < 0 for value in values):
        logger.error(f"Negative value found in {values}.")
        return True
    else:
        return False