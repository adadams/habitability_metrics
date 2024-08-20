from pathlib import Path
from typing import Final

LOCAL_REPOSITORY_PARENT_DIRECTORY: Final[Path] = (
    Path.home() / "Documents" / "Astronomy" / "2022" / "Earth Cousins" / "Code"
)

LOCAL_REPOSITORY_DIRECTORY: Final[Path] = (
    LOCAL_REPOSITORY_PARENT_DIRECTORY / "habitability_metrics"
)

ORIGINAL_MODEL_STORAGE_DIRECTORY: Final[Path] = Path("/Volumes/Research/")

LOCAL_PROCESSED_DATASET_DIRECTORY: Final[Path] = (
    LOCAL_REPOSITORY_PARENT_DIRECTORY / "processed_datasets"
)
