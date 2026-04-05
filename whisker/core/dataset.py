import json
import logging
from enum import Enum
from typing import List

from pydantic import BaseModel, Field

VIDEO_FILE_EXTENSIONS = (
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".wmv",
    ".flv",
    ".webm",
    ".mpeg",
    ".mpg",
    ".3gp",
    ".m4v",
    ".ogv",
    ".vob",
    ".ts",
    ".mts",
    ".m2ts",
    ".seq",
)

IMAGE_FILE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp",
    ".heic",
    ".heif",
    ".raw",
    ".arw",
    ".cr2",
    ".nef",
    ".orf",
    ".sr2",
    ".dng",
    ".ico",
    ".svg",
)

class DatasetType(str, Enum):
    """Enumeration for the different types of datasets."""

    VIDEO_COLLECTION = "VIDEO_COLLECTION"
    IMAGE_COLLECTION = "IMAGE_COLLECTION"
    FRAME_SUBSET = "FRAME_SUBSET"

# Maps the expected file extensions for each DatasetType
GLOB_EXTENSIONS_PER_DATASET_TYPE = {
    DatasetType.VIDEO_COLLECTION: VIDEO_FILE_EXTENSIONS,
    DatasetType.IMAGE_COLLECTION: IMAGE_FILE_EXTENSIONS,
    DatasetType.FRAME_SUBSET: IMAGE_FILE_EXTENSIONS,
}

class Dataset(BaseModel):
    """
    Pydantic model for a single dataset's metadata.
    This provides validation and a clear structure.
    """

    name: str
    type: DatasetType
    base_data_path: str
    files: List[str] = Field(default_factory=list)

    @classmethod
    def from_json(cls, json_str: str) -> "Dataset":
        """
        Create a Dataset object from a JSON string.
        """
        data = json.loads(json_str)
        return cls(
            name=data["name"],
            base_data_path=data["base_data_path"],
            type=DatasetType(data["type"]),
            files=data["files"],
        )

    def show(self, verbose: bool = False) -> None:
        """
        Show the dataset's metadata.
        """
        if verbose:
            logging.info(f"Dataset {self.name} ({self.type.value})")
            logging.info(self.model_dump_json(indent=2))
        else:
            logging.info(
                f"Dataset {self.name} ({self.type.value}) contains {len(self.files)} files"
            )

            num_files_to_show = min(10, len(self.files))
            logging.info(f"First {num_files_to_show} files:")
            for i in range(num_files_to_show):
                logging.info(f"  - {self.files[i]}")
