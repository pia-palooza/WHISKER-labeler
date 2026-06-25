# UPDATE_FILE: whisker/core/workflows/behavior_classification/data_structures.py
import logging
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
from pydantic import BaseModel
import h5py


def _h5_path(path: Path) -> str:
    """Return a path string the HDF5 C library can open on Windows.

    Python's pathlib uses the \\?\ extended-length prefix internally, so
    mkdir() succeeds even for paths near MAX_PATH (260 chars). The HDF5 C
    library receives a plain str and hits the limit, producing errno=2 even
    though the directory exists. Adding \\?\ bypasses MAX_PATH for H5F calls.
    """
    if sys.platform == "win32":
        resolved = str(path.resolve())
        if not resolved.startswith("\\\\"):
            return "\\\\?\\" + resolved
        return resolved
    return str(path)


class Bout(BaseModel):
    """A single hand-annotated behavior interval within one video."""

    start_frame: int
    end_frame: int
    p: Optional[float] = None


class BehaviorDataset:
    """
    Holds all hand-annotated behavior bouts for one video collection.

    Bouts for every video in the collection live in a single ``bouts``
    DataFrame keyed by ``video_key`` (the video filename). The on-disk HDF5
    layout is intentionally identical to the full WHISKER application so that
    label files round-trip between the two tools:

        metadata/behaviors : variable-length UTF-8 string array
        bouts              : table with columns
                             [video_key, behavior, start_frame, end_frame, p]
    """

    _BOUTS_KEY = "bouts"
    _METADATA_KEY = "metadata"

    def __init__(
        self,
        behaviors: List[str] = [],
        bouts: Optional[pd.DataFrame] = None,
    ):
        self.behaviors = list(behaviors)
        self.bouts = bouts if bouts is not None else self._create_empty_bouts_df()

    def _create_empty_bouts_df(self) -> pd.DataFrame:
        """Creates an empty, correctly typed bouts DataFrame."""
        return pd.DataFrame(
            columns=["video_key", "behavior", "start_frame", "end_frame", "p"]
        ).astype(
            {
                "video_key": "str",
                "behavior": "str",
                "start_frame": "int64",
                "end_frame": "int64",
                "p": "float32",
            }
        )

    @classmethod
    def get_video_keys_from_file(cls, file_path: Path) -> set[str]:
        """
        Fast-path to retrieve the set of labeled video keys from the bouts
        table without loading the whole dataset.
        """
        if not file_path.exists():
            return set()

        h5p = _h5_path(file_path)
        try:
            with h5py.File(h5p, "r") as f:
                if cls._BOUTS_KEY not in f:
                    return set()
            df = pd.read_hdf(h5p, key=cls._BOUTS_KEY, columns=["video_key"])
            return set(df["video_key"].unique())
        except Exception as e:
            logging.debug(f"Failed to read video keys from {file_path}: {e}")
        return set()

    @classmethod
    def from_file(cls, file_path: Path) -> "BehaviorDataset":
        """Loads behavior data from an HDF5 (.h5) file."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found at {file_path}")

        behaviors: List[str] = []
        bouts_df: Optional[pd.DataFrame] = None
        h5p = _h5_path(file_path)

        try:
            with h5py.File(h5p, "r") as f:
                meta_key = f"{cls._METADATA_KEY}/behaviors"
                if meta_key in f:
                    behaviors = [
                        b.decode("utf-8") if isinstance(b, bytes) else b
                        for b in f[meta_key][()]
                    ]
                has_bouts = cls._BOUTS_KEY in f

            if has_bouts:
                bouts_df = pd.read_hdf(h5p, key=cls._BOUTS_KEY)
        except Exception as e:
            logging.error(f"Error reading HDF5 file {file_path}: {e}")
            raise IOError(f"Error reading HDF5 file: {e}")

        return cls(behaviors=behaviors, bouts=bouts_df)

    def to_file(self, file_path: Path):
        """Saves the behavior data to an HDF5 (.h5) file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        h5p = _h5_path(file_path)

        # 1. (Re)create the file and write metadata.
        with h5py.File(h5p, "w") as f:
            behaviors_utf8 = [n.encode("utf-8") for n in self.behaviors]
            f.create_dataset(
                f"{self._METADATA_KEY}/behaviors",
                data=behaviors_utf8,
                dtype=h5py.string_dtype("utf-8"),
            )

        # 2. Append the bouts table (only if there is something to save).
        if not self.bouts.empty:
            bouts_to_save = self.bouts.copy()
            bouts_to_save["video_key"] = bouts_to_save["video_key"].astype(str)
            bouts_to_save["behavior"] = bouts_to_save["behavior"].astype(str)
            bouts_to_save.to_hdf(
                h5p,
                key=self._BOUTS_KEY,
                mode="a",
                format="table",
                data_columns=["video_key", "behavior"],
                complevel=9,
                complib="blosc:lz4",
            )

    @property
    def video_keys(self) -> list[str]:
        """Unique video keys that currently have at least one bout."""
        if self.bouts.empty:
            return []
        return list(self.bouts["video_key"].unique())

    def bouts_for_video(self, video_key: str) -> pd.DataFrame:
        """Returns the bouts belonging to a single video, sorted by start."""
        if self.bouts.empty:
            return self._create_empty_bouts_df()
        df = self.bouts[self.bouts["video_key"] == video_key].copy()
        return df.sort_values(by="start_frame")
