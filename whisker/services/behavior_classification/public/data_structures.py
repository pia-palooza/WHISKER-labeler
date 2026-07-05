import dataclasses
import logging
import sys
from pathlib import Path
from typing import List, Optional, Dict

import pandas as pd
import numpy as np
from pydantic import BaseModel
import h5py


def _h5_path(path: Path) -> str:
    """Return a path string the HDF5 C library can open on Windows.

    Python's pathlib uses the \\?\\ extended-length prefix internally, so
    mkdir() succeeds even for paths near MAX_PATH (260 chars). The HDF5 C
    library receives a plain str and hits the limit, producing errno=2 even
    though the directory exists. Adding \\?\\ bypasses MAX_PATH for H5F calls.
    """
    if sys.platform == "win32":
        resolved = str(path.resolve())
        if not resolved.startswith("\\\\"):
            return "\\\\?\\" + resolved
        return resolved
    return str(path)

# DEV_NOTE: This model is moved from the old behavior_labels.py
# It's still useful as a simple data container for the UI.
class Bout(BaseModel):
    start_frame: int
    end_frame: int
    p: Optional[float] = None


class BehaviorDataset:
    """
    Represents all behavior data for a dataset, including per-frame
    probabilities and/or discrete bouts.
    
    This data is stored in a single HDF5 file.
    """

    _PROBS_KEY = "per_frame_probabilities"
    _BOUTS_KEY = "bouts"
    _METADATA_KEY = "metadata"
    # For compatibility with prediction worker output
    _LEGACY_PREDICTIONS_KEY = "predictions"

    def __init__(
        self,
        behaviors: List[str] = [],
        per_frame_probabilities: Optional[pd.DataFrame] = None,
        bouts: Optional[pd.DataFrame] = None,
        pose_run_name: Optional[str] = None,
    ):
        """
        Initializes the dataset.
        
        Args:
            behaviors: Canonical list of behavior names.
            per_frame_probabilities: DataFrame indexed by 'frame_index'
                                     with behavior names as columns.
            bouts: DataFrame with columns ['video_key', 'behavior', 
                                          'start_frame', 'end_frame', 'p'].
            pose_run_name: Optional name of the pose estimation run used for predictions.
        """
        self.behaviors = behaviors
        self.per_frame_probabilities = (
            per_frame_probabilities
            if per_frame_probabilities is not None
            else self._create_empty_probs_df()
        )
        self.bouts = (
            bouts if bouts is not None else self._create_empty_bouts_df()
        )
        self.pose_run_name = pose_run_name

    def _create_empty_probs_df(self) -> pd.DataFrame:
        """Creates an empty, correctly typed per-frame probabilities DataFrame."""
        df = pd.DataFrame(columns=self.behaviors).astype("float32")
        df.index.name = "frame_index"
        return df

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
        Fast-path to retrieve unique video keys from the bouts table
        without loading the entire dataset into memory.
        """
        if not file_path.exists():
            return set()
        
        h5p = _h5_path(file_path)
        try:
            with h5py.File(h5p, "r") as f:
                if cls._BOUTS_KEY in f:
                    # Use pandas to read only the video_key column
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

        behaviors = []
        pose_run_name: Optional[str] = None
        probs_df: Optional[pd.DataFrame] = None
        bouts_df: Optional[pd.DataFrame] = None

        h5p = _h5_path(file_path)
        try:
            with h5py.File(h5p, "r") as f:
                # 1. Load Metadata
                if f"{cls._METADATA_KEY}/behaviors" in f:
                    behaviors_data = f[f"{cls._METADATA_KEY}/behaviors"][()]
                    # Handle bytes from HDF5
                    behaviors = [
                        b.decode("utf-8") if isinstance(b, bytes) else b
                        for b in behaviors_data
                    ]

                if f"{cls._METADATA_KEY}/pose_run_name" in f:
                    pose_run_name_data = f[f"{cls._METADATA_KEY}/pose_run_name"][()]
                    if isinstance(pose_run_name_data, bytes):
                        pose_run_name = pose_run_name_data.decode("utf-8")
                    else:
                        pose_run_name = str(pose_run_name_data)

                # 2. Load Per-Frame Probs
                if cls._PROBS_KEY in f:
                    probs_df = pd.read_hdf(h5p, key=cls._PROBS_KEY)
                # Compatibility: check for prediction worker output
                elif cls._LEGACY_PREDICTIONS_KEY in f:
                    logging.info(
                        f"Found legacy '{cls._LEGACY_PREDICTIONS_KEY}' key. "
                        f"Loading as per-frame probabilities."
                    )
                    probs_df = pd.read_hdf(h5p, key=cls._LEGACY_PREDICTIONS_KEY)
                    # Ensure columns match metadata if available
                    if behaviors and list(probs_df.columns) != behaviors:
                        logging.warning("Legacy predictions columns mismatch metadata.")

                # 3. Load Bouts
                if cls._BOUTS_KEY in f:
                    bouts_df = pd.read_hdf(h5p, key=cls._BOUTS_KEY)

        except Exception as e:
            logging.error(f"Error reading HDF5 file {file_path}: {e}")
            raise IOError(f"Error reading HDF5 file: {e}")

        return cls(
            behaviors=behaviors,
            per_frame_probabilities=probs_df,
            bouts=bouts_df,
            pose_run_name=pose_run_name,
        )

    def to_file(self, file_path: Path):
        """Saves the behavior data to an HDF5 (.h5) file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        h5p = _h5_path(file_path)

        # 1. Save Metadata
        with h5py.File(h5p, "w") as f:
            # Store behaviors as a variable-length string array
            behaviors_utf8 = [n.encode("utf-8") for n in self.behaviors]
            f.create_dataset(
                f"{self._METADATA_KEY}/behaviors",
                data=behaviors_utf8,
                dtype=h5py.string_dtype("utf-8"),
            )

            if self.pose_run_name:
                f.create_dataset(
                    f"{self._METADATA_KEY}/pose_run_name",
                    data=self.pose_run_name.encode("utf-8"),
                    dtype=h5py.string_dtype("utf-8"),
                )

        # 2. Save DataFrames (if they are not empty)
        if not self.per_frame_probabilities.empty:
            self.per_frame_probabilities.to_hdf(
                h5p,
                key=self._PROBS_KEY,
                mode="a",
                format="table",
                complevel=9,
                complib="blosc:lz4",
            )

        if not self.bouts.empty:
            # Ensure string columns are properly saved
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


@dataclasses.dataclass
class BoutExtractionParams:
    min_bout_duration_sec: float
    probability_threshold: float
    max_gap_fill_sec: float

@dataclasses.dataclass
class BehaviorFeatureExtractionParams:
    """
    Configuration parameters for behavior feature extraction.
    """
    # Core Metadata
    model_identities: List[str]
    project_bodyparts: List[str]
    root_bodypart: str
    
    # Reference / Normalization
    root_individual_model_id: Optional[str] = None
    heading_axis: Optional[tuple[str, str]] = None
    
    # Structural Features
    skeleton: Optional[List[tuple[str, str]]] = None
    canvas_size: Optional[tuple[int, int]] = None
    
    # Angle Feature Configuration
    use_angles: bool = True
    custom_angle_triplets: Optional[List[tuple[str, str, str]]] = None    

    # Filtering / Smoothing
    max_vel: float = 50.0
    confidence_smoothing_window: int = 5
    