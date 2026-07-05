# START_FILE: whisker/core/workflows/behavior_classification/operations/verification_operations.py [New file for verification I/O]
import logging
import os
from pathlib import Path
from typing import Optional
import pandas as pd

from whisker.core.base_operation_helper import BaseOperationHelper

class BehaviorVerificationOperations(BaseOperationHelper):
    """
    Handles saving and loading of behavior verification data.
    Structure: workspace/verification/behavior/<run_name>/<dataset_name>/<video_stem>.h5
    """

    def get_verification_path(
        self, run_name: str, dataset_name: str, video_stem: str
    ) -> Path:
        return (
            self._base_dir
            / "behavior" 
            / run_name 
            / dataset_name 
            / f"{video_stem}.h5"
        )

    def load_verification(
        self, run_name: str, dataset_name: str, video_stem: str
    ) -> Optional[pd.DataFrame]:
        """
        Loads the verification DataFrame if it exists.
        Expected columns: [video_key, behavior, start_frame, end_frame, p, status]
        """
        path = self.get_verification_path(run_name, dataset_name, video_stem)
        if not path.exists():
            return None
        
        try:
            df = pd.read_hdf(path, key="verification")
            return df
        except Exception as e:
            logging.error(f"Failed to load verification data from {path}: {e}")
            return None

    def save_verification(
        self, 
        run_name: str, 
        dataset_name: str, 
        video_stem: str, 
        df: pd.DataFrame
    ):
        """Saves the verification DataFrame."""
        path = self.get_verification_path(run_name, dataset_name, video_stem)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Ensure consistent types
            df = df.copy()
            if "status" not in df.columns:
                df["status"] = "unverified"
            
            # String types for HDF5 compatibility
            for col in ["video_key", "behavior", "status"]:
                if col in df.columns:
                    df[col] = df[col].astype(str)

            df.to_hdf(
                path, 
                key="verification", 
                mode="w", 
                format="table", 
                complevel=9, 
                complib="blosc:lz4"
            )
            logging.info(f"Saved verification data for {run_name}/{dataset_name}/{video_stem}")
        except Exception as e:
            logging.error(f"Failed to save verification data to {path}: {e}")
            raise
