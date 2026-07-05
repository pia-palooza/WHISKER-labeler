import logging
import os
from pathlib import Path
from typing import Optional
import pandas as pd
import shutil

import whisker.core.constants as constants
from whisker.core.study.dataset import Dataset, DatasetType, VIDEO_FILE_EXTENSIONS
from whisker.core.base_workflow_operation_helper import BaseWorkflowOperationsHelper
from whisker.core.study.dataset_operations import DatasetOperations
from whisker.core.study.project_operations import ProjectOperations

from .data_structures import BehaviorDataset

class BehaviorLabelOperations(BaseWorkflowOperationsHelper):
    def __init__(
        self,
        base_dir: Path,
        projects: ProjectOperations,
        datasets: DatasetOperations
    ):
        super().__init__(base_dir, projects, datasets)
        self._projects: ProjectOperations = projects
        self._datasets: DatasetOperations = datasets
        self._behavior_labels: dict[str, Optional[BehaviorDataset]] = {}

    def scan_labels(self):
        """Scans for existing behavior label files (lazy load)."""
        self._behavior_labels.clear()
        if self._base_dir.exists():
            discovered_entries = []
            for name in os.listdir(self._base_dir):
                if name.startswith("."): continue
                if (self._base_dir / name / "labels.h5").exists():
                    discovered_entries.append(name)
                    self._behavior_labels[name] = None # Marker for lazy load
            logging.info(
                f"[BehaviorLabelOperations] Completed scanning for behavior labels. "
                f"Found {len(discovered_entries)} datasets with labels."
            )
            for entry in discovered_entries:
                logging.debug(f"  - {entry}")

    def list_labelled_datasets(self) -> set[str]:
        """Dataset names that have a behavior_labels file on disk."""
        return set(self._behavior_labels.keys())

    def get_video_behavior_labels(self, dataset_name: str, video_stem: str) -> Optional[pd.DataFrame]:
        labels = self.get_behavior_labels(dataset_name)
        if labels is None or labels.bouts.empty:
            return None

        # 1. Try exact match
        mask = labels.bouts['video_key'] == video_stem

        # 2. If no match, try appending extensions
        if not mask.any():
            for ext in VIDEO_FILE_EXTENSIONS:
                candidate_key = f"{video_stem}{ext}"
                mask = labels.bouts['video_key'] == candidate_key
                if mask.any():
                    logging.debug(f"Matched video stem '{video_stem}' to label key '{candidate_key}'")
                    break

        video_bouts = labels.bouts[mask]
        
        if video_bouts.empty:
            return None

        return video_bouts.drop(columns=['video_key'])

    def import_behavior_labels(
        self,
        dataset_name: str,
        project_name: str,
        import_path: Path,
        backend: constants.BackendEnum,
        warn_if_exists: constants.WarnIfExistsFunctionType = None,
    ) -> str:
        # 1. Create/Get Dataset
        if dataset_name not in self._datasets.keys():
            self._create_video_dataset_for_import(dataset_name, import_path, warn_if_exists)

        # 2. Clean existing labels
        labels_file = self._base_dir / dataset_name / "labels.h5"
        if labels_file.exists():
            if warn_if_exists and not warn_if_exists(f"Overwrite existing behavior labels for '{dataset_name}'?"):
                return "Import cancelled."
            os.remove(labels_file)

        # 3. Perform Import
        project = self._projects.get(project_name)
        if not project:
            raise ValueError(f"Project '{project_name}' not found.")

        if backend == constants.BackendEnum.MARS:
            import whisker.third_party.mars_api as mars_api
            behavior_dataset, report = mars_api.import_behavior_labels(
                import_path, project.behaviors
            )
        else:
            raise NotImplementedError(f"Behavior import not implemented for {backend}")

        # 4. Save
        self._behavior_labels[dataset_name] = behavior_dataset
        self.save_behavior_labels(dataset_name)
        
        return report
    
    def remove(self, dataset_name: str) -> None:
        dataset = self._behavior_labels.get(dataset_name)
        if not dataset:
            logging.warning(f"Called to remove BehaviorDataset {dataset} but no such BehaviorDataset exists.")
            return
        
        del self._behavior_labels[dataset_name]
        shutil.rmtree(self.base_dir / dataset_name)
        logging.info(f"Removed BehaviorDataset {dataset_name} from the workspace")

    def _create_video_dataset_for_import(self, dataset_name: str, import_path: Path, warn_if_exists):
        """Helper to create a VIDEO_COLLECTION dataset entry if one doesn't exist during import."""
        logging.info(f"Dataset '{dataset_name}' not found. Creating new VIDEO_COLLECTION.")
        
        video_files = []
        for ext in VIDEO_FILE_EXTENSIONS:
            video_files.extend(import_path.rglob(f"*{ext}"))
        
        if not video_files:
            raise FileNotFoundError("Import failed: No video files found in the specified directory.")

        relative_files = [
            str(f.relative_to(import_path)).replace("\\", "/") 
            for f in sorted(video_files)
        ]

        new_dataset = Dataset(
            name=dataset_name,
            base_data_path=str(import_path.resolve()),
            type=DatasetType.VIDEO_COLLECTION,
            files=relative_files
        )
        self._datasets.add_dataset(dataset_name, new_dataset, warn_if_exists=warn_if_exists)
        logging.info(f"Created new VIDEO_COLLECTION dataset '{dataset_name}' for the new behavior labels.")


    def get_behavior_labels(self, dataset_name: str) -> BehaviorDataset:
        """Core lazy-loading logic for behavior labels."""
        if dataset_name in self._behavior_labels:
            if self._behavior_labels[dataset_name]:
                # Case 0: Labels already loaded from disk
                labels = self._behavior_labels[dataset_name]
            else:
                # Case 1: Labels exist on disk but haven't been loaded yet (value is None).
                logging.info(
                    f"[BehaviorLabelOperations] Loading behavior labels for dataset '{dataset_name}'..."
                )
                file_path = self._base_dir / dataset_name / "labels.h5"
                try:
                    labels = BehaviorDataset.from_file(file_path)
                except Exception as e:
                    logging.error(
                        f"Failed to load behavior labels from {file_path}: {e}",
                        exc_info=True,
                    )
                    labels = BehaviorDataset() # Fallback

        # Case 2: No labels were found on disk during the initial startup scan.
        else:
            logging.info(
                f"No behavior labels found for '{dataset_name}', creating new in-memory object."
            )
            behaviors = []
            if self._projects:
                first_project = next(iter(self._projects.values()))
                behaviors = first_project.behaviors
                logging.info(f"Assigning behaviors from project '{first_project.name}' to new BehaviorDataset.")

            labels = BehaviorDataset(behaviors=behaviors)

        self._behavior_labels[dataset_name] = labels
        return labels

    def has_behavior_labels(self, dataset_name: str) -> bool:
        return dataset_name in self._behavior_labels

    def get_behavior_labeled_video_keys(self, dataset_name: str) -> set[str]:
        """
        Returns the set of video keys that have behavior labels for a given dataset.
        Utilizes a fast-path to avoid loading the entire HDF5 file if possible.
        """
        # If already loaded in memory, use that
        if dataset_name in self._behavior_labels and self._behavior_labels[dataset_name]:
            labels = self._behavior_labels[dataset_name]
            if not labels.bouts.empty:
                return set(labels.bouts["video_key"].unique())
            return set()
        
        # Fast-path: peek at the file
        file_path = self._base_dir / dataset_name / "labels.h5"
        return BehaviorDataset.get_video_keys_from_file(file_path)

    def save_behavior_labels(self, dataset_name: str) -> None:
        """Saves the behavior labels for a given dataset to an HDF5 file."""
        labels = self.get_behavior_labels(dataset_name)
        if not labels:
            logging.warning(f"No behavior labels to save for dataset '{dataset_name}'.")
            return

        output_path = self._base_dir / dataset_name / "labels.h5"
        os.makedirs(output_path.parent, exist_ok=True)
        
        try:
            labels.to_file(output_path)
            logging.info(f"Saved behavior labels for '{dataset_name}' to {output_path}")
        except Exception as e:
            logging.error(f"Failed to save behavior labels to {output_path}: {e}", exc_info=True)
            raise

import numpy as np

def create_frame_wise_labels(
    behavior_dataset: BehaviorDataset,
    video_key: str,
    all_behaviors: list[str],
    num_frames: int,
) -> pd.DataFrame:
    """
    Converts bouts from a BehaviorDataset for a specific video into a
    frame-wise, multi-label binary DataFrame.
    """
    # Create the base DataFrame
    labels_df = pd.DataFrame(
        np.zeros((num_frames, len(all_behaviors)), dtype=np.float32),
        index=pd.RangeIndex(num_frames, name="frame_index"),
        columns=all_behaviors,
    )

    if behavior_dataset.bouts.empty:
        logging.info("DEBUG: dataprep: Bouts df is empty.")
        return labels_df

    # --- Robust Key Matching ---
    # 1. Try exact match first
    mask = behavior_dataset.bouts["video_key"] == video_key

    # 2. Fallback: Fuzzy match on filename stem
    if not mask.any():
        available_keys = behavior_dataset.bouts["video_key"].unique()
        target_stem = Path(video_key).stem

        candidate_key = next(
            (k for k in available_keys if Path(str(k)).stem == target_stem), 
            None
        )

        if candidate_key:
            mask = behavior_dataset.bouts["video_key"] == candidate_key

    # Apply mask
    video_bouts_df = behavior_dataset.bouts[mask]

    if video_bouts_df.empty:
        return labels_df

    # Fill in the 1s based on the bouts
    for row in video_bouts_df.itertuples():
        behavior_name = row.behavior
        if behavior_name not in all_behaviors:
            continue

        # Ensure bouts are within the video frame bounds
        start = max(0, int(row.start_frame))
        end = min(num_frames - 1, int(row.end_frame))
        if start <= end:
            # Use .loc to set the values for the frame range
            labels_df.loc[start : end, behavior_name] = 1.0

    return labels_df
