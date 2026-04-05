import logging
import json
import os
from pathlib import Path
import shutil
from typing import Dict, Optional

import pandas as pd

import whisker.core.constants as constants
from whisker.core.dataset import Dataset, DatasetType, IMAGE_FILE_EXTENSIONS
from whisker.core.base_workflow_operation_helper import BaseWorkflowOperationsHelper
from whisker.core.dataset_operations import DatasetOperations
from whisker.core.project_operations import ProjectOperations

from ..data_structures import PoseDataset

class PoseLabelOperations(BaseWorkflowOperationsHelper):
    def __init__(
        self,
        base_dir: Path,
        projects: ProjectOperations,
        datasets: DatasetOperations
    ):
        super().__init__(base_dir, projects, datasets)
        self._projects: ProjectOperations = projects
        self._datasets: DatasetOperations = datasets
        self._pose_labels: dict[str, Optional[PoseDataset]] = {}
        self._pose_label_metadata_cache: dict[str, dict] = {}

    def scan_labels(self):
        """Scans for existing pose label files (lazy load)."""
        self._pose_labels.clear()
        if self._base_dir.exists():
            discovered_entries = []
            for name in os.listdir(self._base_dir):
                if name.startswith("."): continue
                if (self._base_dir / name / "labels.h5").exists():
                    discovered_entries.append(name)
                    self._pose_labels[name] = None # Marker for lazy load
            logging.info(
                f"[PoseLabelOperations] Discovered labels for {len(discovered_entries)} "
                f"datasets: {discovered_entries}"
            )

    def has_pose_labels(self, name: str):
        return name in self._pose_labels

    def _get_pose_label_metadata(self, dataset_name: str) -> Optional[Dict]:
        """Loads and caches the metadata.json file for a given dataset."""
        if dataset_name in self._pose_label_metadata_cache:
            return self._pose_label_metadata_cache[dataset_name]

        metadata_path = self._base_dir / dataset_name / "metadata.json"
        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                self._pose_label_metadata_cache[dataset_name] = metadata
                return metadata
        except (json.JSONDecodeError, IOError) as e:
            logging.warning(
                f"Could not read pose label metadata for '{dataset_name}': {e}"
            )
            return None

    def get_pose_labeled_image_keys_from_summary(self, dataset_name: str) -> set[str]:
        metadata = self._get_pose_label_metadata(dataset_name)
        if metadata and "annotated_images" in metadata:
            return set(metadata["annotated_images"])
        return set()

    def set_pose_labels(
        self,
        dataset_name: str,
        pose_dataset: PoseDataset,
        warn_if_exists: constants.WarnIfExistsFunctionType = None,
    ) -> None:
        if dataset_name in self._pose_labels:
            if warn_if_exists:
                if not warn_if_exists(
                    f"Pose labels for dataset '{dataset_name}' already exist. Overwrite?"
                ):
                    logging.info(f"Set pose labels cancelled by user for dataset '{dataset_name}'.")
                    return
        self._pose_labels[dataset_name] = pose_dataset

    def import_pose_labels(
        self,
        dataset_name: str,
        project_name: str,
        pose_labels_path: Path,
        backend: constants.BackendEnum = constants.BackendEnum.WHISKER,
        warn_if_exists: constants.WarnIfExistsFunctionType = None,
    ) -> str:
        # Uses the helper's dataset_op property to check/create the dataset
        if dataset_name not in self._datasets.keys():
            self._create_dataset_for_import(dataset_name, pose_labels_path, warn_if_exists)

        pose_labels_dir = self._base_dir / dataset_name
        if not self._delete_if_existing(
            pose_labels_dir, "pose labels set", warn_if_exists=warn_if_exists
        ):
            return "Import cancelled by user."

        project = self._projects.get(project_name)
        if not project:
            raise ValueError(f"Project '{project_name}' not found in workspace.")

        report = ""
        project_bodyparts = project.body_parts
        project_identities = project.identities

        if backend == constants.BackendEnum.MARS:
            import whisker.third_party.mars_api as mars_api
            pose_labels, report = mars_api.import_pose_labels(
                pose_labels_path, project_bodyparts, project_identities
            )
        elif backend == constants.BackendEnum.DLC:
            import whisker.third_party.dlc_api.adapter as dlc_api
            pose_labels, report = dlc_api.import_pose_labels(
                pose_labels_path, project_bodyparts, project_identities
            )
        else:
            raise ValueError(f"Unsupported backend for pose import: {backend}")

        self._pose_labels[dataset_name] = pose_labels
        self.save_pose_labels(dataset_name)
        logging.info(
            f"Imported {len(pose_labels.frame_indices)} pose labels "
            f"for dataset {dataset_name}"
        )
        return report

    def remove(self, dataset_name: str) -> None:
        dataset = self._pose_labels.get(dataset_name)
        if not dataset:
            logging.warning(f"Called to remove PoseDataset {dataset} but no such PoseDataset exists.")
            return
        
        del self._pose_labels[dataset_name]
        shutil.rmtree(self.base_dir / dataset_name)
        logging.info(f"Removed PoseDataset {dataset_name} from the workspace")
    
    def fill_labels_for_subsample_dataset(self, target_dataset_name: str, source_dataset_names: list[str]):
        target_dataset = self._datasets.get(target_dataset_name)
        if not target_dataset:
            raise ValueError(f"Target dataset {target_dataset_name} does not exist.")

        # Load all source pose datasets into memory/cache
        sources: dict[str, PoseDataset] = {}
        for src_name in source_dataset_names:
            ds = self.get_pose_dataset(src_name, raise_if_missing=True)
            if ds:
                sources[src_name] = ds

        if not sources:
            raise ValueError("No valid source datasets provided with existing labels.")

        # Prepare to collect matching rows
        matched_dfs = []
        filled_keys = set()
        
        # We assume all sources share the same body_parts/individuals structure for a valid merge
        # Taking metadata from the first source
        first_source = next(iter(sources.values()))
        body_parts = first_source.body_parts
        individuals = first_source.individuals

        for frame_key in target_dataset.files:
            match_found = None
            
            for src_name, src_pose_ds in sources.items():
                if frame_key in src_pose_ds.frame_indices:
                    if match_found:
                        raise ValueError(
                            f"Ambiguity error: Frame '{frame_key}' found in multiple sources: "
                            f"'{match_found}' and '{src_name}'."
                        )
                    
                    # Extract the specific frame data
                    frame_data = src_pose_ds.keypoint_data.xs(frame_key, level='frame_index', drop_level=False)
                    matched_dfs.append(frame_data)
                    match_found = src_name
            
            if not match_found:
                raise ValueError(f"Safety check failed: No source label found for frame '{frame_key}'.")
            
            filled_keys.add(frame_key)

        # Final safety check on count
        if len(filled_keys) != len(target_dataset.files):
            raise ValueError("Target dataset was not fully populated from sources.")

        # Combine and create the new PoseDataset
        combined_df = pd.concat(matched_dfs).sort_index()
        new_pose_dataset = PoseDataset(
            keypoint_data=combined_df,
            body_parts=body_parts,
            individuals=individuals
        )

        self.set_pose_labels(target_dataset_name, new_pose_dataset)
        self.save_pose_labels(target_dataset_name)
        logging.info(f"Successfully filled {len(filled_keys)} labels for '{target_dataset_name}'.")

    def _create_dataset_for_import(self, dataset_name: str, pose_labels_path: Path, warn_if_exists):
        """Helper to create a dataset entry if one doesn't exist during import."""
        logging.info(
            f"Dataset '{dataset_name}' not found. Creating a new one from '{pose_labels_path}'."
        )

        files: list[Path] = []
        for ext in IMAGE_FILE_EXTENSIONS:
            files.extend(pose_labels_path.rglob(f"*{ext}"))

        if not files:
            raise FileNotFoundError("Import failed: No image files found in the specified directory.")

        relative_files: list[str] = [
            str(f.relative_to(pose_labels_path)).replace("\\", "/")
            for f in sorted(files)
        ]

        have_subdirs = any(os.path.dirname(f) != "" for f in relative_files)
        dataset_type = (
            DatasetType.FRAME_SUBSET
            if have_subdirs
            else DatasetType.IMAGE_COLLECTION
        )

        new_dataset = Dataset(
            name=dataset_name,
            base_data_path=str(pose_labels_path.resolve()),
            type=dataset_type,
            files=relative_files,
        )
        self._datasets_op.add_dataset(dataset_name, new_dataset, warn_if_exists=warn_if_exists)
        logging.info(
            f"Created new {dataset_type.value} dataset '{dataset_name}' for the new pose labels."
        )


    def get_labeled_image_count(
        self,
        dataset_name: str,
        video_subset: str = "",
    ) -> int:
        pose_labels = self.get_pose_dataset(dataset_name, raise_if_missing=False)
        if not pose_labels:
            logging.info(f"No pose labels for dataset {dataset_name}")
            return 0

        if video_subset:
            dataset = self._datasets.get(dataset_name)
            if dataset is None:
                logging.error(f"Have pose labels for dataset {dataset_name} but not a dataset")
                return 0

            if dataset.type != DatasetType.FRAME_SUBSET:
                raise ValueError(
                    f"Dataset {dataset_name} is not a video collection. Is {dataset.type}"
                )
            return len(
                [
                    frame_index
                    for frame_index in pose_labels.frame_indices
                    if video_subset == os.path.dirname(frame_index)
                ]
            )
        else:
            return len(pose_labels.frame_indices)

    def get_pose_dataset(
        self, dataset_name: str, raise_if_missing: bool = True
    ) -> Optional[PoseDataset]:
        if dataset_name not in self._pose_labels:
            if raise_if_missing:
                raise ValueError(f"Dataset '{dataset_name}' does not have pose labels.")
            return None

        # Core lazy-loading logic
        if self._pose_labels[dataset_name] is None:
            logging.info(f"[PoseLabelOperations] Loading pose labels for dataset '{dataset_name}'...")
            labels_path = self._base_dir / dataset_name / "labels.h5"
            if not labels_path.exists():
                if raise_if_missing:
                    raise FileNotFoundError(f"Pose label file not found: {labels_path}")
                return None

            try:
                loaded_labels = PoseDataset.from_file(labels_path)
                self._pose_labels[dataset_name] = loaded_labels
            except Exception as e:
                if raise_if_missing:
                    logging.error(
                        f"Failed to load pose labels from {labels_path}: {e}",
                        exc_info=True,
                    )
                    raise
                return None

        return self._pose_labels[dataset_name]

    def get_pose_labeled_image_keys(self, dataset_name: str) -> set[str]:
        pose_labels_for_ds = self.get_pose_dataset(dataset_name, raise_if_missing=False)
        return (
            set(pose_labels_for_ds.frame_indices) if pose_labels_for_ds else set()
        )

    def get_pose_dataset_path(self, dataset_name: str) -> Path:
        if dataset_name not in self._pose_labels:
            raise ValueError(f"Dataset {dataset_name} does not have pose labels.")
        return self._base_dir / dataset_name / "labels.h5"

    def write_poses_file(
        self,
        dataset_name,
        pose_labels: PoseDataset,
        pose_labels_path: Path,
    ):
        dataset_pose_labels_dir = pose_labels_path.parent
        os.makedirs(dataset_pose_labels_dir, exist_ok=True)

        pose_labels.to_file(pose_labels_path)
        logging.info(f"Saved pose labels for dataset {dataset_name}")

        metadata_path = dataset_pose_labels_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(
                {
                    "dataset_name": dataset_name,
                    "body_parts": pose_labels.body_parts,
                    "individuals": pose_labels.individuals,
                    "frame_indices": list(pose_labels.frame_indices),
                },
                f,
                indent=4,
            )
            logging.info(f"Saved metadata for dataset {dataset_name}")

    def save_pose_labels(self, dataset_name: str) -> None:
        pose_labels = self.get_pose_dataset(dataset_name)
        if not pose_labels:
            raise ValueError(f"Dataset {dataset_name} does not have pose labels.")

        self.write_poses_file(
            dataset_name,
            pose_labels,
            self._base_dir / dataset_name / "labels.h5",
        )
        self._pose_label_metadata_cache.pop(dataset_name, None)
