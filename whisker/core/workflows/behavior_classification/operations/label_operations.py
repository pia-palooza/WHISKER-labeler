import logging
import os
from pathlib import Path
from typing import Optional

import whisker.core.constants as constants
from whisker.core.base_workflow_operation_helper import BaseWorkflowOperationsHelper
from whisker.core.dataset_operations import DatasetOperations
from whisker.core.project_operations import ProjectOperations

from ..data_structures import BehaviorDataset


class BehaviorLabelOperations(BaseWorkflowOperationsHelper):
    """
    Manages hand-annotated behavior label files for video datasets.

    One ``labels.h5`` file is stored per dataset under ``<base_dir>/<name>/``,
    mirroring the layout used by :class:`PoseLabelOperations`.
    """

    def __init__(
        self,
        base_dir: Path,
        projects: ProjectOperations,
        datasets: DatasetOperations,
    ):
        super().__init__(base_dir, projects, datasets)
        self._projects: ProjectOperations = projects
        self._datasets: DatasetOperations = datasets
        self._behavior_labels: dict[str, Optional[BehaviorDataset]] = {}

    def scan_labels(self):
        """Scans for existing behavior label files (lazy load)."""
        self._behavior_labels.clear()
        if self._base_dir.exists():
            discovered = []
            for name in os.listdir(self._base_dir):
                if name.startswith("."):
                    continue
                if (self._base_dir / name / "labels.h5").exists():
                    discovered.append(name)
                    self._behavior_labels[name] = None  # Marker for lazy load
            logging.info(
                f"[BehaviorLabelOperations] Discovered behavior labels for "
                f"{len(discovered)} datasets: {discovered}"
            )

    def has_behavior_labels(self, name: str) -> bool:
        return name in self._behavior_labels

    def set_behavior_labels(
        self,
        dataset_name: str,
        behavior_dataset: BehaviorDataset,
        warn_if_exists: constants.WarnIfExistsFunctionType = None,
    ) -> None:
        if dataset_name in self._behavior_labels and warn_if_exists:
            if not warn_if_exists(
                f"Behavior labels for dataset '{dataset_name}' already exist. Overwrite?"
            ):
                logging.info(
                    f"Set behavior labels cancelled by user for dataset '{dataset_name}'."
                )
                return
        self._behavior_labels[dataset_name] = behavior_dataset

    def get_behavior_dataset(
        self, dataset_name: str, raise_if_missing: bool = True
    ) -> Optional[BehaviorDataset]:
        if dataset_name not in self._behavior_labels:
            # Not yet discovered: create a fresh, empty in-memory dataset so the
            # user can start annotating immediately.
            if raise_if_missing:
                raise ValueError(
                    f"Dataset '{dataset_name}' does not have behavior labels."
                )
            self._behavior_labels[dataset_name] = BehaviorDataset()
            return self._behavior_labels[dataset_name]

        if self._behavior_labels[dataset_name] is None:
            logging.info(
                f"[BehaviorLabelOperations] Loading behavior labels for '{dataset_name}'..."
            )
            labels_path = self._base_dir / dataset_name / "labels.h5"
            if not labels_path.exists():
                self._behavior_labels[dataset_name] = BehaviorDataset()
            else:
                try:
                    self._behavior_labels[dataset_name] = BehaviorDataset.from_file(
                        labels_path
                    )
                except Exception as e:
                    if raise_if_missing:
                        logging.error(
                            f"Failed to load behavior labels from {labels_path}: {e}",
                            exc_info=True,
                        )
                        raise
                    self._behavior_labels[dataset_name] = BehaviorDataset()

        return self._behavior_labels[dataset_name]

    def get_behavior_labeled_video_keys(self, dataset_name: str) -> set[str]:
        labels_path = self._base_dir / dataset_name / "labels.h5"
        return BehaviorDataset.get_video_keys_from_file(labels_path)

    def get_behavior_labels_path(self, dataset_name: str) -> Path:
        return self._base_dir / dataset_name / "labels.h5"

    def save_behavior_labels(self, dataset_name: str) -> None:
        behavior_labels = self._behavior_labels.get(dataset_name)
        if not behavior_labels:
            raise ValueError(
                f"Dataset '{dataset_name}' does not have behavior labels to save."
            )

        labels_path = self.get_behavior_labels_path(dataset_name)
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        behavior_labels.to_file(labels_path)
        logging.info(f"Saved behavior labels for dataset {dataset_name}")
