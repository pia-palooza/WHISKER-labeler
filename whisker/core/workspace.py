# UPDATE_FILE: whisker/core/workspace.py
#
# A lightweight workspace for the standalone labeler. It mirrors the on-disk
# layout of the full WHISKER workspace (datasets/, projects/, and
# workflows/<wf>/labels/) so the same folders are interchangeable between the
# two tools -- but it only wires the pieces needed for HAND ANNOTATION
# (datasets, projects, pose labels, behavior labels). No models, predictions,
# services, or training machinery are loaded.
import logging
from pathlib import Path

from .dataset_operations import DatasetOperations
from .project_operations import ProjectOperations
from .workflows.pose_estimation.operations.label_operations import PoseLabelOperations
from .workflows.behavior_classification.operations.label_operations import (
    BehaviorLabelOperations,
)


class Workspace:
    """Opens an existing WHISKER workspace folder for label browsing/editing."""

    def __init__(self, base_dir: Path):
        base_dir = Path(base_dir)
        if not base_dir.exists():
            raise ValueError(f"Workspace base directory does not exist: {base_dir}")

        logging.info(f"Loading workspace at {base_dir}")
        self._base_dir = base_dir

        self.datasets = DatasetOperations(base_dir / "datasets")
        self.projects = ProjectOperations(base_dir / "projects")
        self.pose_labels = PoseLabelOperations(
            base_dir / "workflows" / "pose_estimation" / "labels",
            self.projects,
            self.datasets,
        )
        self.behavior_labels = BehaviorLabelOperations(
            base_dir / "workflows" / "behavior_classification" / "labels",
            self.projects,
            self.datasets,
        )

        self.scan()

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def scan(self):
        self.datasets.scan_datasets()
        self.projects.scan_projects()
        self.pose_labels.scan_labels()
        self.behavior_labels.scan_labels()

    @staticmethod
    def looks_like_workspace(base_dir: Path) -> bool:
        """Heuristic: a workspace has a datasets/ and/or projects/ folder."""
        base_dir = Path(base_dir)
        return (base_dir / "datasets").is_dir() or (base_dir / "projects").is_dir()
