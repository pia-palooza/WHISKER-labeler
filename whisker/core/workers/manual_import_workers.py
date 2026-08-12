"""Background job for importing a dataset from individually-picked component
files/folders. See :mod:`whisker.core.manual_import`.
"""

from pathlib import Path
from typing import Optional

from whisker.base.job import BaseJob
from whisker.core import manual_import
from whisker.core.study.dataset import Dataset
from whisker.core.study.project import Project


class ImportComponentsJob(BaseJob):
    def __init__(
        self,
        workspace,
        dataset_name: str,
        project: Project,
        project_source_path: Path,
        dataset: Dataset,
        media_dir: Path,
        pose_labels_path: Optional[Path] = None,
        pose_metadata_path: Optional[Path] = None,
        behavior_labels_path: Optional[Path] = None,
        overwrite: bool = False,
    ):
        super().__init__()
        self.workspace = workspace
        self.dataset_name = dataset_name
        self.project = project
        self.project_source_path = Path(project_source_path)
        self.dataset = dataset
        self.media_dir = Path(media_dir)
        self.pose_labels_path = Path(pose_labels_path) if pose_labels_path else None
        self.pose_metadata_path = Path(pose_metadata_path) if pose_metadata_path else None
        self.behavior_labels_path = Path(behavior_labels_path) if behavior_labels_path else None
        self.overwrite = overwrite

    def run(self) -> dict:
        return manual_import.import_dataset_from_components(
            self.workspace,
            self.dataset_name,
            self.project,
            self.project_source_path,
            self.dataset,
            self.media_dir,
            pose_labels_path=self.pose_labels_path,
            pose_metadata_path=self.pose_metadata_path,
            behavior_labels_path=self.behavior_labels_path,
            overwrite=self.overwrite,
            progress_cb=self.report_progress,
            cancel_cb=lambda: self.is_cancelled,
        )
