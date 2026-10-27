import json
import logging
import os
from pathlib import Path
from typing import Optional

from whisker.core.base_workflow_operation_helper import BaseWorkflowOperationsHelper
from whisker.core.study.dataset_operations import DatasetOperations
from whisker.core.study.project_operations import ProjectOperations


class RoiLabelOperations(BaseWorkflowOperationsHelper):
    """Read-only access to a dataset's ROI arena config: ``template.json`` (the dataset default), ``rois.json``
    (per-video overrides, keyed by file name), and ``scale.json``/``video_scales.json`` (pixels-to-mm calibration).
    These files are written by full WHISKER's Labeling tab and only ever read here (brief section 4: "Has, and is
    reusable" lists no write path for ROI labels in the labeler)."""

    def __init__(self, base_dir: Path, projects: ProjectOperations, datasets: DatasetOperations):
        super().__init__(base_dir, projects, datasets)
        self._known_datasets: set[str] = set()

    def scan_labels(self):
        """Records which datasets have an ROI label folder (lazy-load the actual JSON on demand)."""
        self._known_datasets.clear()
        if self._base_dir.exists():
            for name in os.listdir(self._base_dir):
                if not name.startswith(".") and (self._base_dir / name).is_dir():
                    self._known_datasets.add(name)

    def has_labels(self, dataset_name: str) -> bool:
        return dataset_name in self._known_datasets

    def _read_json(self, dataset_name: str, filename: str) -> Optional[dict]:
        path = self._base_dir / dataset_name / filename
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as e:
            logging.error(f"Failed to read ROI label file {path}: {e}")
            return None

    def get_template(self, dataset_name: str) -> dict:
        return self._read_json(dataset_name, "template.json") or {}

    def get_video_rois(self, dataset_name: str) -> dict:
        """``{video file name: config}``, each config the same shape as ``template.json`` plus ``has_overrides``."""
        return self._read_json(dataset_name, "rois.json") or {}

    def get_video_scale(self, dataset_name: str, video_name: str) -> dict:
        """The video's pixels-to-mm calibration: its own entry in ``video_scales.json`` (keyed by file name),
        falling back to the dataset-level ``scale.json``."""
        video_scales = self._read_json(dataset_name, "video_scales.json") or {}
        per_video = video_scales.get(Path(video_name).name)
        if per_video is not None:
            return per_video
        return self._read_json(dataset_name, "scale.json") or {}
