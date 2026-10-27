import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional

from whisker.core.base_workflow_operation_helper import BaseWorkflowOperationsHelper
from whisker.core.study.dataset_operations import DatasetOperations
from whisker.core.study.project_operations import ProjectOperations


class RoiPredictionOperations(BaseWorkflowOperationsHelper):
    """Read-only access to ROI tracking prediction runs: ``predictions/<run>/<dataset>/<video_stem>.json``.

    Unlike full WHISKER, the labeler never has ``run_info.json`` (it sits at ``predictions/<run>/`` and is not
    included when full WHISKER exports a Whisker bundle for a dataset — see the port brief, section 5.2), so there is
    no run-settings lookup here: the Verify widget only needs the trajectory itself. The collaborator's edits are
    never written back into this tree -- see ``RoiCorrectionOperations`` for the small corrections file they produce
    instead; applying it to a full prediction run, as ``<run>_uservalidated``, is a full-WHISKER-side job.
    """

    def __init__(self, base_dir: Path, projects: ProjectOperations, datasets: DatasetOperations):
        super().__init__(base_dir, projects, datasets)
        # {run_name: {dataset_name: {video_stem}}} -- which predictions exist, not their (lazily loaded) content.
        self._known: dict[str, dict[str, set[str]]] = {}

    def scan_predictions(self):
        """Scans for existing ROI prediction runs (lazy load the actual JSON on demand)."""
        self._known.clear()
        if not self._base_dir.exists():
            return
        for run_name in os.listdir(self._base_dir):
            run_dir = self._base_dir / run_name
            if not run_dir.is_dir() or run_name.startswith("."):
                continue
            self._known[run_name] = {}
            for dataset_name in os.listdir(run_dir):
                dataset_dir = run_dir / dataset_name
                if not dataset_dir.is_dir() or dataset_name.startswith("."):
                    continue
                stems = {f[:-len(".json")] for f in os.listdir(dataset_dir) if f.endswith(".json")}
                self._known[run_name][dataset_name] = stems

    def list_runs(self, dataset_name: str) -> list[str]:
        """Sorted names of the prediction runs that have at least one prediction for this dataset."""
        return sorted(run for run, datasets in self._known.items() if datasets.get(dataset_name))

    def has_prediction(self, run_name: str, dataset_name: str, video_stem: str) -> bool:
        return video_stem in self._known.get(run_name, {}).get(dataset_name, set())

    def get_prediction_path(self, run_name: str, dataset_name: str, video_stem: str) -> Path:
        return self._base_dir / run_name / dataset_name / f"{video_stem}.json"

    def load_prediction(self, run_name: str, dataset_name: str, video_stem: str) -> Optional[dict]:
        path = self.get_prediction_path(run_name, dataset_name, video_stem)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as e:
            logging.error(f"Failed to load ROI prediction {path}: {e}")
            return None

    def get_prediction_sha256(self, run_name: str, dataset_name: str, video_stem: str) -> Optional[str]:
        """Hash of the prediction file's exact bytes, so a corrections file can refuse to be applied against a
        prediction that has since changed (brief section 5.3)."""
        path = self.get_prediction_path(run_name, dataset_name, video_stem)
        if not path.exists():
            return None
        return hashlib.sha256(path.read_bytes()).hexdigest()
