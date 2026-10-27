import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from whisker.core.base_workflow_operation_helper import BaseWorkflowOperationsHelper
from whisker.core.study.dataset_operations import DatasetOperations
from whisker.core.study.project_operations import ProjectOperations
from whisker.services.roi_tracking.internal.verify_edits import TrackEdits

SCHEMA = "whisker_roi_corrections_v2"


class RoiCorrectionOperations(BaseWorkflowOperationsHelper):
    """Reads and writes the collaborator's hand corrections: ``corrections/<run>/<dataset>/<video_stem>.json``.

    Records are ``[track, frame, x, y, auto]`` (see ``TrackEdits.to_records``/``from_records``), matching full
    WHISKER's own corrections format exactly, keyed by raw track index -- not arena name -- since the arena order
    for one run/dataset doesn't change between saves. Full WHISKER's apply step reads this file and uses it to
    build ``<run>_uservalidated`` (a copy of the whole run, metrics recomputed). This is the labeler's only ROI
    write path -- it never touches the prediction JSON, and is cheap enough to autosave on every edit without a
    background thread (the labeler never recomputes occupancy/transitions/distance)."""

    def __init__(self, base_dir: Path, projects: ProjectOperations, datasets: DatasetOperations):
        super().__init__(base_dir, projects, datasets)

    def get_corrections_path(self, run_name: str, dataset_name: str, video_stem: str) -> Path:
        return self._base_dir / run_name / dataset_name / f"{video_stem}.json"

    def has_corrections(self, run_name: str, dataset_name: str, video_stem: str) -> bool:
        return self.get_corrections_path(run_name, dataset_name, video_stem).exists()

    def load_raw(self, run_name: str, dataset_name: str, video_stem: str) -> Optional[dict]:
        path = self.get_corrections_path(run_name, dataset_name, video_stem)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as e:
            logging.error(f"Failed to load ROI corrections {path}: {e}")
            return None

    def load_edits(self, run_name: str, dataset_name: str, video_stem: str) -> TrackEdits:
        data = self.load_raw(run_name, dataset_name, video_stem)
        if not data:
            return TrackEdits()
        return TrackEdits.from_records(data.get("edits", []))

    def save_edits(self, run_name: str, dataset_name: str, video_name: str, video_stem: str, edits: TrackEdits,
                    total_frames: int, source_run: str, prediction_sha256: Optional[str],
                    annotator: str = "") -> Path:
        """Writes ``edits`` as the corrections file of ``video_name`` in ``run_name`` (see class docstring)."""
        path = self.get_corrections_path(run_name, dataset_name, video_stem)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": SCHEMA,
            "source_run": source_run,
            "dataset": dataset_name,
            "video": video_name,
            "total_frames": total_frames,
            "prediction_sha256": prediction_sha256,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "annotator": annotator,
            "edits": edits.to_records(),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def discard(self, run_name: str, dataset_name: str, video_stem: str) -> None:
        """Removes a video's corrections file (e.g. once every edit has been undone back to nothing)."""
        path = self.get_corrections_path(run_name, dataset_name, video_stem)
        if path.exists():
            path.unlink()

    def corrections_labeled_video_keys(self, dataset_name: str) -> set:
        """Video stems that have at least one correction, in any run, for this dataset (the data-explorer ✓)."""
        keys: set = set()
        if not self._base_dir.exists():
            return keys
        for run_name in os.listdir(self._base_dir):
            dataset_dir = self._base_dir / run_name / dataset_name
            if not dataset_dir.is_dir():
                continue
            for f in os.listdir(dataset_dir):
                if f.endswith(".json"):
                    keys.add(f[: -len(".json")])
        return keys
