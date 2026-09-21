"""Builders for realistic test workspaces (real Workspace objects, real HDF5 label
files, real exported bundles) so import/export tests exercise the actual code paths."""
from __future__ import annotations

import logging
import os
import sys
import tempfile
import traceback
import unittest
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from whisker.core.study.dataset import Dataset, DatasetType
from whisker.core.workspace import Workspace
from whisker.services.behavior_classification.public.data_structures import BehaviorDataset
from whisker.services.pose_estimation.public.data_structures import PoseDataset

BODY_PARTS = ["nose", "tail_base"]
INDIVIDUALS = ["mouse1"]
BEHAVIORS = ["groom", "rear"]


def frame_names(n: int, subdir: str = "") -> List[str]:
    prefix = f"{subdir}/" if subdir else ""
    return [f"{prefix}frame_{i:04d}.png" for i in range(n)]


def make_pose_dataset(keys: Iterable[str], body_parts=None, individuals=None, x0: float = 10.0) -> PoseDataset:
    body_parts = list(body_parts or BODY_PARTS)
    individuals = list(individuals or INDIVIDUALS)
    rows, index = [], []
    for key in keys:
        for ind in individuals:
            for j, bp in enumerate(body_parts):
                index.append((key, ind, bp))
                rows.append({"x": x0 + j, "y": x0 + 2 * j, "c": 1.0})
    idx = pd.MultiIndex.from_tuples(index, names=["frame_index", "individual_id", "body_part"])
    df = pd.DataFrame(rows, index=idx).astype("float32")
    return PoseDataset(keypoint_data=df, body_parts=body_parts, individuals=individuals)


def make_behavior_dataset(video_keys: Iterable[str], behaviors=None, start: int = 0) -> BehaviorDataset:
    behaviors = list(behaviors or BEHAVIORS)
    rows = [
        {"video_key": v, "behavior": behaviors[0], "start_frame": start, "end_frame": start + 9, "p": float("nan")}
        for v in video_keys
    ]
    bouts = pd.DataFrame(rows).astype(
        {"video_key": "str", "behavior": "str", "start_frame": "int64", "end_frame": "int64", "p": "float32"}
    )
    return BehaviorDataset(behaviors=behaviors, bouts=bouts)


class WorkspaceCase(unittest.TestCase):
    """Base class: temp dir + helpers to populate workspaces."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        # Cleanups run last-in-first-out, so this runs before the temp dir is deleted.
        # Workspace() attaches a session-log FileHandler to the root logger; on Windows
        # that open file blocks deleting the directory (and would leak between tests).
        self.addCleanup(self._release_log_handlers)
        # The app logs every scan at INFO; keep test output readable.
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, logging.NOTSET)

    def fail_on_slot_exceptions(self):
        """PyQt6 aborts the whole process on an exception inside a slot, which looks like a test run
        that just stops. Route them to sys.excepthook instead and fail the test with the traceback."""
        errors = []
        previous = sys.excepthook
        sys.excepthook = lambda t, v, tb: errors.append("".join(traceback.format_exception(t, v, tb)))

        def check():
            sys.excepthook = previous
            if errors:
                self.fail("Exception raised inside a Qt slot:\n" + errors[0])
        self.addCleanup(check)

    def _release_log_handlers(self):
        root = logging.getLogger()
        for h in root.handlers[:]:
            if isinstance(h, logging.FileHandler) and str(self.tmp) in h.baseFilename:
                root.removeHandler(h)
                h.close()

    # -- workspaces ------------------------------------------------------
    def new_workspace(self, name: str = "ws") -> Workspace:
        path = self.tmp / name
        path.mkdir(parents=True, exist_ok=True)
        return Workspace(path)

    def add_project(self, ws: Workspace, name="proj", body_parts=None, identities=None, behaviors=None):
        ws.create_project(
            name,
            body_parts=list(body_parts or BODY_PARTS),
            identities=list(identities or INDIVIDUALS),
            behaviors=list(behaviors or BEHAVIORS),
        )
        ws.scan_projects()
        return ws.projects.get(name)

    def add_dataset(
        self,
        ws: Workspace,
        name: str,
        files: List[str],
        dtype: DatasetType = DatasetType.IMAGE_COLLECTION,
        media_root: Optional[Path] = None,
    ) -> Dataset:
        """Write real (tiny) media files and register a dataset that points at them."""
        media_root = Path(media_root or self.tmp / "media" / name)
        for rel in files:
            f = media_root / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(b"x" * 16)
        ds = Dataset(name=name, type=dtype, base_data_path=str(media_root), files=list(files))
        ws.add_dataset(name, ds)
        return ws.datasets.get(name)

    def add_pose_labels(self, ws: Workspace, dataset: str, keys: Iterable[str], **kw) -> PoseDataset:
        pose = make_pose_dataset(keys, **kw)
        ws.pose_labels.write_poses_file(dataset, pose, ws.pose_labels.base_dir / dataset / "labels.h5")
        ws.scan_labels()
        return pose

    def add_behavior_labels(self, ws: Workspace, dataset: str, video_keys: Iterable[str], **kw):
        beh = make_behavior_dataset(video_keys, **kw)
        beh.to_file(ws.behavior_labels.base_dir / dataset / "labels.h5")
        ws.scan_labels()
        return beh
