import json
import os
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# PyQt6 must finish loading its own Qt6Gui.dll before cv2 is imported: cv2 ships a bundled Qt that otherwise wins
# the DLL search on Windows and breaks any later `from PyQt6.QtGui import ...` with "DLL load failed" (observed
# specifically on the first `PyQt6.QtGui` import of the process when cv2 had already been imported first).
import PyQt6.QtWidgets  # noqa: F401

import cv2
import numpy as np
import pytest


@pytest.fixture(scope="session")
def qapp():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def tmp_workspace_dir():
    path = Path(tempfile.mkdtemp())
    yield path
    shutil.rmtree(path, ignore_errors=True)


def _write_tiny_video(path: Path, n_frames: int = 5, size=(64, 48), fps: float = 10.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    for _ in range(n_frames):
        writer.write(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    writer.release()


@pytest.fixture
def roi_workspace(qapp, tmp_workspace_dir):
    """A workspace with one VIDEO_COLLECTION dataset ('ds1', one video 'vid1.mp4'), two arenas (Arena1/Arena2)
    spanning the video's left/right halves, and one ROI prediction run ('run1') with a gap in Arena2's frame 2.
    Returns ``(workspace, video_path)``."""
    from whisker.core.study.dataset import Dataset, DatasetType
    from whisker.core.workspace import Workspace

    ws = Workspace.create(tmp_workspace_dir)

    data_dir = tmp_workspace_dir / "data"
    video_path = data_dir / "vid1.mp4"
    _write_tiny_video(video_path)

    ds = Dataset(name="ds1", base_data_path=str(data_dir), type=DatasetType.VIDEO_COLLECTION, files=["vid1.mp4"])
    ws.add_dataset("ds1", ds)

    labels_dir = tmp_workspace_dir / "workflows" / "roi_tracking" / "labels" / "ds1"
    labels_dir.mkdir(parents=True, exist_ok=True)
    (labels_dir / "template.json").write_text(json.dumps({"arenas": [
        {"arena_name": "Arena1", "coordinates": {"x": 0, "y": 0, "width": 32, "height": 48}},
        {"arena_name": "Arena2", "coordinates": {"x": 32, "y": 0, "width": 32, "height": 48}},
    ]}))

    pred_dir = tmp_workspace_dir / "workflows" / "roi_tracking" / "predictions" / "run1" / "ds1"
    pred_dir.mkdir(parents=True, exist_ok=True)
    (pred_dir / "vid1.json").write_text(json.dumps({"trajectory": [
        [[5, 5], [40, 5]], [[5, 6], [40, 6]], [[5, 7]], [[5, 8], [40, 8]], [[5, 9], [40, 9]],
    ], "total_frames": 5}))

    ws.scan_labels()
    ws.scan_predictions()
    return ws, video_path


@pytest.fixture
def long_gap_workspace(qapp, tmp_workspace_dir):
    """A workspace like ``roi_workspace``, but with a single arena ('Arena1', the whole frame) and a run ('run1')
    whose trajectory has an 11-frame gap (frames 2-12 of 15) -- long enough to be keyframe-filled (keyframes at
    2, 7, 12). Returns ``(workspace, video_path)``."""
    from whisker.core.study.dataset import Dataset, DatasetType
    from whisker.core.workspace import Workspace

    ws = Workspace.create(tmp_workspace_dir)

    data_dir = tmp_workspace_dir / "data"
    video_path = data_dir / "vid1.mp4"
    _write_tiny_video(video_path, n_frames=15)

    ds = Dataset(name="ds1", base_data_path=str(data_dir), type=DatasetType.VIDEO_COLLECTION, files=["vid1.mp4"])
    ws.add_dataset("ds1", ds)

    labels_dir = tmp_workspace_dir / "workflows" / "roi_tracking" / "labels" / "ds1"
    labels_dir.mkdir(parents=True, exist_ok=True)
    (labels_dir / "template.json").write_text(json.dumps({"arenas": [
        {"arena_name": "Arena1", "coordinates": {"x": 0, "y": 0, "width": 64, "height": 48}},
    ]}))

    trajectory = [[[5.0, 5.0]]] * 15
    for frame in range(2, 13):
        trajectory[frame] = []
    pred_dir = tmp_workspace_dir / "workflows" / "roi_tracking" / "predictions" / "run1" / "ds1"
    pred_dir.mkdir(parents=True, exist_ok=True)
    (pred_dir / "vid1.json").write_text(json.dumps({"trajectory": trajectory, "total_frames": 15}))

    ws.scan_labels()
    ws.scan_predictions()
    return ws, video_path
