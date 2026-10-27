"""The ROI Tracking case added to tree_controller.py's LabelStatusIndicator: "labeled" means "has a saved
correction", reported by full file name (to match _normalize_paths's Path(p).name for VIDEO_COLLECTION datasets)
even though corrections are stored on disk keyed by video stem."""
from whisker.core.workflows.workflow_enum import Workflow
from whisker.gui.widgets.data_explorer.tree_controller import LabelStatusIndicator
from whisker.services.roi_tracking.internal.verify_edits import TrackEdits


def test_roi_labeled_keys_are_full_file_names_not_stems(roi_workspace):
    ws, _video_path = roi_workspace
    dataset = ws.datasets.get("ds1")
    indicator = LabelStatusIndicator()

    # No saved correction yet.
    indicator.prepare(ws, dataset, Workflow.ROI_TRACKING)
    icon, fully_labeled = indicator.get_indicator_state(["vid1.mp4"])
    assert fully_labeled is False

    edits = TrackEdits()
    edits.set(0, 1, (2.0, 3.0))
    ws.save_roi_corrections("run1", "ds1", "vid1.mp4", "vid1", edits, total_frames=5, source_run="run1",
                             prediction_sha256=None)

    indicator.prepare(ws, dataset, Workflow.ROI_TRACKING)
    icon, fully_labeled = indicator.get_indicator_state(["vid1.mp4"])
    assert fully_labeled is True


def test_roi_is_never_not_applicable_for_a_video_collection(roi_workspace):
    ws, _video_path = roi_workspace
    dataset = ws.datasets.get("ds1")
    indicator = LabelStatusIndicator()
    indicator.prepare(ws, dataset, Workflow.ROI_TRACKING)
    assert indicator._is_na is False
