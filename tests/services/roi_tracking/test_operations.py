"""Round-trip tests for the ROI tracking service layer (RoiLabelOperations, RoiPredictionOperations,
RoiCorrectionOperations) via the Workspace, using the shared ``roi_workspace`` fixture from tests/conftest.py."""
import json

from whisker.services.roi_tracking.internal.verify_edits import TrackEdits


def test_roi_template_and_runs_are_readable(roi_workspace):
    ws, _video_path = roi_workspace
    template = ws.get_roi_template("ds1")
    assert [a["arena_name"] for a in template["arenas"]] == ["Arena1", "Arena2"]
    assert ws.list_roi_runs("ds1") == ["run1"]
    assert ws.list_roi_runs("no-such-dataset") == []


def test_load_prediction_round_trips_and_sha256_is_stable(roi_workspace):
    ws, _video_path = roi_workspace
    pred = ws.load_roi_prediction("run1", "ds1", "vid1")
    assert pred["total_frames"] == 5
    assert ws.has_roi_prediction("run1", "ds1", "vid1")
    assert not ws.has_roi_prediction("run1", "ds1", "no-such-video")

    sha_a = ws.roi_predictions.get_prediction_sha256("run1", "ds1", "vid1")
    sha_b = ws.roi_predictions.get_prediction_sha256("run1", "ds1", "vid1")
    assert sha_a == sha_b
    assert ws.roi_predictions.get_prediction_sha256("run1", "ds1", "missing") is None


def test_corrections_save_and_load_round_trip_exactly(roi_workspace):
    ws, _video_path = roi_workspace
    assert not ws.roi_corrections.has_corrections("run1", "ds1", "vid1")
    assert len(ws.load_roi_corrections("run1", "ds1", "vid1")) == 0

    edits = TrackEdits()
    edits.set(0, 1, (2.0, 3.0))
    edits.set(0, 2, None)
    sha = ws.get_roi_prediction_sha256("run1", "ds1", "vid1")
    saved_path = ws.save_roi_corrections("run1", "ds1", "vid1.mp4", "vid1", edits, total_frames=5,
                                          source_run="run1", prediction_sha256=sha)
    assert saved_path.exists()
    assert saved_path.name == "vid1.json"

    assert ws.roi_corrections.has_corrections("run1", "ds1", "vid1")
    loaded = ws.load_roi_corrections("run1", "ds1", "vid1")
    assert loaded.get(0, 1) == (2.0, 3.0)
    assert loaded.has(0, 2) and loaded.get(0, 2) is None
    # the original prediction file is untouched
    assert ws.load_roi_prediction("run1", "ds1", "vid1")["total_frames"] == 5


def test_discard_corrections_removes_the_file(roi_workspace):
    ws, _video_path = roi_workspace
    edits = TrackEdits()
    edits.set(0, 1, (2.0, 3.0))
    ws.save_roi_corrections("run1", "ds1", "vid1.mp4", "vid1", edits, total_frames=5, source_run="run1",
                             prediction_sha256=None)
    assert ws.roi_corrections.has_corrections("run1", "ds1", "vid1")
    ws.discard_roi_corrections("run1", "ds1", "vid1")
    assert not ws.roi_corrections.has_corrections("run1", "ds1", "vid1")
    assert len(ws.load_roi_corrections("run1", "ds1", "vid1")) == 0


def test_corrections_labeled_video_keys_reflects_saved_files(roi_workspace):
    ws, _video_path = roi_workspace
    assert ws.get_roi_corrections_labeled_video_keys("ds1") == set()
    edits = TrackEdits()
    edits.set(0, 1, (2.0, 3.0))
    ws.save_roi_corrections("run1", "ds1", "vid1.mp4", "vid1", edits, total_frames=5, source_run="run1",
                             prediction_sha256=None)
    assert ws.get_roi_corrections_labeled_video_keys("ds1") == {"vid1"}


def test_video_scale_falls_back_from_per_video_to_dataset_default(roi_workspace):
    ws, _video_path = roi_workspace
    labels_dir = ws.base_dir / "workflows" / "roi_tracking" / "labels" / "ds1"
    (labels_dir / "scale.json").write_text(json.dumps({"pixels_to_mm": 0.5}))
    assert ws.get_roi_video_scale("ds1", "vid1.mp4") == {"pixels_to_mm": 0.5}

    (labels_dir / "video_scales.json").write_text(json.dumps({"vid1.mp4": {"pixels_to_mm": 0.25}}))
    assert ws.get_roi_video_scale("ds1", "vid1.mp4") == {"pixels_to_mm": 0.25}
