import json
import zipfile
from pathlib import Path

import pytest

from whisker.core import whisker_bundle_import as wbi
from whisker.core.workspace import Workspace


def _write_bundle(path: Path, *, datasets=None, prediction_runs=None, projects=None, models=None, files=None):
    """Builds a minimal, valid Whisker bundle zip for testing. ``files`` is {arcname: bytes-or-str} for the
    per-dataset/workflow payload; the manifest is derived from the other arguments."""
    manifest = {
        "format": wbi.BUNDLE_FORMAT,
        "format_version": 1,
        "whisker_version": "test",
        "created_at": "2026-10-25T00:00:00+00:00",
        "source_workspace": "full_whisker_ws",
        "datasets": datasets or [],
        "prediction_runs": prediction_runs or [],
        "projects": projects or [],
        "models": models or [],
        "file_count": len(files or {}),
        "total_bytes": 1,
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("whisker_bundle.json", json.dumps(manifest))
        for arcname, content in (files or {}).items():
            zf.writestr(arcname, content if isinstance(content, bytes) else json.dumps(content))


def _one_dataset_bundle(path: Path, *, dataset_name="ds1", include_pose=True, include_project=True):
    ds_manifest = {"name": dataset_name, "type": "VIDEO_COLLECTION", "base_data_path": "/original/path",
                   "files": ["vid1.mp4"]}
    label_workflows = ["roi_tracking"] + (["pose_estimation"] if include_pose else [])
    files = {
        f"datasets/{dataset_name}/manifest.json": ds_manifest,
        f"datasets/{dataset_name}/media/vid1.mp4": b"FAKEVIDEOBYTES",
        f"workflows/roi_tracking/labels/{dataset_name}/template.json": {"arenas": [
            {"arena_name": "Arena1", "coordinates": {"x": 0, "y": 0, "width": 10, "height": 10}}]},
        f"workflows/roi_tracking/predictions/run1/{dataset_name}/vid1.json": {
            "trajectory": [[[1, 1]]], "total_frames": 1},
    }
    prediction_runs = [{"workflow": "roi_tracking", "run_name": "run1", "datasets": [dataset_name]}]
    if include_pose:
        files[f"workflows/pose_estimation/labels/{dataset_name}/labels.h5"] = b"FAKEH5BYTES"
        prediction_runs.append({"workflow": "pose_estimation", "run_name": "poserun", "datasets": [dataset_name]})
    projects = []
    if include_project:
        files["projects/proj1.json"] = {"name": "proj1", "body_parts": [], "skeleton": [], "identities": [],
                                         "behaviors": []}
        projects = ["proj1"]

    _write_bundle(
        path,
        datasets=[{"name": dataset_name, "type": "VIDEO_COLLECTION", "file_count": 1, "media_layout": "workspace",
                   "original_base_data_path": "", "media_bytes": 5, "missing_media": [],
                   "label_workflows": label_workflows, "multi_arena": False, "media_included": True}],
        prediction_runs=prediction_runs,
        projects=projects,
        models=[{"workflow": "behavior_classification", "model_name": "m1"}] if include_pose else [],
        files=files,
    )


@pytest.fixture
def workspace(tmp_workspace_dir):
    ws_dir = tmp_workspace_dir / "ws"
    ws_dir.mkdir()
    return Workspace.create(ws_dir)


def test_refuses_a_non_zip_file_without_touching_the_workspace(tmp_workspace_dir, workspace):
    bad = tmp_workspace_dir / "bad.zip"
    bad.write_bytes(b"not a zip")
    with pytest.raises(wbi.BundleError):
        wbi.read_bundle_info(bad)
    with pytest.raises(wbi.BundleError):
        wbi.import_bundle(workspace, bad, wbi.ImportOptions(dataset_names=["x"]))
    assert list((workspace.base_dir / "datasets").iterdir()) == []


def test_refuses_a_zip_with_no_bundle_manifest(tmp_workspace_dir, workspace):
    path = tmp_workspace_dir / "plain.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("hello.txt", "hi")
    with pytest.raises(wbi.BundleError):
        wbi.read_bundle_info(path)


def test_imports_dataset_roi_labels_predictions_and_project(tmp_workspace_dir, workspace):
    path = tmp_workspace_dir / "bundle.zip"
    _one_dataset_bundle(path)

    report = wbi.import_bundle(workspace, path, wbi.ImportOptions(dataset_names=["ds1"]))

    assert report.imported_datasets == ["ds1"]
    assert report.label_sets == ["ds1"]
    assert report.prediction_sets == [("run1", "ds1")]
    assert report.imported_projects == ["proj1"]
    assert "pose estimation labels" in " ".join(report.ignored_content)
    assert "pose estimation predictions" in " ".join(report.ignored_content)
    assert "behavior classification models" in " ".join(report.ignored_content)

    wbi.rescan_workspace(workspace)
    assert "ds1" in workspace.datasets.keys()
    assert (workspace.base_dir / "datasets" / "ds1" / "data" / "vid1.mp4").exists()
    assert workspace.get_roi_template("ds1")["arenas"][0]["arena_name"] == "Arena1"
    assert workspace.list_roi_runs("ds1") == ["run1"]
    assert (workspace.base_dir / "projects" / "proj1.json").exists()
    # pose content was never extracted at all
    assert not (workspace.base_dir / "workflows" / "pose_estimation" / "labels" / "ds1").exists()


def test_imports_without_a_project_when_the_bundle_has_none(tmp_workspace_dir, workspace):
    path = tmp_workspace_dir / "bundle.zip"
    _one_dataset_bundle(path, include_project=False)
    report = wbi.import_bundle(workspace, path, wbi.ImportOptions(dataset_names=["ds1"]))
    assert report.imported_datasets == ["ds1"]
    assert report.imported_projects == []


def test_an_existing_dataset_is_skipped_unless_overwrite_is_requested(tmp_workspace_dir, workspace):
    path = tmp_workspace_dir / "bundle.zip"
    _one_dataset_bundle(path)

    wbi.import_bundle(workspace, path, wbi.ImportOptions(dataset_names=["ds1"]))
    wbi.rescan_workspace(workspace)

    report = wbi.import_bundle(workspace, path, wbi.ImportOptions(dataset_names=["ds1"]))
    assert report.imported_datasets == []
    assert report.skipped_datasets == {"ds1": "already exists in this workspace"}

    report2 = wbi.import_bundle(
        workspace, path, wbi.ImportOptions(dataset_names=["ds1"], overwrite_datasets={"ds1"}))
    assert report2.imported_datasets == ["ds1"]
    assert report2.replaced_datasets == ["ds1"]


def test_an_unknown_dataset_name_is_reported_and_skipped(tmp_workspace_dir, workspace):
    path = tmp_workspace_dir / "bundle.zip"
    _one_dataset_bundle(path)
    report = wbi.import_bundle(workspace, path, wbi.ImportOptions(dataset_names=["no-such-dataset"]))
    assert report.skipped_datasets == {"no-such-dataset": "not in this bundle"}
    assert report.imported_datasets == []


def test_find_import_conflicts_detects_an_existing_dataset_and_project(tmp_workspace_dir, workspace):
    path = tmp_workspace_dir / "bundle.zip"
    _one_dataset_bundle(path)
    wbi.import_bundle(workspace, path, wbi.ImportOptions(dataset_names=["ds1"]))
    wbi.rescan_workspace(workspace)

    info = wbi.read_bundle_info(path)
    conflicts = wbi.find_import_conflicts(workspace, info)
    assert conflicts.datasets == ["ds1"]
    assert conflicts.projects_identical == ["proj1"]  # re-imported unchanged -> identical
