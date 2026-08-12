"""Import a dataset from its individual pieces — a project file, a dataset
info file, a folder of frames/videos, and (optionally) label files — instead
of a single pre-packaged bundle folder.

The old "Import Annotation Bundle..." flow asked the user to pick *one*
folder and tried to figure out whether it was a valid bundle. In practice
that one-folder guess was the single biggest source of confusion: after
unzipping, people would pick the wrong nesting level, or a folder that
wasn't a bundle at all, and get one opaque "not a valid bundle" error with
no indication of which piece was wrong.

This module instead exposes one independent check per component
(:func:`check_project_file`, :func:`check_dataset_file`,
:func:`check_media_folder`, :func:`check_pose_labels_file`,
:func:`check_behavior_labels_file`) so the GUI can tell the user *exactly*
which of the pieces they picked is the problem, as soon as they pick it —
before they ever hit an "Import" button. :func:`import_dataset_from_components`
then performs the actual copy once every required piece has checked out.

Pure filesystem work — no Qt, no in-memory workspace mutation — so it can run
on a background thread like :mod:`whisker.core.bundle` does.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from whisker.core.study.dataset import Dataset, DatasetType
from whisker.core.study.project import Project
from whisker.services.pose_estimation.public.data_structures import PoseDataset
from whisker.services.behavior_classification.public.data_structures import BehaviorDataset

logger = logging.getLogger(__name__)

LABELS_H5_FILENAME = "labels.h5"
POSE_METADATA_FILENAME = "metadata.json"

ProgressCallback = Callable[[str, int], None]


class ManualImportError(Exception):
    """Raised when the import can't proceed even though the individual
    component checks looked fine (e.g. a conflict that appeared between the
    check and the import, or a required piece missing at import time)."""


@dataclass
class ComponentCheck:
    """The result of validating one piece the user picked. ``message`` is
    always meant to be shown to the user verbatim — it should say exactly
    what's wrong (or right), not just pass/fail."""

    ok: bool
    message: str

    @classmethod
    def good(cls, message: str) -> "ComponentCheck":
        return cls(True, message)

    @classmethod
    def bad(cls, message: str) -> "ComponentCheck":
        return cls(False, message)

    @classmethod
    def empty(cls, message: str) -> "ComponentCheck":
        """Nothing picked yet — not an error, just not ready."""
        return cls(False, message)


# ------------------------------------------------------------------ #
# Per-component checks
# ------------------------------------------------------------------ #


def check_project_file(path: Optional[Path]) -> Tuple[ComponentCheck, Optional[Project]]:
    """Validate a project JSON file (the one produced by the labeler's
    project editor, or by a previous "Export..." )."""
    if path is None:
        return ComponentCheck.empty("Select the project file (a .json file)."), None
    path = Path(path)
    if not path.exists() or not path.is_file():
        return ComponentCheck.bad(f"File not found: {path}"), None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        return ComponentCheck.bad(f"Could not open file: {e}"), None
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as e:
        return ComponentCheck.bad(f"This isn't valid JSON ({e}). Is this the right file?"), None
    if not isinstance(raw, dict) or "name" not in raw:
        return (
            ComponentCheck.bad(
                "This JSON file doesn't look like a project (no 'name' field). "
                "Is this the dataset info file instead?"
            ),
            None,
        )
    # A dataset info file also has a top-level "name", so "name" alone isn't
    # enough to tell them apart — check for the fields that are unique to a
    # dataset manifest so we don't silently accept the wrong file here.
    dataset_only_fields = [k for k in ("base_data_path", "files", "type") if k in raw]
    if dataset_only_fields:
        return (
            ComponentCheck.bad(
                "This looks like a dataset info file, not a project file "
                f"(it has {', '.join(repr(f) for f in dataset_only_fields)}). "
                "Use it for the 'Dataset info file' field instead."
            ),
            None,
        )
    try:
        project = Project.from_json(text)
    except Exception as e:
        return ComponentCheck.bad(f"Could not read this as a project file: {e}"), None

    detail = f"Project '{project.name}' — {len(project.body_parts)} body part(s)"
    if project.behaviors:
        detail += f", {len(project.behaviors)} behavior(s)"
    return ComponentCheck.good(detail), project


def check_dataset_file(path: Optional[Path]) -> Tuple[ComponentCheck, Optional[Dataset]]:
    """Validate a dataset info JSON file (a ``manifest.json``, e.g. from
    inside an ``Export...``'d folder's ``dataset/`` subfolder)."""
    if path is None:
        return (
            ComponentCheck.empty(
                "Select the dataset info file (usually named manifest.json)."
            ),
            None,
        )
    path = Path(path)
    if not path.exists() or not path.is_file():
        return ComponentCheck.bad(f"File not found: {path}"), None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        return ComponentCheck.bad(f"Could not open file: {e}"), None
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as e:
        return ComponentCheck.bad(f"This isn't valid JSON ({e}). Is this the right file?"), None
    missing_fields = [k for k in ("name", "type", "base_data_path", "files") if k not in raw]
    if missing_fields:
        return (
            ComponentCheck.bad(
                "This JSON file doesn't look like a dataset info file "
                f"(missing {', '.join(missing_fields)}). Is this the project file instead?"
            ),
            None,
        )
    try:
        dataset = Dataset.from_json(text)
    except Exception as e:
        return ComponentCheck.bad(f"Could not read this as a dataset info file: {e}"), None

    if not dataset.files:
        return (
            ComponentCheck.bad(f"'{dataset.name}' lists 0 files — is this the right file?"),
            dataset,
        )
    detail = f"Dataset '{dataset.name}' — {dataset.type.value}, {len(dataset.files)} file(s)"
    if dataset.is_multi_arena:
        detail += ", multi-arena"
    return ComponentCheck.good(detail), dataset


def media_label_for(dataset: Optional[Dataset]) -> str:
    """What to call the media-folder field, once we know the dataset type."""
    if dataset is None:
        return "Media folder"
    return "Video clips folder" if dataset.type == DatasetType.VIDEO_COLLECTION else "Frames folder"


def check_media_folder(
    path: Optional[Path], dataset: Optional[Dataset]
) -> Tuple[ComponentCheck, List[str]]:
    """Validate the folder holding the actual frame images / video files.

    Checked file-by-file against ``dataset.files`` (from the dataset info
    file) so a wrong-folder pick is reported with concrete missing filenames,
    not a generic failure.
    """
    if path is None:
        return ComponentCheck.empty("Select the folder containing the media files."), []
    path = Path(path)
    if not path.exists() or not path.is_dir():
        return ComponentCheck.bad(f"Folder not found: {path}"), []
    if dataset is None:
        return (
            ComponentCheck.empty(
                "Folder selected — load the dataset info file first so this can be checked."
            ),
            [],
        )
    missing = [rel for rel in dataset.files if not (path / rel).exists()]
    kind = "video" if dataset.type == DatasetType.VIDEO_COLLECTION else "frame"
    total = len(dataset.files)
    if missing:
        sample = ", ".join(missing[:3])
        more = f" (+{len(missing) - 3} more)" if len(missing) > 3 else ""
        return (
            ComponentCheck.bad(
                f"{len(missing)} of {total} {kind}s listed in the dataset info file are not "
                f"in this folder — is this the right folder? e.g. missing {sample}{more}"
            ),
            missing,
        )
    return ComponentCheck.good(f"All {total} {kind}s found in this folder."), missing


def check_pose_labels_file(path: Optional[Path]) -> Tuple[ComponentCheck, Optional[Path]]:
    """Validate an (optional) pose labels HDF5 file. Returns the check plus
    a sibling ``metadata.json`` path if one sits next to the chosen file."""
    if path is None:
        return ComponentCheck.good("No pose labels file — dataset will be imported without pose labels."), None
    path = Path(path)
    if not path.exists() or not path.is_file():
        return ComponentCheck.bad(f"File not found: {path}"), None
    try:
        pose_ds = PoseDataset.from_file(path)
    except Exception as e:
        return ComponentCheck.bad(f"Could not read this as a pose labels file: {e}"), None
    n = len(pose_ds.frame_indices)
    if n == 0:
        return ComponentCheck.bad("This file has 0 labeled frames — is this the right file?"), None
    sibling_meta = path.parent / POSE_METADATA_FILENAME
    return (
        ComponentCheck.good(f"{n} labeled frame(s) found."),
        sibling_meta if sibling_meta.exists() else None,
    )


def check_behavior_labels_file(path: Optional[Path]) -> ComponentCheck:
    """Validate an (optional) behavior labels HDF5 file."""
    if path is None:
        return ComponentCheck.good(
            "No behavior labels file — dataset will be imported without behavior labels."
        )
    path = Path(path)
    if not path.exists() or not path.is_file():
        return ComponentCheck.bad(f"File not found: {path}")
    try:
        keys = BehaviorDataset.get_video_keys_from_file(path)
    except Exception as e:
        return ComponentCheck.bad(f"Could not read this as a behavior labels file: {e}")
    if not keys:
        return ComponentCheck.bad("This file has 0 labeled videos — is this the right file?")
    return ComponentCheck.good(f"{len(keys)} labeled video(s) found.")


# ------------------------------------------------------------------ #
# Workspace conflicts
# ------------------------------------------------------------------ #


@dataclass
class WorkspaceConflicts:
    project_exists: bool = False
    dataset_exists: bool = False
    pose_labels_exist: bool = False
    behavior_labels_exist: bool = False

    @property
    def any(self) -> bool:
        return (
            self.project_exists
            or self.dataset_exists
            or self.pose_labels_exist
            or self.behavior_labels_exist
        )


def check_workspace_conflicts(
    workspace, dataset_name: str, project_name: Optional[str]
) -> WorkspaceConflicts:
    return WorkspaceConflicts(
        project_exists=bool(project_name) and workspace.projects.get(project_name) is not None,
        dataset_exists=bool(dataset_name) and workspace.datasets.get(dataset_name) is not None,
        pose_labels_exist=bool(dataset_name)
        and (workspace.pose_labels.base_dir / dataset_name / LABELS_H5_FILENAME).exists(),
        behavior_labels_exist=bool(dataset_name)
        and (workspace.behavior_labels.base_dir / dataset_name / LABELS_H5_FILENAME).exists(),
    )


# ------------------------------------------------------------------ #
# Import
# ------------------------------------------------------------------ #


def import_dataset_from_components(
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
    progress_cb: Optional[ProgressCallback] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> dict:
    """Copy the picked components into ``workspace`` under ``dataset_name``.

    Every argument here is expected to have already passed its corresponding
    ``check_*`` function — this does the copying, not the validating.
    """

    def _progress(msg: str, pct: int):
        if progress_cb:
            progress_cb(msg, pct)

    def _cancelled() -> bool:
        return bool(cancel_cb and cancel_cb())

    dataset_name = dataset_name.strip()
    if not dataset_name:
        raise ManualImportError("Dataset name is required.")

    _progress("Importing project...", 0)

    # 1. Project definition
    project_installed = False
    project_dst = workspace.projects.base_dir / f"{project.name}.json"
    if not project_dst.exists() or overwrite:
        project_dst.parent.mkdir(parents=True, exist_ok=True)
        project_src = Path(project_source_path)
        if project_src.exists():
            shutil.copy2(project_src, project_dst)
        else:
            with open(project_dst, "w", encoding="utf-8") as f:
                f.write(project.model_dump_json(indent=4))
        project_installed = True

    # 2. Dataset (manifest + media)
    _progress("Preparing dataset...", 2)
    dataset_dir = workspace.datasets.base_dir / dataset_name
    if dataset_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Dataset '{dataset_name}' already exists in the workspace.")
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    rel_paths = list(dataset.files)
    total = len(rel_paths)
    copied = 0
    missing: List[str] = []

    data_dir = dataset_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    media_dir = Path(media_dir)
    for i, rel in enumerate(rel_paths):
        if _cancelled():
            raise ManualImportError("Import cancelled.")
        src = media_dir / rel
        dst = data_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src, dst)
            copied += 1
        except (OSError, shutil.Error) as e:
            logger.warning("Could not copy media %s: %s", src, e)
            missing.append(rel)
        if total and (i % 5 == 0 or i == total - 1):
            pct = 5 + int(80 * (i + 1) / total)
            _progress(f"Copying media ({i + 1}/{total})...", pct)

    imported_dataset = dataset.model_copy(
        update={"name": dataset_name, "base_data_path": str(data_dir.resolve())}
    )
    with open(dataset_dir / "manifest.json", "w", encoding="utf-8") as f:
        f.write(imported_dataset.model_dump_json(indent=4))

    # 3. Pose labels
    pose_imported = False
    if pose_labels_path is not None:
        _progress("Importing pose labels...", 90)
        pose_dst_dir = workspace.pose_labels.base_dir / dataset_name
        if pose_dst_dir.exists() and overwrite:
            shutil.rmtree(pose_dst_dir)
        if not pose_dst_dir.exists():
            pose_dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(pose_labels_path), pose_dst_dir / LABELS_H5_FILENAME)
            pose_imported = True
            if pose_metadata_path is not None and Path(pose_metadata_path).exists():
                shutil.copy2(Path(pose_metadata_path), pose_dst_dir / POSE_METADATA_FILENAME)

    # 4. Behavior labels
    behavior_imported = False
    if behavior_labels_path is not None:
        _progress("Importing behavior labels...", 96)
        bc_dst_dir = workspace.behavior_labels.base_dir / dataset_name
        if bc_dst_dir.exists() and overwrite:
            shutil.rmtree(bc_dst_dir)
        if not bc_dst_dir.exists():
            bc_dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(behavior_labels_path), bc_dst_dir / LABELS_H5_FILENAME)
            behavior_imported = True

    _progress("Import complete.", 100)

    return {
        "dataset_name": dataset_name,
        "project_name": project.name,
        "project_installed": project_installed,
        "media_kind": "videos" if dataset.type == DatasetType.VIDEO_COLLECTION else "frames",
        "num_media": total,
        "num_media_copied": copied,
        "num_missing": len(missing),
        "missing": missing,
        "pose_imported": pose_imported,
        "behavior_imported": behavior_imported,
    }
