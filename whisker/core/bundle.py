"""Tidy, self-describing annotation *bundles* for whisker-labeler.

A bundle packages everything needed to move a labeled dataset between
whisker-labeler workspaces (and into full whisker): the dataset manifest, the
project it was labeled under, the pose/behavior label HDF5s and their metadata,
and (optionally) a copy of the actual media — frame images or video files —
with original filenames and folder structure preserved.

Layout on disk::

    <bundle_name>/
        export_info.json          # machine-readable description of the bundle
        README.txt                # human-readable summary
        project/
            <project_name>.json   # the Project definition
        dataset/
            manifest.json         # the Dataset manifest (incl. multi-arena config)
        pose_labels/              # present only if pose labels exist
            labels.h5
            metadata.json
        behavior_labels/          # present only if behavior labels exist
            labels.h5
        frames/  or  videos/      # the media files (omitted for reference-only)
            <original relative paths ...>

``export_info.json`` records every relative path in the bundle so re-import is
deterministic and does not depend on this app's internal directory layout.

For video datasets the media copy is optional (``media_included``): a
reference-only bundle records the original media paths instead of copying the
(potentially very large) video files, and the importer supplies the videos.

The functions here perform *only* filesystem work (read + copy + write). They
never touch Qt and never mutate the in-memory workspace, so callers can run
them on a background thread and then rescan the workspace on the GUI thread.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

from whisker.core.study.dataset import Dataset, DatasetType
from whisker.core.study.project import Project

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Bundle constants
# ------------------------------------------------------------------ #

BUNDLE_FORMAT_VERSION = "1.0"
GENERATOR = "whisker-labeler"

EXPORT_INFO_FILENAME = "export_info.json"
README_FILENAME = "README.txt"

PROJECT_DIRNAME = "project"
DATASET_DIRNAME = "dataset"
POSE_LABELS_DIRNAME = "pose_labels"
BEHAVIOR_LABELS_DIRNAME = "behavior_labels"
FRAMES_DIRNAME = "frames"
VIDEOS_DIRNAME = "videos"

MANIFEST_FILENAME = "manifest.json"
LABELS_H5_FILENAME = "labels.h5"
POSE_METADATA_FILENAME = "metadata.json"

# All current dataset types are exportable.
EXPORTABLE_DATASET_TYPES = (
    DatasetType.IMAGE_COLLECTION,
    DatasetType.FRAME_SUBSET,
    DatasetType.VIDEO_COLLECTION,
)

ProgressCallback = Callable[[str, int], None]


class BundleError(Exception):
    """Raised when a bundle cannot be built or read."""


def media_kind_for(dataset_type: DatasetType) -> str:
    """'videos' for video datasets, else 'frames'."""
    return "videos" if dataset_type == DatasetType.VIDEO_COLLECTION else "frames"


def media_dirname_for(dataset_type: DatasetType) -> str:
    return VIDEOS_DIRNAME if dataset_type == DatasetType.VIDEO_COLLECTION else FRAMES_DIRNAME


# ------------------------------------------------------------------ #
# Export plan (shared by the dialog preview and the export job)
# ------------------------------------------------------------------ #


@dataclass
class PoseLabelInfo:
    present: bool = False
    labels_h5: Optional[Path] = None
    metadata_json: Optional[Path] = None
    num_labeled_frames: int = 0
    body_parts: List[str] = field(default_factory=list)
    individuals: List[str] = field(default_factory=list)


@dataclass
class BehaviorLabelInfo:
    present: bool = False
    labels_h5: Optional[Path] = None
    num_labeled_videos: int = 0


@dataclass
class BundleExportPlan:
    """Everything the export needs, resolved up front so the dialog can preview
    exactly what the job will write."""

    dataset: Dataset
    dataset_manifest_path: Path
    project: Project
    project_json_path: Optional[Path]
    media_base_path: Path
    media_rel_paths: List[str]
    pose: PoseLabelInfo
    behavior: BehaviorLabelInfo

    @property
    def num_media(self) -> int:
        return len(self.media_rel_paths)

    @property
    def media_kind(self) -> str:
        return media_kind_for(self.dataset.type)

    @property
    def media_dirname(self) -> str:
        return media_dirname_for(self.dataset.type)

    @property
    def is_video(self) -> bool:
        return self.dataset.type == DatasetType.VIDEO_COLLECTION

    def default_bundle_name(self) -> str:
        return f"{self.dataset.name}_bundle"


def _read_pose_metadata(metadata_path: Path) -> dict:
    if not metadata_path.exists():
        return {}
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not read pose metadata %s: %s", metadata_path, e)
        return {}


def build_export_plan(
    workspace, dataset_name: str, project_name: str
) -> BundleExportPlan:
    """Inspect the workspace and resolve everything a bundle for ``dataset_name``
    (labeled under ``project_name``) would contain. Pure reads — no copying."""
    dataset = workspace.datasets.get(dataset_name)
    if dataset is None:
        raise BundleError(f"Dataset '{dataset_name}' not found in workspace.")
    if dataset.type not in EXPORTABLE_DATASET_TYPES:
        raise BundleError(
            f"Dataset '{dataset_name}' has unsupported type {dataset.type.value}."
        )

    project = workspace.projects.get(project_name)
    if project is None:
        raise BundleError(f"Project '{project_name}' not found in workspace.")

    dataset_manifest_path = (
        workspace.datasets.base_dir / dataset_name / MANIFEST_FILENAME
    )
    project_json_path = workspace.projects.base_dir / f"{project_name}.json"
    if not project_json_path.exists():
        project_json_path = None

    # --- Pose labels ---
    pose = PoseLabelInfo()
    if workspace.pose_labels.has_pose_labels(dataset_name):
        pose_dir = workspace.pose_labels.base_dir / dataset_name
        h5 = pose_dir / LABELS_H5_FILENAME
        meta = pose_dir / POSE_METADATA_FILENAME
        if h5.exists():
            metadata = _read_pose_metadata(meta)
            keys = metadata.get("frame_indices") or metadata.get("annotated_images")
            if keys is None:
                try:
                    keys = list(
                        workspace.pose_labels.get_pose_labeled_image_keys(dataset_name)
                    )
                except Exception:  # pragma: no cover - defensive
                    keys = []
            pose = PoseLabelInfo(
                present=True,
                labels_h5=h5,
                metadata_json=meta if meta.exists() else None,
                num_labeled_frames=len(keys),
                body_parts=list(metadata.get("body_parts", [])),
                individuals=list(metadata.get("individuals", [])),
            )

    # --- Behavior labels ---
    behavior = BehaviorLabelInfo()
    if workspace.behavior_labels.has_behavior_labels(dataset_name):
        bc_dir = workspace.behavior_labels.base_dir / dataset_name
        h5 = bc_dir / LABELS_H5_FILENAME
        if h5.exists():
            try:
                keys = workspace.behavior_labels.get_behavior_labeled_video_keys(
                    dataset_name
                )
            except Exception:  # pragma: no cover - defensive
                keys = set()
            behavior = BehaviorLabelInfo(
                present=True,
                labels_h5=h5,
                num_labeled_videos=len(keys),
            )

    return BundleExportPlan(
        dataset=dataset,
        dataset_manifest_path=dataset_manifest_path,
        project=project,
        project_json_path=project_json_path,
        media_base_path=Path(dataset.base_data_path),
        media_rel_paths=list(dataset.files),
        pose=pose,
        behavior=behavior,
    )


# ------------------------------------------------------------------ #
# Export
# ------------------------------------------------------------------ #


def _copy_or_write(src: Optional[Path], dst: Path, fallback_text: Optional[str]):
    """Copy ``src`` to ``dst`` if it exists on disk; otherwise write
    ``fallback_text`` (used when a manifest/project lives only in memory)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src is not None and src.exists():
        shutil.copy2(src, dst)
    elif fallback_text is not None:
        with open(dst, "w", encoding="utf-8") as f:
            f.write(fallback_text)
    else:
        raise BundleError(f"Missing required source file: {src}")


def export_annotation_bundle(
    plan: BundleExportPlan,
    bundle_dir: Path,
    overwrite: bool = False,
    include_media: bool = True,
    progress_cb: Optional[ProgressCallback] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> dict:
    """Write the bundle described by ``plan`` into ``bundle_dir``.

    ``bundle_dir`` is the full path of the bundle folder to create (including
    its name). Raises :class:`FileExistsError` if it already exists and
    ``overwrite`` is False. When ``include_media`` is False the media files are
    not copied (reference-only) and the original media path is recorded instead.
    """

    def _progress(msg: str, pct: int):
        if progress_cb:
            progress_cb(msg, pct)

    def _cancelled() -> bool:
        return bool(cancel_cb and cancel_cb())

    bundle_dir = Path(bundle_dir)
    if bundle_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {bundle_dir}")
        shutil.rmtree(bundle_dir)

    _progress("Preparing bundle...", 0)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # 1. Project definition
    project_dst = bundle_dir / PROJECT_DIRNAME / f"{plan.project.name}.json"
    _copy_or_write(
        plan.project_json_path, project_dst, plan.project.model_dump_json(indent=4)
    )

    # 2. Dataset manifest (carries multi-arena config verbatim, if any)
    manifest_dst = bundle_dir / DATASET_DIRNAME / MANIFEST_FILENAME
    _copy_or_write(
        plan.dataset_manifest_path, manifest_dst, plan.dataset.model_dump_json(indent=4)
    )

    # 3. Pose labels
    pose_info = {"present": False}
    if plan.pose.present and plan.pose.labels_h5 is not None:
        _progress("Copying pose labels...", 4)
        pose_h5_dst = bundle_dir / POSE_LABELS_DIRNAME / LABELS_H5_FILENAME
        _copy_or_write(plan.pose.labels_h5, pose_h5_dst, None)
        pose_info = {
            "present": True,
            "labels_h5": f"{POSE_LABELS_DIRNAME}/{LABELS_H5_FILENAME}",
            "num_labeled_frames": plan.pose.num_labeled_frames,
            "body_parts": plan.pose.body_parts,
            "individuals": plan.pose.individuals,
        }
        if plan.pose.metadata_json is not None and plan.pose.metadata_json.exists():
            meta_dst = bundle_dir / POSE_LABELS_DIRNAME / POSE_METADATA_FILENAME
            _copy_or_write(plan.pose.metadata_json, meta_dst, None)
            pose_info["metadata"] = f"{POSE_LABELS_DIRNAME}/{POSE_METADATA_FILENAME}"

    # 4. Behavior labels
    behavior_info = {"present": False}
    if plan.behavior.present and plan.behavior.labels_h5 is not None:
        _progress("Copying behavior labels...", 7)
        bc_h5_dst = bundle_dir / BEHAVIOR_LABELS_DIRNAME / LABELS_H5_FILENAME
        _copy_or_write(plan.behavior.labels_h5, bc_h5_dst, None)
        behavior_info = {
            "present": True,
            "labels_h5": f"{BEHAVIOR_LABELS_DIRNAME}/{LABELS_H5_FILENAME}",
            "num_labeled_videos": plan.behavior.num_labeled_videos,
        }

    # 5. Media (frames or videos) — the bulk of the work when copied
    media_dirname = plan.media_dirname
    total = plan.num_media
    copied = 0
    missing: List[str] = []
    if include_media:
        media_root = bundle_dir / media_dirname
        for i, rel in enumerate(plan.media_rel_paths):
            if _cancelled():
                raise BundleError("Export cancelled.")
            src = plan.media_base_path / rel
            dst = media_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(src, dst)
                copied += 1
            except (OSError, shutil.Error) as e:
                logger.warning("Could not copy media %s: %s", src, e)
                missing.append(rel)
            if total and (i % 5 == 0 or i == total - 1):
                pct = 10 + int(88 * (i + 1) / total)
                _progress(f"Copying {media_dirname} ({i + 1}/{total})...", pct)

    # 6. export_info.json
    _progress("Writing bundle manifest...", 98)
    dataset_info = {
        "name": plan.dataset.name,
        "type": plan.dataset.type.value,
        "manifest": f"{DATASET_DIRNAME}/{MANIFEST_FILENAME}",
        "media_kind": plan.media_kind,
        "media_dir": media_dirname,
        "media_included": bool(include_media),
        "num_media": total,
        "num_media_copied": copied,
        "original_base_data_path": str(plan.media_base_path),
        "multi_arena": plan.dataset.is_multi_arena,
    }
    if include_media and missing:
        dataset_info["missing_media"] = missing
    if not include_media:
        # Record the referenced media paths so the importer knows what to supply.
        dataset_info["referenced_media"] = list(plan.media_rel_paths)

    export_info = {
        "bundle_format_version": BUNDLE_FORMAT_VERSION,
        "generator": GENERATOR,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project": {
            "name": plan.project.name,
            "file": f"{PROJECT_DIRNAME}/{plan.project.name}.json",
        },
        "dataset": dataset_info,
        "pose_labels": pose_info,
        "behavior_labels": behavior_info,
    }

    with open(bundle_dir / EXPORT_INFO_FILENAME, "w", encoding="utf-8") as f:
        json.dump(export_info, f, indent=4)

    _write_readme(bundle_dir, export_info)

    _progress("Export complete.", 100)

    return {
        "bundle_dir": str(bundle_dir),
        "media_kind": plan.media_kind,
        "media_included": bool(include_media),
        "num_media": total,
        "num_media_copied": copied,
        "num_missing": len(missing),
        "missing": missing,
        "pose_present": pose_info["present"],
        "behavior_present": behavior_info["present"],
    }


def _write_readme(bundle_dir: Path, export_info: dict) -> None:
    ds = export_info["dataset"]
    media_dir = ds["media_dir"]
    if ds["media_included"]:
        media_line = (
            f"  {media_dir}/            the {ds['media_kind']} "
            "(original names preserved)"
        )
    else:
        media_line = (
            f"  ({ds['media_kind']} NOT included — reference only; "
            f"originally at {ds['original_base_data_path']})"
        )
    lines = [
        "whisker-labeler annotation bundle",
        f"Bundle format version: {export_info['bundle_format_version']}",
        f"Created: {export_info['created_at']}",
        "",
        f"Dataset:  {ds['name']} ({ds['type']}, {ds['num_media']} {ds['media_kind']})",
        f"Project:  {export_info['project']['name']}",
        f"Multi-arena:     {'yes' if ds.get('multi_arena') else 'no'}",
        f"Pose labels:     {'yes' if export_info['pose_labels']['present'] else 'no'}",
        f"Behavior labels: {'yes' if export_info['behavior_labels']['present'] else 'no'}",
        "",
        "Layout:",
        f"  {EXPORT_INFO_FILENAME}   machine-readable description of this bundle",
        f"  {PROJECT_DIRNAME}/           the project definition (labels schema)",
        f"  {DATASET_DIRNAME}/           the dataset manifest",
        f"  {POSE_LABELS_DIRNAME}/       pose label HDF5 + metadata (if any)",
        f"  {BEHAVIOR_LABELS_DIRNAME}/   behavior label HDF5 (if any)",
        media_line,
        "",
        "Import into whisker-labeler via the Data Explorer options menu ->",
        "'Import Annotation Bundle...' and select this folder.",
    ]
    with open(bundle_dir / README_FILENAME, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ------------------------------------------------------------------ #
# Import
# ------------------------------------------------------------------ #


def read_bundle_info(bundle_dir: Path) -> dict:
    """Load and validate ``export_info.json`` from a bundle directory."""
    bundle_dir = Path(bundle_dir)
    info_path = bundle_dir / EXPORT_INFO_FILENAME
    if not info_path.exists():
        raise BundleError(
            f"Not a valid annotation bundle: '{EXPORT_INFO_FILENAME}' not found "
            f"in {bundle_dir}"
        )
    try:
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise BundleError(f"Could not read {info_path}: {e}") from e

    version = str(info.get("bundle_format_version", ""))
    if not version:
        raise BundleError(f"{EXPORT_INFO_FILENAME} is missing 'bundle_format_version'.")
    if version.split(".")[0] != BUNDLE_FORMAT_VERSION.split(".")[0]:
        raise BundleError(
            f"Unsupported bundle format version {version} "
            f"(this app understands {BUNDLE_FORMAT_VERSION})."
        )
    if "dataset" not in info or "project" not in info:
        raise BundleError(f"{EXPORT_INFO_FILENAME} is missing required sections.")
    return info


@dataclass
class BundleImportPreview:
    info: dict
    bundle_dir: Path
    dataset_name: str
    dataset_type: str
    media_kind: str
    media_included: bool
    num_media: int
    original_base_data_path: str
    project_name: str
    pose_present: bool
    behavior_present: bool
    multi_arena: bool
    # Conflicts in the *target* workspace
    project_exists: bool
    dataset_exists: bool
    pose_labels_exist: bool
    behavior_labels_exist: bool

    @property
    def has_conflicts(self) -> bool:
        return (
            self.project_exists
            or self.dataset_exists
            or self.pose_labels_exist
            or self.behavior_labels_exist
        )


def build_import_preview(workspace, bundle_dir: Path) -> BundleImportPreview:
    """Read a bundle and detect what it would create/overwrite in ``workspace``."""
    bundle_dir = Path(bundle_dir)
    info = read_bundle_info(bundle_dir)

    ds = info["dataset"]
    dataset_name = ds["name"]
    project_name = info["project"]["name"]
    pose_present = bool(info.get("pose_labels", {}).get("present"))
    behavior_present = bool(info.get("behavior_labels", {}).get("present"))

    project_exists = workspace.projects.get(project_name) is not None
    dataset_exists = workspace.datasets.get(dataset_name) is not None
    pose_labels_exist = (
        workspace.pose_labels.base_dir / dataset_name / LABELS_H5_FILENAME
    ).exists()
    behavior_labels_exist = (
        workspace.behavior_labels.base_dir / dataset_name / LABELS_H5_FILENAME
    ).exists()

    return BundleImportPreview(
        info=info,
        bundle_dir=bundle_dir,
        dataset_name=dataset_name,
        dataset_type=ds.get("type", ""),
        media_kind=ds.get("media_kind", "frames"),
        media_included=bool(ds.get("media_included", True)),
        num_media=int(ds.get("num_media", 0)),
        original_base_data_path=str(ds.get("original_base_data_path", "")),
        project_name=project_name,
        pose_present=pose_present,
        behavior_present=behavior_present,
        multi_arena=bool(ds.get("multi_arena", False)),
        project_exists=project_exists,
        dataset_exists=dataset_exists,
        pose_labels_exist=pose_labels_exist,
        behavior_labels_exist=behavior_labels_exist,
    )


def import_annotation_bundle(
    workspace,
    bundle_dir: Path,
    overwrite: bool = False,
    media_source_dir: Optional[Path] = None,
    progress_cb: Optional[ProgressCallback] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> dict:
    """Import a bundle into ``workspace`` (filesystem only).

    When the bundle includes media, the frames/videos are copied into the
    workspace and the manifest's ``base_data_path`` points at the copy. When the
    bundle is reference-only, no media is copied and ``base_data_path`` is set to
    ``media_source_dir`` (or the recorded original path if not provided).

    Does *not* rescan the in-memory workspace — the caller should do that on the
    GUI thread afterwards.
    """

    def _progress(msg: str, pct: int):
        if progress_cb:
            progress_cb(msg, pct)

    def _cancelled() -> bool:
        return bool(cancel_cb and cancel_cb())

    bundle_dir = Path(bundle_dir)
    info = read_bundle_info(bundle_dir)

    ds_info = info["dataset"]
    dataset_name = ds_info["name"]
    project_name = info["project"]["name"]
    media_included = bool(ds_info.get("media_included", True))

    _progress("Importing project...", 0)

    # 1. Project definition
    project_installed = False
    project_dst = workspace.projects.base_dir / f"{project_name}.json"
    project_src = bundle_dir / info["project"]["file"]
    if not project_dst.exists() or overwrite:
        if project_src.exists():
            project_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(project_src, project_dst)
            project_installed = True
        else:
            raise BundleError(f"Bundle project file missing: {project_src}")

    # 2. Dataset (manifest + media)
    _progress("Preparing dataset...", 2)
    manifest_src = bundle_dir / ds_info["manifest"]
    if not manifest_src.exists():
        raise BundleError(f"Bundle manifest missing: {manifest_src}")
    with open(manifest_src, "rb") as f:
        dataset = Dataset.from_json(f.read().decode())

    dataset_dir = workspace.datasets.base_dir / dataset_name
    if dataset_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Dataset '{dataset_name}' already exists in the workspace."
            )
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    rel_paths = list(dataset.files)
    total = len(rel_paths)
    copied = 0
    missing: List[str] = []

    if media_included:
        data_dir = dataset_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        media_root = bundle_dir / ds_info.get("media_dir", FRAMES_DIRNAME)
        for i, rel in enumerate(rel_paths):
            if _cancelled():
                raise BundleError("Import cancelled.")
            src = media_root / rel
            dst = data_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(src, dst)
                copied += 1
            except (OSError, shutil.Error) as e:
                logger.warning("Could not copy media %s: %s", src, e)
                missing.append(rel)
            if total and (i % 5 == 0 or i == total - 1):
                pct = 5 + int(85 * (i + 1) / total)
                _progress(f"Copying media ({i + 1}/{total})...", pct)
        new_base = str(data_dir.resolve())
    else:
        # Reference-only: point at the media the importer supplied, or the
        # original path recorded in the bundle.
        if media_source_dir:
            new_base = str(Path(media_source_dir).resolve())
        else:
            new_base = ds_info.get("original_base_data_path", "")
        missing = [
            rel for rel in rel_paths if not (Path(new_base) / rel).exists()
        ]

    # Rewrite manifest so it points at the resolved media location.
    imported_dataset = dataset.model_copy(update={"base_data_path": new_base})
    with open(dataset_dir / MANIFEST_FILENAME, "w", encoding="utf-8") as f:
        f.write(imported_dataset.model_dump_json(indent=4))

    # 3. Pose labels
    pose_imported = False
    pose_info = info.get("pose_labels", {})
    if pose_info.get("present"):
        _progress("Importing pose labels...", 92)
        pose_dst_dir = workspace.pose_labels.base_dir / dataset_name
        if pose_dst_dir.exists() and overwrite:
            shutil.rmtree(pose_dst_dir)
        if not pose_dst_dir.exists():
            pose_dst_dir.mkdir(parents=True, exist_ok=True)
            h5_src = bundle_dir / pose_info["labels_h5"]
            if h5_src.exists():
                shutil.copy2(h5_src, pose_dst_dir / LABELS_H5_FILENAME)
                pose_imported = True
            meta_rel = pose_info.get("metadata")
            if meta_rel and (bundle_dir / meta_rel).exists():
                shutil.copy2(
                    bundle_dir / meta_rel, pose_dst_dir / POSE_METADATA_FILENAME
                )

    # 4. Behavior labels
    behavior_imported = False
    bc_info = info.get("behavior_labels", {})
    if bc_info.get("present"):
        _progress("Importing behavior labels...", 96)
        bc_dst_dir = workspace.behavior_labels.base_dir / dataset_name
        if bc_dst_dir.exists() and overwrite:
            shutil.rmtree(bc_dst_dir)
        if not bc_dst_dir.exists():
            bc_dst_dir.mkdir(parents=True, exist_ok=True)
            h5_src = bundle_dir / bc_info["labels_h5"]
            if h5_src.exists():
                shutil.copy2(h5_src, bc_dst_dir / LABELS_H5_FILENAME)
                behavior_imported = True

    _progress("Import complete.", 100)

    return {
        "dataset_name": dataset_name,
        "project_name": project_name,
        "media_kind": ds_info.get("media_kind", "frames"),
        "media_included": media_included,
        "num_media": total,
        "num_media_copied": copied,
        "num_missing": len(missing),
        "missing": missing,
        "project_installed": project_installed,
        "pose_imported": pose_imported,
        "behavior_imported": behavior_imported,
    }
