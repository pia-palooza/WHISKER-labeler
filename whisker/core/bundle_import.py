"""One-pick import of an exported bundle.

The user points at *something* — the export folder, a folder inside it, a folder that
contains it (unzipping adds a level), or any file in it. :func:`locate_bundle` finds
the export root by looking for ``export_info.json``, and says exactly what it looked at
when it can't. :func:`inspect_bundle` then reports what the export contains, part by
part (project, dataset + media, pose labels, behavior labels), so the UI can offer a
checklist and the user can import any subset.

Labels imported *without* their dataset are attached to an existing dataset:
:func:`analyze_pose_labels` / :func:`analyze_behavior_labels` compare the labels with
that dataset's files and report mismatches, and :func:`apply_pose_labels` /
:func:`apply_behavior_labels` add, replace, or merge (choosing who wins on overlap).
Existing labels are backed up during a write and restored if it fails.

Pure filesystem/pandas work — no Qt — so it can run on a background thread. The
piece-by-piece writers live in :mod:`whisker.core.manual_import` and are shared with
the "pick each piece yourself" flow.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from whisker.core import bundle as fmt
from whisker.core import manual_import as mi
from whisker.core.study.dataset import Dataset, DatasetType
from whisker.core.study.project import Project
from whisker.services.behavior_classification.public.data_structures import BehaviorDataset
from whisker.services.pose_estimation.public.data_structures import PoseDataset

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, int], None]

MAX_LEVELS_UP = 6
MAX_LEVELS_DOWN = 3
MAX_DIRS_SCANNED = 3000
# Never descend into these: they're the bundle's own content folders (which can hold
# 100k+ frames) or system clutter.
_SKIP_DIRS = {
    fmt.FRAMES_DIRNAME, fmt.VIDEOS_DIRNAME, fmt.POSE_LABELS_DIRNAME,
    fmt.BEHAVIOR_LABELS_DIRNAME, fmt.PROJECT_DIRNAME, fmt.DATASET_DIRNAME,
    ".git", "__pycache__", "$recycle.bin", "system volume information",
}


class BundleImportError(Exception):
    """The bundle can't be read, or the requested import can't proceed."""


# --------------------------------------------------------------------------- #
# 1. Locating a bundle from any pick
# --------------------------------------------------------------------------- #


@dataclass
class BundleLocation:
    root: Optional[Path] = None          # the export folder, when exactly one was found
    candidates: List[Path] = field(default_factory=list)   # every export found
    message: str = ""                    # plain-English explanation to show the user

    @property
    def found(self) -> bool:
        return self.root is not None


def _search_down(start: Path) -> Tuple[List[Path], bool]:
    """Find export folders at or below ``start`` (breadth-first, bounded)."""
    found: List[Path] = []
    queue = [(start, 0)]
    scanned = 0
    while queue:
        directory, depth = queue.pop(0)
        scanned += 1
        if scanned > MAX_DIRS_SCANNED:
            return found, True
        subdirs = []
        try:
            with os.scandir(directory) as it:
                for entry in it:
                    if entry.name == fmt.EXPORT_INFO_FILENAME and entry.is_file():
                        found.append(directory)
                        subdirs = []
                        break
                    if (depth < MAX_LEVELS_DOWN and entry.is_dir(follow_symlinks=False)
                            and not entry.name.startswith(".")
                            and entry.name.lower() not in _SKIP_DIRS):
                        subdirs.append(Path(entry.path))
        except OSError:
            continue
        queue.extend((d, depth + 1) for d in subdirs)
    return found, False


def locate_bundle(picked: os.PathLike | str) -> BundleLocation:
    """Find the export that ``picked`` refers to (see the module docstring)."""
    p = Path(picked)
    if not p.exists():
        return BundleLocation(message=f"'{p}' doesn't exist.")

    if p.is_file():
        if p.name == fmt.EXPORT_INFO_FILENAME:
            return BundleLocation(root=p.parent, candidates=[p.parent])
        start = p.parent
    else:
        start = p

    if (start / fmt.EXPORT_INFO_FILENAME).is_file():
        return BundleLocation(root=start, candidates=[start])

    below, hit_cap = _search_down(start)
    if len(below) == 1:
        return BundleLocation(
            root=below[0], candidates=below,
            message=f"Found the export '{below[0].name}' inside the folder you picked.",
        )
    if len(below) > 1:
        return BundleLocation(
            candidates=sorted(below),
            message=f"Found {len(below)} exports inside '{start.name}'. Choose the one you want.",
        )

    for level, ancestor in enumerate(start.parents):
        if level >= MAX_LEVELS_UP:
            break
        if (ancestor / fmt.EXPORT_INFO_FILENAME).is_file():
            return BundleLocation(
                root=ancestor, candidates=[ancestor],
                message=f"You picked something inside the export '{ancestor.name}', so I used the whole export.",
            )

    msg = (
        f"Couldn't find a WHISKER export at '{p}'. I looked for a file named "
        f"'{fmt.EXPORT_INFO_FILENAME}' in that folder, in the folders inside it "
        f"(up to {MAX_LEVELS_DOWN} levels down), and in the folders above it "
        f"(up to {MAX_LEVELS_UP} levels up)."
    )
    if hit_cap:
        msg += f" I stopped after {MAX_DIRS_SCANNED} folders, so try a more specific folder."
    if (start / fmt.MANIFEST_FILENAME).is_file() or any(start.glob("*.json")):
        msg += (
            " This folder has .json files but no export description, so it wasn't made by "
            "Export. If you have the pieces separately, use 'Pick pieces manually'."
        )
    else:
        msg += (
            " Pick the folder that File → Export created (it contains "
            f"{fmt.EXPORT_INFO_FILENAME} and README.txt)."
        )
    return BundleLocation(message=msg)


# --------------------------------------------------------------------------- #
# 2. Inspecting a bundle
# --------------------------------------------------------------------------- #


@dataclass
class Part:
    """One importable piece of an export, as shown in the checklist."""
    present: bool = False       # the export contains it
    ok: bool = False            # ...and it can be imported
    summary: str = ""           # one line for the UI
    problem: str = ""           # why it can't be imported (present but broken)
    path: Optional[Path] = None


@dataclass
class BundleContents:
    root: Path
    info: dict
    project: Part = field(default_factory=Part)
    dataset: Part = field(default_factory=Part)       # dataset info + its videos/frames
    pose: Part = field(default_factory=Part)
    behavior: Part = field(default_factory=Part)
    warnings: List[str] = field(default_factory=list)

    project_obj: Optional[Project] = None
    dataset_obj: Optional[Dataset] = None
    media_kind: str = "frames"                        # 'frames' | 'videos'
    media_included: bool = False
    media_dir: Optional[Path] = None                  # where the media are, if known
    media_missing: List[str] = field(default_factory=list)
    original_media_dir: Optional[Path] = None         # where they were on the exporter's machine

    pose_keys: List[str] = field(default_factory=list)
    pose_body_parts: List[str] = field(default_factory=list)
    pose_individuals: List[str] = field(default_factory=list)
    behavior_keys: List[str] = field(default_factory=list)
    behavior_names: List[str] = field(default_factory=list)

    @property
    def needs_media_folder(self) -> bool:
        """The dataset can only be imported once the user says where its media are."""
        return self.dataset.ok and self.media_dir is None

    @property
    def dataset_name(self) -> str:
        return self.dataset_obj.name if self.dataset_obj else ""

    @property
    def has_anything(self) -> bool:
        return any(p.present for p in (self.project, self.dataset, self.pose, self.behavior))


def inspect_bundle(root: Path) -> BundleContents:
    """Read ``export_info.json`` and validate each part of the export."""
    root = Path(root)
    info_path = root / fmt.EXPORT_INFO_FILENAME
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except OSError as e:
        raise BundleImportError(f"Couldn't open {info_path.name}: {e}") from e
    except json.JSONDecodeError as e:
        raise BundleImportError(
            f"{info_path.name} in '{root.name}' isn't readable ({e}). "
            "The export may be damaged or incomplete — try copying it again."
        ) from e
    if not isinstance(info, dict):
        raise BundleImportError(f"{info_path.name} in '{root.name}' isn't a WHISKER export description.")

    c = BundleContents(root=root, info=info)

    try:
        newest_known = int(fmt.BUNDLE_FORMAT_VERSION.split(".")[0])
        if int(str(info.get("bundle_format_version", "1")).split(".")[0]) > newest_known:
            c.warnings.append(
                "This export was made by a newer version of the labeler. I'll import what I "
                "can read, but some details may be missed — consider updating."
            )
    except ValueError:
        pass

    _inspect_project(c)
    _inspect_dataset(c)
    _inspect_pose(c)
    _inspect_behavior(c)
    return c


def _inspect_project(c: BundleContents) -> None:
    pinfo = c.info.get("project") or {}
    rel = pinfo.get("file")
    if not rel or pinfo.get("included") is False:
        c.project = Part(present=False, summary="Not in this export")
        return
    path = c.root / rel
    check, project = mi.check_project_file(path)
    c.project_obj = project
    c.project = Part(present=True, ok=check.ok, path=path,
                     summary=check.message if check.ok else "", problem="" if check.ok else check.message)


def _inspect_dataset(c: BundleContents) -> None:
    dinfo = c.info.get("dataset") or {}
    rel = dinfo.get("manifest") or f"{fmt.DATASET_DIRNAME}/{fmt.MANIFEST_FILENAME}"
    path = c.root / rel
    check, dataset = mi.check_dataset_file(path)
    c.dataset_obj = dataset
    if dataset is None or not check.ok:
        c.dataset = Part(present=path.exists(), ok=False, path=path, problem=check.message)
        return

    kind = fmt.media_kind_for(dataset.type)
    c.media_kind = kind
    c.media_included = bool(dinfo.get("media_included", False))
    n = len(dataset.files)

    if c.media_included:
        media_dir = c.root / (dinfo.get("media_dir") or fmt.media_dirname_for(dataset.type))
        media_check, missing = mi.check_media_folder(media_dir, dataset)
        c.media_missing = missing
        if media_check.ok:
            c.media_dir = media_dir
            c.dataset = Part(True, True, f"{n} {kind}, copied inside this export", path=path)
        else:
            c.dataset = Part(True, False, problem=media_check.message, path=path)
        return

    # Reference-only export: the media live wherever they lived on the exporter's machine.
    original = dinfo.get("original_base_data_path")
    if original:
        c.original_media_dir = Path(original)
        ok, _ = mi.check_media_folder(c.original_media_dir, dataset)
        if ok.ok:
            c.media_dir = c.original_media_dir
            c.dataset = Part(True, True, f"{n} {kind}, found at their original location", path=path)
            return
    c.dataset = Part(True, True, f"{n} {kind} listed — the files aren't in this export, "
                                 "so you'll be asked where they are", path=path)


def _inspect_pose(c: BundleContents) -> None:
    pinfo = c.info.get("pose_labels") or {}
    if not pinfo.get("present"):
        c.pose = Part(present=False, summary="Not in this export")
        return
    path = c.root / (pinfo.get("labels_h5") or f"{fmt.POSE_LABELS_DIRNAME}/{fmt.LABELS_H5_FILENAME}")
    if not path.is_file():
        c.pose = Part(True, False, path=path, problem=f"Listed in the export but the file is missing: {path.name}")
        return
    try:
        pose = PoseDataset.from_file(path)
    except Exception as e:
        c.pose = Part(True, False, path=path, problem=f"Could not read the pose labels: {e}")
        return
    keys = pose.frame_indices
    if not keys:
        c.pose = Part(True, False, path=path, problem="The pose labels file has no labeled frames.")
        return
    c.pose_keys, c.pose_body_parts, c.pose_individuals = list(keys), list(pose.body_parts), list(pose.individuals)
    c.pose = Part(True, True, f"{len(keys)} labeled frame(s), {len(pose.body_parts)} body part(s)", path=path)


def _inspect_behavior(c: BundleContents) -> None:
    binfo = c.info.get("behavior_labels") or {}
    if not binfo.get("present"):
        c.behavior = Part(present=False, summary="Not in this export")
        return
    path = c.root / (binfo.get("labels_h5") or f"{fmt.BEHAVIOR_LABELS_DIRNAME}/{fmt.LABELS_H5_FILENAME}")
    if not path.is_file():
        c.behavior = Part(True, False, path=path, problem=f"Listed in the export but the file is missing: {path.name}")
        return
    try:
        beh = BehaviorDataset.from_file(path)
    except Exception as e:
        c.behavior = Part(True, False, path=path, problem=f"Could not read the behavior labels: {e}")
        return
    keys = sorted(set(beh.bouts["video_key"])) if not beh.bouts.empty else []
    if not keys:
        c.behavior = Part(True, False, path=path, problem="The behavior labels file has no labeled videos.")
        return
    c.behavior_keys, c.behavior_names = keys, list(beh.behaviors)
    c.behavior = Part(True, True, f"{len(keys)} labeled video(s), {len(beh.bouts)} bout(s)", path=path)


# --------------------------------------------------------------------------- #
# 3. Comparing labels with a dataset (the mismatch report)
# --------------------------------------------------------------------------- #


def _norm(key: str) -> str:
    return str(key).replace("\\", "/")


@dataclass
class LabelReport:
    kind: str                                   # 'pose' | 'behavior'
    total: int = 0                              # distinct keys in the imported labels
    matched: List[str] = field(default_factory=list)      # imported keys that match a dataset file
    unmatched: List[str] = field(default_factory=list)    # imported keys with no matching file
    key_map: Dict[str, str] = field(default_factory=dict)  # imported key -> the dataset's own spelling
    unlabeled_files: int = 0                    # dataset files the imported labels don't cover
    dataset_files: int = 0
    existing_total: int = 0                     # keys the dataset already has labels for
    overlapping: List[str] = field(default_factory=list)   # matched keys already labeled (dataset spelling)
    problems: List[str] = field(default_factory=list)      # block merging
    warnings: List[str] = field(default_factory=list)      # worth knowing, don't block

    @property
    def has_existing(self) -> bool:
        return self.existing_total > 0

    @property
    def can_merge(self) -> bool:
        return self.has_existing and not self.problems

    @property
    def nothing_matches(self) -> bool:
        return self.total > 0 and not self.matched


def _project_names(workspace) -> List[str]:
    return list(workspace.projects.keys())


def _projects_covering(workspace, body_parts=(), identities=(), behaviors=()) -> bool:
    for name in _project_names(workspace):
        p = workspace.projects.get(name)
        if p and set(body_parts) <= set(p.body_parts) and set(identities) <= set(p.identities) \
                and set(behaviors) <= set(p.behaviors):
            return True
    return False


def analyze_pose_labels(
    workspace, dataset_name: str, imported_path: Path, existing: Optional[PoseDataset] = None
) -> LabelReport:
    dataset = workspace.datasets.get(dataset_name)
    if dataset is None:
        raise BundleImportError(f"Dataset '{dataset_name}' isn't in this workspace.")
    imported = PoseDataset.from_file(Path(imported_path))
    if existing is None and workspace.pose_labels.has_pose_labels(dataset_name):
        existing = PoseDataset.from_file(workspace.pose_labels.base_dir / dataset_name / fmt.LABELS_H5_FILENAME)

    files = {_norm(f): f for f in dataset.files}
    keys = list(imported.frame_indices)
    r = LabelReport("pose", total=len(keys), dataset_files=len(files))
    for k in keys:
        target = files.get(_norm(k))
        if target is None:
            r.unmatched.append(k)
        else:
            r.matched.append(k)
            r.key_map[k] = target
    r.unlabeled_files = len(files) - len({r.key_map[k] for k in r.matched})

    if existing is not None and existing.frame_indices:
        have = {_norm(k) for k in existing.frame_indices}
        r.existing_total = len(have)
        r.overlapping = sorted(r.key_map[k] for k in r.matched if _norm(k) in have)
        if set(existing.body_parts) != set(imported.body_parts):
            r.problems.append(
                f"The body parts differ — this dataset's labels use {sorted(existing.body_parts)}, "
                f"the imported ones use {sorted(imported.body_parts)}. They can't be combined."
            )
        if set(existing.individuals) != set(imported.individuals):
            r.problems.append(
                f"The animal identities differ — this dataset's labels use {sorted(existing.individuals)}, "
                f"the imported ones use {sorted(imported.individuals)}. They can't be combined."
            )
    if not _projects_covering(workspace, imported.body_parts, imported.individuals):
        r.warnings.append(
            "No project in this workspace has these body parts and identities. The labels will "
            "import, but you'll need a matching project to label with them — import the "
            "export's project too."
        )
    return r


def _behavior_key_index(dataset: Dataset) -> Dict[str, str]:
    """Every label key the app accepts for this dataset -> the app's own spelling.

    The app keys behavior labels by the video's file *name* (``clip.mp4``), or
    ``<stem>_arena<k>`` for multi-arena datasets. A bare stem is accepted too (older
    labels) and is normalised to the file name.
    """
    index: Dict[str, str] = {}
    for f in dataset.files:
        name = Path(_norm(f)).name
        index[name] = name
        index.setdefault(Path(name).stem, name)
    if dataset.multi_arena is not None:
        for _video, _k, arena_stem, _box in dataset.multi_arena.arena_units():
            index[arena_stem] = arena_stem
    return index


def analyze_behavior_labels(
    workspace, dataset_name: str, imported_path: Path, existing: Optional[BehaviorDataset] = None
) -> LabelReport:
    dataset = workspace.datasets.get(dataset_name)
    if dataset is None:
        raise BundleImportError(f"Dataset '{dataset_name}' isn't in this workspace.")
    imported = BehaviorDataset.from_file(Path(imported_path))
    if existing is None and workspace.behavior_labels.has_behavior_labels(dataset_name):
        existing = BehaviorDataset.from_file(workspace.behavior_labels.base_dir / dataset_name / fmt.LABELS_H5_FILENAME)

    index = _behavior_key_index(dataset)
    keys = sorted(set(imported.bouts["video_key"])) if not imported.bouts.empty else []
    r = LabelReport("behavior", total=len(keys), dataset_files=len(dataset.files))
    for k in keys:
        target = index.get(k)
        if target is None:
            r.unmatched.append(k)
        else:
            r.matched.append(k)
            r.key_map[k] = target
    # A video counts as labeled if any key maps to it (an arena key maps to its parent video).
    covered = set()
    for target in r.key_map.values():
        stem = Path(target).stem
        covered.add(stem.rsplit("_arena", 1)[0] if dataset.multi_arena is not None and "_arena" in stem else stem)
    r.unlabeled_files = sum(1 for f in dataset.files if Path(_norm(f)).stem not in covered)

    if existing is not None and not existing.bouts.empty:
        have = set(existing.bouts["video_key"])
        r.existing_total = len(have)
        r.overlapping = sorted(t for t in set(r.key_map.values()) if t in have)
    if not _projects_covering(workspace, behaviors=imported.behaviors):
        r.warnings.append(
            "No project in this workspace lists these behaviors "
            f"({', '.join(imported.behaviors) or 'none'}). Import the export's project too, "
            "or add them in Project Settings."
        )
    return r


def rank_target_datasets(workspace, contents: BundleContents) -> List[Tuple[str, int, int]]:
    """Existing datasets ordered by how well the export's labels fit them:
    ``(name, matching keys, total keys)``. Used to pre-select the likely target."""
    scored = []
    for name in workspace.datasets.keys():
        ds = workspace.datasets.get(name)
        if ds is None:
            continue
        hits, total = 0, 0
        if contents.pose.ok:
            files = {_norm(f) for f in ds.files}
            hits += sum(1 for k in contents.pose_keys if _norm(k) in files)
            total += len(contents.pose_keys)
        if contents.behavior.ok:
            index = _behavior_key_index(ds)
            hits += sum(1 for k in contents.behavior_keys if k in index)
            total += len(contents.behavior_keys)
        scored.append((name, hits, total))
    same_name = contents.dataset_name
    return sorted(scored, key=lambda t: (-t[1], t[0] != same_name, t[0].lower()))


# --------------------------------------------------------------------------- #
# 4. Applying labels to an existing dataset
# --------------------------------------------------------------------------- #


class LabelPolicy(str, Enum):
    ADD = "add"                       # the dataset has no labels of this kind yet
    REPLACE = "replace"               # discard the existing labels, use the imported ones
    MERGE_IMPORTED = "merge_imported"  # combine; where both label a file, imported wins
    MERGE_EXISTING = "merge_existing"  # combine; where both label a file, existing wins
    SKIP = "skip"                     # don't import these labels


def _pose_frame_level(df: pd.DataFrame) -> pd.Index:
    return df.index.get_level_values("frame_index")


def merge_pose(existing: PoseDataset, imported: PoseDataset, imported_wins: bool) -> PoseDataset:
    ex, im = existing.keypoint_data, imported.keypoint_data
    overlap = set(_pose_frame_level(ex)) & set(_pose_frame_level(im))
    if imported_wins:
        ex = ex[~_pose_frame_level(ex).isin(overlap)]
    else:
        im = im[~_pose_frame_level(im).isin(overlap)]
    merged = pd.concat([ex, im])
    return PoseDataset(
        keypoint_data=merged, body_parts=list(existing.body_parts), individuals=list(existing.individuals)
    )


def merge_behavior(existing: BehaviorDataset, imported: BehaviorDataset, imported_wins: bool) -> BehaviorDataset:
    behaviors = list(existing.behaviors) + [b for b in imported.behaviors if b not in existing.behaviors]
    ex, im = existing.bouts, imported.bouts
    overlap = set(ex["video_key"]) & set(im["video_key"])
    # Per video, one side's bouts replace the other's: two people's bouts for the same
    # video would overlap and contradict each other, so we never interleave them.
    if imported_wins:
        ex = ex[~ex["video_key"].isin(overlap)]
    else:
        im = im[~im["video_key"].isin(overlap)]
    bouts = pd.concat([ex, im], ignore_index=True)
    return BehaviorDataset(
        behaviors=behaviors,
        per_frame_probabilities=existing.per_frame_probabilities,
        bouts=bouts,
        pose_run_name=existing.pose_run_name,
    )


class _Backup:
    """Keep a copy of a label folder while it's being rewritten; restore it on failure."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.saved: Optional[Path] = None

    def __enter__(self):
        if self.directory.exists():
            self.saved = self.directory.with_name(self.directory.name + ".import-backup")
            if self.saved.exists():
                shutil.rmtree(self.saved)
            shutil.copytree(self.directory, self.saved)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.saved is None:
            if exc_type is not None and self.directory.exists():
                shutil.rmtree(self.directory, ignore_errors=True)   # a brand-new folder that failed halfway
            return False
        if exc_type is not None:
            shutil.rmtree(self.directory, ignore_errors=True)
            shutil.copytree(self.saved, self.directory)
            logger.error("Label import failed; restored the previous labels in %s", self.directory)
        shutil.rmtree(self.saved, ignore_errors=True)
        return False


def _filter_pose(imported: PoseDataset, report: LabelReport, keep_unmatched: bool) -> PoseDataset:
    df = imported.keypoint_data
    if not keep_unmatched:
        df = df[_pose_frame_level(df).isin(set(report.matched))]
    frames = _pose_frame_level(df).map(lambda k: report.key_map.get(k, k))
    df = df.copy()
    df.index = pd.MultiIndex.from_arrays(
        [frames, df.index.get_level_values("individual_id"), df.index.get_level_values("body_part")],
        names=["frame_index", "individual_id", "body_part"],
    )
    return PoseDataset(keypoint_data=df, body_parts=list(imported.body_parts), individuals=list(imported.individuals))


def _filter_behavior(imported: BehaviorDataset, report: LabelReport, keep_unmatched: bool) -> BehaviorDataset:
    bouts = imported.bouts.copy()
    if not keep_unmatched:
        bouts = bouts[bouts["video_key"].isin(set(report.matched))]
    bouts["video_key"] = bouts["video_key"].map(lambda k: report.key_map.get(k, k))
    return BehaviorDataset(
        behaviors=list(imported.behaviors), per_frame_probabilities=imported.per_frame_probabilities,
        bouts=bouts.reset_index(drop=True), pose_run_name=imported.pose_run_name,
    )


def apply_pose_labels(
    workspace, dataset_name: str, imported_path: Path, policy: LabelPolicy, keep_unmatched: bool = False
) -> dict:
    """Add/replace/merge imported pose labels into ``dataset_name``."""
    if policy == LabelPolicy.SKIP:
        return {"applied": False, "policy": policy.value}
    report = analyze_pose_labels(workspace, dataset_name, imported_path)
    if report.nothing_matches and not keep_unmatched:
        raise BundleImportError(
            f"None of the {report.total} pose labels match a file in '{dataset_name}', "
            "so there's nothing to import. This is probably the wrong dataset."
        )
    imported = _filter_pose(PoseDataset.from_file(Path(imported_path)), report, keep_unmatched)

    target_dir = workspace.pose_labels.base_dir / dataset_name
    target_h5 = target_dir / fmt.LABELS_H5_FILENAME
    existing = PoseDataset.from_file(target_h5) if target_h5.exists() else None

    if existing is None or policy in (LabelPolicy.ADD, LabelPolicy.REPLACE):
        result, mode = imported, "replaced" if existing is not None else "added"
    else:
        if report.problems:
            raise BundleImportError("Can't merge pose labels: " + " ".join(report.problems))
        result = merge_pose(existing, imported, imported_wins=policy == LabelPolicy.MERGE_IMPORTED)
        mode = "merged"

    with _Backup(target_dir):
        workspace.pose_labels.write_poses_file(dataset_name, result, target_h5)
    return {
        "applied": True, "policy": policy.value, "mode": mode,
        "imported_frames": len(imported.frame_indices),
        "dropped_unmatched": 0 if keep_unmatched else len(report.unmatched),
        "overlapping": len(report.overlapping), "total_frames": len(result.frame_indices),
    }


def apply_behavior_labels(
    workspace, dataset_name: str, imported_path: Path, policy: LabelPolicy, keep_unmatched: bool = False
) -> dict:
    """Add/replace/merge imported behavior labels into ``dataset_name``."""
    if policy == LabelPolicy.SKIP:
        return {"applied": False, "policy": policy.value}
    report = analyze_behavior_labels(workspace, dataset_name, imported_path)
    if report.nothing_matches and not keep_unmatched:
        raise BundleImportError(
            f"None of the {report.total} labeled videos match a video in '{dataset_name}', "
            "so there's nothing to import. This is probably the wrong dataset."
        )
    imported = _filter_behavior(BehaviorDataset.from_file(Path(imported_path)), report, keep_unmatched)

    target_dir = workspace.behavior_labels.base_dir / dataset_name
    target_h5 = target_dir / fmt.LABELS_H5_FILENAME
    existing = BehaviorDataset.from_file(target_h5) if target_h5.exists() else None

    if existing is None or policy in (LabelPolicy.ADD, LabelPolicy.REPLACE):
        result, mode = imported, "replaced" if existing is not None else "added"
    else:
        result = merge_behavior(existing, imported, imported_wins=policy == LabelPolicy.MERGE_IMPORTED)
        mode = "merged"

    with _Backup(target_dir):
        result.to_file(target_h5)
    return {
        "applied": True, "policy": policy.value, "mode": mode,
        "imported_videos": len(set(imported.bouts["video_key"])),
        "dropped_unmatched": 0 if keep_unmatched else len(report.unmatched),
        "overlapping": len(report.overlapping), "total_videos": len(set(result.bouts["video_key"])),
    }


# --------------------------------------------------------------------------- #
# 5. Selection, validation, and the import itself
# --------------------------------------------------------------------------- #


@dataclass
class ImportSelection:
    """What the user ticked, plus the answers to any follow-up questions."""
    project: bool = False
    dataset: bool = False               # the dataset info and its videos/frames
    pose: bool = False
    behavior: bool = False
    dataset_name: str = ""              # name for the imported dataset (when ``dataset``)
    media_dir: Optional[Path] = None    # where the media are, when not inside the export
    overwrite_project: bool = False
    overwrite_dataset: bool = False     # also replaces labels installed with it
    # Labels imported without the dataset go onto this existing dataset:
    target_dataset: str = ""
    pose_policy: LabelPolicy = LabelPolicy.ADD
    behavior_policy: LabelPolicy = LabelPolicy.ADD
    keep_unmatched: bool = False

    @property
    def any(self) -> bool:
        return self.project or self.dataset or self.pose or self.behavior

    @property
    def labels_need_target(self) -> bool:
        """Labels are being imported without their dataset, so the user must say which one."""
        return (self.pose or self.behavior) and not self.dataset


def project_relation(workspace, contents: BundleContents) -> str:
    """How the export's project relates to the workspace: 'new', 'identical' or 'different'."""
    proj = contents.project_obj
    if proj is None:
        return "new"
    mine = workspace.projects.get(proj.name)
    if mine is None:
        return "new"
    return "identical" if mine.model_dump() == proj.model_dump() else "different"


def unique_dataset_name(workspace, name: str) -> str:
    existing = set(workspace.datasets.keys())
    if name not in existing:
        return name
    n = 2
    while f"{name}_{n}" in existing:
        n += 1
    return f"{name}_{n}"


def default_selection(workspace, contents: BundleContents) -> ImportSelection:
    """A sensible starting point: tick everything that's importable, except things the
    workspace already has — a re-import most likely wants the *labels*, attached to the
    dataset that's already here."""
    sel = ImportSelection(dataset_name=contents.dataset_name)
    sel.pose, sel.behavior = contents.pose.ok, contents.behavior.ok
    sel.project = contents.project.ok and project_relation(workspace, contents) == "new"
    dataset_exists = bool(contents.dataset_name) and workspace.datasets.get(contents.dataset_name) is not None
    sel.dataset = contents.dataset.ok and not dataset_exists
    if not sel.dataset and (sel.pose or sel.behavior):
        ranked = rank_target_datasets(workspace, contents)
        sel.target_dataset = ranked[0][0] if ranked and ranked[0][1] > 0 else (contents.dataset_name if dataset_exists else "")
    if dataset_exists and contents.dataset.ok:
        sel.dataset_name = unique_dataset_name(workspace, contents.dataset_name)
    sel.media_dir = contents.media_dir
    return sel


def validate_selection(
    workspace, contents: BundleContents, sel: ImportSelection, check_target: bool = True
) -> List[str]:
    """Plain-English reasons the import can't start yet (empty list = good to go).

    ``check_target=False`` skips the "which existing dataset do these labels belong to"
    check, for the import dialog that asks that as a follow-up question."""
    problems: List[str] = []
    if not sel.any:
        return ["Tick at least one thing to import."]
    for flag, part, label in ((sel.project, contents.project, "project"), (sel.dataset, contents.dataset, "dataset"),
                              (sel.pose, contents.pose, "pose labels"), (sel.behavior, contents.behavior, "behavior labels")):
        if flag and not part.ok:
            problems.append(f"The {label} can't be imported: {part.problem or 'not in this export.'}")
    if problems:
        return problems

    if sel.dataset:
        name = sel.dataset_name.strip()
        if not name:
            problems.append("Give the imported dataset a name.")
        elif workspace.datasets.get(name) is not None and not sel.overwrite_dataset:
            problems.append(f"You already have a dataset called '{name}'. Choose another name, or tick 'Replace'.")
        media = sel.media_dir or contents.media_dir
        if media is None:
            problems.append(f"Choose the folder that contains the {contents.media_kind}.")
        elif contents.dataset_obj is not None:
            check, _ = mi.check_media_folder(media, contents.dataset_obj)
            if not check.ok:
                problems.append(check.message)
    elif sel.labels_need_target and check_target:
        if not sel.target_dataset:
            problems.append("Choose which dataset these labels belong to.")
        elif workspace.datasets.get(sel.target_dataset) is None:
            problems.append(f"Dataset '{sel.target_dataset}' isn't in this workspace.")
    return problems


def import_from_bundle(
    workspace,
    contents: BundleContents,
    sel: ImportSelection,
    progress_cb: Optional[ProgressCallback] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> dict:
    """Carry out ``sel``. Callers should have passed :func:`validate_selection`."""
    problems = validate_selection(workspace, contents, sel)
    if problems:
        raise BundleImportError(problems[0])

    def progress(msg: str, pct: int):
        if progress_cb:
            progress_cb(msg, pct)

    result: dict = {"notes": []}

    if sel.project:
        progress("Importing project...", 0)
        installed = mi.install_project(workspace, contents.project_obj, contents.project.path, sel.overwrite_project)
        result["project_name"] = contents.project_obj.name
        result["project_installed"] = installed
        if not installed:
            result["notes"].append(f"Kept your existing project '{contents.project_obj.name}'.")

    target = sel.target_dataset
    if sel.dataset:
        name = sel.dataset_name.strip()
        progress("Preparing dataset...", 2)
        copied, missing = mi.install_dataset(
            workspace, contents.dataset_obj, name, sel.media_dir or contents.media_dir,
            sel.overwrite_dataset, progress_cb, cancel_cb, progress_range=(5, 85),
        )
        result.update(dataset_name=name, media_kind=contents.media_kind, num_media=len(contents.dataset_obj.files),
                      num_media_copied=copied, num_missing=len(missing), missing=missing)
        target = name
        if sel.pose:
            progress("Importing pose labels...", 90)
            ok = mi.install_pose_labels(workspace, name, contents.pose.path, sel.overwrite_dataset)
            result["pose"] = {"applied": ok, "mode": "added" if ok else "kept"}
            if not ok:
                result["notes"].append("Kept the pose labels already saved under that name.")
        if sel.behavior:
            progress("Importing behavior labels...", 96)
            ok = mi.install_behavior_labels(workspace, name, contents.behavior.path, sel.overwrite_dataset)
            result["behavior"] = {"applied": ok, "mode": "added" if ok else "kept"}
            if not ok:
                result["notes"].append("Kept the behavior labels already saved under that name.")
    else:
        result["dataset_name"] = target
        if sel.pose:
            progress("Adding pose labels...", 60)
            result["pose"] = apply_pose_labels(workspace, target, contents.pose.path, sel.pose_policy, sel.keep_unmatched)
        if sel.behavior:
            progress("Adding behavior labels...", 85)
            result["behavior"] = apply_behavior_labels(workspace, target, contents.behavior.path, sel.behavior_policy, sel.keep_unmatched)

    progress("Import complete.", 100)
    return result
