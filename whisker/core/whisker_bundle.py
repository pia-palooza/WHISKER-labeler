"""Whisker bundles: the ZIP file WHISKER and whisker-labeler use to move datasets between them.

A bundle is an ordinary ZIP64 file. Media and label files are stored as they exist in the workspace,
uncompressed and unconverted; only the small JSON files are deflated::

    whisker_bundle.json                                   manifest (read first; nothing is extracted to read it)
    datasets/<dataset>/manifest.json                      the Dataset manifest; its ``files`` list is authoritative
    datasets/<dataset>/media/<relative path>              frames / videos / images, forward slashes, may nest
    workflows/<workflow>/labels/<dataset>/...             pose labels.h5 + metadata.json, behavior labels.h5, ...
    workflows/<workflow>/predictions/<run>/<dataset>/...  prediction runs
    projects/<name>.json                                  project definitions

The workspace uses the same layout (``datasets/<name>/manifest.json``, ``workflows/...``, ``projects/...``),
so most files go in and out unchanged. Only two things are translated: a dataset's ``base_data_path``
(where its media live on *this* machine) and where media are placed.

Importing (:func:`import_bundle`) treats the archive as untrusted: every entry name is checked before anything is
read, files are extracted to a staging folder with the zip library verifying each checksum, and only when the
whole archive has been read cleanly are the pieces moved into place. A corrupt, truncated or cancelled import
leaves the workspace as it was.

Exporting (:func:`export_bundle`) writes to a temporary file, checks it by reading it back, and only then gives
it its real name.

Pure filesystem work: no Qt, no mutation of the in-memory workspace, so callers can run it on a worker thread and
rescan the workspace afterwards.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import stat
import time
import uuid
import zlib
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from whisker.core.study.dataset import Dataset, DatasetType
from whisker.core.study.dataset_operations import _link_or_copy

logger = logging.getLogger(__name__)

FORMAT = "whisker-bundle"
FORMAT_VERSION = 1
MANIFEST_NAME = "whisker_bundle.json"
WHISKER_VERSION = "0.1.0"      # what this tool writes into ``whisker_version``; WHISKER itself writes its own

DATASETS_DIR = "datasets"
PROJECTS_DIR = "projects"
WORKFLOWS_DIR = "workflows"
MEDIA_DIR = "media"
DATASET_MANIFEST = "manifest.json"

_CHUNK = 1024 * 1024
_STAGING_PREFIX = ".wbi_"        # short: staging adds to every path, and Windows stops at 259 characters
_MAX_PATH = 259

ProgressCallback = Callable[[str, int], None]
CancelCallback = Callable[[], bool]


class BundleError(Exception):
    """A bundle can't be read, imported or written; the message says why, in plain words."""


def format_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{n} B"


# --------------------------------------------------------------------------- #
# Untrusted names
# --------------------------------------------------------------------------- #


def check_entry_name(name: str) -> None:
    """Raise :class:`BundleError` unless ``name`` is a plain relative forward-slash path inside the archive.

    Rejects absolute paths, ``..``, empty or ``.`` components, backslashes, drive letters (any ``:``, which on
    Windows also means alternate data streams) and control characters."""
    if not name:
        raise BundleError("The bundle contains a file with no name.")
    shown = name if len(name) <= 80 else name[:77] + "..."
    if any(ord(c) < 32 for c in name):
        raise BundleError(f"The bundle contains a file with an unsafe name ({shown!r}); it won't be opened.")
    if "\\" in name or ":" in name or name.startswith("/"):
        raise BundleError(f"The bundle contains a file with an absolute or unsafe path ({shown!r}); it won't be opened.")
    parts = name[:-1].split("/") if name.endswith("/") else name.split("/")
    if any(p in ("", ".", "..") for p in parts):
        raise BundleError(f"The bundle contains a file with an unsafe path ({shown!r}); it won't be opened.")


def check_component(name: str, what: str) -> str:
    """A dataset / run / workflow / project name from the manifest must be one safe path component."""
    if not isinstance(name, str) or not name or "/" in name:
        raise BundleError(f"The bundle has a {what} with an invalid name ({name!r}).")
    check_entry_name(name)
    if name != name.strip() or name.endswith("."):
        raise BundleError(f"The bundle has a {what} whose name ({name!r}) can't be used as a folder name.")
    return name


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    return stat.S_ISLNK(info.external_attr >> 16)


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


@dataclass
class DatasetEntry:
    """A dataset as the manifest describes it. ``original_base_data_path`` is a record of where the files lived
    on the exporter's machine and is never used."""
    name: str
    type_name: str = ""
    file_count: int = 0
    media_bytes: int = 0
    media_layout: str = ""
    missing_media: List[str] = field(default_factory=list)
    label_workflows: List[str] = field(default_factory=list)
    multi_arena: bool = False
    original_base_data_path: str = ""

    @property
    def type(self) -> Optional[DatasetType]:
        try:
            return DatasetType(self.type_name)
        except ValueError:
            return None

    @property
    def keeps_media_in_workspace(self) -> bool:
        return self.type == DatasetType.FRAME_SUBSET


@dataclass
class PredictionRun:
    workflow: str
    run_name: str
    datasets: List[str] = field(default_factory=list)


@dataclass
class BundleManifest:
    format_version: int
    whisker_version: str = ""
    created_at: str = ""
    source_workspace: str = ""
    datasets: List[DatasetEntry] = field(default_factory=list)
    prediction_runs: List[PredictionRun] = field(default_factory=list)
    projects: List[str] = field(default_factory=list)
    file_count: int = 0
    total_bytes: int = 0

    def dataset(self, name: str) -> Optional[DatasetEntry]:
        return next((d for d in self.datasets if d.name == name), None)


def _as_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_str_list(value) -> List[str]:
    return [str(v) for v in value] if isinstance(value, list) else []


def parse_manifest(raw: object) -> BundleManifest:
    """Read ``whisker_bundle.json``. Fields this version doesn't know are ignored; a newer format version is refused."""
    if not isinstance(raw, dict) or raw.get("format") != FORMAT:
        raise BundleError("This isn't a Whisker bundle (its whisker_bundle.json doesn't say 'whisker-bundle').")
    version = _as_int(raw.get("format_version"), -1)
    if version < 1:
        raise BundleError("This Whisker bundle has no valid format version.")
    if version > FORMAT_VERSION:
        raise BundleError(
            f"This bundle uses format version {version}, which is newer than this version of the labeler "
            f"understands ({FORMAT_VERSION}). Please update the labeler."
        )
    m = BundleManifest(
        format_version=version,
        whisker_version=str(raw.get("whisker_version", "")),
        created_at=str(raw.get("created_at", "")),
        source_workspace=str(raw.get("source_workspace", "")),
        projects=[check_component(p, "project") for p in _as_str_list(raw.get("projects"))],
        file_count=_as_int(raw.get("file_count")),
        total_bytes=_as_int(raw.get("total_bytes")),
    )
    for d in raw.get("datasets") or []:
        if not isinstance(d, dict):
            continue
        m.datasets.append(DatasetEntry(
            name=check_component(d.get("name"), "dataset"),
            type_name=str(d.get("type", "")),
            file_count=_as_int(d.get("file_count")),
            media_bytes=_as_int(d.get("media_bytes")),
            media_layout=str(d.get("media_layout", "")),
            missing_media=_as_str_list(d.get("missing_media")),
            label_workflows=[check_component(w, "workflow") for w in _as_str_list(d.get("label_workflows"))],
            multi_arena=bool(d.get("multi_arena", False)),
            original_base_data_path=str(d.get("original_base_data_path", "")),
        ))
    for r in raw.get("prediction_runs") or []:
        if isinstance(r, dict):
            m.prediction_runs.append(PredictionRun(
                workflow=check_component(r.get("workflow"), "workflow"),
                run_name=check_component(r.get("run_name"), "prediction run"),
                datasets=[check_component(n, "dataset") for n in _as_str_list(r.get("datasets"))],
            ))
    return m


class WhiskerBundle:
    """An opened bundle. Opening reads only the file listing and ``whisker_bundle.json``; nothing is extracted.

        with WhiskerBundle.open(path) as bundle:
            bundle.manifest.datasets ...
    """

    def __init__(self, path: Path, zf: zipfile.ZipFile, manifest: BundleManifest):
        self.path = path
        self.zf = zf
        self.manifest = manifest
        self._infos: Dict[str, zipfile.ZipInfo] = {i.filename: i for i in zf.infolist() if not i.is_dir()}

    @classmethod
    def open(cls, path) -> "WhiskerBundle":
        path = Path(path)
        try:
            zf = zipfile.ZipFile(path, "r")          # ZIP64 and random access: only the central directory is read
        except FileNotFoundError:
            raise BundleError(f"'{path}' doesn't exist.") from None
        except zipfile.BadZipFile:
            raise BundleError(f"'{path.name}' isn't a readable zip file. It may be damaged or incomplete.") from None
        except OSError as e:
            raise BundleError(f"Couldn't open '{path.name}': {e}") from None
        try:
            for info in zf.infolist():
                check_entry_name(info.filename)
                if info.flag_bits & 0x1:
                    raise BundleError("This bundle is password-protected, which isn't supported.")
                if _is_symlink(info):
                    raise BundleError(f"The bundle contains a link ({info.filename!r}), which isn't allowed.")
            if MANIFEST_NAME not in zf.namelist():
                raise BundleError("This zip isn't a Whisker bundle: it has no whisker_bundle.json.")
            try:
                raw = json.loads(zf.read(MANIFEST_NAME).decode("utf-8"))
            except (ValueError, zipfile.BadZipFile) as e:
                raise BundleError(f"whisker_bundle.json can't be read ({e}). The bundle may be damaged.") from None
            return cls(path, zf, parse_manifest(raw))
        except BaseException:
            zf.close()
            raise

    def close(self) -> None:
        self.zf.close()

    def __enter__(self) -> "WhiskerBundle":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- what is inside ---------------------------------------------------------------------------
    def has(self, name: str) -> bool:
        return name in self._infos

    def size(self, name: str) -> int:
        return self._infos[name].file_size

    def read_json(self, name: str):
        try:
            return json.loads(self.zf.read(name).decode("utf-8"))
        except KeyError:
            raise BundleError(f"The bundle is missing {name}.") from None
        except (ValueError, zipfile.BadZipFile) as e:
            raise BundleError(f"{name} in the bundle can't be read ({e}). The bundle may be damaged.") from None

    def dataset(self, name: str) -> Dataset:
        """The dataset's own manifest: ``files`` is the authoritative list of its media."""
        raw = self.read_json(f"{DATASETS_DIR}/{name}/{DATASET_MANIFEST}")
        try:
            return Dataset.from_json(json.dumps(raw))
        except Exception as e:
            raise BundleError(f"The manifest of dataset '{name}' can't be read ({e}).") from None

    def media_entry(self, dataset: str, rel: str) -> str:
        return f"{DATASETS_DIR}/{dataset}/{MEDIA_DIR}/{rel}"

    def media_files(self, name: str, dataset: Dataset) -> Tuple[List[str], List[str]]:
        """``(present, absent)`` relative paths from the dataset's ``files`` list, checked against the archive."""
        present, absent = [], []
        for rel in dataset.files:
            rel = str(rel).replace("\\", "/")
            check_entry_name(rel)
            (present if self.has(self.media_entry(name, rel)) else absent).append(rel)
        return present, absent

    def label_entries(self, dataset: str) -> Dict[str, List[str]]:
        """``{workflow: [entry names]}`` for the labels of ``dataset`` present in the archive."""
        found: Dict[str, List[str]] = {}
        for name in self._infos:
            parts = name.split("/")
            if len(parts) >= 5 and parts[0] == WORKFLOWS_DIR and parts[2] == "labels" and parts[3] == dataset:
                found.setdefault(parts[1], []).append(name)
        return found

    def prediction_entries(self, workflow: str, run: str, dataset: str) -> List[str]:
        prefix = f"{WORKFLOWS_DIR}/{workflow}/predictions/{run}/{dataset}/"
        return [n for n in self._infos if n.startswith(prefix)]

    def project_entry(self, name: str) -> str:
        return f"{PROJECTS_DIR}/{name}.json"


# --------------------------------------------------------------------------- #
# Import: what is in the bundle, and what the workspace already has
# --------------------------------------------------------------------------- #


def dataset_key(name: str) -> str:
    return f"dataset:{name}"


def labels_key(name: str) -> str:
    return f"labels:{name}"


def run_key(workflow: str, run: str) -> str:
    return f"run:{workflow}/{run}"


def project_key(name: str) -> str:
    return f"project:{name}"


@dataclass
class PlanItem:
    """One thing the user can bring in from the bundle."""
    key: str
    kind: str                       # 'dataset' | 'labels' | 'run' | 'project'
    title: str
    detail: str = ""
    bytes: int = 0
    exists: bool = False            # the workspace already has it; bringing it in replaces it
    problem: str = ""               # why it can't be imported at all
    requires_media_folder: bool = False

    @property
    def importable(self) -> bool:
        return not self.problem


def _workspace_has_dataset(workspace, name: str) -> bool:
    return (Path(workspace.datasets.base_dir) / name).exists()


def _labels_dir(workspace, workflow: str, dataset: str) -> Path:
    return Path(workspace.base_dir) / WORKFLOWS_DIR / workflow / "labels" / dataset


def _run_dir(workspace, workflow: str, run: str, dataset: str) -> Path:
    return Path(workspace.base_dir) / WORKFLOWS_DIR / workflow / "predictions" / run / dataset


def _bad_folder_name(name: str) -> str:
    if os.name == "nt" and any(c in name for c in '<>"|?*'):
        return "Its name has characters Windows can't use in a folder name."
    return ""


def plan_import(workspace, bundle: WhiskerBundle) -> List[PlanItem]:
    """List what the bundle offers, marking what the workspace already has. Pure reads."""
    m = bundle.manifest
    items: List[PlanItem] = []
    for d in m.datasets:
        kind = d.type
        problem = ""
        if kind is None:
            problem = f"Unsupported dataset type '{d.type_name}'."
        elif not bundle.has(f"{DATASETS_DIR}/{d.name}/{DATASET_MANIFEST}"):
            problem = "The bundle has no manifest for this dataset."
        else:
            problem = _bad_folder_name(d.name)
        media = "frames" if kind in (DatasetType.FRAME_SUBSET, DatasetType.IMAGE_COLLECTION) else "videos"
        detail = f"{d.type_name.replace('_', ' ').title() if d.type_name else 'Dataset'}, {d.file_count} {media}"
        if d.missing_media:
            detail += f" ({len(d.missing_media)} not in the bundle)"
        items.append(PlanItem(
            key=dataset_key(d.name), kind="dataset", title=d.name, detail=detail, bytes=d.media_bytes,
            exists=_workspace_has_dataset(workspace, d.name), problem=problem,
            requires_media_folder=bool(kind is not None and not d.keeps_media_in_workspace),
        ))
    for d in m.datasets:
        by_workflow = bundle.label_entries(d.name)
        if not by_workflow:
            continue
        items.append(PlanItem(
            key=labels_key(d.name), kind="labels", title=f"Labels for {d.name}",
            detail=", ".join(w.replace("_", " ") for w in sorted(by_workflow)),
            bytes=sum(bundle.size(n) for names in by_workflow.values() for n in names),
            exists=any(_labels_dir(workspace, w, d.name).exists() for w in by_workflow),
        ))
    for r in m.prediction_runs:
        names = [n for ds in r.datasets for n in bundle.prediction_entries(r.workflow, r.run_name, ds)]
        if not names:
            continue
        items.append(PlanItem(
            key=run_key(r.workflow, r.run_name), kind="run", title=f"Predictions: {r.run_name}",
            detail=f"{r.workflow.replace('_', ' ')}, {len(r.datasets)} dataset(s)",
            bytes=sum(bundle.size(n) for n in names),
            exists=any(_run_dir(workspace, r.workflow, r.run_name, ds).exists() for ds in r.datasets),
        ))
    for p in m.projects:
        entry = bundle.project_entry(p)
        items.append(PlanItem(
            key=project_key(p), kind="project", title=f"Project: {p}",
            bytes=bundle.size(entry) if bundle.has(entry) else 0,
            exists=(Path(workspace.projects.base_dir) / f"{p}.json").exists(),
            problem="" if bundle.has(entry) else "The bundle has no file for this project.",
        ))
    return items


def default_choices(items: Iterable[PlanItem]) -> Set[str]:
    """New things are ticked; things the workspace already has are left alone unless the user ticks them."""
    return {i.key for i in items if i.importable and not i.exists}


def validate_choices(workspace, bundle: WhiskerBundle, chosen: Set[str], media_root: Optional[Path]) -> List[str]:
    """Reasons the import can't go ahead, in plain words (empty means it can)."""
    problems: List[str] = []
    items = {i.key: i for i in plan_import(workspace, bundle)}
    unknown = [k for k in chosen if k not in items]
    if unknown:
        problems.append(f"The bundle doesn't contain {unknown[0]}.")
    picked = [items[k] for k in chosen if k in items]
    if not picked:
        problems.append("Choose something to import.")
    for item in picked:
        if item.problem:
            problems.append(f"{item.title}: {item.problem}")
    importing = {i.title for i in picked if i.kind == "dataset"}
    for item in picked:
        if item.kind == "labels":
            ds = item.key.split(":", 1)[1]
            if ds not in importing and not _workspace_has_dataset(workspace, ds):
                problems.append(f"'{ds}' isn't in this workspace, so its labels have nowhere to go. Import the dataset too.")
    needs_folder = [i for i in picked if i.requires_media_folder]
    if needs_folder:
        if media_root is None:
            problems.append("Choose the folder where the videos and images should be stored.")
        elif not Path(media_root).is_dir():
            problems.append(f"The media folder '{media_root}' doesn't exist.")
        else:
            for item in needs_folder:
                target = Path(media_root) / item.title
                if target.exists():                 # never replaced, even when the dataset is: it may be someone's originals
                    problems.append(f"'{target}' already exists, and existing media outside the workspace are never "
                                    "overwritten. Choose another folder for the media.")
    return problems


# --------------------------------------------------------------------------- #
# Import: stage, verify, then move into place
# --------------------------------------------------------------------------- #


def _windows_long_paths_enabled() -> bool:
    """False on Windows unless the system allows paths longer than 259 characters (always True elsewhere)."""
    if os.name != "nt":
        return True
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem") as key:
            return bool(winreg.QueryValueEx(key, "LongPathsEnabled")[0])
    except OSError:
        return False


def _replace(src: Path, dst: Path, attempts: int = 12) -> None:
    """``os.replace``, retried for a few seconds on a permission error. Windows can't rename a folder while another
    program (antivirus, the search indexer) still has one of its freshly written files open; that passes quickly."""
    delay = 0.05
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 1.6, 0.6)


class _Journal:
    """Moves into place that can be undone, so a failure part-way leaves things as they were."""

    def __init__(self, backup_dir: Path):
        self.backup_dir = backup_dir
        self._undo: List[Callable[[], None]] = []
        self._n = 0

    def place(self, staged: Path, final: Path, replace: bool = True) -> None:
        """Move ``staged`` to ``final``. Anything already at ``final`` is kept aside until the import succeeds, then
        deleted with the staging folder. With ``replace=False`` something already there is an error instead."""
        final.parent.mkdir(parents=True, exist_ok=True)
        backup: Optional[Path] = None
        if (final.exists() or final.is_symlink()) and not replace:
            raise BundleError(f"'{final}' already exists, and existing media are never overwritten.")
        if final.exists() or final.is_symlink():
            self._n += 1
            backup = self.backup_dir / str(self._n)
            backup.parent.mkdir(parents=True, exist_ok=True)
            _replace(final, backup)
        try:
            _replace(staged, final)
        except BaseException:
            if backup is not None:
                _replace(backup, final)
            raise

        def undo(final=final, backup=backup):
            _remove(final)
            if backup is not None:
                _replace(backup, final)
        self._undo.append(undo)

    def rollback(self) -> None:
        for undo in reversed(self._undo):
            try:
                undo()
            except Exception:                               # keep undoing the rest
                logger.exception("Could not undo part of a failed import")
        self._undo.clear()


def _remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path, ignore_errors=True)


class _Extractor:
    """Copies archive members to disk in chunks, checking the running total and cancellation."""

    def __init__(self, bundle: WhiskerBundle, total: int, progress_cb, cancel_cb, lo: int = 3, hi: int = 90):
        self.bundle, self.total = bundle, max(1, total)
        self.progress_cb, self.cancel_cb, self.lo, self.hi = progress_cb, cancel_cb, lo, hi
        self.done = 0
        self._last_pct = -1

    def extract(self, entry: str, dest: Path, label: str) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.bundle.zf.open(entry) as src, open(dest, "wb") as out:
                while True:
                    if self.cancel_cb and self.cancel_cb():
                        raise BundleError("Import cancelled.")
                    chunk = src.read(_CHUNK)
                    if not chunk:
                        break
                    out.write(chunk)
                    self.done += len(chunk)
                    self._report(label)
        except (zipfile.BadZipFile, EOFError, zlib.error) as e:   # a failed checksum is raised at the end of the read
            raise BundleError(f"'{entry.split('/')[-1]}' is damaged in the bundle ({e}). Nothing was imported.") from None
        except OSError as e:
            raise BundleError(f"Couldn't write '{dest.name}': {e}. Nothing was imported.") from None

    def _report(self, label: str) -> None:
        if not self.progress_cb:
            return
        pct = self.lo + int((self.hi - self.lo) * min(1.0, self.done / self.total))
        if pct != self._last_pct:
            self._last_pct = pct
            self.progress_cb(label, pct)


def import_bundle(
    workspace,
    bundle: WhiskerBundle,
    chosen: Set[str],
    media_root: Optional[Path] = None,
    progress_cb: Optional[ProgressCallback] = None,
    cancel_cb: Optional[CancelCallback] = None,
) -> dict:
    """Bring the ``chosen`` items (:class:`PlanItem` keys) from ``bundle`` into ``workspace``.

    Things the workspace already has and that are chosen are replaced (the caller asks first). Frame subsets are
    placed inside the workspace; videos and image collections are extracted to ``<media_root>/<dataset>`` and
    linked from the workspace, and each dataset's ``base_data_path`` is rewritten to match. Returns a summary;
    the caller rescans the workspace afterwards.
    """
    problems = validate_choices(workspace, bundle, chosen, media_root)
    if problems:
        raise BundleError(problems[0])
    items = {i.key: i for i in plan_import(workspace, bundle)}
    m = bundle.manifest
    ws_root = Path(workspace.base_dir)
    result: dict = {"datasets": [], "labels": [], "runs": [], "projects": [], "notes": [], "missing": {}}

    def progress(msg: str, pct: int):
        if progress_cb:
            progress_cb(msg, pct)

    # Decide every file to extract before touching the disk.
    stage = ws_root / f"{_STAGING_PREFIX}{uuid.uuid4().hex[:6]}"
    media_stage = Path(media_root) / f"{_STAGING_PREFIX}{uuid.uuid4().hex[:6]}" if media_root else None
    plan_files: List[Tuple[str, Path, str]] = []                      # (entry, staged path, label)
    dataset_moves: List[Tuple[DatasetEntry, Dataset, Path, Path]] = []  # entry, dataset, staged manifest dir, media dir
    placements: List[Tuple[Path, Path]] = []                          # (staged, final) for everything but datasets

    for d in m.datasets:
        if dataset_key(d.name) not in chosen:
            continue
        ds = bundle.dataset(d.name)
        present, absent = bundle.media_files(d.name, ds)
        if absent:
            result["missing"][d.name] = len(absent)
        if d.keeps_media_in_workspace:
            media_final_dir = ws_root / DATASETS_DIR / d.name / "data"
            media_target = stage / DATASETS_DIR / d.name / "data"
        else:
            media_final_dir = Path(media_root) / d.name
            media_target = media_stage / d.name
        for rel in present:
            plan_files.append((bundle.media_entry(d.name, rel), media_target / rel, f"{d.name}: {rel}"))
        dataset_moves.append((d, ds, media_target, media_final_dir))

    for d in m.datasets:
        if labels_key(d.name) not in chosen:
            continue
        for workflow, names in bundle.label_entries(d.name).items():
            root = f"{WORKFLOWS_DIR}/{workflow}/labels/{d.name}/"
            for entry in names:
                plan_files.append((entry, stage / WORKFLOWS_DIR / workflow / "labels" / d.name / entry[len(root):], f"{d.name}: labels"))
            placements.append((stage / WORKFLOWS_DIR / workflow / "labels" / d.name, _labels_dir(workspace, workflow, d.name)))
        result["labels"].append(d.name)

    importing_names = {d.name for d, *_ in dataset_moves}
    for r in m.prediction_runs:
        if run_key(r.workflow, r.run_name) not in chosen:
            continue
        for ds_name in r.datasets:
            if ds_name not in importing_names and not _workspace_has_dataset(workspace, ds_name):
                result["notes"].append(f"Skipped predictions of '{r.run_name}' for '{ds_name}': that dataset isn't in this workspace.")
                continue
            names = bundle.prediction_entries(r.workflow, r.run_name, ds_name)
            if not names:
                continue
            root = f"{WORKFLOWS_DIR}/{r.workflow}/predictions/{r.run_name}/{ds_name}/"
            staged_dir = stage / WORKFLOWS_DIR / r.workflow / "predictions" / r.run_name / ds_name
            for entry in names:
                plan_files.append((entry, staged_dir / entry[len(root):], f"{r.run_name}: predictions"))
            placements.append((staged_dir, _run_dir(workspace, r.workflow, r.run_name, ds_name)))
        result["runs"].append(f"{r.workflow}/{r.run_name}")

    for p in m.projects:
        if project_key(p) in chosen:
            plan_files.append((bundle.project_entry(p), stage / PROJECTS_DIR / f"{p}.json", f"Project {p}"))
            placements.append((stage / PROJECTS_DIR / f"{p}.json", Path(workspace.projects.base_dir) / f"{p}.json"))
            result["projects"].append(p)

    if os.name == "nt" and not _windows_long_paths_enabled():
        longest = max((dest for _e, dest, _l in plan_files), key=lambda d: len(str(d)), default=None)
        if longest is not None and len(str(longest)) > _MAX_PATH:
            raise BundleError(
                f"'{longest.name}' would need a path of {len(str(longest))} characters, and Windows allows {_MAX_PATH}. "
                "Move the workspace (or the media folder) closer to the top of the drive, or turn on Windows long paths, "
                "and try again. Nothing was changed."
            )

    total = sum(bundle.size(entry) for entry, _dst, _lbl in plan_files)
    journal = _Journal(stage / "_previous")
    try:
        stage.mkdir(parents=True)
        if media_stage is not None and any(not d.keeps_media_in_workspace for d, *_ in dataset_moves):
            media_stage.mkdir(parents=True)

        # 1. Extract and verify everything. Nothing in the workspace has changed yet.
        progress("Reading the bundle...", 1)
        extractor = _Extractor(bundle, total, progress_cb, cancel_cb)
        for entry, dest, label in plan_files:
            extractor.extract(entry, dest, label)

        # 2. Dataset manifests, with each dataset's media location rewritten for this machine.
        progress("Preparing datasets...", 92)
        for d, ds, media_target, media_final_dir in dataset_moves:
            manifest_dir = stage / DATASETS_DIR / d.name
            manifest_dir.mkdir(parents=True, exist_ok=True)
            imported = ds.model_copy(update={"name": d.name, "base_data_path": str(media_final_dir.resolve())})
            (manifest_dir / DATASET_MANIFEST).write_text(imported.model_dump_json(indent=4), encoding="utf-8")

        # 3. Move into place. Media of videos / image collections first, so the workspace links can point at them.
        progress("Putting everything in place...", 95)
        for d, ds, media_target, media_final_dir in dataset_moves:
            if not d.keeps_media_in_workspace:
                journal.place(media_target if media_target.exists() else _empty_dir(media_target), media_final_dir, replace=False)
                data_dir = stage / DATASETS_DIR / d.name / "data"
                for rel in bundle.media_files(d.name, ds)[0]:
                    link = data_dir / rel
                    link.parent.mkdir(parents=True, exist_ok=True)
                    _link_or_copy((media_final_dir / rel).resolve(), link)
            elif not media_target.exists():
                media_target.mkdir(parents=True, exist_ok=True)         # a frame subset with no frames still has a data folder
            journal.place(stage / DATASETS_DIR / d.name, Path(workspace.datasets.base_dir) / d.name)
            result["datasets"].append(d.name)
        for staged, final in placements:
            if staged.exists():
                journal.place(staged, final)
    except BaseException as e:
        journal.rollback()
        shutil.rmtree(stage, ignore_errors=True)
        if media_stage is not None:
            shutil.rmtree(media_stage, ignore_errors=True)
        if isinstance(e, OSError):
            raise BundleError(f"The import couldn't be finished ({e}). Another program may be using the files. "
                              "Nothing was changed; try again.") from e
        raise
    shutil.rmtree(stage, ignore_errors=True)                     # also removes what was replaced
    if media_stage is not None:
        shutil.rmtree(media_stage, ignore_errors=True)

    for name, n in result["missing"].items():
        result["notes"].append(f"{n} file(s) of '{name}' weren't in the bundle (they were unreachable when it was made).")
    progress("Import complete.", 100)
    return result


def _empty_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #


@dataclass
class PlanGroup:
    """One line of "what will be in the bundle": a kind of thing, and the files that make it up."""
    dataset: str                    # '' for things that belong to the bundle rather than to a dataset
    kind: str                       # 'manifest' | 'media' | 'labels' | 'predictions' | 'project'
    title: str
    files: List[Tuple[str, int]] = field(default_factory=list)      # (path inside the group, bytes)
    note: str = ""                  # why it is empty, or what is left out
    problem: bool = False           # the note is something the user should act on

    @property
    def bytes(self) -> int:
        return sum(n for _f, n in self.files)


@dataclass
class ExportPlan:
    """Everything an export will write, worked out up front so the dialog can show it and the job can run it."""
    datasets: List[Dataset]
    projects: List[str]
    prediction_runs: List[PredictionRun]
    entries: List[Tuple[str, Path, int]] = field(default_factory=list)      # (archive name, source file, compress type)
    sizes: Dict[str, int] = field(default_factory=dict)                     # archive name -> bytes, measured once
    dataset_info: List[dict] = field(default_factory=list)
    missing: Dict[str, List[str]] = field(default_factory=dict)             # dataset -> files that couldn't be found
    problems: List[str] = field(default_factory=list)

    def add(self, name: str, src: Path, compress: int, size: Optional[int] = None) -> None:
        self.entries.append((name, src, compress))
        self.sizes[name] = src.stat().st_size if size is None else size

    @property
    def total_bytes(self) -> int:
        return sum(self.sizes.values())

    @property
    def media_missing(self) -> int:
        return sum(len(v) for v in self.missing.values())

    def groups(self) -> List[PlanGroup]:
        """What is going into the bundle, grouped the way a person thinks of it: for each dataset its manifest, media,
        labels (per workflow) and prediction results (per run), then the projects. A dataset with no labels says so."""
        out: List[PlanGroup] = []
        for ds in self.datasets:
            word = {DatasetType.FRAME_SUBSET: "frames", DatasetType.IMAGE_COLLECTION: "images"}.get(ds.type, "videos")
            base = f"{DATASETS_DIR}/{ds.name}/"
            media_prefix = base + MEDIA_DIR + "/"
            media = [(n[len(media_prefix):], self.sizes[n]) for n, _p, _c in self.entries if n.startswith(media_prefix)]
            missing = self.missing.get(ds.name, [])
            manifest = base + DATASET_MANIFEST
            out.append(PlanGroup(ds.name, "manifest", "Dataset manifest",
                                 [(DATASET_MANIFEST, self.sizes[manifest])] if manifest in self.sizes else []))
            note = f"{len(missing)} not found on disk, so not included" if missing else ("" if media else "no files")
            out.append(PlanGroup(ds.name, "media", f"Media — {len(media)} {word}", media, note, problem=bool(missing)))
            labels: Dict[str, List[Tuple[str, int]]] = {}
            predictions: Dict[Tuple[str, str], List[Tuple[str, int]]] = {}
            for n, _p, _c in self.entries:
                parts = n.split("/")
                if len(parts) >= 5 and parts[0] == WORKFLOWS_DIR and parts[2] == "labels" and parts[3] == ds.name:
                    labels.setdefault(parts[1], []).append(("/".join(parts[4:]), self.sizes[n]))
                elif len(parts) >= 6 and parts[0] == WORKFLOWS_DIR and parts[2] == "predictions" and parts[4] == ds.name:
                    predictions.setdefault((parts[1], parts[3]), []).append(("/".join(parts[5:]), self.sizes[n]))
            if not labels:
                out.append(PlanGroup(ds.name, "labels", "Labels", [], "none for this dataset"))
            for workflow in sorted(labels):
                out.append(PlanGroup(ds.name, "labels", f"Labels — {workflow.replace('_', ' ')}", labels[workflow]))
            for workflow, run in sorted(predictions):
                out.append(PlanGroup(ds.name, "predictions", f"Prediction results — {run}", predictions[(workflow, run)],
                                     workflow.replace("_", " ")))
        for n, _p, _c in self.entries:
            if n.startswith(PROJECTS_DIR + "/"):
                out.append(PlanGroup("", "project", f"Project — {Path(n).stem}", [(Path(n).name, self.sizes[n])]))
        return out


def list_prediction_runs(workspace, dataset_names: Iterable[str]) -> List[PredictionRun]:
    """Prediction runs in the workspace that have results for any of ``dataset_names``."""
    wanted = set(dataset_names)
    runs: Dict[Tuple[str, str], PredictionRun] = {}
    wf_root = Path(workspace.base_dir) / WORKFLOWS_DIR
    if not wf_root.is_dir():
        return []
    for wf in sorted(p for p in wf_root.iterdir() if p.is_dir()):
        pred_root = wf / "predictions"
        if not pred_root.is_dir():
            continue
        for run in sorted(p for p in pred_root.iterdir() if p.is_dir()):
            names = sorted(d.name for d in run.iterdir() if d.is_dir() and d.name in wanted and any(d.rglob("*")))
            if names:
                runs[(wf.name, run.name)] = PredictionRun(wf.name, run.name, names)
    return list(runs.values())


def _files_under(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file()) if root.is_dir() else []


def _media_source(workspace, dataset: Dataset, rel: str) -> Optional[Path]:
    """Where a dataset's media file really is. The workspace's own ``data`` folder is tried first because a
    manifest's ``base_data_path`` can be stale (a dataset moved from another machine)."""
    for base in (Path(workspace.datasets.base_dir) / dataset.name / "data", Path(dataset.base_data_path)):
        candidate = base / rel
        if candidate.is_file():
            return candidate
    return None


def build_export_plan(
    workspace,
    dataset_names: Sequence[str],
    project_names: Sequence[str] = (),
    run_keys: Iterable[Tuple[str, str]] = (),
) -> ExportPlan:
    """Work out what a bundle of ``dataset_names`` (with the given projects and prediction runs) contains. Pure reads."""
    if not dataset_names:
        raise BundleError("Choose at least one dataset to export.")
    ws_root = Path(workspace.base_dir)
    datasets: List[Dataset] = []
    for name in dataset_names:
        ds = workspace.datasets.get(name)
        if ds is None:
            raise BundleError(f"Dataset '{name}' isn't in this workspace.")
        datasets.append(ds)
    chosen_runs = set(run_keys)
    runs = [r for r in list_prediction_runs(workspace, dataset_names) if (r.workflow, r.run_name) in chosen_runs]
    plan = ExportPlan(datasets=datasets, projects=list(project_names), prediction_runs=runs)
    STORED, DEFLATED = zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED

    for ds in datasets:
        manifest_path = Path(workspace.datasets.base_dir) / ds.name / DATASET_MANIFEST
        if not manifest_path.is_file():
            raise BundleError(f"Dataset '{ds.name}' has no manifest.json on disk; can't export it.")
        plan.add(f"{DATASETS_DIR}/{ds.name}/{DATASET_MANIFEST}", manifest_path, DEFLATED)
        media_bytes, missing = 0, []
        for rel in ds.files:
            rel = str(rel).replace("\\", "/")
            src = _media_source(workspace, ds, rel)
            if src is None:
                missing.append(rel)
                continue
            size = src.stat().st_size
            plan.add(f"{DATASETS_DIR}/{ds.name}/{MEDIA_DIR}/{rel}", src, STORED, size)
            media_bytes += size
        label_workflows = []
        wf_root = ws_root / WORKFLOWS_DIR
        for wf in sorted(p for p in wf_root.iterdir() if p.is_dir()) if wf_root.is_dir() else []:
            files = _files_under(wf / "labels" / ds.name)
            if files:
                label_workflows.append(wf.name)
            for f in files:
                rel = f.relative_to(wf / "labels" / ds.name).as_posix()
                plan.add(f"{WORKFLOWS_DIR}/{wf.name}/labels/{ds.name}/{rel}", f, DEFLATED if f.suffix == ".json" else STORED)
        plan.missing[ds.name] = missing
        plan.dataset_info.append({
            "name": ds.name,
            "type": ds.type.value,
            "file_count": len(ds.files),
            "media_layout": "workspace" if ds.type == DatasetType.FRAME_SUBSET else "external",
            "original_base_data_path": str(ds.base_data_path),
            "media_bytes": media_bytes,
            "missing_media": missing,
            "label_workflows": label_workflows,
            "multi_arena": ds.is_multi_arena,
        })

    for run in runs:
        for ds_name in run.datasets:
            base = ws_root / WORKFLOWS_DIR / run.workflow / "predictions" / run.run_name / ds_name
            for f in _files_under(base):
                rel = f.relative_to(base).as_posix()
                plan.add(f"{WORKFLOWS_DIR}/{run.workflow}/predictions/{run.run_name}/{ds_name}/{rel}", f,
                         DEFLATED if f.suffix == ".json" else STORED)
    for p in project_names:
        src = Path(workspace.projects.base_dir) / f"{p}.json"
        if not src.is_file():
            raise BundleError(f"Project '{p}' isn't in this workspace.")
        plan.add(f"{PROJECTS_DIR}/{p}.json", src, DEFLATED)
    for name, _p, _c in plan.entries:
        check_entry_name(name)
    return plan


def _manifest_json(plan: ExportPlan, workspace) -> bytes:
    manifest = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "whisker_version": WHISKER_VERSION,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_workspace": Path(workspace.base_dir).name,
        "datasets": plan.dataset_info,
        "prediction_runs": [{"workflow": r.workflow, "run_name": r.run_name, "datasets": r.datasets} for r in plan.prediction_runs],
        "projects": plan.projects,
        "file_count": len(plan.entries),
        "total_bytes": plan.total_bytes,
    }
    return json.dumps(manifest, indent=2).encode("utf-8")


def export_bundle(
    workspace,
    plan: ExportPlan,
    dest: Path,
    overwrite: bool = False,
    progress_cb: Optional[ProgressCallback] = None,
    cancel_cb: Optional[CancelCallback] = None,
    verify: bool = True,
) -> dict:
    """Write ``plan`` as a Whisker bundle at ``dest`` (a ``.zip`` path).

    The file is written next to its final name and renamed only once complete (and, with ``verify``, read back
    and checked), so a cancelled or failed export never leaves a partial bundle at ``dest``."""
    dest = Path(dest)
    if dest.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists: {dest}")
    if not dest.parent.is_dir():
        raise BundleError(f"The folder '{dest.parent}' doesn't exist.")
    needed = plan.total_bytes
    try:
        free = shutil.disk_usage(dest.parent).free
    except OSError:
        free = None
    if free is not None and free < needed + 20 * 1024 * 1024:
        raise BundleError(f"There isn't enough free space in {dest.parent}: the bundle needs about {needed / 1e6:.0f} MB "
                          f"but only {free / 1e6:.0f} MB is free.")

    part = dest.with_name(dest.name + ".part")
    total = max(1, needed)
    done = 0

    def progress(msg: str, pct: int):
        if progress_cb:
            progress_cb(msg, max(0, min(100, pct)))

    try:
        progress("Writing the bundle...", 0)
        with zipfile.ZipFile(part, "w", allowZip64=True) as zf:
            zf.writestr(zipfile.ZipInfo(MANIFEST_NAME, date_time=_now_tuple()), _manifest_json(plan, workspace),
                        compress_type=zipfile.ZIP_DEFLATED)
            for name, src, compress in plan.entries:
                if cancel_cb and cancel_cb():
                    raise BundleError("Export cancelled.")
                info = zipfile.ZipInfo.from_file(src, name, strict_timestamps=False)
                info.compress_type = compress
                with open(src, "rb") as f, zf.open(info, "w", force_zip64=info.file_size > 0xFFFFFFFF - 1) as out:
                    while True:
                        if cancel_cb and cancel_cb():
                            raise BundleError("Export cancelled.")
                        chunk = f.read(_CHUNK)
                        if not chunk:
                            break
                        out.write(chunk)
                        done += len(chunk)
                        progress(name.split("/")[-1], int(90 * done / total))
        problems: List[str] = []
        if verify:
            progress("Checking the finished bundle...", 92)
            problems = verify_bundle(part, cancel_cb)
        os.replace(part, dest)
    except BaseException:
        part.unlink(missing_ok=True)                            # never leave a partial bundle behind
        raise
    progress("Export complete.", 100)
    return {
        "path": str(dest),
        "datasets": [d.name for d in plan.datasets],
        "num_files": len(plan.entries),
        "bytes": needed,
        "missing": {k: v for k, v in plan.missing.items() if v},
        "num_missing": plan.media_missing,
        "problems": problems,
    }


def _now_tuple():
    n = datetime.now()
    return (max(n.year, 1980), n.month, n.day, n.hour, n.minute, n.second)


def verify_bundle(path: Path, cancel_cb: Optional[CancelCallback] = None) -> List[str]:
    """Read a finished bundle back the way the importer will: the manifest, every entry name and every checksum.
    Returns what's wrong, in plain words (empty means it is sound)."""
    problems: List[str] = []
    try:
        with WhiskerBundle.open(path) as bundle:
            for d in bundle.manifest.datasets:
                if not bundle.has(f"{DATASETS_DIR}/{d.name}/{DATASET_MANIFEST}"):
                    problems.append(f"Dataset '{d.name}' has no manifest in the bundle.")
                    continue
                present, absent = bundle.media_files(d.name, bundle.dataset(d.name))
                unlisted = set(absent) - set(d.missing_media)
                if unlisted:
                    problems.append(f"{len(unlisted)} file(s) of '{d.name}' are listed but not in the bundle.")
            for info in bundle.zf.infolist():
                if cancel_cb and cancel_cb():
                    raise BundleError("Export cancelled.")
                with bundle.zf.open(info) as f:               # reading to the end verifies the stored checksum
                    while f.read(_CHUNK):
                        pass
    except (zipfile.BadZipFile, EOFError, zlib.error) as e:
        problems.append(f"A file in the bundle is damaged ({e}).")
    except BundleError as e:
        if "cancel" in str(e).lower():
            raise
        problems.append(str(e))
    return problems
