"""Import a full-WHISKER "Whisker bundle" zip, keeping only its ROI tracking content and any projects.

Ported from full WHISKER's ``whisker/core/study/bundle.py`` (read-only reference repo), trimmed to what this app
needs: a dataset's manifest/media, its ``workflows/roi_tracking/labels/<dataset>/`` and
``workflows/roi_tracking/predictions/<run>/<dataset>/`` content, and any ``projects/<name>.json`` -- a project must
exist and be selected for the ROI Tracking tab to work (decision made with the user, 2026-10-25). Pose, behavior,
animal-detection and model content present in the bundle is read from the manifest only to report that it was
ignored; it is never extracted.

This is a SEPARATE importer from this app's own ``core/manual_import.py``, which reads this app's own
folder-format bundle (written by this app's own Export Annotations action). Full WHISKER's bundle is a different,
zip-based format; this module reads only that format.

Verified compatible: the labeler's own ``Dataset`` model (``core/study/dataset.py``) has the identical field set
(name/type/base_data_path/files/multi_arena/arena_boxes) as full WHISKER's, so a manifest exported by full WHISKER
loads with this app's own ``Dataset.from_json`` unchanged.

Qt-free, like the source module, so it can run off the GUI thread.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable, Optional

from .study.dataset import Dataset

BUNDLE_FORMAT = "whisker-bundle"
# Highest bundle format version this importer understands. Mirrors full WHISKER's BUNDLE_FORMAT_VERSION.
BUNDLE_FORMAT_VERSION = 2
BUNDLE_MANIFEST_NAME = "whisker_bundle.json"

ROI_WORKFLOW = "roi_tracking"
MEDIA_LAYOUT_EXTERNAL = "external"

_COPY_CHUNK = 1024 * 1024
_LONG_PREFIX = "\\\\?\\"
_LONG_UNC_PREFIX = "\\\\?\\UNC\\"
_WIN_MAX_PATH = 259
_INVALID_NAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

ProgressFn = Callable[[str, int], None]
CancelFn = Callable[[], bool]


def _noop_progress(message: str, percent: int) -> None:
    return None


def _never_cancelled() -> bool:
    return False


class BundleError(Exception):
    """A bundle could not be read or imported (message is user-facing)."""


class BundleCancelled(BundleError):
    """The operation was cancelled by the user."""


# --------------------------------------------------------------------------- #
# Small helpers (ported from full WHISKER's bundle.py)                        #
# --------------------------------------------------------------------------- #

def _norm_rel(rel: str) -> str:
    return str(rel).replace("\\", "/")


def _fs(path) -> Path:
    r"""A path that is safe to use even beyond Windows' 260-character limit (adds ``\\?\`` on Windows; a no-op
    elsewhere). Every filesystem call this module makes goes through it."""
    p = Path(path)
    if os.name != "nt":
        return p
    text = os.path.abspath(str(p))
    if text.startswith(_LONG_PREFIX):
        return Path(text)
    if text.startswith("\\\\"):
        return Path(_LONG_UNC_PREFIX + text[2:])
    return Path(_LONG_PREFIX + text)


def _unfs(path) -> Path:
    """Inverse of :func:`_fs`: the plain form shown to users."""
    text = str(path)
    if os.name == "nt":
        if text.startswith(_LONG_UNC_PREFIX):
            text = "\\\\" + text[len(_LONG_UNC_PREFIX):]
        elif text.startswith(_LONG_PREFIX):
            text = text[len(_LONG_PREFIX):]
    return Path(text)


def _validate_component(name: str, what: str) -> None:
    """Reject names that would escape their folder or be unusable on Windows."""
    if not name or name in (".", "..") or name != name.strip():
        raise BundleError(f"Invalid {what} name in bundle: {name!r}")
    if _INVALID_NAME_CHARS.search(name):
        raise BundleError(f"Invalid {what} name in bundle (illegal characters): {name!r}")
    if name.endswith("."):
        raise BundleError(f"Invalid {what} name in bundle: {name!r}")


def _safe_join(base: Path, arcname: str) -> Path:
    """``base / arcname`` for an archive member, refusing anything that escapes ``base``."""
    posix = PurePosixPath(arcname)
    if (not posix.parts or posix.is_absolute() or any(part in ("..", "") for part in posix.parts)
            or ":" in posix.parts[0]):
        raise BundleError(f"Unsafe path in bundle: {arcname!r}")
    target = base.joinpath(*posix.parts)
    resolved_base = os.path.normcase(os.path.abspath(base))
    resolved = os.path.normcase(os.path.abspath(target))
    if os.path.commonpath([resolved_base, resolved]) != resolved_base:
        raise BundleError(f"Unsafe path in bundle: {arcname!r}")
    return target


def _dataset_dir(workspace, name: str) -> Path:
    return workspace.base_dir / "datasets" / name


def _roi_dir(workspace, kind: str) -> Path:
    return workspace.base_dir / "workflows" / ROI_WORKFLOW / kind


# --------------------------------------------------------------------------- #
# Reading a bundle's manifest                                                 #
# --------------------------------------------------------------------------- #

@dataclass
class BundleDatasetInfo:
    name: str
    type: str
    media_layout: str
    original_base_data_path: str
    missing_media: list
    label_workflows: list        # every workflow with labels for this dataset, ROI or not (for the ignored report)
    media_included: bool = True


@dataclass
class BundlePredictionInfo:
    workflow: str
    run_name: str
    datasets: list


@dataclass
class BundleInfo:
    path: Path
    format_version: int
    datasets: list
    prediction_runs: list
    projects: list
    models: list = field(default_factory=list)   # workflow names with bundled models (never imported here)

    def dataset(self, name: str) -> Optional[BundleDatasetInfo]:
        return next((d for d in self.datasets if d.name == name), None)

    def roi_run_names(self) -> list:
        return sorted({p.run_name for p in self.prediction_runs if p.workflow == ROI_WORKFLOW})

    def ignored_content_summary(self) -> list:
        """One line per kind of non-ROI content present in the bundle, so the importer can "ignore it and say
        so" (brief section 5.2) instead of silently dropping it."""
        lines = []
        other_label_wfs = {wf for d in self.datasets for wf in d.label_workflows if wf != ROI_WORKFLOW}
        if other_label_wfs:
            lines.append(f"{', '.join(sorted(w.replace('_', ' ') for w in other_label_wfs))} labels")
        other_pred_wfs = {p.workflow for p in self.prediction_runs if p.workflow != ROI_WORKFLOW}
        if other_pred_wfs:
            lines.append(f"{', '.join(sorted(w.replace('_', ' ') for w in other_pred_wfs))} predictions")
        if self.models:
            lines.append(f"{', '.join(sorted(w.replace('_', ' ') for w in set(self.models)))} models")
        return lines


def read_bundle_info(bundle_path: Path) -> BundleInfo:
    """Read a bundle's manifest (fast: does not unpack or verify any media)."""
    bundle_path = Path(bundle_path)
    if not bundle_path.is_file():
        raise BundleError(f"Bundle not found: {bundle_path}")
    try:
        with zipfile.ZipFile(bundle_path) as zf:
            try:
                raw = zf.read(BUNDLE_MANIFEST_NAME)
            except KeyError:
                raise BundleError(
                    "This zip is not a Whisker bundle (it has no "
                    f"{BUNDLE_MANIFEST_NAME}). Bundles are created in full WHISKER with "
                    "'Export Whisker Bundle...'."
                ) from None
    except zipfile.BadZipFile as e:
        raise BundleError(f"This file is not a valid zip archive: {e}") from e

    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise BundleError(f"The bundle's manifest is corrupt: {e}") from e

    if manifest.get("format") != BUNDLE_FORMAT:
        raise BundleError("This zip is not a Whisker bundle (unrecognised manifest format).")
    version = manifest.get("format_version")
    if not isinstance(version, int) or version > BUNDLE_FORMAT_VERSION:
        raise BundleError(
            f"This bundle uses format version {version}, which is newer than this copy of the labeler "
            f"understands (version {BUNDLE_FORMAT_VERSION}). Update the labeler to import it."
        )

    try:
        return BundleInfo(
            path=bundle_path,
            format_version=version,
            datasets=[
                BundleDatasetInfo(
                    name=d["name"],
                    type=d["type"],
                    media_layout=d.get("media_layout", MEDIA_LAYOUT_EXTERNAL),
                    original_base_data_path=d.get("original_base_data_path", ""),
                    missing_media=list(d.get("missing_media", [])),
                    label_workflows=list(d.get("label_workflows", [])),
                    media_included=bool(d.get("media_included", True)),
                )
                for d in manifest.get("datasets", [])
            ],
            prediction_runs=[
                BundlePredictionInfo(p["workflow"], p["run_name"], list(p.get("datasets", [])))
                for p in manifest.get("prediction_runs", [])
            ],
            projects=list(manifest.get("projects", [])),
            models=[m["workflow"] for m in manifest.get("models", [])],
        )
    except (KeyError, TypeError, ValueError) as e:
        raise BundleError(f"The bundle's manifest is incomplete or corrupt: {e!r}") from e


# --------------------------------------------------------------------------- #
# Conflicts (for the import dialog's preview)                                 #
# --------------------------------------------------------------------------- #

@dataclass
class ImportConflicts:
    datasets: list = field(default_factory=list)
    projects_different: list = field(default_factory=list)   # exist, but content differs
    projects_identical: list = field(default_factory=list)   # exist and already match


def find_import_conflicts(workspace, info: BundleInfo) -> ImportConflicts:
    conflicts = ImportConflicts()
    registered = set(workspace.datasets.keys())
    for d in info.datasets:
        if d.name in registered or _dataset_dir(workspace, d.name).exists():
            conflicts.datasets.append(d.name)
    if info.projects:
        with zipfile.ZipFile(info.path) as zf:
            for pname in info.projects:
                existing = workspace.base_dir / "projects" / f"{pname}.json"
                if not existing.is_file():
                    continue
                try:
                    incoming = json.loads(zf.read(f"projects/{pname}.json").decode("utf-8"))
                    current = json.loads(existing.read_text(encoding="utf-8"))
                except (KeyError, ValueError, OSError):
                    conflicts.projects_different.append(pname)
                    continue
                (conflicts.projects_identical if incoming == current else conflicts.projects_different).append(pname)
    return conflicts


# --------------------------------------------------------------------------- #
# Import                                                                      #
# --------------------------------------------------------------------------- #

@dataclass
class ImportOptions:
    dataset_names: list
    # Parent folder for datasets whose media lives outside the workspace; each dataset gets its own subfolder.
    media_dir: Optional[Path] = None
    overwrite_datasets: set = field(default_factory=set)
    prediction_runs: Optional[list] = None     # ROI run names; None = every ROI run in the bundle
    project_names: Optional[list] = None       # None = every project in the bundle
    overwrite_projects: set = field(default_factory=set)
    # For datasets exported without their videos: the folder that already holds them.
    existing_media_dirs: dict = field(default_factory=dict)


@dataclass
class ImportReport:
    imported_datasets: list = field(default_factory=list)
    replaced_datasets: list = field(default_factory=list)
    skipped_datasets: dict = field(default_factory=dict)
    label_sets: list = field(default_factory=list)            # dataset names with ROI labels imported
    prediction_sets: list = field(default_factory=list)       # (run, dataset)
    imported_projects: list = field(default_factory=list)
    skipped_projects: dict = field(default_factory=dict)
    media_locations: dict = field(default_factory=dict)
    ignored_content: list = field(default_factory=list)       # non-ROI content the bundle had, not imported
    warnings: list = field(default_factory=list)

    def summary(self) -> str:
        lines: list = []
        if self.imported_datasets:
            lines.append(f"Imported {len(self.imported_datasets)} dataset(s): {', '.join(self.imported_datasets)}")
        if self.replaced_datasets:
            lines.append(f"  (replaced existing: {', '.join(self.replaced_datasets)})")
        if self.label_sets:
            lines.append(f"ROI labels: {len(self.label_sets)} dataset(s)")
        if self.prediction_sets:
            runs = {run for run, _ds in self.prediction_sets}
            lines.append(f"ROI predictions: {len(runs)} run(s)")
        if self.imported_projects:
            lines.append(f"Projects: {', '.join(self.imported_projects)}")
        for name, reason in self.skipped_datasets.items():
            lines.append(f"Skipped dataset '{name}': {reason}")
        for name, reason in self.skipped_projects.items():
            lines.append(f"Skipped project '{name}': {reason}")
        if self.ignored_content:
            lines.append(f"Ignored (not an ROI Tracking bundle for this app): {', '.join(self.ignored_content)}")
        for w in self.warnings:
            lines.append(f"Warning: {w}")
        return "\n".join(lines) if lines else "Nothing was imported."


def _existing_media_root(ds: BundleDatasetInfo, options: ImportOptions) -> Path:
    """Where the videos of a media-less dataset already are (chosen folder, else its original one)."""
    chosen = options.existing_media_dirs.get(ds.name)
    candidate = Path(chosen) if chosen else (
        Path(ds.original_base_data_path) if ds.original_base_data_path else None
    )
    if candidate is None or not _fs(candidate).is_dir():
        raise BundleError(
            f"'{ds.name}' was exported without its videos. Choose the folder where its videos already are."
        )
    return candidate


def _unique_dir(parent: Path, name: str) -> Path:
    """``parent/name``, or ``parent/name (2)`` etc. if that is taken and non-empty."""
    candidate = parent / name
    n = 2
    while candidate.exists() and any(candidate.iterdir()):
        candidate = parent / f"{name} ({n})"
        n += 1
    return candidate


def _extract_member(zf: zipfile.ZipFile, member: zipfile.ZipInfo, target: Path,
                     tick: Callable[[int], None], is_cancelled: CancelFn) -> None:
    """Stream one archive member to ``target``. ``zipfile`` verifies the CRC at EOF."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(member) as src, open(target, "wb") as out:
        while True:
            chunk = src.read(_COPY_CHUNK)
            if not chunk:
                break
            if is_cancelled():
                raise BundleCancelled("Import cancelled.")
            out.write(chunk)
            tick(len(chunk))


def _link_or_copy(src: Path, dst: Path, report: ImportReport, counters: dict) -> None:
    """Point ``dst`` at ``src``: symlink, else hard link, else copy (Windows without symlink rights)."""
    src_io, dst_io = _fs(src), _fs(dst)
    dst_io.parent.mkdir(parents=True, exist_ok=True)
    if dst_io.is_symlink() or dst_io.exists():
        dst_io.unlink()
    try:
        os.symlink(src, dst_io)
        return
    except OSError:
        pass
    try:
        os.link(src_io, dst_io)
        counters["hardlinked"] += 1
        return
    except OSError:
        pass
    shutil.copy2(src_io, dst_io)
    counters["copied"] += 1


def import_bundle(
    workspace,
    bundle_path: Path,
    options: ImportOptions,
    progress: ProgressFn = _noop_progress,
    is_cancelled: CancelFn = _never_cancelled,
) -> ImportReport:
    """Unpack a Whisker bundle's ROI content (+ projects) into ``workspace``. See :func:`_import_bundle` for the
    guarantees. Callers should call :func:`rescan_workspace` afterwards."""
    try:
        return _import_bundle(workspace, bundle_path, options, progress, is_cancelled)
    except zipfile.BadZipFile as e:
        raise BundleError(
            f"The bundle is corrupt or incomplete ({e}). Nothing was imported; try copying the file again."
        ) from e


def _import_bundle(
    workspace, bundle_path: Path, options: ImportOptions, progress: ProgressFn, is_cancelled: CancelFn,
) -> ImportReport:
    """Unpack a bundle's ROI content into ``workspace``.

    Everything destined for the workspace is first extracted into a temporary staging folder and only moved into
    place once the whole archive has been read and verified, so a corrupt, truncated or cancelled bundle leaves the
    existing workspace untouched. Existing datasets are never modified unless listed in
    ``options.overwrite_datasets``. Pose, behavior, animal-detection and model content in the bundle is left in the
    zip entirely -- never staged, never extracted -- and reported as ignored.
    """
    info = read_bundle_info(bundle_path)
    report = ImportReport()
    report.ignored_content = info.ignored_content_summary()
    ws_base = workspace.base_dir

    selected = [n for n in options.dataset_names if info.dataset(n) is not None]
    unknown = [n for n in options.dataset_names if info.dataset(n) is None]
    for n in unknown:
        report.skipped_datasets[n] = "not in this bundle"

    # ---- decide what actually gets imported --------------------------------
    to_import: list = []
    replacing: set = set()
    existing_roots: dict = {}
    for name in selected:
        ds_info = info.dataset(name)
        _validate_component(name, "dataset")
        if _dataset_dir(workspace, name).exists():
            if name in options.overwrite_datasets:
                replacing.add(name)
            else:
                report.skipped_datasets[name] = "already exists in this workspace"
                continue
        if not ds_info.media_included:
            existing_roots[name] = _existing_media_root(ds_info, options)
        elif ds_info.media_layout == MEDIA_LAYOUT_EXTERNAL and options.media_dir is None:
            raise BundleError(
                f"Choose a folder to store the media for '{name}' (it is kept outside the workspace)."
            )
        to_import.append(ds_info)
    imported_names = {d.name for d in to_import}

    wanted_runs = set(info.roi_run_names()) if options.prediction_runs is None else set(options.prediction_runs)
    wanted_projects = info.projects if options.project_names is None else [
        p for p in options.project_names if p in info.projects
    ]

    with zipfile.ZipFile(bundle_path) as zf:
        members = {m.filename: m for m in zf.infolist() if not m.is_dir()}
        for arc in members:
            _safe_join(ws_base, arc)  # refuse a hostile archive up front

        staging = _fs(ws_base / f".bundle_import_{uuid.uuid4().hex[:8]}")
        media_roots: dict = {}
        created_external: list = []
        jobs: list = []
        media_expected: dict = {}

        for ds in to_import:
            prefix = f"datasets/{ds.name}/"
            if ds.name in existing_roots:
                media_roots[ds.name] = existing_roots[ds.name]
                try:
                    staged_manifest = Dataset.from_json(
                        zf.read(f"datasets/{ds.name}/manifest.json").decode("utf-8")
                    )
                except KeyError:
                    raise BundleError(f"The bundle has no manifest for dataset '{ds.name}'.") from None
                media_expected[ds.name] = [_norm_rel(f) for f in staged_manifest.files]
            elif ds.media_layout == MEDIA_LAYOUT_EXTERNAL:
                folder = _INVALID_NAME_CHARS.sub("_", ds.name)
                media_root = _unique_dir(Path(options.media_dir), folder)
                media_roots[ds.name] = media_root
            for arc, m in members.items():
                if not arc.startswith(prefix):
                    continue
                rest = arc[len(prefix):]
                if rest == "manifest.json":
                    jobs.append((m, staging / "datasets" / ds.name / "manifest.json"))
                elif rest.startswith("extras/"):
                    jobs.append((m, staging / "datasets" / ds.name / rest[len("extras/"):]))
                elif rest.startswith("media/"):
                    rel = rest[len("media/"):]
                    media_expected.setdefault(ds.name, []).append(rel)
                    if ds.media_layout == MEDIA_LAYOUT_EXTERNAL:
                        jobs.append((m, _safe_join(_fs(media_roots[ds.name]), rel)))
                    else:
                        jobs.append((m, staging / "datasets" / ds.name / "data" / rel))

            # ROI labels only -- pose/behavior/other labels in the bundle are never staged or extracted.
            lp = f"workflows/{ROI_WORKFLOW}/labels/{ds.name}/"
            hits = [(a, m) for a, m in members.items() if a.startswith(lp)]
            for a, m in hits:
                jobs.append((m, staging / a))
            if hits:
                report.label_sets.append(ds.name)

        for pred in info.prediction_runs:
            if pred.workflow != ROI_WORKFLOW or pred.run_name not in wanted_runs:
                continue
            _validate_component(pred.run_name, "prediction run")
            for ds_name in pred.datasets:
                if ds_name not in imported_names:
                    continue
                pp = f"workflows/{ROI_WORKFLOW}/predictions/{pred.run_name}/{ds_name}/"
                hits = [(a, m) for a, m in members.items() if a.startswith(pp)]
                for a, m in hits:
                    jobs.append((m, staging / a))
                if hits:
                    report.prediction_sets.append((pred.run_name, ds_name))

        for pname in wanted_projects:
            arc = f"projects/{pname}.json"
            if arc in members:
                jobs.append((members[arc], staging / arc))

        if os.name == "nt":
            staging_extra = len(str(_unfs(staging))) - len(str(ws_base))
            too_long = 0
            for _member, target in jobs:
                length = len(str(_unfs(target)))
                if str(target).startswith(str(staging)):
                    length -= staging_extra
                too_long += length > _WIN_MAX_PATH
            if too_long:
                report.warnings.append(
                    f"{too_long} file path(s) are longer than Windows' 260-character limit and may not "
                    "open in some programs. Choose a shorter workspace or media folder path, or enable "
                    "Windows long-path support."
                )

        # ---- extract ---------------------------------------------------
        total = max(sum(m.file_size for m, _ in jobs), 1)
        done = 0

        def tick(n: int) -> None:
            nonlocal done
            done += n
            progress("Extracting files", min(95, done * 100 // total))

        try:
            staging.mkdir(parents=True, exist_ok=True)
            for ds_name, root in media_roots.items():
                if ds_name in existing_roots:
                    continue
                _fs(root).mkdir(parents=True, exist_ok=True)
                created_external.append(root)
            for member, target in jobs:
                if is_cancelled():
                    raise BundleCancelled("Import cancelled.")
                _extract_member(zf, member, target, tick, is_cancelled)

            # ---- commit: everything is extracted and CRC-verified -------
            progress("Finalizing", 96)
            counters = {"hardlinked": 0, "copied": 0}
            for ds in to_import:
                _commit_dataset(workspace, ds, staging, media_roots.get(ds.name),
                                 media_expected.get(ds.name, []), replacing, report, counters)
            _commit_predictions_and_projects(workspace, staging, report, options, replacing)
            if counters["hardlinked"] or counters["copied"]:
                report.warnings.append(
                    "Symbolic links could not be created on this machine, so "
                    f"{counters['hardlinked']} file(s) were hard-linked and "
                    f"{counters['copied']} file(s) were copied into the workspace instead."
                )
        except BaseException:
            for root in created_external:
                shutil.rmtree(_fs(root), ignore_errors=True)
            raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    for ds in to_import:
        if ds.missing_media:
            report.warnings.append(
                f"'{ds.name}': {len(ds.missing_media)} media file(s) were not available when the bundle "
                "was made and are not included."
            )
    progress("Import complete", 100)
    return report


def _move_into_place(src: Path, dst: Path) -> None:
    src, dst = _fs(src), _fs(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    shutil.move(str(src), str(dst))


def _commit_dataset(workspace, ds: BundleDatasetInfo, staging: Path, media_root: Optional[Path],
                     media_rels: list, replacing: set, report: ImportReport, counters: dict) -> None:
    name = ds.name
    ds_dir = _dataset_dir(workspace, name)
    staged_ds = staging / "datasets" / name

    if name in replacing:
        # Replace this dataset's own artefacts only; other datasets are never touched.
        shutil.rmtree(_fs(ds_dir), ignore_errors=True)
        if (staging / "workflows" / ROI_WORKFLOW / "labels" / name).is_dir():
            shutil.rmtree(_fs(_roi_dir(workspace, "labels") / name), ignore_errors=True)
        report.replaced_datasets.append(name)

    manifest = Dataset.from_json((staged_ds / "manifest.json").read_text(encoding="utf-8"))
    if ds.media_layout == MEDIA_LAYOUT_EXTERNAL:
        assert media_root is not None
        manifest.base_data_path = str(media_root)
        data_dir = staged_ds / "data"
        absent: list = []
        for rel in media_rels:
            target = media_root.joinpath(*PurePosixPath(rel).parts)
            if not ds.media_included and not _fs(target).is_file():
                absent.append(rel)
                continue
            _link_or_copy(target, data_dir.joinpath(*PurePosixPath(rel).parts), report, counters)
        if absent:
            report.warnings.append(
                f"'{name}': {len(absent)} of {len(media_rels)} video(s) were not found in "
                f"{media_root} and are not linked (first missing: {absent[0]})."
            )
        report.media_locations[name] = str(media_root)
    else:
        manifest.base_data_path = str(ds_dir / "data")
        (staged_ds / "data").mkdir(parents=True, exist_ok=True)
        report.media_locations[name] = str(ds_dir / "data")
    (staged_ds / "manifest.json").write_text(manifest.model_dump_json(indent=4), encoding="utf-8")

    _move_into_place(staged_ds, ds_dir)

    staged_labels = staging / "workflows" / ROI_WORKFLOW / "labels" / name
    if staged_labels.is_dir():
        target = _roi_dir(workspace, "labels") / name
        if target.exists():
            report.warnings.append(f"Kept existing ROI labels for '{name}' (a label set was already present).")
            report.label_sets = [x for x in report.label_sets if x != name]
        else:
            _move_into_place(staged_labels, target)

    report.imported_datasets.append(name)


def _commit_predictions_and_projects(workspace, staging: Path, report: ImportReport, options: ImportOptions,
                                      replacing: set) -> None:
    for run, ds_name in list(report.prediction_sets):
        staged = staging / "workflows" / ROI_WORKFLOW / "predictions" / run / ds_name
        if not staged.is_dir():
            continue
        target = _roi_dir(workspace, "predictions") / run / ds_name
        if target.exists() and ds_name not in replacing:
            report.warnings.append(f"Kept existing ROI predictions for '{ds_name}' in run '{run}'.")
            report.prediction_sets.remove((run, ds_name))
            continue
        _move_into_place(staged, target)

    staged_projects = staging / "projects"
    if staged_projects.is_dir():
        for pfile in sorted(staged_projects.glob("*.json")):
            pname = pfile.stem
            target = workspace.base_dir / "projects" / pfile.name
            if target.exists():
                try:
                    same = (json.loads(target.read_text(encoding="utf-8"))
                            == json.loads(pfile.read_text(encoding="utf-8")))
                except (ValueError, OSError):
                    same = False
                if same:
                    report.skipped_projects[pname] = "already present and identical"
                    continue
                if pname not in options.overwrite_projects:
                    report.skipped_projects[pname] = "a different project with this name exists (kept yours)"
                    continue
            _move_into_place(pfile, target)
            report.imported_projects.append(pname)


def rescan_workspace(workspace) -> None:
    """Re-read datasets, projects, labels and predictions from disk after an import."""
    workspace.scan_datasets()
    workspace.scan_projects()
    workspace.scan_labels()
    workspace.scan_models()
    workspace.scan_predictions()
