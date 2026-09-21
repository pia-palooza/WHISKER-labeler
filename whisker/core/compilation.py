"""Compilations: several datasets exported into one package, and imported together.

A compilation is a folder::

    <name>/
        compilation_info.json     # machine-readable list of the datasets inside
        README.txt
        <dataset one>/            # a complete, standard export (see whisker.core.bundle)
            export_info.json ...
        <dataset two>/
            ...

Every inner folder is an ordinary export in the same format a single-dataset export
produces, so anything that reads those (full WHISKER, or importing one dataset on its own)
reads these unchanged. Each keeps its own copy of its project (a few hundred bytes) so it
stays self-contained. The compilation adds only the top-level description.

Importing a compilation applies the single-export import (:mod:`whisker.core.bundle_import`)
to each dataset the user ticks. One dataset failing never stops the others; each project is
installed once even if several datasets share it.

Pure filesystem work — no Qt — so it can run on a worker thread.
"""

from __future__ import annotations

import copy
import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from whisker.core import bundle as fmt
from whisker.core import bundle_import as bi
from whisker.core import manual_import as mi
from whisker.core.bundle_import import BundleContents, BundleImportError, ImportSelection, LabelPolicy

logger = logging.getLogger(__name__)

COMPILATION_FORMAT_VERSION = "1.0"
COMPILATION_INFO_FILENAME = fmt.COMPILATION_INFO_FILENAME
README_FILENAME = fmt.README_FILENAME

ProgressCallback = Callable[[str, int], None]

_INVALID_NAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED_NAMES = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #


@dataclass
class CompilationItem:
    """One dataset going into a compilation, and which parts of it."""
    dataset_name: str
    project_name: str
    include_media: bool = True
    include_pose: bool = True
    include_behavior: bool = True


def is_valid_compilation_name(name: str) -> bool:
    name = name.strip()
    return bool(name) and not _INVALID_NAME_CHARS.search(name) and name.strip(" .") == name \
        and name.lower() not in _RESERVED_NAMES


def safe_folder_name(name: str, taken: set) -> str:
    """A folder name for ``name`` that is legal on Windows/macOS and not in ``taken``
    (compared case-insensitively). Dataset names can contain characters folders can't."""
    base = _INVALID_NAME_CHARS.sub("_", name).strip(" .") or "dataset"
    if base.lower() in _RESERVED_NAMES:
        base += "_"
    candidate, n = base, 2
    while candidate.lower() in taken:
        candidate = f"{base}_{n}"
        n += 1
    taken.add(candidate.lower())
    return candidate


def guess_project(workspace, dataset_name: str, preferred: Optional[str] = None) -> Optional[str]:
    """The project a dataset was most likely labeled under: one that covers its pose labels'
    body parts/identities and its behavior labels' behaviors; failing that ``preferred`` (the
    active project), failing that the first project."""
    names = sorted(workspace.projects.keys())
    if not names:
        return None
    body_parts, identities, behaviors = set(), set(), set()
    try:
        if workspace.pose_labels.has_pose_labels(dataset_name):
            meta = workspace.pose_labels._get_pose_label_metadata(dataset_name) or {}
            body_parts, identities = set(meta.get("body_parts", [])), set(meta.get("individuals", []))
        if workspace.behavior_labels.has_behavior_labels(dataset_name):
            from whisker.services.behavior_classification.public.data_structures import BehaviorDataset
            behaviors = set(BehaviorDataset.from_file(
                workspace.behavior_labels.base_dir / dataset_name / fmt.LABELS_H5_FILENAME).behaviors)
    except Exception:            # a damaged label file shouldn't stop the export dialog opening
        logger.warning("Could not read labels of '%s' to guess its project", dataset_name, exc_info=True)

    def covers(name: str) -> bool:
        p = workspace.projects.get(name)
        return bool(p) and body_parts <= set(p.body_parts) and identities <= set(p.identities) \
            and behaviors <= set(p.behaviors)

    if (body_parts or identities or behaviors):
        if preferred in names and covers(preferred):
            return preferred
        fitting = [n for n in names if covers(n)]
        if fitting:
            return fitting[0]
    return preferred if preferred in names else names[0]


def build_plans(workspace, items: List[CompilationItem]) -> List["fmt.BundleExportPlan"]:
    """Resolve every item into an export plan (pure reads). Raises BundleError with the
    dataset's name if one can't be exported."""
    if not items:
        raise fmt.BundleError("Choose at least one dataset to export.")
    seen = set()
    plans = []
    for item in items:
        if item.dataset_name in seen:
            raise fmt.BundleError(f"Dataset '{item.dataset_name}' is listed twice.")
        seen.add(item.dataset_name)
        try:
            plans.append(fmt.build_export_plan(workspace, item.dataset_name, item.project_name))
        except fmt.BundleError as e:
            raise fmt.BundleError(f"'{item.dataset_name}': {e}") from e
    return plans


def export_compilation(
    items: List[CompilationItem],
    plans: List["fmt.BundleExportPlan"],
    dest_dir: Path,
    name: str,
    overwrite: bool = False,
    progress_cb: Optional[ProgressCallback] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> dict:
    """Write a compilation named ``name`` inside ``dest_dir``. On failure or cancellation the
    partly-written folder is removed."""
    if len(items) != len(plans) or not items:
        raise fmt.BundleError("Choose at least one dataset to export.")
    if not is_valid_compilation_name(name):
        raise fmt.BundleError(
            f"'{name}' can't be used as a folder name. Avoid \\ / : * ? \" < > | and leading or trailing spaces and dots."
        )
    root = Path(dest_dir) / name.strip()
    if root.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists: {root}")

    # Check everything up front, across every dataset, so nothing is written (and an existing package
    # isn't replaced) unless the whole compilation can be made complete.
    if any(item.include_media for item in items):
        if progress_cb:
            progress_cb("Checking the files to copy...", 0)
        problems: List[str] = []
        needed = 0
        for item, plan in zip(items, plans):
            if not item.include_media:
                continue
            missing_sources, size = fmt.scan_media_sources(plan, cancel_cb)
            needed += size
            if missing_sources:
                problems.append(f"'{plan.dataset.name}': " + fmt.missing_media_message(plan, missing_sources))
        if problems:
            raise fmt.BundleError("A complete package can't be made.\n" + "\n".join(problems))
        fmt.ensure_free_space(Path(dest_dir), needed)

    if root.exists():
        shutil.rmtree(root)

    def progress(msg: str, pct: int):
        if progress_cb:
            progress_cb(msg, max(0, min(100, pct)))

    weights = [max(1, plan.num_media if item.include_media else 1) for item, plan in zip(items, plans)]
    total_weight = sum(weights)
    taken: set = set()
    entries: List[dict] = []
    results: List[dict] = []
    done_weight = 0
    root.mkdir(parents=True)
    try:
        progress("Preparing compilation...", 0)
        n = len(items)
        for index, (item, plan, weight) in enumerate(zip(items, plans, weights)):
            if cancel_cb and cancel_cb():
                raise fmt.BundleError("Export cancelled.")
            folder = safe_folder_name(plan.dataset.name, taken)

            def inner_progress(msg, pct, _i=index, _done=done_weight, _w=weight, _name=plan.dataset.name):
                progress(f"{_i + 1}/{n} · {_name}: {msg}", int(98 * (_done + _w * pct / 100) / total_weight))

            result = fmt.export_annotation_bundle(
                plan, root / folder, overwrite=False,
                include_media=item.include_media, include_project=True,
                include_pose=item.include_pose, include_behavior=item.include_behavior,
                progress_cb=inner_progress, cancel_cb=cancel_cb, preflight=False,   # checked above, across all datasets
            )
            done_weight += weight
            results.append({"dataset_name": plan.dataset.name, **result})
            entries.append({
                "name": plan.dataset.name,
                "folder": folder,
                "type": plan.dataset.type.value,
                "project": plan.project.name,
                "media_kind": plan.media_kind,
                "media_included": bool(item.include_media),
                "num_media": plan.num_media,
                "multi_arena": plan.dataset.is_multi_arena,
                "pose_labels": bool(result["pose_present"]),
                "num_labeled_frames": plan.pose.num_labeled_frames if result["pose_present"] else 0,
                "behavior_labels": bool(result["behavior_present"]),
                "num_labeled_videos": plan.behavior.num_labeled_videos if result["behavior_present"] else 0,
            })

        progress("Writing compilation description...", 99)
        info = {
            "compilation_format_version": COMPILATION_FORMAT_VERSION,
            "bundle_format_version": fmt.BUNDLE_FORMAT_VERSION,
            "generator": fmt.GENERATOR,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "name": name.strip(),
            "projects": sorted({e["project"] for e in entries}),
            "datasets": entries,
        }
        (root / COMPILATION_INFO_FILENAME).write_text(json.dumps(info, indent=4), encoding="utf-8")
        _write_readme(root, info)
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise

    progress("Export complete.", 100)
    return {
        "compilation_dir": str(root),
        "num_datasets": len(entries),
        "datasets": results,
        "num_media": sum(r["num_media"] for r in results),
        "num_media_copied": sum(r["num_media_copied"] for r in results),
        "num_missing": sum(r["num_missing"] for r in results),
        "problems": [f"'{r['dataset_name']}': {p}" for r in results for p in r["problems"]],
    }


def _write_readme(root: Path, info: dict) -> None:
    lines = [
        "whisker-labeler compilation",
        f"Exported: {info['created_at']}",
        f"Name:     {info['name']}",
        "",
        f"{len(info['datasets'])} dataset(s), {len(info['projects'])} project(s): {', '.join(info['projects'])}",
        "",
    ]
    for e in info["datasets"]:
        parts = []
        parts.append(f"{e['num_media']} {e['media_kind']}" + ("" if e["media_included"] else " (not copied)"))
        if e["pose_labels"]:
            parts.append(f"pose labels ({e['num_labeled_frames']} frames)")
        if e["behavior_labels"]:
            parts.append(f"behavior labels ({e['num_labeled_videos']} videos)")
        lines.append(f"  {e['folder']}/   {e['name']}: " + ", ".join(parts))
    lines += [
        "",
        "Each folder above is a complete export of one dataset, in the same format a",
        "single-dataset export produces.",
        "",
        "To import into whisker-labeler:",
        "  File -> Import...  and choose THIS folder. Tick the datasets, and the parts of",
        "  each one, that you want. (Choosing one dataset's folder imports just that one.)",
    ]
    (root / README_FILENAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Inspecting
# --------------------------------------------------------------------------- #


@dataclass
class CompilationEntry:
    name: str
    folder: str
    root: Path
    contents: Optional[BundleContents] = None
    problem: str = ""               # why this dataset can't be imported at all

    @property
    def ok(self) -> bool:
        return self.contents is not None


@dataclass
class CompilationContents:
    root: Path
    info: dict
    name: str = ""
    created_at: str = ""
    entries: List[CompilationEntry] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def project_names(self) -> List[str]:
        return sorted({e.contents.project_obj.name for e in self.entries
                       if e.ok and e.contents.project.ok and e.contents.project_obj})

    def entry(self, name: str) -> Optional[CompilationEntry]:
        return next((e for e in self.entries if e.name == name), None)


def peek_compilation(root: Path) -> Tuple[str, int]:
    """``(name, number of datasets)`` without inspecting the exports; ``("", 0)`` if unreadable."""
    try:
        info = json.loads((Path(root) / COMPILATION_INFO_FILENAME).read_text(encoding="utf-8"))
        return str(info.get("name") or Path(root).name), len(info.get("datasets", []))
    except (OSError, ValueError, AttributeError):
        return "", 0


def inspect_compilation(root: Path) -> CompilationContents:
    """Read ``compilation_info.json`` and inspect every export it lists."""
    root = Path(root)
    path = root / COMPILATION_INFO_FILENAME
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise BundleImportError(f"Couldn't open {path.name}: {e}") from e
    except json.JSONDecodeError as e:
        raise BundleImportError(
            f"{path.name} in '{root.name}' isn't readable ({e}). The compilation may be damaged — try copying it again."
        ) from e
    if not isinstance(info, dict) or not isinstance(info.get("datasets"), list):
        raise BundleImportError(f"{path.name} in '{root.name}' isn't a WHISKER compilation description.")

    c = CompilationContents(root=root, info=info, name=str(info.get("name") or root.name),
                            created_at=str(info.get("created_at", "")))
    try:
        if int(str(info.get("compilation_format_version", "1")).split(".")[0]) > int(COMPILATION_FORMAT_VERSION.split(".")[0]):
            c.warnings.append("This compilation was made by a newer version of the labeler; some details may be missed.")
    except ValueError:
        pass

    seen = set()
    for d in info["datasets"]:
        folder = str(d.get("folder") or "")
        name = str(d.get("name") or folder)
        if name in seen:
            continue
        seen.add(name)
        bundle_root = root / folder
        if not folder or not (bundle_root / fmt.EXPORT_INFO_FILENAME).is_file():
            c.entries.append(CompilationEntry(name, folder, bundle_root,
                                              problem=f"Its folder '{folder}' is missing or isn't an export."))
            continue
        try:
            c.entries.append(CompilationEntry(name, folder, bundle_root, contents=bi.inspect_bundle(bundle_root)))
        except BundleImportError as e:
            c.entries.append(CompilationEntry(name, folder, bundle_root, problem=str(e)))
    return c


# --------------------------------------------------------------------------- #
# Import selection
# --------------------------------------------------------------------------- #


@dataclass
class CompilationSelection:
    items: Dict[str, ImportSelection] = field(default_factory=dict)   # by dataset name in the compilation
    import_projects: bool = True
    # For each project the compilation uses: ("new", the name to save it as) or ("existing",
    # one of your own projects to use instead). A project with no entry is added under its own name.
    project_choices: Dict[str, Tuple[str, str]] = field(default_factory=dict)
    overwrite_projects: bool = False
    existing_labels_policy: LabelPolicy = LabelPolicy.MERGE_EXISTING   # when a dataset already has labels
    keep_unmatched: bool = False

    def active_items(self) -> Dict[str, ImportSelection]:
        return {n: s for n, s in self.items.items() if s.dataset or s.pose or s.behavior}

    def choice_for(self, project_name: str) -> Tuple[str, str]:
        return self.project_choices.get(project_name, ("new", project_name))

    def adds_projects(self, contents: "CompilationContents") -> bool:
        return self.import_projects and any(self.choice_for(p)[0] == "new" for p in contents.project_names)


def default_compilation_selection(workspace, contents: CompilationContents) -> CompilationSelection:
    sel = CompilationSelection()
    for e in contents.entries:
        if not e.ok:
            sel.items[e.name] = ImportSelection(dataset_name=e.name)
            continue
        s = bi.default_selection(workspace, e.contents)
        s.project = False                       # projects are handled once, for the whole compilation
        if not s.dataset and (s.pose or s.behavior):
            # In a compilation the natural home for labels is the dataset of the same name.
            s.target_dataset = e.name if workspace.datasets.get(e.name) is not None else ""
        sel.items[e.name] = s
    for name in contents.project_names:
        # A project you already have is used as it is; otherwise it's added under its own name.
        sel.project_choices[name] = ("existing", name) if workspace.projects.get(name) is not None else ("new", name)
    sel.import_projects = bool(contents.project_names)
    return sel


def _item_for_import(item: ImportSelection) -> ImportSelection:
    it = copy.copy(item)
    it.project = False
    return it


def validate_compilation_selection(workspace, contents: CompilationContents, sel: CompilationSelection) -> List[str]:
    problems: List[str] = []
    active = sel.active_items()
    if not active and not sel.adds_projects(contents):
        return ["Tick at least one thing to import."]
    problems.extend(_project_problems(workspace, contents, sel))
    new_names: Dict[str, str] = {}
    for e in contents.entries:
        item = active.get(e.name)
        if item is None:
            continue
        if not e.ok:
            problems.append(f"'{e.name}': {e.problem}")
            continue
        for p in bi.validate_selection(workspace, e.contents, _item_for_import(item)):
            problems.append(f"'{e.name}': {p}")
        if item.dataset:
            key = item.dataset_name.strip().lower()
            if key in new_names:
                problems.append(f"'{e.name}' and '{new_names[key]}' would both be imported as '{item.dataset_name.strip()}'.")
            new_names[key] = e.name
    return problems


def _project_problems(workspace, contents: CompilationContents, sel: CompilationSelection) -> List[str]:
    problems: List[str] = []
    if not sel.import_projects:
        return problems
    taken: Dict[str, str] = {}
    for name in contents.project_names:
        mode, target = sel.choice_for(name)
        target = target.strip()
        if mode == "existing":
            if workspace.projects.get(target) is None:
                problems.append(f"Project '{name}': choose which of your projects to use.")
        elif not bi.is_valid_name(target):
            problems.append(f"Project '{name}': the new name can't be empty or contain \\ / : * ? \" < > |.")
        elif workspace.projects.get(target) is not None and not sel.overwrite_projects:
            problems.append(f"Project '{name}': you already have a project called '{target}' \u2014 "
                            "choose another name or use your existing one.")
        elif target.lower() in taken:
            problems.append(f"Projects '{taken[target.lower()]}' and '{name}' would both be saved as '{target}'.")
        else:
            taken[target.lower()] = name
    return problems


def attach_notes(workspace, entry: CompilationEntry, item: ImportSelection) -> List[Tuple[str, str]]:
    """How labels going onto an existing dataset compare with it: ``[(level, text)]`` where
    level is 'ok', 'warn' or 'bad'. For the compilation dialog's notes column."""
    notes: List[Tuple[str, str]] = []
    if item.dataset or not (item.pose or item.behavior) or not item.target_dataset or not entry.ok:
        return notes
    target = item.target_dataset
    for kind, wanted, path, analyze, unit in (
        ("Pose", item.pose, entry.contents.pose.path, bi.analyze_pose_labels, "frames"),
        ("Behavior", item.behavior, entry.contents.behavior.path, bi.analyze_behavior_labels, "videos"),
    ):
        if not wanted:
            continue
        try:
            r = analyze(workspace, target, path)
        except Exception as e:
            notes.append(("bad", f"{kind}: couldn't read ({e})"))
            continue
        if r.nothing_matches:
            notes.append(("bad", f"{kind}: none of {r.total} {unit} match '{target}' — will be skipped"))
            continue
        text = f"{kind}: {len(r.matched)}/{r.total} {unit} match"
        if r.unmatched:
            text += f" ({len(r.unmatched)} skipped)"
        if r.has_existing:
            text += f"; '{target}' already has labels ({len(r.overlapping)} overlap)"
        notes.append(("warn" if r.unmatched or (r.has_existing and r.problems) else "ok", text))
        for problem in r.problems:
            notes.append(("bad", f"{kind}: can't combine — {problem}"))
    return notes


# --------------------------------------------------------------------------- #
# Import
# --------------------------------------------------------------------------- #


def _plan_attach(workspace, entry: CompilationEntry, item: ImportSelection, sel: CompilationSelection) -> List[str]:
    """Fill in ``item``'s label policies for labels going onto an existing dataset, skipping
    (with a note) any kind that can't be applied, so one mismatch never aborts the dataset."""
    notes: List[str] = []
    target = item.target_dataset
    for kind, wanted, path, analyze in (
        ("Pose labels", item.pose, entry.contents.pose.path, bi.analyze_pose_labels),
        ("Behavior labels", item.behavior, entry.contents.behavior.path, bi.analyze_behavior_labels),
    ):
        if not wanted:
            continue
        attr = "pose_policy" if kind.startswith("Pose") else "behavior_policy"
        r = analyze(workspace, target, path)
        if r.nothing_matches:
            setattr(item, attr, LabelPolicy.SKIP)
            notes.append(f"{kind} skipped: none matched a file in '{target}'.")
        elif not r.has_existing:
            setattr(item, attr, LabelPolicy.ADD)
        elif r.problems and sel.existing_labels_policy in (LabelPolicy.MERGE_EXISTING, LabelPolicy.MERGE_IMPORTED):
            setattr(item, attr, LabelPolicy.SKIP)
            notes.append(f"{kind} not combined ({r.problems[0]}) — your existing labels were left unchanged.")
        else:
            setattr(item, attr, sel.existing_labels_policy)
    return notes


def import_compilation(
    workspace,
    contents: CompilationContents,
    sel: CompilationSelection,
    progress_cb: Optional[ProgressCallback] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
) -> dict:
    """Import what ``sel`` ticks. Returns ``{"projects": [...], "datasets": [{"name", "result",
    "error"}], "cancelled": bool}``; a dataset that fails is recorded and the rest continue."""
    problems = validate_compilation_selection(workspace, contents, sel)
    if problems:
        raise BundleImportError(problems[0])

    def progress(msg: str, pct: int):
        if progress_cb:
            progress_cb(msg, max(0, min(100, pct)))

    out: dict = {"projects": [], "datasets": [], "cancelled": False}

    if sel.import_projects:
        progress("Importing projects...", 0)
        done = set()
        for e in contents.entries:
            if not e.ok or not e.contents.project.ok or e.contents.project_obj is None:
                continue
            proj = e.contents.project_obj
            if proj.name in done:
                continue
            done.add(proj.name)
            mode, target = sel.choice_for(proj.name)
            if mode == "existing":
                out["projects"].append({"name": proj.name, "installed": False, "existing": target})
                continue
            installed = mi.install_project(workspace, proj, e.contents.project.path, sel.overwrite_projects,
                                           name=target.strip())
            out["projects"].append({"name": proj.name, "installed": installed, "as": target.strip()})

    active = [(e, sel.active_items()[e.name]) for e in contents.entries if e.name in sel.active_items()]
    n = len(active)
    for index, (e, item) in enumerate(active):
        if cancel_cb and cancel_cb():
            out["cancelled"] = True
            break
        lo, hi = 5 + 95 * index // max(n, 1), 5 + 95 * (index + 1) // max(n, 1)

        def inner(msg, pct, _e=e, _lo=lo, _hi=hi, _i=index):
            progress(f"{_i + 1}/{n} · {_e.name}: {msg}", _lo + (_hi - _lo) * pct // 100)

        record = {"name": e.name, "result": None, "error": None}
        out["datasets"].append(record)
        try:
            prepared = _item_for_import(item)
            prepared.keep_unmatched = sel.keep_unmatched
            notes: List[str] = []
            if not prepared.dataset and (prepared.pose or prepared.behavior):
                notes = _plan_attach(workspace, e, prepared, sel)
            result = bi.import_from_bundle(workspace, e.contents, prepared, inner, cancel_cb)
            result["notes"] = notes + result.get("notes", [])
            record["result"] = result
        except Exception as err:      # isolate: one dataset failing must not stop the others
            if "cancel" in str(err).lower():
                record["error"] = "Cancelled."
                out["cancelled"] = True
                break
            logger.warning("Compilation import of '%s' failed: %s", e.name, err, exc_info=True)
            record["error"] = str(err)

    progress("Import complete.", 100)
    return out
