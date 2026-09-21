"""Helpers shared by the Whisker bundle tests: building hand-made bundles and comparing workspaces."""
import hashlib
import json
import zipfile
from pathlib import Path

from fixtures import WorkspaceCase
from whisker.core import whisker_bundle as wb


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree(ws_root: Path) -> dict:
    """What a workspace holds that an import may change (datasets, projects, workflows), plus stray staging folders."""
    out = {}
    for top in ("datasets", "projects", "workflows"):
        for p in (ws_root / top).rglob("*"):
            out[p.relative_to(ws_root).as_posix()] = p.stat().st_size if p.is_file() else -1
    out["<staging>"] = sorted(p.name for p in ws_root.iterdir() if p.name.startswith(".wbi_"))
    return out


def write_zip(path: Path, entries: dict, manifest: dict = None, stored: bool = True):
    """A hand-made bundle: ``entries`` maps archive names to bytes."""
    with zipfile.ZipFile(path, "w", allowZip64=True) as zf:
        if manifest is not None:
            zf.writestr(wb.MANIFEST_NAME, json.dumps(manifest))
        for name, data in entries.items():
            zf.writestr(zipfile.ZipInfo(name), data, compress_type=zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED)


def manifest_for(*datasets, projects=(), runs=()) -> dict:
    return {"format": "whisker-bundle", "format_version": 1, "whisker_version": "0.8.0", "datasets": list(datasets),
            "prediction_runs": list(runs), "projects": list(projects)}


def ds_entry(name, type_="FRAME_SUBSET", files=(), **kw) -> dict:
    return {"name": name, "type": type_, "file_count": len(files), **kw}


def ds_manifest(name, type_, files) -> bytes:
    return json.dumps({"name": name, "type": type_, "base_data_path": "D:\\somewhere\\else", "files": list(files),
                       "multi_arena": None, "arena_boxes": None}).encode()


class BundleCase(WorkspaceCase):
    def make_bundle(self, name="b.zip", dataset="ds", type_="FRAME_SUBSET", files=("a/f1.png", "a/f2.png"),
                    labels=True, project=None, extra_manifest=None):
        entries = {f"datasets/{dataset}/manifest.json": ds_manifest(dataset, type_, files)}
        for rel in files:
            entries[f"datasets/{dataset}/media/{rel}"] = f"media:{rel}".encode() * 10
        if labels:
            entries[f"workflows/pose_estimation/labels/{dataset}/labels.h5"] = b"h5-bytes" * 50
            entries[f"workflows/pose_estimation/labels/{dataset}/metadata.json"] = b'{"dataset_name": "%s"}' % dataset.encode()
        if project:
            entries[f"projects/{project}.json"] = json.dumps({"name": project, "body_parts": ["nose"]}).encode()
        m = manifest_for(ds_entry(dataset, type_, files, label_workflows=["pose_estimation"] if labels else []),
                         projects=[project] if project else [])
        m.update(extra_manifest or {})
        path = self.tmp / name
        write_zip(path, entries, m)
        return path

    def open(self, path):
        b = wb.WhiskerBundle.open(path)
        self.addCleanup(b.close)
        return b
