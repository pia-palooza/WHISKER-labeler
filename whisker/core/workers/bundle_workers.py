"""Background jobs for exporting and importing annotation bundles.

They wrap the pure-filesystem work in :mod:`whisker.core.bundle` (export) and
:mod:`whisker.core.bundle_import` (import) so the GUI can run it on a worker thread
with progress reporting. All conflict (overwrite / merge) decisions are resolved by the
caller *before* a job starts, so a job never pops dialogs and never mutates the
in-memory workspace; the caller rescans the workspace afterwards.
"""

from pathlib import Path

from whisker.base.job import BaseJob
from whisker.core import bundle, bundle_import, compilation


class ExportBundleJob(BaseJob):
    def __init__(
        self,
        plan: "bundle.BundleExportPlan",
        bundle_dir: Path,
        overwrite: bool = False,
        include_media: bool = True,
        include_project: bool = True,
        include_pose: bool = True,
        include_behavior: bool = True,
    ):
        super().__init__()
        self.plan = plan
        self.bundle_dir = Path(bundle_dir)
        self.overwrite = overwrite
        self.include_media = include_media
        self.include_project = include_project
        self.include_pose = include_pose
        self.include_behavior = include_behavior

    def run(self) -> dict:
        return bundle.export_annotation_bundle(
            self.plan,
            self.bundle_dir,
            overwrite=self.overwrite,
            include_media=self.include_media,
            include_project=self.include_project,
            include_pose=self.include_pose,
            include_behavior=self.include_behavior,
            progress_cb=self.report_progress,
            cancel_cb=lambda: self.is_cancelled,
        )


class ExportCompilationJob(BaseJob):
    def __init__(self, items, plans, dest_dir: Path, name: str, overwrite: bool = False):
        super().__init__()
        self.items = items
        self.plans = plans
        self.dest_dir = Path(dest_dir)
        self.name = name
        self.overwrite = overwrite

    def run(self) -> dict:
        return compilation.export_compilation(
            self.items,
            self.plans,
            self.dest_dir,
            self.name,
            overwrite=self.overwrite,
            progress_cb=self.report_progress,
            cancel_cb=lambda: self.is_cancelled,
        )


class ImportCompilationJob(BaseJob):
    def __init__(self, workspace, contents: "compilation.CompilationContents",
                 selection: "compilation.CompilationSelection"):
        super().__init__()
        self.workspace = workspace
        self.contents = contents
        self.selection = selection

    def run(self) -> dict:
        return compilation.import_compilation(
            self.workspace,
            self.contents,
            self.selection,
            progress_cb=self.report_progress,
            cancel_cb=lambda: self.is_cancelled,
        )


class ImportBundleJob(BaseJob):
    def __init__(self, workspace, contents: "bundle_import.BundleContents",
                 selection: "bundle_import.ImportSelection"):
        super().__init__()
        self.workspace = workspace
        self.contents = contents
        self.selection = selection

    def run(self) -> dict:
        return bundle_import.import_from_bundle(
            self.workspace,
            self.contents,
            self.selection,
            progress_cb=self.report_progress,
            cancel_cb=lambda: self.is_cancelled,
        )
