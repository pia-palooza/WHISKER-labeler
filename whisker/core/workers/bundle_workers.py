"""Background jobs for exporting/importing annotation bundles.

These wrap the pure-filesystem functions in :mod:`whisker.core.bundle` so the
GUI can run them on a worker thread with progress reporting. All conflict
(overwrite) decisions are resolved by the caller *before* the job starts, so
the jobs never pop dialogs and never mutate the in-memory workspace.
"""

from pathlib import Path

from whisker.base.job import BaseJob
from whisker.core import bundle


class ExportBundleJob(BaseJob):
    def __init__(
        self,
        plan: "bundle.BundleExportPlan",
        bundle_dir: Path,
        overwrite: bool = False,
        include_media: bool = True,
    ):
        super().__init__()
        self.plan = plan
        self.bundle_dir = Path(bundle_dir)
        self.overwrite = overwrite
        self.include_media = include_media

    def run(self) -> dict:
        return bundle.export_annotation_bundle(
            self.plan,
            self.bundle_dir,
            overwrite=self.overwrite,
            include_media=self.include_media,
            progress_cb=self.report_progress,
            cancel_cb=lambda: self.is_cancelled,
        )


class ImportBundleJob(BaseJob):
    def __init__(
        self,
        workspace,
        bundle_dir: Path,
        overwrite: bool = False,
        media_source_dir=None,
    ):
        super().__init__()
        self.workspace = workspace
        self.bundle_dir = Path(bundle_dir)
        self.overwrite = overwrite
        self.media_source_dir = media_source_dir

    def run(self) -> dict:
        return bundle.import_annotation_bundle(
            self.workspace,
            self.bundle_dir,
            overwrite=self.overwrite,
            media_source_dir=self.media_source_dir,
            progress_cb=self.report_progress,
            cancel_cb=lambda: self.is_cancelled,
        )
