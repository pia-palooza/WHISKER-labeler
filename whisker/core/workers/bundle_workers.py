"""Background job for exporting an annotation bundle.

Wraps the pure-filesystem export in :mod:`whisker.core.bundle` so the GUI can
run it on a worker thread with progress reporting. All conflict (overwrite)
decisions are resolved by the caller *before* the job starts, so the job
never pops dialogs and never mutates the in-memory workspace.

(Dataset *import* no longer goes through a "bundle" — see
:mod:`whisker.core.manual_import` and
:class:`whisker.core.workers.manual_import_workers.ImportComponentsJob`.)
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
