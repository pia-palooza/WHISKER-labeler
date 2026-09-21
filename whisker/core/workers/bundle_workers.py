"""Background jobs for importing and exporting Whisker bundles.

They wrap the pure-filesystem work in :mod:`whisker.core.whisker_bundle` so the GUI can run it on a worker thread
with progress and cancellation. Every decision (what to bring in, what to replace, where media go) is made by the
caller *before* a job starts, so a job never pops dialogs and never touches the in-memory workspace; the caller
rescans the workspace afterwards.
"""

from pathlib import Path
from typing import Iterable, Optional

from whisker.base.job import BaseJob
from whisker.core import whisker_bundle as wb


class ExportBundleJob(BaseJob):
    def __init__(self, workspace, plan: "wb.ExportPlan", dest: Path, overwrite: bool = False):
        super().__init__()
        self.workspace = workspace
        self.plan = plan
        self.dest = Path(dest)
        self.overwrite = overwrite

    def run(self) -> dict:
        return wb.export_bundle(
            self.workspace,
            self.plan,
            self.dest,
            overwrite=self.overwrite,
            progress_cb=self.report_progress,
            cancel_cb=lambda: self.is_cancelled,
        )


class ImportBundleJob(BaseJob):
    def __init__(self, workspace, bundle_path: Path, chosen: Iterable[str], media_root: Optional[Path] = None):
        super().__init__()
        self.workspace = workspace
        self.bundle_path = Path(bundle_path)
        self.chosen = set(chosen)
        self.media_root = Path(media_root) if media_root else None

    def run(self) -> dict:
        # Opened here, on the worker thread, so the archive is never shared between threads.
        with wb.WhiskerBundle.open(self.bundle_path) as bundle:
            return wb.import_bundle(
                self.workspace,
                bundle,
                self.chosen,
                self.media_root,
                progress_cb=self.report_progress,
                cancel_cb=lambda: self.is_cancelled,
            )
