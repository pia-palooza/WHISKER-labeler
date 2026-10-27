"""Dialog for ``File > Import Whisker Bundle...``: pick a full-WHISKER bundle zip, preview what it has, choose
which datasets/ROI runs/projects to bring in, and run the import (``whisker.core.whisker_bundle_import``).

Modeled on ``import_dataset_dialog.py``'s preview/overwrite-checkbox/Import-Complete convention, trimmed to what a
Whisker-bundle-zip import needs (it is a different format from this app's own dataset-component importer)."""
from pathlib import Path
from typing import Dict, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QDialog, QDialogButtonBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

from whisker.core import whisker_bundle_import as wbi
from whisker.core.workspace import Workspace


class ImportWhiskerBundleDialog(QDialog):
    """Import ROI Tracking content (+ projects) from a full-WHISKER "Whisker bundle" zip (brief section 5.2)."""

    def __init__(self, workspace: Workspace, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Import Whisker Bundle...")
        self.setMinimumWidth(540)
        self._workspace = workspace
        self._info: Optional[wbi.BundleInfo] = None
        self._conflicts: Optional[wbi.ImportConflicts] = None
        self._dataset_checks: Dict[str, QCheckBox] = {}
        self._dataset_overwrite_checks: Dict[str, QCheckBox] = {}
        self._run_checks: Dict[str, QCheckBox] = {}
        self._project_checks: Dict[str, QCheckBox] = {}
        self._media_dir: Optional[Path] = None
        self.report: Optional[wbi.ImportReport] = None

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        path_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("Choose a .zip bundle exported from full WHISKER...")
        path_row.addWidget(self.path_edit, 1)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse)
        path_row.addWidget(browse_btn)
        layout.addLayout(path_row)

        self.ignored_label = QLabel("")
        self.ignored_label.setWordWrap(True)
        self.ignored_label.setStyleSheet("color: #888;")
        layout.addWidget(self.ignored_label)

        self.datasets_group = QGroupBox("Datasets")
        self.datasets_layout = QVBoxLayout(self.datasets_group)
        layout.addWidget(self.datasets_group)

        self.runs_group = QGroupBox("ROI prediction runs")
        self.runs_layout = QVBoxLayout(self.runs_group)
        layout.addWidget(self.runs_group)

        self.projects_group = QGroupBox("Projects")
        self.projects_layout = QVBoxLayout(self.projects_group)
        layout.addWidget(self.projects_group)

        media_row = QHBoxLayout()
        self.media_label = QLabel("")
        self.media_label.setWordWrap(True)
        self.media_browse_btn = QPushButton("Choose media folder...")
        self.media_browse_btn.clicked.connect(self._choose_media_dir)
        self.media_browse_btn.setVisible(False)
        media_row.addWidget(self.media_label, 1)
        media_row.addWidget(self.media_browse_btn)
        layout.addLayout(media_row)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.ok_button.setText("Import")
        self.ok_button.setEnabled(False)
        buttons.accepted.connect(self._do_import)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # --- Choosing the bundle ---
    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose a Whisker Bundle", "", "Whisker Bundle (*.zip)")
        if not path:
            return
        try:
            info = wbi.read_bundle_info(Path(path))
        except wbi.BundleError as e:
            QMessageBox.warning(self, "Could Not Read Bundle", str(e))
            return
        self._info = info
        self._conflicts = wbi.find_import_conflicts(self._workspace, info)
        self._media_dir = None
        self.path_edit.setText(path)
        self._populate_preview()

    @staticmethod
    def _clear_layout(layout):
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _populate_preview(self):
        info, conflicts = self._info, self._conflicts
        self._dataset_checks.clear()
        self._dataset_overwrite_checks.clear()
        self._run_checks.clear()
        self._project_checks.clear()
        self._clear_layout(self.datasets_layout)
        self._clear_layout(self.runs_layout)
        self._clear_layout(self.projects_layout)

        ignored = info.ignored_content_summary()
        self.ignored_label.setText(
            f"This bundle also has {', '.join(ignored)}, which this app ignores (it only reads ROI Tracking "
            "content)." if ignored else ""
        )

        for ds in info.datasets:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            conflict = ds.name in conflicts.datasets
            cb = QCheckBox(f"{ds.name} ({ds.type})" + ("  -- already in this workspace" if conflict else ""))
            cb.setChecked(not conflict)
            self._dataset_checks[ds.name] = cb
            row_layout.addWidget(cb, 1)
            if conflict:
                overwrite_cb = QCheckBox("overwrite")
                overwrite_cb.toggled.connect(cb.setChecked)
                self._dataset_overwrite_checks[ds.name] = overwrite_cb
                row_layout.addWidget(overwrite_cb)
            self.datasets_layout.addWidget(row)
        if not info.datasets:
            self.datasets_layout.addWidget(QLabel("This bundle has no datasets."))

        roi_runs = info.roi_run_names()
        for run_name in roi_runs:
            cb = QCheckBox(run_name)
            cb.setChecked(True)
            self._run_checks[run_name] = cb
            self.runs_layout.addWidget(cb)
        if not roi_runs:
            self.runs_layout.addWidget(QLabel("This bundle has no ROI Tracking prediction runs."))

        for pname in info.projects:
            label = pname
            if pname in conflicts.projects_identical:
                label += "  (already present, identical)"
            elif pname in conflicts.projects_different:
                label += "  (a different project with this name already exists here -- kept unless you tick it)"
            cb = QCheckBox(label)
            cb.setChecked(pname not in conflicts.projects_identical)
            self._project_checks[pname] = cb
            self.projects_layout.addWidget(cb)
        if not info.projects:
            note = QLabel(
                "This bundle has no project. ROI Tracking still needs a project selected in this app (any "
                "project will do) before its Label tab will show anything."
            )
            note.setWordWrap(True)
            self.projects_layout.addWidget(note)

        self.media_browse_btn.setVisible(self._any_needs_media_dir(info.datasets))
        self._update_media_label()
        self.ok_button.setEnabled(True)

    @staticmethod
    def _any_needs_media_dir(datasets) -> bool:
        return any(ds.media_layout == wbi.MEDIA_LAYOUT_EXTERNAL and ds.media_included for ds in datasets)

    # --- Media destination (only shown when the bundle keeps media outside the workspace) ---
    def _choose_media_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Choose a Folder to Store This Bundle's Media")
        if path:
            self._media_dir = Path(path)
        self._update_media_label()

    def _update_media_label(self):
        if self._media_dir:
            self.media_label.setText(f"Media will be stored under: {self._media_dir}")
        elif self.media_browse_btn.isVisible():
            self.media_label.setText("Choose a folder to store this bundle's media.")
        else:
            self.media_label.setText("")

    # --- Import ---
    def _do_import(self):
        if not self._info:
            return
        dataset_names = [name for name, cb in self._dataset_checks.items() if cb.isChecked()]
        if not dataset_names:
            QMessageBox.warning(self, "Nothing Selected", "Choose at least one dataset to import.")
            return
        selected_infos = [d for d in self._info.datasets if d.name in dataset_names]
        if self._any_needs_media_dir(selected_infos) and not self._media_dir:
            QMessageBox.warning(self, "Media Folder Needed", "Choose a folder to store this bundle's media first.")
            return

        overwrite_datasets = {name for name, cb in self._dataset_overwrite_checks.items() if cb.isChecked()}
        run_names = [name for name, cb in self._run_checks.items() if cb.isChecked()]
        project_names = [name for name, cb in self._project_checks.items() if cb.isChecked()]
        overwrite_projects = {
            name for name, cb in self._project_checks.items()
            if cb.isChecked() and self._conflicts and name in self._conflicts.projects_different
        }

        options = wbi.ImportOptions(
            dataset_names=dataset_names,
            media_dir=self._media_dir,
            overwrite_datasets=overwrite_datasets,
            prediction_runs=run_names,
            project_names=project_names,
            overwrite_projects=overwrite_projects,
        )

        self.ok_button.setEnabled(False)
        self.status_label.setText("Importing...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.report = wbi.import_bundle(self._workspace, Path(self.path_edit.text()), options)
        except wbi.BundleError as e:
            QApplication.restoreOverrideCursor()
            self.ok_button.setEnabled(True)
            self.status_label.setText("")
            QMessageBox.critical(self, "Import Failed", str(e))
            return
        QApplication.restoreOverrideCursor()
        self.accept()
