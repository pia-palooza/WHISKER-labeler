"""Dialog for importing a dataset from its individual pieces.

Replaces the old "pick one bundle folder and hope it's valid" flow. Here the
user browses to each piece separately — the project file, the dataset info
file, the folder of frames/videos, and (optionally) label files — and each
one is checked the moment it's picked, with a specific plain-English reason
if it's wrong. Nothing is copied until every required piece has checked out.
"""

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QDialog,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QDialogButtonBox,
    QGroupBox,
    QCheckBox,
    QFrame,
)

from whisker.core import manual_import as mi
from whisker.core.workspace import Workspace

_OK_STYLE = "color: #2e7d32;"
_BAD_STYLE = "color: #c0392b;"
_EMPTY_STYLE = "color: gray;"


class ImportDatasetDialog(QDialog):
    """Pick + validate each component, then import."""

    def __init__(self, workspace: Workspace, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._workspace = workspace

        self._project_path: Optional[Path] = None
        self._dataset_path: Optional[Path] = None
        self._media_dir: Optional[Path] = None
        self._pose_path: Optional[Path] = None
        self._pose_meta_path: Optional[Path] = None
        self._behavior_path: Optional[Path] = None

        self._project = None
        self._dataset = None

        self._name_autofill_value = ""

        self.setWindowTitle("Import Dataset")

        screen = QApplication.primaryScreen()
        self._dpi = screen.logicalDotsPerInch() / 96.0 if screen else 1.0
        self.setMinimumWidth(int(640 * self._dpi))

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(
            self._hint_label(
                "Select each piece below. Every field is checked as soon as you pick "
                "it, so you'll see exactly what's wrong instead of a generic error."
            )
        )

        # --- Required pieces ---
        required_group = QGroupBox("Required")
        required_layout = QVBoxLayout(required_group)

        self.project_edit, self.project_status = self._add_file_field(
            required_layout,
            "Project file",
            "Select the project .json file...",
            "Select Project File",
            "JSON files (*.json)",
            on_browse=self._on_browse_project,
            on_clear=None,
        )

        self.dataset_edit, self.dataset_status = self._add_file_field(
            required_layout,
            "Dataset info file (manifest.json)",
            "Select the dataset's manifest.json...",
            "Select Dataset Info File",
            "JSON files (*.json)",
            on_browse=self._on_browse_dataset,
            on_clear=None,
        )

        self.media_label_widget, self.media_edit, self.media_status = self._add_folder_field(
            required_layout,
            "Media folder",
            "Select the folder containing the frames/videos...",
            "Select Media Folder",
            on_browse=self._on_browse_media,
        )

        self.name_edit, self.name_status = self._add_name_field(required_layout)

        main_layout.addWidget(required_group)

        # --- Optional pieces ---
        optional_group = QGroupBox("Optional — labels")
        optional_layout = QVBoxLayout(optional_group)

        self.pose_edit, self.pose_status = self._add_file_field(
            optional_layout,
            "Pose labels file",
            "(optional) Select a pose labels.h5 file...",
            "Select Pose Labels File",
            "HDF5 files (*.h5)",
            on_browse=self._on_browse_pose,
            on_clear=self._on_clear_pose,
        )

        self.behavior_edit, self.behavior_status = self._add_file_field(
            optional_layout,
            "Behavior labels file",
            "(optional) Select a behavior labels.h5 file...",
            "Select Behavior Labels File",
            "HDF5 files (*.h5)",
            on_browse=self._on_browse_behavior,
            on_clear=self._on_clear_behavior,
        )

        main_layout.addWidget(optional_group)

        # --- Conflicts / overwrite ---
        self.conflict_label = QLabel("")
        self.conflict_label.setWordWrap(True)
        self.conflict_label.setStyleSheet("color: #e67e22;")
        self.conflict_label.setVisible(False)
        main_layout.addWidget(self.conflict_label)

        self.overwrite_checkbox = QCheckBox("Overwrite existing items in this workspace")
        self.overwrite_checkbox.setVisible(False)
        self.overwrite_checkbox.toggled.connect(self._revalidate)
        main_layout.addWidget(self.overwrite_checkbox)

        # --- Buttons ---
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setText("Import")
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        main_layout.addWidget(self.button_box)

        self._revalidate()

    # -- widget-building helpers ----------------------------------------

    def _hint_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet("color: gray;")
        return label

    def _add_file_field(
        self, layout, title, placeholder, dialog_title, name_filter, on_browse, on_clear
    ):
        if layout.count():
            layout.addWidget(self._divider())
        title_label = QLabel(f"<b>{title}</b>")
        layout.addWidget(title_label)

        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        edit.textChanged.connect(self._revalidate)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(lambda: on_browse(edit, dialog_title, name_filter))
        h.addWidget(edit)
        h.addWidget(browse_btn)
        if on_clear is not None:
            clear_btn = QPushButton("Clear")
            clear_btn.clicked.connect(lambda: on_clear(edit))
            h.addWidget(clear_btn)
        layout.addWidget(row)

        status = QLabel("")
        status.setWordWrap(True)
        status.setStyleSheet(_EMPTY_STYLE)
        layout.addWidget(status)

        return edit, status

    def _add_folder_field(self, layout, title, placeholder, dialog_title, on_browse):
        if layout.count():
            layout.addWidget(self._divider())
        title_label = QLabel(f"<b>{title}</b>")
        layout.addWidget(title_label)

        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        edit.textChanged.connect(self._revalidate)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(lambda: on_browse(edit, dialog_title))
        h.addWidget(edit)
        h.addWidget(browse_btn)
        layout.addWidget(row)

        status = QLabel("")
        status.setWordWrap(True)
        status.setStyleSheet(_EMPTY_STYLE)
        layout.addWidget(status)

        return title_label, edit, status

    def _add_name_field(self, layout):
        layout.addWidget(self._divider())
        layout.addWidget(QLabel("<b>Dataset name</b>"))
        edit = QLineEdit()
        edit.setPlaceholderText("Filled in automatically once the dataset info file loads...")
        edit.textChanged.connect(self._on_name_edited)
        layout.addWidget(edit)
        status = QLabel("")
        status.setWordWrap(True)
        status.setStyleSheet(_EMPTY_STYLE)
        layout.addWidget(status)
        return edit, status

    def _divider(self) -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        return line

    # -- browse handlers -------------------------------------------------

    def _on_browse_project(self, edit: QLineEdit, title: str, name_filter: str):
        start = edit.text().strip() or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(self, title, start, name_filter)
        if path:
            edit.setText(path)

    def _on_browse_dataset(self, edit: QLineEdit, title: str, name_filter: str):
        start = edit.text().strip() or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(self, title, start, name_filter)
        if path:
            edit.setText(path)

    def _on_browse_media(self, edit: QLineEdit, title: str):
        start = edit.text().strip() or str(Path.home())
        path = QFileDialog.getExistingDirectory(self, title, start)
        if path:
            edit.setText(path)

    def _on_browse_pose(self, edit: QLineEdit, title: str, name_filter: str):
        start = edit.text().strip() or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(self, title, start, name_filter)
        if path:
            edit.setText(path)

    def _on_clear_pose(self, edit: QLineEdit):
        edit.clear()

    def _on_browse_behavior(self, edit: QLineEdit, title: str, name_filter: str):
        start = edit.text().strip() or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(self, title, start, name_filter)
        if path:
            edit.setText(path)

    def _on_clear_behavior(self, edit: QLineEdit):
        edit.clear()

    def _on_name_edited(self):
        if self.name_edit.text() != self._name_autofill_value:
            # User is typing their own name; stop auto-filling from the dataset file.
            self._name_autofill_value = None
        self._revalidate()

    # -- validation --------------------------------------------------------

    def _set_status(self, label: QLabel, check: mi.ComponentCheck):
        label.setText(check.message)
        label.setStyleSheet(_OK_STYLE if check.ok else _BAD_STYLE)

    def _revalidate(self):
        # Project
        self._project_path = self._path_or_none(self.project_edit.text())
        project_check, self._project = mi.check_project_file(self._project_path)
        self._set_status(self.project_status, project_check)

        # Dataset
        self._dataset_path = self._path_or_none(self.dataset_edit.text())
        dataset_check, self._dataset = mi.check_dataset_file(self._dataset_path)
        self._set_status(self.dataset_status, dataset_check)

        # Media folder label + check (depends on dataset type once known)
        self.media_label_widget.setText(f"<b>{mi.media_label_for(self._dataset)}</b>")
        self._media_dir = self._path_or_none(self.media_edit.text())
        media_check, _missing = mi.check_media_folder(self._media_dir, self._dataset)
        self._set_status(self.media_status, media_check)

        # Dataset name auto-fill (only while the user hasn't typed their own)
        if self._dataset is not None and self._name_autofill_value is not None:
            if self.name_edit.text().strip() in ("", self._name_autofill_value):
                self.name_edit.blockSignals(True)
                self.name_edit.setText(self._dataset.name)
                self.name_edit.blockSignals(False)
                self._name_autofill_value = self._dataset.name
        dataset_name = self.name_edit.text().strip()
        if not dataset_name:
            self._set_status(
                self.name_status,
                mi.ComponentCheck.empty("Required — pick a name for this dataset."),
            )
        else:
            self._set_status(
                self.name_status, mi.ComponentCheck.good(f"Will be imported as '{dataset_name}'.")
            )

        # Pose labels (optional)
        self._pose_path = self._path_or_none(self.pose_edit.text())
        pose_check, self._pose_meta_path = mi.check_pose_labels_file(self._pose_path)
        self._set_status(self.pose_status, pose_check)

        # Behavior labels (optional)
        self._behavior_path = self._path_or_none(self.behavior_edit.text())
        behavior_check = mi.check_behavior_labels_file(self._behavior_path)
        self._set_status(self.behavior_status, behavior_check)

        # Workspace conflicts
        project_name = self._project.name if self._project else None
        conflicts = mi.check_workspace_conflicts(self._workspace, dataset_name, project_name)
        if conflicts.any:
            existing = []
            if conflicts.project_exists:
                existing.append(f"project '{project_name}'")
            if conflicts.dataset_exists:
                existing.append(f"dataset '{dataset_name}'")
            if conflicts.pose_labels_exist:
                existing.append("pose labels")
            if conflicts.behavior_labels_exist:
                existing.append("behavior labels")
            self.conflict_label.setText(
                "Already present in this workspace: "
                + ", ".join(existing)
                + ". Check the box below to overwrite them."
            )
            self.conflict_label.setVisible(True)
            self.overwrite_checkbox.setVisible(True)
        else:
            self.conflict_label.setVisible(False)
            self.overwrite_checkbox.setVisible(False)

        required_ok = (
            project_check.ok
            and dataset_check.ok
            and media_check.ok
            and bool(dataset_name)
            and pose_check.ok
            and behavior_check.ok
        )
        ok = required_ok and (not conflicts.any or self.overwrite_checkbox.isChecked())
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(ok)

    @staticmethod
    def _path_or_none(text: str) -> Optional[Path]:
        text = text.strip()
        return Path(text) if text else None

    # -- results -------------------------------------------------------

    @property
    def project(self):
        return self._project

    @property
    def project_source_path(self) -> Optional[Path]:
        return self._project_path

    @property
    def dataset(self):
        return self._dataset

    @property
    def media_dir(self) -> Optional[Path]:
        return self._media_dir

    @property
    def dataset_name(self) -> str:
        return self.name_edit.text().strip()

    @property
    def pose_labels_path(self) -> Optional[Path]:
        return self._pose_path

    @property
    def pose_metadata_path(self) -> Optional[Path]:
        return self._pose_meta_path

    @property
    def behavior_labels_path(self) -> Optional[Path]:
        return self._behavior_path

    @property
    def overwrite(self) -> bool:
        return self.overwrite_checkbox.isChecked()
