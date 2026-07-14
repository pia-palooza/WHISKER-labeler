"""Dialog for importing an annotation bundle into the current workspace.

Lets the user pick a bundle folder (one produced by "Export Annotations..."),
previews what it contains and what — if anything — it would overwrite in the
current workspace, and requires an explicit confirmation before overwriting.
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
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QDialogButtonBox,
    QGroupBox,
    QCheckBox,
    QTreeWidget,
    QTreeWidgetItem,
)

from whisker.core import bundle
from whisker.core.workspace import Workspace


class ImportBundleDialog(QDialog):
    """Select + confirm import of an annotation bundle."""

    def __init__(self, workspace: Workspace, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._workspace = workspace
        self._preview: Optional[bundle.BundleImportPreview] = None

        self.setWindowTitle("Import Annotation Bundle")

        screen = QApplication.primaryScreen()
        dpi_scale = screen.logicalDotsPerInch() / 96.0 if screen else 1.0
        self.setMinimumWidth(int(620 * dpi_scale))

        main_layout = QVBoxLayout(self)

        # --- Bundle folder selector ---
        grid = QGridLayout()
        grid.setSpacing(int(8 * dpi_scale))
        grid.addWidget(self._right_label("Bundle folder:"), 0, 0)
        path_container = QWidget()
        path_hbox = QHBoxLayout(path_container)
        path_hbox.setContentsMargins(0, 0, 0, 0)
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Select a bundle folder to import...")
        self.path_edit.textChanged.connect(self._on_path_changed)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._on_browse)
        path_hbox.addWidget(self.path_edit)
        path_hbox.addWidget(browse_btn)
        grid.addWidget(path_container, 0, 1)
        grid.setColumnStretch(1, 1)
        main_layout.addLayout(grid)

        # --- Contents preview ---
        contents_group = QGroupBox("Bundle contents")
        contents_layout = QVBoxLayout(contents_group)
        self.contents_tree = QTreeWidget()
        self.contents_tree.setColumnCount(2)
        self.contents_tree.setHeaderLabels(["Item", "Details"])
        self.contents_tree.setRootIsDecorated(False)
        self.contents_tree.setMinimumHeight(int(150 * dpi_scale))
        contents_layout.addWidget(self.contents_tree)
        main_layout.addWidget(contents_group)

        # --- Locate media (reference-only bundles) ---
        self.media_group = QGroupBox("Locate media (not included in bundle)")
        media_grid = QGridLayout(self.media_group)
        media_grid.setSpacing(int(8 * dpi_scale))
        media_grid.addWidget(self._right_label("Media folder:"), 0, 0)
        media_container = QWidget()
        media_hbox = QHBoxLayout(media_container)
        media_hbox.setContentsMargins(0, 0, 0, 0)
        self.media_edit = QLineEdit()
        self.media_edit.setPlaceholderText(
            "Folder containing the referenced videos..."
        )
        media_browse = QPushButton("Browse...")
        media_browse.clicked.connect(self._on_media_browse)
        media_hbox.addWidget(self.media_edit)
        media_hbox.addWidget(media_browse)
        media_grid.addWidget(media_container, 0, 1)
        media_grid.setColumnStretch(1, 1)
        self.media_group.setVisible(False)
        main_layout.addWidget(self.media_group)

        # --- Conflicts / overwrite ---
        self.conflict_label = QLabel("")
        self.conflict_label.setWordWrap(True)
        self.conflict_label.setStyleSheet("color: #e67e22;")
        self.conflict_label.setVisible(False)
        main_layout.addWidget(self.conflict_label)

        self.overwrite_checkbox = QCheckBox(
            "Overwrite existing items in this workspace"
        )
        self.overwrite_checkbox.setVisible(False)
        self.overwrite_checkbox.toggled.connect(self._validate)
        main_layout.addWidget(self.overwrite_checkbox)

        # --- Buttons ---
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setText("Import")
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        main_layout.addWidget(self.button_box)

        self._validate()

    # -- helpers -------------------------------------------------------

    def _right_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        return label

    def _add_row(self, item: str, details: str):
        self.contents_tree.addTopLevelItem(QTreeWidgetItem([item, details]))

    def _on_browse(self):
        start = self.path_edit.text().strip() or str(Path.home())
        path = QFileDialog.getExistingDirectory(
            self, "Select Annotation Bundle Folder", start
        )
        if path:
            self.path_edit.setText(path)

    def _on_media_browse(self):
        start = self.media_edit.text().strip() or str(Path.home())
        path = QFileDialog.getExistingDirectory(
            self, "Select Folder Containing the Media", start
        )
        if path:
            self.media_edit.setText(path)

    def _on_path_changed(self):
        self.contents_tree.clear()
        self._preview = None
        self.conflict_label.setVisible(False)
        self.overwrite_checkbox.setVisible(False)
        self.media_group.setVisible(False)

        path = self.path_edit.text().strip()
        if not path:
            self._validate()
            return

        try:
            preview = bundle.build_import_preview(self._workspace, Path(path))
        except bundle.BundleError as e:
            self._add_row("Not a valid bundle", str(e))
            self._validate()
            return

        self._preview = preview
        self._add_row(
            "Dataset",
            f"{preview.dataset_name} ({preview.dataset_type})"
            + (" [multi-arena]" if preview.multi_arena else ""),
        )
        media_detail = f"{preview.num_media} {preview.media_kind}"
        media_detail += " (included)" if preview.media_included else " (reference only)"
        self._add_row("Media", media_detail)
        self._add_row("Project", preview.project_name)
        self._add_row("Pose labels", "yes" if preview.pose_present else "no")
        self._add_row(
            "Behavior labels", "yes" if preview.behavior_present else "no"
        )
        self.contents_tree.resizeColumnToContents(0)

        # Reference-only bundles need the importer to point at the media.
        if not preview.media_included:
            self.media_group.setVisible(True)
            if not self.media_edit.text().strip():
                self.media_edit.setText(preview.original_base_data_path)

        if preview.has_conflicts:
            existing = []
            if preview.dataset_exists:
                existing.append(f"dataset '{preview.dataset_name}'")
            if preview.project_exists:
                existing.append(f"project '{preview.project_name}'")
            if preview.pose_labels_exist:
                existing.append("pose labels")
            if preview.behavior_labels_exist:
                existing.append("behavior labels")
            self.conflict_label.setText(
                "Already present in this workspace: "
                + ", ".join(existing)
                + ". Check the box below to overwrite them."
            )
            self.conflict_label.setVisible(True)
            self.overwrite_checkbox.setVisible(True)

        self._validate()

    def _validate(self):
        ok = self._preview is not None and (
            not self._preview.has_conflicts or self.overwrite_checkbox.isChecked()
        )
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(ok)

    # -- results -------------------------------------------------------

    @property
    def bundle_dir(self) -> Optional[Path]:
        if self._preview is None:
            return None
        return self._preview.bundle_dir

    @property
    def overwrite(self) -> bool:
        return self.overwrite_checkbox.isChecked()

    @property
    def media_source_dir(self) -> Optional[Path]:
        """Folder the importer picked for a reference-only bundle's media."""
        if self._preview is None or self._preview.media_included:
            return None
        text = self.media_edit.text().strip()
        return Path(text) if text else None
