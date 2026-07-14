"""Confirmation dialog for exporting a dataset's annotations as a bundle.

Shown when the user picks "Export Annotations..." on an image or frame dataset.
It previews exactly what will be written (project, manifest, label HDF5s, frame
count) and lets the user confirm the project the dataset was labeled under and
where the bundle should be saved.
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
    QComboBox,
    QCheckBox,
    QSizePolicy,
    QTreeWidget,
    QTreeWidgetItem,
)

from whisker.core import bundle
from whisker.core.workspace import Workspace, DatasetType


class ExportAnnotationsDialog(QDialog):
    """Preview + confirm dialog for exporting an annotation bundle."""

    def __init__(
        self,
        workspace: Workspace,
        dataset_name: str,
        default_project_name: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._workspace = workspace
        self._dataset_name = dataset_name
        self._plan: Optional[bundle.BundleExportPlan] = None

        self.setWindowTitle(f"Export Annotations — {dataset_name}")

        screen = QApplication.primaryScreen()
        dpi_scale = screen.logicalDotsPerInch() / 96.0 if screen else 1.0
        self.setMinimumWidth(int(620 * dpi_scale))

        main_layout = QVBoxLayout(self)

        # --- Top: dataset + project selectors ---
        grid = QGridLayout()
        grid.setSpacing(int(8 * dpi_scale))

        dataset = workspace.datasets.get(dataset_name)
        type_str = dataset.type.value if dataset else "?"
        num_files = len(dataset.files) if dataset else 0

        grid.addWidget(self._right_label("Dataset:"), 0, 0)
        grid.addWidget(
            QLabel(f"<b>{dataset_name}</b>  ({type_str}, {num_files} files)"), 0, 1
        )

        grid.addWidget(self._right_label("Labeled under project:"), 1, 0)
        self.project_combo = QComboBox()
        self.project_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        project_names = sorted(workspace.projects.keys())
        if project_names:
            self.project_combo.addItems(project_names)
            if default_project_name and default_project_name in project_names:
                self.project_combo.setCurrentText(default_project_name)
        else:
            self.project_combo.addItem("No projects found in workspace")
            self.project_combo.setEnabled(False)
        self.project_combo.currentIndexChanged.connect(self._refresh_preview)
        grid.addWidget(self.project_combo, 1, 1)
        grid.setColumnStretch(1, 1)
        main_layout.addLayout(grid)

        # --- Middle: what will be exported ---
        contents_group = QGroupBox("The following will be exported")
        contents_layout = QVBoxLayout(contents_group)
        self.contents_tree = QTreeWidget()
        self.contents_tree.setColumnCount(2)
        self.contents_tree.setHeaderLabels(["Item", "Details"])
        self.contents_tree.setRootIsDecorated(False)
        self.contents_tree.setMinimumHeight(int(170 * dpi_scale))
        contents_layout.addWidget(self.contents_tree)
        main_layout.addWidget(contents_group)

        # --- Include video files (video datasets only) ---
        self._is_video = bool(
            dataset and dataset.type == DatasetType.VIDEO_COLLECTION
        )
        self.include_media_checkbox = QCheckBox(
            "Include video files in the bundle (uncheck for a smaller, "
            "reference-only bundle)"
        )
        self.include_media_checkbox.setChecked(True)
        self.include_media_checkbox.setVisible(self._is_video)
        self.include_media_checkbox.toggled.connect(self._refresh_preview)
        main_layout.addWidget(self.include_media_checkbox)

        # --- Destination ---
        dest_group = QGroupBox("Save bundle to")
        dest_grid = QGridLayout(dest_group)
        dest_grid.setSpacing(int(8 * dpi_scale))

        dest_grid.addWidget(self._right_label("Folder:"), 0, 0)
        dest_container = QWidget()
        dest_hbox = QHBoxLayout(dest_container)
        dest_hbox.setContentsMargins(0, 0, 0, 0)
        self.dest_edit = QLineEdit()
        self.dest_edit.setPlaceholderText("Choose where to save the bundle...")
        self.dest_edit.textChanged.connect(self._update_full_path)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._on_browse)
        dest_hbox.addWidget(self.dest_edit)
        dest_hbox.addWidget(browse_btn)
        dest_grid.addWidget(dest_container, 0, 1)

        dest_grid.addWidget(self._right_label("Bundle name:"), 1, 0)
        self.name_edit = QLineEdit()
        self.name_edit.textChanged.connect(self._update_full_path)
        dest_grid.addWidget(self.name_edit, 1, 1)

        dest_grid.addWidget(self._right_label("Full path:"), 2, 0)
        self.full_path_label = QLabel("")
        self.full_path_label.setWordWrap(True)
        self.full_path_label.setStyleSheet("color: gray;")
        dest_grid.addWidget(self.full_path_label, 2, 1)
        dest_grid.setColumnStretch(1, 1)
        main_layout.addWidget(dest_group)

        # --- Buttons ---
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setText("Export")
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        main_layout.addWidget(self.button_box)

        # Sensible default destination: next to the original source data.
        if dataset and dataset.base_data_path:
            default_parent = Path(dataset.base_data_path).parent
            if default_parent.exists():
                self.dest_edit.setText(str(default_parent))

        self._refresh_preview()

    # -- helpers -------------------------------------------------------

    def _right_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        return label

    def _add_row(self, item: str, details: str):
        node = QTreeWidgetItem([item, details])
        self.contents_tree.addTopLevelItem(node)

    def _refresh_preview(self):
        """Rebuild the export plan for the currently selected project and refresh
        the contents preview + default bundle name."""
        self.contents_tree.clear()
        self._plan = None

        if not self.project_combo.isEnabled():
            self._update_full_path()
            return

        project_name = self.project_combo.currentText()
        try:
            self._plan = bundle.build_export_plan(
                self._workspace, self._dataset_name, project_name
            )
        except bundle.BundleError as e:
            self._add_row("Error", str(e))
            self._update_full_path()
            return

        plan = self._plan
        self._add_row(f"project/{plan.project.name}.json", "project definition")
        self._add_row(
            "dataset/manifest.json",
            f"{plan.num_media} {plan.media_kind} listed"
            + (", multi-arena" if plan.dataset.is_multi_arena else ""),
        )
        if plan.pose.present:
            self._add_row(
                "pose_labels/labels.h5",
                f"{plan.pose.num_labeled_frames} labeled frames"
                + (
                    f", {len(plan.pose.body_parts)} body parts"
                    if plan.pose.body_parts
                    else ""
                ),
            )
            if plan.pose.metadata_json is not None:
                self._add_row("pose_labels/metadata.json", "pose label metadata")
        if plan.behavior.present:
            self._add_row(
                "behavior_labels/labels.h5",
                f"{plan.behavior.num_labeled_videos} labeled keys",
            )
        if self._include_media():
            self._add_row(
                f"{plan.media_dirname}/",
                f"{plan.num_media} {plan.media_kind} files (copied)",
            )
        else:
            self._add_row(
                f"({plan.media_kind} not copied)",
                f"reference only — {plan.num_media} original paths recorded",
            )
        self._add_row("export_info.json", "bundle description")
        self.contents_tree.resizeColumnToContents(0)

        if not self.name_edit.text().strip():
            self.name_edit.setText(plan.default_bundle_name())
        self._update_full_path()

    def _on_browse(self):
        start = self.dest_edit.text().strip() or str(Path.home())
        path = QFileDialog.getExistingDirectory(
            self, "Select Folder to Save Bundle In", start
        )
        if path:
            self.dest_edit.setText(path)

    def _update_full_path(self):
        full = self.bundle_dir
        self.full_path_label.setText(str(full) if full else "")
        ok_enabled = (
            self._plan is not None
            and bool(self.dest_edit.text().strip())
            and bool(self.name_edit.text().strip())
        )
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(
            ok_enabled
        )

    def _include_media(self) -> bool:
        """Frames are always copied; videos only when the box is checked."""
        if not self._is_video:
            return True
        return self.include_media_checkbox.isChecked()

    # -- results -------------------------------------------------------

    @property
    def include_media(self) -> bool:
        return self._include_media()

    @property
    def selected_project_name(self) -> str:
        return self.project_combo.currentText()

    @property
    def plan(self) -> Optional[bundle.BundleExportPlan]:
        return self._plan

    @property
    def bundle_dir(self) -> Optional[Path]:
        parent = self.dest_edit.text().strip()
        name = self.name_edit.text().strip()
        if not parent or not name:
            return None
        return Path(parent) / name
