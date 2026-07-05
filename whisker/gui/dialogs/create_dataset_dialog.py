from pathlib import Path
from typing import List, Optional, Tuple

from PyQt6.QtWidgets import (
    QWidget,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QDialogButtonBox,
    QFileDialog,
)

from whisker.core.workspace import Workspace
from whisker.core.study.dataset import DatasetType
from whisker.gui.constants import VIDEO_EXTENSIONS, IMAGE_EXTENSIONS
from PyQt6.QtWidgets import QMessageBox


class CreateDatasetDialog(QDialog):
    """A custom dialog for creating a new dataset from a folder."""

    def __init__(self, workspace: Workspace, parent: QWidget | None = None):
        super().__init__(parent)
        self._workspace = workspace
        self.found_files: List[Path] = []
        self.setWindowTitle("Add New Dataset")

        layout = QVBoxLayout(self)

        # --- Name and Type ---
        layout.addWidget(QLabel("Dataset Name:"))
        self.name_edit = QLineEdit()
        self.name_edit.textChanged.connect(self._validate_state)
        layout.addWidget(self.name_edit)

        # --- Folder Selection ---
        folder_layout = QHBoxLayout()
        self.folder_label = QLabel("<i>No folder selected...</i>")
        select_folder_btn = QPushButton("Select Folder...")
        select_folder_btn.clicked.connect(self._select_folder)
        folder_layout.addWidget(QLabel("Source Folder:"))
        folder_layout.addWidget(self.folder_label)
        folder_layout.addStretch()
        folder_layout.addWidget(select_folder_btn)
        layout.addLayout(folder_layout)

        # --- File Preview ---
        layout.addWidget(QLabel("File Preview:"))
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        layout.addWidget(self.file_list)
        self.inferred_type = None

        # --- OK and Cancel Buttons ---
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        self._validate_state()  # Initially disable OK button

    def _select_folder(self):
        """Opens a dialog to select a directory and then scans it for media."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Source Folder", str(self._workspace.base_dir)
        )
        if not folder:
            return

        folder_path = Path(folder)
        video_files: List[Path] = []
        image_files: List[Path] = []

        # Recursively find all supported files
        for ext in VIDEO_EXTENSIONS:
            video_files.extend(folder_path.rglob(f"*{ext}"))
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(folder_path.rglob(f"*{ext}"))

        # Determine dataset type based on found files
        has_videos = bool(video_files)
        has_images = bool(image_files)
        inferred_type = None

        if has_videos and not has_images:
            inferred_type = DatasetType.VIDEO_COLLECTION
        elif has_images and not has_videos:
            inferred_type = DatasetType.IMAGE_COLLECTION
        elif has_videos and has_images:
            QMessageBox.critical(
                self,
                "Error",
                "Cannot infer dataset type from a the provided folder. "
                "Please select a folder containing only videos or only images.",
            )
            return

        # Only update state if the folder is valid
        if inferred_type is not None:
            self.folder_path = folder_path
            self.folder_label.setText(f"<b>{self.folder_path.name}</b>")
            self.name_edit.setText(self.folder_path.name)
            self.file_list.clear()
            self.found_files = sorted(video_files + image_files)

            # Display up to 100 found files for a quick preview
            for path in self.found_files[:100]:
                self.file_list.addItem(str(path.relative_to(self.folder_path)))
            if len(self.found_files) > 100:
                self.file_list.addItem(
                    f"...and {len(self.found_files) - 100} more files."
                )

            self.inferred_type = inferred_type

        self._validate_state()

    def _validate_state(self):
        """Enables or disables the OK button based on the dialog's state."""
        is_name_valid = bool(self.name_edit.text().strip())
        is_folder_valid = bool(self.found_files)
        # DEV_NOTE: Added a check to ensure we've successfully inferred a type.
        # This prevents creating a dataset from a mixed-content folder for now.
        is_type_valid = self.inferred_type is not None

        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(
            is_name_valid and is_folder_valid and is_type_valid
        )

    def get_dataset_info(self) -> Optional[Tuple[str, DatasetType, Path]]:
        """Returns the entered data if the dialog was accepted."""
        if self.result() == QDialog.DialogCode.Accepted:
            assert self.inferred_type is not None
            assert self.folder_path is not None

            name = self.name_edit.text().strip()
            return name, self.inferred_type, self.folder_path
        return None
