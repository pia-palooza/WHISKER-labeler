# UPDATE_FILE: whisker/gui/dialogs/new_project_dataset_dialog.py
#
# Popup for adding a dataset to the labeler's own workspace. The user points at
# a folder of videos or frames and either creates a NEW project (defining body
# parts, identities, behaviors, optional skeleton) or REUSES an existing project
# from the labeler workspace. Everything is written to the labeler's internal
# workspace -- the source media folder and the full WHISKER workspace are never
# modified.
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QMessageBox, QComboBox, QFrame,
)

from whisker.core.dataset import (
    DatasetType, VIDEO_FILE_EXTENSIONS, IMAGE_FILE_EXTENSIONS,
)
from whisker.core.project import Project

NEW_PROJECT_SENTINEL = "➕ Create new project…"


@dataclass
class NewProjectDatasetSpec:
    data_folder: Path
    dataset_name: str
    project_name: str
    use_existing_project: bool = False
    body_parts: List[str] = field(default_factory=list)
    identities: List[str] = field(default_factory=list)
    behaviors: List[str] = field(default_factory=list)
    skeleton: List[Tuple[str, str]] = field(default_factory=list)
    dataset_type: Optional[DatasetType] = None
    files: List[str] = field(default_factory=list)


def detect_dataset(folder: Path) -> Tuple[DatasetType, List[str]]:
    """
    Inspect a folder and decide whether it is a video or image/frame dataset,
    returning the dataset type and the media files relative to the folder.
    Videos take priority if both are present.
    """
    def _collect(exts) -> List[str]:
        found: set[str] = set()
        for ext in exts:
            for f in folder.rglob(f"*{ext}"):
                found.add(str(f.relative_to(folder)).replace("\\", "/"))
            for f in folder.rglob(f"*{ext.upper()}"):
                found.add(str(f.relative_to(folder)).replace("\\", "/"))
        return sorted(found)

    videos = _collect(VIDEO_FILE_EXTENSIONS)
    if videos:
        return DatasetType.VIDEO_COLLECTION, videos

    images = _collect(IMAGE_FILE_EXTENSIONS)
    if images:
        has_subdirs = any("/" in rel for rel in images)
        return (
            DatasetType.FRAME_SUBSET if has_subdirs else DatasetType.IMAGE_COLLECTION
        ), images

    raise ValueError("No video or image files were found in that folder.")


def _parse_list(text: str) -> List[str]:
    return [t.strip() for t in text.replace("\n", ",").split(",") if t.strip()]


def _parse_skeleton(text: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for token in _parse_list(text):
        if "-" in token:
            a, b = token.split("-", 1)
            a, b = a.strip(), b.strip()
            if a and b:
                pairs.append((a, b))
    return pairs


class NewProjectDatasetDialog(QDialog):
    """Collects everything needed to create a dataset (and optionally a project)."""

    def __init__(self, parent=None, projects: Optional[Dict[str, Project]] = None,
                 existing_datasets: Optional[set] = None):
        super().__init__(parent)
        self.setWindowTitle("New Project / Dataset")
        self._projects: Dict[str, Project] = projects or {}
        self._existing_datasets = existing_datasets or set()
        self._spec: Optional[NewProjectDatasetSpec] = None
        self._build_ui()
        self._update_project_mode()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Point at a folder of videos (for behavior labeling) or frames/"
            "images (for pose labeling)."
        ))

        grid = QGridLayout()
        row = 0
        grid.addWidget(QLabel("Data Folder:"), row, 0)
        self.folder_edit = QLineEdit()
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse_folder)
        grid.addWidget(self.folder_edit, row, 1)
        grid.addWidget(browse, row, 2)
        row += 1

        grid.addWidget(QLabel("Dataset Name:"), row, 0)
        self.dataset_edit = QLineEdit()
        grid.addWidget(self.dataset_edit, row, 1, 1, 2)
        row += 1
        layout.addLayout(grid)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep)

        # Project: reuse existing or create new
        proj_grid = QGridLayout()
        r = 0
        proj_grid.addWidget(QLabel("Project:"), r, 0)
        self.project_mode_combo = QComboBox()
        self.project_mode_combo.addItem(NEW_PROJECT_SENTINEL)
        self.project_mode_combo.addItems(sorted(self._projects.keys()))
        self.project_mode_combo.currentIndexChanged.connect(self._update_project_mode)
        proj_grid.addWidget(self.project_mode_combo, r, 1)
        r += 1

        proj_grid.addWidget(QLabel("New Project Name:"), r, 0)
        self.project_name_edit = QLineEdit()
        proj_grid.addWidget(self.project_name_edit, r, 1)
        r += 1

        proj_grid.addWidget(QLabel("Body Parts:"), r, 0)
        self.bodyparts_edit = QLineEdit()
        self.bodyparts_edit.setPlaceholderText("e.g. nose, left_ear, right_ear, tail_base")
        proj_grid.addWidget(self.bodyparts_edit, r, 1)
        r += 1

        proj_grid.addWidget(QLabel("Identities:"), r, 0)
        self.identities_edit = QLineEdit()
        self.identities_edit.setPlaceholderText("e.g. resident, intruder")
        proj_grid.addWidget(self.identities_edit, r, 1)
        r += 1

        proj_grid.addWidget(QLabel("Behaviors:"), r, 0)
        self.behaviors_edit = QLineEdit()
        self.behaviors_edit.setPlaceholderText("e.g. attack, sniff, mount, groom")
        proj_grid.addWidget(self.behaviors_edit, r, 1)
        r += 1

        proj_grid.addWidget(QLabel("Skeleton (optional):"), r, 0)
        self.skeleton_edit = QLineEdit()
        self.skeleton_edit.setPlaceholderText("pairs like  nose-neck, neck-tail_base")
        proj_grid.addWidget(self.skeleton_edit, r, 1)
        r += 1
        layout.addLayout(proj_grid)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        create = QPushButton("Create")
        create.setDefault(True)
        create.clicked.connect(self._on_create)
        btn_row.addWidget(cancel)
        btn_row.addWidget(create)
        layout.addLayout(btn_row)

        self.setMinimumWidth(580)

    def _is_new_project(self) -> bool:
        return self.project_mode_combo.currentText() == NEW_PROJECT_SENTINEL

    def _update_project_mode(self):
        """Show editable definition fields for a new project; show the chosen
        existing project's definition (read-only) when reusing."""
        new_mode = self._is_new_project()
        self.project_name_edit.setEnabled(new_mode)

        def_fields = [
            self.bodyparts_edit, self.identities_edit,
            self.behaviors_edit, self.skeleton_edit,
        ]
        for f in def_fields:
            f.setReadOnly(not new_mode)

        if new_mode:
            return

        # Reusing an existing project: display its definition (read-only).
        project = self._projects.get(self.project_mode_combo.currentText())
        if not project:
            return
        self.project_name_edit.setText(project.name)
        self.bodyparts_edit.setText(", ".join(project.body_parts))
        self.identities_edit.setText(", ".join(project.identities))
        self.behaviors_edit.setText(", ".join(project.behaviors))
        self.skeleton_edit.setText(
            ", ".join(f"{a}-{b}" for a, b in project.skeleton)
        )

    def _browse_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Folder with Videos or Frames")
        if path:
            self.folder_edit.setText(path)
            if not self.dataset_edit.text().strip():
                self.dataset_edit.setText(Path(path).name)
            if self._is_new_project() and not self.project_name_edit.text().strip():
                self.project_name_edit.setText(Path(path).name)

    def _on_create(self):
        folder_str = self.folder_edit.text().strip()
        dataset_name = self.dataset_edit.text().strip()

        if not folder_str or not Path(folder_str).is_dir():
            QMessageBox.warning(self, "Invalid Folder", "Please choose a valid data folder.")
            return
        if not dataset_name:
            QMessageBox.warning(self, "Missing Name", "Please enter a dataset name.")
            return

        use_existing = not self._is_new_project()
        if use_existing:
            project = self._projects[self.project_mode_combo.currentText()]
            project_name = project.name
            body_parts = list(project.body_parts)
            identities = list(project.identities)
            behaviors = list(project.behaviors)
            skeleton = [tuple(p) for p in project.skeleton]
        else:
            project_name = self.project_name_edit.text().strip()
            if not project_name:
                QMessageBox.warning(self, "Missing Name", "Please enter a project name.")
                return
            body_parts = _parse_list(self.bodyparts_edit.text())
            identities = _parse_list(self.identities_edit.text())
            behaviors = _parse_list(self.behaviors_edit.text())
            skeleton = _parse_skeleton(self.skeleton_edit.text())

        folder = Path(folder_str)
        try:
            ds_type, files = detect_dataset(folder)
        except ValueError as e:
            QMessageBox.warning(self, "No Media Found", str(e))
            return

        # Light guidance only when creating a brand-new project.
        if not use_existing:
            if ds_type == DatasetType.VIDEO_COLLECTION and not behaviors:
                if not self._confirm("This is a video dataset but no behaviors were "
                                     "defined. Create anyway?"):
                    return
            if ds_type != DatasetType.VIDEO_COLLECTION and not (body_parts and identities):
                if not self._confirm("This is an image dataset but body parts and/or "
                                     "identities are missing. Create anyway?"):
                    return
            if project_name in self._projects and not self._confirm(
                f"A project named '{project_name}' already exists. Overwrite it?"
            ):
                return

        if dataset_name in self._existing_datasets and not self._confirm(
            f"A dataset named '{dataset_name}' already exists. Overwrite it?"
        ):
            return

        self._spec = NewProjectDatasetSpec(
            data_folder=folder,
            dataset_name=dataset_name,
            project_name=project_name,
            use_existing_project=use_existing,
            body_parts=body_parts,
            identities=identities,
            behaviors=behaviors,
            skeleton=skeleton,
            dataset_type=ds_type,
            files=files,
        )
        self.accept()

    def _confirm(self, message: str) -> bool:
        return QMessageBox.question(
            self, "Confirm", message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        ) == QMessageBox.StandardButton.Yes

    def get_spec(self) -> Optional[NewProjectDatasetSpec]:
        return self._spec
