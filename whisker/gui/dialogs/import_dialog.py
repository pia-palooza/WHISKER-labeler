# UPDATE_FILE: whisker/gui/dialogs/import_dialog.py
#
# Popup for importing existing projects and labels into the labeler's own
# workspace. The user points at a source folder laid out like a WHISKER
# workspace (or a labeler export): projects/*.json, datasets/<name>/manifest.json,
# workflows/<wf>/labels/<name>/labels.h5. The source is only READ; everything is
# copied into the labeler's internal workspace, so the source (e.g. a full
# WHISKER workspace) is never modified.
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QFileDialog, QMessageBox, QListWidget, QListWidgetItem, QGroupBox,
)

POSE_WF = "pose_estimation"
BEHAVIOR_WF = "behavior_classification"


@dataclass
class SourceDatasetInfo:
    name: str
    type: str = ""
    has_pose: bool = False
    has_behavior: bool = False


@dataclass
class ImportSelection:
    source: Path
    projects: List[str] = field(default_factory=list)
    datasets: List[str] = field(default_factory=list)


def _labels_path(source: Path, wf: str, name: str) -> Path:
    return source / "workflows" / wf / "labels" / name / "labels.h5"


def scan_source(source: Path) -> tuple[List[str], List[SourceDatasetInfo]]:
    """Read-only scan of a source workspace/export folder."""
    projects: List[str] = []
    proj_dir = source / "projects"
    if proj_dir.is_dir():
        projects = sorted(p.stem for p in proj_dir.glob("*.json"))

    datasets: List[SourceDatasetInfo] = []
    ds_dir = source / "datasets"
    if ds_dir.is_dir():
        for d in sorted(p for p in ds_dir.iterdir() if p.is_dir()):
            manifest = d / "manifest.json"
            if not manifest.is_file():
                continue
            dtype = ""
            try:
                dtype = json.loads(manifest.read_text(encoding="utf-8")).get("type", "")
            except Exception:
                pass
            datasets.append(SourceDatasetInfo(
                name=d.name,
                type=dtype,
                has_pose=_labels_path(source, POSE_WF, d.name).is_file(),
                has_behavior=_labels_path(source, BEHAVIOR_WF, d.name).is_file(),
            ))
    return projects, datasets


class ImportDialog(QDialog):
    """Select projects and datasets (with their labels) to import."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import Projects / Labels")
        self._selection: Optional[ImportSelection] = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Choose a source folder (a WHISKER workspace or a labeler export). "
            "Its projects, datasets, and labels are copied into this labeler — "
            "the source is never changed."
        ))

        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("Source Folder:"))
        self.source_edit = QLineEdit()
        self.source_edit.textChanged.connect(self._rescan)
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse)
        src_row.addWidget(self.source_edit, 1)
        src_row.addWidget(browse)
        layout.addLayout(src_row)

        lists_row = QHBoxLayout()

        proj_group = QGroupBox("Projects")
        proj_layout = QVBoxLayout(proj_group)
        self.projects_list = QListWidget()
        proj_layout.addWidget(self.projects_list)
        lists_row.addWidget(proj_group)

        ds_group = QGroupBox("Datasets (with labels)")
        ds_layout = QVBoxLayout(ds_group)
        self.datasets_list = QListWidget()
        ds_layout.addWidget(self.datasets_list)
        lists_row.addWidget(ds_group)

        layout.addLayout(lists_row)

        btn_row = QHBoxLayout()
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self._select_all)
        btn_row.addWidget(self.select_all_btn)
        btn_row.addStretch()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        self.import_btn = QPushButton("Import")
        self.import_btn.setDefault(True)
        self.import_btn.clicked.connect(self._on_import)
        btn_row.addWidget(cancel)
        btn_row.addWidget(self.import_btn)
        layout.addLayout(btn_row)

        self.setMinimumSize(640, 420)

    def _browse(self):
        path = QFileDialog.getExistingDirectory(self, "Select Source Folder")
        if path:
            self.source_edit.setText(path)

    def _rescan(self, text: str):
        self.projects_list.clear()
        self.datasets_list.clear()
        source = Path(text.strip())
        if not text.strip() or not source.is_dir():
            return

        projects, datasets = scan_source(source)
        for name in projects:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.projects_list.addItem(item)

        for ds in datasets:
            tags = []
            if ds.has_pose:
                tags.append("pose")
            if ds.has_behavior:
                tags.append("behavior")
            label = ds.name + (f"   [{ds.type}]" if ds.type else "")
            label += f"   ({', '.join(tags)} labels)" if tags else "   (no labels)"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, ds.name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.datasets_list.addItem(item)

        if not projects and not datasets:
            QMessageBox.information(
                self, "Nothing Found",
                "No projects or datasets were found in that folder.\n\n"
                "Expected a 'projects/' folder and/or 'datasets/<name>/manifest.json'.",
            )

    def _select_all(self):
        for lst in (self.projects_list, self.datasets_list):
            for i in range(lst.count()):
                lst.item(i).setCheckState(Qt.CheckState.Checked)

    def _checked(self, lst: QListWidget, use_data: bool = False) -> List[str]:
        out = []
        for i in range(lst.count()):
            item = lst.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                out.append(item.data(Qt.ItemDataRole.UserRole) if use_data else item.text())
        return out

    def _on_import(self):
        source = Path(self.source_edit.text().strip())
        if not source.is_dir():
            QMessageBox.warning(self, "Invalid Source", "Please choose a valid source folder.")
            return
        projects = self._checked(self.projects_list)
        datasets = self._checked(self.datasets_list, use_data=True)
        if not projects and not datasets:
            QMessageBox.information(self, "Nothing Selected",
                                    "Check at least one project or dataset to import.")
            return
        self._selection = ImportSelection(source=source, projects=projects, datasets=datasets)
        self.accept()

    def get_selection(self) -> Optional[ImportSelection]:
        return self._selection
