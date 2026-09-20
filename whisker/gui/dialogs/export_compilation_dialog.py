"""Export several datasets into one package (a compilation).

A table lists every dataset in the workspace. Tick the ones to include, choose the project each
was labeled under, and tick which parts of each to copy: its videos/frames, pose labels, behavior
labels. See :mod:`whisker.core.compilation` for the format.
"""

from datetime import date
from pathlib import Path
from typing import List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from whisker.core import compilation as comp
from whisker.core.compilation import CompilationItem
from whisker.core.study.dataset import DatasetType
from whisker.core.workspace import Workspace

_DATASET, _TYPE, _PROJECT, _MEDIA, _POSE, _BEHAVIOR = range(6)
_CHECKABLE = Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled
_PLAIN = Qt.ItemFlag.ItemIsEnabled
_OFF = Qt.ItemFlag.NoItemFlags


class ExportCompilationDialog(QDialog):
    def __init__(
        self,
        workspace: Workspace,
        default_project_name: Optional[str] = None,
        preselect: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._ws = workspace
        self._rows: List[dict] = []
        self._loading = True
        self._name_touched = False

        self.setWindowTitle("Export Several Datasets")
        screen = QApplication.primaryScreen()
        dpi = screen.logicalDotsPerInch() / 96.0 if screen else 1.0
        self.setMinimumWidth(int(820 * dpi))

        root = QVBoxLayout(self)
        intro = QLabel(
            "Tick the datasets to put in one package. Each keeps its own folder inside it, in the same "
            "format as a single-dataset export, so the package imports back here or into full WHISKER."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: gray;")
        root.addWidget(intro)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Dataset", "Type", "Labeled under project", "Videos / frames", "Pose labels", "Behavior labels"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(_DATASET, QHeaderView.ResizeMode.Stretch)
        for col in (_TYPE, _PROJECT, _MEDIA, _POSE, _BEHAVIOR):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.table.setMinimumHeight(int(220 * dpi))
        self.table.itemChanged.connect(self._on_item_changed)
        root.addWidget(self.table)

        buttons = QHBoxLayout()
        all_btn, none_btn = QPushButton("Tick all"), QPushButton("Untick all")
        all_btn.clicked.connect(lambda: self._tick_all(True))
        none_btn.clicked.connect(lambda: self._tick_all(False))
        buttons.addWidget(all_btn)
        buttons.addWidget(none_btn)
        buttons.addStretch()
        self.summary = QLabel("")
        buttons.addWidget(self.summary)
        root.addLayout(buttons)

        self.notice = QLabel("")
        self.notice.setWordWrap(True)
        self.notice.setStyleSheet("color: #e67e22;")
        root.addWidget(self.notice)

        dest = QGroupBox("Save the package to")
        g = QGridLayout(dest)
        g.addWidget(QLabel("Folder:"), 0, 0, Qt.AlignmentFlag.AlignRight)
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        self.dest_edit = QLineEdit()
        self.dest_edit.setPlaceholderText("Choose where to save it...")
        self.dest_edit.textChanged.connect(self._update_state)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        h.addWidget(self.dest_edit)
        h.addWidget(browse)
        g.addWidget(row, 0, 1)
        g.addWidget(QLabel("Package name:"), 1, 0, Qt.AlignmentFlag.AlignRight)
        self.name_edit = QLineEdit(f"whisker_export_{date.today().isoformat()}")
        self.name_edit.textEdited.connect(lambda _t: setattr(self, "_name_touched", True))
        self.name_edit.textChanged.connect(self._update_state)
        g.addWidget(self.name_edit, 1, 1)
        g.addWidget(QLabel("Full path:"), 2, 0, Qt.AlignmentFlag.AlignRight)
        self.full_path = QLabel("")
        self.full_path.setWordWrap(True)
        self.full_path.setStyleSheet("color: gray;")
        g.addWidget(self.full_path, 2, 1)
        g.setColumnStretch(1, 1)
        root.addWidget(dest)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.ok_btn = self.button_box.button(QDialogButtonBox.StandardButton.Ok)
        self.ok_btn.setText("Export")
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        root.addWidget(self.button_box)

        self._populate(default_project_name, preselect)
        self._loading = False
        first = next((r for r in self._rows if r["include"].checkState() == Qt.CheckState.Checked), None)
        first_ds = self._ws.datasets.get(first["name"]) if first else None
        base = Path(first_ds.base_data_path).parent if first_ds and first_ds.base_data_path else None
        self.dest_edit.setText(str(base) if base and base.exists() else str(Path.home()))
        self._update_state()

    # ------------------------------------------------------------ building

    def _populate(self, default_project: Optional[str], preselect: Optional[str]):
        projects = sorted(self._ws.projects.keys())
        names = sorted(self._ws.datasets.keys(), key=str.lower)
        self.table.setRowCount(len(names))
        for r, name in enumerate(names):
            ds = self._ws.datasets.get(name)
            kind = "videos" if ds.type == DatasetType.VIDEO_COLLECTION else "frames"

            include = QTableWidgetItem(name)
            include.setFlags(_CHECKABLE)
            include.setCheckState(Qt.CheckState.Checked if name == preselect else Qt.CheckState.Unchecked)
            self.table.setItem(r, _DATASET, include)

            type_item = QTableWidgetItem(ds.type.value.replace("_", " ").title())
            type_item.setFlags(_PLAIN)
            self.table.setItem(r, _TYPE, type_item)

            combo = QComboBox()
            combo.addItems(projects)
            guess = comp.guess_project(self._ws, name, default_project)
            if guess in projects:
                combo.setCurrentText(guess)
            combo.currentIndexChanged.connect(self._update_state)
            self.table.setCellWidget(r, _PROJECT, combo)

            media = QTableWidgetItem(f"{len(ds.files)} {kind}")
            has_pose = self._ws.pose_labels.has_pose_labels(name)
            has_beh = self._ws.behavior_labels.has_behavior_labels(name)
            pose_n = len(self._ws.pose_labels.get_pose_labeled_image_keys_from_summary(name)) if has_pose else 0
            beh_n = len(self._ws.behavior_labels.get_behavior_labeled_video_keys(name)) if has_beh else 0
            pose = QTableWidgetItem(f"{pose_n} labeled frames" if has_pose else "none")
            beh = QTableWidgetItem(f"{beh_n} labeled videos" if has_beh else "none")
            for col, item in ((_MEDIA, media), (_POSE, pose), (_BEHAVIOR, beh)):
                self.table.setItem(r, col, item)
            self._rows.append({
                "name": name, "include": include, "combo": combo, "media": media, "pose": pose, "behavior": beh,
                "has_pose": has_pose, "has_behavior": has_beh,
            })
            self._sync_row(self._rows[-1])

    def _sync_row(self, row: dict):
        """A part's checkbox is live only while its dataset is ticked and the part exists;
        otherwise the cell shows no checkbox at all."""
        on = row["include"].checkState() == Qt.CheckState.Checked
        row["combo"].setEnabled(on)
        for key, present in (("media", True), ("pose", row["has_pose"]), ("behavior", row["has_behavior"])):
            item = row[key]
            has_box = item.data(Qt.ItemDataRole.CheckStateRole) is not None
            if on and present:
                if not has_box:
                    item.setFlags(_CHECKABLE)
                    item.setCheckState(Qt.CheckState.Checked)
            else:
                item.setData(Qt.ItemDataRole.CheckStateRole, None)
                item.setFlags(_PLAIN if present else _OFF)

    # ------------------------------------------------------------ state

    def _on_item_changed(self, item: QTableWidgetItem):
        if self._loading:
            return
        self._loading = True
        try:
            for row in self._rows:
                if row["include"] is item:
                    self._sync_row(row)
        finally:
            self._loading = False
        self._update_state()

    def _tick_all(self, on: bool):
        self._loading = True
        for row in self._rows:
            row["include"].setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
            self._sync_row(row)
        self._loading = False
        self._update_state()

    def _browse(self):
        start = self.dest_edit.text().strip() or str(Path.home())
        path = QFileDialog.getExistingDirectory(self, "Choose where to save the package", start)
        if path:
            self.dest_edit.setText(path)

    def _checked(self, item: QTableWidgetItem) -> bool:
        return bool(item.flags() & Qt.ItemFlag.ItemIsUserCheckable) and item.checkState() == Qt.CheckState.Checked

    def items(self) -> List[CompilationItem]:
        out = []
        for row in self._rows:
            if row["include"].checkState() != Qt.CheckState.Checked:
                continue
            out.append(CompilationItem(
                dataset_name=row["name"],
                project_name=row["combo"].currentText(),
                include_media=self._checked(row["media"]),
                include_pose=self._checked(row["pose"]),
                include_behavior=self._checked(row["behavior"]),
            ))
        return out

    @property
    def destination_dir(self) -> Path:
        return Path(self.dest_edit.text().strip())

    @property
    def name(self) -> str:
        return self.name_edit.text().strip()

    def _update_state(self, *_):
        if self._loading:
            return
        items = self.items()
        copying = sum(len(self._ws.datasets.get(i.dataset_name).files) for i in items if i.include_media)
        self.summary.setText(f"{len(items)} dataset(s) ticked" + (f" · {copying} media file(s) will be copied" if copying else ""))

        problems = []
        if not self._ws.projects.keys():
            problems.append("This workspace has no projects, and every export records the project the dataset was labeled under. Create a project first.")
        if items and not self.name:
            problems.append("Give the package a name.")
        elif self.name and not comp.is_valid_compilation_name(self.name):
            problems.append("The name can't contain \\ / : * ? \" < > | and can't start or end with a space or dot.")
        self.notice.setText("\n".join(problems))
        self.full_path.setText(str(self.destination_dir / self.name) if self.dest_edit.text().strip() and self.name else "")
        self.ok_btn.setEnabled(bool(items) and not problems and bool(self.dest_edit.text().strip()))
