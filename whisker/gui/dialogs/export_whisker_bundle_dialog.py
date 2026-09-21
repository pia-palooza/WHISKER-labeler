"""Export a Whisker bundle: pick datasets (their media and labels always go in), and optionally projects and
prediction results, then where to save the .zip. The result opens in Whisker with Import Whisker Bundle.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor, QPalette
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from whisker.core import whisker_bundle as wb

_UNSAFE_IN_FILENAME = re.compile(r'[\\/:*?"<>|\[\]]')


def suggest_filename(dataset_names: Sequence[str], today: Optional[datetime] = None) -> str:
    """``whisker_bundle_<dataset>_<yyyymmdd>.zip``, with brackets and characters files can't hold turned into ``_``."""
    stamp = (today or datetime.now()).strftime("%Y%m%d")
    if len(dataset_names) == 1:
        return f"whisker_bundle_{_UNSAFE_IN_FILENAME.sub('_', dataset_names[0]).strip()}_{stamp}.zip"
    return f"whisker_bundle_{len(dataset_names)}_datasets_{stamp}.zip" if dataset_names else f"whisker_bundle_{stamp}.zip"


class ExportWhiskerBundleDialog(QDialog):
    def __init__(
        self,
        workspace,
        active_project: Optional[str] = None,
        preselect: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Export Whisker Bundle")
        self.setMinimumWidth(700)
        self.resize(760, 800)
        self._workspace = workspace
        self._plan: Optional[wb.ExportPlan] = None
        self._name_touched = False
        self._run_checks: dict = {}

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Exports datasets, with their media and labels, as one Whisker bundle (.zip) that Whisker "
            "or Whisker Labeler can import."
        ))

        datasets_box = QGroupBox("Datasets")
        d_layout = QVBoxLayout(datasets_box)
        self.datasets_list = QListWidget()
        self.datasets_list.setMinimumHeight(120)
        for name in sorted(workspace.datasets.keys()):
            ds = workspace.datasets.get(name)
            item = QListWidgetItem(f"{name}  —  {ds.type.value.replace('_', ' ').title()}, {len(ds.files)} files" if ds else name)
            item.setData(Qt.ItemDataRole.UserRole, name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if name == preselect else Qt.CheckState.Unchecked)
            self.datasets_list.addItem(item)
        d_layout.addWidget(self.datasets_list)
        layout.addWidget(datasets_box)

        self.projects_box = QGroupBox("Projects to include (optional)")
        p_layout = QVBoxLayout(self.projects_box)
        self.projects_list = QListWidget()
        self.projects_list.setMaximumHeight(90)
        for name in sorted(workspace.projects.keys()):
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if name == active_project else Qt.CheckState.Unchecked)
            self.projects_list.addItem(item)
        p_layout.addWidget(self.projects_list)
        layout.addWidget(self.projects_box)

        self.runs_box = QGroupBox("Prediction results to include (optional; these can be large)")
        r_layout = QVBoxLayout(self.runs_box)
        self.runs_list = QListWidget()
        self.runs_list.setMaximumHeight(90)
        r_layout.addWidget(self.runs_list)
        layout.addWidget(self.runs_box)

        self.contents_box = QGroupBox("What will be in the bundle")
        c_layout = QVBoxLayout(self.contents_box)
        self.contents_tree = QTreeWidget()
        self.contents_tree.setHeaderLabels(["Item", "Files", "Size"])
        self.contents_tree.setMinimumHeight(190)
        header = self.contents_tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)     # long run names must not be cut off
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        c_layout.addWidget(self.contents_tree)
        layout.addWidget(self.contents_box, 1)

        dest = QHBoxLayout()
        dest.addWidget(QLabel("Save as:"))
        self.dest_edit = QLineEdit()
        self.dest_edit.setPlaceholderText("Choose where to save the bundle...")
        self.browse_btn = QPushButton("Browse...")
        dest.addWidget(self.dest_edit, 1)
        dest.addWidget(self.browse_btn)
        layout.addLayout(dest)

        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.export_btn = self.button_box.button(QDialogButtonBox.StandardButton.Ok)
        self.export_btn.setText("Export")
        layout.addWidget(self.button_box)

        self._default_dir = self._initial_dir()
        self._refresh_runs()
        self._refresh()
        # Connect last: a slot that raises aborts the whole app, so nothing fires while the dialog is half built.
        self.datasets_list.itemChanged.connect(lambda _i: self._on_datasets_changed())
        self.projects_list.itemChanged.connect(lambda _i: self._refresh())
        self.runs_list.itemChanged.connect(lambda _i: self._refresh())
        self.dest_edit.textEdited.connect(self._on_name_typed)
        self.dest_edit.textChanged.connect(lambda _t: self._refresh())
        self.browse_btn.clicked.connect(self._on_browse)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

    @staticmethod
    def _initial_dir() -> Path:
        docs = Path.home() / "Documents"
        return docs if docs.is_dir() else Path.home()

    # -- what is chosen -----------------------------------------------------------------------------
    @staticmethod
    def _checked(widget: QListWidget) -> List[str]:
        return [widget.item(i).data(Qt.ItemDataRole.UserRole) for i in range(widget.count())
                if widget.item(i).checkState() == Qt.CheckState.Checked]

    def selected_datasets(self) -> List[str]:
        return self._checked(self.datasets_list)

    def selected_projects(self) -> List[str]:
        return self._checked(self.projects_list)

    def selected_runs(self) -> List[Tuple[str, str]]:
        return [tuple(k.split("/", 1)) for k in self._checked(self.runs_list)]

    @property
    def destination(self) -> Optional[Path]:
        text = self.dest_edit.text().strip()
        if not text:
            return None
        path = Path(text)
        return path if path.suffix.lower() == ".zip" else path.with_name(path.name + ".zip")

    @property
    def plan(self) -> Optional[wb.ExportPlan]:
        return self._plan

    # -- keeping the dialog honest ------------------------------------------------------------------
    def _on_datasets_changed(self):
        self._refresh_runs()
        if not self._name_touched:
            self.dest_edit.blockSignals(True)
            names = self.selected_datasets()
            self.dest_edit.setText(str(self._default_dir / suggest_filename(names)) if names else "")
            self.dest_edit.blockSignals(False)
        self._refresh()

    def _on_name_typed(self, _text: str):
        self._name_touched = True

    def _refresh_runs(self):
        """Prediction runs that have results for the ticked datasets; what was ticked stays ticked."""
        ticked = set(self._checked(self.runs_list))
        self.runs_list.blockSignals(True)
        self.runs_list.clear()
        try:
            runs = wb.list_prediction_runs(self._workspace, self.selected_datasets())
        except Exception:
            runs = []
        for run in runs:
            key = f"{run.workflow}/{run.run_name}"
            item = QListWidgetItem(f"{run.run_name}  —  {run.workflow.replace('_', ' ')}, {len(run.datasets)} dataset(s)")
            item.setData(Qt.ItemDataRole.UserRole, key)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if key in ticked else Qt.CheckState.Unchecked)
            self.runs_list.addItem(item)
        self.runs_box.setVisible(bool(runs))
        self.runs_list.blockSignals(False)

    def _refresh(self):
        self._plan = None
        names = self.selected_datasets()
        problem, text = "", ""
        if not names:
            text = "Choose at least one dataset."
            self._populate_contents(None)
        else:
            try:
                plan = wb.build_export_plan(self._workspace, names, self.selected_projects(), self.selected_runs())
                self._plan = plan
                self._populate_contents(plan)
                text = f"{len(plan.entries)} files, {wb.format_bytes(plan.total_bytes)}."
                if plan.media_missing:
                    where = ", ".join(f"'{n}' ({len(v)})" for n, v in plan.missing.items() if v)
                    problem = (f"{plan.media_missing} media file(s) can't be found on disk and won't be in the bundle: {where}. "
                               "They are listed as missing in the bundle, as Whisker does.")
            except Exception as e:                      # never let a slot raise: Qt would abort the whole app
                problem = str(e)
                self._plan = None
                self._populate_contents(None)
        dest = self.destination
        ready = bool(self._plan) and dest is not None and dest.parent.is_dir()
        if names and dest is None:
            text += " Choose where to save it."
        elif dest is not None and not dest.parent.is_dir():
            problem = problem or f"The folder '{dest.parent}' doesn't exist."
        self.summary_label.setText((text + ("\n" if text and problem else "") + problem).strip())
        self.summary_label.setStyleSheet("color: #b00020;" if problem and not self._plan else "")
        self.export_btn.setEnabled(ready)

    # -- what will be in the bundle -----------------------------------------------------------------
    def _populate_contents(self, plan: Optional[wb.ExportPlan]):
        """Show, under the choices, every part of the bundle: for each ticked dataset its manifest, media, labels and
        prediction results, then the projects. Things that exist but aren't ticked are listed as not included."""
        tree = self.contents_tree
        tree.clear()
        muted = QBrush(self.palette().color(QPalette.ColorRole.PlaceholderText))
        if plan is None:
            self._row(tree, "Tick a dataset above to see everything that will be exported.", muted=muted)
            return
        self._row(tree, "Bundle description", "whisker_bundle.json", tooltip="Says what is inside the bundle. Always included.")
        parents, totals = {}, {}
        for group in plan.groups():
            key = group.dataset or "\0projects"
            if key not in parents:
                parents[key] = self._row(tree, group.dataset or "Projects", bold=True)
                totals[key] = 0
            totals[key] += group.bytes
            self._row(parents[key], group.title, self._files_text(group), wb.format_bytes(group.bytes) if group.files else "",
                      tooltip=self._files_tooltip(group), problem=group.problem)
        for key, parent in parents.items():
            parent.setText(2, wb.format_bytes(totals[key]))
        hints = self._not_included()
        if hints:
            head = self._row(tree, "Not included", muted=muted)
            for hint in hints:
                self._row(head, hint, muted=muted)
        tree.expandAll()

    def _not_included(self) -> List[str]:
        chosen = set(self.selected_runs())
        try:
            runs = wb.list_prediction_runs(self._workspace, self.selected_datasets())
        except Exception:
            runs = []
        hints = [f"Prediction results — {r.run_name}  (tick it above to include)" for r in runs
                 if (r.workflow, r.run_name) not in chosen]
        if not self.selected_projects() and self.projects_list.count():
            hints.append("Project definitions  (tick one above to include)")
        return hints

    @staticmethod
    def _files_text(group: wb.PlanGroup) -> str:
        names = [f for f, _n in group.files]
        text = "" if group.kind == "media" else ", ".join(names[:3]) + (f" (+{len(names) - 3} more)" if len(names) > 3 else "")
        if group.note:
            text = f"{text}  —  {group.note}" if text else group.note
        return text

    @staticmethod
    def _files_tooltip(group: wb.PlanGroup) -> str:
        names = [f for f, _n in group.files]
        lines = names[:30] + ([f"...and {len(names) - 30} more"] if len(names) > 30 else [])
        if group.note:
            lines.append(group.note)
        return "\n".join(lines)

    def _row(self, parent, title: str, files: str = "", size: str = "", tooltip: str = "", problem: bool = False,
             bold: bool = False, muted: Optional[QBrush] = None) -> QTreeWidgetItem:
        item = QTreeWidgetItem(parent, [title, files, size])
        item.setTextAlignment(2, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if bold:
            font = item.font(0)
            font.setBold(True)
            for column in range(3):
                item.setFont(column, font)
        brush = muted or (QBrush(QColor("#b00020")) if problem else None)
        if brush is not None:
            for column in range(3):
                item.setForeground(column, brush)
        if tooltip:
            for column in range(3):
                item.setToolTip(column, tooltip)
        return item

    def _on_browse(self):
        start = self.dest_edit.text().strip() or str(self._default_dir / suggest_filename(self.selected_datasets()))
        path, _ = QFileDialog.getSaveFileName(self, "Save Whisker bundle", start, "Whisker bundle (*.zip)")
        if path:
            self._name_touched = True
            self.dest_edit.setText(path)

    def accept(self):
        if self._plan is None or self.destination is None:
            QMessageBox.warning(self, "Can't Export", "Choose at least one dataset and where to save the bundle.")
            return
        super().accept()
