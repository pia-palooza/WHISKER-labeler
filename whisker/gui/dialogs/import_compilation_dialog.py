"""Import from a compilation: tick which datasets, and which parts of each.

One row per dataset. Datasets you already have start with their videos/frames unticked, so a
re-import brings in just the labels for the dataset of the same name; labels that would land on
existing labels are combined according to one choice below the table. See
:mod:`whisker.core.compilation`.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from whisker.core import bundle_import as bi
from whisker.core import compilation as comp
from whisker.core import manual_import as mi
from whisker.core.bundle_import import LabelPolicy
from whisker.core.workspace import Workspace

_NAME, _MEDIA, _POSE, _BEHAVIOR, _AS, _NOTES = range(6)
_CHECKABLE = Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled
_PLAIN = Qt.ItemFlag.ItemIsEnabled
_EDITABLE = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsEditable
_OFF = Qt.ItemFlag.NoItemFlags
_COLORS = {"ok": "#2e7d32", "warn": "#e67e22", "bad": "#c0392b", "muted": "gray"}


class ImportCompilationDialog(QDialog):
    def __init__(self, workspace: Workspace, root: Path, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._ws = workspace
        self._root = Path(root)
        self._loading = True
        self._rows: List[dict] = []
        self._media: Dict[str, Path] = {}                 # media folders the user located, by dataset
        self._media_error: Dict[str, str] = {}            # why the last folder they tried was rejected
        self._notes_cache: Dict[tuple, List[Tuple[str, str]]] = {}
        self.contents: Optional[comp.CompilationContents] = None
        self.selection: Optional[comp.CompilationSelection] = None

        self.setWindowTitle("Import from a Compilation")
        screen = QApplication.primaryScreen()
        dpi = screen.logicalDotsPerInch() / 96.0 if screen else 1.0
        self.setMinimumWidth(int(1050 * dpi))

        layout = QVBoxLayout(self)
        self.header = QLabel("")
        self.header.setWordWrap(True)
        layout.addWidget(self.header)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Dataset", "Videos / frames", "Pose labels", "Behavior labels", "Import as", "Notes"])
        head = self.table.horizontalHeader()
        for col in (_NAME, _MEDIA, _POSE, _BEHAVIOR):
            head.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        head.setSectionResizeMode(_AS, QHeaderView.ResizeMode.Interactive)
        self.table.setColumnWidth(_AS, int(170 * dpi))
        head.setSectionResizeMode(_NOTES, QHeaderView.ResizeMode.Stretch)     # gets whatever is left
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.table.setWordWrap(True)
        self.table.setMinimumHeight(int(220 * dpi))
        self.table.itemChanged.connect(self._on_changed)
        # Row heights depend on how wide Notes ends up, which isn't known until the dialog is shown or resized.
        head.sectionResized.connect(lambda *_: self.table.resizeRowsToContents())
        self.table.cellDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self.table)

        row = QHBoxLayout()
        all_btn, none_btn = QPushButton("Tick everything importable"), QPushButton("Untick all")
        all_btn.clicked.connect(lambda: self._tick_all(True))
        none_btn.clicked.connect(lambda: self._tick_all(False))
        row.addWidget(all_btn)
        row.addWidget(none_btn)
        row.addStretch()
        layout.addLayout(row)

        self.projects_check = QCheckBox()
        self.projects_check.toggled.connect(self._revalidate)
        layout.addWidget(self.projects_check)

        self.policy_box = QGroupBox("Where a dataset already has labels and the import brings more")
        pl = QVBoxLayout(self.policy_box)
        self._policy_radios: Dict[LabelPolicy, QRadioButton] = {}
        for policy, text in (
            (LabelPolicy.MERGE_EXISTING, "Combine — where both have a label, keep mine"),
            (LabelPolicy.MERGE_IMPORTED, "Combine — where both have a label, use the imported one"),
            (LabelPolicy.REPLACE, "Replace my labels with the imported ones (mine are lost)"),
        ):
            radio = QRadioButton(text)
            self._policy_radios[policy] = radio
            pl.addWidget(radio)
        self._policy_radios[LabelPolicy.MERGE_EXISTING].setChecked(True)
        for radio in self._policy_radios.values():
            radio.toggled.connect(self._revalidate)
        layout.addWidget(self.policy_box)

        self.problem_label = QLabel("")
        self.problem_label.setWordWrap(True)
        self.problem_label.setStyleSheet(f"color: {_COLORS['warn']};")
        layout.addWidget(self.problem_label)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.import_btn = self.button_box.button(QDialogButtonBox.StandardButton.Ok)
        self.import_btn.setText("Import")
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        self._load()

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(0, self.table.resizeRowsToContents)

    # ------------------------------------------------------------ loading

    def _load(self):
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.contents = comp.inspect_compilation(self._root)
            self.selection = comp.default_compilation_selection(self._ws, self.contents)
        except bi.BundleImportError as e:
            self.header.setText(str(e))
            self.header.setStyleSheet(f"color: {_COLORS['bad']};")
            self.import_btn.setEnabled(False)
            self.projects_check.setVisible(False)
            self.policy_box.setVisible(False)
            self._loading = False
            return
        finally:
            QApplication.restoreOverrideCursor()

        c, sel = self.contents, self.selection
        self.header.setText(
            f"<b>{c.name}</b> — {len(c.entries)} dataset(s)" + (f", exported {c.created_at[:10]}" if c.created_at else "")
            + "<br><span style='color:gray'>Tick what you want. Double-click a note that asks for a folder to locate the files.</span>"
        )
        if c.warnings:
            self.header.setText(self.header.text() + "<br>" + "<br>".join(c.warnings))

        self.table.setRowCount(len(c.entries))
        for r, entry in enumerate(c.entries):
            self._add_row(r, entry, sel.items[entry.name])

        projects = c.project_names
        self.projects_check.setVisible(bool(projects))
        if projects:
            bits = []
            for p in projects:
                entry = next(e for e in c.entries if e.ok and e.contents.project_obj and e.contents.project_obj.name == p)
                rel = bi.project_relation(self._ws, entry.contents)
                bits.append(p + {"new": " (new)", "identical": " (you already have it)", "different": " (yours differs — yours is kept)"}[rel])
            self.projects_check.setText("Also import the project(s) these datasets use: " + ", ".join(bits))
            self.projects_check.setChecked(sel.import_projects)
        self._loading = False
        self._revalidate()

    def _add_row(self, r: int, entry: comp.CompilationEntry, item: bi.ImportSelection):
        name_item = QTableWidgetItem(entry.name)
        name_item.setFlags(_PLAIN)
        self.table.setItem(r, _NAME, name_item)
        row = {"entry": entry, "name": name_item}
        cells = {}
        for key, col, part_of, text in (
            ("media", _MEDIA, lambda c: c.dataset, self._media_text),
            ("pose", _POSE, lambda c: c.pose, lambda c: f"{len(c.pose_keys)} frames"),
            ("behavior", _BEHAVIOR, lambda c: c.behavior, lambda c: f"{len(c.behavior_keys)} videos"),
        ):
            cell = QTableWidgetItem("")
            if entry.ok and part_of(entry.contents).ok:
                cell.setText(text(entry.contents))
                cell.setToolTip(part_of(entry.contents).summary)
                cell.setFlags(_CHECKABLE)
                wanted = {"media": item.dataset, "pose": item.pose, "behavior": item.behavior}[key]
                cell.setCheckState(Qt.CheckState.Checked if wanted else Qt.CheckState.Unchecked)
            else:
                cell.setText("—" if entry.ok else "")
                cell.setFlags(_OFF)
            self.table.setItem(r, col, cell)
            cells[key] = cell
        as_item = QTableWidgetItem(item.dataset_name or entry.name)
        as_item.setFlags(_EDITABLE)
        self.table.setItem(r, _AS, as_item)
        notes = QTableWidgetItem("")
        notes.setFlags(_PLAIN)
        self.table.setItem(r, _NOTES, notes)
        row.update(cells=cells, as_item=as_item, notes=notes)
        self._rows.append(row)

    def _media_text(self, c: bi.BundleContents) -> str:
        n = len(c.dataset_obj.files) if c.dataset_obj else 0
        return f"{n} {c.media_kind}"

    # ------------------------------------------------------------ reading state

    def _ticked(self, cell: QTableWidgetItem) -> bool:
        return bool(cell.flags() & Qt.ItemFlag.ItemIsUserCheckable) and cell.checkState() == Qt.CheckState.Checked

    def _build_selection(self) -> comp.CompilationSelection:
        sel = comp.CompilationSelection()
        for row in self._rows:
            entry = row["entry"]
            item = bi.ImportSelection(
                dataset=self._ticked(row["cells"]["media"]),
                pose=self._ticked(row["cells"]["pose"]),
                behavior=self._ticked(row["cells"]["behavior"]),
                dataset_name=row["as_item"].text().strip(),
            )
            if entry.ok:
                item.media_dir = self._media.get(entry.name) or entry.contents.media_dir
            if not item.dataset and (item.pose or item.behavior):
                item.target_dataset = entry.name if self._ws.datasets.get(entry.name) is not None else ""
            sel.items[entry.name] = item
        sel.import_projects = self.projects_check.isVisibleTo(self) and self.projects_check.isChecked()
        sel.existing_labels_policy = next((p for p, r in self._policy_radios.items() if r.isChecked()), LabelPolicy.MERGE_EXISTING)
        return sel

    # ------------------------------------------------------------ reacting

    def _on_changed(self, _item):
        if not self._loading:
            self._revalidate()

    def _tick_all(self, on: bool):
        self._loading = True
        for row in self._rows:
            entry = row["entry"]
            default = self.selection.items.get(entry.name)
            for key, cell in row["cells"].items():
                if cell.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                    cell.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
            if on and default is not None and entry.ok and self._ws.datasets.get(entry.name) is not None:
                # "everything" must not mean overwriting a dataset you already have
                row["cells"]["media"].setCheckState(Qt.CheckState.Unchecked)
        self._loading = False
        self._revalidate()

    def _on_double_click(self, r: int, col: int):
        if col != _NOTES or not (0 <= r < len(self._rows)):
            return
        entry = self._rows[r]["entry"]
        if not entry.ok or entry.contents.media_dir is not None or not entry.contents.dataset.ok:
            return
        path = QFileDialog.getExistingDirectory(self, f"Choose the folder containing the {entry.contents.media_kind} of '{entry.name}'",
                                                str(self._media.get(entry.name, Path.home())))
        if not path:
            return
        check, _missing = mi.check_media_folder(Path(path), entry.contents.dataset_obj)
        if check.ok:
            self._media[entry.name] = Path(path)
            self._media_error.pop(entry.name, None)
            self._rows[r]["cells"]["media"].setCheckState(Qt.CheckState.Checked)
        else:
            self._media.pop(entry.name, None)
            self._media_error[entry.name] = check.message      # shown by _notes_for, which recomputes the cell
        self._revalidate()

    def _notes_for(self, row: dict, item: bi.ImportSelection) -> List[Tuple[str, str]]:
        entry = row["entry"]
        if not entry.ok:
            return [("bad", entry.problem)]
        out: List[Tuple[str, str]] = []
        c = entry.contents
        if item.dataset:
            name = item.dataset_name.strip()
            if not name:
                out.append(("bad", "Give it a name."))
            elif self._ws.datasets.get(name) is not None:
                out.append(("bad", f"You already have a dataset called '{name}' — choose another name."))
            elif c.media_dir is None and item.media_dir is None:
                if entry.name in self._media_error:
                    out.append(("bad", f"{self._media_error[entry.name]} Double-click to choose another folder."))
                else:
                    out.append(("warn", f"The {c.media_kind} aren't in the compilation — double-click here to locate them."))
            elif c.media_dir is None:
                out.append(("ok", f"{c.media_kind.capitalize()} found in {item.media_dir}"))
        elif item.pose or item.behavior:
            if not item.target_dataset:
                out.append(("bad", f"You don't have a dataset called '{entry.name}' — tick its {c.media_kind} too, so the labels have somewhere to go."))
            else:
                key = (entry.name, item.pose, item.behavior, item.target_dataset)
                if key not in self._notes_cache:
                    self._notes_cache[key] = comp.attach_notes(self._ws, entry, item)
                out.extend(self._notes_cache[key])
        return out

    def _revalidate(self, *_):
        if self._loading or self.contents is None:
            return
        sel = self._build_selection()
        self.selection = sel
        # Updating the cells below fires itemChanged, which would call this again forever.
        self._loading = True
        try:
            needs_policy = False
            for row in self._rows:
                item = sel.items[row["entry"].name]
                # the "import as" name only matters while the dataset itself is being imported
                row["as_item"].setFlags(_EDITABLE if item.dataset else _OFF)
                notes = self._notes_for(row, item)
                row["notes"].setText("  •  ".join(t for _l, t in notes))
                worst = "bad" if any(l == "bad" for l, _t in notes) else "warn" if any(l == "warn" for l, _t in notes) else "ok" if notes else "muted"
                row["notes"].setForeground(QBrush(QColor(_COLORS[worst])))
                if not item.dataset and (item.pose or item.behavior) and item.target_dataset:
                    needs_policy = needs_policy or any("already has labels" in t for _l, t in notes)
            self.policy_box.setVisible(needs_policy)
            self.table.resizeRowsToContents()
        finally:
            self._loading = False
        problems = comp.validate_compilation_selection(self._ws, self.contents, sel)
        self.problem_label.setText("\n".join(problems[:4]) + (f"\n…and {len(problems) - 4} more" if len(problems) > 4 else ""))
        self.import_btn.setEnabled(not problems)
