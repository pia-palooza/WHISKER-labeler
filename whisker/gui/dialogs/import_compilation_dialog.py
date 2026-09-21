"""Import from a compilation: tick which datasets, and which parts of each.

One row per dataset. For each dataset you choose whether to *add it as new* (its videos/frames
are copied, under the name in "Add as new") or *use one you already have* (nothing is copied and
its labels are added to that dataset). Each project the package uses gets the same choice. Datasets
you already have start on "use existing", so a re-import brings in just the labels for the dataset
of the same name; where labels would land on existing labels, one choice below the table decides how
to combine them. See :mod:`whisker.core.compilation`.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
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

_NAME, _MEDIA, _POSE, _BEHAVIOR, _AS, _USE, _NOTES = range(7)
_CHECKABLE = Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled
_PLAIN = Qt.ItemFlag.ItemIsEnabled
_EDITABLE = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsEditable
_OFF = Qt.ItemFlag.NoItemFlags
_COLORS = {"ok": "#2e7d32", "warn": "#e67e22", "bad": "#c0392b", "muted": "gray"}
_ADD_NEW = "(add as new)"


class ImportCompilationDialog(QDialog):
    def __init__(self, workspace: Workspace, root: Path, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._ws = workspace
        self._root = Path(root)
        self._loading = True
        self._rows: List[dict] = []
        self._project_rows: Dict[str, dict] = {}
        self._media: Dict[str, Path] = {}                 # media folders the user located, by dataset
        self._media_error: Dict[str, str] = {}            # why the last folder they tried was rejected
        self._notes_cache: Dict[tuple, List[Tuple[str, str]]] = {}
        self.contents: Optional[comp.CompilationContents] = None
        self.selection: Optional[comp.CompilationSelection] = None

        self.setWindowTitle("Import from a Compilation")
        screen = QApplication.primaryScreen()
        dpi = screen.logicalDotsPerInch() / 96.0 if screen else 1.0
        self.setMinimumWidth(int(1200 * dpi))

        layout = QVBoxLayout(self)
        self.header = QLabel("")
        self.header.setWordWrap(True)
        layout.addWidget(self.header)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["Dataset", "Videos / frames", "Pose labels", "Behavior labels", "Add as new", "Or use existing", "Notes"])
        head = self.table.horizontalHeader()
        for col in (_NAME, _MEDIA, _POSE, _BEHAVIOR):
            head.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        for col, width in ((_AS, 150), (_USE, 200)):
            head.setSectionResizeMode(col, QHeaderView.ResizeMode.Interactive)
            self.table.setColumnWidth(col, int(width * dpi))
        head.setSectionResizeMode(_NOTES, QHeaderView.ResizeMode.Stretch)     # gets whatever is left
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.table.setWordWrap(True)
        self.table.setMinimumHeight(int(200 * dpi))
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

        self.projects_check = QCheckBox("Also handle the project(s) these datasets use:")
        self.projects_check.toggled.connect(self._revalidate)
        layout.addWidget(self.projects_check)
        self.projects_box = QGroupBox()
        self._projects_layout = QVBoxLayout(self.projects_box)
        layout.addWidget(self.projects_box)

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
            for w in (self.projects_check, self.projects_box, self.policy_box):
                w.setVisible(False)
            self._loading = False
            return
        finally:
            QApplication.restoreOverrideCursor()

        c, sel = self.contents, self.selection
        self.header.setText(
            f"<b>{c.name}</b> — {len(c.entries)} dataset(s)" + (f", exported {c.created_at[:10]}" if c.created_at else "")
            + "<br><span style='color:gray'>For each dataset, add it as new or use one you already have. "
              "Double-click a note that asks for a folder to locate the files.</span>"
        )
        if c.warnings:
            self.header.setText(self.header.text() + "<br>" + "<br>".join(c.warnings))

        self.table.setRowCount(len(c.entries))
        for r, entry in enumerate(c.entries):
            self._add_row(r, entry, sel.items[entry.name])

        projects = c.project_names
        self.projects_check.setVisible(bool(projects))
        self.projects_box.setVisible(bool(projects))
        if projects:
            self.projects_check.setChecked(sel.import_projects)
            self._build_project_rows(projects, sel)
        self._loading = False
        self._revalidate()

    def _build_project_rows(self, names: List[str], sel: comp.CompilationSelection):
        existing = sorted(self._ws.projects.keys())
        for name in names:
            mode_kind, mode_value = sel.choice_for(name)
            box = QWidget()
            h = QHBoxLayout(box)
            h.setContentsMargins(0, 0, 0, 0)
            label = QLabel(f"<b>{name}</b>")
            label.setMinimumWidth(120)
            mode = QComboBox()
            mode.addItem("Add as new project named:", "new")
            for p in existing:
                mode.addItem(f"Use my existing project: {p}", f"existing:{p}")
            edit = QLineEdit(mode_value if mode_kind == "new" else name)
            if mode_kind == "existing":
                mode.setCurrentIndex(max(0, mode.findData(f"existing:{mode_value}")))
            note = QLabel("")
            note.setWordWrap(True)
            h.addWidget(label)
            h.addWidget(mode, 1)
            h.addWidget(edit, 1)
            wrapper = QWidget()
            v = QVBoxLayout(wrapper)
            v.setContentsMargins(0, 0, 0, 0)
            v.addWidget(box)
            v.addWidget(note)
            self._projects_layout.addWidget(wrapper)
            mode.currentIndexChanged.connect(self._revalidate)
            edit.textChanged.connect(self._revalidate)
            self._project_rows[name] = {"mode": mode, "name": edit, "note": note}

    def _add_row(self, r: int, entry: comp.CompilationEntry, item: bi.ImportSelection):
        name_item = QTableWidgetItem(entry.name)
        name_item.setFlags(_PLAIN)
        self.table.setItem(r, _NAME, name_item)
        row = {"entry": entry, "name": name_item}
        existing_mode = (not item.dataset) and bool(item.target_dataset)
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
                wanted = {"media": item.dataset, "pose": item.pose, "behavior": item.behavior}[key]
                if key == "media" and existing_mode:
                    cell.setFlags(_OFF)            # nothing is copied when you use a dataset you have
                    cell.setCheckState(Qt.CheckState.Unchecked)
                else:
                    cell.setFlags(_CHECKABLE)
                    cell.setCheckState(Qt.CheckState.Checked if wanted else Qt.CheckState.Unchecked)
            else:
                cell.setText("—" if entry.ok else "")
                cell.setFlags(_OFF)
            self.table.setItem(r, col, cell)
            cells[key] = cell
        as_item = QTableWidgetItem(item.dataset_name or entry.name)
        as_item.setFlags(_EDITABLE)
        self.table.setItem(r, _AS, as_item)

        use = QComboBox()
        use.addItem(_ADD_NEW, "")
        if entry.ok:
            for name, hits, total in bi.rank_target_datasets(self._ws, entry.contents):
                use.addItem(f"{name}    — {hits}/{total}" if total else name, name)
        else:
            for name in sorted(self._ws.datasets.keys(), key=str.lower):
                use.addItem(name, name)
        if existing_mode:
            use.setCurrentIndex(max(0, use.findData(item.target_dataset)))
        use.setEnabled(entry.ok)
        self.table.setCellWidget(r, _USE, use)

        notes = QTableWidgetItem("")
        notes.setFlags(_PLAIN)
        self.table.setItem(r, _NOTES, notes)
        row.update(cells=cells, as_item=as_item, use=use, notes=notes)
        self._rows.append(row)
        use.currentIndexChanged.connect(lambda _i, _row=row: self._on_use_changed(_row))

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
            use = row["use"].currentData() or ""
            item = bi.ImportSelection(
                dataset=self._ticked(row["cells"]["media"]) and not use,
                pose=self._ticked(row["cells"]["pose"]),
                behavior=self._ticked(row["cells"]["behavior"]),
                dataset_name=row["as_item"].text().strip(),
                target_dataset=use,
            )
            if entry.ok:
                item.media_dir = self._media.get(entry.name) or entry.contents.media_dir
            sel.items[entry.name] = item
        for name, pr in self._project_rows.items():
            data = pr["mode"].currentData() or "new"
            sel.project_choices[name] = (
                ("existing", data.split(":", 1)[1]) if data.startswith("existing:") else ("new", pr["name"].text().strip()))
        sel.import_projects = self.projects_check.isVisibleTo(self) and self.projects_check.isChecked()
        sel.existing_labels_policy = next((p for p, r in self._policy_radios.items() if r.isChecked()), LabelPolicy.MERGE_EXISTING)
        return sel

    # ------------------------------------------------------------ reacting

    def _on_changed(self, _item):
        if not self._loading:
            self._revalidate()

    def _apply_use_state(self, row: dict):
        """Using a dataset you have means nothing is copied: the media cell goes inert; back on
        'add as new' it's ticked again (if the export has usable media)."""
        cell, entry = row["cells"]["media"], row["entry"]
        if row["use"].currentData():
            cell.setFlags(_OFF)
            cell.setCheckState(Qt.CheckState.Unchecked)
        elif entry.ok and entry.contents.dataset.ok:
            cell.setFlags(_CHECKABLE)
            cell.setCheckState(Qt.CheckState.Checked)

    def _on_use_changed(self, row: dict):
        if self._loading:
            return
        self._loading = True
        try:
            self._apply_use_state(row)
        finally:
            self._loading = False
        self._revalidate()

    def _tick_all(self, on: bool):
        self._loading = True
        try:
            for row in self._rows:
                entry = row["entry"]
                if on and entry.ok:
                    # "everything" never means replacing a dataset you already have: those go on 'use existing'
                    same = row["use"].findData(entry.name) if self._ws.datasets.get(entry.name) is not None else -1
                    row["use"].setCurrentIndex(same if same >= 0 else 0)
                    self._apply_use_state(row)
                for key in ("pose", "behavior"):
                    cell = row["cells"][key]
                    if cell.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                        cell.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
                media = row["cells"]["media"]
                if not on and media.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                    media.setCheckState(Qt.CheckState.Unchecked)
        finally:
            self._loading = False
        self._revalidate()

    def _on_double_click(self, r: int, col: int):
        if col != _NOTES or not (0 <= r < len(self._rows)):
            return
        row = self._rows[r]
        entry = row["entry"]
        if (row["use"].currentData() or not entry.ok or entry.contents.media_dir is not None
                or not entry.contents.dataset.ok):
            return
        path = QFileDialog.getExistingDirectory(self, f"Choose the folder containing the {entry.contents.media_kind} of '{entry.name}'",
                                                str(self._media.get(entry.name, Path.home())))
        if not path:
            return
        check, _missing = mi.check_media_folder(Path(path), entry.contents.dataset_obj)
        if check.ok:
            self._media[entry.name] = Path(path)
            self._media_error.pop(entry.name, None)
            row["cells"]["media"].setCheckState(Qt.CheckState.Checked)
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
                out.append(("bad", f"You already have a dataset called '{name}' — choose another name, or use it under 'Or use existing'."))
            elif c.media_dir is None and item.media_dir is None:
                if entry.name in self._media_error:
                    out.append(("bad", f"{self._media_error[entry.name]} Double-click to choose another folder."))
                else:
                    out.append(("warn", f"The {c.media_kind} aren't in the compilation — double-click here to locate them."))
            elif c.media_dir is None:
                out.append(("ok", f"{c.media_kind.capitalize()} found in {item.media_dir}"))
        elif item.pose or item.behavior:
            if not item.target_dataset:
                out.append(("bad", f"These labels need a dataset: tick its {c.media_kind} to add it as new, "
                                   "or choose one of yours under 'Or use existing'."))
            else:
                key = (entry.name, item.pose, item.behavior, item.target_dataset)
                if key not in self._notes_cache:
                    self._notes_cache[key] = comp.attach_notes(self._ws, entry, item)
                out.extend(self._notes_cache[key])
        return out

    def _project_note(self, name: str, sel: comp.CompilationSelection) -> Tuple[str, str]:
        mode, target = sel.choice_for(name)
        if mode != "existing":
            return "", "muted"
        gaps = set()
        for e in self.contents.entries:
            item = sel.items.get(e.name)
            if item and e.ok and e.contents.project_obj and e.contents.project_obj.name == name and (item.pose or item.behavior):
                gaps.update(bi.project_gaps(self._ws, e.contents, target, pose=item.pose, behavior=item.behavior))
        if gaps:
            return f"'{target}' doesn't define {'; '.join(sorted(gaps))}, which these labels use — they'll import, but won't show up when labeling with it.", "warn"
        return "Nothing is added; your project is used as it is.", "muted"

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
                # the "add as new" name only matters while the dataset itself is being added
                row["as_item"].setFlags(_EDITABLE if item.dataset else _OFF)
                notes = self._notes_for(row, item)
                row["notes"].setText("  •  ".join(t for _l, t in notes))
                worst = "bad" if any(l == "bad" for l, _t in notes) else "warn" if any(l == "warn" for l, _t in notes) else "ok" if notes else "muted"
                row["notes"].setForeground(QBrush(QColor(_COLORS[worst])))
                if not item.dataset and (item.pose or item.behavior) and item.target_dataset:
                    needs_policy = needs_policy or any("already has labels" in t for _l, t in notes)
            self.policy_box.setVisible(needs_policy)
            self.projects_box.setEnabled(self.projects_check.isChecked())
            for name, pr in self._project_rows.items():
                pr["name"].setEnabled((pr["mode"].currentData() or "new") == "new")
                text, level = self._project_note(name, sel)
                pr["note"].setText(text)
                pr["note"].setStyleSheet(f"color: {_COLORS[level]};")
                pr["note"].setVisible(bool(text))
            self.table.resizeRowsToContents()
        finally:
            self._loading = False
        problems = comp.validate_compilation_selection(self._ws, self.contents, sel)
        self.problem_label.setText("\n".join(problems[:4]) + (f"\n…and {len(problems) - 4} more" if len(problems) > 4 else ""))
        self.import_btn.setEnabled(not problems)
