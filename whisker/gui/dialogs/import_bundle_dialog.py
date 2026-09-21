"""Import an exported dataset with one pick.

The user chooses (or drops) the export folder — or anything inside it. The app finds
the export, shows what's in it as a checklist, and imports exactly what's ticked. The
older "pick every piece yourself" form is still available for data that didn't come
from Export (button at the bottom).
"""

from pathlib import Path
from typing import Dict, Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from whisker.core import bundle_import as bi
from whisker.core import compilation as comp
from whisker.core.workspace import Workspace
from whisker.gui.dialogs.attach_labels_dialog import AttachLabelsDialog

_OK = "color: #2e7d32;"
_BAD = "color: #c0392b;"
_WARN = "color: #e67e22;"
_MUTED = "color: gray;"

ADVANCED_RESULT = 2       # dialog.exec() value when the user chose "Import from separate files"
COMPILATION_RESULT = 3    # ...or picked a compilation (several datasets); see ``compilation_root``


class ImportBundleDialog(QDialog):
    """One pick -> checklist -> import."""

    def __init__(self, workspace: Workspace, parent: Optional[QWidget] = None,
                 active_project_name: Optional[str] = None):
        super().__init__(parent)
        self._workspace = workspace
        self._active_project = active_project_name
        self.contents: Optional[bi.BundleContents] = None
        self.selection: Optional[bi.ImportSelection] = None
        self.compilation_root: Optional[Path] = None
        self._loading = False

        self.setWindowTitle("Import")
        self.setAcceptDrops(True)
        screen = QApplication.primaryScreen()
        dpi = screen.logicalDotsPerInch() / 96.0 if screen else 1.0
        self.setMinimumWidth(int(700 * dpi))

        root = QVBoxLayout(self)

        hint = QLabel(
            "Choose the export folder you were given — the one File → Export created. "
            "You can also drag it onto this window, or pick a folder that contains it."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(_MUTED)
        root.addWidget(hint)

        row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Folder of an export...")
        self.path_edit.textChanged.connect(self._schedule_locate)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        row.addWidget(self.path_edit)
        row.addWidget(browse)
        root.addLayout(row)

        self.location_label = QLabel("")
        self.location_label.setWordWrap(True)
        root.addWidget(self.location_label)

        self.candidate_combo = QComboBox()
        self.candidate_combo.setVisible(False)
        self.candidate_combo.activated.connect(self._on_candidate_chosen)
        root.addWidget(self.candidate_combo)

        # --- what's in the export ---
        self.parts_box = QGroupBox("What's in this export — tick what you want to import")
        grid = QGridLayout(self.parts_box)
        grid.setColumnStretch(1, 1)
        self._rows: Dict[str, tuple] = {}
        for r, (key, title) in enumerate(
            (("project", "Project"), ("dataset", "Frames"), ("pose", "Pose labels"), ("behavior", "Behavior labels"))
        ):
            check = QCheckBox(title)
            check.toggled.connect(self._on_part_toggled)
            note = QLabel("")
            note.setWordWrap(True)
            grid.addWidget(check, r, 0, Qt.AlignmentFlag.AlignTop)
            grid.addWidget(note, r, 1)
            self._rows[key] = (check, note)
        root.addWidget(self.parts_box)
        self.parts_box.setVisible(False)

        # --- follow-up details (only what applies) ---
        self.details_box = QGroupBox("Details")
        d = QVBoxLayout(self.details_box)

        # Project: add the export's as a new one, or use one you already have.
        self.project_group = QWidget()
        pg = QVBoxLayout(self.project_group)
        pg.setContentsMargins(0, 0, 0, 0)
        pg.addWidget(QLabel("<b>Project</b>"))
        self.project_new_radio = QRadioButton("Add as a new project named:")
        self.project_name_edit = QLineEdit()
        pg.addWidget(self._choice_row(self.project_new_radio, self.project_name_edit))
        self.project_existing_radio = QRadioButton("Use my existing project:")
        self.project_combo = QComboBox()
        pg.addWidget(self._choice_row(self.project_existing_radio, self.project_combo))
        self.project_replace = QCheckBox()
        pg.addWidget(self.project_replace)
        self.project_note = QLabel("")
        self.project_note.setWordWrap(True)
        pg.addWidget(self.project_note)
        self._project_mode = QButtonGroup(self)
        self._project_mode.addButton(self.project_new_radio)
        self._project_mode.addButton(self.project_existing_radio)
        d.addWidget(self.project_group)

        # Dataset: add the export's as a new one, or attach to one you already have.
        self.dataset_group = QWidget()
        dg = QVBoxLayout(self.dataset_group)
        dg.setContentsMargins(0, 0, 0, 0)
        self.dataset_title = QLabel("<b>Dataset</b>")
        dg.addWidget(self.dataset_title)
        self.dataset_new_radio = QRadioButton("Add as a new dataset named:")
        self.name_edit = QLineEdit()
        self.name_row = self._choice_row(self.dataset_new_radio, self.name_edit)
        dg.addWidget(self.name_row)
        self.dataset_replace = QCheckBox()
        dg.addWidget(self.dataset_replace)
        self.media_info = QLabel("")
        self.media_info.setWordWrap(True)
        dg.addWidget(self.media_info)
        media_row = QWidget()
        mr = QHBoxLayout(media_row)
        mr.setContentsMargins(0, 0, 0, 0)
        self.media_edit = QLineEdit()
        media_browse = QPushButton("Browse...")
        media_browse.clicked.connect(self._browse_media)
        mr.addWidget(self.media_edit, 1)
        mr.addWidget(media_browse)
        self.media_row = media_row
        dg.addWidget(media_row)
        self.dataset_existing_radio = QRadioButton("Use my existing dataset:")
        self.dataset_combo = QComboBox()
        dg.addWidget(self._choice_row(self.dataset_existing_radio, self.dataset_combo))
        self.dataset_note = QLabel("")
        self.dataset_note.setWordWrap(True)
        dg.addWidget(self.dataset_note)
        self._dataset_mode = QButtonGroup(self)
        self._dataset_mode.addButton(self.dataset_new_radio)
        self._dataset_mode.addButton(self.dataset_existing_radio)
        d.addWidget(self.dataset_group)

        self.labels_note = QLabel("")
        self.labels_note.setWordWrap(True)
        self.labels_note.setStyleSheet(_MUTED)
        d.addWidget(self.labels_note)
        root.addWidget(self.details_box)
        self.details_box.setVisible(False)

        for radio in (self.project_new_radio, self.project_existing_radio, self.dataset_new_radio, self.dataset_existing_radio):
            radio.toggled.connect(self._revalidate)
        for edit in (self.project_name_edit, self.name_edit, self.media_edit):
            edit.textChanged.connect(self._revalidate)
        for combo in (self.project_combo, self.dataset_combo):
            combo.currentIndexChanged.connect(self._revalidate)
        for box in (self.project_replace, self.dataset_replace):
            box.toggled.connect(self._revalidate)

        self.problem_label = QLabel("")
        self.problem_label.setWordWrap(True)
        self.problem_label.setStyleSheet(_WARN)
        root.addWidget(self.problem_label)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        root.addWidget(line)

        self.button_box = QDialogButtonBox()
        self.manual_btn = self.button_box.addButton("Import from separate files...", QDialogButtonBox.ButtonRole.ActionRole)
        self.manual_btn.setToolTip("For files that didn't come from Export: use your existing project and dataset, "
                                   "and browse only for the label files or media that are new.")
        self.manual_btn.clicked.connect(lambda: self.done(ADVANCED_RESULT))
        self.import_btn = self.button_box.addButton("Import", QDialogButtonBox.ButtonRole.AcceptRole)
        self.button_box.addButton(QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self._on_import_clicked)
        self.button_box.rejected.connect(self.reject)
        root.addWidget(self.button_box)

        self._locate_timer = QTimer(self)
        self._locate_timer.setSingleShot(True)
        self._locate_timer.setInterval(400)
        self._locate_timer.timeout.connect(self._locate)

        self._set_status("", _MUTED)
        self.import_btn.setEnabled(False)

    @staticmethod
    def _choice_row(radio: QRadioButton, field: QWidget) -> QWidget:
        """``( ) label  [ field ]`` on one line."""
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(radio)
        h.addWidget(field, 1)
        return row

    # ------------------------------------------------------------ picking

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            if url.isLocalFile():
                self.set_path(url.toLocalFile())
                event.acceptProposedAction()
                return

    def set_path(self, path: str):
        self.path_edit.setText(path)
        self._locate_timer.stop()
        self._locate()

    def _browse(self):
        start = self.path_edit.text().strip() or str(Path.home())
        path = QFileDialog.getExistingDirectory(self, "Choose the export folder", start)
        if path:
            self.set_path(path)

    def _browse_media(self):
        start = self.media_edit.text().strip() or str(Path.home())
        path = QFileDialog.getExistingDirectory(self, f"Choose the folder containing the {self._kind()}", start)
        if path:
            self.media_edit.setText(path)

    def _schedule_locate(self):
        self._locate_timer.start()

    def _set_status(self, text: str, style: str):
        self.location_label.setText(text)
        self.location_label.setStyleSheet(style)

    def _reset(self):
        self.contents = self.selection = self.compilation_root = None
        self.import_btn.setText("Import")
        self.parts_box.setVisible(False)
        self.details_box.setVisible(False)
        self.candidate_combo.setVisible(False)
        self.problem_label.setText("")
        self.import_btn.setEnabled(False)

    def _locate(self):
        text = self.path_edit.text().strip()
        if not text:
            self._reset()
            self._set_status("", _MUTED)
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            loc = bi.locate_bundle(text)
        finally:
            QApplication.restoreOverrideCursor()
        if loc.is_compilation:
            self.candidate_combo.setVisible(False)
            self._show_compilation(loc.root, loc.message)
        elif loc.found:
            self.candidate_combo.setVisible(False)
            self._load(loc.root, loc.message)
        elif loc.candidates:
            self._reset()
            self._set_status(loc.message, _WARN)
            self.candidate_combo.clear()
            base = Path(text)
            for c in loc.candidates:
                try:
                    label = str(c.relative_to(base if base.is_dir() else base.parent))
                except ValueError:
                    label = str(c)
                self.candidate_combo.addItem(label, str(c))
            self.candidate_combo.setCurrentIndex(-1)
            self.candidate_combo.setPlaceholderText("Choose an export...")
            self.candidate_combo.setVisible(True)
        else:
            self._reset()
            self._set_status(loc.message, _BAD)

    def _on_candidate_chosen(self, index: int):
        root = self.candidate_combo.itemData(index)
        if not root:
            return
        if bi.kind_of(Path(root)) == "compilation":
            self._show_compilation(Path(root), "")
        else:
            self._load(Path(root), "")

    def _show_compilation(self, root: Path, note: str):
        """A compilation holds several datasets; the next step is choosing among them."""
        self._reset()
        self.compilation_root = root
        name, count = comp.peek_compilation(root)
        summary = note or f"Found the compilation '{name or root.name}'."
        self._set_status(f"{summary} It holds {count} dataset(s) \u2014 click Choose datasets to pick what to import.", _OK)
        self.import_btn.setText("Choose datasets...")
        self.import_btn.setEnabled(True)

    # ------------------------------------------------------------ loading

    def _load(self, root: Path, note: str):
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            contents = bi.inspect_bundle(root)
            selection = bi.default_selection(self._workspace, contents)
        except bi.BundleImportError as e:
            self._reset()
            self._set_status(str(e), _BAD)
            return
        finally:
            QApplication.restoreOverrideCursor()

        self._loading = True
        self.contents, self.selection = contents, selection
        summary = note or f"Found the export '{root.name}'."
        for w in contents.warnings:
            summary += "\n" + w
        self._set_status(summary, _OK)

        self.parts_box.setVisible(True)
        kind = self._kind().capitalize()
        self._rows["dataset"][0].setText(kind)
        parts = {"project": contents.project, "dataset": contents.dataset, "pose": contents.pose, "behavior": contents.behavior}
        # The dataset row means "the export's dataset is part of this import"; whether it's added
        # as new or matched to one you have is the choice underneath.
        ticked = {"project": selection.project, "dataset": contents.dataset.ok, "pose": selection.pose, "behavior": selection.behavior}
        for key, part in parts.items():
            check, note_label = self._rows[key]
            check.setEnabled(part.ok)
            check.setChecked(part.ok and ticked[key])
            if part.ok:
                text, style = part.summary, _MUTED
            elif part.present:
                text, style = part.problem, _BAD
            else:
                text, style = part.summary or "Not in this export", _MUTED
            if key == "dataset" and part.ok and self._workspace.datasets.get(contents.dataset_name):
                text += f" — you already have a dataset called '{contents.dataset_name}'"
            if key == "project" and part.ok:
                relation = bi.project_relation(self._workspace, contents)
                if relation == "identical":
                    text += " — you already have this exact project"
                elif relation == "different":
                    text += " — you have a project with this name that differs"
            note_label.setText(text)
            note_label.setStyleSheet(style)

        self._load_choices(contents, selection)
        self.media_edit.setText(str(selection.media_dir) if selection.media_dir and not contents.media_dir else "")
        self._loading = False
        self._refresh_details()
        self._revalidate()

    def _load_choices(self, contents: bi.BundleContents, selection: bi.ImportSelection):
        """Fill the project and dataset pickers and set the starting choices."""
        ws = self._workspace
        projects = sorted(ws.projects.keys())
        self.project_combo.clear()
        for name in projects:
            self.project_combo.addItem(name, name)
        self.project_existing_radio.setEnabled(bool(projects))
        self.project_name_edit.setText(selection.project_name or (contents.project_obj.name if contents.project_obj else ""))
        use_existing = selection.project_mode == "existing" and bool(projects)
        (self.project_existing_radio if use_existing else self.project_new_radio).setChecked(True)
        if selection.existing_project:
            self.project_combo.setCurrentIndex(max(0, self.project_combo.findData(selection.existing_project)))
        elif self._active_project in projects:
            # switching to "use mine" should land on the project they're working in
            self.project_combo.setCurrentIndex(self.project_combo.findData(self._active_project))

        ranked = bi.rank_target_datasets(ws, contents)
        self.dataset_combo.clear()
        for name, hits, total in ranked:
            self.dataset_combo.addItem(f"{name}    \u2014 {hits} of {total} labels match" if total else name, name)
        self.dataset_existing_radio.setEnabled(bool(ranked))
        use_existing = (not selection.dataset) and bool(selection.target_dataset) and contents.dataset.ok
        (self.dataset_existing_radio if use_existing else self.dataset_new_radio).setChecked(True)
        if selection.target_dataset:
            self.dataset_combo.setCurrentIndex(max(0, self.dataset_combo.findData(selection.target_dataset)))
        self.name_edit.setText(selection.dataset_name)

    def _kind(self) -> str:
        return self.contents.media_kind if self.contents else "frames"

    # ------------------------------------------------------------ details

    def _on_part_toggled(self):
        if self._loading:
            return
        self._refresh_details()
        self._revalidate()

    def _refresh_details(self):
        """Show only the follow-up questions that apply to the current ticks and choices."""
        c = self.contents
        if c is None:
            return
        ws = self._workspace
        want = {k: self._rows[k][0].isChecked() for k in self._rows}
        labels = want["pose"] or want["behavior"]

        # --- project
        show_project = want["project"] and c.project_obj is not None
        self.project_group.setVisible(show_project)
        if show_project:
            new_mode = self.project_new_radio.isChecked()
            self.project_name_edit.setEnabled(new_mode)
            self.project_combo.setEnabled(not new_mode)
            name = self.project_name_edit.text().strip()
            taken = new_mode and ws.projects.get(name) is not None
            self.project_replace.setVisible(taken)
            if taken:
                self.project_replace.setText(f"Replace my project '{name}' with the one in this export")
            if new_mode:
                relation = bi.project_relation(ws, c)
                self.project_note.setText(
                    "You have a project with this name that differs from the export's." if relation == "different" else "")
                self.project_note.setStyleSheet(_WARN)
            else:
                chosen = self.project_combo.currentData() or ""
                gaps = bi.project_gaps(ws, c, chosen, pose=want["pose"], behavior=want["behavior"]) if chosen else []
                if gaps:
                    self.project_note.setText(
                        f"'{chosen}' doesn't define {'; '.join(gaps)}, which these labels use \u2014 they'll import, "
                        "but those won't show up when labeling with it.")
                    self.project_note.setStyleSheet(_WARN)
                else:
                    self.project_note.setText("Nothing is added; your project is used as it is.")
                    self.project_note.setStyleSheet(_MUTED)
            self.project_note.setVisible(bool(self.project_note.text()))

        # --- dataset
        show_dataset = want["dataset"]
        self.dataset_group.setVisible(show_dataset)
        existing_mode = show_dataset and self.dataset_existing_radio.isChecked()
        new_mode = show_dataset and not existing_mode
        if show_dataset:
            self.name_edit.setEnabled(new_mode)
            self.dataset_combo.setEnabled(existing_mode)
            name = self.name_edit.text().strip()
            taken = new_mode and ws.datasets.get(name) is not None
            self.dataset_replace.setVisible(taken)
            if taken:
                self.dataset_replace.setText(f"Replace the dataset I already have called '{name}' (its media and labels)")
            self.media_info.setVisible(new_mode)
            self.media_row.setVisible(new_mode and c.media_dir is None)
            if new_mode:
                if c.media_dir is None:
                    self.media_info.setText(
                        f"The {c.media_kind} aren't inside this export. Choose the folder that contains them:")
                    self.media_info.setStyleSheet(_WARN)
                else:
                    where = "inside the export" if c.media_included else "at their original location"
                    self.media_info.setText(f"The {c.media_kind} will be copied from {where}.")
                    self.media_info.setStyleSheet(_MUTED)
            self.dataset_note.setVisible(existing_mode)
            if existing_mode:
                target = self.dataset_combo.currentData() or ""
                if labels:
                    self.dataset_note.setText(f"No {c.media_kind} are copied. The labels are added to '{target}'.")
                    self.dataset_note.setStyleSheet(_MUTED)
                else:
                    self.dataset_note.setText("Using an existing dataset only adds labels to it \u2014 tick pose or "
                                              "behavior labels, or choose to add the dataset as new.")
                    self.dataset_note.setStyleSheet(_WARN)

        # --- labels
        goes_with_new_dataset = show_dataset and new_mode
        if labels and goes_with_new_dataset:
            self.labels_note.setText("The labels will be attached to the dataset you're importing.")
        elif labels and existing_mode:
            self.labels_note.setText(
                f"Next you'll see how the labels compare with '{self.dataset_combo.currentData() or ''}' "
                "and choose how to combine them with any labels it already has.")
        elif labels:
            self.labels_note.setText(
                "Next you'll choose which of your datasets these labels belong to, and how to "
                "combine them with any labels it already has.")
        else:
            self.labels_note.setText("")
        self.labels_note.setVisible(labels)

        self.details_box.setVisible(self.project_group.isVisibleTo(self.details_box)
                                    or self.dataset_group.isVisibleTo(self.details_box) or labels)
        self.import_btn.setText("Next..." if labels and not goes_with_new_dataset else "Import")

    def _sync_selection(self) -> bi.ImportSelection:
        sel = self.selection
        sel.project = self._rows["project"][0].isChecked()
        sel.project_mode = "existing" if self.project_existing_radio.isChecked() else "new"
        sel.project_name = self.project_name_edit.text().strip()
        sel.existing_project = self.project_combo.currentData() or ""
        sel.pose = self._rows["pose"][0].isChecked()
        sel.behavior = self._rows["behavior"][0].isChecked()
        dataset_ticked = self._rows["dataset"][0].isChecked()
        existing = dataset_ticked and self.dataset_existing_radio.isChecked()
        sel.dataset = dataset_ticked and not existing
        if dataset_ticked:
            sel.target_dataset = (self.dataset_combo.currentData() or "") if existing else ""
        sel.dataset_name = self.name_edit.text().strip()
        sel.overwrite_project = self.project_replace.isVisibleTo(self) and self.project_replace.isChecked()
        sel.overwrite_dataset = self.dataset_replace.isVisibleTo(self) and self.dataset_replace.isChecked()
        media_text = self.media_edit.text().strip()
        sel.media_dir = self.contents.media_dir or (Path(media_text) if media_text else None)
        return sel

    def _revalidate(self):
        if self._loading or self.contents is None:
            return
        self._refresh_details()
        sel = self._sync_selection()
        problems = bi.validate_selection(self._workspace, self.contents, sel, check_target=False)
        if (self._rows["dataset"][0].isChecked() and self.dataset_existing_radio.isChecked()
                and not (sel.pose or sel.behavior)):
            problems.insert(0, "Using an existing dataset only adds labels to it \u2014 tick pose or behavior "
                               "labels, or choose to add the dataset as new.")
        self.problem_label.setText("\n".join(problems[:3]))
        self.import_btn.setEnabled(not problems)

    # ------------------------------------------------------------ finishing

    def _on_import_clicked(self):
        if self.compilation_root is not None:
            self.done(COMPILATION_RESULT)
            return
        sel = self._sync_selection()
        if sel.labels_need_target:
            # If they already chose which dataset the labels go on, don't ask again; the follow-up
            # is then only about how the labels compare and combine.
            chose_target = self._rows["dataset"][0].isChecked() and self.dataset_existing_radio.isChecked()
            dialog = AttachLabelsDialog(self._workspace, self.contents, sel, self, choose_target=not chose_target)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return          # back to this dialog so they can change their mind
        self.accept()
