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
    QVBoxLayout,
    QWidget,
)

from whisker.core import bundle_import as bi
from whisker.core.workspace import Workspace
from whisker.gui.dialogs.attach_labels_dialog import AttachLabelsDialog

_OK = "color: #2e7d32;"
_BAD = "color: #c0392b;"
_WARN = "color: #e67e22;"
_MUTED = "color: gray;"

ADVANCED_RESULT = 2   # dialog.exec() value when the user chose "Pick pieces manually"


class ImportBundleDialog(QDialog):
    """One pick -> checklist -> import."""

    def __init__(self, workspace: Workspace, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._workspace = workspace
        self.contents: Optional[bi.BundleContents] = None
        self.selection: Optional[bi.ImportSelection] = None
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

        self.project_replace = QCheckBox()
        self.project_replace.toggled.connect(self._revalidate)
        d.addWidget(self.project_replace)

        name_row = QWidget()
        nr = QHBoxLayout(name_row)
        nr.setContentsMargins(0, 0, 0, 0)
        self.name_label = QLabel("Name for the imported dataset:")
        self.name_edit = QLineEdit()
        self.name_edit.textChanged.connect(self._revalidate)
        nr.addWidget(self.name_label)
        nr.addWidget(self.name_edit, 1)
        self.name_row = name_row
        d.addWidget(name_row)

        self.dataset_replace = QCheckBox()
        self.dataset_replace.toggled.connect(self._revalidate)
        d.addWidget(self.dataset_replace)

        self.media_info = QLabel("")
        self.media_info.setWordWrap(True)
        d.addWidget(self.media_info)
        media_row = QWidget()
        mr = QHBoxLayout(media_row)
        mr.setContentsMargins(0, 0, 0, 0)
        self.media_edit = QLineEdit()
        self.media_edit.textChanged.connect(self._revalidate)
        media_browse = QPushButton("Browse...")
        media_browse.clicked.connect(self._browse_media)
        mr.addWidget(self.media_edit, 1)
        mr.addWidget(media_browse)
        self.media_row = media_row
        d.addWidget(media_row)

        self.labels_note = QLabel("")
        self.labels_note.setWordWrap(True)
        self.labels_note.setStyleSheet(_MUTED)
        d.addWidget(self.labels_note)
        root.addWidget(self.details_box)
        self.details_box.setVisible(False)

        self.problem_label = QLabel("")
        self.problem_label.setWordWrap(True)
        self.problem_label.setStyleSheet(_WARN)
        root.addWidget(self.problem_label)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        root.addWidget(line)

        self.button_box = QDialogButtonBox()
        self.manual_btn = self.button_box.addButton("Pick pieces manually...", QDialogButtonBox.ButtonRole.ActionRole)
        self.manual_btn.setToolTip("For files that didn't come from Export: choose the project, dataset, media and label files one by one.")
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
        self.contents = self.selection = None
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
        if loc.found:
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
        if root:
            self._load(Path(root), "")

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
        ticked = {"project": selection.project, "dataset": selection.dataset, "pose": selection.pose, "behavior": selection.behavior}
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

        self.name_edit.setText(selection.dataset_name)
        self.media_edit.setText(str(selection.media_dir) if selection.media_dir and not contents.media_dir else "")
        self._loading = False
        self._refresh_details()
        self._revalidate()

    def _kind(self) -> str:
        return self.contents.media_kind if self.contents else "frames"

    # ------------------------------------------------------------ details

    def _on_part_toggled(self):
        if self._loading:
            return
        self._refresh_details()
        self._revalidate()

    def _refresh_details(self):
        """Show only the follow-up questions that apply to the current ticks."""
        c = self.contents
        if c is None:
            return
        ws = self._workspace
        want = {k: self._rows[k][0].isChecked() for k in self._rows}

        # project
        relation = bi.project_relation(ws, c) if c.project_obj else "new"
        show_project = want["project"] and relation != "new"
        self.project_replace.setVisible(show_project)
        if show_project:
            self.project_replace.setText(f"Replace my project '{c.project_obj.name}' with the one in this export")
            self.project_replace.setToolTip("Unticked, your existing project is kept and this one is skipped.")

        # dataset
        self.name_row.setVisible(want["dataset"])
        exists = want["dataset"] and ws.datasets.get(self.name_edit.text().strip()) is not None
        self.dataset_replace.setVisible(exists)
        if exists:
            self.dataset_replace.setText(
                f"Replace the dataset I already have called '{self.name_edit.text().strip()}' "
                "(its media and labels)"
            )
        show_media = want["dataset"]
        self.media_info.setVisible(show_media)
        self.media_row.setVisible(show_media and c.media_dir is None)
        if show_media:
            if c.media_dir is None:
                self.media_info.setText(
                    f"The {c.media_kind} aren't inside this export. Choose the folder that contains them:"
                )
                self.media_info.setStyleSheet(_WARN)
            else:
                where = "inside the export" if c.media_included else "at their original location"
                self.media_info.setText(f"The {c.media_kind} will be copied from {where}.")
                self.media_info.setStyleSheet(_MUTED)

        # labels
        labels = want["pose"] or want["behavior"]
        if labels and not want["dataset"]:
            self.labels_note.setText(
                "Next you'll choose which of your datasets these labels belong to, and how to "
                "combine them with any labels it already has."
            )
        elif labels:
            self.labels_note.setText("The labels will be attached to the dataset you're importing.")
        else:
            self.labels_note.setText("")
        self.labels_note.setVisible(labels)

        self.details_box.setVisible(any(w.isVisibleTo(self.details_box) for w in (
            self.project_replace, self.name_row, self.dataset_replace, self.media_info, self.media_row, self.labels_note)))
        self.import_btn.setText("Next..." if labels and not want["dataset"] else "Import")

    def _sync_selection(self) -> bi.ImportSelection:
        sel = self.selection
        sel.project = self._rows["project"][0].isChecked()
        sel.dataset = self._rows["dataset"][0].isChecked()
        sel.pose = self._rows["pose"][0].isChecked()
        sel.behavior = self._rows["behavior"][0].isChecked()
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
        self.problem_label.setText("\n".join(problems[:3]))
        self.import_btn.setEnabled(not problems)

    # ------------------------------------------------------------ finishing

    def _on_import_clicked(self):
        sel = self._sync_selection()
        if sel.labels_need_target:
            dialog = AttachLabelsDialog(self._workspace, self.contents, sel, self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return          # back to this dialog so they can change their mind
        self.accept()
