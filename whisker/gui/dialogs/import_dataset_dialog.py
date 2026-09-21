"""Import from separate files: for files that didn't come from Export.

Nothing here should make you dig through folders for something you already have in your workspace.
Pick your existing project (your active one is offered first) and your existing dataset from lists;
browse only for what's genuinely new: a project file, a new dataset's info file and media, and label
files. Each field is checked as soon as it's set, with a specific reason if it's wrong.

Like the one-pick import, this builds a description of the chosen pieces
(:func:`whisker.core.bundle_import.contents_from_files`) and a selection, so labels going onto an
existing dataset get the same mismatch report and combine / replace choices, and it runs through the
same import code.
"""

from pathlib import Path
from typing import Dict, Optional

from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
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
from whisker.core import manual_import as mi
from whisker.core.workspace import Workspace
from whisker.gui.dialogs.attach_labels_dialog import AttachLabelsDialog

_COLORS = {"ok": "#2e7d32", "bad": "#c0392b", "warn": "#e67e22", "muted": "gray"}


class ImportDatasetDialog(QDialog):
    """Choose existing things from lists, browse only for new ones, then import."""

    def __init__(self, workspace: Workspace, parent: Optional[QWidget] = None,
                 active_project_name: Optional[str] = None):
        super().__init__(parent)
        self._ws = workspace
        self._active_project = active_project_name
        self._loading = True
        self._key: Optional[tuple] = None
        self._files = bi.contents_from_files()
        self._name_touched = {"project": False, "dataset": False}
        self._dataset_touched = False          # has the user picked a dataset themselves?
        self.contents: Optional[bi.BundleContents] = None      # set when the user accepts
        self.selection: Optional[bi.ImportSelection] = None

        self.setWindowTitle("Import from Separate Files")
        self.setMinimumWidth(720)
        root = QVBoxLayout(self)
        hint = QLabel(
            "For files that didn't come from Export. Choose things you already have from the lists; "
            "browse only for what's new."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {_COLORS['muted']};")
        root.addWidget(hint)

        root.addWidget(self._build_project_group())
        root.addWidget(self._build_dataset_group())
        root.addWidget(self._build_labels_group())

        self.problem_label = QLabel("")
        self.problem_label.setWordWrap(True)
        self.problem_label.setStyleSheet(f"color: {_COLORS['warn']};")
        root.addWidget(self.problem_label)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.import_btn = self.button_box.button(QDialogButtonBox.StandardButton.Ok)
        self.import_btn.setText("Import")
        self.button_box.accepted.connect(self._on_import_clicked)
        self.button_box.rejected.connect(self.reject)
        root.addWidget(self.button_box)

        self._populate_choices()
        for radio in (self.project_existing_radio, self.project_file_radio,
                      self.dataset_existing_radio, self.dataset_new_radio):
            radio.toggled.connect(self._revalidate)
        for edit in (self.project_edit, self.project_name_edit, self.dataset_edit, self.media_edit,
                     self.dataset_name_edit, self.pose_edit, self.behavior_edit):
            edit.textChanged.connect(self._revalidate)
        for combo in (self.project_combo, self.dataset_combo):
            combo.currentIndexChanged.connect(self._revalidate)
        self.dataset_combo.activated.connect(lambda _i: setattr(self, "_dataset_touched", True))    # a real user pick
        for box in (self.project_replace, self.dataset_replace):
            box.toggled.connect(self._revalidate)
        self.project_name_edit.textEdited.connect(lambda _t: self._name_touched.update(project=True))
        self.dataset_name_edit.textEdited.connect(lambda _t: self._name_touched.update(dataset=True))

        self._loading = False
        self._revalidate()

    # ------------------------------------------------------------ building

    @staticmethod
    def _row(*widgets) -> QWidget:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        for i, w in enumerate(widgets):
            h.addWidget(w, 1 if i == len(widgets) - 1 else 0)
        return row

    def _status(self) -> QLabel:
        label = QLabel("")
        label.setWordWrap(True)
        return label

    def _path_field(self, placeholder: str, browse, clearable: bool = False):
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        button = QPushButton("Browse...")
        button.clicked.connect(lambda: browse(edit))
        parts = [edit, button]
        if clearable:
            clear = QPushButton("Clear")
            clear.clicked.connect(edit.clear)
            parts.append(clear)
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(edit, 1)
        for b in parts[1:]:
            h.addWidget(b)
        return row, edit

    def _build_project_group(self) -> QGroupBox:
        box = QGroupBox("Project")
        v = QVBoxLayout(box)
        self.project_existing_radio = QRadioButton("Use my existing project:")
        self.project_combo = QComboBox()
        v.addWidget(self._row(self.project_existing_radio, self.project_combo))
        self.project_note = self._status()
        v.addWidget(self.project_note)
        self.project_file_radio = QRadioButton("Add a project from a file:")
        file_row, self.project_edit = self._path_field("Select the project .json file...", self._browse_project)
        v.addWidget(self.project_file_radio)
        v.addWidget(file_row)
        self.project_name_edit = QLineEdit()
        self.project_name_row = self._row(QLabel("Save it as:"), self.project_name_edit)
        v.addWidget(self.project_name_row)
        self.project_replace = QCheckBox()
        v.addWidget(self.project_replace)
        self.project_status = self._status()
        v.addWidget(self.project_status)
        group = QButtonGroup(self)
        group.addButton(self.project_existing_radio)
        group.addButton(self.project_file_radio)
        self._project_group = group
        return box

    def _build_dataset_group(self) -> QGroupBox:
        box = QGroupBox("Dataset")
        v = QVBoxLayout(box)
        self.dataset_existing_radio = QRadioButton("Use my existing dataset (its labels are added; nothing is copied):")
        self.dataset_combo = QComboBox()
        v.addWidget(self.dataset_existing_radio)
        v.addWidget(self.dataset_combo)
        self.dataset_new_radio = QRadioButton("Add a new dataset from files:")
        v.addWidget(self.dataset_new_radio)
        info_row, self.dataset_edit = self._path_field("Select the dataset's manifest.json...", self._browse_dataset)
        self.dataset_info_label = QLabel("Dataset info file:")
        v.addWidget(self.dataset_info_label)
        v.addWidget(info_row)
        self.dataset_status = self._status()
        v.addWidget(self.dataset_status)
        self.media_label = QLabel("Media folder:")
        media_row, self.media_edit = self._path_field("Select the folder containing the frames/videos...", self._browse_media)
        v.addWidget(self.media_label)
        v.addWidget(media_row)
        self.media_status = self._status()
        v.addWidget(self.media_status)
        self.dataset_name_edit = QLineEdit()
        self.dataset_name_row = self._row(QLabel("Name:"), self.dataset_name_edit)
        v.addWidget(self.dataset_name_row)
        self.dataset_replace = QCheckBox()
        v.addWidget(self.dataset_replace)
        group = QButtonGroup(self)
        group.addButton(self.dataset_existing_radio)
        group.addButton(self.dataset_new_radio)
        self._dataset_group = group
        return box

    def _build_labels_group(self) -> QGroupBox:
        box = QGroupBox("Labels (optional)")
        v = QVBoxLayout(box)
        v.addWidget(QLabel("Pose labels file (.h5):"))
        row, self.pose_edit = self._path_field("(optional) Select a pose labels.h5 file...", self._browse_h5, clearable=True)
        v.addWidget(row)
        self.pose_status = self._status()
        v.addWidget(self.pose_status)
        v.addWidget(QLabel("Behavior labels file (.h5):"))
        row, self.behavior_edit = self._path_field("(optional) Select a behavior labels.h5 file...", self._browse_h5, clearable=True)
        v.addWidget(row)
        self.behavior_status = self._status()
        v.addWidget(self.behavior_status)
        return box

    def _populate_choices(self):
        projects = sorted(self._ws.projects.keys())
        for name in projects:
            self.project_combo.addItem(name, name)
        if self._active_project in projects:
            self.project_combo.setCurrentIndex(self.project_combo.findData(self._active_project))
        self.project_existing_radio.setEnabled(bool(projects))
        (self.project_existing_radio if projects else self.project_file_radio).setChecked(True)

        self._fill_dataset_combo()
        datasets = self.dataset_combo.count()
        self.dataset_existing_radio.setEnabled(bool(datasets))
        (self.dataset_existing_radio if datasets else self.dataset_new_radio).setChecked(True)

    def _fill_dataset_combo(self):
        """Your datasets, best fit for the chosen label files first; alphabetical until there are any."""
        current = self.dataset_combo.currentData()
        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()
        ranked = self._files.pose.ok or self._files.behavior.ok
        if ranked:
            for name, hits, total in bi.rank_target_datasets(self._ws, self._files):
                self.dataset_combo.addItem(f"{name}    — {hits} of {total} labels match" if total else name, name)
        else:
            for name in sorted(self._ws.datasets.keys(), key=str.lower):
                self.dataset_combo.addItem(name, name)
        # A dataset the user picked stays picked; until they pick, the default follows the best fit.
        if current and (self._dataset_touched or not ranked):
            self.dataset_combo.setCurrentIndex(max(0, self.dataset_combo.findData(current)))
        self.dataset_combo.blockSignals(False)

    # ------------------------------------------------------------ browsing

    def _browse_file(self, edit: QLineEdit, title: str, name_filter: str):
        start = edit.text().strip() or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(self, title, start, name_filter)
        if path:
            edit.setText(path)

    def _browse_project(self, edit):
        self._browse_file(edit, "Select Project File", "JSON files (*.json)")

    def _browse_dataset(self, edit):
        self._browse_file(edit, "Select Dataset Info File", "JSON files (*.json)")

    def _browse_h5(self, edit):
        self._browse_file(edit, "Select Labels File", "HDF5 files (*.h5)")

    def _browse_media(self, edit):
        start = edit.text().strip() or str(Path.home())
        path = QFileDialog.getExistingDirectory(self, "Select Media Folder", start)
        if path:
            edit.setText(path)

    # ------------------------------------------------------------ state

    @staticmethod
    def _path_or_none(text: str) -> Optional[Path]:
        text = text.strip()
        return Path(text) if text else None

    def _set(self, label: QLabel, text: str, level: str = "muted"):
        label.setText(text)
        label.setStyleSheet(f"color: {_COLORS[level]};")
        label.setVisible(bool(text))

    def _build_selection(self) -> bi.ImportSelection:
        sel = bi.ImportSelection()
        if self.project_existing_radio.isChecked():
            sel.project, sel.project_mode = True, "existing"
            sel.existing_project = self.project_combo.currentData() or ""
        else:
            sel.project = bool(self.project_edit.text().strip())
            sel.project_name = self.project_name_edit.text().strip()
            sel.overwrite_project = self.project_replace.isVisibleTo(self) and self.project_replace.isChecked()
        sel.pose = bool(self.pose_edit.text().strip())
        sel.behavior = bool(self.behavior_edit.text().strip())
        if self.dataset_existing_radio.isChecked():
            sel.target_dataset = self.dataset_combo.currentData() or ""
        else:
            sel.dataset = bool(self.dataset_edit.text().strip())
            sel.dataset_name = self.dataset_name_edit.text().strip()
            sel.media_dir = self._path_or_none(self.media_edit.text())
            sel.overwrite_dataset = self.dataset_replace.isVisibleTo(self) and self.dataset_replace.isChecked()
        return sel

    def _reload_files_if_changed(self):
        project_mode = self.project_file_radio.isChecked()
        dataset_mode = self.dataset_new_radio.isChecked()
        key = (
            self.project_edit.text().strip() if project_mode else "",
            self.dataset_edit.text().strip() if dataset_mode else "",
            self.media_edit.text().strip() if dataset_mode else "",
            self.pose_edit.text().strip(),
            self.behavior_edit.text().strip(),
        )
        if key == self._key:
            return
        self._key = key
        self._files = bi.contents_from_files(*(self._path_or_none(k) for k in key))
        c = self._files
        # offer the files' own names, made unique, until the user types their own
        for kind, edit, obj, unique in (
            ("project", self.project_name_edit, c.project_obj, bi.unique_project_name),
            ("dataset", self.dataset_name_edit, c.dataset_obj, bi.unique_dataset_name),
        ):
            if obj is not None and not self._name_touched[kind]:
                edit.blockSignals(True)
                edit.setText(unique(self._ws, obj.name))
                edit.blockSignals(False)
        self._fill_dataset_combo()

    def _revalidate(self, *_):
        if self._loading:
            return
        self._reload_files_if_changed()
        c = self._files
        project_file = self.project_file_radio.isChecked()
        dataset_new = self.dataset_new_radio.isChecked()

        # --- show only what applies
        self.project_combo.setEnabled(not project_file)
        for w in (self.project_edit, self.project_name_row):
            w.setEnabled(project_file)
        for w in (self.dataset_edit, self.media_edit, self.dataset_name_row):
            w.setEnabled(dataset_new)
        self.dataset_combo.setEnabled(not dataset_new)
        sel = self._build_selection()

        project_name = self.project_name_edit.text().strip()
        self.project_replace.setVisible(project_file and bool(project_name) and self._ws.projects.get(project_name) is not None)
        self.project_replace.setText(f"Replace my project '{project_name}' with this one")
        dataset_name = self.dataset_name_edit.text().strip()
        self.dataset_replace.setVisible(dataset_new and bool(dataset_name) and self._ws.datasets.get(dataset_name) is not None)
        self.dataset_replace.setText(f"Replace the dataset I already have called '{dataset_name}' (its media and labels)")
        sel = self._build_selection()

        # --- per-field status
        if not project_file:
            gaps = bi.project_gaps(self._ws, c, sel.existing_project, pose=sel.pose, behavior=sel.behavior) if sel.existing_project else []
            if gaps:
                self._set(self.project_note, f"'{sel.existing_project}' doesn't define {'; '.join(gaps)}, which these labels use — "
                                             "they'll import, but those won't show up when labeling with it.", "warn")
            else:
                self._set(self.project_note, "Nothing is added; your project is used as it is." if sel.existing_project else "")
        else:
            self._set(self.project_note, "")
        text = self.project_edit.text().strip()
        self._set(self.project_status,
                  (c.project.summary or c.project.problem) if project_file and text else "",
                  "ok" if c.project.ok else "bad")
        if dataset_new:
            info = self.dataset_edit.text().strip()
            if info:
                check, _ds = mi.check_dataset_file(Path(info))
                self._set(self.dataset_status, check.message, "ok" if check.ok else "bad")
            else:
                self._set(self.dataset_status, "")
            media = self.media_edit.text().strip()
            if media and c.dataset_obj is not None:
                check, _missing = mi.check_media_folder(Path(media), c.dataset_obj)
                self._set(self.media_status, check.message, "ok" if check.ok else "bad")
            else:
                self._set(self.media_status, "")
        else:
            self._set(self.dataset_status, "")
            self._set(self.media_status, "")
        for part, label, text_edit in ((c.pose, self.pose_status, self.pose_edit), (c.behavior, self.behavior_status, self.behavior_edit)):
            if text_edit.text().strip():
                self._set(label, part.summary if part.ok else part.problem, "ok" if part.ok else "bad")
            else:
                self._set(label, "")

        # --- can we import?
        problems = []
        if project_file and not self.project_edit.text().strip():
            problems.append("Choose the project file, or use one of your existing projects.")
        if dataset_new and not self.dataset_edit.text().strip():
            problems.append("Choose the dataset info file (manifest.json), or use one of your existing datasets.")
        if not problems and not (sel.adds_project or sel.dataset or sel.pose or sel.behavior):
            problems.append("Choose something to import: label files, a project file, or a new dataset.")
        if not problems:
            problems = bi.validate_selection(self._ws, c, sel)
        self.problem_label.setText("\n".join(problems[:3]))
        self.import_btn.setText("Next..." if sel.labels_need_target else "Import")
        self.import_btn.setEnabled(not problems)

    # ------------------------------------------------------------ finishing

    def _on_import_clicked(self):
        sel = self._build_selection()
        if sel.labels_need_target:
            # The dataset was chosen here, so the follow-up is only how the labels compare and combine.
            dialog = AttachLabelsDialog(self._ws, self._files, sel, self, choose_target=False)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
        self.contents, self.selection = self._files, sel
        self.accept()
