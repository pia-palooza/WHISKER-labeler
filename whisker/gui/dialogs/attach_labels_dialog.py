"""Follow-up question for labels imported without their dataset: which of your datasets
do they belong to? Checks the labels against that dataset's files, reports mismatches,
and lets the user choose how to combine them with labels the dataset already has."""

from html import escape
from typing import Callable, Dict, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QLabel,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from whisker.core import bundle_import as bi
from whisker.core.bundle_import import LabelPolicy

_OK = "color: #2e7d32;"
_BAD = "color: #c0392b;"
_WARN = "color: #e67e22;"
_MUTED = "color: gray;"


def _examples(keys, n: int = 4) -> str:
    shown = ", ".join(escape(str(k)) for k in keys[:n])
    return shown + (f" and {len(keys) - n} more" if len(keys) > n else "")


class AttachLabelsDialog(QDialog):
    """Choose the target dataset and how to combine labels. Writes its answers into
    ``selection`` (target_dataset, pose_policy, behavior_policy, keep_unmatched) on accept."""

    def __init__(self, workspace, contents: bi.BundleContents, selection: bi.ImportSelection, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._ws = workspace
        self._contents = contents
        self._sel = selection
        self._reports: Dict[str, bi.LabelReport] = {}
        self._policy_of: Dict[str, Callable[[], LabelPolicy]] = {}

        self.setWindowTitle("Which dataset are these labels for?")
        screen = QApplication.primaryScreen()
        dpi = screen.logicalDotsPerInch() / 96.0 if screen else 1.0
        self.setMinimumWidth(int(640 * dpi))

        layout = QVBoxLayout(self)
        intro = QLabel(
            "These labels aren't being imported with their videos/frames, so choose the dataset "
            "you already have that they belong to. They'll be checked against its files first."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.dataset_combo = QComboBox()
        ranked = bi.rank_target_datasets(workspace, contents)
        for name, hits, total in ranked:
            self.dataset_combo.addItem(f"{name}    — {hits} of {total} labels match its files", name)
        if selection.target_dataset:
            i = self.dataset_combo.findData(selection.target_dataset)
            if i >= 0:
                self.dataset_combo.setCurrentIndex(i)
        layout.addWidget(self.dataset_combo)

        self.sections = QVBoxLayout()
        layout.addLayout(self.sections)

        self.keep_unmatched = QCheckBox()
        self.keep_unmatched.setVisible(False)
        layout.addWidget(self.keep_unmatched)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.ok_btn = self.button_box.button(QDialogButtonBox.StandardButton.Ok)
        self.ok_btn.setText("Import")
        self.button_box.accepted.connect(self._accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        self.dataset_combo.currentIndexChanged.connect(self._refresh)
        if not ranked:
            intro.setText("You don't have any datasets yet. Import the dataset (tick the frames/videos) "
                          "together with these labels, or create a dataset first.")
            intro.setStyleSheet(_BAD)
            self.dataset_combo.setEnabled(False)
            self.ok_btn.setEnabled(False)
        else:
            self._refresh()

    # ---------------------------------------------------------------- report

    def _target(self) -> str:
        return self.dataset_combo.currentData() or ""

    def _clear_sections(self):
        while self.sections.count():
            item = self.sections.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._policy_of.clear()
        self._reports.clear()

    def _refresh(self):
        self._clear_sections()
        target = self._target()
        if not target:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            if self._sel.pose:
                self._add_section("pose", "Pose labels", "labeled frames", "files",
                                  lambda: bi.analyze_pose_labels(self._ws, target, self._contents.pose.path))
            if self._sel.behavior:
                self._add_section("behavior", "Behavior labels", "labeled videos", "videos",
                                  lambda: bi.analyze_behavior_labels(self._ws, target, self._contents.behavior.path))
        finally:
            QApplication.restoreOverrideCursor()

        # Only labels that could actually be imported count: a kind that matches nothing is
        # already blocked as "probably the wrong dataset".
        unmatched = sum(len(r.unmatched) for r in self._reports.values() if not r.nothing_matches)
        self.keep_unmatched.setVisible(unmatched > 0)
        if unmatched:
            self.keep_unmatched.setText(
                f"Also import the {unmatched} label(s) that don't match a file (normally they're skipped)"
            )
            self.keep_unmatched.setChecked(False)
        self._update_ok()

    def _add_section(self, kind: str, title: str, unit: str, file_word: str, analyze):
        box = QGroupBox(title)
        v = QVBoxLayout(box)
        try:
            r = analyze()
        except Exception as e:
            err = QLabel(f"Couldn't read these labels: {escape(str(e))}")
            err.setStyleSheet(_BAD)
            err.setWordWrap(True)
            v.addWidget(err)
            self._policy_of[kind] = lambda: LabelPolicy.SKIP
            self.sections.addWidget(box)
            return
        self._reports[kind] = r

        lines = []
        if r.nothing_matches:
            lines.append(f"<span style='color:#c0392b'>None of the {r.total} {unit} match a file in this dataset "
                         "— this is probably the wrong dataset.</span>")
        else:
            lines.append(f"<span style='color:#2e7d32'>✔ {len(r.matched)} of {r.total} {unit} match {file_word} "
                         "in this dataset.</span>")
            if r.unmatched:
                lines.append(f"<span style='color:#e67e22'>⚠ {len(r.unmatched)} don't match any of its {file_word} "
                             f"({_examples(r.unmatched)}).</span>")
            if r.unlabeled_files:
                lines.append(f"<span style='color:gray'>{r.unlabeled_files} of the dataset's {r.dataset_files} "
                             f"{file_word} have no imported label.</span>")
        for w in r.warnings:
            lines.append(f"<span style='color:#e67e22'>⚠ {escape(w)}</span>")
        summary = QLabel("<br>".join(lines))
        summary.setWordWrap(True)
        summary.setTextFormat(Qt.TextFormat.RichText)
        v.addWidget(summary)

        if r.nothing_matches:
            self._policy_of[kind] = lambda: LabelPolicy.SKIP
        elif not r.has_existing:
            v.addWidget(self._muted("This dataset has none yet, so these will simply be added."))
            self._policy_of[kind] = lambda: LabelPolicy.ADD
        else:
            v.addWidget(self._muted(
                f"This dataset already has {unit} for {r.existing_total} "
                f"of its {file_word}; {len(r.overlapping)} of those are also in the import."
            ))
            self._policy_of[kind] = self._add_policy_choices(v, r)
        self.sections.addWidget(box)

    def _muted(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet(_MUTED)
        return label

    def _add_policy_choices(self, layout, r: bi.LabelReport) -> Callable[[], LabelPolicy]:
        group = QButtonGroup(layout.parent())
        choices = []
        if r.overlapping:
            choices.append((LabelPolicy.MERGE_EXISTING, "Combine — where both have a label, keep mine", r.can_merge))
            choices.append((LabelPolicy.MERGE_IMPORTED, "Combine — where both have a label, use the imported one", r.can_merge))
        else:
            choices.append((LabelPolicy.MERGE_IMPORTED, "Combine — add the imported labels to the ones I have", r.can_merge))
        choices.append((LabelPolicy.REPLACE, "Replace all my labels with the imported ones (mine are lost)", True))
        choices.append((LabelPolicy.SKIP, "Don't import these labels", True))

        default = next((p for p, _t, enabled in choices if enabled), LabelPolicy.SKIP)
        default = default if r.can_merge else LabelPolicy.SKIP    # never default to something destructive
        radios: Dict[LabelPolicy, QRadioButton] = {}
        for policy, text, enabled in choices:
            radio = QRadioButton(text)
            radio.setEnabled(enabled)
            group.addButton(radio)
            layout.addWidget(radio)
            radios[policy] = radio
        radios[default].setChecked(True)
        # Connect only now: toggled fires while the default is being ticked, before this
        # section's getter is registered, and an exception in a slot aborts a PyQt6 app.
        for radio in radios.values():
            radio.toggled.connect(self._update_ok)
        for problem in r.problems:
            note = QLabel(f"Can't combine: {escape(problem)}")
            note.setWordWrap(True)
            note.setStyleSheet(_BAD)
            layout.addWidget(note)
        return lambda: next((p for p, rb in radios.items() if rb.isChecked()), LabelPolicy.SKIP)

    # ---------------------------------------------------------------- result

    def _update_ok(self, *_):
        applicable = [
            k for k, r in self._reports.items()
            if k in self._policy_of and not r.nothing_matches and self._policy_of[k]() != LabelPolicy.SKIP
        ]
        self.ok_btn.setEnabled(bool(applicable))

    def _accept(self):
        self._sel.target_dataset = self._target()
        self._sel.pose_policy = self._policy_of["pose"]() if "pose" in self._policy_of else LabelPolicy.SKIP
        self._sel.behavior_policy = self._policy_of["behavior"]() if "behavior" in self._policy_of else LabelPolicy.SKIP
        self._sel.keep_unmatched = self.keep_unmatched.isVisibleTo(self) and self.keep_unmatched.isChecked()
        self.accept()
