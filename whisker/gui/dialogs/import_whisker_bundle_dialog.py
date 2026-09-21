"""Import a Whisker bundle: pick the zip, see what is in it, tick what to bring in.

Opening the bundle reads only its listing and manifest. Each thing in it is marked New or Already here; nothing that
already exists changes unless it is ticked, and replacing asks first. Frame subsets are stored inside the workspace;
videos and image collections need a folder, which is asked for only when one of them is ticked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from whisker.core import whisker_bundle as wb

_SECTIONS = (("dataset", "Datasets"), ("labels", "Labels"), ("run", "Prediction results"), ("project", "Projects"))


class ImportWhiskerBundleDialog(QDialog):
    def __init__(self, workspace, parent: Optional[QWidget] = None, path: Optional[str] = None):
        super().__init__(parent)
        self.setWindowTitle("Import Whisker Bundle")
        self.setMinimumWidth(680)
        self._workspace = workspace
        self._bundle: Optional[wb.WhiskerBundle] = None
        self._path: Optional[Path] = None
        self._items: List[wb.PlanItem] = []
        self._rows: Dict[str, QTreeWidgetItem] = {}

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("A Whisker bundle is a .zip file made by Whisker or Whisker Labeler."))

        pick = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("Choose a Whisker bundle (.zip)...")
        self.browse_btn = QPushButton("Browse...")
        pick.addWidget(self.path_edit, 1)
        pick.addWidget(self.browse_btn)
        layout.addLayout(pick)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Import", "Details", "In this workspace"])
        self.tree.setRootIsDecorated(True)
        self.tree.setMinimumHeight(220)
        self.tree.setColumnWidth(0, 300)
        self.tree.setColumnWidth(1, 200)
        layout.addWidget(self.tree, 1)

        self.media_row = QWidget()
        media = QHBoxLayout(self.media_row)
        media.setContentsMargins(0, 0, 0, 0)
        media.addWidget(QLabel("Store videos and images in:"))
        self.media_edit = QLineEdit()
        self.media_edit.setPlaceholderText("Choose a folder...")
        self.media_browse_btn = QPushButton("Browse...")
        media.addWidget(self.media_edit, 1)
        media.addWidget(self.media_browse_btn)
        layout.addWidget(self.media_row)
        self.media_note = QLabel("Each dataset gets its own folder inside it. Existing media outside the workspace are never overwritten.")
        self.media_note.setWordWrap(True)
        layout.addWidget(self.media_note)

        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.import_btn = self.button_box.button(QDialogButtonBox.StandardButton.Ok)
        self.import_btn.setText("Import")
        layout.addWidget(self.button_box)

        self._refresh()
        # Connect last: a slot that raises aborts the whole app, so nothing fires while the dialog is half built.
        self.browse_btn.clicked.connect(self._on_browse)
        self.media_browse_btn.clicked.connect(self._on_browse_media)
        self.media_edit.textChanged.connect(lambda _t: self._refresh())
        self.tree.itemChanged.connect(lambda _i, _c: self._refresh())
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        if path:
            self.set_path(path)

    # -- choosing the bundle ------------------------------------------------------------------------
    def _on_browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose a Whisker bundle", "", "Whisker bundle (*.zip);;All files (*)")
        if path:
            self.set_path(path)

    def set_path(self, path: str):
        """Open ``path`` and list what it holds. Anything wrong with it is said in words, not raised."""
        self._close_bundle()
        self._path = None
        self.path_edit.setText(str(path))
        self._items, self._rows = [], {}
        self.tree.blockSignals(True)
        self.tree.clear()
        try:
            self._bundle = wb.WhiskerBundle.open(path)
            self._items = wb.plan_import(self._workspace, self._bundle)
        except wb.BundleError as e:
            self._close_bundle()
            self.tree.blockSignals(False)
            self._set_status(str(e), error=True)
            self._refresh()
            return
        self._path = Path(path)
        self._populate()
        self.tree.blockSignals(False)
        m = self._bundle.manifest
        made = f"Made by Whisker {m.whisker_version}" if m.whisker_version else "Made by Whisker"
        when = f" on {m.created_at[:10]}" if m.created_at else ""
        self._set_status(f"{made}{when}. Tick what you want to bring in.")
        self._refresh()

    def _populate(self):
        default = wb.default_choices(self._items)
        for kind, heading in _SECTIONS:
            members = [i for i in self._items if i.kind == kind]
            if not members:
                continue
            header = QTreeWidgetItem(self.tree, [heading])
            header.setFlags(Qt.ItemFlag.ItemIsEnabled)
            for item in members:
                row = QTreeWidgetItem(header, [item.title, item.detail, self._status_text(item)])
                flags = Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled
                row.setFlags(flags if item.importable else Qt.ItemFlag.NoItemFlags)
                row.setCheckState(0, Qt.CheckState.Checked if item.key in default else Qt.CheckState.Unchecked)
                if item.problem:
                    row.setToolTip(2, item.problem)
                elif item.exists:
                    row.setToolTip(2, "Already here. Ticking it replaces what you have.")
                self._rows[item.key] = row
            header.setExpanded(True)

    @staticmethod
    def _status_text(item: wb.PlanItem) -> str:
        if item.problem:
            return "Can't import"
        return "Already here (tick to replace)" if item.exists else "New"

    # -- what is chosen -----------------------------------------------------------------------------
    @property
    def bundle_path(self) -> Optional[Path]:
        return self._path

    @property
    def chosen(self) -> frozenset:
        return frozenset(k for k, row in self._rows.items() if row.checkState(0) == Qt.CheckState.Checked)

    @property
    def media_root(self) -> Optional[Path]:
        text = self.media_edit.text().strip()
        return Path(text) if text else None

    def _picked(self) -> List[wb.PlanItem]:
        chosen = self.chosen
        return [i for i in self._items if i.key in chosen]

    def _on_browse_media(self):
        path = QFileDialog.getExistingDirectory(self, "Where should the videos and images be stored?", self.media_edit.text())
        if path:
            self.media_edit.setText(path)

    # -- keeping the dialog honest ------------------------------------------------------------------
    def _set_status(self, text: str, error: bool = False):
        self.status_label.setText(text)
        self.status_label.setStyleSheet("color: #b00020;" if error else "")

    def _refresh(self):
        picked = self._picked()
        needs_folder = any(i.requires_media_folder for i in picked)
        self.media_row.setVisible(needs_folder)
        self.media_note.setVisible(needs_folder)

        problems: List[str] = []
        if self._bundle is not None and picked:
            try:
                problems = wb.validate_choices(self._workspace, self._bundle, set(self.chosen), self.media_root)
            except Exception as e:                      # never let a slot raise: Qt would abort the whole app
                problems = [f"Couldn't check the bundle: {e}"]
        replaced = [i for i in picked if i.exists]
        if not picked:
            summary = "Nothing is ticked." if self._bundle is not None else ""
        else:
            summary = f"{len(picked)} item(s) selected, {wb.format_bytes(sum(i.bytes for i in picked))}."
            if replaced:
                summary += f" {len(replaced)} of them already exist here and will be replaced."
        if problems:
            summary += ("\n" if summary else "") + problems[0]
        self.summary_label.setText(summary)
        self.summary_label.setStyleSheet("color: #b00020;" if problems else "")
        self.import_btn.setEnabled(self._bundle is not None and bool(picked) and not problems)

    def accept(self):
        replaced = [i for i in self._picked() if i.exists]
        if replaced:
            names = "\n".join(f"• {i.title}" for i in replaced[:12]) + (f"\n...and {len(replaced) - 12} more" if len(replaced) > 12 else "")
            answer = QMessageBox.question(
                self, "Replace Existing?",
                f"These already exist in this workspace and will be replaced by what is in the bundle:\n\n{names}\n\nReplace them?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        super().accept()

    def _close_bundle(self):
        if self._bundle is not None:
            self._bundle.close()
            self._bundle = None

    def done(self, result: int):
        # The job reopens the file on its own thread, so this handle is no longer needed (and would lock the zip on Windows).
        self._close_bundle()
        super().done(result)
