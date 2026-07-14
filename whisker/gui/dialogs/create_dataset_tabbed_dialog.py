from typing import Optional, Tuple
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget,
    QDialog,
    QVBoxLayout,
    QTabWidget,
    QMessageBox,
)

from whisker.core.workspace import Workspace
from whisker.core.study.dataset import DatasetType
from whisker.gui.dialogs.create_dataset_dialog import CreateDatasetDialog
from whisker.gui.dialogs.multi_arena import MultiArenaDatasetPanel


class CreateDatasetTabbedDialog(QDialog):
    """
    Two-tab container for creating a new dataset.

    Tab 1 ("Single Arena Datasets") is the *existing* ``CreateDatasetDialog``
    embedded intact — its widgets, validation, and result contract are reused
    unchanged so the single-arena flow behaves exactly as it does today. This
    container adds no logic of its own to that tab; it only bridges the embedded
    dialog's accept/reject to itself and delegates ``get_dataset_info()``.

    Tab 2 ("Multi-Arena Datasets") hosts the arena-box placement editor. It is
    in-memory only for now; serialization and dataset creation from its
    configuration are wired in a later phase.
    """

    def __init__(self, workspace: Workspace, parent: QWidget | None = None):
        super().__init__(parent)
        self._workspace = workspace
        self.setWindowTitle("Add New Dataset")
        # Which tab produced an accepted result: "single" or "multi".
        self._accepted_mode: Optional[str] = None

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # --- Tab 1: existing single-arena dialog, embedded intact ---
        # Reparenting the QDialog into the tab strips its top-level window
        # behaviour and renders it as a plain panel. We never call exec() on it,
        # so it is not modal; its own OK/Cancel drive this container instead.
        self.single_arena_dialog = CreateDatasetDialog(workspace, parent=self)
        self.single_arena_dialog.setWindowFlags(Qt.WindowType.Widget)
        self.tabs.addTab(self.single_arena_dialog, "Single Arena Datasets")

        # --- Tab 2: multi-arena placement editor (in-memory; persistence
        # and dataset creation are wired in a later phase) ---
        self.multi_arena_panel = MultiArenaDatasetPanel()
        self.tabs.addTab(self.multi_arena_panel, "Multi-Arena Datasets")

        # Single Arena is the default selected tab.
        self.tabs.setCurrentIndex(0)

        # Bridge the embedded dialog's result signals to this container so that
        # clicking OK/Cancel inside Tab 1 accepts/rejects the whole popup.
        self.single_arena_dialog.accepted.connect(self._on_single_accepted)
        self.single_arena_dialog.rejected.connect(self.reject)

        # Tab 2's own Create/Cancel buttons drive the container.
        self.multi_arena_panel.create_requested.connect(self._on_multi_create)
        self.multi_arena_panel.cancel_requested.connect(self.reject)

    def _on_single_accepted(self):
        self._accepted_mode = "single"
        self.accept()

    def _on_multi_create(self):
        info = self.multi_arena_panel.get_multi_arena_info()
        if info is None:
            QMessageBox.warning(
                self,
                "Incomplete Configuration",
                "Please provide a dataset name, a source folder, and place at "
                "least one arena box before creating the dataset.",
            )
            return
        self._accepted_mode = "multi"
        self.accept()

    def get_dataset_info(self) -> Optional[Tuple[str, DatasetType, Path]]:
        """Delegates to the embedded single-arena dialog's result contract."""
        return self.single_arena_dialog.get_dataset_info()

    def get_result(self) -> Optional[Tuple[str, object]]:
        """
        Unified result accessor for the caller.

        Returns ``("single", (name, DatasetType, folder))`` for the single-arena
        tab, ``("multi", info_dict)`` for the multi-arena tab, or ``None`` if the
        dialog was not accepted. The multi-arena ``info_dict`` has keys
        ``name``, ``folder_path``, ``box_width``, ``box_height``, ``placements``.
        """
        if self.result() != QDialog.DialogCode.Accepted:
            return None
        if self._accepted_mode == "multi":
            return "multi", self.multi_arena_panel.get_multi_arena_info()
        if self._accepted_mode == "single":
            return "single", self.single_arena_dialog.get_dataset_info()
        return None
