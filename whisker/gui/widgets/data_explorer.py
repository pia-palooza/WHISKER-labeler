# UPDATE_FILE: whisker/gui/widgets/data_explorer.py
#
# A lightweight data explorer: a tree of the workspace's datasets, each
# expandable to its media files, with a checkmark on files that already have
# labels. Selecting a dataset or a file emits a signal so the main window can
# route it to the matching workflow tab. This is the standalone labeler's
# equivalent of the full WHISKER Data Explorer (minus the create/sample/delete
# actions, which belong to the full training pipeline).
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTreeWidget, QTreeWidgetItem, QLineEdit,
)

from whisker.core.dataset import DatasetType
from whisker.core.workspace import Workspace

# Item roles
ROLE_KIND = Qt.ItemDataRole.UserRole          # "dataset" | "file"
ROLE_DATASET = Qt.ItemDataRole.UserRole + 1   # dataset name
ROLE_RELPATH = Qt.ItemDataRole.UserRole + 2   # file rel path
ROLE_POPULATED = Qt.ItemDataRole.UserRole + 3 # bool, dataset children loaded


class DataExplorerWidget(QWidget):
    """Tree view over a workspace's datasets and their media files."""

    dataset_activated = pyqtSignal(str)        # dataset_name
    file_activated = pyqtSignal(str, str)      # dataset_name, rel_path

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._workspace: Optional[Workspace] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(QLabel("Data Explorer"))

        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter datasets...")
        self.filter_edit.textChanged.connect(self._apply_filter)
        layout.addWidget(self.filter_edit)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemExpanded.connect(self._on_item_expanded)
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.tree)

    # --- Population ---

    def set_workspace(self, workspace: Optional[Workspace]):
        self._workspace = workspace
        self.tree.clear()
        if not workspace:
            return

        for name in sorted(workspace.datasets.keys()):
            ds = workspace.datasets.get(name)
            if not ds:
                continue
            node = QTreeWidgetItem(self.tree)
            node.setText(0, f"{name}   [{ds.type.value}]   ({len(ds.files)})")
            node.setData(0, ROLE_KIND, "dataset")
            node.setData(0, ROLE_DATASET, name)
            node.setData(0, ROLE_POPULATED, False)
            # Add a placeholder child so the expand arrow appears.
            if ds.files:
                QTreeWidgetItem(node).setText(0, "Loading...")

    def _on_item_expanded(self, node: QTreeWidgetItem):
        if node.data(0, ROLE_KIND) != "dataset" or node.data(0, ROLE_POPULATED):
            return
        node.takeChildren()  # remove placeholder
        name = node.data(0, ROLE_DATASET)
        ds = self._workspace.datasets.get(name)
        labeled = self._labeled_keys(name, ds.type)

        for rel in ds.files:
            child = QTreeWidgetItem(node)
            child.setData(0, ROLE_KIND, "file")
            child.setData(0, ROLE_DATASET, name)
            child.setData(0, ROLE_RELPATH, rel)
            child.setText(0, self._file_label(rel, ds.type, labeled))
        node.setData(0, ROLE_POPULATED, True)

    def _labeled_keys(self, dataset_name: str, ds_type: DatasetType) -> set[str]:
        if not self._workspace:
            return set()
        try:
            if ds_type == DatasetType.VIDEO_COLLECTION:
                return self._workspace.behavior_labels.get_behavior_labeled_video_keys(
                    dataset_name
                )
            return self._workspace.pose_labels.get_pose_labeled_image_keys(dataset_name)
        except Exception:
            return set()

    def _file_label(self, rel: str, ds_type: DatasetType, labeled: set[str]) -> str:
        key = Path(rel).name if ds_type == DatasetType.VIDEO_COLLECTION else rel
        return f"✓ {rel}" if key in labeled else rel

    # --- Selection / navigation ---

    def _on_selection_changed(self):
        items = self.tree.selectedItems()
        if not items:
            return
        item = items[0]
        kind = item.data(0, ROLE_KIND)
        if kind == "dataset":
            self.dataset_activated.emit(item.data(0, ROLE_DATASET))
        elif kind == "file":
            self.file_activated.emit(
                item.data(0, ROLE_DATASET), item.data(0, ROLE_RELPATH)
            )

    def select_first_file_of_dataset(self, dataset_name: str):
        node = self._find_dataset_node(dataset_name)
        if not node:
            return
        node.setExpanded(True)  # triggers lazy population
        if node.childCount() > 0:
            self.tree.setCurrentItem(node.child(0))

    def step_selection(self, delta: int):
        """Move file selection by delta within the current dataset."""
        items = self.tree.selectedItems()
        if not items:
            return
        item = items[0]
        if item.data(0, ROLE_KIND) != "file":
            return
        parent = item.parent()
        idx = parent.indexOfChild(item)
        new_idx = idx + delta
        if 0 <= new_idx < parent.childCount():
            self.tree.setCurrentItem(parent.child(new_idx))

    def refresh_labels_for(self, dataset_name: str):
        node = self._find_dataset_node(dataset_name)
        if not node or not node.data(0, ROLE_POPULATED):
            return
        ds = self._workspace.datasets.get(dataset_name)
        labeled = self._labeled_keys(dataset_name, ds.type)
        for i in range(node.childCount()):
            child = node.child(i)
            rel = child.data(0, ROLE_RELPATH)
            child.setText(0, self._file_label(rel, ds.type, labeled))

    def _find_dataset_node(self, dataset_name: str) -> Optional[QTreeWidgetItem]:
        for i in range(self.tree.topLevelItemCount()):
            node = self.tree.topLevelItem(i)
            if node.data(0, ROLE_DATASET) == dataset_name:
                return node
        return None

    def _apply_filter(self, text: str):
        text = text.strip().lower()
        for i in range(self.tree.topLevelItemCount()):
            node = self.tree.topLevelItem(i)
            name = (node.data(0, ROLE_DATASET) or "").lower()
            node.setHidden(bool(text) and text not in name)
