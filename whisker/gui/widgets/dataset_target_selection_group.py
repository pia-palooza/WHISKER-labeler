from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QVBoxLayout, QTreeWidget, QTreeWidgetItem, QGroupBox, QHeaderView, QSplitter,
)

from whisker.core.workspace import Workspace, Dataset, DatasetType

# Input: Dataset. Output: (Parent Col Strings, Child File -> Col Strings)
HeaderLabelStatusHandler = Callable[
    [Dataset],
    tuple[
        list[str],
        dict[str, list[str]],  
    ]
]

# Input: Item, Dataset, specific_file_path (None if parent)
ItemRenderHandler = Callable[[QTreeWidgetItem, Dataset, Optional[str]], None]

class DatasetTargetSelectionGroup(QGroupBox):
    def __init__(
        self,
        data_tree_tuples: list[tuple[DatasetType, str]],
        title: str | None,
        header_labels: list[str] | None = None,
        header_label_status_handler: HeaderLabelStatusHandler | None = None,
        item_render_handler: ItemRenderHandler | None = None,
    ):
        super().__init__(title)
        self._header_labels = header_labels
        self._header_label_status_handler = header_label_status_handler
        self._item_render_handler = item_render_handler
        
        target_layout = QVBoxLayout(self)
        
        # Initialize separate item trees for each dataset item type
        header_labels = [] if header_labels is None else header_labels
        splitter = QSplitter(Qt.Orientation.Vertical)
        self._item_trees: dict[DatasetType, QTreeWidget] = {}
        
        for (dataset_type, tree_title) in data_tree_tuples:
            item_tree = QTreeWidget()
            item_tree.setHeaderLabels([tree_title] + header_labels)
            item_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            # Resize status columns to contents
            for i in range(1, len(header_labels) + 1):
                item_tree.header().setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
            
            # Connect change signal for manual tri-state logic
            item_tree.itemChanged.connect(self._on_item_changed)
            
            self._item_trees[dataset_type] = item_tree
            splitter.addWidget(item_tree)
            
        target_layout.addWidget(splitter)
    
    def clear(self):
        for item_tree in self._item_trees.values():
            item_tree.clear()
    
    def get_tree(self, item_type: DatasetType) -> QTreeWidget:
        if item_type not in self._item_trees:
            raise ValueError(
                f"{item_type} is not a valid DatasetType for this group. "
                f"Valid choices: {list(self._item_trees.keys())}"
            )
        return self._item_trees[item_type]

    def populate(self, workspace: Workspace):
        # Block signals to prevent triggering logic during bulk insertion
        for tree in self._item_trees.values():
            tree.blockSignals(True)

        try:
            # Iterate all datasets and categorize
            for ds in workspace.datasets.values():
                if self._header_label_status_handler:
                    parent_header_label_status, children_header_label_status = self._header_label_status_handler(ds)
                else:
                    parent_header_label_status = (
                        len(self._header_labels) * ["UNKNOWN"]
                        if self._header_labels else []
                    )
                    children_header_label_status = {}

                if ds.type not in self._item_trees:
                    continue

                # Add parent item for selecting/deselecting entire dataset
                parent = QTreeWidgetItem(self.get_tree(ds.type), [ds.name] + parent_header_label_status)
                # DEV_NOTE: Removed ItemIsAutoTristate to handle logic manually
                parent.setFlags(parent.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                parent.setCheckState(0, Qt.CheckState.Unchecked)
                
                # Allow custom styling of the parent item
                if self._item_render_handler:
                    self._item_render_handler(parent, ds, None)

                if ds.type == DatasetType.VIDEO_COLLECTION:
                    # Add child items for selecting / deselecting individual videos
                    for f in sorted(ds.files):
                        video_stem = Path(f).stem
                        video_header_label_status = children_header_label_status.get(
                            f,
                            len(self._header_labels) * ["UNKNOWN"]
                            if self._header_labels else []
                        )
                        item = QTreeWidgetItem(parent, [Path(f).name] + video_header_label_status)
                        item.setData(0, Qt.ItemDataRole.UserRole, (ds.name, video_stem))
                        item.setCheckState(0, Qt.CheckState.Unchecked)
                        
                        # Allow custom styling of the child item (which may disable it)
                        if self._item_render_handler:
                            self._item_render_handler(item, ds, f)
                else:
                    # Need to set parent's user role so that its target is appended when selected
                    parent.setData(0, Qt.ItemDataRole.UserRole, (ds.name, "_IMAGES_"))
        
        finally:
            # Re-enable signals
            for tree in self._item_trees.values():
                tree.blockSignals(False)
            
            # Manually trigger parent state update for all items
            for tree in self._item_trees.values():
                root = tree.invisibleRootItem()
                for i in range(root.childCount()):
                    self._update_parent_check_state(root.child(i))

    def _on_item_changed(self, item: QTreeWidgetItem, column: int):
        """
        Manually handles checking/unchecking to skip disabled items.
        """
        if column != 0:
            return

        tree = item.treeWidget()
        # Block signals to prevent recursive loops when we programmatically set states
        tree.blockSignals(True)
        try:
            if item.parent() is None:
                # Case 1: Parent Item Changed -> Propagate to valid children
                new_state = item.checkState(0)
                # If the user clicked a partially checked parent, it usually goes to Checked.
                # We enforce simple Checked/Unchecked logic for the click.
                if new_state == Qt.CheckState.PartiallyChecked:
                    new_state = Qt.CheckState.Checked
                    item.setCheckState(0, new_state)

                for i in range(item.childCount()):
                    child = item.child(i)
                    # CRITICAL FIX: Only propagate to Enabled items
                    if child.flags() & Qt.ItemFlag.ItemIsEnabled:
                        child.setCheckState(0, new_state)
            else:
                # Case 2: Child Item Changed -> Update Parent state
                self._update_parent_check_state(item.parent())
        finally:
            tree.blockSignals(False)

    def _update_parent_check_state(self, parent: QTreeWidgetItem):
        """
        Scans all ENABLED children to determine the parent's state.
        Disabled children are ignored in the calculation.
        """
        tree = parent.treeWidget()
        if tree:
            tree.blockSignals(True)
            
        try:
            enabled_children = []
            for i in range(parent.childCount()):
                child = parent.child(i)
                if child.flags() & Qt.ItemFlag.ItemIsEnabled:
                    enabled_children.append(child)
            
            # If no children are enabled, the parent is effectively unchecked/disabled contextually
            if not enabled_children:
                parent.setCheckState(0, Qt.CheckState.Unchecked)
                return

            all_checked = all(c.checkState(0) == Qt.CheckState.Checked for c in enabled_children)
            any_checked = any(c.checkState(0) != Qt.CheckState.Unchecked for c in enabled_children)

            if all_checked:
                parent.setCheckState(0, Qt.CheckState.Checked)
            elif any_checked:
                parent.setCheckState(0, Qt.CheckState.PartiallyChecked)
            else:
                parent.setCheckState(0, Qt.CheckState.Unchecked)
        finally:
            if tree:
                tree.blockSignals(False)

    def get_selected_targets(self) -> list[tuple[str, str]]:
        targets: list[tuple[str, str]] = []
        
        def _append_target_if_selected(target_item: QTreeWidgetItem):
            # Only include if Checked AND Enabled (Double check security)
            if (target_item.checkState(0) == Qt.CheckState.Checked and 
                (target_item.flags() & Qt.ItemFlag.ItemIsEnabled)):
                targets.append(target_item.data(0, Qt.ItemDataRole.UserRole))

        for dataset_type, data_tree in self._item_trees.items():
            root = data_tree.invisibleRootItem()
            if dataset_type == DatasetType.VIDEO_COLLECTION:
                # Children are selected in video collections
                for i in range(root.childCount()):
                    video_dataset_item = root.child(i)
                    for j in range(video_dataset_item.childCount()):
                        _append_target_if_selected(video_dataset_item.child(j))
            else:
                # Parents are selected in all other dataset types
                for i in range(root.childCount()):
                    _append_target_if_selected(root.child(i))
        return targets