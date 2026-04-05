from typing import Dict, Optional, Tuple
import pandas as pd

from PyQt6.QtCore import Qt, QPoint, pyqtSignal
from PyQt6.QtGui import QColor, QAction
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QCheckBox, QSpinBox, QGroupBox, QTreeWidget,
    QTreeWidgetItem, QTreeWidgetItemIterator, QMenu
)

from whisker.core.project import Project
from whisker.gui.constants import CHECKMARK_INDICATOR, KEYPOINT_QCOLORS

class PoseLabelingSideControlsWidget(QWidget):
    font_size_changed = pyqtSignal(int)
    drag_mode_toggled = pyqtSignal(bool)
    show_names_toggled = pyqtSignal(bool)
    selection_changed = pyqtSignal()
    clear_keypoint_requested = pyqtSignal(str, int)
    clear_all_keypoint_requested = pyqtSignal()
    swap_identities_requested = pyqtSignal(str, str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._colors_by_identity_id: Dict[str, QColor] = {}
        self._project_identities: list[str] = []
        self._init_ui()
        self._connect_internal_signals()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        self.drag_mode_checkbox = QCheckBox("Drag Keypoints (Right-Click)")
        layout.addWidget(self.drag_mode_checkbox)

        self.autocycle_checkbox = QCheckBox("Auto-cycle to Next Keypoint (`Ctrl+C`)")
        self.autocycle_checkbox.setChecked(True)
        layout.addWidget(self.autocycle_checkbox)

        self.show_names_checkbox = QCheckBox("Show Keypoint Names (`Ctrl+N`)")
        self.show_names_checkbox.setChecked(True)
        layout.addWidget(self.show_names_checkbox)

        font_size_layout = QHBoxLayout()
        self.font_size_label = QLabel("Font Size:")
        self.font_size_spinbox = QSpinBox()
        self.font_size_spinbox.setRange(6, 24)
        self.font_size_spinbox.setValue(8)
        font_size_layout.addWidget(self.font_size_label)
        font_size_layout.addWidget(self.font_size_spinbox)
        layout.addLayout(font_size_layout)

        group = QGroupBox("Labeling Targets")
        group_layout = QVBoxLayout(group)
        self.labeling_tree = QTreeWidget()
        self.labeling_tree.setHeaderHidden(True)
        self.labeling_tree.setToolTip(
            "W/S: Previous/Next Body Part\n"
            "A/D: Previous/Next Identity\n"
            "Delete/X: Clear selected keypoint"
        )
        self.labeling_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        group_layout.addWidget(self.labeling_tree)

        self.clear_keypoint_button = QPushButton("Clear Selected (`Del`)")
        group_layout.addWidget(self.clear_keypoint_button)
        self.clear_all_button = QPushButton("Clear All (`Ctrl+Del`)")
        group_layout.addWidget(self.clear_all_button)
        layout.addWidget(group, 1)

    def _connect_internal_signals(self):
        self.drag_mode_checkbox.toggled.connect(self._on_drag_toggled_internal)
        self.show_names_checkbox.toggled.connect(self._on_show_names_internal)
        self.font_size_spinbox.valueChanged.connect(self.font_size_changed.emit)
        self.labeling_tree.currentItemChanged.connect(self.selection_changed.emit)
        self.labeling_tree.customContextMenuRequested.connect(self._show_context_menu)
        self.clear_keypoint_button.clicked.connect(self.request_clear_current)
        self.clear_all_button.clicked.connect(self.clear_all_keypoint_requested.emit)

    def set_enabled_state(self, enabled: bool):
        self.setEnabled(enabled)

    def get_current_selection(self) -> Tuple[Optional[str], Optional[int]]:
        current = self.labeling_tree.currentItem()
        if current and current.parent():
            return current.data(0, Qt.ItemDataRole.UserRole)
        return None, None

    def is_autocycle_enabled(self) -> bool:
        return self.autocycle_checkbox.isChecked()
    
    def toggle_checkbox(self, checkbox_name: str):
        mapping = {
            'drag': self.drag_mode_checkbox,
            'names': self.show_names_checkbox,
            'auto_cycle': self.autocycle_checkbox
        }
        if cb := mapping.get(checkbox_name):
            cb.toggle()
    
    def set_checkbox(self, checkbox_name: str, value: bool):
        mapping = {
            'drag': self.drag_mode_checkbox,
            'names': self.show_names_checkbox,
            'auto_cycle': self.autocycle_checkbox
        }
        if cb := mapping.get(checkbox_name):
            cb.setChecked(value)

    def update_tree(self, project: Project | None, image_df: pd.DataFrame):
        self.labeling_tree.blockSignals(True)
        self.labeling_tree.clear()
        self._colors_by_identity_id.clear()
        if project is None:
            return

        self._project_identities = project.identities

        if image_df is None:
            self.labeling_tree.blockSignals(False)
            return

        for i, identity in enumerate(project.identities):
            color = KEYPOINT_QCOLORS[i % len(KEYPOINT_QCOLORS)]
            self._colors_by_identity_id[identity] = color

            identity_item = QTreeWidgetItem(self.labeling_tree, [identity])
            identity_item.setForeground(0, color)
            font = identity_item.font(0)
            font.setBold(True)
            identity_item.setFont(0, font)
            identity_item.setFlags(identity_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)

            for part_idx, part_name in enumerate(project.body_parts):
                is_labeled = False
                try:
                    is_labeled = pd.notna(image_df.loc[(identity, part_name), 'c'])
                except KeyError:
                    pass

                display_text = f"{part_name} {CHECKMARK_INDICATOR}" if is_labeled else part_name
                body_part_item = QTreeWidgetItem(identity_item, [display_text])
                body_part_item.setData(0, Qt.ItemDataRole.UserRole, (identity, part_idx))

        self.labeling_tree.expandAll()
        
        if self.labeling_tree.topLevelItemCount() > 0:
            top = self.labeling_tree.topLevelItem(0)
            if top.childCount() > 0 and not self.labeling_tree.currentItem():
                self.labeling_tree.setCurrentItem(top.child(0))
        
        self.labeling_tree.blockSignals(False)

    def restore_selection(self, identity_id: str, part_idx: int):
        iterator = QTreeWidgetItemIterator(self.labeling_tree)
        while iterator.value():
            item = iterator.value()
            if item.data(0, Qt.ItemDataRole.UserRole) == (identity_id, part_idx):
                self.labeling_tree.setCurrentItem(item)
                return
            iterator += 1

    def cycle_selection(self, direction: str):
        curr = self.labeling_tree.currentItem()
        if not curr or not curr.parent(): return

        target = None
        if direction == 'next':
            target = self.labeling_tree.itemBelow(curr)
        elif direction == 'prev':
            target = self.labeling_tree.itemAbove(curr)
        elif direction in ('next_id', 'prev_id'):
            parent = curr.parent()
            adj_parent = (self.labeling_tree.itemBelow(parent) if direction == 'next_id' 
                          else self.labeling_tree.itemAbove(parent))
            if adj_parent:
                idx = parent.indexOfChild(curr)
                if idx < adj_parent.childCount():
                    target = adj_parent.child(idx)
        
        if target:
            if not target.parent() and direction == 'next':
                 target = target.child(0) if target.childCount() > 0 else None
            if target:
                self.labeling_tree.setCurrentItem(target)

    def cycle_to_next_auto(self):
        curr = self.labeling_tree.currentItem()
        next_item = self.labeling_tree.itemBelow(curr)
        if next_item and not next_item.data(0, Qt.ItemDataRole.UserRole):
            next_item = next_item.child(0) if next_item.childCount() > 0 else None
        
        if not next_item:
            first_id = self.labeling_tree.topLevelItem(0)
            if first_id and first_id.childCount() > 0:
                next_item = first_id.child(0)
        
        if next_item:
            self.labeling_tree.setCurrentItem(next_item)

    def get_color_for_identity(self, identity: str) -> QColor:
        return self._colors_by_identity_id.get(identity, QColor("white"))

    def clear_selection(self):
        self.labeling_tree.clearSelection()
        self.labeling_tree.setCurrentItem(None)

    def request_clear_current(self):
        identity_id, part_idx = self.get_current_selection()
        if identity_id is not None:
            self.clear_keypoint_requested.emit(identity_id, part_idx)

    def _on_drag_toggled_internal(self, checked: bool):
        self.autocycle_checkbox.setEnabled(not checked)
        self.labeling_tree.setEnabled(not checked)
        if checked:
            self.labeling_tree.clearSelection()
        self.drag_mode_toggled.emit(checked)

    def _on_show_names_internal(self, checked: bool):
        self.font_size_label.setEnabled(checked)
        self.font_size_spinbox.setEnabled(checked)
        self.show_names_toggled.emit(checked)

    def _show_context_menu(self, position: QPoint):
        item = self.labeling_tree.itemAt(position)
        if not item: return

        identity_item = item.parent() if item.parent() else item
        selected_identity = identity_item.text(0)

        menu = QMenu()
        swap_menu = menu.addMenu("Swap Identities >")
        added = False
        
        for identity in self._project_identities:
            if identity == selected_identity: continue
            action = QAction(identity, self)
            action.triggered.connect(
                lambda chk, o=identity: self.swap_identities_requested.emit(selected_identity, o)
            )
            swap_menu.addAction(action)
            added = True

        if added:
            menu.exec(self.labeling_tree.viewport().mapToGlobal(position))
