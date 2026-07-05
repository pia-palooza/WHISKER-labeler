from typing import List, Tuple
import logging

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QListWidget,
    QGroupBox,
    QComboBox,
    QMessageBox,
)

from whisker.core.workspace import Project

class ProjectSettingsWidget(QWidget):
    behavior_added = pyqtSignal(str, str)
    behavior_renamed = pyqtSignal(str, str, str)
    behavior_removed = pyqtSignal(str, str)
    body_part_added = pyqtSignal(str, str)
    body_part_renamed = pyqtSignal(str, str, str)
    body_part_removed = pyqtSignal(str, str)
    identity_added = pyqtSignal(str, str)
    identity_renamed = pyqtSignal(str, str, str)
    identity_removed = pyqtSignal(str, str)
    skeleton_edge_added = pyqtSignal(str, str, str)
    skeleton_edge_removed = pyqtSignal(str, str)
    template_updated = pyqtSignal(str, dict, tuple)
    define_template_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._project_name: str | None = None

        main_layout = QVBoxLayout(self)

        # Create and add the manager UIs
        (
            self.bp_group,
            self.bp_list,
            self.bp_input,
            self.bp_add,
            self.bp_rename,
            self.bp_remove,
        ) = self._create_list_manager_group("Body Parts")
        main_layout.addWidget(self.bp_group)

        (
            self.sk_group,
            self.sk_list,
            self.sk_combo1,
            self.sk_combo2,
            self.sk_add,
            self.sk_remove,
        ) = self._create_skeleton_manager_group("Skeleton")
        main_layout.addWidget(self.sk_group)

        (
            self.id_group,
            self.id_list,
            self.id_input,
            self.id_add,
            self.id_rename,
            self.id_remove,
        ) = self._create_list_manager_group("Identities")
        main_layout.addWidget(self.id_group)

        (
            self.bh_group,
            self.bh_list,
            self.bh_input,
            self.bh_add,
            self.bh_rename,
            self.bh_remove,
        ) = self._create_list_manager_group("Behaviors")
        main_layout.addWidget(self.bh_group)

        self.tmpl_group = QGroupBox("Template Skeleton & Feature Engineering")
        tmpl_layout = QHBoxLayout(self.tmpl_group)
        
        self.tmpl_info_label = QLabel("No template defined.")
        self.tmpl_btn = QPushButton("Define Template...")
        self.tmpl_btn.clicked.connect(self._on_define_template)
        
        tmpl_layout.addWidget(self.tmpl_info_label)
        tmpl_layout.addStretch()
        tmpl_layout.addWidget(self.tmpl_btn)
        
        main_layout.addWidget(self.tmpl_group)

        # --- CONNECT SIGNALS ---
        # Connect signals for the behavior manager
        self.bh_add.clicked.connect(self._on_add_behavior)
        self.bh_rename.clicked.connect(self._on_rename_behavior)
        self.bh_remove.clicked.connect(self._on_remove_behavior)

        # Connect signals for the body part manager
        self.bp_add.clicked.connect(self._on_add_body_part)
        self.bp_rename.clicked.connect(self._on_rename_body_part)
        self.bp_remove.clicked.connect(self._on_remove_body_part)

        # Connect signals for the identity manager
        self.id_add.clicked.connect(self._on_add_identity)
        self.id_rename.clicked.connect(self._on_rename_identity)
        self.id_remove.clicked.connect(self._on_remove_identity)

        # Connect signals for skeleton manager
        self.sk_add.clicked.connect(self._on_add_skeleton_edge)
        self.sk_remove.clicked.connect(self._on_remove_skeleton_edge)
        # --- END CONNECT SIGNALS ---

    def set_project_context(self, project: Project):
        """
        Sets the current project context and populates the UI.
        This should be called whenever the selected project changes.
        """
        self._project_name = project.name
        self.populate(project)

    def populate(self, project: Project):
        """Loads data into all list widgets from a Project object."""
        self._current_project_obj = project # Cache object for dialog access
        self.bh_list.clear()
        self.bh_list.addItems(project.behaviors)

        body_parts = self._load_list(self.bp_list, project.body_parts)
        self._load_skeleton(project.skeleton, body_parts)
        self._load_list(self.id_list, project.identities)

        if project.heading_axis:
            start, end = project.heading_axis
            self.tmpl_info_label.setText(f"Heading Axis: <b>{start} → {end}</b>")
        else:
            self.tmpl_info_label.setText("No template defined (Standard features only).")


    def _create_list_manager_group(self, title: str):
        """Factory method to create a standardized list management groupbox."""
        group = QGroupBox(title)
        layout = QVBoxLayout(group)

        list_widget = QListWidget()
        layout.addWidget(list_widget)

        edit_layout = QHBoxLayout()
        input_field = QLineEdit()
        input_field.setPlaceholderText("Enter name and click action...")
        add_btn = QPushButton("Add")
        rename_btn = QPushButton("Rename")
        remove_btn = QPushButton("Remove")

        edit_layout.addWidget(input_field)
        edit_layout.addWidget(add_btn)
        edit_layout.addWidget(rename_btn)
        edit_layout.addWidget(remove_btn)
        layout.addLayout(edit_layout)

        return group, list_widget, input_field, add_btn, rename_btn, remove_btn

    def _create_skeleton_manager_group(self, title: str):
        """Factory method to create the skeleton management groupbox."""
        group = QGroupBox(title)
        layout = QVBoxLayout(group)

        list_widget = QListWidget()
        list_widget.setToolTip("List of connections between body parts.")
        layout.addWidget(list_widget)

        add_layout = QHBoxLayout()
        combo1 = QComboBox()
        combo2 = QComboBox()
        add_btn = QPushButton("Add")
        remove_btn = QPushButton("Remove")

        add_layout.addWidget(combo1)
        add_layout.addWidget(QLabel("→"))
        add_layout.addWidget(combo2)
        add_layout.addStretch()
        add_layout.addWidget(add_btn)
        add_layout.addWidget(remove_btn)
        layout.addLayout(add_layout)

        return group, list_widget, combo1, combo2, add_btn, remove_btn

    def _load_list(self, list_widget: QListWidget, items: List[str]):
        """Generic method to load items into a list widget."""
        list_widget.clear()
        if items:
            list_widget.addItems(items)
        return items

    def _update_skeleton_combos(self, body_parts: list[str]):
        """Populates the skeleton comboboxes with current body parts."""
        self.sk_combo1.clear()
        self.sk_combo2.clear()
        self.sk_combo1.addItems(body_parts)
        self.sk_combo2.addItems(body_parts)

    def _load_skeleton(self, skeleton: List[Tuple[str, str]], body_parts: list[str]):
        """Loads the skeleton and populates the UI."""
        self._update_skeleton_combos(body_parts)
        self.sk_list.clear()
        try:
            for part1, part2 in skeleton:
                self.sk_list.addItem(f"{part1} → {part2}")
        except Exception as e:
            logging.warning(f"Failed to load skeleton: {e}")

    # --- Generic helper for validation ---
    def _validate_and_emit_add(
        self,
        list_widget: QListWidget,
        input_field: QLineEdit,
        signal: pyqtSignal,
        item_type: str,
    ):
        if not self._project_name:
            return
        new_name = input_field.text().strip()
        if not new_name:
            QMessageBox.warning(
                self, "Input Error", f"{item_type} name cannot be empty."
            )
            return

        existing = [
            list_widget.item(i).text() for i in range(list_widget.count())
        ]
        if new_name in existing:
            QMessageBox.warning(
                self, "Input Error", f"{item_type} '{new_name}' already exists."
            )
            return

        signal.emit(self._project_name, new_name)
        input_field.clear()

    def _validate_and_emit_rename(
        self,
        list_widget: QListWidget,
        input_field: QLineEdit,
        signal: pyqtSignal,
        item_type: str,
    ):
        if not self._project_name:
            return

        selected_items = list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(
                self, "Selection Error", f"Please select a {item_type} to rename."
            )
            return

        old_name = selected_items[0].text()
        new_name = input_field.text().strip()

        if not new_name:
            QMessageBox.warning(
                self, "Input Error", f"New {item_type} name cannot be empty."
            )
            return

        if old_name == new_name:
            return  # No change

        existing = [
            list_widget.item(i).text() for i in range(list_widget.count())
        ]
        if new_name in existing:
            QMessageBox.warning(
                self, "Input Error", f"{item_type} '{new_name}' already exists."
            )
            return

        signal.emit(self._project_name, old_name, new_name)
        input_field.clear()

    def _validate_and_emit_remove(
        self, list_widget: QListWidget, signal: pyqtSignal, item_type: str
    ):
        if not self._project_name:
            return
        selected_items = list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(
                self, "Selection Error", f"Please select a {item_type} to remove."
            )
            return

        name_to_remove = selected_items[0].text()
        signal.emit(self._project_name, name_to_remove)

    # --- Body Part Slots ---
    def _on_add_body_part(self):
        self._validate_and_emit_add(
            self.bp_list, self.bp_input, self.body_part_added, "Body Part"
        )

    def _on_rename_body_part(self):
        self._validate_and_emit_rename(
            self.bp_list, self.bp_input, self.body_part_renamed, "Body Part"
        )

    def _on_remove_body_part(self):
        self._validate_and_emit_remove(
            self.bp_list, self.body_part_removed, "Body Part"
        )

    # --- Identity Slots ---
    def _on_add_identity(self):
        self._validate_and_emit_add(
            self.id_list, self.id_input, self.identity_added, "Identity"
        )

    def _on_rename_identity(self):
        self._validate_and_emit_rename(
            self.id_list, self.id_input, self.identity_renamed, "Identity"
        )

    def _on_remove_identity(self):
        self._validate_and_emit_remove(
            self.id_list, self.identity_removed, "Identity"
        )

    # --- Behavior Slots (Existing) ---
    def _on_add_behavior(self):
        self._validate_and_emit_add(
            self.bh_list, self.bh_input, self.behavior_added, "Behavior"
        )

    def _on_rename_behavior(self):
        self._validate_and_emit_rename(
            self.bh_list, self.bh_input, self.behavior_renamed, "Behavior"
        )

    def _on_remove_behavior(self):
        self._validate_and_emit_remove(
            self.bh_list, self.behavior_removed, "Behavior"
        )

    # --- NEW SLOTS for Skeleton ---
    def _on_add_skeleton_edge(self):
        if not self._project_name:
            return
        bp1 = self.sk_combo1.currentText()
        bp2 = self.sk_combo2.currentText()
        if not bp1 or not bp2:
            QMessageBox.warning(
                self, "Selection Error", "Both body parts must be selected."
            )
            return
        if bp1 == bp2:
            QMessageBox.warning(
                self, "Selection Error", "Cannot connect a body part to itself."
            )
            return
        # Check for duplicates (bp1->bp2 or bp2->bp1)
        existing = [self.sk_list.item(i).text() for i in range(self.sk_list.count())]
        if f"{bp1} → {bp2}" in existing or f"{bp2} → {bp1}" in existing:
            QMessageBox.warning(self, "Duplicate", "That skeleton edge already exists.")
            return
        self.skeleton_edge_added.emit(self._project_name, bp1, bp2)

    def _on_remove_skeleton_edge(self):
        if not self._project_name:
            return
        selected_items = self.sk_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(
                self, "Selection Error", "Please select an edge to remove."
            )
            return
        edge_string = selected_items[0].text()
        self.skeleton_edge_removed.emit(self._project_name, edge_string)

    def _on_define_template(self):
        if not self._project_name or not hasattr(self, '_current_project_obj'):
            return
            
        project = self._current_project_obj
        
        if len(project.body_parts) < 2:
            QMessageBox.warning(self, "Not Enough Points", "Please define at least 2 body parts first.")
            return

        self.define_template_requested.emit()