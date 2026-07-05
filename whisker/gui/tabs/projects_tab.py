import logging
from typing import List

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QHBoxLayout,
    QStackedWidget,
    QMessageBox,
    # --- New Imports ---
    QGroupBox,
    QTreeWidget,
    QTreeWidgetItem,
)

from whisker.core.workspace import Project
from whisker.gui.widgets import ProjectSettingsWidget
from whisker.gui.widgets.template_skeleton_widget import TemplateSkeletonWidget
from whisker.gui.signals import MessageBus
from .base_tab import BaseTab


class ProjectsTab(BaseTab):
    _PLACEHOLDER_MESSAGE = (
        "Select an active project from the dropdown menu above to see its settings."
    )
    _DASHBOARD_PLACEHOLDER_MESSAGE = (
        "Select an active project to see its related models and predictions."
    )

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._init_ui()
        self._connect_signals()

        bus = MessageBus.get()
        bus.subscribe("workspace/models/refreshed", lambda t, p: self._update_dashboard(self._project))
        bus.subscribe("workspace/predictions/refreshed", lambda t, p: self._update_dashboard(self._project))

    def _init_ui(self):
        self.main_stack = QStackedWidget(self)

        self.main_view = QWidget()
        main_layout = QHBoxLayout(self.main_view)

        # --- Left Panel (New Dashboard) ---
        self.dashboard_panel = self._create_dashboard_panel()
        main_layout.addWidget(self.dashboard_panel, stretch=2)

        # --- Right Panel (Settings) ---
        self.view_stack = QStackedWidget()

        # 0: Placeholder
        self.placeholder_widget = QLabel(self._PLACEHOLDER_MESSAGE)
        self.placeholder_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder_widget.setWordWrap(True)
        self.view_stack.addWidget(self.placeholder_widget)
        self.view_stack.setCurrentWidget(self.placeholder_widget)

        # 1: Project Settings
        self.project_settings_widget = ProjectSettingsWidget()
        self.view_stack.addWidget(self.project_settings_widget)

        main_layout.addWidget(self.view_stack, stretch=1)
        self.main_stack.addWidget(self.main_view)

        tab_layout = QVBoxLayout(self)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.addWidget(self.main_stack)

    def _create_dashboard_panel(self) -> QWidget:
        """Creates the new left-hand panel for a project asset dashboard."""
        panel = QWidget()
        main_layout = QVBoxLayout(panel)
        main_layout.setContentsMargins(10, 10, 10, 10)

        self.dashboard_stack = QStackedWidget()
        main_layout.addWidget(self.dashboard_stack)

        # Page 0: Placeholder
        self.dashboard_placeholder = QLabel(self._DASHBOARD_PLACEHOLDER_MESSAGE)
        self.dashboard_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.dashboard_placeholder.setWordWrap(True)
        self.dashboard_stack.addWidget(self.dashboard_placeholder)

        # Page 1: Content
        self.dashboard_content_widget = QWidget()
        content_layout = QVBoxLayout(self.dashboard_content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)

        # --- Pose Models ---
        pose_models_group = QGroupBox("Pose Models (Training Runs)")
        pose_models_layout = QVBoxLayout(pose_models_group)
        self.pose_models_tree = QTreeWidget()
        self.pose_models_tree.setHeaderHidden(True)
        pose_models_layout.addWidget(self.pose_models_tree)
        content_layout.addWidget(pose_models_group)

        # --- Behavior Models ---
        behavior_models_group = QGroupBox("Behavior Models (Training Runs)")
        behavior_models_layout = QVBoxLayout(behavior_models_group)
        self.behavior_models_tree = QTreeWidget()
        self.behavior_models_tree.setHeaderHidden(True)
        behavior_models_layout.addWidget(self.behavior_models_tree)
        content_layout.addWidget(behavior_models_group)

        # --- Pose Predictions ---
        pose_preds_group = QGroupBox("Pose Predictions")
        pose_preds_layout = QVBoxLayout(pose_preds_group)
        self.pose_predictions_tree = QTreeWidget()
        self.pose_predictions_tree.setHeaderHidden(True)
        pose_preds_layout.addWidget(self.pose_predictions_tree)
        content_layout.addWidget(pose_preds_group)

        # --- Behavior Predictions ---
        behavior_preds_group = QGroupBox("Behavior Predictions")
        behavior_preds_layout = QVBoxLayout(behavior_preds_group)
        self.behavior_predictions_tree = QTreeWidget()
        self.behavior_predictions_tree.setHeaderHidden(True)
        behavior_preds_layout.addWidget(self.behavior_predictions_tree)
        content_layout.addWidget(behavior_preds_group)

        self.dashboard_stack.addWidget(self.dashboard_content_widget)
        self.dashboard_stack.setCurrentWidget(self.dashboard_placeholder)

        return panel

    def _connect_signals(self):
        # Behavior signals
        self.project_settings_widget.behavior_added.connect(self._on_behavior_added)
        self.project_settings_widget.behavior_renamed.connect(self._on_behavior_renamed)
        self.project_settings_widget.behavior_removed.connect(self._on_behavior_removed)

        # Body Part signals
        self.project_settings_widget.body_part_added.connect(self._on_body_part_added)
        self.project_settings_widget.body_part_renamed.connect(
            self._on_body_part_renamed
        )
        self.project_settings_widget.body_part_removed.connect(
            self._on_body_part_removed
        )

        # Identity signals
        self.project_settings_widget.identity_added.connect(self._on_identity_added)
        self.project_settings_widget.identity_renamed.connect(
            self._on_identity_renamed
        )
        self.project_settings_widget.identity_removed.connect(
            self._on_identity_removed
        )

        # --- NEW CONNECTIONS ---
        # Skeleton signals
        self.project_settings_widget.skeleton_edge_added.connect(
            self._on_skeleton_edge_added
        )
        self.project_settings_widget.skeleton_edge_removed.connect(
            self._on_skeleton_edge_removed
        )
        # --- END NEW CONNECTIONS ---

        self.project_settings_widget.template_updated.connect(
            self._on_template_updated
        )
        self.project_settings_widget.define_template_requested.connect(
            self._show_template_editor
        )

    def set_project(self, project: Project | None):
        super().set_project(project)
        if hasattr(self, 'main_stack'):
            self.main_stack.setCurrentIndex(0)

        if project:
            self.project_settings_widget.set_project_context(project)
            self.view_stack.setCurrentIndex(1)
        else:
            self.view_stack.setCurrentIndex(0)
        
        # --- New call to update the dashboard ---
        self._update_dashboard(project)

    def _show_template_editor(self):
        project = self._project
        if not project:
            return

        self.template_widget = TemplateSkeletonWidget(
            body_parts=project.body_parts,
            skeleton=project.skeleton,
            initial_coords=project.template_coords,
            initial_axis=project.heading_axis,
            parent=self,
        )

        self.template_widget.accepted.connect(self._on_template_widget_accepted)
        self.template_widget.rejected.connect(self._on_template_widget_rejected)

        # Remove existing widget at index 1 if present
        if self.main_stack.count() > 1:
            old_widget = self.main_stack.widget(1)
            self.main_stack.removeWidget(old_widget)
            old_widget.deleteLater()

        self.main_stack.addWidget(self.template_widget)
        self.main_stack.setCurrentIndex(1)

    def _on_template_widget_accepted(self):
        if not self._project or not hasattr(self, 'template_widget'):
            return

        new_coords, new_axis = self.template_widget.get_result()

        # Update project immediately
        self._project.template_coords = new_coords
        self._project.heading_axis = new_axis

        # Call the existing save/update handler
        self._on_template_updated(self._project.name, new_coords, new_axis)

        # Return to main view
        self.main_stack.setCurrentIndex(0)

    def _on_template_widget_rejected(self):
        # Return to main view without saving
        self.main_stack.setCurrentIndex(0)

    def _update_dashboard(self, project: Project | None):
        """Populates the dashboard trees based on the active project."""
        # Clear all trees first
        self.pose_models_tree.clear()
        self.behavior_models_tree.clear()
        self.pose_predictions_tree.clear()
        self.behavior_predictions_tree.clear()

        if not project or not self._workspace:
            self.dashboard_stack.setCurrentWidget(self.dashboard_placeholder)
            return

        self.dashboard_stack.setCurrentWidget(self.dashboard_content_widget)
        project_name = project.name

        # --- 1. Populate Pose Models (from workspace/pose_models/) ---
        pose_models_path = self._workspace.pose_models.base_dir
        if pose_models_path.is_dir():
            for run_dir in sorted(pose_models_path.iterdir()):
                if run_dir.is_dir() and run_dir.name.startswith(project_name):
                    item = QTreeWidgetItem([run_dir.name])
                    self.pose_models_tree.addTopLevelItem(item)

        # --- 2. Populate Behavior Models (from workspace/behavior_models/) ---
        behavior_models_path = self._workspace.behavior_models.base_dir
        if behavior_models_path.is_dir():
            for run_dir in sorted(behavior_models_path.iterdir()):
                if run_dir.is_dir() and run_dir.name.startswith(project_name):
                    item = QTreeWidgetItem([run_dir.name])
                    self.behavior_models_tree.addTopLevelItem(item)

        # --- 3. Populate Pose Predictions (from memory) ---
        for run_name, datasets in sorted(self._workspace.pose_predictions.items()):
            if run_name.startswith(project_name):
                run_item = QTreeWidgetItem([run_name])
                self.pose_predictions_tree.addTopLevelItem(run_item)
                for dataset_name in sorted(datasets.keys()):
                    dataset_item = QTreeWidgetItem([dataset_name])
                    run_item.addChild(dataset_item)
        self.pose_predictions_tree.expandAll()

        # --- 4. Populate Behavior Predictions (from workspace/behavior_predictions/) ---
        behavior_preds_path = self._workspace.behavior_predictions.base_dir
        if behavior_preds_path.is_dir():
            for run_dir in sorted(behavior_preds_path.iterdir()):
                if run_dir.is_dir() and run_dir.name.startswith(project_name):
                    run_item = QTreeWidgetItem([run_dir.name])
                    self.behavior_predictions_tree.addTopLevelItem(run_item)
                    # These runs contain dataset subdirectories
                    for dataset_dir in sorted(run_dir.iterdir()):
                        if dataset_dir.is_dir():
                            dataset_item = QTreeWidgetItem([dataset_dir.name])
                            run_item.addChild(dataset_item)
        self.behavior_predictions_tree.expandAll()

    # --- Generic Handlers (to avoid repetition) ---

    def _get_project(self, project_name: str) -> Project | None:
        if not self._workspace:
            return None
        project = self._workspace.projects.get(project_name)
        if not project:
            logging.error(f"Project '{project_name}' not found in workspace.")
        return project

    def _handle_project_list_add(
        self,
        project_name: str,
        item_name: str,
        project_list: List[str],
        item_type_name: str,
    ):
        """Generic logic to add an item to a project list."""
        project = self._get_project(project_name)
        if not project:
            return

        if item_name not in project_list:
            project_list.append(item_name)
            self._workspace.save_project(project_name)
            self.project_settings_widget.set_project_context(project)
        else:
            QMessageBox.warning(
                self,
                "Duplicate",
                f"{item_type_name} '{item_name}' already exists.",
            )

    def _handle_project_list_rename(
        self,
        project_name: str,
        old_name: str,
        new_name: str,
        project_list: List[str],
        item_type_name: str,
    ):
        """Generic logic to rename an item in a project list with a warning."""
        project = self._get_project(project_name)
        if not project:
            return

        # Show a warning that this does not cascade to data
        reply = QMessageBox.warning(
            self,
            "Confirm Rename",
            f"Are you sure you want to rename '{old_name}' to '{new_name}'?\n\n"
            f"This action will NOT update existing pose labels. "
            f"It can lead to data inconsistency if '{old_name}' is already in use.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.No:
            return

        try:
            index = project_list.index(old_name)
            project_list[index] = new_name
            project_list.sort()
            self._workspace.save_project(project_name)
            self.project_settings_widget.set_project_context(project)
        except ValueError:
            logging.error(
                f"{item_type_name} '{old_name}' not found in project list."
            )

    def _handle_project_list_remove(
        self,
        project_name: str,
        item_name: str,
        project_list: List[str],
        item_type_name: str,
    ):
        """Generic logic to remove an item from a project list with a warning."""
        project = self._get_project(project_name)
        if not project:
            return

        # Show a strong warning
        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to remove the {item_type_name.lower()} '{item_name}'?\n\n"
            f"This action will NOT remove annotations from existing pose labels. "
            f"It can lead to data inconsistency. This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.No:
            return

        try:
            project_list.remove(item_name)
            self._workspace.save_project(project_name)
            self.project_settings_widget.set_project_context(project)
        except ValueError:
            logging.error(
                f"Cannot remove: {item_type_name} '{item_name}' not found in project."
            )

    # --- Body Part Slots ---

    def _on_body_part_added(self, project_name: str, bp_name: str):
        project = self._get_project(project_name)
        if project:
            self._handle_project_list_add(
                project_name, bp_name, project.body_parts, "Body Part"
            )

    def _on_body_part_renamed(
        self, project_name: str, old_name: str, new_name: str
    ):
        project = self._get_project(project_name)
        if project:
            self._handle_project_list_rename(
                project_name, old_name, new_name, project.body_parts, "Body Part"
            )

    def _on_body_part_removed(self, project_name: str, bp_name: str):
        project = self._get_project(project_name)
        if project:
            self._handle_project_list_remove(
                project_name, bp_name, project.body_parts, "Body Part"
            )

    # --- Identity Slots ---

    def _on_identity_added(self, project_name: str, id_name: str):
        project = self._get_project(project_name)
        if project:
            self._handle_project_list_add(
                project_name, id_name, project.identities, "Identity"
            )

    def _on_identity_renamed(
        self, project_name: str, old_name: str, new_name: str
    ):
        project = self._get_project(project_name)
        if project:
            self._handle_project_list_rename(
                project_name, old_name, new_name, project.identities, "Identity"
            )

    def _on_identity_removed(self, project_name: str, id_name: str):
        project = self._get_project(project_name)
        if project:
            self._handle_project_list_remove(
                project_name, id_name, project.identities, "Identity"
            )

    # --- Behavior Slots (Existing) ---

    def _on_behavior_added(self, project_name: str, behavior_name: str):
        # This one uses the generic helper
        project = self._get_project(project_name)
        if project:
            self._handle_project_list_add(
                project_name, behavior_name, project.behaviors, "Behavior"
            )

    def _on_behavior_renamed(self, project_name: str, old_name: str, new_name: str):
        # This one has custom logic, so it's NOT generic
        project = self._get_project(project_name)
        if not project:
            return

        # 1. Update the project's behavior list
        try:
            index = project.behaviors.index(old_name)
            project.behaviors[index] = new_name
            project.behaviors.sort()
        except ValueError:
            logging.error(f"Behavior '{old_name}' not found in project.")
            return

        self._workspace.save_project(project_name)

        # 2. Update all behavior label files
        for dataset_name in self._workspace.datasets.keys():
            labels = self._workspace.get_behavior_labels(dataset_name)
            # Find all rows with the old behavior name and update them
            rows_to_update = labels.bouts["behavior"] == old_name
            if rows_to_update.any():
                labels.bouts.loc[rows_to_update, "behavior"] = new_name
                self._workspace.save_behavior_labels(dataset_name)
                logging.info(
                    f"Updated behavior '{old_name}' to '{new_name}' in dataset '{dataset_name}'."
                )

        # 3. Refresh the UI
        self.project_settings_widget.set_project_context(project)

    def _on_behavior_removed(self, project_name: str, behavior_name: str):
        # This one also has custom logic
        project = self._get_project(project_name)
        if not project:
            return

        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to remove the behavior '{behavior_name}'?\n"
            "This will remove the behavior from the project and delete all "
            "associated annotations from all datasets. This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.No:
            return

        # 1. Update the project's behavior list
        try:
            project.behaviors.remove(behavior_name)
        except ValueError:
            logging.error(
                f"Cannot remove: Behavior '{behavior_name}' not found in project."
            )
            return

        self._workspace.save_project(project_name)

        # 2. Remove associated annotations from all label files
        for dataset_name in self._workspace.datasets.keys():
            labels = self._workspace.get_behavior_labels(dataset_name)
            
            original_count = len(labels.bouts)
            # Keep all rows where the behavior is NOT the one to be removed
            labels.bouts = labels.bouts[labels.bouts["behavior"] != behavior_name]
            
            if len(labels.bouts) < original_count:
                self._workspace.save_behavior_labels(dataset_name)
                logging.info(
                    f"Removed {original_count - len(labels.bouts)} annotations for "
                    f"'{behavior_name}' in dataset '{dataset_name}'."
                )

        # 3. Refresh the UI
        self.project_settings_widget.set_project_context(project)

        QMessageBox.information(
            self,
            "Remove Successful",
            f"Removed behavior '{behavior_name}' from the project.",
        )

    def _on_skeleton_edge_added(self, project_name: str, bp1: str, bp2: str):
        project = self._get_project(project_name)
        if not project:
            return

        new_edge = (bp1, bp2)
        # Double-check for duplicates (e.g., (bp2, bp1))
        if new_edge not in project.skeleton and (bp2, bp1) not in project.skeleton:
            project.skeleton.append(new_edge)
            self._workspace.save_project(project_name)
            # Refresh the UI
            self.project_settings_widget.set_project_context(project)
        else:
            QMessageBox.warning(self, "Duplicate", "That skeleton edge already exists.")

    def _on_skeleton_edge_removed(self, project_name: str, edge_string: str):
        project = self._get_project(project_name)
        if not project:
            return

        try:
            bp1, bp2 = edge_string.split(" → ")
            edge1 = (bp1, bp2)
            edge2 = (bp2, bp1)

            if edge1 in project.skeleton:
                project.skeleton.remove(edge1)
            elif edge2 in project.skeleton:
                project.skeleton.remove(edge2)
            else:
                raise ValueError(
                    f"Edge '{edge_string}' not found in project.skeleton list."
                )

            self._workspace.save_project(project_name)
            # Refresh the UI
            self.project_settings_widget.set_project_context(project)
        except Exception as e:
            logging.error(f"Error removing skeleton edge '{edge_string}': {e}")
            QMessageBox.critical(self, "Error", f"Could not remove edge: {e}")

    # --- Removed the old static image method ---

    def _on_template_updated(self, project_name: str, coords: dict, axis: tuple):
        """
        Updates the project's template skeleton and saves it to disk.
        """
        project = self._get_project(project_name)
        if not project:
            return

        # Update the project object
        project.template_coords = coords
        project.heading_axis = axis
        
        # Save to JSON
        try:
            self._workspace.save_project(project_name)
            logging.info(f"Saved template skeleton for project '{project_name}'.")
            
            # Refresh the settings widget to ensure it reflects the saved state
            self.project_settings_widget.set_project_context(project)
            
        except Exception as e:
            logging.error(f"Failed to save template for project '{project_name}': {e}")
            QMessageBox.critical(self, "Save Error", f"Could not save template:\n{e}")