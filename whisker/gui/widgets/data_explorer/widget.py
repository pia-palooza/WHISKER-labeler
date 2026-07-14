import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal, QThreadPool, QSignalBlocker
from PyQt6.QtWidgets import (
    QWidget,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QComboBox,
    QCheckBox,
    QLabel,
    QMenu,
)

from whisker.gui.signals import MessageBus
from whisker.gui.base.collapsible_panel import VerticalCollapsiblePanel
from whisker.core.workspace import Workspace, Project
from .constants import Selection, ItemGroupEnum, FilterOptions
from .tree_controller import TreeController
from .action_handler import ActionHandler
from .selection_logic import get_item_hierarchy, infer_selection_item_type

class Widget(VerticalCollapsiblePanel):
    item_selected = pyqtSignal(Selection)
    item_deselected = pyqtSignal()
    blind_mode_toggled = pyqtSignal(bool)

    def __init__(self, parent: QWidget | None = None):
        self.title_label = QLabel("DATA EXPLORER")
        self.title_label.setObjectName("HeaderLabel")

        super().__init__(self.title_label, parent=parent, drag_edges=None)
        self._workspace: Optional[Workspace] = None
        self.thread_pool = QThreadPool()

        # Minimize margins to match NavigationPanel
        self.content_layout.setContentsMargins(4, 4, 4, 4)

        # Options button in the expanded header (next to title)
        self.options_button = QPushButton("...")
        self.options_button.setStyleSheet("""
            QPushButton { 
                font-weight: bold; 
                text-align: center;
            }
            QPushButton::menu-indicator { 
                image: none; 
                width: 0px;
                padding: 0px;
            }
        """)
        self.options_button.setToolTip("Creation & Import Actions")
        self.expanded_header_layout.addWidget(self.options_button)

        # --- Options Menu ---
        self.options_menu = QMenu(self)
        self.create_project_action = self.options_menu.addAction("Create Project...")
        self.create_dataset_action = self.options_menu.addAction("Create Dataset...")
        self.import_labels_action = self.options_menu.addAction("Import Pose Labels...")
        self.import_bundle_action = self.options_menu.addAction(
            "Import Annotation Bundle..."
        )

        self.options_button.setMenu(self.options_menu)

        # Connect Menu Actions
        self.create_project_action.triggered.connect(self.show_create_project_dialog)
        self.create_dataset_action.triggered.connect(self.show_create_dataset_dialog)
        self.import_labels_action.triggered.connect(self.show_import_pose_labels_dialog)
        self.import_bundle_action.triggered.connect(self.show_import_bundle_dialog)

        # 1. Controls (Dropdown + Blind Mode)
        self.dropdown = QComboBox()
        self.dropdown.currentIndexChanged.connect(self._update_data_tree)
        self.content_layout.addWidget(self.dropdown)

        self.blind_mode_checkbox = QCheckBox("Blind Mode")
        self.blind_mode_checkbox.setToolTip("Anonymize dataset entries.")
        self.blind_mode_checkbox.toggled.connect(self._on_blind_mode_toggled)
        self.content_layout.addWidget(self.blind_mode_checkbox)

        # 2. Filter Dropdown
        self.filter_dropdown = QComboBox()
        self.filter_dropdown.setToolTip("Filter items.")
        self.filter_dropdown.addItem(FilterOptions.ALL.value, FilterOptions.ALL)
        self.filter_dropdown.addItem(FilterOptions.LABELED.value, FilterOptions.LABELED)
        self.filter_dropdown.addItem(
            FilterOptions.UNLABELED.value, FilterOptions.UNLABELED
        )
        self.filter_dropdown.setVisible(False)
        self.filter_dropdown.currentIndexChanged.connect(self._apply_filter)
        self.content_layout.addWidget(self.filter_dropdown)

        # 3. Tree Widget
        self.data_tree = QTreeWidget()
        self.data_tree.setHeaderHidden(True)
        self.data_tree.currentItemChanged.connect(self._on_item_changed)
        self.data_tree.itemExpanded.connect(self._on_item_expanded)
        self.data_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.data_tree.customContextMenuRequested.connect(self._on_context_menu)
        self.content_layout.addWidget(self.data_tree)

        # Setup Components
        self.controller = TreeController(self.data_tree)
        self.action_handler = ActionHandler(self, self.thread_pool)

        # Message Bus Subscriptions
        bus = MessageBus.get()
        bus.subscribe("workspace/datasets/refreshed", lambda t, p: self._on_global_refresh())
        bus.subscribe("workspace/labels/refreshed", lambda t, p: self._on_global_refresh())
        bus.subscribe("workspace/projects/refreshed", lambda t, p: self._on_global_refresh())
        bus.subscribe("workspace/predictions/refreshed", lambda t, p: self._update_data_tree())
        bus.subscribe("workspace/models/refreshed", lambda t, p: self._update_data_tree())
        bus.subscribe("workspace/files/refreshed", lambda t, p: self._update_data_tree())
        bus.subscribe("workspace/*/model/selected", self._on_bus_model_selected)
        
        # Subscribe to active model/prediction changes from the global toolbar
        bus.subscribe("active/*/changed", self._on_active_context_changed)

        self.show_item_groups([])

        # Set default expanded size to minimum size hint needed for content
        self.expanded_size = self.sizeHint().width()
        self.setFixedWidth(self.expanded_size)
        self.setMinimumWidth(0)
        self.setMaximumWidth(16777215)

    def _on_active_context_changed(self, topic: str, payload: dict):
        """Updates the run name when the active model or prediction changes."""
        from whisker.core.workflows.workflow_enum import Workflow
        
        # Map topics to workflows
        topic_map = {
            "active/pose_model/changed": Workflow.POSE_ESTIMATION,
            "active/pose_prediction/changed": Workflow.POSE_ESTIMATION,
            "active/behavior_model/changed": Workflow.BEHAVIOR_CLASSIFICATION,
            "active/behavior_prediction/changed": Workflow.BEHAVIOR_CLASSIFICATION,
        }
        
        if topic in topic_map:
            workflow = topic_map[topic]
            if workflow == self.controller.current_workflow():
                # Use 'name' from the payload as published by RunContextToolbar
                run_name = payload.get("name", "")
                self.set_run_name(run_name)

    def _on_bus_model_selected(self, topic: str, payload: dict):
        """Filters model selection events to only act on those matching the active workflow."""
        from whisker.core.workflows.workflow_enum import Workflow
        
        topic_to_workflow = {
            "workspace/pose_estimation/model/selected": Workflow.POSE_ESTIMATION,
            "workspace/behavior_classification/model/selected": Workflow.BEHAVIOR_CLASSIFICATION,
        }
        
        if topic in topic_to_workflow:
            event_workflow = topic_to_workflow[topic]
            if event_workflow == self.controller.current_workflow():
                run_name = payload.get("run_name", "")
                self.set_run_name(run_name)

    # --- Public API (Restored) ---

    def show_item_groups(self, groups: list[ItemGroupEnum]):
        if not groups:
            groups = list(ItemGroupEnum.__members__.values())

        # Check if groups are the same as current
        current_groups = set()
        for i in range(self.dropdown.count()):
            data = self.dropdown.itemData(i)
            if data is None:
                continue

            current_groups.add(data)
        
        if current_groups == set(groups):
            return

        blocker = QSignalBlocker(self.dropdown)

        self.dropdown.setCurrentText(groups[0].value)
        self.dropdown.clear()
        for group in groups:
            self.dropdown.addItem(group.value, group)
        
        self._update_data_tree()
        del blocker  # Release the signal blocker

    def set_workflow(self, workflow):
        if self.controller.current_workflow() == workflow:
            return
        
        logging.debug(f"Data Explorer Widget updating workflow to {workflow}")
        self.controller.set_workflow(workflow)
        self._update_data_tree()
    
    def set_run_name(self, run_name: Optional[str]):
        if self.controller.current_run_name() == run_name:
            return
        
        logging.debug(f"Data Explorer Widget updating run name to {run_name}")
        self.controller.set_run_name(run_name)
        self._update_data_tree()

    def show_create_project_dialog(self):
        """Facade method to trigger project creation dialog."""
        self.action_handler.show_create_project_dialog()

    def show_create_dataset_dialog(self):
        """Facade method to trigger dataset creation dialog."""
        self.action_handler.show_create_dataset_dialog()

    def show_import_pose_labels_dialog(self):
        """Facade method to trigger labels import."""
        self.action_handler.show_import_pose_labels_dialog()

    def show_import_bundle_dialog(self):
        """Facade method to trigger annotation-bundle import."""
        self.action_handler.show_import_bundle_dialog()



    def update_workspace(self, workspace: Optional[Workspace]):
        self._workspace = workspace
        self.controller.update_workspace(workspace)
        self.action_handler.update_workspace(workspace)
        self._update_data_tree()

    def set_active_project(self, project: Optional[Project]):
        self.action_handler.set_active_project(project)

    def set_item_group(self, group: ItemGroupEnum):
        self.dropdown.setCurrentText(group.value)
        self._update_data_tree()

    def is_blind_mode_enabled(self) -> bool:
        return self.blind_mode_checkbox.isChecked()
    
    def select_item_by_path(self, file_path: Path):
        self.controller.select_item_by_path(file_path)

    def select_previous_image(self):
        self.controller.select_sibling_image(-1)

    def select_next_image(self):
        self.controller.select_sibling_image(1)

    # --- Signal Handlers (Delegated) ---

    def _on_global_refresh(self):
        # Refresh if looking at Datasets
        if self.dropdown.currentText() == ItemGroupEnum.DATASETS.value:
            self._update_data_tree()

    def _on_blind_mode_toggled(self, checked: bool):
        self.controller.set_blind_mode(checked)
        self._update_data_tree()
        self.blind_mode_toggled.emit(checked)

    def _apply_filter(self):
        self.controller.apply_filter(self.filter_dropdown.currentData())

    def _on_item_expanded(self, item: QTreeWidgetItem):
        # Delegate lazy loading check
        self.controller.handle_expansion_event(item, self.dropdown.currentText())

    def _on_context_menu(self, pos):
        if not self.dropdown.currentText():
            return

        group = ItemGroupEnum(self.dropdown.currentText())
        self.action_handler.handle_context_menu_request(pos, self.data_tree, group)

    def _update_data_tree(self):
        if not self.dropdown.currentText():
            return

        current_group = ItemGroupEnum(self.dropdown.currentText())
        is_ds_view = current_group == ItemGroupEnum.DATASETS
        has_ws = self._workspace is not None

        # Visibility Toggle
        self.import_labels_action.setVisible(is_ds_view)
        self.import_bundle_action.setVisible(is_ds_view)
        self.blind_mode_checkbox.setVisible(has_ws and is_ds_view)
        self.filter_dropdown.setVisible(has_ws and is_ds_view)

        # Enable State
        self.options_button.setEnabled(has_ws)
        self.create_project_action.setEnabled(has_ws)
        self.create_dataset_action.setEnabled(has_ws)
        self.import_labels_action.setEnabled(has_ws)
        self.import_bundle_action.setEnabled(has_ws)

        # Populate
        self.controller.populate_tree(current_group)
        self._apply_filter()

    # --- Selection Logic ---

    def _on_item_changed(self, current: Optional[QTreeWidgetItem], previous=None):
        if not current or not self._workspace:
            self.item_deselected.emit()
            return
            
        # Ignore dummy "Loading..." nodes
        if current.text(0) == "Loading...": 
            return

        # Build hierarchy for signal
        hierarchy = get_item_hierarchy(current)
        
        # Logging
        log_h = []
        temp = current
        while temp:
            log_h.append(temp.text(0))
            temp = temp.parent()
        logging.debug(f"Selected: {' -> '.join(reversed(log_h))}")

        if not self.dropdown.currentText():
            logging.debug("No item group selected in dropdown, cannot infer selection type.")
            return

        group = ItemGroupEnum(self.dropdown.currentText())
        try:
            item_type = infer_selection_item_type(group, current, self._workspace)
            self.item_selected.emit(
                Selection(
                    group=group,
                    type=item_type,
                    item=hierarchy,
                    is_leaf=current.childCount() == 0,
                )
            )
        except (ValueError, AttributeError):
            # Happens on structural nodes (e.g. "Video Collection")
            self.item_deselected.emit()