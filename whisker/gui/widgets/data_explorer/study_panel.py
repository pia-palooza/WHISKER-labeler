import logging
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtWidgets import QLabel, QWidget, QVBoxLayout

from whisker.core.workspace import Workspace
from whisker.gui.signals import MessageBus
from whisker.core.workflows.workflow_enum import Workflow
from whisker.gui.widgets.expanding_combo_box import ExpandingComboBox
from whisker.gui.base.collapsible_panel import HorizontalCollapsiblePanel

from .tree_controller import (
    strip_model_label,
    format_model_display_name,
    is_model_hidden,
    strip_prediction_label,
    format_prediction_display_name,
    is_prediction_hidden,
)


class StudyPanel(HorizontalCollapsiblePanel):
    """
    A collapsible panel within the Data Explorer that provides global selection
    of models and prediction runs for the active workspace.
    Dynamically updates visibility of selectors based on active workflow and tab.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        title = QLabel("STUDY")
        title.setObjectName("HeaderLabel")
        super().__init__(title, parent=parent, drag_edges=Qt.Edge.TopEdge)

        self._workspace: Optional[Workspace] = None
        self._current_workflow: Workflow = Workflow.POSE_ESTIMATION
        self._current_tab: Optional[str] = None

        self._setup_ui()
        self._connect_signals()
        self._update_visibility()

    def _setup_ui(self):
        # Create a vertical layout to stack controls neatly inside the panel's content area
        self.vertical_layout = QVBoxLayout()
        self.vertical_layout.setContentsMargins(0, 0, 0, 0)
        self.vertical_layout.setSpacing(4)
        self.content_layout.addLayout(self.vertical_layout)

        # --- Active Project Selection ---
        self.project_label = QLabel("Active Project:")
        self.project_selector = ExpandingComboBox()
        self.project_selector.setPlaceholderText("Select Project...")
        self.vertical_layout.addWidget(self.project_label)
        self.vertical_layout.addWidget(self.project_selector)

        # --- Pose Model Selection ---
        self.pose_model_label = QLabel("Pose Model:")
        self.pose_model_selector = ExpandingComboBox()
        self.pose_model_selector.setPlaceholderText("Select Pose Model...")
        self.vertical_layout.addWidget(self.pose_model_label)
        self.vertical_layout.addWidget(self.pose_model_selector)

        # --- Pose Prediction Selection ---
        self.pose_pred_label = QLabel("Pose Prediction:")
        self.pose_pred_selector = ExpandingComboBox()
        self.pose_pred_selector.setPlaceholderText("Select Pose Prediction...")
        self.vertical_layout.addWidget(self.pose_pred_label)
        self.vertical_layout.addWidget(self.pose_pred_selector)

        # --- Behavior Model Selection ---
        self.behavior_model_label = QLabel("Behavior Model:")
        self.behavior_model_selector = ExpandingComboBox()
        self.behavior_model_selector.setPlaceholderText("Select Behavior Model...")
        self.vertical_layout.addWidget(self.behavior_model_label)
        self.vertical_layout.addWidget(self.behavior_model_selector)

        # --- Behavior Prediction Selection ---
        self.behavior_pred_label = QLabel("Behavior Prediction:")
        self.behavior_pred_selector = ExpandingComboBox()
        self.behavior_pred_selector.setPlaceholderText("Select Behavior Prediction...")
        self.vertical_layout.addWidget(self.behavior_pred_label)
        self.vertical_layout.addWidget(self.behavior_pred_selector)


    def _connect_signals(self):
        # Internal Changes -> Message Bus
        self.pose_model_selector.currentTextChanged.connect(
            lambda text: MessageBus.get().publish(
                "active/pose_model/changed", {"name": strip_model_label(text)}
            )
        )
        self.pose_pred_selector.currentTextChanged.connect(
            lambda text: MessageBus.get().publish(
                "active/pose_prediction/changed", {"name": strip_prediction_label(text)}
            )
        )
        self.behavior_model_selector.currentTextChanged.connect(
            lambda text: MessageBus.get().publish(
                "active/behavior_model/changed", {"name": strip_model_label(text)}
            )
        )
        self.behavior_pred_selector.currentTextChanged.connect(
            self._on_behavior_pred_selected
        )

        # External Refreshes
        MessageBus.get().subscribe(
            "workspace/models/refreshed", lambda t, p: self.refresh_selectors()
        )
        MessageBus.get().subscribe(
            "workspace/predictions/refreshed", lambda t, p: self.refresh_selectors()
        )

    def set_workspace(self, workspace: Optional[Workspace]):
        self._workspace = workspace
        self.refresh_selectors()

    @pyqtSlot(Workflow)
    def set_workflow(self, workflow: Workflow):
        self._current_workflow = workflow
        self._update_visibility()
        self._publish_current_selections()

    @pyqtSlot(str)
    def set_current_tab(self, tab_name: str):
        self._current_tab = tab_name
        self._update_visibility()
        self._publish_current_selections()

    def _publish_current_selections(self):
        """Re-publishes the current text of all selectors to ensure all components are in sync."""
        bus = MessageBus.get()
        is_pose = self._current_workflow == Workflow.POSE_ESTIMATION
        is_behavior = self._current_workflow == Workflow.BEHAVIOR_CLASSIFICATION

        is_predict_tab = self._current_tab == "Predict"
        is_evaluate_tab = self._current_tab == "Evaluate"
        is_amend_tab = self._current_tab == "Amend"
        is_debug_tab = self._current_tab == "Debug"

        if is_pose and is_predict_tab:
            bus.publish(
                "active/pose_model/changed",
                {"name": strip_model_label(self.pose_model_selector.currentText())},
            )

        show_pose_pred = (
            (is_pose and (is_amend_tab or is_evaluate_tab or is_debug_tab))
            or (is_behavior and (is_predict_tab or is_debug_tab))
        )
        if show_pose_pred:
            bus.publish(
                "active/pose_prediction/changed",
                {"name": strip_prediction_label(self.pose_pred_selector.currentText())},
            )

        if is_behavior and is_predict_tab:
            bus.publish(
                "active/behavior_model/changed",
                {"name": strip_model_label(self.behavior_model_selector.currentText())},
            )

        show_behavior_pred = is_behavior and (
            is_amend_tab or is_evaluate_tab or is_debug_tab
        )
        if show_behavior_pred:
            bus.publish(
                "active/behavior_prediction/changed",
                {
                    "name": strip_prediction_label(
                        self.behavior_pred_selector.currentText()
                    )
                },
            )


    def _update_visibility(self):
        is_pose = self._current_workflow == Workflow.POSE_ESTIMATION
        is_behavior = self._current_workflow == Workflow.BEHAVIOR_CLASSIFICATION

        # Tab checks
        is_predict_tab = self._current_tab == "Predict"
        is_evaluate_tab = self._current_tab == "Evaluate"
        is_amend_tab = self._current_tab == "Amend"
        is_debug_tab = self._current_tab == "Debug"

        # Pose Estimation
        # -> "Pose Model" visible only in Predict tab.
        # -> "Pose Predictions" visible only in Amend, Evaluate, or Debug tab.
        show_pose_model = is_pose and is_predict_tab
        show_pose_pred = (
            (is_pose and (is_amend_tab or is_evaluate_tab or is_debug_tab))
            or (is_behavior and (is_predict_tab or is_debug_tab))
        )

        # Behavior Classification
        # -> "Behavior Model" is visible only in Predict tab.
        # -> "Behavior Predictions" is visible only in Amend or Evaluate tab.
        show_behavior_model = is_behavior and is_predict_tab
        show_behavior_pred = is_behavior and (
            is_amend_tab or is_evaluate_tab or is_debug_tab
        )

        # Apply visibility to widgets
        self.pose_model_label.setVisible(show_pose_model)
        self.pose_model_selector.setVisible(show_pose_model)

        self.pose_pred_label.setVisible(show_pose_pred)
        self.pose_pred_selector.setVisible(show_pose_pred)

        self.behavior_model_label.setVisible(show_behavior_model)
        self.behavior_model_selector.setVisible(show_behavior_model)
        self.behavior_pred_label.setVisible(show_behavior_pred)
        self.behavior_pred_selector.setVisible(show_behavior_pred)

        # Clear cached expanded size to force recalculation based on new visible widgets
        self.expanded_size = None
        self.update_height()

    def update_height(self):
        if self.current_size() != self.collapsed_size:
            # Force layout recalculation
            self.vertical_layout.activate()
            self.content_container.adjustSize()
            self.adjustSize()
            self.setFixedHeight(self.sizeHint().height())

    def refresh_selectors(self):
        if not self._workspace:
            self.pose_model_selector.clear()
            self.pose_pred_selector.clear()
            self.behavior_model_selector.clear()
            self.behavior_pred_selector.clear()
            return

        self._populate_pose_models()
        self._populate_pose_predictions()
        self._populate_behavior_models()
        self._populate_behavior_predictions()

    def _populate_pose_models(self):
        self.pose_model_selector.blockSignals(True)
        current = self.pose_model_selector.currentText()
        current_raw = strip_model_label(current)
        self.pose_model_selector.clear()

        model_dir = self._workspace.pose_models.base_dir
        if model_dir.is_dir():
            names = sorted(
                [d.name for d in model_dir.iterdir() if d.is_dir()], reverse=True
            )
            workspace_path = str(self._workspace.base_dir)
            visible_names = [
                n for n in names if not is_model_hidden(workspace_path, n)
            ]
            display_items = [""] + [
                format_model_display_name(workspace_path, n) for n in visible_names
            ]
            self.pose_model_selector.addItems(display_items)
            if current_raw in visible_names:
                disp = format_model_display_name(workspace_path, current_raw)
                self.pose_model_selector.setCurrentText(disp)
        self.pose_model_selector.blockSignals(False)

    def _populate_pose_predictions(self):
        self.pose_pred_selector.blockSignals(True)
        current = self.pose_pred_selector.currentText()
        current_raw = strip_prediction_label(current)
        self.pose_pred_selector.clear()

        pred_dir = self._workspace.pose_predictions.base_dir
        if pred_dir.is_dir():
            names = sorted(
                [d.name for d in pred_dir.iterdir() if d.is_dir()], reverse=True
            )
            workspace_path = str(self._workspace.base_dir)
            visible_names = [
                n for n in names if not is_prediction_hidden(workspace_path, n)
            ]
            display_items = [""] + [
                format_prediction_display_name(workspace_path, n)
                for n in visible_names
            ]
            self.pose_pred_selector.addItems(display_items)
            if current_raw in visible_names:
                disp = format_prediction_display_name(workspace_path, current_raw)
                self.pose_pred_selector.setCurrentText(disp)
        self.pose_pred_selector.blockSignals(False)

    def _populate_behavior_models(self):
        self.behavior_model_selector.blockSignals(True)
        current = self.behavior_model_selector.currentText()
        current_raw = strip_model_label(current)
        self.behavior_model_selector.clear()

        model_dir = self._workspace.behavior_models.base_dir
        if model_dir.is_dir():
            names = sorted(
                [d.name for d in model_dir.iterdir() if d.is_dir()], reverse=True
            )
            workspace_path = str(self._workspace.base_dir)
            visible_names = [
                n for n in names if not is_model_hidden(workspace_path, n)
            ]
            display_items = [""] + [
                format_model_display_name(workspace_path, n) for n in visible_names
            ]
            self.behavior_model_selector.addItems(display_items)
            if current_raw in visible_names:
                disp = format_model_display_name(workspace_path, current_raw)
                self.behavior_model_selector.setCurrentText(disp)
        self.behavior_model_selector.blockSignals(False)

    def _populate_behavior_predictions(self):
        self.behavior_pred_selector.blockSignals(True)
        current = self.behavior_pred_selector.currentText()
        current_raw = strip_prediction_label(current)
        self.behavior_pred_selector.clear()

        pred_dir = self._workspace.behavior_predictions.base_dir
        if pred_dir.is_dir():
            names = sorted(
                [d.name for d in pred_dir.iterdir() if d.is_dir()], reverse=True
            )
            workspace_path = str(self._workspace.base_dir)
            visible_names = [
                n for n in names if not is_prediction_hidden(workspace_path, n)
            ]
            display_items = [""] + [
                format_prediction_display_name(workspace_path, n)
                for n in visible_names
            ]
            self.behavior_pred_selector.addItems(display_items)
            if current_raw in visible_names:
                disp = format_prediction_display_name(workspace_path, current_raw)
                self.behavior_pred_selector.setCurrentText(disp)
        self.behavior_pred_selector.blockSignals(False)


    @pyqtSlot(str)
    def _on_behavior_pred_selected(self, text: str):
        MessageBus.get().publish(
            "active/behavior_prediction/changed",
            {"name": strip_prediction_label(text)},
        )

        if not text or not self._workspace:
            return

        # Future refinement: scan the behavior prediction metadata to auto-select
        # the input pose prediction and behavior model.
        pass
