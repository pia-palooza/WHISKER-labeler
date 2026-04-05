import logging
from pathlib import Path
from typing import Optional
import pandas as pd

from PyQt6.QtCore import Qt, QPointF, pyqtSignal, QTimer
from PyQt6.QtGui import QPixmap, QKeyEvent
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QSplitter, QMessageBox, QStackedWidget
)

from whisker.core import Dataset, Project
from whisker.core.dataset_operations import DatasetOperations
from whisker.core.workflows.pose_estimation.data_structures import PoseDataset
from whisker.core.workflows.pose_estimation.operations.label_operations import PoseLabelOperations
from whisker.core.workflows.pose_estimation.operations.prediction_operations import PosePredictionOperations
from whisker.gui.widgets.interactive_image_label import InteractiveImageLabel
from .model import PoseLabelingModel
from .top_controls import PoseLabelingTopControlsWidget
from .bottom_controls import PoseLabelingBottomControlsWidget
from .side_controls import PoseLabelingSideControlsWidget

class PoseLabelingWidget(QWidget):
    labels_saved = pyqtSignal(str, str)
    data_changed = pyqtSignal()
    request_select_prev_image = pyqtSignal()
    request_select_next_image = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._pose_label_operations: Optional[PoseLabelOperations] = None
        self._project: Optional[Project] = None
        self._selected_dataset: Optional[Dataset] = None
        self._image_path: Optional[Path] = None

        self.model = PoseLabelingModel(self)
        
        self._init_ui()
        self._connect_signals()
        self.set_media(None, None, None)

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        self.controls = PoseLabelingSideControlsWidget()
        self.top_controls = PoseLabelingTopControlsWidget()
        self.bottom_controls = PoseLabelingBottomControlsWidget()

        self.image_viewer = InteractiveImageLabel()
        self.image_viewer.setStyleSheet("background-color: #222;")
        self.image_viewer.set_show_names(self.controls.show_names_checkbox.isChecked())

        self.placeholder = QLabel()
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setWordWrap(True)

        viewer_container = QWidget()
        stack_layout = QVBoxLayout(viewer_container)
        stack_layout.setContentsMargins(0, 0, 0, 0)
        
        stack_layout.addWidget(self.top_controls)
        self.top_controls.setVisible(False)
        
        self.center_stack = QStackedWidget()
        self.center_stack.addWidget(self.image_viewer)
        self.center_stack.addWidget(self.placeholder)
        stack_layout.addWidget(self.center_stack, 1)

        stack_layout.addWidget(self.bottom_controls)

        splitter.addWidget(self.controls)
        splitter.addWidget(viewer_container)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    def _connect_signals(self):
        self.image_viewer.point_placed.connect(self._on_point_placed)
        self.image_viewer.point_dragged.connect(self._on_point_dragged)

        self.controls.drag_mode_toggled.connect(self.image_viewer.set_drag_mode)
        self.controls.show_names_toggled.connect(self.image_viewer.set_show_names)
        self.controls.font_size_changed.connect(self.image_viewer.set_font_size)
        self.controls.clear_keypoint_requested.connect(self.model.clear_keypoint)
        self.controls.clear_all_keypoint_requested.connect(self.model.clear_all_keypoints)
        self.controls.swap_identities_requested.connect(self.model.swap_identities)

        self.bottom_controls.save_requested.connect(self._on_save_clicked)
        self.bottom_controls.swap_identities_requested.connect(self.model.swap_identities)
        self.bottom_controls.request_select_next_image.connect(self.request_select_next_image)
        self.bottom_controls.request_select_prev_image.connect(self.request_select_prev_image)

        self.top_controls.display_predictions_requested.connect(
            lambda pts: self.image_viewer.set_prediction_display_data(pts, None)
        )
        self.top_controls.clear_predictions_requested.connect(
            lambda: self.image_viewer.set_prediction_display_data([], None)
        )
        self.top_controls.accept_predictions_requested.connect(self.model.accept_predictions)

        self.model.model_updated.connect(self._on_model_update)
        self.model.dirty_state_changed.connect(self.data_changed)

    def set_context(self, pose_label_operations: Optional[PoseLabelOperations], pose_prediction_operations: Optional[PosePredictionOperations]):
        self._pose_label_operations = pose_label_operations
        self._pose_prediction_operations = pose_prediction_operations
        self.top_controls.set_context(self._pose_prediction_operations, self._project, self._selected_dataset, self._image_path)
        if not self._pose_label_operations or not self._pose_prediction_operations:
            self.set_media(None, None, None)

        self.top_controls.setVisible(self._pose_prediction_operations is not None)

    def set_project(self, project: Optional[Project]):
        self._project = project
        self.bottom_controls.set_identities(project.identities if project else [])
        self.top_controls.set_context(self._pose_prediction_operations, project, self._selected_dataset, self._image_path)
        self._update_model_data()

    def set_media(self, dataset_operations: Optional[DatasetOperations], dataset_name: Optional[str], image_path: Optional[Path]):        
        self._selected_dataset = dataset_operations.get_dataset(dataset_name) if (dataset_operations and dataset_name) else None
        self._image_path = image_path
        
        is_valid = all([dataset_operations, self._selected_dataset, image_path])
        self.center_stack.setCurrentWidget(self.image_viewer if is_valid else self.placeholder)
        self.controls.set_enabled_state(is_valid)
        self.top_controls.setEnabled(is_valid)
        self.bottom_controls.setEnabled(is_valid)

        if not is_valid:
            self.placeholder.setText("Select an image from the Data Explorer to begin.")
            self.image_viewer.set_full_pixmap(None)
            self._update_model_data()
            return

        self.image_viewer.set_full_pixmap(QPixmap(str(image_path)))
        self.controls.clear_selection()
        
        self.top_controls.set_context(self._pose_prediction_operations, self._project, self._selected_dataset, self._image_path)
        self._update_model_data()

        self.controls.set_checkbox('drag', False)

    def _update_model_data(self):
        pose_labels, image_key, dataset_name = None, None, None
        if self._pose_label_operations and self._selected_dataset and self._image_path and self._project:
            dataset_name = self._selected_dataset.name
            pose_labels = self._pose_label_operations.get_pose_dataset(dataset_name, raise_if_missing=False)
            if not pose_labels:
                pose_labels = PoseDataset(
                    body_parts=self._project.body_parts,
                    individuals=self._project.identities
                )
                self._pose_label_operations.set_pose_labels(dataset_name, pose_labels)

            dataset_base = Path(self._selected_dataset.base_data_path)
            image_key = str(self._image_path.relative_to(dataset_base)).replace("\\", "/")

        self.model.set_data(pose_labels, image_key, self._project, dataset_name)

    def _on_model_update(self):
        image_df = self.model.get_current_image_data()
        project = self.model.get_project()
        
        sel_id, sel_part = self.controls.get_current_selection()
        
        self.controls.update_tree(project, image_df)
        self._update_viewer_points(image_df, project)
        
        if sel_id is not None:
            self.controls.restore_selection(sel_id, sel_part)

    def _update_viewer_points(self, image_df, project):
        if image_df is None or not project:
            self.image_viewer.set_display_data([], None)
            return

        points_data = []
        for identity_id in project.identities:
            try:
                inst_df = image_df.loc[identity_id]
            except KeyError: continue

            labeled_kps = inst_df.dropna(subset=['c'])
            points = {name: QPointF(row['x'], row['y']) for name, row in labeled_kps.iterrows()}
            lines = [
                (points[p1], points[p2]) 
                for p1, p2 in project.skeleton 
                if p1 in points and p2 in points
            ]

            points_data.append({
                "name": identity_id,
                "color": self.controls.get_color_for_identity(identity_id),
                "points": points,
                "lines": lines,
            })
        self.image_viewer.set_display_data(points_data, None)

    def _on_point_placed(self, image_coords: QPointF):
        id_id, part_idx = self.controls.get_current_selection()
        if id_id is None:
            QMessageBox.warning(self, "No Selection", "Please select a body part in the tree to label.")
            return

        self.model.update_keypoint_position(id_id, part_idx, image_coords)
        
        if self.controls.is_autocycle_enabled():
            QTimer.singleShot(0, self.controls.cycle_to_next_auto)

    def _on_point_dragged(self, identity_id: str, part_name: str, coords: QPointF):
        project = self.model.get_project()
        if project:
            try:
                self.model.update_keypoint_position(identity_id, project.body_parts.index(part_name), coords)
            except ValueError: pass

    def _on_save_clicked(self):
        if not all([self._pose_label_operations, self._selected_dataset, self._project]):
            QMessageBox.warning(self, "Error", "Cannot save. Incomplete context.")
            return
        try:
            self.model.save(self._pose_label_operations)
            self.labels_saved.emit(self._selected_dataset.name, str(self._image_path))
        except Exception as e:
            logging.error(f"Failed to save: {e}", exc_info=True)
            QMessageBox.critical(self, "Save Failed", f"Could not save annotations:\n{e}")

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        mods = event.modifiers()
        
        if mods == Qt.KeyboardModifier.ControlModifier and key == Qt.Key.Key_S:
            self._on_save_clicked()
            return
        if mods == Qt.KeyboardModifier.ControlModifier and key == Qt.Key.Key_I:
            self.bottom_controls.swap_identities_shortcut()
            return
        if mods == Qt.KeyboardModifier.ControlModifier and key == Qt.Key.Key_N:
            self.controls.toggle_checkbox('names')
            return
        if mods == Qt.KeyboardModifier.ControlModifier and key == Qt.Key.Key_C:
            self.controls.toggle_checkbox('auto_cycle')
            return
        if mods == Qt.KeyboardModifier.ControlModifier and key == Qt.Key.Key_Delete:
            self.model.clear_all_keypoints()
            return

        actions = {
            Qt.Key.Key_Q: lambda: self.controls.toggle_checkbox('drag'),
            Qt.Key.Key_Delete: self.controls.request_clear_current,
            Qt.Key.Key_X: self.controls.request_clear_current,
            Qt.Key.Key_W: lambda: self.controls.cycle_selection('prev'),
            Qt.Key.Key_S: lambda: self.controls.cycle_selection('next'),
            Qt.Key.Key_A: lambda: self.controls.cycle_selection('prev_id'),
            Qt.Key.Key_D: lambda: self.controls.cycle_selection('next_id'),
            Qt.Key.Key_M: lambda: self.request_select_prev_image.emit,
            Qt.Key.Key_N: lambda: self.request_select_next_image.emit,
        }

        if action := actions.get(key):
            action()
        else:
            super().keyPressEvent(event)