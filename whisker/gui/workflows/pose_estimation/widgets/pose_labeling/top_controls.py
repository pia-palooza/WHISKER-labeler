import logging
from pathlib import Path
from typing import Optional
import pandas as pd

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox, QWidget, QHBoxLayout, QLabel, QPushButton, QCheckBox
)

from whisker.services.pose_estimation.public.prediction_operations import PosePredictionOperations
from whisker.core.study.dataset import Dataset
from whisker.core.study.project import Project
from whisker.gui.constants import KEYPOINT_QCOLORS
from whisker.gui.signals import MessageBus
from ..pose_overlay import build_pose_overlay_points


class PoseLabelingTopControlsWidget(QWidget):
    display_predictions_requested = pyqtSignal(list)
    clear_predictions_requested = pyqtSignal()
    accept_predictions_requested = pyqtSignal(pd.DataFrame, float)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._pose_prediction_operations: Optional[PosePredictionOperations] = None
        self._project: Optional[Project] = None
        self._dataset: Optional[Dataset] = None
        self._image_path: Optional[Path] = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 0, 5)

        layout.addWidget(QLabel("<b>Primer Predictions:</b>"))
        self.pred_model_combo = QComboBox()
        self.pred_model_combo.setPlaceholderText("Select model...")
        layout.addWidget(self.pred_model_combo)

        self.show_preds_checkbox = QCheckBox("Overlay Predictions")
        layout.addWidget(self.show_preds_checkbox)

        self.accept_preds_button = QPushButton("Accept Predictions as Labels")
        self.accept_preds_button.setEnabled(False)
        layout.addWidget(self.accept_preds_button)
        layout.addStretch()

        self.pred_model_combo.currentTextChanged.connect(self._on_context_changed)
        self.show_preds_checkbox.toggled.connect(self._on_context_changed)
        self.show_preds_checkbox.toggled.connect(self.accept_preds_button.setEnabled)
        self.accept_preds_button.clicked.connect(self._on_accept)

        MessageBus.get().subscribe("workspace/predictions/refreshed", lambda t, p: self._populate_models())

    def set_context(
        self, 
        pose_prediction_operations: Optional[PosePredictionOperations], 
        project: Optional[Project], 
        dataset: Optional[Dataset], 
        image_path: Optional[Path]
    ):
        self._pose_prediction_operations = pose_prediction_operations
        self._project = project
        self._dataset = dataset
        self._image_path = image_path
        self._populate_models()
        self.show_preds_checkbox.setChecked(False)

    def _populate_models(self):
        self.pred_model_combo.blockSignals(True)
        self.pred_model_combo.clear()
        if self._pose_prediction_operations:
            pred_dir = self._pose_prediction_operations.base_dir
            if pred_dir.is_dir():
                runs = sorted([d.name for d in pred_dir.iterdir() if d.is_dir()], reverse=True)
                if runs:
                    self.pred_model_combo.addItems([""] + runs)
        self.pred_model_combo.blockSignals(False)

    def _on_context_changed(self):
        show_preds = self.show_preds_checkbox.isChecked()
        model_name = self.pred_model_combo.currentText()

        if not show_preds or not model_name or not self._pose_prediction_operations or not self._image_path or not self._dataset:
            self.clear_predictions_requested.emit()
            return

        try:
            image_df = self._pose_prediction_operations.get_pose_predictions_for_image(
                model_name, self._dataset, self._image_rel_path()
            )
            if image_df is not None:
                self._format_and_display(image_df)
            else:
                self.clear_predictions_requested.emit()
        except Exception as e:
            logging.error(f"[Primer] Error loading predictions: {e}")
            self.clear_predictions_requested.emit()

    def _image_rel_path(self) -> str:
        dataset_base = Path(self._dataset.base_data_path)
        return str(self._image_path.relative_to(dataset_base)).replace("\\", "/")

    def _format_and_display(self, image_df):
        def color_for(identity_id):
            color_idx = self._project.identities.index(identity_id)
            return KEYPOINT_QCOLORS[color_idx % len(KEYPOINT_QCOLORS)]

        points_data = build_pose_overlay_points(
            image_df, self._project.identities, self._project.skeleton, color_for
        )
        self.display_predictions_requested.emit(points_data)

    def _on_accept(self):
        model_name = self.pred_model_combo.currentText()
        if not model_name or not self._pose_prediction_operations or not self._dataset:
            return

        try:
            image_df = self._pose_prediction_operations.get_pose_predictions_for_image(
                model_name, self._dataset, self._image_rel_path()
            )
            if image_df is not None:
                self.accept_predictions_requested.emit(image_df, 0.1)
                self.show_preds_checkbox.setChecked(False)
        except Exception as e:
            logging.error(f"[Primer] Failed to accept predictions: {e}")
