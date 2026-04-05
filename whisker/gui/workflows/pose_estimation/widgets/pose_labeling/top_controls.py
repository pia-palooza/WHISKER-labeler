import logging
from pathlib import Path
from typing import Optional
import pandas as pd

from PyQt6.QtCore import QPointF, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox, QWidget, QHBoxLayout, QLabel, QPushButton, QCheckBox
)

from whisker.core.workflows.pose_estimation.operations.prediction_operations import PosePredictionOperations
from whisker.core.dataset import Dataset
from whisker.core.project import Project
from whisker.core.workflows.pose_estimation.data_structures import PoseDataset
from whisker.gui.constants import KEYPOINT_QCOLORS


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

        if not show_preds or not model_name or not self._pose_prediction_operations or not self._image_path:
            self.clear_predictions_requested.emit()
            return

        dataset_name = self._dataset.name
        try:
            preds = self._pose_prediction_operations.get_pose_predictions(
                model_name, dataset_name, video_stem=None, raise_if_missing=False
            )
            if preds:
                self._format_and_display(preds)
            else:
                self.clear_predictions_requested.emit()
        except Exception as e:
            logging.error(f"[Primer] Error loading predictions: {e}")
            self.clear_predictions_requested.emit()

    def _format_and_display(self, pose_data: PoseDataset):
        dataset_base = Path(self._dataset.base_data_path)
        image_key = str(self._image_path.relative_to(dataset_base)).replace("\\", "/")
        
        try:
            image_df = pose_data.keypoint_data.loc[image_key]
        except KeyError:
            self.clear_predictions_requested.emit()
            return

        points_data = []
        for identity_id in self._project.identities:
            try:
                inst_df = image_df.loc[identity_id]
            except KeyError: 
                continue

            points = {}
            if isinstance(inst_df, pd.Series):
                if pd.notna(inst_df.get('c')) and inst_df.get('c', 0) > 0.1:
                    points[inst_df.name] = QPointF(inst_df['x'], inst_df['y'])
            else:
                valid_rows = inst_df[(inst_df['c'] > 0.1) & inst_df['c'].notna()]
                for bp_name, row in valid_rows.iterrows():
                    points[str(bp_name)] = QPointF(row['x'], row['y'])

            lines = [
                (points[p1], points[p2]) 
                for p1, p2 in self._project.skeleton 
                if p1 in points and p2 in points
            ]

            if points or lines:
                color_idx = self._project.identities.index(identity_id)
                color = KEYPOINT_QCOLORS[color_idx % len(KEYPOINT_QCOLORS)]
                points_data.append({
                    "name": identity_id,
                    "color": color,
                    "points": points,
                    "lines": lines,
                })
                
        self.display_predictions_requested.emit(points_data)

    def _on_accept(self):
        model_name = self.pred_model_combo.currentText()
        if not model_name or not self._pose_prediction_operations:
            return
            
        dataset_base = Path(self._dataset.base_data_path)
        image_key = str(self._image_path.relative_to(dataset_base)).replace("\\", "/")

        try:
            preds = self._pose_prediction_operations.get_pose_predictions(
                model_name, self._dataset.name, video_stem=None, raise_if_missing=False
            )
            if preds:
                image_df = preds.keypoint_data.loc[image_key]
                self.accept_predictions_requested.emit(image_df, 0.1)
                self.show_preds_checkbox.setChecked(False)
        except Exception as e:
            logging.error(f"[Primer] Failed to accept predictions: {e}")
