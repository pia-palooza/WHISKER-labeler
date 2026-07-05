# START_FILE: whisker/gui/widgets/behaviors_labeling_widget.py [Add keyboard shortcuts and UI hints]
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PyQt6.QtCore import pyqtSlot, pyqtSignal, Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
    QMessageBox,
    QSplitter,
    QFrame,
    QCheckBox,
    QSpinBox,
    QGroupBox,
    QFormLayout,
)

from whisker.core.workspace import Workspace
from whisker.core.study.project import Project
from whisker.services.behavior_classification.public.data_structures import Bout, BehaviorDataset
from whisker.services.behavior_classification.public.label_operations import create_frame_wise_labels
from whisker.gui.widgets.media_viewer import MediaViewerWidget
from whisker.gui.widgets.expanding_combo_box import ExpandingComboBox
from ..widgets.probability_plot import ProbabilityPlotPanel


class BehaviorsLabelingWidget(QWidget):
    """A self-contained widget for annotating behavior bouts in a single video."""

    labels_saved = pyqtSignal(str, str)
    data_changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._workspace: Optional[Workspace] = None
        self._project: Optional[Project] = None
        self._dataset_name: Optional[str] = None
        self._video_path: Optional[Path] = None
        self._behaviors: list[str] = []
        self._behavior_dataset: Optional[BehaviorDataset] = None
        self._video_key = None
        
        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        """Initializes the main UI for labeling a single video."""
        main_layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        viewer_panel = QWidget()
        viewer_layout = QVBoxLayout(viewer_panel)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_layout.setSpacing(0)

        self.media_viewer = MediaViewerWidget()
        self.media_viewer.setToolTip(
            "Space: Play/Pause | Arrow Keys: Seek Frame | Shift+Arrow: Skip"
        )
        viewer_layout.addWidget(self.media_viewer)

        # --- Predictions Timeline ---
        self.predictions_control_layout = QHBoxLayout()
        self.predictions_control_layout.setContentsMargins(10, 2, 10, 2)
        self.predictions_control_layout.setSpacing(10)

        self.show_predictions_checkbox = QCheckBox("Show Predictions")
        self.show_predictions_checkbox.setStyleSheet("font-size: 11px; color: #888;")

        self.run_label = QLabel("Run:")
        self.run_label.setStyleSheet("font-size: 11px; color: #888;")

        self.predictions_combo = ExpandingComboBox()
        self.predictions_combo.setPlaceholderText("Select prediction run...")
        self.predictions_combo.setMaximumHeight(24)
        self.predictions_combo.setStyleSheet("font-size: 11px;")

        self.predictions_control_layout.addWidget(self.show_predictions_checkbox)
        self.predictions_control_layout.addWidget(self.run_label)
        self.predictions_control_layout.addWidget(self.predictions_combo)
        self.predictions_control_layout.addStretch()
        viewer_layout.addLayout(self.predictions_control_layout)

        self.prob_plot = ProbabilityPlotPanel(title_text="BEHAVIOR TIMELINE")
        viewer_layout.addWidget(self.prob_plot)

        splitter.addWidget(viewer_panel)

        control_panel = self._create_control_panel()
        splitter.addWidget(control_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

    def _create_control_panel(self) -> QWidget:
        """Builds the right-hand side panel with all the controls."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # --- Annotation Editor ---
        editor_group = QGroupBox("Annotation Editor")
        editor_layout = QFormLayout(editor_group)

        self.start_frame_spinbox = QSpinBox()
        self.set_start_btn = QPushButton("Set from Current (`T`)")
        start_layout = QHBoxLayout()
        start_layout.addWidget(self.start_frame_spinbox)
        start_layout.addWidget(self.set_start_btn)
        editor_layout.addRow("Start Frame:", start_layout)

        self.end_frame_spinbox = QSpinBox()
        self.set_end_btn = QPushButton("Set from Current (`E`)")
        end_layout = QHBoxLayout()
        end_layout.addWidget(self.end_frame_spinbox)
        end_layout.addWidget(self.set_end_btn)
        editor_layout.addRow("End Frame:", end_layout)

        self.behavior_combo = QComboBox()
        self.behavior_combo.setPlaceholderText("Select Behavior...")
        editor_layout.addRow("Behavior:", self.behavior_combo)

        action_layout = QHBoxLayout()
        self.create_update_btn = QPushButton("Create (`C`)")
        self.clear_selection_btn = QPushButton("Clear / New (`Esc`)")
        action_layout.addWidget(self.create_update_btn)
        action_layout.addWidget(self.clear_selection_btn)
        editor_layout.addRow(action_layout)

        layout.addWidget(editor_group)

        # --- Annotations Table ---
        table_group = QGroupBox("Annotations for this video")
        table_layout = QVBoxLayout(table_group)
        self.bout_table = QTableWidget()
        self.bout_table.setColumnCount(3)
        self.bout_table.setHorizontalHeaderLabels(["Behavior", "Start", "End"])
        self.bout_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.bout_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.bout_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table_layout.addWidget(self.bout_table)
        self.remove_bout_btn = QPushButton("Remove Selected (`Del`)")
        table_layout.addWidget(self.remove_bout_btn, 0, Qt.AlignmentFlag.AlignRight)
        layout.addWidget(table_group)

        # --- Global Controls ---
        self.save_btn = QPushButton("Save (`Ctrl+S`)")
        layout.addWidget(self.save_btn, 0, Qt.AlignmentFlag.AlignRight)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator)
        self.overlay_checkbox = QCheckBox("Show Behavior Overlay (`O`)")
        self.overlay_checkbox.setChecked(True)
        layout.addWidget(self.overlay_checkbox)
        font_size_layout = QHBoxLayout()
        font_size_layout.addWidget(QLabel("Overlay Font Size:"))
        self.overlay_font_size_spinbox = QSpinBox()
        self.overlay_font_size_spinbox.setRange(8, 100)
        self.overlay_font_size_spinbox.setValue(
            MediaViewerWidget.DEFAULT_OVERLAY_FONT_SIZE
        )
        font_size_layout.addWidget(self.overlay_font_size_spinbox)
        font_size_layout.addStretch()
        layout.addLayout(font_size_layout)

        layout.addStretch()
        return panel

    def keyPressEvent(self, event: QKeyEvent):
        """Handle keyboard shortcuts for behavior labeling."""
        key = event.key()
        modifiers = event.modifiers()

        if modifiers == Qt.KeyboardModifier.NoModifier:
            if key in (Qt.Key.Key_Escape,):
                self._enter_create_mode()
            elif key == Qt.Key.Key_Space:
                self.media_viewer.toggle_play_pause()
            elif key == Qt.Key.Key_Left:
                self.media_viewer._single_step_backward()
            elif key == Qt.Key.Key_Right:
                self.media_viewer._single_step_forward()
            elif key == Qt.Key.Key_T:
                self._on_set_start_clicked()
            elif key == Qt.Key.Key_E:
                self._on_set_end_clicked()
            elif key == Qt.Key.Key_C:
                self._on_create_update_clicked()
            elif key == Qt.Key.Key_O:
                self.overlay_checkbox.toggle()
            elif key == Qt.Key.Key_Delete:
                self._on_remove_bout()
            else:
                super().keyPressEvent(event)
        elif modifiers == Qt.KeyboardModifier.ShiftModifier:
            if key == Qt.Key.Key_Left:
                self.media_viewer._skip_backward()
            elif key == Qt.Key.Key_Right:
                self.media_viewer._skip_forward()
            else:
                super().keyPressEvent(event)
        elif modifiers == Qt.KeyboardModifier.ControlModifier and key == Qt.Key.Key_S:
            self._on_save()
        else:
            super().keyPressEvent(event)

    def _connect_signals(self):
        """Connects all the internal signals and slots for the widget."""
        self.set_start_btn.clicked.connect(self._on_set_start_clicked)
        self.set_end_btn.clicked.connect(self._on_set_end_clicked)
        self.create_update_btn.clicked.connect(self._on_create_update_clicked)
        self.clear_selection_btn.clicked.connect(self._enter_create_mode)

        self.remove_bout_btn.clicked.connect(self._on_remove_bout)
        self.save_btn.clicked.connect(self._on_save)
        self.bout_table.itemSelectionChanged.connect(self._on_table_selection_changed)
        self.media_viewer.frame_changed.connect(self._on_frame_changed)
        self.media_viewer.frame_changed.connect(self.prob_plot.set_current_frame)
        self.prob_plot.bout_clicked.connect(self.media_viewer.seek_to_frame)
        self.show_predictions_checkbox.toggled.connect(self._on_show_predictions_toggled)
        self.predictions_combo.currentTextChanged.connect(self._update_plot)
        self.data_changed.connect(self._update_plot)
        self.overlay_checkbox.toggled.connect(self.media_viewer.set_overlay_visible)
        self.overlay_font_size_spinbox.valueChanged.connect(
            self.media_viewer.set_overlay_font_size
        )

    def set_project(self, project: Optional[Project]):
        """Sets the active project to get behavior definitions from."""
        self._project = project
        self._behaviors = project.behaviors if project else []
        self.behavior_combo.clear()
        if self._behaviors:
            self.behavior_combo.addItems(self._behaviors)
        
        # Pass behaviors to the dataset object if it exists
        if self._behavior_dataset:
            self._behavior_dataset.behaviors = self._behaviors

        if self._project:
            self.prob_plot.set_project_name(self._project.name)
            
        self._load_data()

    def set_media(
        self,
        workspace: Optional[Workspace],
        dataset_name: Optional[str],
        video_path: Optional[Path],
    ):
        """Sets the video to be labeled and loads all necessary context."""
        self._workspace = workspace
        self._dataset_name = dataset_name
        self._video_path = video_path

        is_valid_context = all([workspace, dataset_name, video_path, self._project])
        self.setEnabled(is_valid_context)
        self.media_viewer.set_media(video_path)
        
        self._video_key = video_path.name if video_path else None
        self._behavior_dataset = None

        if is_valid_context:
            # Load the dataset for this dataset_name
            self._behavior_dataset = self._workspace.get_behavior_labels(self._dataset_name)
            # Ensure the dataset's behavior list matches the project's
            self._behavior_dataset.behaviors = self._behaviors

            self.media_viewer.set_overlay_visible(self.overlay_checkbox.isChecked())
            self.media_viewer.set_overlay_font_size(
                self.overlay_font_size_spinbox.value()
            )
            # Set spinbox range based on video length
            max_frame = self.media_viewer.total_frames - 1
            if max_frame > 0:
                self.start_frame_spinbox.setRange(0, max_frame)
                self.end_frame_spinbox.setRange(0, max_frame)

            # Populate prediction runs
            current_run = self.predictions_combo.currentText()
            self.predictions_combo.blockSignals(True)
            self.predictions_combo.clear()
            self.predictions_combo.addItem("")
            if self._workspace:
                pred_dir = self._workspace.behavior_predictions.base_dir
                if pred_dir.is_dir():
                    run_names = sorted(
                        [d.name for d in pred_dir.iterdir() if d.is_dir()],
                        reverse=True,
                    )
                    self.predictions_combo.addItems(run_names)
                    if current_run in run_names:
                        self.predictions_combo.setCurrentText(current_run)
            self.predictions_combo.blockSignals(False)
        self._load_data()

    def _load_data(self):
        """Loads annotations based on the current workspace, dataset, and video."""
        if not all(
            [self._workspace, self._dataset_name, self._video_key, self._project]
        ):
            # Clear everything if context is incomplete
            self._behavior_dataset = None
            self._video_key = None
        
        # Data is already loaded in set_media via get_behavior_labels
        # We just need to refresh the UI
        self._populate_table()
        self._update_plot()
        self._enter_create_mode()
        self._on_frame_changed(self.media_viewer.get_current_frame())

    def _enter_create_mode(self):
        """Resets the editor to its default state for creating new bouts."""
        self.bout_table.clearSelection()
        self.start_frame_spinbox.setValue(0)
        self.end_frame_spinbox.setValue(0)
        self.behavior_combo.setCurrentIndex(-1)
        self.create_update_btn.setText("Create (`C`)")

    def _enter_edit_mode(self, behavior: str, bout: Bout):
        """Populates the editor with data from a selected bout."""
        self.start_frame_spinbox.setValue(bout.start_frame)
        self.end_frame_spinbox.setValue(bout.end_frame)
        self.behavior_combo.setCurrentText(behavior)
        self.create_update_btn.setText("Update (`C`)")

    @pyqtSlot()
    def _on_table_selection_changed(self):
        """When a table row is selected, enter edit mode for that bout."""
        selected_items = self.bout_table.selectedItems()
        if not selected_items:
            # When selection is cleared, revert to create mode by resetting the editor
            self.start_frame_spinbox.setValue(0)
            self.end_frame_spinbox.setValue(0)
            self.behavior_combo.setCurrentIndex(-1)
            self.create_update_btn.setText("Create (`C`)")
            return

        # We are now in edit mode
        row = selected_items[0].row()
        item_data = self.bout_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        if item_data:
            behavior, bout = item_data
            self._enter_edit_mode(behavior, bout)
            self.media_viewer.seek_to_frame(bout.start_frame)

    @pyqtSlot(int)
    def _on_frame_changed(self, frame: int):
        """Updates the overlay text with active behaviors for the current frame."""
        if not self._behavior_dataset or not self._video_key or frame < 0:
            self.media_viewer.set_overlay_text("Labels: None")
            return

        active_behaviors = []
        
        # Filter bouts DataFrame for the current video and frame
        bouts_df = self._behavior_dataset.bouts
        if not bouts_df.empty:
            active_df = bouts_df[
                (bouts_df["video_key"] == self._video_key)
                & (bouts_df["start_frame"] <= frame)
                & (bouts_df["end_frame"] >= frame)
            ]
            if not active_df.empty:
                active_behaviors = active_df["behavior"].unique().tolist()

        text = (
            f"Labels: {', '.join(sorted(active_behaviors))}"
            if active_behaviors
            else "Labels: None"
        )
        self.media_viewer.set_overlay_text(text)

    def _populate_table(self):
        """Fills the annotation table from the current data model."""
        self.bout_table.setRowCount(0)
        if not self._behavior_dataset or not self._video_key:
            return

        bouts_df = self._behavior_dataset.bouts
        if bouts_df.empty:
            return
            
        video_bouts_df = bouts_df[bouts_df["video_key"] == self._video_key].copy()
        if video_bouts_df.empty:
            return

        video_bouts_df.sort_values(by="start_frame", inplace=True)

        self.bout_table.setRowCount(len(video_bouts_df))
        for i, row in enumerate(video_bouts_df.itertuples()):
            behavior = row.behavior
            bout = Bout(
                start_frame=row.start_frame,
                end_frame=row.end_frame,
                p=row.p if pd.notna(row.p) else None,
            )
            
            behavior_item = QTableWidgetItem(behavior)
            # Store the identifying info to find this row later
            behavior_item.setData(Qt.ItemDataRole.UserRole, (behavior, bout))
            self.bout_table.setItem(i, 0, behavior_item)
            self.bout_table.setItem(i, 1, QTableWidgetItem(str(bout.start_frame)))
            self.bout_table.setItem(i, 2, QTableWidgetItem(str(bout.end_frame)))

    def _on_set_start_clicked(self):
        """Sets the start frame spinbox to the current video frame."""
        frame = self.media_viewer.get_current_frame()
        if frame >= 0:
            self.start_frame_spinbox.setValue(frame)

    def _on_set_end_clicked(self):
        """Sets the end frame spinbox to the current video frame."""
        frame = self.media_viewer.get_current_frame()
        if frame >= 0:
            self.end_frame_spinbox.setValue(frame)

    def _on_create_update_clicked(self):
        """Handles both creating a new bout and updating an existing one."""
        if not self._behavior_dataset:
            return

        start = self.start_frame_spinbox.value()
        end = self.end_frame_spinbox.value()
        behavior = self.behavior_combo.currentText()

        if end < start:
            QMessageBox.warning(
                self, "Invalid Range", "End frame must be after start frame."
            )
            return
        if not behavior:
            QMessageBox.warning(self, "No Behavior", "Please select a behavior.")
            return

        # Decide whether to update or create based on table selection
        selected_items = self.bout_table.selectedItems()
        if selected_items:  # Update mode
            row = selected_items[0].row()
            item_data = self.bout_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
            if not item_data:
                logging.error("Could not find item data for update. Aborting.")
                return

            old_behavior, old_bout = item_data
            
            # Find the row in the main DataFrame
            bouts_df = self._behavior_dataset.bouts
            row_index = bouts_df[
                (bouts_df["video_key"] == self._video_key) &
                (bouts_df["behavior"] == old_behavior) &
                (bouts_df["start_frame"] == old_bout.start_frame) &
                (bouts_df["end_frame"] == old_bout.end_frame)
            ].index
            
            if row_index.empty:
                logging.error("Could not find corresponding row in DataFrame to update.")
                return
                
            # Update the row
            self._behavior_dataset.bouts.loc[
                row_index, ["behavior", "start_frame", "end_frame"]
            ] = [behavior, start, end]
            
        else:  # Create mode
            new_row = pd.DataFrame([{
                "video_key": self._video_key,
                "behavior": behavior,
                "start_frame": start,
                "end_frame": end,
                "p": np.nan
            }])
            self._behavior_dataset.bouts = pd.concat(
                [self._behavior_dataset.bouts, new_row], ignore_index=True
            )

        self._populate_table()
        self._update_plot()
        self._on_frame_changed(self.media_viewer.get_current_frame())
        self.data_changed.emit()
        self._enter_create_mode()

    def _on_remove_bout(self):
        """Removes the selected bout from the table and data model."""
        if not self._behavior_dataset:
            return
            
        selected_items = self.bout_table.selectedItems()
        if not selected_items:
            QMessageBox.warning(
                self,
                "Selection Error",
                "Please select a bout from the table to remove.",
            )
            return

        row = selected_items[0].row()
        item_data = self.bout_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        if not item_data:
            logging.error("Could not find item data for removal. Aborting.")
            return

        behavior, bout_to_remove = item_data
        
        # Find the row in the main DataFrame
        bouts_df = self._behavior_dataset.bouts
        row_index = bouts_df[
            (bouts_df["video_key"] == self._video_key) &
            (bouts_df["behavior"] == behavior) &
            (bouts_df["start_frame"] == bout_to_remove.start_frame) &
            (bouts_df["end_frame"] == bout_to_remove.end_frame)
        ].index

        if not row_index.empty:
            self._behavior_dataset.bouts = self._behavior_dataset.bouts.drop(
                index=row_index
            )
            self._populate_table()
            self._update_plot()
            self._on_frame_changed(self.media_viewer.get_current_frame())
            self.data_changed.emit()
            self._enter_create_mode()
        else:
            logging.warning(
                "Could not find bout to remove. Table might be out of sync."
            )

    def _on_save(self):
        """Saves the entire behavior labels file for the current dataset."""
        if not all([self._workspace, self._dataset_name, self._video_path]):
            QMessageBox.critical(self, "Error", "No workspace or dataset context.")
            return

        try:
            self._workspace.save_behavior_labels(self._dataset_name)
            QMessageBox.information(self, "Success", "Behavior annotations saved!")
            self.labels_saved.emit(self._dataset_name, str(self._video_path))
        except Exception as e:
            logging.error(f"Failed to save behavior labels: {e}", exc_info=True)
            QMessageBox.critical(
                self, "Save Failed", f"Could not save annotations:\n{e}"
            )

    def _update_plot(self):
        """Updates the Probability Plot with Ground Truth and optionally Predictions."""
        if not self._behavior_dataset or not self._video_key:
            self.prob_plot.clear()
            return

        num_frames = self.media_viewer.total_frames
        fps = self.media_viewer.fps

        # Create ground truth df
        gt_df = create_frame_wise_labels(
            self._behavior_dataset, self._video_key, self._behaviors, num_frames
        )

        # Check if predictions should be loaded
        probs_df = None
        pred_binary_df = None
        if self.show_predictions_checkbox.isChecked():
            run_name = self.predictions_combo.currentText()
            if run_name:
                video_stem = Path(self._video_key).stem
                try:
                    pred_ds = self._workspace.get_behavior_predictions(
                        run_name, self._dataset_name, video_stem, raise_if_missing=False
                    )
                    if pred_ds:
                        if not pred_ds.per_frame_probabilities.empty:
                            probs_df = pred_ds.per_frame_probabilities

                        if not pred_ds.bouts.empty:
                            behaviors = pred_ds.behaviors or list(
                                pred_ds.bouts["behavior"].unique()
                            )
                            pred_key = (
                                self._video_key
                                if self._video_key in pred_ds.bouts["video_key"].values
                                else video_stem
                            )
                            pred_binary_df = create_frame_wise_labels(
                                pred_ds, pred_key, behaviors, num_frames
                            )
                except Exception as e:
                    logging.warning(f"Could not load predictions for {video_stem}: {e}")

        self.prob_plot.plot_probabilities(
            probs_df=probs_df, gt_df=gt_df, pred_binary_df=pred_binary_df, fps=fps
        )
        self.prob_plot.set_current_frame(self.media_viewer.get_current_frame())

    @pyqtSlot(bool)
    def _on_show_predictions_toggled(self, checked: bool):
        """Trigger a plot refresh when prediction visibility changes."""
        self._update_plot()
