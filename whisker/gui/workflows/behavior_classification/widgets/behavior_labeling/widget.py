# UPDATE_FILE: whisker/gui/workflows/behavior_classification/widgets/behavior_labeling/widget.py
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PyQt6.QtCore import pyqtSlot, pyqtSignal, Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QMessageBox, QSplitter, QFrame, QCheckBox, QSpinBox, QGroupBox, QFormLayout,
)

from whisker.core import Dataset, Project
from whisker.core.dataset_operations import DatasetOperations
from whisker.core.workflows.behavior_classification.data_structures import (
    Bout, BehaviorDataset,
)
from whisker.core.workflows.behavior_classification.operations.label_operations import (
    BehaviorLabelOperations,
)
from .media_viewer import MediaViewerWidget
from .timeline import BehaviorTimelineWidget


class BehaviorLabelingWidget(QWidget):
    """A self-contained widget for hand-annotating behavior bouts in a video."""

    labels_saved = pyqtSignal(str, str)
    data_changed = pyqtSignal()
    request_select_prev_video = pyqtSignal()
    request_select_next_video = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._label_operations: Optional[BehaviorLabelOperations] = None
        self._project: Optional[Project] = None
        self._selected_dataset: Optional[Dataset] = None
        self._video_path: Optional[Path] = None
        self._dataset_name: Optional[str] = None
        self._video_key: Optional[str] = None
        self._behaviors: list[str] = []
        self._behavior_dataset: Optional[BehaviorDataset] = None

        self._init_ui()
        self._connect_signals()
        self.set_media(None, None, None)

    # --- UI construction ---

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        viewer_panel = QWidget()
        viewer_layout = QVBoxLayout(viewer_panel)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_layout.setSpacing(0)

        self.media_viewer = MediaViewerWidget()
        self.media_viewer.setToolTip(
            "Space: Play/Pause | Arrows: Step Frame | Shift+Arrows: Skip"
        )
        viewer_layout.addWidget(self.media_viewer, 1)

        self.timeline = BehaviorTimelineWidget()
        viewer_layout.addWidget(self.timeline)

        splitter.addWidget(viewer_panel)

        control_panel = self._create_control_panel()
        splitter.addWidget(control_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

    def _create_control_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Annotation editor
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

        # Annotations table
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

    def _connect_signals(self):
        self.set_start_btn.clicked.connect(self._on_set_start_clicked)
        self.set_end_btn.clicked.connect(self._on_set_end_clicked)
        self.create_update_btn.clicked.connect(self._on_create_update_clicked)
        self.clear_selection_btn.clicked.connect(self._enter_create_mode)
        self.remove_bout_btn.clicked.connect(self._on_remove_bout)
        self.save_btn.clicked.connect(self._on_save)
        self.bout_table.itemSelectionChanged.connect(self._on_table_selection_changed)

        self.media_viewer.frame_changed.connect(self._on_frame_changed)
        self.media_viewer.frame_changed.connect(self.timeline.set_current_frame)
        self.timeline.seek_requested.connect(self.media_viewer.seek_to_frame)

        self.overlay_checkbox.toggled.connect(self.media_viewer.set_overlay_visible)
        self.overlay_font_size_spinbox.valueChanged.connect(
            self.media_viewer.set_overlay_font_size
        )

    # --- Context wiring (mirrors the pose labeling widget) ---

    def set_context(self, label_operations: Optional[BehaviorLabelOperations]):
        self._label_operations = label_operations

    def set_project(self, project: Optional[Project]):
        self._project = project
        self._behaviors = project.behaviors if project else []
        self.behavior_combo.clear()
        if self._behaviors:
            self.behavior_combo.addItems(self._behaviors)
        self.behavior_combo.setCurrentIndex(-1)
        self.timeline.set_project_behaviors(self._behaviors)
        if self._behavior_dataset is not None:
            self._behavior_dataset.behaviors = self._behaviors
        self._load_data()

    def set_media(
        self,
        dataset_operations: Optional[DatasetOperations],
        dataset_name: Optional[str],
        video_path: Optional[Path],
    ):
        self._selected_dataset = (
            dataset_operations.get_dataset(dataset_name)
            if (dataset_operations and dataset_name) else None
        )
        self._dataset_name = dataset_name
        self._video_path = video_path
        self._video_key = video_path.name if video_path else None

        is_valid = all([
            self._label_operations, self._selected_dataset, video_path, self._project
        ])
        self.setEnabled(bool(is_valid) or self._project is not None)

        self.media_viewer.set_media(video_path)
        self.media_viewer.set_overlay_visible(self.overlay_checkbox.isChecked())
        self.media_viewer.set_overlay_font_size(self.overlay_font_size_spinbox.value())

        self._behavior_dataset = None
        if is_valid:
            self._behavior_dataset = self._label_operations.get_behavior_dataset(
                self._dataset_name, raise_if_missing=False
            )
            self._behavior_dataset.behaviors = self._behaviors

            max_frame = max(0, self.media_viewer.total_frames - 1)
            self.start_frame_spinbox.setRange(0, max_frame)
            self.end_frame_spinbox.setRange(0, max_frame)

        self._load_data()

    # --- Data refresh ---

    def _load_data(self):
        self._refresh_timeline()
        self._populate_table()
        self._enter_create_mode()
        self._on_frame_changed(self.media_viewer.get_current_frame())

    def _current_video_bouts(self) -> pd.DataFrame:
        if not self._behavior_dataset or not self._video_key:
            return pd.DataFrame(columns=["video_key", "behavior", "start_frame", "end_frame", "p"])
        return self._behavior_dataset.bouts_for_video(self._video_key)

    def _refresh_timeline(self):
        self.timeline.set_data(
            self._current_video_bouts(),
            self.media_viewer.total_frames,
            self.media_viewer.fps,
        )
        self.timeline.set_current_frame(max(0, self.media_viewer.get_current_frame()))

    def _enter_create_mode(self):
        self.bout_table.clearSelection()
        self.start_frame_spinbox.setValue(0)
        self.end_frame_spinbox.setValue(0)
        self.behavior_combo.setCurrentIndex(-1)
        self.create_update_btn.setText("Create (`C`)")

    def _enter_edit_mode(self, behavior: str, bout: Bout):
        self.start_frame_spinbox.setValue(bout.start_frame)
        self.end_frame_spinbox.setValue(bout.end_frame)
        self.behavior_combo.setCurrentText(behavior)
        self.create_update_btn.setText("Update (`C`)")

    @pyqtSlot()
    def _on_table_selection_changed(self):
        selected_items = self.bout_table.selectedItems()
        if not selected_items:
            self.start_frame_spinbox.setValue(0)
            self.end_frame_spinbox.setValue(0)
            self.behavior_combo.setCurrentIndex(-1)
            self.create_update_btn.setText("Create (`C`)")
            return

        row = selected_items[0].row()
        item_data = self.bout_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        if item_data:
            behavior, bout = item_data
            self._enter_edit_mode(behavior, bout)
            self.media_viewer.seek_to_frame(bout.start_frame)

    @pyqtSlot(int)
    def _on_frame_changed(self, frame: int):
        self.timeline.set_current_frame(max(0, frame))
        if not self._behavior_dataset or not self._video_key or frame < 0:
            self.media_viewer.set_overlay_text("Labels: None")
            return

        active = []
        bouts_df = self._behavior_dataset.bouts
        if not bouts_df.empty:
            active_df = bouts_df[
                (bouts_df["video_key"] == self._video_key)
                & (bouts_df["start_frame"] <= frame)
                & (bouts_df["end_frame"] >= frame)
            ]
            if not active_df.empty:
                active = active_df["behavior"].unique().tolist()

        text = f"Labels: {', '.join(sorted(active))}" if active else "Labels: None"
        self.media_viewer.set_overlay_text(text)

    def _populate_table(self):
        self.bout_table.setRowCount(0)
        video_bouts_df = self._current_video_bouts()
        if video_bouts_df.empty:
            return

        self.bout_table.setRowCount(len(video_bouts_df))
        for i, row in enumerate(video_bouts_df.itertuples()):
            behavior = row.behavior
            bout = Bout(
                start_frame=int(row.start_frame),
                end_frame=int(row.end_frame),
                p=row.p if pd.notna(row.p) else None,
            )
            behavior_item = QTableWidgetItem(behavior)
            behavior_item.setData(Qt.ItemDataRole.UserRole, (behavior, bout))
            self.bout_table.setItem(i, 0, behavior_item)
            self.bout_table.setItem(i, 1, QTableWidgetItem(str(bout.start_frame)))
            self.bout_table.setItem(i, 2, QTableWidgetItem(str(bout.end_frame)))

    # --- Editor actions ---

    def _on_set_start_clicked(self):
        frame = self.media_viewer.get_current_frame()
        if frame >= 0:
            self.start_frame_spinbox.setValue(frame)

    def _on_set_end_clicked(self):
        frame = self.media_viewer.get_current_frame()
        if frame >= 0:
            self.end_frame_spinbox.setValue(frame)

    def _on_create_update_clicked(self):
        if not self._behavior_dataset or not self._video_key:
            return

        start = self.start_frame_spinbox.value()
        end = self.end_frame_spinbox.value()
        behavior = self.behavior_combo.currentText()

        if end < start:
            QMessageBox.warning(self, "Invalid Range", "End frame must be after start frame.")
            return
        if not behavior:
            QMessageBox.warning(self, "No Behavior", "Please select a behavior.")
            return

        selected_items = self.bout_table.selectedItems()
        if selected_items:  # Update existing
            row = selected_items[0].row()
            item_data = self.bout_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
            if not item_data:
                logging.error("Could not find item data for update. Aborting.")
                return
            old_behavior, old_bout = item_data
            bouts_df = self._behavior_dataset.bouts
            row_index = bouts_df[
                (bouts_df["video_key"] == self._video_key)
                & (bouts_df["behavior"] == old_behavior)
                & (bouts_df["start_frame"] == old_bout.start_frame)
                & (bouts_df["end_frame"] == old_bout.end_frame)
            ].index
            if row_index.empty:
                logging.error("Could not find corresponding row in DataFrame to update.")
                return
            self._behavior_dataset.bouts.loc[
                row_index, ["behavior", "start_frame", "end_frame"]
            ] = [behavior, start, end]
        else:  # Create new
            new_row = pd.DataFrame([{
                "video_key": self._video_key,
                "behavior": behavior,
                "start_frame": start,
                "end_frame": end,
                "p": np.nan,
            }])
            self._behavior_dataset.bouts = pd.concat(
                [self._behavior_dataset.bouts, new_row], ignore_index=True
            )

        self._populate_table()
        self._refresh_timeline()
        self._on_frame_changed(self.media_viewer.get_current_frame())
        self.data_changed.emit()
        self._enter_create_mode()

    def _on_remove_bout(self):
        if not self._behavior_dataset:
            return
        selected_items = self.bout_table.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Selection Error",
                                "Please select a bout from the table to remove.")
            return

        row = selected_items[0].row()
        item_data = self.bout_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        if not item_data:
            logging.error("Could not find item data for removal. Aborting.")
            return

        behavior, bout_to_remove = item_data
        bouts_df = self._behavior_dataset.bouts
        row_index = bouts_df[
            (bouts_df["video_key"] == self._video_key)
            & (bouts_df["behavior"] == behavior)
            & (bouts_df["start_frame"] == bout_to_remove.start_frame)
            & (bouts_df["end_frame"] == bout_to_remove.end_frame)
        ].index

        if not row_index.empty:
            self._behavior_dataset.bouts = self._behavior_dataset.bouts.drop(index=row_index)
            self._populate_table()
            self._refresh_timeline()
            self._on_frame_changed(self.media_viewer.get_current_frame())
            self.data_changed.emit()
            self._enter_create_mode()
        else:
            logging.warning("Could not find bout to remove. Table might be out of sync.")

    def _on_save(self):
        if not all([self._label_operations, self._dataset_name, self._video_path]):
            QMessageBox.critical(self, "Error", "Cannot save. Incomplete context.")
            return
        try:
            # Make sure the (possibly newly created) dataset is registered before save.
            self._label_operations.set_behavior_labels(
                self._dataset_name, self._behavior_dataset
            )
            self._label_operations.save_behavior_labels(self._dataset_name)
            self.labels_saved.emit(self._dataset_name, str(self._video_path))
        except Exception as e:
            logging.error(f"Failed to save behavior labels: {e}", exc_info=True)
            QMessageBox.critical(self, "Save Failed", f"Could not save annotations:\n{e}")

    # --- Keyboard shortcuts ---

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        mods = event.modifiers()

        if mods == Qt.KeyboardModifier.ControlModifier and key == Qt.Key.Key_S:
            self._on_save()
            return

        if mods == Qt.KeyboardModifier.NoModifier:
            handlers = {
                Qt.Key.Key_Escape: self._enter_create_mode,
                Qt.Key.Key_Space: self.media_viewer.toggle_play_pause,
                Qt.Key.Key_Left: self.media_viewer._single_step_backward,
                Qt.Key.Key_Right: self.media_viewer._single_step_forward,
                Qt.Key.Key_T: self._on_set_start_clicked,
                Qt.Key.Key_E: self._on_set_end_clicked,
                Qt.Key.Key_C: self._on_create_update_clicked,
                Qt.Key.Key_O: self.overlay_checkbox.toggle,
                Qt.Key.Key_Delete: self._on_remove_bout,
                Qt.Key.Key_N: self.request_select_next_video.emit,
                Qt.Key.Key_M: self.request_select_prev_video.emit,
            }
            if handler := handlers.get(key):
                handler()
                return
        elif mods == Qt.KeyboardModifier.ShiftModifier:
            if key == Qt.Key.Key_Left:
                self.media_viewer._skip_backward()
                return
            elif key == Qt.Key.Key_Right:
                self.media_viewer._skip_forward()
                return

        super().keyPressEvent(event)
