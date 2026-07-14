import logging
from pathlib import Path
from typing import Set, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QSplitter, QComboBox,
    QMessageBox, QAbstractItemView, QTabWidget, QFormLayout, QSpinBox
)

from whisker.core.workspace import Workspace, DatasetType
from whisker.services.pose_estimation.internal.core.utils.sampling import save_manual_frames
from whisker.gui.widgets.media_viewer import MediaViewerWidget
from whisker.gui.workflows.behavior_classification.widgets.probability_plot import ProbabilityPlotPanel
from whisker.gui.signals import MessageBus
from whisker.gui.worker_wrapper import Worker

class FrameSamplingWidget(QWidget):
    request_launch_worker = pyqtSignal(str, object)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._workspace: Optional[Workspace] = None
        self._dataset_name: str = ""
        self._video_rel_path: str = ""
        self._video_path: Optional[Path] = None

        self._saved_frames: Set[int] = set() 
        self._pending_frames: Set[int] = set()
        self._current_subset_name: Optional[str] = None
        self._enable_edits: bool = True

        # Multi-arena state: boxes for the current video and the selected arena
        # index (None = "All arenas", i.e. show every ROI outline, no mask).
        self._arena_boxes: list = []
        self._arena_index: Optional[int] = None

        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # --- Left: Video & Timeline ---
        viewer_container = QWidget()
        viewer_layout = QVBoxLayout(viewer_container)
        viewer_layout.setContentsMargins(0, 0, 0, 0)

        viewer_layout.addWidget(QLabel("<b>Video Frame Sampler View</b>: Sample frames from a video to create a frame subset dataset."))

        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("View Subset:"))
        self.subset_combo = QComboBox()
        top_bar.addWidget(self.subset_combo, 1)

        # Arena selector (multi-arena datasets only). Choosing an arena masks the
        # preview to that arena and saves frames masked, into a per-arena subdir.
        self.arena_widget = QWidget()
        arena_layout = QHBoxLayout(self.arena_widget)
        arena_layout.setContentsMargins(8, 0, 0, 0)
        arena_layout.addWidget(QLabel("Arena:"))
        self.arena_combo = QComboBox()
        self.arena_combo.setToolTip(
            "Pick an arena to sample one animal at a time. Saved frames are "
            "masked to the arena (single mouse) for pose labeling/training."
        )
        arena_layout.addWidget(self.arena_combo)
        self.arena_widget.setVisible(False)
        top_bar.addWidget(self.arena_widget)

        viewer_layout.addLayout(top_bar)

        self.media_viewer = MediaViewerWidget()
        self.timeline = ProbabilityPlotPanel(title_text="SAMPLING TIMELINE")
        self.timeline.setFixedHeight(150)
        self.timeline.toggle_behaviors_btn.setChecked(False)
        self.timeline.sidebar.setVisible(False)

        viewer_layout.addWidget(self.media_viewer, 1)
        viewer_layout.addWidget(self.timeline)
        splitter.addWidget(viewer_container)

        # --- Right: Controls (Tabs) ---
        self.tabs = QTabWidget()
        
        # Tab 1: Manual Sampling
        self.manual_tab = QWidget()
        manual_layout = QVBoxLayout(self.manual_tab)

        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Time", "Frame"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

        manual_layout.addWidget(QLabel("<b>Selected Frames</b>"))
        manual_layout.addWidget(self.table)

        btn_layout = QVBoxLayout()
        self.mark_btn = QPushButton("Mark Current Frame")
        self.remove_btn = QPushButton("Remove Selected")
        self.save_btn = QPushButton("Save to Dataset")

        btn_layout.addWidget(self.mark_btn)
        btn_layout.addWidget(self.remove_btn)
        btn_layout.addSpacing(10)
        btn_layout.addWidget(self.save_btn)
        manual_layout.addLayout(btn_layout)



        self.tabs.addTab(self.manual_tab, "Manual Sampling")
        splitter.addWidget(self.tabs)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

    def _connect_signals(self):
        self.media_viewer.frame_changed.connect(self._on_frame_changed)
        self.timeline.bout_clicked.connect(self.media_viewer.seek_to_frame)
        
        self.mark_btn.clicked.connect(self._mark_current_frame)
        self.remove_btn.clicked.connect(self._remove_selected_frame)
        self.save_btn.clicked.connect(self._save_frames)
        self.subset_combo.currentIndexChanged.connect(self._on_subset_changed)
        self.arena_combo.currentIndexChanged.connect(self._on_arena_changed)
        self.table.cellClicked.connect(self._on_table_clicked)

        MessageBus.get().subscribe("workspace/datasets/refreshed", lambda t, p: self._refresh_subsets_list())

    def set_media(self, workspace: Workspace, dataset_name: str, video_rel_path: str):
        self._workspace = workspace
        self._dataset_name = dataset_name
        self._video_rel_path = video_rel_path

        dataset = workspace.datasets.get(dataset_name)
        if not dataset: return

        self._video_path = Path(dataset.base_data_path) / video_rel_path
        self.media_viewer.set_media(self._video_path)

        self._setup_arenas(dataset, video_rel_path)

        self._saved_frames.clear()
        self._pending_frames.clear()
        self._refresh_subsets_list()

        self.subset_combo.setCurrentIndex(0)
        self._update_ui()

    def _setup_arenas(self, dataset, video_rel_path: str):
        """Populate the arena selector for a multi-arena dataset and show all
        arena ROIs on the video. Hidden for ordinary datasets."""
        video_rel_norm = str(video_rel_path).replace("\\", "/")
        boxes = (
            dataset.multi_arena.boxes_for(video_rel_norm)
            if getattr(dataset, "is_multi_arena", False) else []
        )
        self._arena_boxes = boxes
        self._arena_index = None

        self.arena_combo.blockSignals(True)
        self.arena_combo.clear()
        if boxes:
            self.arena_combo.addItem("All arenas", None)
            for k in range(len(boxes)):
                self.arena_combo.addItem(f"Arena {k + 1}", k)
        self.arena_combo.blockSignals(False)

        self.arena_widget.setVisible(bool(boxes))
        if boxes:
            self.arena_combo.setCurrentIndex(0)  # "All arenas"
            self._apply_arena_overlay()
        else:
            self.media_viewer.clear_arena_boxes()
            self.media_viewer.clear_arena_mask()

    def _apply_arena_overlay(self):
        """Reflect the current arena selection on the video: all outlines for
        'All arenas', or a black-out mask + highlighted outline for one arena."""
        if not self._arena_boxes:
            self.media_viewer.clear_arena_boxes()
            self.media_viewer.clear_arena_mask()
            return
        if self._arena_index is None:
            self.media_viewer.clear_arena_mask()
            self.media_viewer.set_arena_boxes(self._arena_boxes, active_index=None)
        else:
            self.media_viewer.set_arena_mask(self._arena_boxes[self._arena_index])
            self.media_viewer.set_arena_boxes(self._arena_boxes, active_index=self._arena_index)

    def _on_arena_changed(self, _index: int):
        self._arena_index = self.arena_combo.currentData()
        self._apply_arena_overlay()

    def _refresh_subsets_list(self):
        """Finds all FRAME_SUBSET datasets that contain frames from this video."""
        if not self._workspace or not self._video_path: return

        current_text = self.subset_combo.currentText()
        self.subset_combo.blockSignals(True)
        self.subset_combo.clear()
        sample_dataset_name = f"{self._dataset_name} [Manual Samples]"
        self.subset_combo.addItem(sample_dataset_name, sample_dataset_name)

        video_stem = self._video_path.stem

        for ds in self._workspace.datasets.values():
            if ds.name == sample_dataset_name:
                continue

            if ds.type == DatasetType.FRAME_SUBSET:
                # Check if this dataset contains files from our video
                # Files in FRAME_SUBSET are usually "video_stem/frame_XXXX.png"
                has_video = any(self._file_belongs_to_video(f, video_stem) for f in ds.files)
                if has_video:
                    self.subset_combo.addItem(ds.name, ds.name)

        # Restore selection if possible
        idx = self.subset_combo.findText(current_text)
        if idx >= 0:
            self.subset_combo.setCurrentIndex(idx)

        self.subset_combo.blockSignals(False)
        self._on_subset_changed()

    @staticmethod
    def _file_belongs_to_video(rel_file: str, video_stem: str) -> bool:
        """A FRAME_SUBSET file belongs to this video if it lives under the
        video's subdir or any of its per-arena subdirs ("{stem}_arena{k}")."""
        p0 = Path(rel_file).parts[0]
        return p0 == video_stem or p0.startswith(f"{video_stem}_arena")

    def _on_subset_changed(self):
        subset_name = self.subset_combo.currentData()
        self._current_subset_name = subset_name
        self._saved_frames.clear()
        self._pending_frames.clear()

        if subset_name:
            ds = self._workspace.datasets.get(subset_name)
            video_stem = self._video_path.stem

            if ds:
                for f in ds.files:
                    path = Path(f)
                    if self._file_belongs_to_video(f, video_stem):
                        # Parse "frame_XXXXXX.png"
                        try:
                            idx = int(path.stem.split('_')[-1])
                            self._saved_frames.add(idx)
                        except ValueError: pass
                
            self._set_enable_edits("[Manual Samples]" in subset_name)

        self._update_ui()

    def _set_enable_edits(self, enable: bool):
        logging.info(f"Setting enable_edits state to {enable}")
        self._enable_edits = enable
        self.save_btn.setEnabled(enable)
        self.mark_btn.setEnabled(enable)
        self.remove_btn.setEnabled(enable)

        if enable:
            self.mark_btn.setStyleSheet("font-weight: bold; background-color: #d6eaf8;")
            self.save_btn.setStyleSheet(None)
            self.remove_btn.setStyleSheet(None)
        else:
            self.mark_btn.setStyleSheet("font-weight: bold; background-color: #909090;")
            self.save_btn.setStyleSheet("background-color: #909090;")
            self.remove_btn.setStyleSheet("background-color: #909090")

    def _on_frame_changed(self, frame: int):
        self.timeline.set_current_frame(frame)

    def _mark_current_frame(self):
        frame = self.media_viewer.get_current_frame()
        if frame < 0: return

        if frame in self._saved_frames:
            # Already saved, maybe warn? For now just ignore.
            return

        if frame not in self._pending_frames:
            self._pending_frames.add(frame)
            self._update_ui()

    def _remove_selected_frame(self):
        row = self.table.currentRow()
        if row < 0: return

        frame_item = self.table.item(row, 1)
        if not frame_item: return

        frame = int(frame_item.text())

        if frame in self._pending_frames:
            self._pending_frames.remove(frame)
            self._update_ui()
        elif frame in self._saved_frames:
            QMessageBox.information(
                self, "Cannot Remove",
                "This frame is already saved in the dataset. "
                "To remove it, you would need to edit the dataset directly (not implemented here)."
            )

    def _save_frames(self):
        if not self._pending_frames:
            QMessageBox.warning(self, "Empty", "No new frames to save.")
            return

        # For multi-arena datasets, require a specific arena so frames are masked
        # to one animal (otherwise the saved frames would show every arena).
        if self._arena_boxes and self._arena_index is None:
            QMessageBox.warning(
                self, "Select an Arena",
                "This is a multi-arena dataset. Choose a specific arena (not "
                "'All arenas') before saving, so frames are masked to one animal."
            )
            return

        try:
            target_ds = self._current_subset_name
            # If "New Selection" is selected, we assume saving to "{source} [Manual Samples]"
            if not target_ds:
                target_ds = f"{self._dataset_name} [Manual Samples]"

            new_ds_name = save_manual_frames(
                self._workspace,
                self._dataset_name,
                self._video_rel_path,
                list(self._pending_frames),
                target_dataset_name=target_ds,
                arena_index=self._arena_index,
            )

            QMessageBox.information(self, "Success", f"Saved {len(self._pending_frames)} frames to '{new_ds_name}'.")
            MessageBus.get().publish("request/workspace/datasets/refresh")

            # Move pending to saved
            self._saved_frames.update(self._pending_frames)
            self._pending_frames.clear()

            # Select the dataset we just saved to
            idx = self.subset_combo.findData(new_ds_name)
            if idx >= 0:
                self.subset_combo.setCurrentIndex(idx)
            else:
                self._refresh_subsets_list() # Should appear now
                idx = self.subset_combo.findData(new_ds_name)
                if idx >= 0: self.subset_combo.setCurrentIndex(idx)

            self._update_ui()

        except Exception as e:
            logging.error(f"Error saving frames: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save frames: {e}")

    def _update_ui(self):
        # 1. Update Table
        self.table.setRowCount(0)

        # Combine and sort for display
        all_frames = []
        for f in self._saved_frames: all_frames.append((f, "Saved"))
        for f in self._pending_frames: all_frames.append((f, "Pending"))
        all_frames.sort(key=lambda x: x[0])

        self.table.setRowCount(len(all_frames))
        fps = self.media_viewer.fps if self.media_viewer.fps > 0 else 30.0

        for i, (frame, status) in enumerate(all_frames):
            # Time
            seconds = frame / fps
            m, s = divmod(int(seconds), 60)
            time_str = f"{m:02d}:{s:02d}.{int((seconds%1)*100):02d}"

            self.table.setItem(i, 0, QTableWidgetItem(time_str))
            self.table.setItem(i, 1, QTableWidgetItem(str(frame)))

            # Color code
            color = Qt.GlobalColor.white
            if status == "Saved":
                color = QTableWidgetItem("").background().color() # Default
            else:
                color = Qt.GlobalColor.cyan # Highlight pending
                self.table.item(i, 0).setBackground(color)
                self.table.item(i, 1).setBackground(color)

        # 2. Update Timeline
        # We use ProbabilityPlotWidget by creating a Ground Truth DataFrame
        import pandas as pd
        num_frames = self.media_viewer.total_frames
        if num_frames <= 0: return

        behaviors = ["Saved", "Pending"]
        gt_df = pd.DataFrame(0.0, index=pd.RangeIndex(num_frames), columns=behaviors)
        
        for f in self._saved_frames:
            if 0 <= f < num_frames:
                gt_df.loc[f, "Saved"] = 1.0
        
        for f in self._pending_frames:
            if 0 <= f < num_frames:
                gt_df.loc[f, "Pending"] = 1.0

        self.timeline.plot_probabilities(
            probs_df=None,
            gt_df=gt_df,
            fps=self.media_viewer.fps
        )
        self.timeline.set_current_frame(self.media_viewer.get_current_frame())

    def _on_table_clicked(self, row, col):
        frame_item = self.table.item(row, 1)
        if frame_item:
            self.media_viewer.seek_to_frame(int(frame_item.text()))
