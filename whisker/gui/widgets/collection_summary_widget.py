import os
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from PyQt6.QtCore import pyqtSignal, Qt, QObject, QRunnable, QThreadPool, pyqtSlot
from PyQt6.QtGui import QPixmap, QResizeEvent
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QGridLayout,
    QSlider, QGroupBox, QTableWidget, QHeaderView, QTableWidgetItem,
    QFrame, QPushButton, QSplitter, QFormLayout, QSpinBox, QComboBox, QMessageBox,
    QLineEdit
)

from whisker.core.workspace import Workspace
from whisker.core.utils import generate_video_thumbnail
from whisker.gui.constants import VIDEO_EXTENSIONS
from .scalable_image_label import ScalableImageLabel
from whisker.core.topics import SubsampleParams, SamplingTechnique
from whisker.gui.worker_wrapper import Worker
from whisker.gui.signals import MessageBus

class ClickableLabel(ScalableImageLabel):
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

class ThumbnailLoader(QRunnable):
    class Signals(QObject):
        finished = pyqtSignal(str, QPixmap)

    def __init__(self, file_path: Path, cache_dir: Path):
        super().__init__()
        self.file_path = file_path
        self.cache_dir = cache_dir
        self.signals = self.Signals()

    @pyqtSlot()
    def run(self):
        is_video = self.file_path.suffix.lower() in VIDEO_EXTENSIONS
        pixmap = QPixmap()
        if is_video:
            cache_file = generate_video_thumbnail(self.file_path, self.cache_dir)
            if cache_file and cache_file.exists():
                pixmap = QPixmap(str(cache_file))
        else:
            if self.file_path.exists():
                pixmap = QPixmap(str(self.file_path))
        self.signals.finished.emit(str(self.file_path), pixmap)

class NumericTableWidgetItem(QTableWidgetItem):
    def __lt__(self, other):
        try:
            return int(self.text()) < int(other.text())
        except (ValueError, TypeError):
            return super().__lt__(other)

class NumericFloatTableWidgetItem(QTableWidgetItem):
    def __lt__(self, other):
        try:
            return float(self.text()) < float(other.text())
        except (ValueError, TypeError):
            return super().__lt__(other)

class CollectionSummaryWidget(QWidget):
    media_selected = pyqtSignal(Path)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._cache_dir: Optional[Path] = None
        self._workspace: Optional[Workspace] = None
        self._current_dataset_name: Optional[str] = None
        self._sampling_name_edited = False
        self._subsample_name_edited = False

        self._thumbnail_widgets: Dict[str, ClickableLabel] = {}
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(
            QThreadPool.globalInstance().maxThreadCount() // 2 or 1
        )

        self._all_file_paths: List[str] = []
        self._items_per_page = 100
        self._current_page = 0
        self._total_pages = 0

        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)

        summary_box = QGroupBox("Collection Summary")
        self.summary_layout = QVBoxLayout(summary_box)
        self.summary_text = QLabel()
        self.summary_text.setWordWrap(True)
        self.summary_layout.addWidget(self.summary_text)

        self._create_sampling_panel()
        self.summary_layout.addWidget(self.sampling_panel)
        
        self._create_subsampling_panel()
        self.summary_layout.addWidget(self.subsampling_panel)

        self.behavior_totals_table = QTableWidget()
        self.behavior_totals_table.setVisible(False)
        self.summary_layout.addWidget(self.behavior_totals_table)

        self.video_bout_table = QTableWidget()
        self.video_bout_table.setVisible(False)
        self.summary_layout.addWidget(self.video_bout_table)

        gallery_box = QGroupBox("Thumbnail Gallery")
        gallery_layout = QVBoxLayout(gallery_box)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(summary_box)
        splitter.addWidget(gallery_box)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)
        self._create_gallery_controls(gallery_layout)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        gallery_layout.addWidget(self.scroll_area)

        self.gallery_widget = QWidget()
        self.gallery_layout = QGridLayout(self.gallery_widget)
        self.scroll_area.setWidget(self.gallery_widget)

    def _create_sampling_panel(self):
        self.sampling_panel = QGroupBox("Sample Frames (Video)")
        sampling_layout = QFormLayout(self.sampling_panel)

        self.num_frames_spinbox = QSpinBox()
        self.num_frames_spinbox.setRange(1, 1000)
        self.num_frames_spinbox.setValue(20)
        sampling_layout.addRow("Frames per video:", self.num_frames_spinbox)

        self.sampling_combo = QComboBox()
        self.sampling_combo.addItems([
            "Uniform", 
            "K-Means (Visual Diversity)", 
            "K-Means (Pose/Morphological)"
        ])
        sampling_layout.addRow("Technique:", self.sampling_combo)

        self.sampling_name_edit = QLineEdit()
        self.sampling_name_edit.setPlaceholderText("Output dataset name...")
        self.sampling_name_edit.textEdited.connect(self._on_sampling_name_edited)
        sampling_layout.addRow("Output Name:", self.sampling_name_edit)

        self.sample_frames_btn = QPushButton("Sample Frames")
        self.sample_frames_btn.clicked.connect(self._on_sample_frames_clicked)
        sampling_layout.addRow(self.sample_frames_btn)

        self.sampling_panel.setVisible(False)

    def _create_subsampling_panel(self):
        self.subsampling_panel = QGroupBox("Subsample Dataset")
        layout = QFormLayout(self.subsampling_panel)

        self.subsample_spinbox = QSpinBox()
        self.subsample_spinbox.setRange(1, 999999)
        layout.addRow("Total Frames to Keep:", self.subsample_spinbox)

        self.subsample_combo = QComboBox()
        self.subsample_combo.addItems([
            "Uniform", 
            "K-Means (Visual Diversity)", 
            "K-Means (Pose/Morphological)"
        ])
        layout.addRow("Technique:", self.subsample_combo)

        self.subsample_name_edit = QLineEdit()
        self.subsample_name_edit.setPlaceholderText("Output dataset name...")
        self.subsample_name_edit.textEdited.connect(self._on_subsample_name_edited)
        layout.addRow("Output Name:", self.subsample_name_edit)

        self.subsample_btn = QPushButton("Create Subsample")
        self.subsample_btn.clicked.connect(self._on_subsample_clicked)
        layout.addRow(self.subsample_btn)

        self.subsampling_panel.setVisible(False)

    def _on_sampling_name_edited(self):
        self._sampling_name_edited = True

    def _on_subsample_name_edited(self):
        self._subsample_name_edited = True

    def _update_dynamic_names(self):
        if not self._current_dataset_name: return

        if not self._sampling_name_edited:
            tech = self.sampling_combo.currentText()
            num = self.num_frames_spinbox.value()
            self.sampling_name_edit.setText(f"{self._current_dataset_name} [{tech} {num}]")

        if not self._subsample_name_edited:
            tech = self.subsample_combo.currentText()
            num = self.subsample_spinbox.value()
            self.subsample_name_edit.setText(f"{self._current_dataset_name} [{tech} {num}]")

    def _on_sample_frames_clicked(self):
        if not self._workspace or not self._current_dataset_name: return

        from whisker.services.pose_estimation.internal.workers.sampler import SamplerJob
        worker = Worker(
            SamplerJob(
                workspace=self._workspace,
                sampling_params=SubsampleParams(
                    source_dataset_name=self._current_dataset_name,
                    num_frames=self.num_frames_spinbox.value(),
                    technique=SamplingTechnique(self.sampling_combo.currentText()),
                    target_dataset_name=self.sampling_name_edit.text().strip() or None
                )
            )
        )
        worker.signals.finished.connect(self._on_sampling_finished)
        worker.signals.error.connect(self._on_sampling_error)

        self.sample_frames_btn.setEnabled(False)
        self.sample_frames_btn.setText("Sampling...")
        self.thread_pool.start(worker)

    def _on_sampling_finished(self, new_dataset_name: str):
        self.sample_frames_btn.setEnabled(True)
        self.sample_frames_btn.setText("Sample Frames")
        QMessageBox.information(self, "Success", f"Successfully created new dataset: '{new_dataset_name}'")
        MessageBus.get().publish("request/workspace/datasets/refresh")

    def _on_sampling_error(self, error_message: str):
        self.sample_frames_btn.setEnabled(True)
        self.sample_frames_btn.setText("Sample Frames")
        QMessageBox.critical(self, "Sampling Failed", f"An error occurred during sampling:\n{error_message}")

    def _on_subsample_clicked(self):
        if not self._workspace or not self._current_dataset_name: return

        from whisker.core.workers.subsample_dataset import SubsampleDatasetJob
        worker = Worker(
            SubsampleDatasetJob(
                workspace=self._workspace, 
                params=SubsampleParams(
                    source_dataset_name=self._current_dataset_name,
                    num_frames=self.subsample_spinbox.value(),
                    technique=SamplingTechnique(self.subsample_combo.currentText()),
                    target_dataset_name=self.subsample_name_edit.text().strip() or None
                )
            )
        )
        worker.signals.finished.connect(self._on_subsampling_finished)
        worker.signals.error.connect(self._on_subsampling_error)

        self.subsample_btn.setEnabled(False)
        self.subsample_btn.setText("Subsampling...")
        self.thread_pool.start(worker)

    def _on_subsampling_finished(self, new_dataset_name: str):
        self.subsample_btn.setEnabled(True)
        self.subsample_btn.setText("Create Subsample")
        QMessageBox.information(self, "Success", f"Successfully created subsampled dataset: '{new_dataset_name}'")
        MessageBus.get().publish("request/workspace/datasets/refresh")

    def _on_subsampling_error(self, error_message: str):
        self.subsample_btn.setEnabled(True)
        self.subsample_btn.setText("Create Subsample")
        QMessageBox.critical(self, "Subsampling Failed", f"An error occurred:\n{error_message}")

    def _create_gallery_controls(self, parent_layout: QVBoxLayout):
        controls_layout = QVBoxLayout()
        controls_layout.setSpacing(10)

        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Thumbnail Size:"))
        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setRange(80, 400)
        self.size_slider.setValue(150)
        slider_layout.addWidget(self.size_slider)
        controls_layout.addLayout(slider_layout)

        self.pagination_widget = QWidget()
        pagination_layout = QHBoxLayout(self.pagination_widget)
        pagination_layout.setContentsMargins(0, 0, 0, 0)
        self.prev_button = QPushButton("< Previous")
        self.next_button = QPushButton("Next >")
        self.page_label = QLabel()
        pagination_layout.addWidget(self.prev_button)
        pagination_layout.addStretch()
        pagination_layout.addWidget(self.page_label)
        pagination_layout.addStretch()
        pagination_layout.addWidget(self.next_button)
        controls_layout.addWidget(self.pagination_widget)

        parent_layout.addLayout(controls_layout)

    def _connect_signals(self):
        self.num_frames_spinbox.valueChanged.connect(self._update_total_sample_frames_text)
        self.size_slider.valueChanged.connect(self._resize_thumbnails)
        self.prev_button.clicked.connect(self._go_to_prev_page)
        self.next_button.clicked.connect(self._go_to_next_page)
        
        # Wire up dynamic name updates
        self.num_frames_spinbox.valueChanged.connect(self._update_dynamic_names)
        self.sampling_combo.currentTextChanged.connect(self._update_dynamic_names)
        self.subsample_spinbox.valueChanged.connect(self._update_dynamic_names)
        self.subsample_combo.currentTextChanged.connect(self._update_dynamic_names)

    def set_workspace(self, workspace: Optional[Workspace]):
        self._workspace = workspace
        if workspace:
            self._cache_dir = workspace.base_dir / ".whisker_cache" / "thumbnails"
        else:
            self._cache_dir = None

    def _reset_gallery(self):
        self.thread_pool.clear()
        self._thumbnail_widgets.clear()

        if self.gallery_widget:
            self.gallery_widget.deleteLater()

        self.gallery_widget = QWidget()
        self.gallery_layout = QGridLayout(self.gallery_widget)
        self.gallery_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.scroll_area.setWidget(self.gallery_widget)
    
    def _update_total_sample_frames_text(self):
        btn_text = self.sample_frames_btn.text()
        btn_text = re.sub(r' \[Makes \d+ frames\]', '', btn_text)
        
        if self._workspace and self._current_dataset_name:
            num_videos = len(self._all_file_paths)
            frames_per_video = self.num_frames_spinbox.value()
            total_frames = frames_per_video * num_videos
            btn_text += f" [Makes {total_frames} frames]"
        
        self.sample_frames_btn.setText(btn_text)

    def set_image_collection_data(
        self, title: str, file_paths: List[str], labeled_files_count: int, dataset_name: str
    ):
        self._reset_gallery()
        self.summary_text.clear()
        self.video_bout_table.setVisible(False)
        self.sampling_panel.setVisible(False)
        self.subsampling_panel.setVisible(True)

        self._current_dataset_name = dataset_name
        self._sampling_name_edited = False
        self._subsample_name_edited = False
        total_files = len(file_paths)
        
        self.subsample_spinbox.setMaximum(max(1, total_files))
        self.subsample_spinbox.setValue(min(100, max(1, total_files // 2)))
        self._update_dynamic_names()
        
        summary = (
            f"<b>{title} ({total_files})</b><br>"
            f"{labeled_files_count} of {total_files} images have pose labels."
        )
        self.summary_text.setText(summary)
        self._set_files_for_gallery(file_paths)

    def set_video_collection_data(
        self, dataset, behaviors: Optional[List[str]] = None,
        per_video_counts: Optional[Dict[str, Dict[str, int]]] = None,
        behavior_totals: Optional[Dict[str, Dict]] = None,
        show_sampling_panel: bool = True
    ):
        self._reset_gallery()
        self.summary_text.clear()
        self.sampling_panel.setVisible(show_sampling_panel)
        self.subsampling_panel.setVisible(False)
        self._current_dataset_name = dataset.name
        self._sampling_name_edited = False
        self._subsample_name_edited = False

        total_files = len(dataset.files)
        title = f"{dataset.name} ({dataset.type.value})"

        if behaviors and per_video_counts:
            labeled_videos_count = sum(1 for counts in per_video_counts.values() if sum(counts.values()) > 0)
            summary = (
                f"<b>{title} ({total_files})</b><br>"
                f"{labeled_videos_count} of {total_files} videos have behavior labels."
            )
            self.video_bout_table.setVisible(True)
            self._populate_bout_table(behaviors, per_video_counts)
        else:
            summary = f"<b>{title} ({total_files})</b><br>Video Collection"
            self.video_bout_table.setVisible(False)

        if behavior_totals:
            self.behavior_totals_table.setVisible(True)
            self._populate_behavior_totals_table(behavior_totals)
        else:
            self.behavior_totals_table.setVisible(False)

        self.summary_text.setText(summary)

        file_list = [os.path.join(dataset.base_data_path, f) for f in dataset.files]
        self._set_files_for_gallery(file_list)
        self._update_total_sample_frames_text()
        self._update_dynamic_names()

    def _set_files_for_gallery(self, file_paths: List[str]):
        self._all_file_paths = file_paths
        self._current_page = 0
        item_count = len(self._all_file_paths)

        self.pagination_widget.setVisible(False)
        if item_count == 0:
            self._total_pages = 0
            return

        self._total_pages = (item_count + self._items_per_page - 1) // self._items_per_page
        self.pagination_widget.setVisible(self._total_pages > 1)
        self._render_current_page()

    def _render_current_page(self):
        start_index = self._current_page * self._items_per_page
        end_index = min(start_index + self._items_per_page, len(self._all_file_paths))
        page_files = self._all_file_paths[start_index:end_index]

        if not self._cache_dir:
            logging.warning("Cannot render thumbnails; cache directory is not set.")
            return

        for path_str in page_files:
            path = Path(path_str)
            thumbnail = ClickableLabel()
            thumbnail.setText("Loading...")
            thumbnail.setAlignment(Qt.AlignmentFlag.AlignCenter)
            thumbnail.setStyleSheet("background-color: #EEE; border: 1px solid #CCC;")
            thumbnail.setToolTip(path.name)
            thumbnail.clicked.connect(lambda p=path: self.media_selected.emit(p))
            self._thumbnail_widgets[str(path)] = thumbnail

            worker = ThumbnailLoader(path, self._cache_dir)
            worker.signals.finished.connect(self._on_thumbnail_loaded)
            self.thread_pool.start(worker)

        self._resize_thumbnails()
        self._update_pagination_controls()

    def _update_pagination_controls(self):
        if self._total_pages <= 1:
            self.page_label.setText("")
        else:
            self.page_label.setText(f"Page {self._current_page + 1} of {self._total_pages}")

        self.prev_button.setEnabled(self._current_page > 0)
        self.next_button.setEnabled(self._current_page < self._total_pages - 1)

    def _go_to_prev_page(self):
        if self._current_page > 0:
            self._current_page -= 1
            self._reset_gallery()
            self._render_current_page()

    def _go_to_next_page(self):
        if self._current_page < self._total_pages - 1:
            self._current_page += 1
            self._reset_gallery()
            self._render_current_page()

    @pyqtSlot(str, QPixmap)
    def _on_thumbnail_loaded(self, path_str: str, pixmap: QPixmap):
        if path_str in self._thumbnail_widgets:
            label = self._thumbnail_widgets[path_str]
            if not pixmap.isNull():
                label.setPixmap(pixmap)
                label.setText("")
            else:
                label.setText("Error")
                label.setStyleSheet("background-color: #FDD; border: 1px solid #DCC;")

    def _resize_thumbnails(self):
        if not self._thumbnail_widgets: return

        size = self.size_slider.value()
        max_cols = max(1, self.gallery_widget.width() // (size + 10))

        row, col = 0, 0
        for thumb_widget in self._thumbnail_widgets.values():
            thumb_widget.setFixedSize(size, size)
            self.gallery_layout.addWidget(thumb_widget, row, col)
            col += 1
            if col >= max_cols:
                col = 0
                row += 1

    def resizeEvent(self, event: QResizeEvent | None):
        super().resizeEvent(event)
        self._resize_thumbnails()

    def _populate_behavior_totals_table(self, behavior_totals: Dict[str, Dict]):
        from PyQt6.QtGui import QFont
        # Disable sorting before inserting — Qt physically reorders rows on each
        # setItem() call when sorting is active, corrupting row indices mid-loop.
        # Sorting is intentionally left OFF so the TOTAL row stays pinned at the bottom.
        self.behavior_totals_table.setSortingEnabled(False)
        self.behavior_totals_table.clearContents()

        sorted_behaviors = sorted(behavior_totals.keys())
        row_count = len(sorted_behaviors) + 1  # +1 for TOTAL row
        headers = ["Behavior", "# Bouts", "Frames Labeled", "Time Labeled (min)"]

        self.behavior_totals_table.setColumnCount(len(headers))
        self.behavior_totals_table.setHorizontalHeaderLabels(headers)
        self.behavior_totals_table.setRowCount(row_count)

        grand_bouts = 0
        grand_frames = 0
        grand_time = 0.0

        for i, behavior in enumerate(sorted_behaviors):
            stats = behavior_totals[behavior]
            total_bouts = stats['total_bouts']
            total_frames = stats['total_frames']
            total_time = stats['total_time_min']
            grand_bouts += total_bouts
            grand_frames += total_frames
            grand_time += total_time

            self.behavior_totals_table.setItem(i, 0, QTableWidgetItem(behavior))

            bouts_item = NumericTableWidgetItem(str(total_bouts))
            bouts_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.behavior_totals_table.setItem(i, 1, bouts_item)

            frames_item = NumericTableWidgetItem(str(total_frames))
            frames_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.behavior_totals_table.setItem(i, 2, frames_item)

            time_item = NumericFloatTableWidgetItem(f"{total_time:.2f}")
            time_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.behavior_totals_table.setItem(i, 3, time_item)

        # Bold TOTAL row
        bold_font = QFont()
        bold_font.setBold(True)
        total_row = len(sorted_behaviors)
        for col, (val, cls) in enumerate([
            ("TOTAL", QTableWidgetItem),
            (str(grand_bouts), NumericTableWidgetItem),
            (str(grand_frames), NumericTableWidgetItem),
            (f"{grand_time:.2f}", NumericFloatTableWidgetItem),
        ]):
            item = cls(val)
            item.setFont(bold_font)
            if col > 0:
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.behavior_totals_table.setItem(total_row, col, item)

        self.behavior_totals_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.behavior_totals_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.behavior_totals_table.resizeRowsToContents()
        header_h = self.behavior_totals_table.horizontalHeader().height()
        rows_h = sum(self.behavior_totals_table.rowHeight(r) for r in range(row_count))
        self.behavior_totals_table.setMaximumHeight(header_h + rows_h + 4)

    def _populate_bout_table(
        self, behaviors: List[str], per_video_counts: Dict[str, Dict[str, int]]
    ):
        self.video_bout_table.setSortingEnabled(False)
        self.video_bout_table.clearContents()

        sorted_behaviors = sorted(behaviors)
        headers = ["Video"] + sorted_behaviors + ["Total"]
        self.video_bout_table.setColumnCount(len(headers))
        self.video_bout_table.setHorizontalHeaderLabels(headers)
        self.video_bout_table.setRowCount(len(per_video_counts))

        for i, (video_path_str, counts_dict) in enumerate(sorted(per_video_counts.items())):
            video_name_item = QTableWidgetItem(Path(video_path_str).name)
            self.video_bout_table.setItem(i, 0, video_name_item)
            row_total = 0
            for j, behavior_name in enumerate(sorted_behaviors):
                count = counts_dict.get(behavior_name, 0)
                row_total += count
                count_item = NumericTableWidgetItem(str(count))
                count_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.video_bout_table.setItem(i, j + 1, count_item)
            total_item = NumericTableWidgetItem(str(row_total))
            total_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.video_bout_table.setItem(i, len(headers) - 1, total_item)

        self.video_bout_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.video_bout_table.setSortingEnabled(True)
        self.video_bout_table.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        self.video_bout_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)