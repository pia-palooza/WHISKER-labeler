from pathlib import Path
from typing import Optional

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QStackedWidget,
)

from whisker.core.workspace import Workspace, Project

from whisker.gui.widgets import data_explorer
from whisker.gui.widgets.collection_summary_widget import CollectionSummaryWidget
from whisker.gui.tabs.base_tab import BaseTab
from whisker.gui.signals import MessageBus
from ..widgets.behavior_labeling import BehaviorsLabelingWidget

class LabelingBehaviorsTab(BaseTab):
    """
    A controller widget that switches between a summary view for a video
    collection and a detailed labeling view for a single video. It also manages
    the selected project context for behavior definitions.
    """

    labels_saved = pyqtSignal(str, str)
    media_selected = pyqtSignal(Path)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._current_selection: Optional[data_explorer.Selection] = None

        self._init_ui()
        self._connect_signals()
        self.view_stack.setCurrentWidget(self.placeholder)

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)

        self.view_stack = QStackedWidget()
        main_layout.addWidget(self.view_stack, 1)

        self.placeholder = QLabel(
            "Select a project, then a video from the Data Explorer."
        )
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setWordWrap(True)
        self.view_stack.addWidget(self.placeholder)

        self.labeling_widget = BehaviorsLabelingWidget()
        self.view_stack.addWidget(self.labeling_widget)

        self.summary_widget = CollectionSummaryWidget()
        self.view_stack.addWidget(self.summary_widget)

    def _connect_signals(self):
        self.labeling_widget.labels_saved.connect(self._on_labels_saved)
        self.labeling_widget.data_changed.connect(self._on_data_changed)
        self.summary_widget.media_selected.connect(self.media_selected)

    def _on_labels_saved(self, dataset_name, media_path):
        """When labels are saved, the tab is no longer dirty."""
        self.set_dirty(False)
        MessageBus.get().publish("request/workspace/labels/refresh")
        self.labels_saved.emit(dataset_name, media_path)

    def _on_data_changed(self):
        """When data changes, the tab becomes dirty."""
        self.set_dirty(True)

    def set_workspace(self, workspace: Optional[Workspace]):
        super().set_workspace(workspace)
        self.summary_widget.set_workspace(workspace)
        self._current_selection = None
        self.view_stack.setCurrentWidget(self.placeholder)
        self.set_dirty(False)

    def set_project(self, project: Optional[Project]):
        super().set_project(project)
        self.labeling_widget.set_project(self._project)
        self.set_dirty(False)
        self._update_view()

    def on_data_explorer_item_selected(self, selection: data_explorer.Selection):
        self._current_selection = selection
        self._update_view()

    def _update_view(self):
        """Main logic to decide which widget to show based on current state."""
        selection = self._current_selection

        if self.is_dirty():
            # This check is handled by the main window before switching tabs.
            pass

        # Condition 1: No selection or no project selected yet
        if not selection or not self._project or not self._workspace:
            self.view_stack.setCurrentWidget(self.placeholder)
            return

        # Condition 2: A video collection is selected -> show summary
        if selection.type == data_explorer.ItemTypeEnum.DATASET_BASE:
            dataset_name = selection.item[1]
            self.set_dirty(False)
            self._show_collection_summary(dataset_name, self._project)
            return

        # Condition 3: A video file is selected -> show labeling widget
        if selection.type == data_explorer.ItemTypeEnum.DATASET_VIDEO:
            dataset_name = selection.item[1]
            dataset = self._workspace.datasets.get(dataset_name)
            video_path = Path(dataset.base_data_path) / selection.item[-1]

            self.set_dirty(False)
            self.labeling_widget.set_media(self._workspace, dataset_name, video_path)
            self.view_stack.setCurrentWidget(self.labeling_widget)
            return

        # Fallback to placeholder
        self.set_dirty(False)
        self.view_stack.setCurrentWidget(self.placeholder)

    def _show_collection_summary(self, dataset_name: str, project: Project):
        if not self._workspace:
            return

        dataset = self._workspace.datasets.get(dataset_name)
        if not dataset:
            self.view_stack.setCurrentWidget(self.placeholder)
            return

        # Pre-fetch labeled video keys using fast-path
        labeled_video_keys = self._workspace.get_behavior_labeled_video_keys(dataset_name)
        all_behaviors = project.behaviors
        
        # Only load the full labels if we actually have labeled videos to summarize
        labels = None
        all_bouts_df = None
        if labeled_video_keys:
            labels = self._workspace.get_behavior_labels(dataset_name)
            all_bouts_df = labels.bouts
        
        per_video_counts = {}
        for video_path_str in dataset.files:
            video_name = Path(video_path_str).name
            counts_for_this_video = {b: 0 for b in all_behaviors}

            if all_bouts_df is not None and not all_bouts_df.empty and video_name in labeled_video_keys:
                video_bouts_df = all_bouts_df[all_bouts_df["video_key"] == video_name]
                if not video_bouts_df.empty:
                    video_counts = video_bouts_df.groupby("behavior").size().to_dict()
                    for b, c in video_counts.items():
                        if b in counts_for_this_video:
                            counts_for_this_video[b] = c

            per_video_counts[video_name] = counts_for_this_video

        # Aggregate per-behavior stats from GT labels
        behavior_totals = {}
        if all_bouts_df is not None and not all_bouts_df.empty:
            # Read FPS from the first video (all videos in a dataset share the same FPS)
            fps = 30.0
            if dataset.files:
                try:
                    import cv2
                    first_video = Path(dataset.base_data_path) / dataset.files[0]
                    cap = cv2.VideoCapture(str(first_video))
                    if cap.isOpened():
                        f = cap.get(cv2.CAP_PROP_FPS)
                        if f > 0:
                            fps = f
                    cap.release()
                except Exception:
                    pass

            for behavior in sorted(all_bouts_df["behavior"].unique()):
                bouts = all_bouts_df[all_bouts_df["behavior"] == behavior]
                total_frames = int((bouts["end_frame"] - bouts["start_frame"]).sum())
                behavior_totals[behavior] = {
                    "total_bouts": len(bouts),
                    "total_frames": total_frames,
                    "total_time_min": total_frames / fps / 60.0,
                }

        self.summary_widget.set_video_collection_data(
            dataset, all_behaviors, per_video_counts, behavior_totals or None,
            show_sampling_panel=False
        )
        self.view_stack.setCurrentWidget(self.summary_widget)
