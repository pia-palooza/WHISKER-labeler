from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QStackedWidget,
)

from whisker.core.workspace import Workspace, Project, DatasetType
from whisker.gui.tabs.base_tab import BaseTab
from whisker.gui.widgets import (
    data_explorer,
    CollectionSummaryWidget,
)
from whisker.gui.signals import MessageBus
from whisker.gui.widgets.frame_sampling import FrameSamplingWidget
from ..widgets.pose_labeling import PoseLabelingWidget

class LabelingPosesTab(BaseTab):
    labels_saved = pyqtSignal(str, str)
    media_selected = pyqtSignal(Path)
    request_select_prev_image = pyqtSignal()
    request_select_next_image = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.view_stack = QStackedWidget()
        main_layout.addWidget(self.view_stack)

        # --- View 0: Placeholder for when nothing is selected ---
        self.placeholder_widget = QLabel(
            "Select an image or frame collection from the Data Explorer to begin."
        )
        self.placeholder_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder_widget.setWordWrap(True)
        self.view_stack.addWidget(self.placeholder_widget)

        # --- View 1: Main labeling widget for a single image ---
        self.labeling_widget = PoseLabelingWidget()
        self.view_stack.addWidget(self.labeling_widget)

        # --- View 2: Summary widget for a collection ---
        self.summary_widget = CollectionSummaryWidget()
        self.view_stack.addWidget(self.summary_widget)

        # --- View 3: Frame Sampling (New) ---
        self.sampling_widget = FrameSamplingWidget()
        self.view_stack.addWidget(self.sampling_widget)

        self.view_stack.setCurrentWidget(self.placeholder_widget)
        self._connect_signals()

    def _connect_signals(self):
        self.labeling_widget.labels_saved.connect(self._on_labels_saved)
        self.labeling_widget.data_changed.connect(self._on_data_changed)
        self.labeling_widget.request_select_prev_image.connect(self.request_select_prev_image)
        self.labeling_widget.request_select_next_image.connect(self.request_select_next_image)
        self.summary_widget.media_selected.connect(self.media_selected)
        self.sampling_widget.request_launch_worker.connect(self.request_launch_worker)

    def _on_labels_saved(self, dataset_name, media_path):
        """When labels are saved, the tab is no longer dirty."""
        self.set_dirty(False)
        MessageBus.get().publish("request/workspace/labels/refresh")
        # Forward the signal to the parent tab.
        self.labels_saved.emit(dataset_name, media_path)

    def _on_data_changed(self):
        """When data changes, the tab becomes dirty."""
        self.set_dirty(True)

    def set_workspace(self, workspace: Optional[Workspace]):
        """Passes the workspace context down to child widgets."""
        super().set_workspace(workspace)
        self.summary_widget.set_workspace(workspace)
        if workspace:
            self.labeling_widget.set_context(workspace.pose_labels, workspace.pose_predictions)
        else:
            self.labeling_widget.set_context(None, None)
        self.set_dirty(False)

    def set_project(self, project: Optional[Project]):
        """Passes the project context down to the main labeling widget."""
        super().set_project(project)
        self.labeling_widget.set_project(project)
        self.set_dirty(False)

    def on_data_explorer_item_selected(self, selection: data_explorer.Selection):
        """
        Public slot to update the view based on the selected item.
        This now correctly handles files vs. different kinds of collections.
        """
        if self.is_dirty():
            # This check is now handled by the main window before switching tabs,
            # so we can simply proceed with the item selection logic.
            pass

        if not self._workspace:
            self.view_stack.setCurrentWidget(self.placeholder_widget)
            return

        # Case 1: A collection-like item is selected (a whole dataset or a video sub-group)
        if selection.type in (
            data_explorer.ItemTypeEnum.DATASET_BASE,
            data_explorer.ItemTypeEnum.DATASET_VIDEO_FRAME_SUBSET,
        ):
            self._show_summary(selection)
        # Case 2: A single image/frame is selected
        elif selection.type in (
            data_explorer.ItemTypeEnum.DATASET_IMAGE,
            data_explorer.ItemTypeEnum.DATASET_VIDEO_FRAME,
        ):
            self._show_image(selection)

        # Case 3: Video Selection (Trigger Sampling)
        elif selection.type == data_explorer.ItemTypeEnum.DATASET_VIDEO:
            self._show_sampling(selection)

        # Otherwise, show the placeholder
        else:
            self.view_stack.setCurrentWidget(self.placeholder_widget)

    def _show_summary(self, selection: data_explorer.Selection):
        """
        Calculates stats and populates the summary widget for any collection
        or sub-collection represented by a tree item.
        """
        if not self._workspace:
            return

        dataset_name = selection.item[1]
        dataset = self._workspace.datasets.get(dataset_name)
        if not dataset:
            self.view_stack.setCurrentWidget(self.placeholder_widget)
            return

        if dataset.type == DatasetType.VIDEO_COLLECTION and selection.type == data_explorer.ItemTypeEnum.DATASET_BASE:
             self.summary_widget.set_video_collection_data(dataset)
             self.view_stack.setCurrentWidget(self.summary_widget)
             return

        files_in_scope = []
        title = ""

        if selection.type == data_explorer.ItemTypeEnum.DATASET_BASE:
            # Summary for the entire dataset
            title = f"{dataset.name} ({dataset.type.value})"
            files_in_scope = [
                str(Path(dataset.base_data_path) / f) for f in dataset.files
            ]
            labeled_in_subset = self._workspace.get_labeled_image_count(dataset_name)
        elif selection.type == data_explorer.ItemTypeEnum.DATASET_VIDEO_FRAME_SUBSET:
            # Summary for a specific video's frames within a Frame Subset
            video_name = selection.item[2]
            title = f"Frames from '{video_name}' (in {dataset.name})"
            # Filter files to only those in the selected video's subdirectory
            files_in_scope = [
                str(Path(dataset.base_data_path) / f)
                for f in dataset.files
                if Path(f).parts[0] == video_name
            ]
            labeled_in_subset = self._workspace.get_labeled_image_count(
                dataset_name, video_name
            )
        else:
            labeled_in_subset = 0

        if labeled_in_subset > len(files_in_scope):
            raise RuntimeError(
                f"Number of labeled images in subset ({labeled_in_subset}) is greater "
                f"than the number of images in the scope ({len(files_in_scope)})"
            )

        self.summary_widget.set_image_collection_data(
            title, files_in_scope, labeled_in_subset, dataset_name
        )
        self.view_stack.setCurrentWidget(self.summary_widget)

    def _show_image(self, selection: data_explorer.Selection):
        """
        Shows the image and associated keypoints for a single image.
        """
        if not self._workspace:
            self.view_stack.setCurrentWidget(self.placeholder_widget)
            return

        dataset_name = selection.item[1]
        dataset = self._workspace.datasets.get(dataset_name)
        if not dataset:
            self.view_stack.setCurrentWidget(self.placeholder_widget)
            return

        # The full relative path is the last item in the hierarchy
        image_relative_path = selection.item[-1]
        image_path = Path(dataset.base_data_path) / image_relative_path

        self.labeling_widget.set_media(
            dataset_operations=self._workspace.datasets, dataset_name=dataset_name, image_path=image_path
        )
        self.view_stack.setCurrentWidget(self.labeling_widget)

    def _show_sampling(self, selection: data_explorer.Selection):
        """Shows the frame sampling widget for a selected video."""
        dataset_name = selection.item[1]
        video_rel_path = selection.item[-1]

        self.sampling_widget.set_media(self._workspace, dataset_name, video_rel_path)
        self.view_stack.setCurrentWidget(self.sampling_widget)
