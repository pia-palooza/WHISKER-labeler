from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QLabel, QStackedWidget, QVBoxLayout, QWidget

from whisker.core.workspace import Project, Workspace
from whisker.gui.tabs.base_tab import BaseTab
from whisker.gui.widgets import data_explorer

from ..widgets.roi_summary_widget import RoiSummaryWidget
from ..widgets.roi_verify_widget import RoiVerifyWidget


class RoiTrackingLabelingTab(BaseTab):
    """Switches between a placeholder, the per-video :class:`RoiVerifyWidget`, and the per-dataset
    :class:`RoiSummaryWidget`, modeled on ``behavior_classification/tabs/labeling.py::LabelingBehaviorsTab``.

    Like the other workflows, a project must be selected for this tab to show anything (decision made with the user:
    the labeler's existing project-required gate stays unchanged; a project's actual content -- body parts, behaviors
    -- is irrelevant to ROI tracking, but the gate itself is not special-cased away)."""

    media_selected = pyqtSignal(Path)

    def __init__(self, parent: Optional[QWidget] = None):
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

        self.placeholder = QLabel("Select a project, then a video from the Data Explorer.")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setWordWrap(True)
        self.view_stack.addWidget(self.placeholder)

        self.verify_widget = RoiVerifyWidget()
        self.view_stack.addWidget(self.verify_widget)

        self.summary_widget = RoiSummaryWidget()
        self.view_stack.addWidget(self.summary_widget)

    def _connect_signals(self):
        self.summary_widget.video_selected.connect(self._on_summary_video_selected)
        self.summary_widget.video_selected.connect(self.media_selected)

    def set_workspace(self, workspace: Optional[Workspace]):
        super().set_workspace(workspace)
        self.summary_widget.set_workspace(workspace)
        self._current_selection = None
        self.view_stack.setCurrentWidget(self.placeholder)
        self.set_dirty(False)

    def set_project(self, project: Optional[Project]):
        super().set_project(project)
        self.set_dirty(False)
        self._update_view()

    def on_data_explorer_item_selected(self, selection: data_explorer.Selection):
        self._current_selection = selection
        self._update_view()

    def _update_view(self):
        selection = self._current_selection

        if not selection or not self._project or not self._workspace:
            self.view_stack.setCurrentWidget(self.placeholder)
            return

        if selection.type == data_explorer.ItemTypeEnum.DATASET_BASE:
            dataset_name = selection.item[1]
            self.summary_widget.set_dataset(dataset_name)
            self.view_stack.setCurrentWidget(self.summary_widget)
            return

        if selection.type == data_explorer.ItemTypeEnum.DATASET_VIDEO:
            dataset_name = selection.item[1]
            dataset = self._workspace.datasets.get(dataset_name)
            video_path = Path(dataset.base_data_path) / selection.item[-1]
            self.verify_widget.set_media(self._workspace, dataset_name, video_path)
            self.view_stack.setCurrentWidget(self.verify_widget)
            return

        self.view_stack.setCurrentWidget(self.placeholder)

    def _on_summary_video_selected(self, video_path: Path):
        if self._workspace:
            self.verify_widget.set_media(self._workspace, self.summary_widget._dataset_name, video_path)
            self.view_stack.setCurrentWidget(self.verify_widget)
