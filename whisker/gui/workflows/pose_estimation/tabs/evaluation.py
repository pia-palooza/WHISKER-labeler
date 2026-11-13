"""The Pose Estimation "Evaluate" task: pick a video and a prediction run, scrub the video, and
see the predicted keypoints overlaid - including per-arena predictions for a multi-arena video
(each arena was predicted as its own "pseudo-video", keyed "{video_stem}_arena{k}"; see
whisker.core.study.dataset.MultiArenaConfig and PosePredictionOperations.get_pose_predictions).

Deliberately much smaller than full WHISKER's Evaluate tab: no ground-truth/metrics comparison,
no multi-run comparison, no detector/tracker debug - this labeler has no training, so there is
nothing to evaluate against, only predictions to look at.
"""
from pathlib import Path
from typing import List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QStackedWidget, QVBoxLayout, QWidget

from whisker.core.study.dataset import Dataset, DatasetType, MultiArenaConfig
from whisker.core.workspace import Project, Workspace
from whisker.gui.constants import KEYPOINT_QCOLORS
from whisker.gui.tabs.base_tab import BaseTab
from whisker.gui.widgets import data_explorer
from whisker.gui.widgets.media_viewer import MediaViewerWidget
from ..widgets.pose_overlay import build_pose_overlay_points

_ALL_ARENAS = "all"
_ALL_ARENAS_UNMASKED = "all_unmasked"


class EvaluationPoseTab(BaseTab):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._dataset: Optional[Dataset] = None
        self._video_rel_path: Optional[str] = None
        self._arena_boxes: list = []

        self._init_ui()
        self._connect_signals()
        self.view_stack.setCurrentWidget(self.placeholder)

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Prediction run:"))
        self.run_combo = QComboBox()
        self.run_combo.setPlaceholderText("Select a run...")
        controls.addWidget(self.run_combo, 1)

        self.arena_label = QLabel("Arena:")
        self.arena_combo = QComboBox()
        self.arena_combo.setToolTip("Choose which arena's predictions to view.")
        self.arena_label.setVisible(False)
        self.arena_combo.setVisible(False)
        controls.addWidget(self.arena_label)
        controls.addWidget(self.arena_combo)
        controls.addStretch()
        main_layout.addLayout(controls)

        self.view_stack = QStackedWidget()
        main_layout.addWidget(self.view_stack, 1)

        self.placeholder = QLabel("Select a video from the Data Explorer to see its predictions.")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setWordWrap(True)
        self.view_stack.addWidget(self.placeholder)

        self.media_viewer = MediaViewerWidget()
        self.view_stack.addWidget(self.media_viewer)

    def _connect_signals(self):
        self.run_combo.currentIndexChanged.connect(self._on_run_changed)
        self.arena_combo.currentIndexChanged.connect(self._on_arena_changed)
        self.media_viewer.frame_changed.connect(self._on_frame_changed)

    # -- BaseTab overrides -----------------------------------------------------------------

    def set_workspace(self, workspace: Optional[Workspace]):
        super().set_workspace(workspace)
        self._reset()

    def on_data_explorer_item_selected(self, selection: data_explorer.Selection):
        if self._workspace and selection.type == data_explorer.ItemTypeEnum.DATASET_VIDEO:
            dataset_name = selection.item[1]
            dataset = self._workspace.datasets.get(dataset_name)
            if dataset and dataset.type == DatasetType.VIDEO_COLLECTION:
                self._load_video(dataset, selection.item[-1])
                return
        self._reset()

    # -- Loading a video ---------------------------------------------------------------------

    def _reset(self):
        self._dataset = None
        self._video_rel_path = None
        self._arena_boxes = []
        self.run_combo.blockSignals(True)
        self.run_combo.clear()
        self.run_combo.blockSignals(False)
        self.arena_label.setVisible(False)
        self.arena_combo.setVisible(False)
        self.media_viewer.set_media(None)
        self.view_stack.setCurrentWidget(self.placeholder)

    def _load_video(self, dataset: Dataset, video_rel_path: str):
        self._dataset = dataset
        self._video_rel_path = str(video_rel_path).replace("\\", "/")

        run_names = sorted(
            self._workspace.pose_predictions.get_models_with_predictions_for_dataset(dataset.name),
            reverse=True,
        )
        self.run_combo.blockSignals(True)
        self.run_combo.clear()
        self.run_combo.addItems(run_names)
        if run_names:
            # setPlaceholderText otherwise leaves a freshly populated combo at currentIndex -1.
            self.run_combo.setCurrentIndex(0)
        self.run_combo.blockSignals(False)

        self._arena_boxes = (
            dataset.multi_arena.boxes_for(self._video_rel_path) if dataset.is_multi_arena else []
        )
        self.arena_combo.blockSignals(True)
        self.arena_combo.clear()
        if self._arena_boxes:
            self.arena_combo.addItem("All Arenas", _ALL_ARENAS)
            self.arena_combo.addItem("All Arenas (unmasked)", _ALL_ARENAS_UNMASKED)
            for k in range(len(self._arena_boxes)):
                self.arena_combo.addItem(f"Arena {k + 1}", k)
        self.arena_combo.blockSignals(False)
        self.arena_label.setVisible(bool(self._arena_boxes))
        self.arena_combo.setVisible(bool(self._arena_boxes))

        video_path = Path(dataset.base_data_path) / video_rel_path
        self.media_viewer.set_media(video_path)
        self.view_stack.setCurrentWidget(self.media_viewer)
        self._apply_arena_view()
        self._on_frame_changed(self.media_viewer.get_current_frame())

    # -- Reacting to the run/arena/frame controls ---------------------------------------------

    def _on_run_changed(self):
        self._on_frame_changed(self.media_viewer.get_current_frame())

    def _on_arena_changed(self):
        self._apply_arena_view()
        self._on_frame_changed(self.media_viewer.get_current_frame())

    def _apply_arena_view(self):
        """Mask/outline the video for the current arena selection (not every frame - only when
        the selection itself changes)."""
        if not self._arena_boxes:
            self.media_viewer.clear_arena_mask()
            self.media_viewer.clear_arena_boxes()
            self.media_viewer.set_zoom_to_roi_disabled(False)
            return
        selection = self.arena_combo.currentData()
        if selection == _ALL_ARENAS:
            self.media_viewer.set_arena_mask(list(self._arena_boxes))
            self.media_viewer.set_arena_boxes(self._arena_boxes)
            self.media_viewer.set_zoom_to_roi_disabled(True)
        elif selection == _ALL_ARENAS_UNMASKED:
            self.media_viewer.clear_arena_mask()
            self.media_viewer.set_arena_boxes(self._arena_boxes)
            self.media_viewer.set_zoom_to_roi_disabled(True)
        elif isinstance(selection, int):
            self.media_viewer.set_arena_mask(self._arena_boxes[selection])
            self.media_viewer.set_arena_boxes(self._arena_boxes, active_index=selection)
            self.media_viewer.set_zoom_to_roi_disabled(False)

    def _effective_arena_indices(self) -> Optional[List[int]]:
        """``None`` for a non-multi-arena (or not-yet-loaded) video - one implicit target.
        Otherwise every arena index currently requested on screen: all of them for "All Arenas"/
        "(unmasked)", or just the one picked."""
        if not self._arena_boxes:
            return None
        selection = self.arena_combo.currentData()
        return [selection] if isinstance(selection, int) else list(range(len(self._arena_boxes)))

    def _on_frame_changed(self, frame_idx: int):
        if not self._dataset or not self._workspace or not self._video_rel_path:
            return
        run_name = self.run_combo.currentText()
        if not run_name:
            self.media_viewer.set_pose_data([], [])
            return

        arena_indices = self._effective_arena_indices()
        color_by_arena = arena_indices is not None and len(arena_indices) > 1
        targets = arena_indices if arena_indices is not None else [None]

        points_data = []
        for arena_index in targets:
            points_data.extend(self._points_for(run_name, frame_idx, arena_index, color_by_arena))
        self.media_viewer.set_pose_data(points_data, [])

    def _points_for(self, run_name: str, frame_idx: int, arena_index: Optional[int], color_by_arena: bool) -> list:
        video_stem = (
            MultiArenaConfig.arena_stem(self._video_rel_path, arena_index)
            if arena_index is not None else Path(self._video_rel_path).stem
        )
        preds = self._workspace.pose_predictions.get_pose_predictions(
            run_name, self._dataset.name, video_stem=video_stem, raise_if_missing=False
        )
        if preds is None:
            return []
        frame_key = f"{self._video_rel_path}/frame_{frame_idx:06d}"
        try:
            image_df = preds.keypoint_data.loc[frame_key]
        except KeyError:
            return []

        identities = list(preds.individuals)
        skeleton = self._project.skeleton if self._project else []
        if color_by_arena:
            color = KEYPOINT_QCOLORS[arena_index % len(KEYPOINT_QCOLORS)]
            return build_pose_overlay_points(image_df, identities, skeleton, color_for=lambda _id: color)
        return build_pose_overlay_points(
            image_df, identities, skeleton,
            color_for=lambda identity_id: KEYPOINT_QCOLORS[identities.index(identity_id) % len(KEYPOINT_QCOLORS)],
        )
