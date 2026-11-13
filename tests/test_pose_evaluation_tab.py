"""EvaluationPoseTab: the Pose Estimation "Evaluate" task - pick a video + prediction run, scrub
the video, see keypoints overlaid, including per-arena predictions for a multi-arena video.
"""
import os
import tempfile
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication

from fixtures import WorkspaceCase, make_pose_dataset
from whisker.core.study.dataset import DatasetType, MultiArenaConfig
from whisker.gui.widgets import data_explorer
from whisker.gui.workflows.pose_estimation.tabs.evaluation import EvaluationPoseTab, _ALL_ARENAS

_app = QApplication.instance() or QApplication(["whisker_test_pose_eval"])


def write_tiny_video(path, n_frames=5, size=(40, 40)):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, size)
    for _ in range(n_frames):
        writer.write(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    writer.release()


def video_selection(dataset_name, video_rel_path):
    return data_explorer.Selection(
        group=data_explorer.ItemGroupEnum.DATASETS,
        type=data_explorer.ItemTypeEnum.DATASET_VIDEO,
        item=["Datasets", dataset_name, video_rel_path],
    )


class EvaluationPoseTabCase(WorkspaceCase):
    def setUp(self):
        super().setUp()
        self.fail_on_slot_exceptions()
        self.ws = self.new_workspace()
        self.project = self.add_project(self.ws)
        # QMediaPlayer keeps a video file open on Windows even after set_media(None), so this
        # can't live under self.tmp (WorkspaceCase's strict cleanup); tolerate a lingering lock.
        media_tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(media_tmp.cleanup)
        self.media_root = Path(media_tmp.name)
        self.tab = EvaluationPoseTab()
        self.tab.show()
        self.addCleanup(self.tab.deleteLater)

    def make_video_dataset(self, multi_arena=None):
        ds = self.add_dataset(self.ws, "vid_ds", ["myvideo.mp4"], dtype=DatasetType.VIDEO_COLLECTION,
                               media_root=self.media_root)
        write_tiny_video(self.media_root / "myvideo.mp4", n_frames=5)
        if multi_arena is not None:
            ds.multi_arena = multi_arena
            self.ws.save_dataset("vid_ds")
        return ds


class NothingSelected(EvaluationPoseTabCase):
    def test_starts_on_the_placeholder(self):
        self.assertEqual(self.tab.view_stack.currentWidget(), self.tab.placeholder)

    def test_a_non_video_selection_shows_the_placeholder(self):
        self.tab.set_workspace(self.ws)
        self.tab.on_data_explorer_item_selected(data_explorer.Selection(
            group=data_explorer.ItemGroupEnum.DATASETS,
            type=data_explorer.ItemTypeEnum.DATASET_BASE,
            item=["Datasets", "vid_ds"],
        ))
        self.assertEqual(self.tab.view_stack.currentWidget(), self.tab.placeholder)


class SingleArenaVideo(EvaluationPoseTabCase):
    def test_selecting_a_video_populates_the_run_combo_and_shows_the_player(self):
        self.make_video_dataset()
        self.ws.pose_predictions.add_pose_predictions(
            "runA", "vid_ds", make_pose_dataset(["myvideo.mp4/frame_000001"]), video_stem="myvideo"
        )
        self.ws.scan_predictions()
        self.tab.set_workspace(self.ws)

        self.tab.on_data_explorer_item_selected(video_selection("vid_ds", "myvideo.mp4"))

        self.assertEqual(self.tab.view_stack.currentWidget(), self.tab.media_viewer)
        self.assertEqual([self.tab.run_combo.itemText(i) for i in range(self.tab.run_combo.count())], ["runA"])
        self.assertFalse(self.tab.arena_combo.isVisible())

    def test_scrubbing_to_a_predicted_frame_overlays_its_points(self):
        self.make_video_dataset()
        pose = make_pose_dataset(["myvideo.mp4/frame_000001"], x0=7.0)
        self.ws.pose_predictions.add_pose_predictions("runA", "vid_ds", pose, video_stem="myvideo")
        self.ws.scan_predictions()
        self.tab.set_workspace(self.ws)
        self.tab.on_data_explorer_item_selected(video_selection("vid_ds", "myvideo.mp4"))

        with mock.patch.object(self.tab.media_viewer, "set_pose_data") as set_pose_data:
            self.tab._on_frame_changed(1)

        set_pose_data.assert_called_once()
        points_data, gt = set_pose_data.call_args.args
        self.assertEqual(gt, [])
        self.assertEqual(len(points_data), 1)
        self.assertIn("nose", points_data[0]["points"])

    def test_a_frame_with_no_prediction_clears_the_overlay(self):
        self.make_video_dataset()
        pose = make_pose_dataset(["myvideo.mp4/frame_000001"])
        self.ws.pose_predictions.add_pose_predictions("runA", "vid_ds", pose, video_stem="myvideo")
        self.ws.scan_predictions()
        self.tab.set_workspace(self.ws)
        self.tab.on_data_explorer_item_selected(video_selection("vid_ds", "myvideo.mp4"))

        with mock.patch.object(self.tab.media_viewer, "set_pose_data") as set_pose_data:
            self.tab._on_frame_changed(3)   # no prediction written for frame 3

        set_pose_data.assert_called_once_with([], [])


class MultiArenaVideo(EvaluationPoseTabCase):
    def setUp(self):
        super().setUp()
        config = MultiArenaConfig(box_width=10, box_height=10, placements={"myvideo.mp4": [(0, 0), (20, 0)]})
        self.make_video_dataset(multi_arena=config)
        self.pose_a0 = make_pose_dataset(["myvideo.mp4/frame_000001"], x0=1.0)
        self.pose_a1 = make_pose_dataset(["myvideo.mp4/frame_000001"], x0=99.0)
        self.ws.pose_predictions.add_pose_predictions("runA", "vid_ds", self.pose_a0, video_stem="myvideo_arena0")
        self.ws.pose_predictions.add_pose_predictions("runA", "vid_ds", self.pose_a1, video_stem="myvideo_arena1")
        self.ws.scan_predictions()
        self.tab.set_workspace(self.ws)
        self.tab.on_data_explorer_item_selected(video_selection("vid_ds", "myvideo.mp4"))

    def test_arena_combo_is_populated_and_defaults_to_all_arenas(self):
        self.assertTrue(self.tab.arena_combo.isVisible())
        items = [self.tab.arena_combo.itemText(i) for i in range(self.tab.arena_combo.count())]
        self.assertEqual(items, ["All Arenas", "All Arenas (unmasked)", "Arena 1", "Arena 2"])
        self.assertEqual(self.tab.arena_combo.currentData(), _ALL_ARENAS)

    def test_all_arenas_shows_both_arenas_points(self):
        with mock.patch.object(self.tab.media_viewer, "set_pose_data") as set_pose_data:
            self.tab._on_frame_changed(1)
        points_data, _ = set_pose_data.call_args.args
        self.assertEqual(len(points_data), 2)
        xs = sorted(p["points"]["nose"].x() for p in points_data)
        self.assertEqual(xs, [1.0, 99.0])

    def test_picking_one_arena_shows_only_its_points(self):
        self.tab.arena_combo.setCurrentIndex(self.tab.arena_combo.findData(1))  # "Arena 2"
        with mock.patch.object(self.tab.media_viewer, "set_pose_data") as set_pose_data:
            self.tab._on_frame_changed(1)
        points_data, _ = set_pose_data.call_args.args
        self.assertEqual(len(points_data), 1)
        self.assertEqual(points_data[0]["points"]["nose"].x(), 99.0)
