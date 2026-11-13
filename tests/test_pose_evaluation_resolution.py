"""PosePredictionOperations.get_pose_predictions_for_image: the direct lookup (predictions made
straight on the dataset being viewed) and the fallback for a FRAME_SUBSET sampled from a video
(see sampling.save_manual_frames) - predictions for that video live under the video dataset's
own name and video_stem ("{video}_arena{k}" for a multi-arena video), not the sample's.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pandas as pd

from fixtures import WorkspaceCase, frame_names, make_pose_dataset
from whisker.core.study.dataset import DatasetType, MultiArenaConfig
from whisker.services.pose_estimation.internal.core.utils.sampling import save_manual_frames


def write_tiny_video(path, n_frames=5, size=(40, 40)):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, size)
    for _ in range(n_frames):
        writer.write(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    writer.release()


class DirectLookup(WorkspaceCase):
    """The existing behavior - predictions made straight on the dataset being viewed - is unaffected."""

    def test_predictions_made_on_the_frame_subset_itself_are_found_directly(self):
        ws = self.new_workspace()
        self.add_project(ws)
        files = frame_names(2)
        ds = self.add_dataset(ws, "frames", files)
        pose = make_pose_dataset([files[0]])
        ws.pose_predictions.add_pose_predictions("run1", "frames", pose)
        ws.scan_predictions()

        result = ws.pose_predictions.get_pose_predictions_for_image("run1", ds, files[0])
        self.assertIsNotNone(result)

    def test_no_predictions_anywhere_returns_none(self):
        ws = self.new_workspace()
        self.add_project(ws)
        ds = self.add_dataset(ws, "frames", frame_names(1))
        self.assertIsNone(ws.pose_predictions.get_pose_predictions_for_image("run1", ds, frame_names(1)[0]))

    def test_a_dataset_not_named_like_a_sample_never_falls_back(self):
        ws = self.new_workspace()
        self.add_project(ws)
        ds = self.add_dataset(ws, "frames", frame_names(1))
        # Even if a same-named "source" dataset existed, a plain dataset name must not trigger
        # the fallback path (only the " [Manual Samples]" convention does).
        self.add_dataset(ws, "frames_source", ["vid.mp4"], dtype=DatasetType.VIDEO_COLLECTION)
        self.assertIsNone(ws.pose_predictions.get_pose_predictions_for_image("run1", ds, frame_names(1)[0]))


class SampledFromVideo(WorkspaceCase):
    """Predictions made on the source video (optionally multi-arena) are found when viewing
    frames sampled from it."""

    def setUp(self):
        super().setUp()
        self.ws = self.new_workspace()
        self.project = self.add_project(self.ws)
        self.media_root = self.tmp / "video_media"
        self.media_root.mkdir()

    def make_video_dataset(self, multi_arena=None):
        # add_dataset writes a placeholder file for each path; overwrite it with a real,
        # readable video afterward so sampling.save_manual_frames can open it.
        ds = self.add_dataset(self.ws, "vid_ds", ["myvideo.mp4"], dtype=DatasetType.VIDEO_COLLECTION,
                               media_root=self.media_root)
        write_tiny_video(self.media_root / "myvideo.mp4", n_frames=5)
        if multi_arena is not None:
            ds.multi_arena = multi_arena
            self.ws.save_dataset("vid_ds")
        return ds

    def test_single_arena_video_predictions_are_found_for_a_sampled_frame(self):
        self.make_video_dataset()
        self.ws.pose_predictions.add_pose_predictions(
            "run1", "vid_ds", make_pose_dataset(["myvideo.mp4/frame_000002"]), video_stem="myvideo"
        )
        self.ws.scan_predictions()

        sample_name = save_manual_frames(self.ws, "vid_ds", "myvideo.mp4", [2])
        sample_ds = self.ws.datasets.get(sample_name)
        self.assertEqual(sample_name, "vid_ds [Manual Samples]")

        result = self.ws.pose_predictions.get_pose_predictions_for_image(
            "run1", sample_ds, "myvideo/frame_000002.png"
        )
        self.assertIsNotNone(result)

    def test_each_arena_of_a_multiarena_video_resolves_to_its_own_predictions(self):
        config = MultiArenaConfig(box_width=10, box_height=10, placements={"myvideo.mp4": [(0, 0), (20, 0)]})
        self.make_video_dataset(multi_arena=config)

        pose_a0 = make_pose_dataset(["myvideo.mp4/frame_000001"], x0=1.0)
        pose_a1 = make_pose_dataset(["myvideo.mp4/frame_000001"], x0=99.0)
        self.ws.pose_predictions.add_pose_predictions("run1", "vid_ds", pose_a0, video_stem="myvideo_arena0")
        self.ws.pose_predictions.add_pose_predictions("run1", "vid_ds", pose_a1, video_stem="myvideo_arena1")
        self.ws.scan_predictions()

        sample_name = save_manual_frames(self.ws, "vid_ds", "myvideo.mp4", [1], arena_index=0)
        save_manual_frames(self.ws, "vid_ds", "myvideo.mp4", [1], target_dataset_name=sample_name, arena_index=1)
        sample_ds = self.ws.datasets.get(sample_name)

        row0 = self.ws.pose_predictions.get_pose_predictions_for_image(
            "run1", sample_ds, "myvideo_arena0/frame_000001.png"
        )
        row1 = self.ws.pose_predictions.get_pose_predictions_for_image(
            "run1", sample_ds, "myvideo_arena1/frame_000001.png"
        )
        self.assertIsNotNone(row0)
        self.assertIsNotNone(row1)
        # The two arenas' predictions are genuinely different rows, not a coincidental match.
        self.assertNotEqual(row0["x"].iloc[0] if hasattr(row0["x"], "iloc") else row0["x"],
                             row1["x"].iloc[0] if hasattr(row1["x"], "iloc") else row1["x"])

    def test_a_source_video_that_is_not_video_collection_never_falls_back(self):
        # "vid_ds" exists but isn't a VIDEO_COLLECTION - the fallback must refuse it.
        self.add_dataset(self.ws, "vid_ds", frame_names(1))
        sample_ds = self.add_dataset(self.ws, "vid_ds [Manual Samples]", ["myvideo/frame_000001.png"])
        self.assertIsNone(self.ws.pose_predictions.get_pose_predictions_for_image(
            "run1", sample_ds, "myvideo/frame_000001.png"
        ))
