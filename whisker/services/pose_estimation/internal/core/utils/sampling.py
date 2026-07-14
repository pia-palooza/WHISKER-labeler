import enum
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from whisker.core.workspace import Workspace
from whisker.core.study.dataset import Dataset, DatasetType
from whisker.core.topics import SubsampleParams, SamplingTechnique
from whisker.core.utils.masking import mask_frame

# Arena "pseudo-video" subdirs are named "{video_stem}_arena{k}".
_ARENA_SUBDIR_RE = re.compile(r"_arena(\d+)$")


def _recover_box_from_masked_frame(img_path: Path) -> Optional[Tuple[int, int, int, int]]:
    """Recover an arena box (x, y, w, h) from a masked frame: everything outside
    the arena box was blacked out, so the box is the bounding rectangle of the
    non-black pixels. Returns None if the frame can't be read or is all black."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()), int(ys.max())
    return (x1, y1, x2 - x1 + 1, y2 - y1 + 1)


def backfill_arena_boxes(
    workspace: Workspace, dataset_name: str, persist: bool = True
) -> Dict[str, Tuple[int, int, int, int]]:
    """Recover ``arena_boxes`` for a legacy arena-sampled FRAME_SUBSET whose
    frames predate box persistence. Arena subdirs ("*_arena{k}") are detected by
    name and each box recovered from a representative masked frame (the bounding
    rect of non-black pixels). Boxes already recorded are kept. Returns the
    resulting box map.

    ``persist`` controls whether the recovered boxes are written back to the
    dataset manifest. The user-facing migration keeps ``persist=True``; callers
    that only need the boxes for the current run (e.g. arena-crop training) pass
    ``persist=False`` so an *approximate* pixel-recovered box is never silently
    saved as authoritative.

    No-op (returns {}) for datasets with no arena-style subdirs, so it is safe to
    call opportunistically.
    """
    dataset = workspace.datasets.get(dataset_name)
    if not dataset or dataset.type != DatasetType.FRAME_SUBSET:
        return {}

    base_path = Path(dataset.base_data_path)
    boxes: Dict[str, Tuple[int, int, int, int]] = dict(dataset.arena_boxes or {})

    # Group files by their leading subdir, keeping one sample per arena subdir.
    sample_per_subdir: Dict[str, str] = {}
    for rel in dataset.files:
        subdir = Dataset.arena_subdir_of(rel)
        parts = Path(str(rel).replace("\\", "/")).parts
        if len(parts) < 2:
            continue
        if _ARENA_SUBDIR_RE.search(subdir) and subdir not in boxes and subdir not in sample_per_subdir:
            sample_per_subdir[subdir] = rel

    for subdir, rel in sample_per_subdir.items():
        box = _recover_box_from_masked_frame(base_path / rel)
        if box is not None:
            boxes[subdir] = box
            logging.info(f"Recovered arena box for '{dataset_name}/{subdir}': {box}")
        else:
            logging.warning(f"Could not recover arena box for '{dataset_name}/{subdir}'.")

    if persist and boxes and boxes != (dataset.arena_boxes or {}):
        dataset.arena_boxes = boxes
        workspace.save_dataset(dataset_name)

    return boxes


class FrameSampler:
    """Handles the core logic of sampling frames from a video collection."""

    def __init__(
        self,
        workspace: Workspace,
        sampling_params: SubsampleParams,
    ):
        self._workspace = workspace
        self._source_dataset_name = sampling_params.source_dataset_name
        self._sampling_params = sampling_params

    def run(self) -> str:
        """
        Executes the frame sampling process.

        This creates a new FRAME_SUBSET dataset containing the sampled frames.
        The new dataset is self-contained within the workspace's datasets
        directory.

        Returns:
            The name of the newly created dataset.
        """
        source_dataset = self._workspace.datasets.get(self._source_dataset_name)
        if not source_dataset:
            raise ValueError(
                f"Source dataset '{self._source_dataset_name}' not found in workspace."
            )
        if source_dataset.type != DatasetType.VIDEO_COLLECTION:
            raise ValueError(
                f"Source dataset '{self._source_dataset_name}' is not a video collection. "
                f"Its type is '{source_dataset.type.value}'."
            )

        new_dataset_name = self._get_unique_dataset_name()
        new_dataset_dir = self._workspace.datasets.base_dir / new_dataset_name
        # DEV_NOTE: The 'data' subdir will store the actual images and act as the base path.
        # This creates a clean, self-contained dataset structure.
        output_data_dir = new_dataset_dir / "data"
        output_data_dir.mkdir(parents=True, exist_ok=True)

        self._save_params(new_dataset_dir)

        new_dataset_files = []
        # Maps each arena subdir ("{stem}_arena{k}") to the fixed arena box its
        # masked frames were cut from, so arena-crop training can recover it.
        arena_box_map: dict[str, Tuple[int, int, int, int]] = {}
        # DEV_NOTE: We must construct full, absolute paths to the source videos
        # by combining the dataset's base path with the relative file paths.
        num_videos = len(source_dataset.files)

        for i, rel in enumerate(source_dataset.files):
            video_path = Path(source_dataset.base_data_path) / rel
            video_rel = str(rel).replace("\\", "/")
            logging.info(
                f"Sampling video {i + 1}/{num_videos}: '{video_path.name}'"
            )
            try:
                sampled_frames = self._sample_frames_from_video(video_path)

                # Multi-arena: emit one masked, per-arena frame set per placed box
                # (each arena becomes its own subdir so it labels/trains as an
                # independent unit). Frames stay full-size; only pixels outside
                # the arena box are blacked out, so coordinates remain full-frame.
                arena_boxes = (
                    source_dataset.multi_arena.boxes_for(video_rel)
                    if source_dataset.is_multi_arena
                    else []
                )
                if arena_boxes:
                    for arena_idx, box in enumerate(arena_boxes):
                        subdir = f"{video_path.stem}_arena{arena_idx}"
                        arena_box_map[subdir] = tuple(int(v) for v in box)
                        for frame_index, frame_data in sampled_frames:
                            masked = mask_frame(frame_data, box)
                            relative_path = self._save_frame(
                                masked, frame_index, video_path, output_data_dir,
                                subdir_name=subdir,
                            )
                            new_dataset_files.append(str(relative_path))
                else:
                    for frame_index, frame_data in sampled_frames:
                        # DEV_NOTE: Save the frame and get its path relative to the
                        # new dataset's 'data' directory.
                        relative_path = self._save_frame(
                            frame_data, frame_index, video_path, output_data_dir
                        )
                        new_dataset_files.append(str(relative_path))
            except Exception as e:
                logging.error(
                    f"Could not sample frames from '{video_path.name}': {e}",
                    exc_info=True,
                )
                continue

        if not new_dataset_files:
            # DEV_NOTE: Clean up the empty directory if no frames were sampled.
            shutil.rmtree(new_dataset_dir)
            raise RuntimeError("No frames were sampled successfully.")

        # DEV_NOTE: The base_data_path is now correctly set to the directory
        # containing the actual image files.
        new_dataset = Dataset(
            name=new_dataset_name,
            type=DatasetType.FRAME_SUBSET,
            base_data_path=str(output_data_dir.resolve()),
            files=sorted(new_dataset_files),
            arena_boxes=arena_box_map or None,
        )
        # DEV_NOTE: The call to add_dataset is now correct and avoids overwriting.
        self._workspace.add_dataset(
            new_dataset_name, new_dataset, overwrite_existing=False
        )

        return new_dataset_name

    def _get_unique_dataset_name(self) -> str:
        """Generates a unique dataset name to avoid collisions."""
        if self._sampling_params.target_dataset_name:
            return self._sampling_params.target_dataset_name

        # DEV_NOTE: A more descriptive name is generated, e.g., "MyVideos [K-Means 20]".
        base_name = (
            f"{self._source_dataset_name} "
            f"[{self._sampling_params.technique.value} {self._sampling_params.num_frames}]"
        )

        i = 0
        while True:
            candidate_name = base_name if i == 0 else f"{base_name}_{i}"
            if not self._workspace.datasets.get_dataset(candidate_name):
                return candidate_name
            i += 1

    def _sample_frames_from_video(
        self, video_path: Path
    ) -> List[Tuple[int, np.ndarray]]:
        """Dispatcher to select and run the chosen sampling technique (optimized for speed)."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                logging.warning(f"Video '{video_path.name}' has no frames.")
                return []

            num_to_sample = self._sampling_params.num_frames
            if total_frames < num_to_sample:
                logging.warning(
                    f"Video '{video_path.name}' has fewer frames ({total_frames}) "
                    f"than requested ({num_to_sample}). Sampling all frames."
                )
                num_to_sample = total_frames

            if self._sampling_params.technique == SamplingTechnique.UNIFORM:
                indices = self._sample_uniform(total_frames, num_to_sample)
            else:
                raise ValueError(
                    f"Unsupported sampling technique: '{self._sampling_params.technique}' (K-Means/Visual techniques disabled in standalone labeler)"
                )

            # --- Optimized frame extraction ---
            # Read frames in a single pass for efficiency
            indices_set = set(indices)
            frames = []
            idx_iter = iter(sorted(indices))
            next_idx = next(idx_iter, None)
            current_frame = 0

            if not indices_set:
                return []

            # DEV_NOTE: It's crucial to rewind the video before this sequential read,
            # especially after the k-means analysis might have moved the read head.
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while next_idx is not None and current_frame < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                if current_frame == next_idx:
                    frames.append((current_frame, frame.copy()))
                    next_idx = next(idx_iter, None)
                current_frame += 1
                if len(frames) == len(indices):
                    break
            return frames
        finally:
            cap.release()

    def _sample_uniform(self, total_frames: int, num_to_sample: int) -> List[int]:
        """Returns `num_to_sample` indices sampled uniformly."""
        if num_to_sample <= 0:
            return []
        return np.linspace(0, total_frames - 1, num_to_sample, dtype=int).tolist()

    def _sample_kmeans(
        self, cap: cv2.VideoCapture, total_frames: int, num_to_sample: int
    ) -> List[int]:
        """
        Dispatcher for k-means sampling. Chooses GPU-accelerated path if
        available, otherwise falls back to the optimized CPU path.
        """
        # DEV_NOTE: Import torch just-in-time to speed up application startup.
        import torch

        if torch.cuda.is_available():
            logging.info("CUDA device found. Using GPU-accelerated K-Means sampling.")
            try:
                return self._sample_kmeans_gpu(cap, total_frames, num_to_sample)
            except Exception as e:
                logging.warning(
                    f"GPU K-Means sampling failed with error: {e}. "
                    "Falling back to CPU implementation."
                )
                # Fallback to CPU if the GPU method fails for any reason
                return self._sample_kmeans_cpu(cap, total_frames, num_to_sample)
        else:
            logging.info("No CUDA device found. Using CPU-based K-Means sampling.")
            return self._sample_kmeans_cpu(cap, total_frames, num_to_sample)

    def _sample_pose_kmeans(
        self, pose_dataset, total_frames: int, num_to_sample: int
    ) -> List[int]:
        """Clusters morphological states from pose data."""
        # DEV_NOTE: Lazy imports to keep startup fast
        from sklearn.cluster import KMeans
        from sklearn.impute import SimpleImputer

        df = pose_dataset.keypoint_data
        if df.empty:
            raise ValueError("Pose dataset is empty; cannot sample poses.")

        # Pivot to wide format: rows=frame_index, cols=(individual_id, body_part, [x,y,c])
        df_wide = df.unstack(level=['individual_id', 'body_part'])
        valid_frames = df_wide.index.get_level_values('frame_index').to_numpy()

        if len(valid_frames) <= num_to_sample:
            logging.warning("Fewer tracked frames than requested. Returning all.")
            return [int(f) for f in valid_frames]

        imputer = SimpleImputer(strategy='mean')
        pose_features = imputer.fit_transform(df_wide.values)

        kmeans = KMeans(n_clusters=num_to_sample, random_state=42, n_init='auto')
        distances = kmeans.fit_transform(pose_features)

        # Find the frame closest to the center of each cluster
        closest_idx_in_features = np.argmin(distances, axis=0)
        selected_frames = valid_frames[closest_idx_in_features]
        unique_frames = np.unique(selected_frames)

        # Fallback padding if multiple clusters collapse onto the same frame
        if len(unique_frames) < num_to_sample:
            shortfall = num_to_sample - len(unique_frames)
            available_idx = list(set(range(total_frames)) - set(unique_frames))
            
            # Pad with remaining random frames up to the shortfall limit
            padding_idx = np.random.choice(
                available_idx, min(shortfall, len(available_idx)), replace=False
            )
            unique_frames = np.concatenate([unique_frames, padding_idx])

        return [int(f) for f in unique_frames]

    def _sample_kmeans_gpu(
        self, cap: cv2.VideoCapture, total_frames: int, num_to_sample: int
    ) -> List[int]:
        """
        Performs K-Means sampling using GPU acceleration via PyTorch and
        kmeans-pytorch for maximum performance.
        """
        # DEV_NOTE: Lazy-import torch, torchvision, and kmeans_pytorch here to avoid startup errors.
        import torch
        from torchvision import transforms
        from kmeans_pytorch import kmeans

        pre_sample_count = min(num_to_sample * 10, 1000, total_frames)
        pre_sample_indices = np.linspace(
            0, total_frames - 1, pre_sample_count, dtype=int
        )

        device = torch.device("cuda")
        # Define image transforms that will run on the GPU
        gpu_transforms = transforms.Compose(
            [
                transforms.ConvertImageDtype(torch.float),
                transforms.Resize((64, 64), antialias=True),
                transforms.Grayscale(),
            ]
        )

        features = []
        valid_indices = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        logging.info("Pre-sampling indices")
        # --- Optimized batch frame extraction for speed ---
        # Read all needed frames in a single pass, minimizing cap.set calls
        pre_sample_indices_set = set(int(idx) for idx in pre_sample_indices)
        frames_buffer = []
        idx_iter = iter(sorted(pre_sample_indices_set))
        next_idx = next(idx_iter, None)
        current_frame = 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while next_idx is not None and current_frame < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if current_frame == next_idx:
                frames_buffer.append((current_frame, frame))
                next_idx = next(idx_iter, None)
            # DEV_NOTE: This was an infinite loop bug. The frame counter must always increment.
            current_frame += 1
            if len(frames_buffer) == len(pre_sample_indices_set):
                break

        # Batch process frames on the GPU
        if frames_buffer:
            # Convert all frames to RGB and stack as a batch tensor
            rgb_frames = [
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for _, frame in frames_buffer
            ]
            frame_tensors = torch.stack(
                [torch.from_numpy(f).permute(2, 0, 1) for f in rgb_frames]
            ).to(device)

            # Apply transforms in batch (resize, grayscale, dtype)
            processed_tensors = gpu_transforms(frame_tensors)
            # Flatten each processed tensor and collect features
            features = [t.flatten() for t in processed_tensors]
            valid_indices = [idx for idx, _ in frames_buffer]

        logging.info("Pre-sampling complete")

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rewind for final extraction

        if not features or len(features) < num_to_sample:
            return self._sample_uniform(total_frames, num_to_sample)

        features_tensor = torch.stack(features)

        # Run K-Means on the GPU
        logging.info("Running k-means...")
        _, cluster_centers = kmeans(
            X=features_tensor,
            num_clusters=num_to_sample,
            device=device,
            tqdm_flag=False,  # Disable internal progress bar
        )

        # Find the closest point in the original data to each cluster center
        # by calculating batched distances on the GPU.
        logging.info("Finding images closest to each cluster center...")
        closest_indices = []
        for center in cluster_centers:
            distances = torch.norm(features_tensor - center, dim=1)
            closest_idx = torch.argmin(distances).item()
            closest_indices.append(closest_idx)

        # Map feature indices back to original frame indices
        final_indices = [valid_indices[i] for i in closest_indices]
        logging.info("Done.")
        return list(set(final_indices))

    def _sample_kmeans_cpu(
        self, cap: cv2.VideoCapture, total_frames: int, num_to_sample: int
    ) -> List[int]:
        """
        Samples frames by clustering to maximize visual diversity using an
        optimized CPU-based approach.
        """
        # DEV_NOTE: Import scikit-learn just-in-time to speed up application startup.
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.metrics import pairwise_distances_argmin_min

        pre_sample_count = min(num_to_sample * 10, 1000, total_frames)
        if pre_sample_count <= num_to_sample:
            logging.warning(
                "Not enough frames for meaningful k-means; falling back to uniform."
            )
            return self._sample_uniform(total_frames, num_to_sample)

        pre_sample_indices = np.linspace(
            0, total_frames - 1, pre_sample_count, dtype=int
        ).tolist()

        features = []
        valid_indices = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        for idx in pre_sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                small_frame = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
                gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                features.append(gray_frame.flatten())
                valid_indices.append(idx)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if not features or len(features) < num_to_sample:
            logging.warning("Feature extraction failed; falling back to uniform.")
            return self._sample_uniform(total_frames, num_to_sample)

        features_arr = np.array(features)
        kmeans = MiniBatchKMeans(
            n_clusters=num_to_sample,
            n_init="auto",
            random_state=42,
            batch_size=min(256, len(features)),
        )
        kmeans.fit(features_arr)

        closest_indices_in_features, _ = pairwise_distances_argmin_min(
            kmeans.cluster_centers_, features_arr
        )

        final_indices = [valid_indices[i] for i in closest_indices_in_features]
        return list(set(final_indices))

    def _save_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        video_path: Path,
        output_data_dir: Path,
        subdir_name: Optional[str] = None,
    ) -> Path:
        """
        Saves a single frame to a hierarchical output directory and returns
        the path relative to that directory.

        ``subdir_name`` overrides the per-video output subdir (used to keep each
        arena's masked frames in their own group); defaults to the video stem.
        """
        # Create a subdirectory for the video to avoid filename clashes
        video_specific_dir = output_data_dir / (subdir_name or video_path.stem)
        video_specific_dir.mkdir(exist_ok=True)

        filename = f"frame_{frame_index:06d}.png"
        full_output_path = video_specific_dir / filename
        cv2.imwrite(str(full_output_path), frame)

        # DEV_NOTE: The returned path must be relative to the dataset's base_data_path.
        return full_output_path.relative_to(output_data_dir)

    def _save_params(self, new_dataset_dir: Path):
        """Saves sampling parameters to a JSON file in the new dataset's root."""
        params_path = new_dataset_dir / "sampling_params.json"

        class EnumEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, enum.Enum):
                    return obj.value
                return json.JSONEncoder.default(self, obj)

        with params_path.open("w", encoding="utf-8") as f:
            json.dump(self._sampling_params.model_dump(), f, indent=4, cls=EnumEncoder)

def save_manual_frames(
    workspace: Workspace,
    source_dataset_name: str,
    video_rel_path: str,
    frame_indices: List[int],
    target_dataset_name: str = None,
    arena_index: Optional[int] = None,
) -> str:
    """
    Extracts specific frames from a video and saves them to a FRAME_SUBSET dataset.
    If target_dataset_name is provided and exists, it appends to it.
    Otherwise, it creates '{source_dataset_name} [Manual Samples]'.

    ``arena_index``: when the source is a multi-arena dataset, masks each saved
    frame to that arena's box and stores it under a per-arena subdir
    ("{video_stem}_arena{k}"), so pose labeling/training see one masked arena
    per frame (single mouse). None (default) keeps the original whole-frame
    behavior.
    """
    source_dataset = workspace.datasets.get(source_dataset_name)
    if not source_dataset:
        raise ValueError(f"Source dataset '{source_dataset_name}' not found.")

    if not target_dataset_name:
        target_dataset_name = f"{source_dataset_name} [Manual Samples]"

    # 1. Prepare Target Dataset Directory
    target_dataset = workspace.datasets.get(target_dataset_name)

    if target_dataset:
        if target_dataset.type != DatasetType.FRAME_SUBSET:
            raise ValueError(f"Target dataset '{target_dataset_name}' exists but is not a FRAME_SUBSET.")
        # Ensure we use the resolved path or base_data_path if it exists
        if hasattr(target_dataset, 'base_data_path') and target_dataset.base_data_path:
             target_dir = Path(target_dataset.base_data_path)
        else:
             # Fallback if base_data_path is not set (should not happen for valid dataset)
             target_dir = workspace.datasets.base_dir / target_dataset_name / "data"

    else:
        # Create new
        dataset_dir = workspace.datasets.base_dir / target_dataset_name
        target_dir = dataset_dir / "data"
        target_dir.mkdir(parents=True, exist_ok=True)

    # 2. Extract Frames
    # source_dataset.base_data_path should be absolute or relative to CWD?
    # Usually in workspace it is absolute.
    video_path = Path(source_dataset.base_data_path) / video_rel_path
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    video_stem = video_path.stem

    # Multi-arena: mask to the chosen arena and use its per-arena subdir.
    arena_box = None
    out_subdir = video_stem
    if arena_index is not None and source_dataset.is_multi_arena:
        video_rel_norm = str(video_rel_path).replace("\\", "/")
        boxes = source_dataset.multi_arena.boxes_for(video_rel_norm)
        if 0 <= arena_index < len(boxes):
            arena_box = boxes[arena_index]
            out_subdir = source_dataset.multi_arena.arena_stem(video_rel_norm, arena_index)

    video_out_dir = target_dir / out_subdir
    logging.info(f"Ensuring {video_out_dir} exists")
    video_out_dir.mkdir(exist_ok=True, parents=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    new_files = []

    try:
        # Sort indices to optimize seeking
        sorted_indices = sorted(list(set(frame_indices)))

        for frame_idx in sorted_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                if arena_box is not None:
                    frame = mask_frame(frame, arena_box)
                filename = f"frame_{frame_idx:06d}.png"
                out_path = video_out_dir / filename
                logging.info(f"Writing {out_path}")
                cv2.imwrite(str(out_path), frame)

                # Rel path from dataset base (which is target_dir aka 'data' subdir)
                # The Dataset class expects files relative to base_data_path
                rel_path = f"{out_subdir}/{filename}"
                new_files.append(rel_path)
            else:
                logging.warning(f"Could not read frame {frame_idx} from {video_path.name}")
    finally:
        cap.release()

    # 3. Update/Create Dataset Record
    if target_dataset:
        # Append unique new files
        existing_set = set(target_dataset.files)
        added_count = 0
        for f in new_files:
            if f not in existing_set:
                target_dataset.files.append(f)
                added_count += 1
        target_dataset.files.sort()
        # Record this arena's box so arena-crop training can recover it.
        if arena_box is not None:
            boxes = dict(target_dataset.arena_boxes or {})
            boxes[out_subdir] = tuple(int(v) for v in arena_box)
            target_dataset.arena_boxes = boxes
        workspace.save_dataset(target_dataset_name)
        logging.info(f"Added {added_count} frames to existing dataset '{target_dataset_name}'.")
    else:
        # Create new dataset
        new_dataset = Dataset(
            name=target_dataset_name,
            type=DatasetType.FRAME_SUBSET,
            base_data_path=str(target_dir.resolve()),
            files=sorted(new_files),
            arena_boxes={out_subdir: tuple(int(v) for v in arena_box)} if arena_box is not None else None,
        )
        workspace.add_dataset(target_dataset_name, new_dataset, overwrite_existing=False)
        logging.info(f"Created new dataset '{target_dataset_name}' with {len(new_files)} frames.")

    return target_dataset_name
