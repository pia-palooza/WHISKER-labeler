import logging
import shutil
from pathlib import Path
from typing import List, Union

import numpy as np

from whisker.core.workspace import Workspace, Dataset
from whisker.core.workers.registry import JobRegistry
from whisker.core.topics import SubsampleParams, SamplingTechnique
from whisker.base.job import BaseJob

@JobRegistry.register(
    "subsample",
    "Create a frame subset from a video collection (Uniform/K-Means).",
    SubsampleParams
)
class SubsampleDatasetJob(BaseJob):
    """Worker job to subsample an image dataset using various techniques."""

    def __init__(
        self,
        workspace: Workspace,
        params: SubsampleParams,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.workspace = workspace
        self.params = params

    def run(self) -> str:
        source_dataset = self.workspace.datasets.get(self.params.source_dataset_name)
        if not source_dataset:
            raise ValueError(f"Dataset '{self.params.source_dataset_name}' not found.")

        new_dataset_name = self._get_unique_dataset_name()
        if self.workspace.datasets.get(new_dataset_name):
            raise ValueError(f"New dataset name {new_dataset_name} already exists.")

        files = sorted(source_dataset.files)
        total_files = len(files)
        num_to_sample = self.params.num_frames

        if total_files <= num_to_sample:
            raise ValueError(
                f"Dataset has {total_files} files, <= requested {num_to_sample}."
            )

        technique = self.params.technique
        if technique == SamplingTechnique.UNIFORM:
            selected_files = self._sample_uniform(files, num_to_sample)
        else:
            raise ValueError(f"Unsupported sampling technique (K-Means visual & pose features are disabled in standalone labeler): {technique}")

        new_dataset_dir = self.workspace.datasets.base_dir / new_dataset_name
        output_data_dir = new_dataset_dir / "data"
        output_data_dir.mkdir(parents=True, exist_ok=True)

        new_files = []
        base_data_path = Path(source_dataset.base_data_path)

        for f in selected_files:
            src_path = base_data_path / f
            if src_path.exists():
                dst_path = output_data_dir / Path(f).name
                shutil.copy2(src_path, dst_path)
                new_files.append(dst_path.name)
            else:
                logging.warning(f"File not found during subsampling: {src_path}")

        if not new_files:
            shutil.rmtree(new_dataset_dir)
            raise RuntimeError("No files were successfully copied.")

        new_dataset = Dataset(
            name=new_dataset_name,
            type=source_dataset.type,
            base_data_path=str(output_data_dir.resolve()),
            files=sorted(new_files),
        )

        self.workspace.add_dataset(
            new_dataset_name, new_dataset, overwrite_existing=False
        )

        if self.workspace.pose_labels.get_pose_dataset(self.params.source_dataset_name, raise_if_missing=False):
            self.workspace.pose_labels.fill_labels_for_subsample_dataset(
                new_dataset_name, [self.params.source_dataset_name]
            )

        return new_dataset_name

    def _get_unique_dataset_name(self) -> str:
        if self.params.target_dataset_name:
            return self.params.target_dataset_name

        base_name = (
            f"{self.params.source_dataset_name} "
            f"[{self.params.technique.value} {self.params.num_frames}]"
        )
        i = 0
        while True:
            candidate = base_name if i == 0 else f"{base_name}_{i}"
            if not self.workspace.datasets.get_dataset(candidate):
                return candidate
            i += 1

    def _sample_uniform(self, files: List[str], num_to_sample: int) -> List[str]:
        indices = np.linspace(0, len(files) - 1, num_to_sample, dtype=int)
        return [files[i] for i in indices]

    def _sample_kmeans_visual(
        self, dataset: Dataset, files: List[str], num_to_sample: int
    ) -> List[str]:
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.metrics import pairwise_distances
        from pathlib import Path
        import cv2
        import numpy as np
        import logging

        base_path = Path(dataset.base_data_path)
        features, valid_files = [], []

        # Cluster a representative subset to keep it snappy
        pre_sample_count = min(num_to_sample * 10, 5000, len(files))
        pre_sample_indices = np.linspace(0, len(files) - 1, pre_sample_count, dtype=int)
        pre_sample_files = [files[i] for i in pre_sample_indices]

        for f in pre_sample_files:
            img = cv2.imread(str(base_path / f), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                features.append(cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA).flatten())
                valid_files.append(f)

        if len(features) < num_to_sample:
            logging.warning("Insufficient valid images for K-Means. Using uniform.")
            return self._sample_uniform(valid_files or files, num_to_sample)

        features_arr = np.array(features)
        kmeans = MiniBatchKMeans(
            n_clusters=num_to_sample, n_init="auto", random_state=42, batch_size=256
        ).fit(features_arr)

        # Prevent duplicate samples by finding unique nearest neighbors for each centroid
        dist_matrix = pairwise_distances(kmeans.cluster_centers_, features_arr)
        closest_indices = []
        used_indices = set()

        for distances in dist_matrix:
            # Sort indices of features by distance to current cluster center
            for idx in np.argsort(distances):
                if idx not in used_indices:
                    closest_indices.append(idx)
                    used_indices.add(idx)
                    break
        
        return [valid_files[i] for i in closest_indices]

    def _sample_kmeans_pose(
        self, dataset: Dataset, files: List[str], num_to_sample: int
    ) -> List[str]:
        from sklearn.cluster import KMeans
        from sklearn.impute import SimpleImputer

        try:
            # Fetch ground truth labels directly
            pose_ds = self.workspace.get_pose_dataset(
                dataset_name=self.source_dataset_name,
                raise_if_missing=True,
            )
        except Exception as e:
            raise ValueError(f"Missing ground truth pose data for '{self.source_dataset_name}': {e}")

        if not pose_ds or pose_ds.keypoint_data.empty:
            raise ValueError("Pose dataset is empty. Cannot sample poses.")

        df = pose_ds.keypoint_data
        df_wide = df.unstack(level=["individual_id", "body_part"])
        valid_frames = df_wide.index.get_level_values("frame_index").to_numpy()

        if len(valid_frames) <= num_to_sample:
            logging.warning("Fewer tracked frames than requested. Returning all.")
            selected_files = self._map_frames_to_files(valid_frames, files)
            return selected_files[:num_to_sample]

        imputer = SimpleImputer(strategy="mean")
        pose_features = imputer.fit_transform(df_wide.values)

        kmeans = KMeans(n_clusters=num_to_sample, random_state=42, n_init="auto")
        distances = kmeans.fit_transform(pose_features)

        closest_idx_in_features = np.argmin(distances, axis=0)
        selected_frames = valid_frames[closest_idx_in_features]
        unique_frames = np.unique(selected_frames)

        if len(unique_frames) < num_to_sample:
            shortfall = num_to_sample - len(unique_frames)
            available_frames = list(set(valid_frames) - set(unique_frames))
            padding_frames = np.random.choice(
                available_frames, min(shortfall, len(available_frames)), replace=False
            )
            unique_frames = np.concatenate([unique_frames, padding_frames])

        return self._map_frames_to_files(unique_frames, files)

    def _map_frames_to_files(
        self, frames: Union[List, np.ndarray], files: List[str]
    ) -> List[str]:
        """Maps varying pose frame_index formats back to the actual dataset files."""
        selected_files = []
        files_set = set(files)
        for f in frames:
            if isinstance(f, str) and f in files_set:
                selected_files.append(f)
            elif isinstance(f, (int, np.integer)) and 0 <= f < len(files):
                selected_files.append(files[f])
            else:
                # DEV_NOTE: Handles cases where numeric indices are stored as strings
                try:
                    idx = int(f)
                    if 0 <= idx < len(files):
                        selected_files.append(files[idx])
                except (ValueError, TypeError):
                    pass
        return list(set(selected_files))