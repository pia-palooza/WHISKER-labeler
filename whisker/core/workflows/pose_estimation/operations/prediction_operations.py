# UPDATE_FILE: whisker/core/workflows/pose_estimation/operations/prediction_operations.py
import logging
import json
import os
import numpy as np
from pathlib import Path
from typing import Optional, Set, Dict, List, Any

from whisker.core.base_workflow_operation_helper import BaseWorkflowOperationsHelper
from whisker.core.dataset import DatasetType
from whisker.core.dataset_operations import DatasetOperations
from whisker.core.project_operations import ProjectOperations

from ..data_structures import PoseDataset
from .label_operations import PoseLabelOperations

class PosePredictionOperations(BaseWorkflowOperationsHelper):
    def __init__(
        self,
        base_dir: Path,
        projects: ProjectOperations,
        datasets: DatasetOperations,
        labels: PoseLabelOperations
    ):
        super().__init__(base_dir, projects, datasets)
        self._projects: ProjectOperations = projects
        self._datasets: DatasetOperations = datasets
        self._pose_labels: PoseLabelOperations = labels
        
        self._pose_predictions: Dict[str, Dict[str, Dict[Optional[str], Optional[PoseDataset]]]] = {}

    def scan_predictions(self):
        self._pose_predictions.clear()
        if self._base_dir.exists():
            for run_name in os.listdir(self._base_dir):
                run_dir = self._base_dir / run_name
                if not run_dir.is_dir() or run_name.startswith("."): continue

                self._pose_predictions[run_name] = {}
                
                for dataset_name in os.listdir(run_dir):
                    dataset_dir = run_dir / dataset_name
                    if not dataset_dir.is_dir() or dataset_name.startswith("."): continue

                    self._pose_predictions[run_name][dataset_name] = {}

                    if (dataset_dir / "predictions.h5").exists():
                        self._pose_predictions[run_name][dataset_name][None] = None
                    else:
                        for video_stem in os.listdir(dataset_dir):
                            video_dir = dataset_dir / video_stem
                            if not video_dir.is_dir(): continue

                            if (video_dir / "predictions.h5").exists():
                                self._pose_predictions[run_name][dataset_name][video_stem] = None

    def add_pose_predictions(
        self,
        training_run_name: str,
        dataset_name: str,
        pose_labels: PoseDataset,
        video_stem: Optional[str] = None,
    ):
        file_path = self.get_pose_predictions_path(training_run_name, dataset_name, video_stem)
        
        description = f"{dataset_name} (Run: {training_run_name})"
        if video_stem:
            description += f" (Video: {video_stem})"

        os.makedirs(file_path.parent, exist_ok=True)
        logging.info(f"Predictions will be stored in {file_path}")

        self._pose_labels.write_poses_file( 
            dataset_name=description,
            pose_labels=pose_labels,
            pose_labels_path=file_path,
        )

        self._pose_predictions.setdefault(training_run_name, {}).setdefault(
            dataset_name, {}
        )[video_stem] = pose_labels

    def save_debug_metrics(
        self,
        training_run_name: str,
        dataset_name: str,
        debug_data: List[Dict[str, Any]],
        video_stem: Optional[str] = None
    ):
        """
        Saves debug info (graph tracker costs) to a sidecar JSON file.
        Handles numpy serialization.
        """
        base_path = self.get_pose_predictions_path(training_run_name, dataset_name, video_stem)
        base_path.parent.mkdir(exist_ok=True, parents=True)
        debug_path = base_path.parent / "debug_metrics.json"
        
        def np_encoder(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)

        try:
            with open(debug_path, 'w') as f:
                json.dump(debug_data, f, indent=2, default=np_encoder)
            logging.info(f"Saved debug metrics to {debug_path}")
        except Exception as e:
            logging.error(f"Failed to save debug metrics: {e}")

    def get_pose_predictions_path(
        self,
        training_run_name: str,
        dataset_name: str,
        video_stem: Optional[str] = None,
    ) -> Path:
        base_path = self._base_dir / training_run_name / dataset_name
        if video_stem:
            return base_path / video_stem / "predictions.h5"
        else:
            return base_path / "predictions.h5"

    def get_pose_prediction_frame_keys(
        self, training_run_name: str, dataset_name: str
    ) -> Set[str]:
        all_keys: Set[str] = set()
        run_data = self._pose_predictions.get(training_run_name, {}).get(dataset_name, {})

        if not run_data:
            return all_keys

        for video_stem in run_data.keys():
            path = self.get_pose_predictions_path(training_run_name, dataset_name, video_stem)
            metadata_path = path.parent / "metadata.json"
            
            if metadata_path.exists():
                try:
                    with open(metadata_path, "r") as f:
                        all_keys.update(json.load(f).get("frame_indices", []))
                except (json.JSONDecodeError, IOError) as e:
                    logging.warning(f"Could not read pose prediction metadata: {metadata_path}. Error: {e}")

        return all_keys

    def has_pose_prediction(
        self,
        training_run_name: str,
        dataset_name: str,
        video_stem: Optional[str] = None,
    ) -> bool:
        return video_stem in self._pose_predictions.get(training_run_name, {}).get(dataset_name, {})

    def get_pose_predictions_for_run(
        self,
        training_run_name: str
    ) -> Dict[str, Dict[Optional[str], Optional[PoseDataset]]] | None:
        return self._pose_predictions.get(training_run_name)

    def get_pose_predictions(
        self,
        training_run_name: str,
        dataset_name: str,
        video_stem: Optional[str] = None,
        raise_if_missing: bool = True,
    ) -> Dict[Optional[str], Optional[PoseDataset]] | PoseDataset | None:
        dataset = self._datasets.get(dataset_name)
        if dataset is None:
            if raise_if_missing:
                raise RuntimeError(f"Dataset {dataset_name} does not exist.")
            return None

        dataset_predictions = self._pose_predictions.get(training_run_name, {}).get(dataset_name, {})

        if dataset.type == DatasetType.VIDEO_COLLECTION:
            if video_stem is None:
                return dataset_predictions
            
            if video_stem not in dataset_predictions:
                if raise_if_missing:
                    raise ValueError(f"No predictions found for video '{video_stem}'.")
                return None

            if dataset_predictions[video_stem] is None:
                logging.info(f"Lazily loading pose predictions for run '{training_run_name}', dataset '{dataset_name}', video '{video_stem}'...")
                self._load_and_cache(training_run_name, dataset_name, video_stem, dataset_predictions)

            return dataset_predictions[video_stem]

        else:
            if None not in dataset_predictions:
                if raise_if_missing:
                    raise ValueError(f"No predictions found for dataset '{dataset_name}'.")
                return None
                
            if dataset_predictions[None] is None:
                logging.info(f"Lazily loading pose predictions for run '{training_run_name}', dataset '{dataset_name}'...")
                self._load_and_cache(training_run_name, dataset_name, None, dataset_predictions)
                
            return dataset_predictions[None]

    def get_models_with_predictions_for_dataset(self, dataset_name: str) -> list[str]:
        models = []
        for model_name, datasets in self._pose_predictions.items():
            if dataset_name in datasets:
                models.append(model_name)
        return models
    
    def get_evaluation_metrics(self, model_name: str, dataset_name: str) -> dict | None:
        eval_metrics_path = self._base_dir / model_name / dataset_name / "evaluation_metrics.json"
        if eval_metrics_path.exists():
            with open(eval_metrics_path, "r") as f:
                return json.load(f)
        return None

    def _load_and_cache(self, run_name, ds_name, stem, container_dict):
        labels_path = self.get_pose_predictions_path(run_name, ds_name, stem)
        if not labels_path.exists():
            raise FileNotFoundError(f"Pose prediction file not found: {labels_path}")

        try:
            loaded_labels = PoseDataset.from_file(labels_path)
            container_dict[stem] = loaded_labels
            logging.info(f"Successfully loaded predictions.")
        except Exception as e:
            logging.error(f"Failed to load pose predictions from {labels_path}: {e}", exc_info=True)
            raise

    def items(self):
        return list(self._pose_predictions.items())