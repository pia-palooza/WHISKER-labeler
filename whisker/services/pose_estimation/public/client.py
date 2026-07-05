import logging
import pprint
from pathlib import Path
from typing import Dict, Optional, Set

from ....base.node import Node
from ....core.study.dataset_operations import DatasetOperations
from ....core.study.project_operations import ProjectOperations
from ....services import register_service
from ....services.pose_estimation.public.data_structures import PoseDataset
from .file_operations import PoseEstimationFileOperations

SERVICE_NAME = "Pose Estimation"

# ================================================================= #
# Pose Label & Prediction Functions (Delegated)                     #
# ================================================================= #
def get_pose_labeled_image_keys_from_summary(self, dataset_name: str) -> Set[str]:
    return self.pose_labels.get_pose_labeled_image_keys_from_summary(dataset_name)

def set_pose_labels(self, dataset_name: str, pose_dataset: PoseDataset, warn_if_exists = None):
    return self.pose_labels.set_pose_labels(dataset_name, pose_dataset, warn_if_exists)

def import_pose_labels(self, dataset_name: str, project_name: str, pose_labels_path: Path, backend = None, warn_if_exists = None) -> str:
    return self.pose_labels.import_pose_labels(dataset_name, project_name, pose_labels_path, backend, warn_if_exists)

def get_labeled_image_count(self, dataset_name: str, video_subset: str = "") -> int:
    return self.pose_labels.get_labeled_image_count(dataset_name, video_subset)

def get_pose_dataset(self, dataset_name: str, raise_if_missing: bool = True) -> Optional[PoseDataset]:
    return self.pose_labels.get_pose_dataset(dataset_name, raise_if_missing)

def get_pose_labeled_image_keys(self, dataset_name: str) -> Set[str]:
    return self.pose_labels.get_pose_labeled_image_keys(dataset_name)

def get_pose_dataset_path(self, dataset_name: str) -> Path:
    return self.pose_labels.get_pose_dataset_path(dataset_name)

def write_poses_file(self, dataset_name, pose_labels: PoseDataset, pose_labels_path: Path):
    return self.pose_labels.write_poses_file(dataset_name, pose_labels, pose_labels_path)

def save_pose_labels(self, dataset_name: str) -> None:
    self.pose_labels.save_pose_labels(dataset_name)

def add_pose_predictions(self, training_run_name: str, dataset_name: str, pose_labels: PoseDataset, video_stem: Optional[str] = None):
    self.pose_predictions.add_pose_predictions(training_run_name, dataset_name, pose_labels, video_stem)

def get_pose_predictions_path(self, training_run_name: str, dataset_name: str, video_stem: Optional[str] = None) -> Path:
    return self.pose_predictions.get_pose_predictions_path(training_run_name, dataset_name, video_stem)

def get_pose_prediction_frame_keys(self, training_run_name: str, dataset_name: str) -> Set[str]:
    return self.pose_predictions.get_pose_prediction_frame_keys(training_run_name, dataset_name)

def has_pose_prediction(self, training_run_name: str, dataset_name: str, video_stem: Optional[str] = None) -> bool:
    return self.pose_predictions.has_pose_prediction(training_run_name, dataset_name, video_stem)

def get_pose_predictions(self, training_run_name: str, dataset_name: str, video_stem: Optional[str] = None, raise_if_missing: bool = True) -> Dict[Optional[str], Optional[PoseDataset]] | PoseDataset | None:
    return self.pose_predictions.get_pose_predictions(training_run_name, dataset_name, video_stem, raise_if_missing)

def _log_pose_label_set_metadata(self, pose_label_set_name: str):
    metadata = self.pose_labels._get_pose_label_metadata(pose_label_set_name)
    if metadata:
        metadata_copy = metadata.copy()
        # Support both legacy and current keys
        keys = metadata_copy.get("frame_indices") or metadata_copy.get("annotated_images") or []
        metadata_copy["num_annotated_images"] = len(keys)
        metadata_copy.pop("annotated_images", None)
        metadata_copy.pop("frame_indices", None)
        logging.info(f"Pose labels metadata:\n{pprint.pformat(metadata_copy)}")
    else:
        expected_file_path = self._relpath(
            self.pose_labels.base_dir
            / pose_label_set_name
            / 'metadata.json'
        )
        logging.warning(
            f"Pose labels set {pose_label_set_name} does not have metadata file: {expected_file_path}"
        )

# ================================================================= #
# Properties                                                        #
# ================================================================= #
@property
def pose_labels(self):
    return self.clients[PoseEstimationServiceClient.SERVICE_NAME].file_operations.labels

@property
def pose_models(self):
    return self.clients[PoseEstimationServiceClient.SERVICE_NAME].file_operations.models

@property
def pose_predictions(self):
    return self.clients[PoseEstimationServiceClient.SERVICE_NAME].file_operations.predictions


@register_service(SERVICE_NAME)
class PoseEstimationServiceClient(Node):
    SERVICE_NAME = SERVICE_NAME

    def __init__(
        self,
        file_operations: PoseEstimationFileOperations
    ):
        super().__init__(self.__class__.__name__)
        self.file_operations = file_operations

    @classmethod
    def make_service_client(cls, base_dir: Path, projects: ProjectOperations, datasets: DatasetOperations):
        return PoseEstimationServiceClient(
            PoseEstimationFileOperations(
                base_dir / 'workflows' / 'pose_estimation', projects, datasets
            )
        )
    
    @classmethod
    def get_workspace_attributes(cls):
        return {
            # Properties
            "pose_labels": pose_labels,
            "pose_models": pose_models,
            "pose_predictions": pose_predictions,
            
            # Methods
            "get_pose_labeled_image_keys_from_summary": get_pose_labeled_image_keys_from_summary,
            "set_pose_labels": set_pose_labels,
            "import_pose_labels": import_pose_labels,
            "get_labeled_image_count": get_labeled_image_count,
            "get_pose_dataset": get_pose_dataset,
            "get_pose_labeled_image_keys": get_pose_labeled_image_keys,
            "get_pose_dataset_path": get_pose_dataset_path,
            "write_poses_file": write_poses_file,
            "save_pose_labels": save_pose_labels,
            "add_pose_predictions": add_pose_predictions,
            "get_pose_predictions_path": get_pose_predictions_path,
            "get_pose_prediction_frame_keys": get_pose_prediction_frame_keys,
            "has_pose_prediction": has_pose_prediction,
            "get_pose_predictions": get_pose_predictions,
            "_log_pose_label_set_metadata": _log_pose_label_set_metadata
        }
