from pathlib import Path
from typing import Optional
import pandas as pd

from ....base.node import Node
from ....core.study.dataset_operations import DatasetOperations
from ....core.study.project_operations import ProjectOperations
from ....services import register_service
from ....services.behavior_classification.public.data_structures import BehaviorDataset
from .file_operations import BehaviorClassificationFileOperations

SERVICE_NAME = "Behavior Classification"

# ================================================================= #
# Behavior Label & Prediction Functions (Delegated)                 #
# ================================================================= #
def has_behavior_prediction(self, training_run_name: str, dataset_name: str, video_stem: str) -> bool:
    return self.behavior_predictions.has_behavior_prediction(training_run_name, dataset_name, video_stem)

def get_behavior_predictions(self, training_run_name: str, dataset_name: str, video_stem: str, raise_if_missing: bool = True) -> Optional[BehaviorDataset]:
    return self.behavior_predictions.get_behavior_predictions(training_run_name, dataset_name, video_stem, raise_if_missing)

def add_video_behavior_predictions(self, training_run_name: str, dataset_name: str, video_stem: str, behavior_dataset: BehaviorDataset):
    self.behavior_predictions.add_video_behavior_predictions(training_run_name, dataset_name, video_stem, behavior_dataset)

def get_behavior_probabilities(self, run_name: str, dataset_name: str, video_stem: str) -> Optional[pd.DataFrame]:
    return self.behavior_predictions.get_behavior_probabilities(run_name, dataset_name, video_stem)

def add_behavior_bouts(self, run_name: str, dataset_name: str, video_stem: str, bouts_df: pd.DataFrame):
    self.behavior_predictions.add_behavior_bouts(run_name, dataset_name, video_stem, bouts_df)

def get_video_behavior_labels(self, dataset_name: str, video_stem: str) -> Optional[pd.DataFrame]:
    return self.behavior_labels.get_video_behavior_labels(dataset_name, video_stem)

def import_behavior_labels(self, dataset_name: str, project_name: str, import_path: Path, backend, warn_if_exists = None) -> str:
    return self.behavior_labels.import_behavior_labels(dataset_name, project_name, import_path, backend, warn_if_exists)

def get_behavior_labels(self, dataset_name: str) -> BehaviorDataset:
    return self.behavior_labels.get_behavior_labels(dataset_name)

def get_behavior_labeled_video_keys(self, dataset_name: str) -> set[str]:
    return self.behavior_labels.get_behavior_labeled_video_keys(dataset_name)

def save_behavior_labels(self, dataset_name: str) -> None:
    self.behavior_labels.save_behavior_labels(dataset_name)

@property
def behavior_labels(self):
    return self.clients[BehaviorClassificationServiceClient.SERVICE_NAME].file_operations.labels

@property
def behavior_models(self):
    return self.clients[BehaviorClassificationServiceClient.SERVICE_NAME].file_operations.models

@property
def behavior_predictions(self):
    return self.clients[BehaviorClassificationServiceClient.SERVICE_NAME].file_operations.predictions

@property
def behavior_verification(self):
    return self.clients[BehaviorClassificationServiceClient.SERVICE_NAME].file_operations.verification
    

# ================================================================= #
# Behavior Verification Functions (Delegated)                      #
# ================================================================= #
def load_behavior_verification(self, run_name: str, dataset_name: str, video_stem: str) -> Optional[pd.DataFrame]:
    return self.behavior_verification.load_verification(run_name, dataset_name, video_stem)

def save_behavior_verification(self, run_name: str, dataset_name: str, video_stem: str, df: pd.DataFrame):
    self.behavior_verification.save_verification(run_name, dataset_name, video_stem, df)


@register_service(SERVICE_NAME)
class BehaviorClassificationServiceClient(Node):
    SERVICE_NAME = SERVICE_NAME

    def __init__(
        self,
        file_operations: BehaviorClassificationFileOperations
    ):
        super().__init__(self.__class__.__name__)
        self.file_operations = file_operations

    @classmethod
    def make_service_client(cls, base_dir: Path, projects: ProjectOperations, datasets: DatasetOperations):
        return BehaviorClassificationServiceClient(
            BehaviorClassificationFileOperations(
                base_dir / 'workflows' / 'behavior_classification', projects, datasets
            )
        )
    
    @classmethod
    def get_workspace_attributes(cls):
        return {
            # Properties
            "behavior_labels": behavior_labels,
            "behavior_models": behavior_models,
            "behavior_predictions": behavior_predictions,
            "behavior_verification": behavior_verification,
            
            # Label & Prediction Methods
            "has_behavior_prediction": has_behavior_prediction,
            "get_behavior_predictions": get_behavior_predictions,
            "add_video_behavior_predictions": add_video_behavior_predictions,
            "get_behavior_probabilities": get_behavior_probabilities,
            "add_behavior_bouts": add_behavior_bouts,
            "get_video_behavior_labels": get_video_behavior_labels,
            "import_behavior_labels": import_behavior_labels,
            "get_behavior_labels": get_behavior_labels,
            "get_behavior_labeled_video_keys": get_behavior_labeled_video_keys,
            "save_behavior_labels": save_behavior_labels,
            
            # Verification Methods
            "load_behavior_verification": load_behavior_verification,
            "save_behavior_verification": save_behavior_verification,
        }

