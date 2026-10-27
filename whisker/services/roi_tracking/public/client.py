from pathlib import Path
from typing import List, Optional

from ....base.node import Node
from ....core.study.dataset_operations import DatasetOperations
from ....core.study.project_operations import ProjectOperations
from ....services import register_service
from ....services.roi_tracking.internal.verify_edits import TrackEdits
from .file_operations import RoiTrackingFileOperations

SERVICE_NAME = "ROI Tracking"


# ================================================================= #
# ROI Label Functions (Delegated)                                   #
# ================================================================= #
def get_roi_template(self, dataset_name: str) -> dict:
    return self.roi_labels.get_template(dataset_name)

def get_roi_video_rois(self, dataset_name: str) -> dict:
    return self.roi_labels.get_video_rois(dataset_name)

def get_roi_video_scale(self, dataset_name: str, video_name: str) -> dict:
    return self.roi_labels.get_video_scale(dataset_name, video_name)


# ================================================================= #
# ROI Prediction Functions (Delegated)                               #
# ================================================================= #
def list_roi_runs(self, dataset_name: str) -> List[str]:
    return self.roi_predictions.list_runs(dataset_name)

def has_roi_prediction(self, run_name: str, dataset_name: str, video_stem: str) -> bool:
    return self.roi_predictions.has_prediction(run_name, dataset_name, video_stem)

def load_roi_prediction(self, run_name: str, dataset_name: str, video_stem: str) -> Optional[dict]:
    return self.roi_predictions.load_prediction(run_name, dataset_name, video_stem)

def get_roi_prediction_sha256(self, run_name: str, dataset_name: str, video_stem: str) -> Optional[str]:
    return self.roi_predictions.get_prediction_sha256(run_name, dataset_name, video_stem)


# ================================================================= #
# ROI Correction Functions (Delegated)                               #
# ================================================================= #
def load_roi_corrections(self, run_name: str, dataset_name: str, video_stem: str) -> TrackEdits:
    return self.roi_corrections.load_edits(run_name, dataset_name, video_stem)

def save_roi_corrections(self, run_name: str, dataset_name: str, video_name: str, video_stem: str, edits: TrackEdits,
                          total_frames: int, source_run: str, prediction_sha256: Optional[str],
                          annotator: str = "") -> Path:
    return self.roi_corrections.save_edits(
        run_name, dataset_name, video_name, video_stem, edits, total_frames, source_run, prediction_sha256,
        annotator)

def discard_roi_corrections(self, run_name: str, dataset_name: str, video_stem: str) -> None:
    self.roi_corrections.discard(run_name, dataset_name, video_stem)

def get_roi_corrections_labeled_video_keys(self, dataset_name: str) -> set:
    return self.roi_corrections.corrections_labeled_video_keys(dataset_name)


@property
def roi_labels(self):
    return self.clients[RoiTrackingServiceClient.SERVICE_NAME].file_operations.labels

@property
def roi_predictions(self):
    return self.clients[RoiTrackingServiceClient.SERVICE_NAME].file_operations.predictions

@property
def roi_corrections(self):
    return self.clients[RoiTrackingServiceClient.SERVICE_NAME].file_operations.corrections


@register_service(SERVICE_NAME)
class RoiTrackingServiceClient(Node):
    SERVICE_NAME = SERVICE_NAME

    def __init__(self, file_operations: RoiTrackingFileOperations):
        super().__init__(self.__class__.__name__)
        self.file_operations = file_operations

    @classmethod
    def make_service_client(cls, base_dir: Path, projects: ProjectOperations, datasets: DatasetOperations):
        return RoiTrackingServiceClient(
            RoiTrackingFileOperations(base_dir / 'workflows' / 'roi_tracking', projects, datasets)
        )

    @classmethod
    def get_workspace_attributes(cls):
        return {
            # Properties
            "roi_labels": roi_labels,
            "roi_predictions": roi_predictions,
            "roi_corrections": roi_corrections,

            # Label Methods
            "get_roi_template": get_roi_template,
            "get_roi_video_rois": get_roi_video_rois,
            "get_roi_video_scale": get_roi_video_scale,

            # Prediction Methods
            "list_roi_runs": list_roi_runs,
            "has_roi_prediction": has_roi_prediction,
            "load_roi_prediction": load_roi_prediction,
            "get_roi_prediction_sha256": get_roi_prediction_sha256,

            # Correction Methods
            "load_roi_corrections": load_roi_corrections,
            "save_roi_corrections": save_roi_corrections,
            "discard_roi_corrections": discard_roi_corrections,
            "get_roi_corrections_labeled_video_keys": get_roi_corrections_labeled_video_keys,
        }
