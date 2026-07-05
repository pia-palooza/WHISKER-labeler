from dataclasses import dataclass
import enum

LABELED_INDICATOR = "🟢"
PARTIALLY_LABELED_INDICATOR = "🟡"
NOT_LABELED_INDICATOR = "⚪"
LABELED_NOT_APPLICABLE_INDICATOR = "⚫"
LABELED_DIRTY_STATE_INDICATOR = "🟣"

PREDICTED_INDICATOR = "🟩"
PARTIALLY_PREDICTED_INDICATOR = "🟨"
NOT_PREDICTED_INDICATOR = "⬜"
PREDICTED_NOT_APPLICABLE_INDICATOR = "⬛"

class ItemGroupEnum(str, enum.Enum):
    DATASETS = "Datasets"
    WORKSPACE_FILES = "Workspace Files"
    MODELS = "Models"
    PREDICTIONS = "Predictions"

class ItemTypeEnum(str, enum.Enum):
    WORKSPACE_DIR = "Workspace Directory"
    WORKSPACE_FILE = "Workspace File"
    DATASET_TYPE = "Dataset Type"
    DATASET_BASE = "Dataset Base"
    DATASET_VIDEO_FRAME_SUBSET = "Dataset Video Frame Subset"
    DATASET_VIDEO_FRAME = "Dataset Video Frame"
    DATASET_IMAGE = "Dataset Image"
    DATASET_VIDEO = "Dataset Video"
    POSE_ESTIMATION_MODEL_PROJECT = "Pose Estimation Model Project"
    POSE_ESTIMATION_MODEL = "Pose Estimation Model"
    BEHAVIOR_CLASSIFICATION_MODEL_PROJECT = "Behavior Model Project"
    BEHAVIOR_CLASSIFICATION_MODEL = "Behavior Model"
    ANIMAL_DETECTION_MODEL_PROJECT = "Animal Detection Model Project"
    ANIMAL_DETECTION_MODEL = "Animal Detection Model"
    POSE_ESTIMATION_PREDICTION_PROJECT = "Pose Prediction Project"
    POSE_ESTIMATION_PREDICTION = "Pose Prediction"
    BEHAVIOR_CLASSIFICATION_PREDICTION_PROJECT = "Behavior Prediction Project"
    BEHAVIOR_CLASSIFICATION_PREDICTION = "Behavior Prediction"
    ANIMAL_DETECTION_PREDICTION_PROJECT = "Animal Prediction Project"
    ANIMAL_DETECTION_PREDICTION = "Animal Prediction"


class FilterOptions(str, enum.Enum):
    ALL = "Display All"
    LABELED = "Display Labeled Only"
    UNLABELED = "Display Unlabeled Only"


@dataclass
class Selection:
    group: ItemGroupEnum
    type: ItemTypeEnum
    item: list[str]
    is_leaf: bool = True
