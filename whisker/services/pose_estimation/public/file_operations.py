import json
import logging
from pathlib import Path


from ....core.study.dataset_operations import DatasetOperations
from ....core.study.project_operations import ProjectOperations
from .label_operations import PoseLabelOperations
from .model_operations import PoseModelOperations
from .prediction_operations import PosePredictionOperations


class PoseEstimationFileOperations:
    def __init__(self, base_dir: Path, projects: ProjectOperations, datasets: DatasetOperations):
        self.labels = PoseLabelOperations(base_dir / 'labels', projects, datasets)
        self.models = PoseModelOperations(base_dir / 'models', projects, datasets)
        self.predictions = PosePredictionOperations(
            base_dir / 'predictions',
            projects,
            datasets,
            self.labels
        )
