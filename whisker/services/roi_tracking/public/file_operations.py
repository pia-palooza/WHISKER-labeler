from pathlib import Path

from whisker.core.study.dataset_operations import DatasetOperations
from whisker.core.study.project_operations import ProjectOperations

from .correction_operations import RoiCorrectionOperations
from .label_operations import RoiLabelOperations
from .model_operations import RoiModelOperations
from .prediction_operations import RoiPredictionOperations


class RoiTrackingFileOperations:
    def __init__(self, base_dir: Path, projects: ProjectOperations, datasets: DatasetOperations):
        self.labels = RoiLabelOperations(base_dir / 'labels', projects, datasets)
        self.models = RoiModelOperations(base_dir / 'models', projects, datasets)
        self.predictions = RoiPredictionOperations(base_dir / 'predictions', projects, datasets)
        self.corrections = RoiCorrectionOperations(base_dir / 'corrections', projects, datasets)
