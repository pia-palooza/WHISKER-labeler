import json
import logging
from pathlib import Path


from ....core.study.dataset_operations import DatasetOperations
from ....core.study.project_operations import ProjectOperations
from .label_operations import BehaviorLabelOperations
from .model_operations import BehaviorModelOperations
from .prediction_operations import BehaviorPredictionOperations
from .verification_operations import BehaviorVerificationOperations


class BehaviorClassificationFileOperations:
    def __init__(self, base_dir: Path, projects: ProjectOperations, datasets: DatasetOperations):
        self.labels = BehaviorLabelOperations(base_dir / 'labels', projects, datasets)
        self.models = BehaviorModelOperations(base_dir / 'models', projects, datasets)
        self.predictions = BehaviorPredictionOperations(
            base_dir / 'predictions',
            projects,
            datasets
        )
        self.verification = BehaviorVerificationOperations(base_dir / 'verifications')
