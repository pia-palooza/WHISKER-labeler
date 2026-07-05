import os
from pathlib import Path

from whisker.core.base_workflow_operation_helper import BaseWorkflowOperationsHelper
from whisker.core.study.dataset_operations import DatasetOperations
from whisker.core.study.project_operations import ProjectOperations

class BehaviorModelOperations(BaseWorkflowOperationsHelper):
    def __init__(
        self,
        base_dir: Path,
        projects: ProjectOperations,
        datasets: DatasetOperations
    ):
        super().__init__(base_dir, projects, datasets)
        self._projects: ProjectOperations = projects
        self._datasets: DatasetOperations = datasets
        self._behavior_models: list[str] = []

    def scan_models(self):
        """Scans for existing behavior models."""
        self._behavior_models.clear()
        if self._base_dir.exists():
            self._behavior_models = [
                d for d in os.listdir(self._base_dir) 
                if (self._base_dir / d).is_dir() and not d.startswith(".")
            ]

    def get(self) -> list[str]:
        """Returns a list of all behavior model names."""
        return list(self._behavior_models)
