import logging
import os
import shutil
from pathlib import Path
from typing import  Optional, Callable

from .base_operation_helper import BaseOperationHelper
from .dataset_operations import DatasetOperations
from .project_operations import ProjectOperations

class BaseWorkflowOperationsHelper(BaseOperationHelper):
    """Base class for all operation helpers to share common utility methods."""
    def __init__(
        self,
        base_dir: Path,
        projects: ProjectOperations,
        datasets: DatasetOperations
    ):
        super().__init__(base_dir)
        self._projects: ProjectOperations = projects
        self._datasets: DatasetOperations = datasets

    def get_models_by_project(self) -> dict[str, list[str]]:
        """Get all models grouped by project for a given workflow."""
        logging.error("Must be implemented by subclass!")
        return {}