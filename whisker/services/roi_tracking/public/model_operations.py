from pathlib import Path

from whisker.core.base_workflow_operation_helper import BaseWorkflowOperationsHelper
from whisker.core.study.dataset_operations import DatasetOperations
from whisker.core.study.project_operations import ProjectOperations


class RoiModelOperations(BaseWorkflowOperationsHelper):
    """ROI tracking has no trained model (the blob tracker runs in full WHISKER, never in the labeler) -- this stub
    exists only because :class:`whisker.core.workspace.Workspace` calls ``.models.scan_models()`` on every
    registered service unconditionally."""

    def __init__(self, base_dir: Path, projects: ProjectOperations, datasets: DatasetOperations):
        super().__init__(base_dir, projects, datasets)

    def scan_models(self):
        pass

    def get_models_by_project(self) -> dict[str, list[str]]:
        return {}
