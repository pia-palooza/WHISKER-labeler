from pathlib import Path
from whisker.base.job import BaseJob
from whisker.core.workspace import Workspace
from whisker.core.constants import BackendEnum, WarnIfExistsFunctionType

class LabelImporterJob(BaseJob):
    def __init__(
        self,
        workspace: Workspace,
        dataset_name: str,
        project_name: str,
        import_path: Path,
        data_type_str: str, # "poses" or "behaviors"
        backend: BackendEnum,
        warn_if_exists: WarnIfExistsFunctionType,
    ):
        super().__init__()
        self.workspace = workspace
        self.dataset_name = dataset_name
        self.project_name = project_name
        self.import_path = import_path
        self.data_type_str = data_type_str
        self.backend = backend
        self.warn_if_exists = warn_if_exists

    def run(self) -> str:
        self.report_progress("Importing...", 0)
        
        if self.data_type_str == "poses":
            report = self.workspace.import_pose_labels(
                self.dataset_name, self.project_name, self.import_path, 
                self.backend, self.warn_if_exists
            )
        elif self.data_type_str == "behaviors":
            report = self.workspace.import_behavior_labels(
                self.dataset_name, self.project_name, self.import_path,
                self.backend, self.warn_if_exists
            )
        else:
            raise NotImplementedError(f"Unknown type: {self.data_type_str}")

        self.report_progress("Finished!", 100)
        return f"Successfully imported labels for '{self.dataset_name}'.\n\n{report}"