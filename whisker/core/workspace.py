import logging
import pprint
import os
from pathlib import Path
from typing import Optional, Callable

from whisker.third_party.third_party_manager import ThirdPartyManager
from whisker.base.logger import configure_workspace_logging
from .base_operation_helper import BaseOperationHelper
from .study.dataset import Dataset, DatasetType
from .study.project import Project
from .workflows.workflow_enum import Workflow
from ..services import make_service_clients, inject_workspace_attributes
from ..services.pose_estimation.public.client import PoseEstimationServiceClient
from ..services.behavior_classification.public.client import BehaviorClassificationServiceClient

from .study.dataset_operations import DatasetOperations
from .study.project_operations import ProjectOperations

class Workspace:
    def __init__(self, base_dir: Path):
        logging.info(f"Loading workspace at {base_dir}")
        if not base_dir.exists():
            raise ValueError(f"Workspace base directory does not exist: {base_dir}")

        configure_workspace_logging(base_dir)

        self._base_dir = base_dir
        self.third_party_manager = ThirdPartyManager(self._base_dir / 'third_party')
        
        # 2. Initialize Helper Classes
        self.datasets = DatasetOperations(
            self.base_dir / 'datasets'
        )
        self.projects = ProjectOperations(
            self.base_dir / 'projects'
        )

        self.clients = make_service_clients(self.base_dir, self.projects, self.datasets)
        self = inject_workspace_attributes(self)
        
        # 4. Initial full scan (Delegated)
        self.scan_datasets()
        self.scan_projects()
        self.scan_labels()
        self.scan_models() 
        self.scan_predictions()

    # --- Directory Properties (Required for helpers via Protocol) ---
    @property
    def base_dir(self) -> Path:
        return self._base_dir
    
    # --- Utility Functions ---
    def _relpath(self, path: Path) -> Path:
        """Return a relative path to a file within the workspace."""
        return Path("${WS}") / os.path.relpath(path, self._base_dir)
        
    @classmethod
    def _delete_if_existing(
        cls,
        base_path: Path,
        dir_type_name: str,
        warn_if_exists: Optional[Callable[[str], bool]] = None,
    ) -> bool:
        return BaseOperationHelper._delete_if_existing(base_path, dir_type_name, warn_if_exists)
            
    def show(self) -> None:
        logging.info(f"Workspace base directory: {self.base_dir}")
    
    # ================================================================= #
    # Scan Functions (Delegated)                                        #
    # ================================================================= #

    def scan_datasets(self):
        self.datasets.scan_datasets()
    def scan_projects(self):
        self.projects.scan_projects()
    def scan_labels(self):
        for service in self.clients.values():
            if hasattr(service.file_operations, 'labels'):
                service.file_operations.labels.scan_labels()
    def scan_models(self):
        for service in self.clients.values():
            service.file_operations.models.scan_models()
    def scan_predictions(self):
        for service in self.clients.values():
            service.file_operations.predictions.scan_predictions()

    def delete_dataset(self, dataset_name: str):
        self.datasets.remove(dataset_name)
        for service in self.clients.values():
            if hasattr(service.file_operations, 'labels'):
                service.file_operations.labels.remove(dataset_name)

    # ================================================================= #
    # Project Functions (Delegated)                                     #
    # ================================================================= #
    def create_project(self, project_name: str, body_parts: list[str] = [], identities: list[str] = [], skeleton: list[tuple[str, str]] = [], behaviors: list[str] = [], warn_if_exists: Optional[Callable[[str], bool]] = None,) -> Optional[Project]:
        return self.projects.create_project(project_name, body_parts, identities, skeleton, behaviors, warn_if_exists)
    def save_project(self, project_name: str):
        self.projects.save_project(project_name)

    # ================================================================= #
    # Dataset Functions (Delegated)                                     #
    # ================================================================= #
    def get_dataset(self, dataset_name: str) -> Optional[Dataset]:
        return self.datasets.get_dataset(dataset_name)
    def create_dataset(self, dataset_name: str, dataset_type: DatasetType, dataset_data_dir: Path, warn_if_exists: Optional[Callable[[str], bool]] = None,):
        return self.datasets.create_dataset(dataset_name, dataset_type, dataset_data_dir, warn_if_exists)
    def add_dataset(self, dataset_name: str, dataset: Dataset, warn_if_exists: Optional[Callable[[str], bool]] = None, overwrite_existing: bool = True,):
        return self.datasets.add_dataset(dataset_name, dataset, warn_if_exists, overwrite_existing)
    def save_dataset(self, dataset_name: str) -> None:
        self.datasets.save_dataset(dataset_name)
    def show_dataset(self, dataset_name: Optional[str] = None, verbose: bool = False):
        self.datasets.show_dataset(dataset_name, verbose)
    def find_dataset_by_file_path(self, absolute_path: Path) -> Optional[Dataset]:
        return self.datasets.find_dataset_by_file_path(absolute_path)



    # ================================================================= #
    # HPO Functions                                                     #
    # ================================================================= #
    def get_hpo_runs(self) -> list[str]:
        hpo_dir = self.base_dir / "runs" / "hpo"
        if not hpo_dir.exists():
            return []
        return sorted([d.name for d in hpo_dir.iterdir() if d.is_dir()])

    def get_hpo_study_path(self, run_name: str) -> Path:
        return self.base_dir / "runs" / "hpo" / run_name / "study.db"

    def get_models_by_project_for_workflow(self, workflow: Workflow) -> dict[str, list[str]]:
        """Get all models grouped by project for a given workflow."""
        for service in self.clients.values():
            if service.SERVICE_NAME == workflow.value:
                return service.file_operations.models.get_models_by_project()

        raise ValueError(f"Unknown workflow: {workflow}")

    # ================================================================= #
    # Static Workspace Functions (Delegated)                            #
    # ================================================================= #

    @classmethod
    def create(
        cls, base_dir: Path, warn_if_exists: Optional[Callable[[str], bool]] = None
    ) -> Optional["Workspace"]:
        logging.info(f"Creating workspace base directory: {base_dir}")
        if not BaseOperationHelper._delete_if_existing(base_dir, "workspace", warn_if_exists):
            return None

        os.makedirs(base_dir, exist_ok=True)
        logging.info("Created new workspace base directory.")

        return Workspace(base_dir)

    @classmethod
    def delete(
        cls, base_dir: Path, warn_if_exists: Optional[Callable[[str], bool]] = None
    ) -> bool:
        if not base_dir.exists():
            logging.info(f"Workspace base directory does not exist: {base_dir}")
            return False
        else:
            logging.info(f"Deleting workspace base directory: {base_dir}")
            return BaseOperationHelper._delete_if_existing(base_dir, "workspace", warn_if_exists)