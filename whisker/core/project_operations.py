import logging
import os
from pathlib import Path
from typing import Optional, Callable

from .base_operation_helper import BaseOperationHelper
from .project import Project

class ProjectOperations(BaseOperationHelper):
    def __init__(self, base_dir: Path):
        super().__init__(base_dir)
        self._projects: dict[str, Project] = {}

    def scan_projects(self):
        """Scans the projects directory and reloads metadata."""
        self._projects.clear()
        if not self._base_dir.exists(): return

        for project_file in os.listdir(self._base_dir):
            if not project_file.endswith(".json"): continue
            
            project_name = Path(project_file).stem
            project_path = self._base_dir / project_file
            new_project = None
            try:
                with open(project_path, "rb") as f:
                    new_project = Project.from_json(f.read().decode())
                    self._projects[project_name] = new_project
            except Exception as e:
                logging.error(f"Failed to load project {project_name}: {e}")

            if new_project:
                logging.info(
                    f"[ProjectOperations] Loaded project {new_project.name}\n"
                    f"  {len(new_project.body_parts):3d} Body Parts : {new_project.body_parts}\n"
                    f"  {len(new_project.identities):3d} Identities : {new_project.identities}"
                )

    def create_project(
        self,
        project_name: str,
        body_parts: list[str] = [],
        identities: list[str] = [],
        skeleton: list[tuple[str, str]] = [],
        behaviors: list[str] = [],
        warn_if_exists: Optional[Callable[[str], bool]] = None,
    ) -> Optional[Project]:
        if project_name in self._projects:
            if warn_if_exists:
                logging.warning(f"Project {project_name} already exists!")
                if not warn_if_exists(f"Overwrite existing project at {project_name}?"):
                    return None

        self._projects[project_name] = Project(
            name=project_name,
            body_parts=body_parts,
            skeleton=skeleton,
            identities=identities,
            behaviors=behaviors,
        )
        self.save_project(project_name)
        logging.info(f"Added project {project_name} to workspace")

    def save_project(self, project_name: str):
        project = self._projects[project_name]
        project_path = self._base_dir / f"{project_name}.json"
        os.makedirs(self._base_dir, exist_ok=True)
        with open(project_path, "w", encoding="utf-8") as f:
            json_str = project.model_dump_json(indent=4)
            f.write(json_str)
        logging.info(f"Saved project {project_name} to {project_path}")

    def get(self, name: str):
        return self._projects.get(name)
    
    def keys(self) -> list[str]:
        return list(self._projects.keys())
    
    def values(self) -> list[Project]:
        return list(self._projects.values())

