import logging
from typing import Any, Type, Dict, Optional, Type
from pathlib import Path
from types import MethodType
from enum import Enum

from ..core.study.dataset_operations import DatasetOperations
from ..core.study.project_operations import ProjectOperations


_SERVICE_REGISTRY: Dict[str, Type] = {}

def register_service(service: Enum | str):
    """Registers a service client class to a service string or Enum value."""
    logging.debug(f"Registering service: {service}")
    def wrapper(cls: Type) -> Type:
        service_str = service.value if isinstance(service, Enum) else service
        cls.SERVICE_NAME = service_str
        _SERVICE_REGISTRY[service_str] = cls
        return cls
    return wrapper

def get_type_for_service(service: str) -> Optional[Type]:
    return _SERVICE_REGISTRY.get(service)

def make_service_clients(base_dir: Path, projects: ProjectOperations, datasets: DatasetOperations) -> dict[str, Any]:
    return {
        key : value.make_service_client(base_dir, projects, datasets)
        for key, value in _SERVICE_REGISTRY.items()
    }

def inject_workspace_attributes(workspace: Any) -> Any:
    """
    Dynamically routes and injects attributes. 
    Properties are attached to the instance's class, while callables are bound as methods.
    """
    workspace_class = workspace.__class__

    for service_cls in _SERVICE_REGISTRY.values():
        attrs = service_cls.get_workspace_attributes()
        logging.debug(f"Injecting workspace methods from {service_cls.__name__}: {list(attrs.keys())}")
        
        for name, attr in attrs.items():
            if isinstance(attr, property):
                # Properties MUST be set on the class to trigger descriptor lookup
                setattr(workspace_class, name, attr)
            elif callable(attr):
                # Regular functions get bound to the specific instance
                setattr(workspace, name, MethodType(attr, workspace))
            else:
                # Fallback for any static configurations or constants
                setattr(workspace, name, attr)
                
    return workspace