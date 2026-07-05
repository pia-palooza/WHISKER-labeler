from typing import Dict, Type, Optional
from pydantic import BaseModel
from whisker.base.job import BaseJob

class JobSchema(BaseModel):
    name: str
    description: str
    params_type: Type[BaseModel]

class JobRegistry:
    _jobs: Dict[str, JobSchema] = {}
    _job_classes: Dict[str, Type[BaseJob]] = {}
    _initialized = False

    @classmethod
    def _ensure_registered(cls):
        if cls._initialized:
            return
        cls._initialized = True
        try:
            from whisker.core.workers.subsample_dataset import SubsampleDatasetJob
            from whisker.services.pose_estimation.internal.workers.pose_export import PoseExportJob
            from whisker.services.behavior_classification.internal.workers.behavior_export import BehaviorExportJob
        except ImportError as e:
            import logging
            logging.warning(f"Could not import some jobs dynamically: {e}")

    @classmethod
    def register(cls, name: str, description: str, params_type: Type[BaseModel]):
        """Decorator factory to register a BaseJob class."""
        def decorator(job_class: Type[BaseJob]) -> Type[BaseJob]:
            cls._jobs[name] = JobSchema(name=name, description=description, params_type=params_type)
            cls._job_classes[name] = job_class
            return job_class
        return decorator

    @classmethod
    def get_job_schema(cls, name: str) -> Optional[JobSchema]:
        cls._ensure_registered()
        return cls._jobs.get(name)

    @classmethod
    def get_job_class(cls, name: str) -> Optional[Type[BaseJob]]:
        cls._ensure_registered()
        return cls._job_classes.get(name)

    @classmethod
    def list_jobs(cls) -> Dict[str, JobSchema]:
        cls._ensure_registered()
        return cls._jobs


