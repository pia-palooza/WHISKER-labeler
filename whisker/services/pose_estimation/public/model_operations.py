import logging
import os
from pathlib import Path

from whisker.core.base_workflow_operation_helper import BaseWorkflowOperationsHelper
from whisker.core.study.dataset_operations import DatasetOperations
from whisker.core.study.project_operations import ProjectOperations
import yaml

class PoseModelOperations(BaseWorkflowOperationsHelper):
    def __init__(
        self,
        base_dir: Path,
        projects: ProjectOperations,
        datasets: DatasetOperations
    ):
        super().__init__(base_dir, projects, datasets)
        self._projects: ProjectOperations = projects
        self._datasets: DatasetOperations = datasets
        self._pose_models: list[str] = []
        self._pose_model_bodyparts_cache: dict[str, list[str]] = {}

    def scan_models(self):
        """Scans for existing pose models."""
        self._pose_models.clear()
        if self._base_dir.exists():
            self._pose_models = [
                d for d in os.listdir(self._base_dir) 
                if (self._base_dir / d).is_dir() and not d.startswith(".")
            ]
            self._pose_model_bodyparts_cache = {}
            for model_name in self._pose_models:
                self._pose_model_bodyparts_cache[model_name] = self._get_model_body_parts(model_name)

    def has(self, model_name: str):
        return model_name in self._pose_models
    
    def get(self) -> list[str]:
        """Returns a list of all pose model names."""
        return list(self._pose_models)

    def get_models_by_project(self) -> dict[str, list[str]]:
        """Get all models grouped by project for pose estimation workflow."""
        models_by_project: dict[str, list[str]] = {}
        for project in self._projects.values():
            project_body_parts = set(project.body_parts)

            matching_models = []
            for model_name in self._pose_models:
                model_body_parts = set(self._get_model_body_parts(model_name))
                if model_body_parts == project_body_parts:
                    matching_models.append(model_name)
            
            logging.info(f"Found {len(matching_models)} matching models for project '{project.name}'")
            if matching_models:
                models_by_project[project.name] = matching_models
    
        return models_by_project

    def _get_model_body_parts(self, model_name: str) -> list[str]:
        if model_name in self._pose_model_bodyparts_cache:
            return self._pose_model_bodyparts_cache[model_name]

        metadata = self.get_training_config(model_name)
        if metadata:
            self._pose_model_bodyparts_cache[model_name] = metadata.get('bodyparts', [])
            return metadata.get('bodyparts', [])
        return []

    def get_training_config(self, model_name: str) -> dict | None:
        metadata_path = self._base_dir / model_name / "metadata.yaml"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return yaml.safe_load(f)
        return None

    def get_training_metadata(self, model_name: str) -> dict | None:
        """Parsed ``training_metadata.json`` for the run, or None if absent.

        The file is written by PoseTrainingJob and includes
        ``source_datasets`` — the dataset names the model was trained on.
        Used by the Figure Maker pose page's per-row auto-detect Train
        default.
        """
        import json
        meta_path = self._base_dir / model_name / "training_metadata.json"
        if not meta_path.exists():
            return None
        try:
            with open(meta_path, 'r') as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    def get_source_datasets(self, model_name: str) -> list[str]:
        """Names of the datasets a model was trained on (empty if unknown)."""
        meta = self.get_training_metadata(model_name)
        if not meta:
            return []
        out = meta.get("source_datasets") or []
        return [str(s) for s in out if isinstance(s, str)]

    def get_training_history(self, model_name: str) -> list[dict] | None:
        history_path = self._base_dir / model_name / "checkpoints" / "training_metrics.csv"
        if history_path.exists():
            import pandas as pd
            df = pd.read_csv(history_path)
            return df.to_dict(orient="records")
        
        # Fallback: try loading from best_model.pth
        checkpoint_path = self._base_dir / model_name / "checkpoints" / "best_model.pth"
        if checkpoint_path.exists():
            import torch
            try:
                # Load on CPU and only the metadata if possible, but torch.load loads the whole thing
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                history = checkpoint.get("history")
                if history:
                    # Convert history dict of lists to list of dicts
                    epochs = history.get("epochs", [])
                    records = []
                    for i, epoch in enumerate(epochs):
                        record = {"epochs": epoch}
                        for k, v in history.items():
                            if k != "epochs" and i < len(v):
                                record[k] = v[i]
                        records.append(record)
                    return records
            except Exception as e:
                logging.warning(f"Failed to load history from checkpoint {checkpoint_path}: {e}")
        
        return None

    def get_split_info(self, model_name: str) -> dict | None:
        split_path = self._base_dir / model_name / "split_data.yaml"
        if split_path.exists():
            with open(split_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Fallback: Count samples in H5 files
        data_dir = self._base_dir / model_name / "data"
        if data_dir.exists():
            try:
                import pandas as pd
                train_h5 = data_dir / "train_data.h5"
                val_h5 = data_dir / "val_data.h5"
                
                info = {"split_data": {"train": [], "val": []}}
                if train_h5.exists():
                    with pd.HDFStore(train_h5, 'r') as store:
                        # For 'table' format, we can check nrows
                        strio = store.get_storer('keypoints')
                        if strio:
                            count = strio.nrows
                            info["split_data"]["train"] = [None] * count
                if val_h5.exists():
                    with pd.HDFStore(val_h5, 'r') as store:
                        strio = store.get_storer('keypoints')
                        if strio:
                            count = strio.nrows
                            info["split_data"]["val"] = [None] * count
                return info
            except Exception as e:
                logging.warning(f"Failed to estimate dataset size from H5 files: {e}")
        
        return None

    def prune(self) -> dict[Path, str]:
        """
        Scans for corrupt or incomplete pose models.
        Returns:
            dict[Path, str]: Mapping of folder path to reason for pruning.
        """
        findings = {}
        if not self._base_dir.exists():
            return findings

        for model_name in os.listdir(self._base_dir):
            if model_name.startswith("."): continue
            
            model_dir = self._base_dir / model_name
            if not model_dir.is_dir(): continue

            metadata_path = model_dir / "metadata.yaml"
            
            # 1. Missing Metadata
            if not metadata_path.exists():
                findings[model_dir] = "Missing metadata.yaml"
                continue

            # 2. Corrupt Metadata
            try:
                with open(metadata_path, 'r') as f:
                    yaml.safe_load(f)
            except Exception as e:
                findings[model_dir] = f"Corrupt metadata.yaml: {e}"

        return findings