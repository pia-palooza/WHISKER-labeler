import logging
import os
from pathlib import Path
from typing import Optional
import pandas as pd

from whisker.core.base_workflow_operation_helper import BaseWorkflowOperationsHelper
from whisker.core.study.dataset_operations import DatasetOperations
from whisker.core.study.project_operations import ProjectOperations
from .data_structures import BehaviorDataset

class BehaviorPredictionOperations(BaseWorkflowOperationsHelper):
    def __init__(
        self,
        base_dir: Path,
        projects: ProjectOperations,
        datasets: DatasetOperations
    ):
        super().__init__(base_dir, projects, datasets)
        self._projects: ProjectOperations = projects
        self._datasets: DatasetOperations = datasets
        self._behavior_predictions: dict[str, dict[str, dict[str, Optional[BehaviorDataset]]]] = {}

    def scan_predictions(self):
        """Scans for existing behavior prediction runs (lazy load)."""
        self._behavior_predictions.clear()
        if self._base_dir.exists():
            for run_name in os.listdir(self._base_dir):
                run_dir = self._base_dir / run_name
                if not run_dir.is_dir() or run_name.startswith("."): continue

                self._behavior_predictions[run_name] = {}
                for dataset_name in os.listdir(run_dir):
                    dataset_dir = run_dir / dataset_name
                    if not dataset_dir.is_dir() or dataset_name.startswith("."): continue

                    self._behavior_predictions[run_name][dataset_name] = {}
                    for video_stem in os.listdir(dataset_dir):
                        video_dir = dataset_dir / video_stem
                        if not video_dir.is_dir(): continue
                        
                        if (video_dir / "predictions.h5").exists():
                            self._behavior_predictions[run_name][dataset_name][video_stem] = None

    def has_behavior_prediction(
        self,
        training_run_name: str,
        dataset_name: str,
        video_stem: str = ''
    ) -> bool:
        all_predictions_for_training_run = (
            self._behavior_predictions.get(training_run_name, {})
        )
        if video_stem:
            return video_stem in all_predictions_for_training_run.get(dataset_name, {})
        else:
            return dataset_name in all_predictions_for_training_run

    def get_behavior_predictions(
        self,
        training_run_name: str,
        dataset_name: str,
        video_stem: str,
        raise_if_missing: bool = True,
    ) -> Optional[BehaviorDataset]:
        """Core lazy-loading logic for behavior predictions."""
        dataset_predictions = self._behavior_predictions.get(training_run_name, {}).get(dataset_name, {})
        
        if video_stem not in dataset_predictions:
            if raise_if_missing:
                raise ValueError("No behavior predictions found.")
            return None

        if dataset_predictions[video_stem] is None:
            labels_path = (
                self._base_dir / training_run_name / dataset_name / video_stem / "predictions.h5"
            )

            if not labels_path.exists():
                if raise_if_missing:
                    raise FileNotFoundError(f"Behavior prediction file not found: {labels_path}")
                return None

            try:
                loaded_labels = BehaviorDataset.from_file(labels_path)
                dataset_predictions[video_stem] = loaded_labels
            except Exception as e:
                logging.error(f"Failed to load behavior predictions from {labels_path}: {e}", exc_info=True)
                if raise_if_missing: raise
                return None

        return dataset_predictions[video_stem]

    def get_behavior_predictions_for_run(self, run_name: str):
        return self._behavior_predictions.get(run_name)

    def get_behavior_predictions_for_dataset(self, run_name: str, dataset_name: str):
        return self._behavior_predictions.get(run_name, {}).get(dataset_name, {})

    def list_runs(self) -> list[str]:
        """Sorted names of all known prediction runs."""
        return sorted(self._behavior_predictions.keys())

    def list_runs_for_video(self, dataset_name: str, video_stem: str) -> set[str]:
        """Run names that have predictions for this (dataset, video_stem)."""
        return {
            run_name
            for run_name, datasets in self._behavior_predictions.items()
            if video_stem in datasets.get(dataset_name, {})
        }

    def add_video_behavior_predictions(
        self,
        training_run_name: str,
        dataset_name: str,
        video_stem: str,
        behavior_dataset: BehaviorDataset,
    ):
        output_dir = (
            self._base_dir
            / training_run_name
            / dataset_name
            / video_stem
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        labels_path = output_dir / "predictions.h5"
        
        try:
            behavior_dataset.to_file(labels_path)
            logging.info(f"Saved behavior predictions for {training_run_name}/{dataset_name}/{video_stem}")
        except Exception as e:
            logging.error(f"Failed to save behavior predictions to {labels_path}: {e}", exc_info=True)
            raise

        # Update the in-memory cache
        self._behavior_predictions.setdefault(training_run_name, {}).setdefault(
            dataset_name, {}
        )[video_stem] = behavior_dataset

    def get_behavior_probabilities(self, run_name: str, dataset_name: str, video_stem: str) -> Optional[pd.DataFrame]:
        preds = self.get_behavior_predictions(run_name, dataset_name, video_stem, raise_if_missing=False)
        return preds.per_frame_probabilities if preds else None

    def add_behavior_bouts(self, run_name: str, dataset_name: str, video_stem: str, bouts_df: pd.DataFrame):
        # --- FIX: Ensure video_key column exists if we have data ---
        if not bouts_df.empty and "video_key" not in bouts_df.columns:
            bouts_df = bouts_df.copy()
            bouts_df["video_key"] = video_stem
        # -----------------------------------------------------------

        existing_preds = self.get_behavior_predictions(
            run_name, dataset_name, video_stem, raise_if_missing=False
        )
        
        if existing_preds:
            existing_preds.bouts = bouts_df
            new_preds = existing_preds
        else:
            if not bouts_df.empty:
                behavior_names = list(bouts_df['behavior'].unique())
            else:
                behavior_names = []
                
            new_preds = BehaviorDataset(behaviors=behavior_names, bouts=bouts_df)

        self.add_video_behavior_predictions(
            training_run_name=run_name,
            dataset_name=dataset_name,
            video_stem=video_stem,
            behavior_dataset=new_preds
        )