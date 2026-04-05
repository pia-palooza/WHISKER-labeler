import logging
from typing import Optional, List, Dict, Tuple
import pandas as pd
import numpy as np

from PyQt6.QtCore import QObject, QPointF, pyqtSignal

from whisker.core import Project
from whisker.core.workflows.pose_estimation.data_structures import PoseDataset, VisibilityFlag
from whisker.core.workflows.pose_estimation.operations.label_operations import PoseLabelOperations


class PoseLabelingModel(QObject):
    """
    Manages the data and business logic for the pose labeling widget.

    This class centralizes data modification and state management (e.g., dirty state)
    to follow a Model-View pattern. The widget (View) observes this model and
    sends user actions to it, and the model emits signals when data changes,
    prompting the view to update.
    """

    model_updated = pyqtSignal()
    dirty_state_changed = pyqtSignal(bool)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._pose_labels: Optional[PoseDataset] = None
        self._project: Optional[Project] = None
        self._current_image_key: Optional[str] = None
        self._current_dataset_name: Optional[str] = None

        # Stores dirty dataframes for (dataset_name, image_key) -> dataframe
        self._dirty_data_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
        
        # This holds the data for *only* the currently viewed image
        self._current_image_df: Optional[pd.DataFrame] = None
        self._current_image_dirty = False

    def is_dirty(self) -> bool:
        """Returns True if the model has unsaved changes."""
        return bool(self._dirty_data_cache) or self._current_image_dirty

    def set_dirty(self, dirty: bool):
        """Sets the dirty state for the current image and emits a signal if the global dirty state changes."""
        was_dirty = self.is_dirty()
        self._current_image_dirty = dirty

        if self.is_dirty() != was_dirty:
            self.dirty_state_changed.emit(dirty)

    def set_data(
        self,
        pose_labels: Optional[PoseDataset],
        image_key: Optional[str],
        project: Optional[Project],
        dataset_name: Optional[str] = None,
    ):
        """
        Sets the model's context to a specific image within a dataset and project.
        This also synchronizes the annotation structure with the project's settings.
        """
        # Save pending changes for the previous image into cache
        if self._current_image_dirty and self._current_dataset_name and self._current_image_key and self._current_image_df is not None:
            entry = (self._current_dataset_name, self._current_image_key)
            logging.info(f"Caching pending pose label changes for {entry}")
            self._dirty_data_cache[entry] = self._current_image_df.copy()
            self.set_dirty(True)

        self._pose_labels = pose_labels
        self._project = project
        self._current_image_key = image_key
        self._current_dataset_name = dataset_name
        self._current_image_df = None # Clear old data

        if pose_labels and image_key and project and dataset_name:
            # Check if we have a dirty version in cache
            if (dataset_name, image_key) in self._dirty_data_cache:
                self._current_image_df = self._dirty_data_cache[(dataset_name, image_key)].copy()
                self._current_image_dirty = True # It came from cache, so it is dirty relative to disk
            else:
                try:
                    # Select all rows for the current image
                    existing_data_df = pose_labels.keypoint_data.loc[image_key]
                except (KeyError, AttributeError):
                    # This image has no labels in the DataFrame yet

                    # DEV_NOTE: Specify dtypes to match the main PoseDataset dataframe.
                    # This prevents a TypeError during HDF5 serialization.
                    column_dtypes = {
                        'x': 'float32',
                        'y': 'float32',
                        'c': 'float32',
                    }
                    existing_data_df = pd.DataFrame(columns=['x', 'y', 'c']).astype(column_dtypes)
                
                self._synchronize_df(existing_data_df)
                self._current_image_dirty = False

        else:
             self._current_image_dirty = False
        
        self.model_updated.emit()

    def _synchronize_df(self, df: pd.DataFrame):
        """
        Ensures the local DataFrame slice matches the project's identities 
        and body parts, preserving existing labels.
        """
        if not self._project:
            self._current_image_df = pd.DataFrame(columns=['x', 'y', 'c'])
            return

        # Create the canonical index for this image based on the project
        target_index = pd.MultiIndex.from_product(
            [self._project.identities, self._project.body_parts], 
            names=['individual_id', 'body_part']
        )
        
        # Reindex the existing data to match the project.
        # This adds rows for new (ind, bp) pairs as all-NaN
        # and drops rows for (ind, bp) pairs not in the project.
        self._current_image_df = df.reindex(target_index)

    def get_current_image_data(self) -> Optional[pd.DataFrame]:
        return self._current_image_df

    def get_project(self) -> Optional[Project]:
        return self._project

    def update_keypoint_position(self, identity_id: str, part_idx: int, pos: QPointF):
        """Updates a specific keypoint's position and visibility."""
        if self._current_image_df is None or self._project is None:
            return

        try:
            # Handle both string names and integer indices for part_idx
            if isinstance(part_idx, str):
                 part_name = part_idx
            else:
                 part_name = self._project.body_parts[part_idx]

            idx = (identity_id, part_name)
            
            # Set x, y, and c. Mark as VISIBLE.
            self._current_image_df.loc[idx, ['x', 'y', 'c']] = [
                np.float32(pos.x()),
                np.float32(pos.y()),
                VisibilityFlag.VISIBLE
            ]

            self.set_dirty(True)
            self.model_updated.emit()
        except (IndexError, KeyError):
            logging.warning(f"Could not update keypoint: {identity_id}, {part_idx}")

    def accept_predictions(self, prediction_df: pd.DataFrame, min_conf: float = 0.0):
        if self._current_image_df is None or prediction_df.empty:
            logging.warning("[Primer Model] Cannot accept predictions: target df is None or prediction df is empty.")
            return

        logging.info(f"[Primer Model] Parsing prediction dataframe (type: {type(prediction_df)})")
        
        # Robust handling in case the slice collapsed to a Series
        if isinstance(prediction_df, pd.Series):
            prediction_df = prediction_df.to_frame().T

        valid_preds = prediction_df[prediction_df['c'] > min_conf]
        logging.info(f"[Primer Model] Found {len(valid_preds)} valid points > {min_conf} conf.")
        
        accepted_count = 0
        for idx, row in valid_preds.iterrows():
            try:
                # idx should be (individual_id, body_part)
                self._current_image_df.loc[idx, ['x', 'y', 'c']] = [
                    np.float32(row['x']),
                    np.float32(row['y']),
                    VisibilityFlag.VISIBLE
                ]
                accepted_count += 1
            except KeyError:
                logging.warning(f"[Primer Model] Key '{idx}' not found in current image DataFrame.")

        logging.info(f"[Primer Model] Successfully accepted {accepted_count} points into active state.")
        self.set_dirty(True)
        self.model_updated.emit()

    def clear_keypoint(self, identity_id: str, part_idx: int):
        """Clears a keypoint by setting its data to NaN."""
        if self._current_image_df is None or self._project is None:
            return

        try:
            # Handle both string names and integer indices for part_idx
            if isinstance(part_idx, str):
                 part_name = part_idx
            else:
                 part_name = self._project.body_parts[part_idx]

            idx = (identity_id, part_name)
            
            # Set all values to NaN to mark as unlabeled/unknown
            self._current_image_df.loc[idx, ['x', 'y', 'c']] = [np.nan, np.nan, np.nan]
            
            self.set_dirty(True)
            self.model_updated.emit()
        except (IndexError, KeyError):
            logging.warning(f"Could not clear keypoint: {identity_id}, {part_idx}")

    def clear_all_keypoints(self):
        """Clears all keypoints for the current image."""
        if self._current_image_df is None:
            return

        self._current_image_df.loc[:, ['x', 'y', 'c']] = np.nan
        self.set_dirty(True)
        self.model_updated.emit()

    def swap_identities(self, identity1: str, identity2: str):
        """
        Swaps all keypoints between two individuals in the current frame.
        """
        if self._current_image_df is None:
            return

        try:
            # We need to swap data for all body parts
            # Since the index is MultiIndex(individual_id, body_part), we can select by individual_id

            # Copy data for identity 1 and identity 2
            # Use .copy() to ensure we don't hold references that get overwritten mid-swap
            data1 = self._current_image_df.loc[identity1].copy()
            data2 = self._current_image_df.loc[identity2].copy()

            # Assign data2 to identity 1's rows
            # We iterate body parts to ensure we assign to the correct (id, part) index
            for body_part in data1.index:
                if body_part in data2.index:
                    self._current_image_df.loc[(identity1, body_part)] = data2.loc[body_part]

            # Assign data1 to identity 2's rows
            for body_part in data2.index:
                if body_part in data1.index:
                    self._current_image_df.loc[(identity2, body_part)] = data1.loc[body_part]

            self.set_dirty(True)
            self.model_updated.emit()

        except KeyError as e:
            logging.error(f"Failed to swap identities {identity1} and {identity2}: {e}")


    def save(self, pose_label_operations: PoseLabelOperations):
        """
        Commits all pending changes (from the current image and the cache) to their
        respective datasets in the workspace.
        """
        # 1. Ensure current image is in cache if dirty
        if (
            self._current_image_dirty
            and self._current_dataset_name
            and self._current_image_key
            and self._current_image_df is not None
        ):
             self._dirty_data_cache[(self._current_dataset_name, self._current_image_key)] = self._current_image_df.copy()

        if not self._dirty_data_cache:
            return

        # 2. Group changes by dataset
        changes_by_dataset: Dict[str, List[Tuple[str, pd.DataFrame]]] = {}
        for (d_name, i_key), df in self._dirty_data_cache.items():
            if d_name not in changes_by_dataset:
                changes_by_dataset[d_name] = []
            changes_by_dataset[d_name].append((i_key, df))

        # 3. Process each dataset
        for dataset_name, changes in changes_by_dataset.items():
            pose_labels = pose_label_operations.get_pose_dataset(dataset_name)

            if not pose_labels:
                logging.error(f"PoseLabelingModel.save: No pose label dataset found for '{dataset_name}'. Skipping save for this dataset.")
                continue

            # Update metadata
            if self._project:
                pose_labels.body_parts = self._project.body_parts
                pose_labels.individuals = self._project.identities
                    
            for image_key, image_df in changes:
                # Same logic as before to update single image data
                labeled_data_df = image_df.dropna(subset=['c'])

                try:
                    pose_labels.keypoint_data = pose_labels.keypoint_data.drop(
                        index=image_key, level='frame_index'
                    )
                except KeyError:
                    pass

                if not labeled_data_df.empty:
                    new_index = pd.MultiIndex.from_arrays(
                        [
                            [image_key] * len(labeled_data_df),
                            labeled_data_df.index.get_level_values('individual_id'),
                            labeled_data_df.index.get_level_values('body_part')
                        ],
                        names=['frame_index', 'individual_id', 'body_part']
                    )
                    labeled_data_df.index = new_index

                    pose_labels.keypoint_data = pd.concat(
                        [pose_labels.keypoint_data, labeled_data_df]
                    )
            
            # Sort once after all updates for this dataset
            pose_labels.keypoint_data.sort_index(inplace=True)

            # Trigger workspace save for this dataset
            pose_label_operations.save_pose_labels(dataset_name)

        # 4. Clear cache and dirty state
        self._dirty_data_cache.clear()
        self._current_image_dirty = False
        self.dirty_state_changed.emit(False)
