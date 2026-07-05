from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QTreeWidgetItem

from whisker.core.workspace import Workspace, DatasetType
from .constants import (
    ItemGroupEnum,
    ItemTypeEnum,
    LABELED_INDICATOR,
    PARTIALLY_LABELED_INDICATOR,
    NOT_LABELED_INDICATOR,
    LABELED_NOT_APPLICABLE_INDICATOR,
    LABELED_DIRTY_STATE_INDICATOR 
)

# DEV_NOTE: This matches the dummy text in tree_controller.
# Kept here to strip it from hierarchy paths if present.
_DUMMY_CHILD_TEXT = "Loading..."


def get_item_hierarchy(item: QTreeWidgetItem) -> list[str]:
    """
    Returns a list of the hierarchy of items, starting from the root.
    For leaf frame nodes, uses the full relative path stored in UserRole.
    """
    hierarchy = []
    temp_item = item
    while temp_item:
        # Use stored full relative path for leaf frame nodes
        user_data = temp_item.data(0, Qt.ItemDataRole.UserRole)
        text_to_add = user_data if user_data is not None else temp_item.text(0)
        
        cleaned_text = (
            str(text_to_add)
            .replace(LABELED_INDICATOR, "")
            .replace(PARTIALLY_LABELED_INDICATOR, "")
            .replace(NOT_LABELED_INDICATOR, "")
            .replace(LABELED_NOT_APPLICABLE_INDICATOR, "")
            .replace(LABELED_DIRTY_STATE_INDICATOR, "")
            .replace(_DUMMY_CHILD_TEXT, "")
            .lstrip()
        )
        hierarchy.append(cleaned_text)
        temp_item = temp_item.parent()
    return hierarchy[::-1]


def infer_selection_item_type(
    item_group: ItemGroupEnum,
    tree_item: QTreeWidgetItem,
    workspace: Workspace,
) -> ItemTypeEnum:
    """Determines the specific ItemTypeEnum based on tree depth and dataset type."""
    level = 0
    temp_item = tree_item
    while temp_item.parent() is not None:
        temp_item = temp_item.parent()
        level += 1

    match item_group.value:
        case ItemGroupEnum.WORKSPACE_FILES.value:
            is_dir = tree_item.childCount() > 0
            return (
                ItemTypeEnum.WORKSPACE_DIR 
                if is_dir 
                else ItemTypeEnum.WORKSPACE_FILE
            )

        case ItemGroupEnum.DATASETS.value:
            hierarchy = get_item_hierarchy(tree_item)
            if len(hierarchy) < 2:
                return ItemTypeEnum.DATASET_TYPE

            # dataset name is always the second item in the hierarchy
            dataset_name = hierarchy[1]
            dataset = workspace.datasets.get(dataset_name)
            if not dataset:
                raise ValueError(f"Dataset '{dataset_name}' not found.")
            
            return _resolve_dataset_level_type(level, dataset.type)
        case ItemGroupEnum.MODELS.value:
            root_item = tree_item
            while root_item.parent() is not None:
                root_item = root_item.parent()
            root_text = root_item.text(0)

            if root_text == "Pose Estimation Models":
                if tree_item.parent() is None:
                    return ItemTypeEnum.POSE_ESTIMATION_MODEL_PROJECT
                else:
                    return ItemTypeEnum.POSE_ESTIMATION_MODEL
            elif root_text == "Behavior Models":
                if tree_item.parent() is None:
                    return ItemTypeEnum.BEHAVIOR_CLASSIFICATION_MODEL_PROJECT
                else:
                    return ItemTypeEnum.BEHAVIOR_CLASSIFICATION_MODEL
            elif root_text == "Animal Detection Models":
                if tree_item.parent() is None:
                    return ItemTypeEnum.ANIMAL_DETECTION_MODEL_PROJECT
                else:
                    return ItemTypeEnum.ANIMAL_DETECTION_MODEL
            else:
                raise ValueError(f"Unknown root node text: {root_text}")
        case ItemGroupEnum.PREDICTIONS.value:
            root_item = tree_item
            while root_item.parent() is not None:
                root_item = root_item.parent()
            root_text = root_item.text(0)

            if root_text == "Pose Estimation Predictions":
                if tree_item.parent() is None:
                    return ItemTypeEnum.POSE_ESTIMATION_PREDICTION_PROJECT
                else:
                    return ItemTypeEnum.POSE_ESTIMATION_PREDICTION
            elif root_text == "Behavior Predictions":
                if tree_item.parent() is None:
                    return ItemTypeEnum.BEHAVIOR_CLASSIFICATION_PREDICTION_PROJECT
                else:
                    return ItemTypeEnum.BEHAVIOR_CLASSIFICATION_PREDICTION
            elif root_text == "Animal Detection Predictions":
                if tree_item.parent() is None:
                    return ItemTypeEnum.ANIMAL_DETECTION_PREDICTION_PROJECT
                else:
                    return ItemTypeEnum.ANIMAL_DETECTION_PREDICTION
            else:
                raise ValueError(f"Unknown root node text: {root_text}")
        case _:
            raise ValueError(f"Invalid item group: {item_group.value}")


def _resolve_dataset_level_type(level: int, ds_type: DatasetType) -> ItemTypeEnum:
    """Helper to map tree depth to item type based on dataset definition."""
    match level:
        case 0:
            return ItemTypeEnum.DATASET_TYPE
        case 1:
            return ItemTypeEnum.DATASET_BASE
        case 2:
            if ds_type == DatasetType.IMAGE_COLLECTION:
                return ItemTypeEnum.DATASET_IMAGE
            if ds_type == DatasetType.VIDEO_COLLECTION:
                return ItemTypeEnum.DATASET_VIDEO
            if ds_type == DatasetType.FRAME_SUBSET:
                return ItemTypeEnum.DATASET_VIDEO_FRAME_SUBSET
            raise ValueError(f"Unhandled dataset type {ds_type} at level 2")
        case 3:
            if ds_type == DatasetType.FRAME_SUBSET:
                return ItemTypeEnum.DATASET_VIDEO_FRAME
            raise ValueError(f"Invalid selection at level 3 for {ds_type}")
        case _:
            raise ValueError(f"Invalid level for dataset selection: {level}")


def get_labeled_indicator(
    is_fully_labeled: bool, 
    is_partially_labeled: bool = False,
    is_not_applicable: bool = False,
    is_dirty_state: bool = False
) -> str:
    if is_dirty_state:
        return LABELED_DIRTY_STATE_INDICATOR
    
    if is_not_applicable:
        return LABELED_NOT_APPLICABLE_INDICATOR
    
    if is_fully_labeled:
        return LABELED_INDICATOR
    elif is_partially_labeled:
        return PARTIALLY_LABELED_INDICATOR
    return NOT_LABELED_INDICATOR