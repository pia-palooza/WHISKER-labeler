# UPDATE_FILE: whisker/gui/widgets/data_explorer/tree_controller.py
import logging
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Set, List, Any, Tuple

from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtWidgets import (
    QTreeWidget,
    QTreeWidgetItem,
    QTreeWidgetItemIterator,
)

EMOJI_LABELS = ["⭐", "1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣"]

def get_model_settings_key(workspace_path: str, model_name: str) -> str:
    normalized_path = workspace_path.replace("\\", "/")
    return f"models/{normalized_path}/{model_name}"

def get_model_label(workspace_path: str, model_name: str) -> Optional[str]:
    settings = QSettings("whisker", "model_metadata")
    return settings.value(get_model_settings_key(workspace_path, model_name) + "/label", None, type=str)

def set_model_label(workspace_path: str, model_name: str, label: Optional[str]):
    settings = QSettings("whisker", "model_metadata")
    key = get_model_settings_key(workspace_path, model_name) + "/label"
    if label:
        settings.setValue(key, label)
    else:
        settings.remove(key)

def is_model_hidden(workspace_path: str, model_name: str) -> bool:
    settings = QSettings("whisker", "model_metadata")
    return settings.value(get_model_settings_key(workspace_path, model_name) + "/hidden", False, type=bool)

def set_model_hidden(workspace_path: str, model_name: str, hidden: bool):
    settings = QSettings("whisker", "model_metadata")
    key = get_model_settings_key(workspace_path, model_name) + "/hidden"
    if hidden:
        settings.setValue(key, True)
    else:
        settings.remove(key)

def format_model_display_name(workspace_path: str, model_name: str) -> str:
    lbl = get_model_label(workspace_path, model_name)
    if lbl:
        return f"{lbl} {model_name}"
    return model_name

def strip_model_label(text: str) -> str:
    if not text:
        return text
    for emoji in EMOJI_LABELS:
        if text.startswith(emoji):
            return text[len(emoji):].strip()
    return text

def get_prediction_settings_key(workspace_path: str, pred_name: str) -> str:
    normalized_path = workspace_path.replace("\\", "/")
    return f"predictions/{normalized_path}/{pred_name}"

def get_prediction_label(workspace_path: str, pred_name: str) -> Optional[str]:
    settings = QSettings("whisker", "prediction_metadata")
    return settings.value(get_prediction_settings_key(workspace_path, pred_name) + "/label", None, type=str)

def set_prediction_label(workspace_path: str, pred_name: str, label: Optional[str]):
    settings = QSettings("whisker", "prediction_metadata")
    key = get_prediction_settings_key(workspace_path, pred_name) + "/label"
    if label:
        settings.setValue(key, label)
    else:
        settings.remove(key)

def is_prediction_hidden(workspace_path: str, pred_name: str) -> bool:
    settings = QSettings("whisker", "prediction_metadata")
    return settings.value(get_prediction_settings_key(workspace_path, pred_name) + "/hidden", False, type=bool)

def set_prediction_hidden(workspace_path: str, pred_name: str, hidden: bool):
    settings = QSettings("whisker", "prediction_metadata")
    key = get_prediction_settings_key(workspace_path, pred_name) + "/hidden"
    if hidden:
        settings.setValue(key, True)
    else:
        settings.remove(key)

def format_prediction_display_name(workspace_path: str, pred_name: str) -> str:
    lbl = get_prediction_label(workspace_path, pred_name)
    if lbl:
        return f"{lbl} {pred_name}"
    return pred_name

def strip_prediction_label(text: str) -> str:
    if not text:
        return text
    for emoji in EMOJI_LABELS:
        if text.startswith(emoji):
            return text[len(emoji):].strip()
    return text

from whisker.core.workspace import Workspace, DatasetType, Dataset
from whisker.core.workflows.workflow_enum import Workflow
from .constants import (
    ItemGroupEnum, 
    ItemTypeEnum, 
    FilterOptions,
    PREDICTED_INDICATOR,
    PARTIALLY_PREDICTED_INDICATOR,
    NOT_PREDICTED_INDICATOR,
    PREDICTED_NOT_APPLICABLE_INDICATOR,
    LABELED_INDICATOR,
    PARTIALLY_LABELED_INDICATOR,
    NOT_LABELED_INDICATOR,
    LABELED_NOT_APPLICABLE_INDICATOR,
    LABELED_DIRTY_STATE_INDICATOR,
)
from .selection_logic import (
    get_item_hierarchy,
    infer_selection_item_type,
    get_labeled_indicator,
    _DUMMY_CHILD_TEXT
)

_JUNK_DIRS = {
    ".git", ".venv", "venv", ".vscode", "__pycache__", "node_modules",
    ".pytest_cache", ".mypy_cache", "build", "dist", ".egg-info",
}

ORDERED_DATASET_TYPES_BY_WORKFLOW = {
    Workflow.POSE_ESTIMATION: [
        DatasetType.IMAGE_COLLECTION,
        DatasetType.FRAME_SUBSET,
        DatasetType.VIDEO_COLLECTION,
    ],
    Workflow.BEHAVIOR_CLASSIFICATION: [
        DatasetType.VIDEO_COLLECTION,
    ],
}

class TreeItemIndicator(ABC):
    """
    Abstract base class for item indicators.
    Encapsulates logic for determining item status, icons, and filter data.
    """
    @abstractmethod
    def prepare(
        self, 
        workspace: Workspace, 
        dataset: Dataset, 
        workflow: Workflow,
        run_name: Optional[str] = None
    ) -> None:
        """
        Called once per dataset population to pre-calculate state.
        
        Args:
            workspace: The active workspace.
            dataset: The dataset being populated.
            workflow: The current active workflow.
            run_name: The currently selected training run/model name (optional).
        """
        pass

    @abstractmethod
    def get_indicator_state(self, paths: List[str]) -> Tuple[str, Any]:
        """
        Returns a tuple of (icon_string, filter_data).
        paths: A list of file paths associated with the item (single for leaf, multiple for group/dataset).
        """
        pass

    @property
    @abstractmethod
    def data_role(self) -> Optional[int]:
        """Returns the Qt DataRole used to store this indicator's state, or None."""
        pass


class LabelStatusIndicator(TreeItemIndicator):
    """
    Manages the 'Labeled' status, including rules for N/A, partial, and full labeling.
    """
    def __init__(self):
        # UserRole + 1 used for filtering state (Matches original LabeledStateRole)
        self._role = Qt.ItemDataRole.UserRole + 1
        self._labeled_keys: Set[str] = set()
        self._is_na = False
        self._dataset_type: Optional[DatasetType] = None
        self._workflow: Optional[Workflow] = None

    @property
    def data_role(self) -> int:
        return self._role

    def prepare(
        self, 
        workspace: Workspace, 
        dataset: Dataset, 
        workflow: Workflow,
        run_name: Optional[str] = None
    ) -> None:
        del run_name
        self._dataset_type = dataset.type
        self._workflow = workflow
        self._labeled_keys = self._get_labeled_keys(workspace, dataset, workflow)
        self._is_na = self._check_not_applicable(dataset, workflow)

    def get_indicator_state(self, paths: List[str]) -> Tuple[str, Any]:
        if not paths:
            return "", False

        # Normalize paths to keys based on dataset type
        keys = self._normalize_paths(paths)
        
        has_files = bool(keys)
        # Determine status
        is_fully_labeled = has_files and keys.issubset(self._labeled_keys)
        is_partially_labeled = bool(keys & self._labeled_keys)

        indicator_icon = get_labeled_indicator(
            is_fully_labeled,
            is_partially_labeled,
            is_not_applicable=self._is_na
        )
        
        # For filtering, we typically care if it is "fully labeled" (True) or not (False)
        return indicator_icon, is_fully_labeled

    def _normalize_paths(self, paths: List[str]) -> Set[str]:
        """Normalize file paths to keys expected by the label system."""
        if self._dataset_type == DatasetType.VIDEO_COLLECTION:
            return {Path(p).name for p in paths}
        else:
            # Image collections and Frame subsets use relative paths with forward slashes
            return {p.replace("\\", "/") for p in paths}

    def _get_labeled_keys(self, workspace: Workspace, dataset: Dataset, workflow: Workflow) -> Set[str]:
        match workflow:
            case Workflow.POSE_ESTIMATION:
                return workspace.get_pose_labeled_image_keys_from_summary(dataset.name)
            case Workflow.BEHAVIOR_CLASSIFICATION:
                return workspace.get_behavior_labeled_video_keys(dataset.name)
        return set()

    def _check_not_applicable(self, dataset: Dataset, workflow: Workflow) -> bool:
        match workflow:
            case Workflow.POSE_ESTIMATION:
                return dataset.type == DatasetType.VIDEO_COLLECTION
            case Workflow.BEHAVIOR_CLASSIFICATION:
                return dataset.type in [DatasetType.IMAGE_COLLECTION, DatasetType.FRAME_SUBSET]
            case _:
                return False


class PredictedStatusIndicator(TreeItemIndicator):
    """
    Manages the 'Predicted' status based on the selected run.
    """
    def __init__(self):
        # UserRole + 2 used for predicted state
        self._role = Qt.ItemDataRole.UserRole + 2
        self._predicted_keys: Set[str] = set()
        self._is_na = False
        self._dataset_type: Optional[DatasetType] = None
        self._run_name: Optional[str] = None

    @property
    def data_role(self) -> int:
        return self._role

    def prepare(
        self, 
        workspace: Workspace, 
        dataset: Dataset, 
        workflow: Workflow,
        run_name: Optional[str] = None
    ) -> None:
        self._dataset_type = dataset.type
        self._run_name = run_name
        
        # Prediction status is N/A if no run is selected, or if workflow constraints apply
        # However, if no run is selected, we usually just show nothing (handled in get_indicator_state)
        # Here we check if the DATASET TYPE is applicable for predictions in this workflow
        self._is_na = self._check_not_applicable(dataset, workflow)

        if not run_name or self._is_na:
            self._predicted_keys = set()
        else:
            self._predicted_keys = self._get_predicted_keys(workspace, dataset, workflow, run_name)

    def get_indicator_state(self, paths: List[str]) -> Tuple[str, Any]:
        # If no run is selected, do not show any indicator (even N/A) to avoid clutter
        if not self._run_name:
            return "", False

        if self._is_na:
            return PREDICTED_NOT_APPLICABLE_INDICATOR, False

        if not paths:
            return "", False

        keys = self._normalize_paths(paths)
        has_files = bool(keys)
        
        is_fully_predicted = has_files and keys.issubset(self._predicted_keys)
        is_partially_predicted = bool(keys & self._predicted_keys)

        if is_fully_predicted:
            return PREDICTED_INDICATOR, True
        elif is_partially_predicted:
            return PARTIALLY_PREDICTED_INDICATOR, False
        else:
            return NOT_PREDICTED_INDICATOR, False

    def _normalize_paths(self, paths: List[str]) -> Set[str]:
        if self._dataset_type == DatasetType.VIDEO_COLLECTION:
            # Predictions usually key off the video STEM (filename without extension)
            return {Path(p).stem for p in paths}
        else:
            # Images/Frame Subsets use relative paths with forward slashes
            return {p.replace("\\", "/") for p in paths}

    def _check_not_applicable(self, dataset: Dataset, workflow: Workflow) -> bool:
        # Predictions are generally applicable to all supported dataset types 
        # for a given workflow, but we can refine this if needed.
        # e.g. Behavior mostly applies to Videos.
        match workflow:
            case Workflow.BEHAVIOR_CLASSIFICATION:
                return dataset.type != DatasetType.VIDEO_COLLECTION
            case _:
                return False

    def _get_predicted_keys(
        self, workspace: Workspace, dataset: Dataset, workflow: Workflow, run_name: str
    ) -> Set[str]:
        predicted_keys = set()
        
        # Helper to get all video stems in dataset
        def _get_all_stems():
            return {Path(f).stem for f in dataset.files}

        match workflow:
            case Workflow.POSE_ESTIMATION:
                if dataset.type == DatasetType.VIDEO_COLLECTION:
                    # Check existence for each video
                    # Optimization: Iterate dataset files (stems) and check helper
                    for stem in _get_all_stems():
                        if workspace.has_pose_prediction(run_name, dataset.name, stem):
                            predicted_keys.add(stem)
                else:
                    # For images/frames, get the set of specific frame keys
                    predicted_keys = workspace.get_pose_prediction_frame_keys(run_name, dataset.name)

            case Workflow.BEHAVIOR_CLASSIFICATION:
                if dataset.type == DatasetType.VIDEO_COLLECTION:
                    for stem in _get_all_stems():
                        if workspace.has_behavior_prediction(run_name, dataset.name, stem):
                            predicted_keys.add(stem)

        return predicted_keys


class TreeController:
    """
    Manages the population, filtering, and navigation of the QTreeWidget.
    """
    def __init__(self, tree_widget: QTreeWidget):
        self.tree = tree_widget
        self._workspace: Optional[Workspace] = None
        self._blind_mode = False
        self._shuffled_file_maps: Dict[str, List[str]] = {}
        self._workflow: Optional[Workflow] = None
        self._run_name: Optional[str] = None
        
        # Initialize Indicators
        self._indicators: List[TreeItemIndicator] = [
            LabelStatusIndicator(),
            PredictedStatusIndicator()
        ]
        
        # Expose primary role for legacy access if needed
        self.LabeledStateRole = self._indicators[0].data_role

    def update_workspace(self, workspace: Optional[Workspace]):
        self._workspace = workspace

    def set_blind_mode(self, enabled: bool):
        self._blind_mode = enabled
        self._shuffled_file_maps.clear()

    def set_workflow(self, workflow):
        self._workflow = workflow
    
    def set_run_name(self, run_name: Optional[str]):
        """Sets the active run context for predictions."""
        self._run_name = run_name

    def current_workflow(self):
        return self._workflow

    def current_run_name(self) -> Optional[str]:
        return self._run_name

    # --- Population Logic ---

    def populate_tree(self, group: ItemGroupEnum):
        """Main entry point to clear and repopulate the tree."""
        logging.debug(f"TreeController populating tree for group {group}")
        # Preserve state
        expanded_paths = self._get_expanded_item_paths()
        selected_path = None
        if self.tree.currentItem():
            selected_path = tuple(get_item_hierarchy(self.tree.currentItem()))

        self.tree.clear()

        if not self._workspace:
            return

        match group.value:
            case ItemGroupEnum.WORKSPACE_FILES.value:
                root_path = self._workspace.base_dir
                root_item = QTreeWidgetItem([root_path.name])
                root_item.setToolTip(0, root_path.name)
                self.tree.addTopLevelItem(root_item)
                self._populate_workspace_directory_lazily(root_item, root_path)
                root_item.setExpanded(True)
            case ItemGroupEnum.DATASETS.value:
                self._populate_workspace_datasets()
            case ItemGroupEnum.MODELS.value:
                self._populate_workflow_items(group)
            case ItemGroupEnum.PREDICTIONS.value:
                self._populate_workflow_predictions(group)

        # Restore State
        self._restore_expanded_paths(expanded_paths)
        self._restore_selection(selected_path)

    def _populate_workflow_items(self, group: ItemGroupEnum):
        if not self._workflow or not self._workspace:
            return
        
        match self._workflow:
            case Workflow.POSE_ESTIMATION:
                models = self._workspace.pose_models.get()
                header = "Pose Estimation Models"
            case Workflow.BEHAVIOR_CLASSIFICATION:
                models = self._workspace.behavior_models.get()
                header = "Behavior Models"
            case _:
                return

        parent_item = QTreeWidgetItem([header])
        parent_item.setToolTip(0, header)
        self.tree.addTopLevelItem(parent_item)
        parent_item.setExpanded(True)
        
        workspace_path = str(self._workspace.base_dir)
        for model_name in sorted(models):
            is_hidden = is_model_hidden(workspace_path, model_name)
            label = get_model_label(workspace_path, model_name)
            display_name = f"{label} {model_name}" if label else model_name

            model_item = QTreeWidgetItem([display_name])
            model_item.setToolTip(0, display_name)
            model_item.setData(0, Qt.ItemDataRole.UserRole, model_name)

            if is_hidden:
                model_item.setForeground(0, Qt.GlobalColor.gray)
                font = model_item.font(0)
                font.setItalic(True)
                model_item.setFont(0, font)

            parent_item.addChild(model_item)

    def _populate_workflow_predictions(self, group: ItemGroupEnum):
        if not self._workflow or not self._workspace:
            return
        
        match self._workflow:
            case Workflow.POSE_ESTIMATION:
                predictions = list(self._workspace.pose_predictions._pose_predictions.keys())
                header = "Pose Estimation Predictions"
            case Workflow.BEHAVIOR_CLASSIFICATION:
                predictions = list(self._workspace.behavior_predictions._behavior_predictions.keys())
                header = "Behavior Predictions"
            case _:
                return

        parent_item = QTreeWidgetItem([header])
        parent_item.setToolTip(0, header)
        self.tree.addTopLevelItem(parent_item)
        parent_item.setExpanded(True)
        
        workspace_path = str(self._workspace.base_dir)
        for pred_name in sorted(predictions):
            is_hidden = is_prediction_hidden(workspace_path, pred_name)
            label = get_prediction_label(workspace_path, pred_name)
            display_name = f"{label} {pred_name}" if label else pred_name

            pred_item = QTreeWidgetItem([display_name])
            pred_item.setToolTip(0, display_name)
            pred_item.setData(0, Qt.ItemDataRole.UserRole, pred_name)

            if is_hidden:
                pred_item.setForeground(0, Qt.GlobalColor.gray)
                font = pred_item.font(0)
                font.setItalic(True)
                pred_item.setFont(0, font)

            parent_item.addChild(pred_item)

    def _populate_workspace_directory_lazily(
        self, parent_item: QTreeWidgetItem, path: Path
    ):
        """Scans a single directory. Adds dummy children for lazy loading."""
        if not path.is_dir():
            return

        try:
            entries = sorted(
                path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())
            )
        except (PermissionError, FileNotFoundError) as e:
            logging.warning(f"Error scanning {path}: {e}")
            return

        for entry in entries:
            if entry.name in _JUNK_DIRS:
                continue
            
            item = QTreeWidgetItem([entry.name])
            item.setToolTip(0, entry.name)
            parent_item.addChild(item)
            if entry.is_dir():
                item.addChild(QTreeWidgetItem([_DUMMY_CHILD_TEXT]))

    def handle_expansion_event(self, item: QTreeWidgetItem, current_group: str):
        """Called by the View when a node is expanded."""
        # Check if needs loading (has dummy child)
        if (item.childCount() != 1 or 
                item.child(0).text(0) != _DUMMY_CHILD_TEXT):
            return 

        item.takeChild(0) # Remove dummy

        # Reconstruct path or identify dataset
        if not self._workspace: return
        
        hierarchy = get_item_hierarchy(item)
        
        if current_group == ItemGroupEnum.WORKSPACE_FILES.value:
            parts = hierarchy[1:] # Remove root
            current_path = self._workspace.base_dir.joinpath(*parts)
            self._populate_workspace_directory_lazily(item, current_path)
        
        elif current_group == ItemGroupEnum.DATASETS.value:
            # Check if this item is a dataset (it will be a second-level item, or child of a type header)
            if item.parent() and item.parent().parent() is None:
                # It's a dataset node
                dataset_name = item.data(0, Qt.ItemDataRole.UserRole)
                dataset = self._workspace.datasets.get(dataset_name)
                if dataset is not None:
                    # Prepare indicators (may have already been prepared but safe to re-run)
                    for ind in self._indicators:
                        ind.prepare(self._workspace, dataset, self._workflow, self._run_name)
                    
                    if dataset.type == DatasetType.VIDEO_COLLECTION:
                        self._populate_video_collection(item, dataset)
                    elif dataset.type in [DatasetType.IMAGE_COLLECTION, DatasetType.FRAME_SUBSET]:
                        self._populate_image_collection(item, dataset)

    def _populate_workspace_datasets(self):
        if not self._workspace or not self._workspace.datasets or not self._workflow:
            return

        datasets_by_type: Dict[DatasetType, List[Dataset]] = {}
        for dataset in self._workspace.datasets.values():
            datasets_by_type.setdefault(dataset.type, []).append(dataset)

        populate_order = ORDERED_DATASET_TYPES_BY_WORKFLOW.get(self._workflow, [])
        for dtype in populate_order:
            # Format header: "Video Collection"
            header = " ".join(
                [c.capitalize() for c in str(dtype.value).replace("_", " ").split(" ")]
            )
            type_item = QTreeWidgetItem([header])
            type_item.setToolTip(0, header)
            self.tree.addTopLevelItem(type_item)
            type_item.setExpanded(True)

            for dataset in sorted(datasets_by_type.get(dtype, []), key=lambda d: d.name):
                # Prepare indicators for this dataset context (just for the header icon)
                for ind in self._indicators:
                    ind.prepare(
                        self._workspace, 
                        dataset, 
                        self._workflow, 
                        self._run_name
                    )

                ds_item = QTreeWidgetItem([dataset.name])
                ds_item.setToolTip(0, dataset.name)
                ds_item.setData(0, Qt.ItemDataRole.UserRole, dataset.name)
                
                # Apply indicators to the dataset header text
                indicator_icons = []
                for ind in self._indicators:
                    icon, _ = ind.get_indicator_state(list(dataset.files))
                    if icon: indicator_icons.append(icon)
                prefix = "".join(indicator_icons)
                ds_item.setText(0, f"{prefix} {dataset.name}")
                
                type_item.addChild(ds_item)
                
                # Add dummy child for lazy loading
                ds_item.addChild(QTreeWidgetItem([_DUMMY_CHILD_TEXT]))

    def _get_display_items(self, cache_key: str, items: list) -> list:
        """
        Returns the list of items to display, shuffling them if blind mode is active.
        """
        if not self._blind_mode:
            return items
        
        if cache_key not in self._shuffled_file_maps:
            shuffled = list(items)
            random.shuffle(shuffled)
            self._shuffled_file_maps[cache_key] = shuffled
        return self._shuffled_file_maps[cache_key]

    def _create_tree_item(
        self, 
        display_text: str, 
        user_data: str, 
        paths: List[str]
    ) -> QTreeWidgetItem:
        """
        Factory method to create a standardized QTreeWidgetItem.
        Iterates through registered indicators to generate icons and data roles.
        """
        indicator_icons = []
        
        item = QTreeWidgetItem()
        item.setData(0, Qt.ItemDataRole.UserRole, user_data)
        
        for ind in self._indicators:
            icon, filter_data = ind.get_indicator_state(paths)
            if icon:
                indicator_icons.append(icon)
            
            # Set filter data if role is defined
            if ind.data_role is not None:
                item.setData(0, ind.data_role, filter_data)

        # Join all icons with the display text
        prefix = "".join(indicator_icons)
        final_text = f"{prefix} {display_text}" if prefix else display_text
        item.setText(0, final_text)
        item.setToolTip(0, display_text)

        return item

    # --- Dataset Type Specific Population ---

    def _populate_video_collection(self, dataset_item: QTreeWidgetItem, dataset: Dataset):
        # Update Dataset Header using all files
        all_files = list(dataset.files)
        
        # Rewrite header with indicators
        indicator_icons = []
        for ind in self._indicators:
            icon, _ = ind.get_indicator_state(all_files)
            if icon: indicator_icons.append(icon)
        
        prefix = "".join(indicator_icons)
        dataset_item.setText(0, f"{prefix} {dataset.name}")

        # Populate Items
        display_files = self._get_display_items(dataset.name, all_files)
        
        for i, file_str in enumerate(display_files):
            display = f"Video {i+1}" if self._blind_mode else file_str
            item = self._create_tree_item(
                display_text=display,
                user_data=file_str,
                paths=[file_str]
            )
            dataset_item.addChild(item)

    def _populate_image_collection(self, dataset_item: QTreeWidgetItem, dataset: Dataset):
        # Update Dataset Header
        all_files = list(dataset.files)
        
        indicator_icons = []
        for ind in self._indicators:
            icon, _ = ind.get_indicator_state(all_files)
            if icon: indicator_icons.append(icon)
            
        prefix = "".join(indicator_icons)
        dataset_item.setText(0, f"{prefix} {dataset.name}")

        if dataset.type == DatasetType.FRAME_SUBSET:
            self._populate_frame_subset(dataset_item, dataset)
        else:
            self._populate_flat_images(dataset_item, dataset)

    def _populate_frame_subset(self, parent, dataset):
        # Group by video
        frames_by_video = {}
        for f in dataset.files:
            v_name = Path(f).parts[0]
            frames_by_video.setdefault(v_name, []).append(f)

        video_groups = sorted(frames_by_video.keys())
        display_groups = self._get_display_items(dataset.name, video_groups)

        for i, v_name in enumerate(display_groups):
            frame_files = frames_by_video[v_name]
            
            # Group Item (Video)
            group_display = f"Video Group {i+1}" if self._blind_mode else v_name
            v_item = self._create_tree_item(
                display_text=group_display,
                user_data=v_name,
                paths=frame_files
            )
            parent.addChild(v_item)

            # Leaf Items (Frames)
            for j, f_file in enumerate(sorted(frame_files)):
                f_display = f"Frame {j+1}" if self._blind_mode else Path(f_file).name
                f_item = self._create_tree_item(
                    display_text=f_display,
                    user_data=f_file,
                    paths=[f_file]
                )
                v_item.addChild(f_item)

    def _populate_flat_images(self, parent, dataset):
        display_files = self._get_display_items(dataset.name, dataset.files)

        for i, f_str in enumerate(display_files):
            display = f"Image {i+1}" if self._blind_mode else f_str
            item = self._create_tree_item(
                display_text=display,
                user_data=f_str,
                paths=[f_str]
            )
            parent.addChild(item)

    # --- Filtering ---

    def apply_filter(self, filter_option: FilterOptions):
        """Hides/Shows items based on LabeledStateRole."""
        if not filter_option: return

        iterator = QTreeWidgetItemIterator(self.tree)
        parents_to_check = {}

        # 1. Leaf visibility
        while iterator.value():
            item = iterator.value()
            # DEV_NOTE: This assumes the filter logic is strictly about "Labeling".
            # If generic filtering is needed later, this method will need refactoring
            # to accept a role or a callback.
            is_labeled = item.data(0, self.LabeledStateRole)
            
            if is_labeled is not None:
                should_hide = False
                if filter_option == FilterOptions.LABELED and not is_labeled:
                    should_hide = True
                elif filter_option == FilterOptions.UNLABELED and is_labeled:
                    should_hide = True
                
                if item.isHidden() != should_hide:
                    item.setHidden(should_hide)
                    if item.parent():
                        parents_to_check[id(item.parent())] = item.parent()
            iterator += 1
        
        # 2. Parent visibility (Recursive)
        for parent in parents_to_check.values():
            self._update_parent_visibility(parent)

    def _update_parent_visibility(self, item):
        if not item: return
        all_hidden = True
        for i in range(item.childCount()):
            if not item.child(i).isHidden():
                all_hidden = False
                break
        
        if item.isHidden() != all_hidden:
            item.setHidden(all_hidden)
            self._update_parent_visibility(item.parent())

    # --- State Restoration ---

    def _get_expanded_item_paths(self) -> Set[tuple[str, ...]]:
        paths = set()
        iterator = QTreeWidgetItemIterator(self.tree)
        while iterator.value():
            item = iterator.value()
            if item.isExpanded():
                paths.add(tuple(get_item_hierarchy(item)))
            iterator += 1
        return paths

    def _restore_expanded_paths(self, expanded_paths):
        if not expanded_paths: return
        
        # Stack-based expansion to handle lazy nodes appearing dynamically
        stack = []
        for i in range(self.tree.topLevelItemCount()):
            stack.append(self.tree.topLevelItem(i))
            
        while stack:
            item = stack.pop()
            path = tuple(get_item_hierarchy(item))
            
            if path in expanded_paths:
                item.setExpanded(True) # May trigger lazy load logic in view
                for i in range(item.childCount() - 1, -1, -1):
                    stack.append(item.child(i))

    def _restore_selection(self, selected_path):
        if not selected_path: return
        
        # Longest Prefix Match
        best_item = None
        best_len = 0
        
        iterator = QTreeWidgetItemIterator(self.tree)
        while iterator.value():
            item = iterator.value()
            path = tuple(get_item_hierarchy(item))
            
            if len(path) <= len(selected_path) and path == selected_path[:len(path)]:
                if len(path) > best_len:
                    best_len = len(path)
                    best_item = item
            iterator += 1
            
        try:
            if best_item:
                self.tree.setCurrentItem(best_item)
                self.tree.scrollToItem(best_item)
        except RuntimeError as e:
            logging.error(f"Couldn't scroll data explorer tree to item: {e}")
    # --- Public Selectors ---

    def select_item_by_path(self, file_path: Path):
        """Finds item by matching stored UserRole path."""
        if not self._workspace: return

        iterator = QTreeWidgetItemIterator(self.tree)
        while iterator.value():
            item = iterator.value()
            data = item.data(0, Qt.ItemDataRole.UserRole)
            
            if isinstance(data, str):
                # 1. Direct path match
                if Path(data).as_posix() == file_path.as_posix():
                    self.tree.setCurrentItem(item)
                    return

                # 2. Dataset relative match
                try:
                    hier = get_item_hierarchy(item)
                    if len(hier) > 2 and hier[1] in self._workspace.datasets:
                        ds = self._workspace.datasets[hier[1]]
                        full = Path(ds.base_data_path) / data
                        if full.resolve() == file_path.resolve():
                            try:
                                self.tree.setCurrentItem(item)
                                self.tree.scrollToItem(item)
                            except RuntimeError:
                                logging.error(f"Couldn't scroll data explorer tree to item: {traceback.format_exc()}")
                            return
                except Exception:
                    pass
            iterator += 1

    def select_sibling_image(self, direction: int):
        """
        direction: -1 for previous, 1 for next.
        """
        current = self.tree.currentItem()
        if not current: return
        
        step = self.tree.itemAbove if direction < 0 else self.tree.itemBelow
        target = step(current)
        
        while target:
            if self._is_valid_image(target):
                self.tree.setCurrentItem(target)
                return
            target = step(target)

    def _is_valid_image(self, item):
        # Quick check without expensive inference
        # Valid images/frames have a boolean LabeledStateRole set
        return item.data(0, self.LabeledStateRole) is not None