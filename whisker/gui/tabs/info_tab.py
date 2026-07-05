import logging
import os
import shutil
import datetime
import queue
import json
from typing import Callable
from pathlib import Path

import cv2
import h5py
import numpy as np
from PyQt6.QtCore import Qt, QUrl, QObject, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QDesktopServices
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QStackedWidget,
    QGroupBox,
    QFormLayout,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QHeaderView,
    QMenu,
    QMessageBox,
    QComboBox,
)

from .base_tab import BaseTab
from .base_workflow_tab import WorkflowItemHandler
from whisker.core.workspace import Workspace
from whisker.services.behavior_classification.public.data_structures import BehaviorDataset
from whisker.services.pose_estimation.public.data_structures import PoseDataset
from whisker.gui.constants import (
    VIDEO_EXTENSIONS,
    IMAGE_EXTENSIONS,
    TEXT_EXTENSIONS,
    HDF5_EXTENSIONS,
)
from whisker.gui.widgets import data_explorer, ScalableImageLabel, MediaViewerWidget
from whisker.gui.signals import MessageBus

_PLACEHOLDER_TEXT = "Select an item on the Data Explorer for more information."


class SizeCalculator(QObject):
    size_calculated = pyqtSignal(str, str)  # (path_str, size_str)


class WorkspaceExplorerWidget(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._workspace = None
        self._size_cache = {}
        self._pending_sizes = {}
        self._size_queue = queue.Queue()
        self._worker_thread = None

        self.size_calc = SizeCalculator()
        self.size_calc.size_calculated.connect(self._on_size_calculated)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Title/header label to make it look premium
        self.header_label = QLabel("Workspace Directory Size Analyzer")
        self.header_label.setStyleSheet("font-weight: bold; font-size: 14px; padding-bottom: 5px;")
        layout.addWidget(self.header_label)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Name", "Last Modified", "Total Size"])
        self.tree.setColumnCount(3)
        self.tree.setAlternatingRowColors(True)

        header = self.tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)

        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_context_menu)
        self.tree.itemExpanded.connect(self._on_item_expanded)

        layout.addWidget(self.tree)

        # Subscribe to files refreshed event so we automatically stay in sync
        MessageBus.get().subscribe("workspace/files/refreshed", lambda t, p: self.refresh())

    def set_workspace(self, workspace):
        self._workspace = workspace
        self.refresh()

    def refresh(self):
        # Clear the queue of any pending paths
        while not self._size_queue.empty():
            try:
                self._size_queue.get_nowait()
                self._size_queue.task_done()
            except (queue.Empty, ValueError):
                break
        self._pending_sizes.clear()
        self.tree.clear()
        if not self._workspace or not self._workspace.base_dir:
            return

        root_path = Path(self._workspace.base_dir)
        if not root_path.exists():
            return

        # Use same name format as tree controller
        root_item = QTreeWidgetItem([
            root_path.name,
            self._get_last_modified_str(root_path),
            "Calculating..."
        ])
        root_item.setData(0, Qt.ItemDataRole.UserRole, str(root_path))
        self.tree.addTopLevelItem(root_item)
        self._update_item_size(root_item, root_path)

        self._populate_directory_lazily(root_item, root_path)
        root_item.setExpanded(True)

    def _populate_directory_lazily(self, parent_item: QTreeWidgetItem, path: Path):
        if not path.is_dir():
            return

        try:
            entries = sorted(
                path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())
            )
        except (PermissionError, FileNotFoundError) as e:
            logging.warning(f"WorkspaceExplorerWidget error scanning {path}: {e}")
            return

        junk_dirs = {
            ".git", ".venv", "venv", ".vscode", "__pycache__", "node_modules",
            ".pytest_cache", ".mypy_cache", "build", "dist", ".egg-info",
        }

        for entry in entries:
            if entry.name in junk_dirs:
                continue

            item = QTreeWidgetItem([
                entry.name,
                self._get_last_modified_str(entry),
                "Calculating..."
            ])
            item.setData(0, Qt.ItemDataRole.UserRole, str(entry))
            parent_item.addChild(item)
            self._update_item_size(item, entry)

            if entry.is_dir():
                # Add dummy child for lazy loading
                item.addChild(QTreeWidgetItem(["Loading...", "", ""]))

    def _on_item_expanded(self, item: QTreeWidgetItem):
        # Check if needs loading (has dummy child)
        if item.childCount() == 1 and item.child(0).text(0) == "Loading...":
            item.takeChild(0) # Remove dummy
            path_str = item.data(0, Qt.ItemDataRole.UserRole)
            if path_str:
                path = Path(path_str)
                self._populate_directory_lazily(item, path)

    def _get_last_modified_str(self, path: Path) -> str:
        try:
            mtime = path.stat().st_mtime
            return datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return "N/A"

    def _update_item_size(self, item: QTreeWidgetItem, path: Path):
        path_str = str(path)
        if path_str in self._size_cache:
            item.setText(2, self._size_cache[path_str])
            return

        # Keep track of which items are waiting for this path's size
        self._pending_sizes.setdefault(path_str, []).append(item)

        # If it's already calculating, don't start another thread
        if len(self._pending_sizes[path_str]) > 1:
            return

        self._size_queue.put(path)
        self._start_worker_if_needed()

    def _start_worker_if_needed(self):
        import threading
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return

        def worker():
            while not self._size_queue.empty():
                try:
                    path = self._size_queue.get_nowait()
                except queue.Empty:
                    break
                
                path_str = str(path)
                try:
                    if path.is_file():
                        size = path.stat().st_size
                    else:
                        size = self._get_dir_size(path)
                    formatted = self._format_size(size)
                except Exception:
                    formatted = "N/A"
                
                self.size_calc.size_calculated.emit(path_str, formatted)
                self._size_queue.task_done()

        self._worker_thread = threading.Thread(target=worker, daemon=True)
        self._worker_thread.start()

    def _on_size_calculated(self, path_str: str, size_str: str):
        self._size_cache[path_str] = size_str
        items = self._pending_sizes.pop(path_str, [])
        for item in items:
            try:
                item.setText(2, size_str)
            except RuntimeError:
                pass

    def _get_dir_size(self, path: Path) -> int:
        total_size = 0
        junk_dirs = {
            ".git", ".venv", "venv", ".vscode", "__pycache__", "node_modules",
            ".pytest_cache", ".mypy_cache", "build", "dist", ".egg-info",
        }
        try:
            for root, dirs, files in os.walk(path):
                dirs[:] = [d for d in dirs if d not in junk_dirs]
                for f in files:
                    try:
                        total_size += os.path.getsize(os.path.join(root, f))
                    except Exception:
                        pass
        except Exception:
            pass
        return total_size

    def _format_size(self, size_bytes: int) -> str:
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024**2:
            return f"{size_bytes / 1024:.2f} KB"
        elif size_bytes < 1024**3:
            return f"{size_bytes / 1024**2:.2f} MB"
        else:
            return f"{size_bytes / 1024**3:.2f} GB"

    def _on_context_menu(self, pos):
        item = self.tree.itemAt(pos)
        if not item:
            return

        path_str = item.data(0, Qt.ItemDataRole.UserRole)
        if not path_str or not os.path.exists(path_str):
            return

        menu = QMenu(self)

        delete_action = menu.addAction("Delete from Disk...")
        delete_action.triggered.connect(lambda: self._delete_path(path_str))

        open_browser_action = menu.addAction("Open in File Browser...")
        open_browser_action.triggered.connect(lambda: self._open_browser(path_str))

        menu.exec(self.tree.viewport().mapToGlobal(pos))

    def _delete_path(self, path_str: str):
        msg = QMessageBox(self)
        msg.setWindowTitle("Confirm Item Deletion")
        msg.setIcon(QMessageBox.Icon.Warning)
        highlighted_path = (
            f'<span style="color: #e74c3c; font-weight: bold;">{path_str}</span>'
        )
        msg.setText(
            "The selected action will permanently delete the following path:<br><br>"
            f"{highlighted_path}<br><br><b>Are you sure?</b>"
        )
        msg.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel
        )
        if msg.exec() == QMessageBox.StandardButton.Yes:
            try:
                if os.path.isdir(path_str):
                    shutil.rmtree(path_str)
                else:
                    os.remove(path_str)
                logging.info(f"Deleted path: {path_str}")
            except Exception as e:
                logging.error(f"Failed to delete {path_str}: {str(e)}")
                QMessageBox.critical(self, "Error", f"Failed to delete path:\n{e}")

            # Clear cache for this path so it can be recalculated/refreshed
            self._size_cache.pop(path_str, None)
            # If it has parent, clear parent's size cache too
            parent_path = str(Path(path_str).parent)
            self._size_cache.pop(parent_path, None)

            self.refresh()

            # Publish refresh notifications via MessageBus
            bus = MessageBus.get()
            bus.publish("workspace/files/refreshed")
            bus.publish("request/workspace/projects/refresh")
            bus.publish("request/workspace/datasets/refresh")
            bus.publish("request/workspace/labels/refresh")

    def _open_browser(self, path_str: str):
        url = QUrl.fromLocalFile(
            path_str if os.path.isdir(path_str) else os.path.dirname(path_str)
        )
        QDesktopServices.openUrl(url)



class InfoTab(BaseTab):
    def __init__(
        self,
        parent: QWidget | None = None,
        workflow_item_handlers: dict[str, WorkflowItemHandler] | None = None,
    ):
        super().__init__(parent)
        # DEV_NOTE: This will cache the current selection from the data explorer
        self._current_selection: data_explorer.Selection | None = None

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)  # Use consistent margins

        self.view_stack = QStackedWidget()
        main_layout.addWidget(self.view_stack)

        # --- View 0: Placeholder ---
        self.placeholder_widget = QLabel(_PLACEHOLDER_TEXT)
        self.placeholder_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder_widget.setWordWrap(True)
        self.view_stack.addWidget(self.placeholder_widget)

        # --- View 1: Content view ---
        self.content_widget = QWidget()
        content_layout = QVBoxLayout(self.content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        self._create_metadata_panel()
        content_layout.addWidget(self.metadata_panel)
        content_layout.addStretch()
        self.view_stack.addWidget(self.content_widget)

        # --- View 2: Text file view ---
        self.text_viewer = QTextEdit()
        self.text_viewer.setReadOnly(True)
        self.view_stack.addWidget(self.text_viewer)

        # --- View 3: Video player view ---
        self.video_viewer = MediaViewerWidget()
        self.view_stack.addWidget(self.video_viewer)

        # --- View 4: Workspace Explorer view ---
        self.workspace_explorer = WorkspaceExplorerWidget(self)
        self.view_stack.addWidget(self.workspace_explorer)



        self.view_stack.setCurrentWidget(self.placeholder_widget)

        # --- Custom file handlers to display workflow files ---
        self._workflow_item_handlers = workflow_item_handlers or {}
        if workflow_item_handlers:
            logging.debug(
                f"Registered file handlers for workflows: '{list(workflow_item_handlers.keys())}'"
            )
            for (_, widget) in self._workflow_item_handlers.values():
                self.view_stack.addWidget(widget)

    def set_workspace(self, workspace: Workspace | None):
        super().set_workspace(workspace)
        self.workspace_explorer.set_workspace(workspace)

    def on_data_explorer_item_selected(self, selection: data_explorer.Selection):
        logging.debug(f"InfoTab received data explorer selection: {selection}")
        self._current_selection = selection

        if not self._workspace:
            self.view_stack.setCurrentWidget(self.placeholder_widget)
            return

        # Handle Workspace File Explorer selections
        if selection.group == data_explorer.ItemGroupEnum.WORKSPACE_FILES:
            # Check if root node is selected (selection.item has length 1)
            if len(selection.item) == 1:
                self.workspace_explorer.set_workspace(self._workspace)
                self.view_stack.setCurrentWidget(self.workspace_explorer)
                return

            # The full path is the join of all items in the hierarchy
            relative_path = Path(*selection.item[1:])
            file_path = self._workspace.base_dir / relative_path

            # Check if this is a workflow file
            if "workflows" in file_path.parts:
                try:
                    workflows_index = file_path.parts.index("workflows")
                    if len(file_path.parts) > workflows_index + 2:
                        workflow_name = file_path.parts[workflows_index + 1]
                    
                        if self._show_workflow_item_preview(workflow_name, selection):
                            return
                except (ValueError, IndexError) as e:
                    logging.error(
                        f"Error showing workflow file preview for '{file_path}', "
                        f"falling back to generic preview.\nException: [{e}]"
                    )

            if selection.type == data_explorer.ItemTypeEnum.WORKSPACE_FILE:                    
                self._show_file_preview(file_path)
            else:
                # It's a directory or something else, show placeholder
                self.view_stack.setCurrentWidget(self.placeholder_widget)
            return

        # Handle Datasets Explorer selections
        if selection.group == data_explorer.ItemGroupEnum.DATASETS:
            if selection.type == data_explorer.ItemTypeEnum.DATASET_TYPE:
                self.view_stack.setCurrentWidget(self.placeholder_widget)
                return

            self.view_stack.setCurrentWidget(self.content_widget)
            dataset_name = selection.item[1]
            dataset = self._workspace.datasets.get(dataset_name)

            if not dataset:
                self.view_stack.setCurrentWidget(self.placeholder_widget)
                return

            # Case 1: A whole dataset is selected (e.g., "MyVideos")
            if selection.type == data_explorer.ItemTypeEnum.DATASET_BASE:
                self.metadata_stack.setCurrentIndex(1)  # Show dataset info
                self.ds_meta_name.setText(f"<b>{dataset.name}</b>")
                self.ds_meta_type.setText(dataset.type.value)
                self.ds_meta_files.setText(str(len(dataset.files)))
                return

            # Case 2: A group of frames from one video is selected
            if selection.type == data_explorer.ItemTypeEnum.DATASET_VIDEO_FRAME_SUBSET:
                self.metadata_stack.setCurrentIndex(1)  # Show dataset info view
                video_name = selection.item[2]
                num_frames = sum(
                    1 for f in dataset.files if Path(f).parts[0] == video_name
                )
                self.ds_meta_name.setText(f"<b>{video_name}</b> (from {dataset.name})")
                self.ds_meta_type.setText("Video Frame Group")
                self.ds_meta_files.setText(str(num_frames))
                return

            # Case 3: A single file/frame is selected from a Dataset
            if selection.type in (
                data_explorer.ItemTypeEnum.DATASET_IMAGE,
                data_explorer.ItemTypeEnum.DATASET_VIDEO,
                data_explorer.ItemTypeEnum.DATASET_VIDEO_FRAME,
            ):
                relative_path = selection.item[-1]
                file_path = Path(dataset.base_data_path) / relative_path
                self._show_file_preview(file_path, is_dataset_file=True)
            else:
                self.view_stack.setCurrentWidget(self.placeholder_widget)
            return

        if selection.group == data_explorer.ItemGroupEnum.MODELS:
            if selection.type in (
                data_explorer.ItemTypeEnum.POSE_ESTIMATION_MODEL_PROJECT,
                data_explorer.ItemTypeEnum.POSE_ESTIMATION_MODEL,
            ):
                self._show_workflow_item_preview("pose_estimation", selection)
            elif selection.type in (
                data_explorer.ItemTypeEnum.BEHAVIOR_CLASSIFICATION_MODEL_PROJECT,
                data_explorer.ItemTypeEnum.BEHAVIOR_CLASSIFICATION_MODEL,
            ):
                self._show_workflow_item_preview("behavior_classification", selection)
            elif selection.type in (
                data_explorer.ItemTypeEnum.ANIMAL_DETECTION_MODEL_PROJECT,
                data_explorer.ItemTypeEnum.ANIMAL_DETECTION_MODEL,
            ):
                self._show_workflow_item_preview("animal_detection", selection)
            else:
                self.view_stack.setCurrentWidget(self.placeholder_widget)
            return

        # Fallback for other groups (Experiments, etc.)
        self.view_stack.setCurrentWidget(self.placeholder_widget)

    def _show_workflow_item_preview(self, workflow_name: str, selection: data_explorer.Selection) -> bool:
        """Shows a preview for a workflow-specific item (e.g., a pose estimation model)."""
        if not self._workspace:
            return False

        handler = self._workflow_item_handlers.get(workflow_name)
        if handler:
            callback_function, widget = handler
            handled = callback_function(self._workspace, selection)
            if handled:
                self.view_stack.setCurrentWidget(widget)
                return True
        else:
            logging.warning(f"No item handler registered for workflow '{workflow_name}'.")
        return False

    def _show_file_preview(self, file_path: Path, is_dataset_file: bool = False):
        """Shows a preview of the selected file based on its type."""
        logging.info(f"Showing file preview for {file_path}")
        suffix = file_path.suffix.lower()

        if not file_path.exists():
            self.view_stack.setCurrentWidget(self.placeholder_widget)
            self.placeholder_widget.setText("File not found.")
            return



        if suffix in TEXT_EXTENSIONS:
            try:
                # Limit the size of the file we try to read
                if file_path.stat().st_size > 1024 * 1024:  # 1 MB limit
                    self.text_viewer.setPlainText(
                        f"File is too large to display (> 1MB)."
                    )
                else:
                    content = file_path.read_text(encoding="utf-8")
                    self.text_viewer.setPlainText(content)
            except Exception as e:
                logging.error(f"Could not read text file {file_path}: {e}")
                self.text_viewer.setPlainText(f"Error reading file:\n{e}")
            self.view_stack.setCurrentWidget(self.text_viewer)
        elif suffix in HDF5_EXTENSIONS:
            self._show_h5_preview(file_path)

        elif suffix in IMAGE_EXTENSIONS:
            # For all images (dataset or workspace), we'll use the metadata view
            # which contains the ScalableImageLabel for the preview.
            self.view_stack.setCurrentWidget(self.content_widget)
            self.metadata_stack.setCurrentIndex(2)  # Show file info panel
            self._update_file_metadata_common(file_path)
            self._update_image_metadata(file_path)

        elif suffix in VIDEO_EXTENSIONS:
            # For dataset videos, we show the metadata panel with a thumbnail.
            if is_dataset_file:
                self.view_stack.setCurrentWidget(self.content_widget)
                self.metadata_stack.setCurrentIndex(2)  # Show file info panel
                self._update_file_metadata_common(file_path)
                self._update_video_metadata(file_path)
            # For workspace videos, we use the full media player.
            else:
                self.video_viewer.set_media(file_path)
                self.view_stack.setCurrentWidget(self.video_viewer)
        else:
            self.view_stack.setCurrentWidget(self.placeholder_widget)
            self.placeholder_widget.setText(f"Unsupported file type: {suffix}")

    def _show_h5_preview(self, file_path: Path):
        """Dispatches to a specialized H5 viewer based on the file's path."""
        path_parts = {p.name for p in file_path.parents}

        if "behavior_classification" in path_parts:
            self._show_behavior_h5_preview(file_path)
        elif "pose_estimation" in path_parts:
            self._show_pose_h5_preview(file_path)
        else:
            self._show_generic_h5_preview(file_path)

    def _show_behavior_h5_preview(self, file_path: Path):
        """Shows a human-readable preview of a BehaviorDataset HDF5 file."""
        try:
            ds = BehaviorDataset.from_file(file_path)
            content = [f"Behavior Data: {file_path.name}\n"]
            content.append("--- Metadata ---")
            content.append(f"Behaviors: {', '.join(ds.behaviors) or 'N/A'}")
            content.append("\n--- Data ---")

            if not ds.per_frame_probabilities.empty:
                content.append(
                    f"[Per-Frame Probabilities]\n"
                    f" - Shape: {ds.per_frame_probabilities.shape}\n"
                    f" - Frames: {len(ds.per_frame_probabilities)}\n"
                    f" - Behaviors: {list(ds.per_frame_probabilities.columns)}"
                )
            else:
                content.append("[Per-Frame Probabilities]\n - (No data)")

            if not ds.bouts.empty:
                num_bouts = len(ds.bouts)
                num_videos = ds.bouts["video_key"].nunique()
                content.append(
                    f"\n[Bouts]\n"
                    f" - Total Bouts: {num_bouts}\n"
                    f" - Videos: {num_videos}"
                )
                # Show top 5 bouts
                content.append(" - First 5 Bouts:")
                for row in ds.bouts.head(5).itertuples():
                    content.append(
                        f"    - {row.video_key} | {row.behavior} "
                        f"[{row.start_frame} - {row.end_frame}]"
                    )
            else:
                content.append("\n[Bouts]\n - (No data)")

            self.text_viewer.setPlainText("\n".join(content))
        except Exception as e:
            logging.error(f"Could not read BehaviorDataset file {file_path}: {e}")
            self.text_viewer.setPlainText(f"Error reading HDF5 file:\n{e}")
        self.view_stack.setCurrentWidget(self.text_viewer)

    def _show_pose_h5_preview(self, file_path: Path):
        """Shows a summary of a pose estimation HDF5 file."""
        try:
            pose_dataset = PoseDataset.from_file(file_path)
            content = ["Pose Estimation Data\n"]

            df = pose_dataset.keypoint_data
            if not df.empty:
                num_frames = df.index.get_level_values('frame_index').nunique()
                num_individuals = len(pose_dataset.individuals)
                num_body_parts = len(pose_dataset.body_parts)

                content.append(f"- Labeled Frames: {num_frames}")
                content.append(f"- Individuals: {num_individuals} ({', '.join(pose_dataset.individuals)})")
                content.append(f"- Body Parts: {num_body_parts} ({', '.join(pose_dataset.body_parts)})")

                content.append(df.head(1000).to_string())
            else:
                content.append("(No keypoint data found)")

            self.text_viewer.setPlainText("\n".join(content))
        except Exception as e:
            logging.error(f"Could not read pose data file {file_path}: {e}")
            self.text_viewer.setPlainText(f"Error reading pose data file:\n{e}")
        self.view_stack.setCurrentWidget(self.text_viewer)

    def _show_generic_h5_preview(self, file_path: Path):
        """Shows a human-readable preview of a generic HDF5 file structure."""
        try:
            with h5py.File(file_path, "r") as f:
                content = [f"HDF5 File Structure: {file_path.name}\n"]

                def visitor_func(name, obj):
                    indent = "  " * name.count('/')
                    basename = name.split('/')[-1] if '/' in name else name
                    prefix = indent + "└── "

                    if isinstance(obj, h5py.Group):
                        content.append(f"{prefix}{basename}/")
                    elif isinstance(obj, h5py.Dataset):
                        dataset_info = f"{prefix}{basename}"
                        dataset_info += f" (Shape: {obj.shape}, DType: {obj.dtype})"
                        content.append(dataset_info)

                content.append("📂 / (Root)")
                f.visititems(visitor_func)

                if len(content) == 2:  # Only header and root
                    content.append("  └── (empty)")

                self.text_viewer.setPlainText("\n".join(content))

        except Exception as e:
            logging.error(f"Could not read HDF5 file {file_path}: {e}")
            self.text_viewer.setPlainText(f"Error reading HDF5 file:\n{e}")
        self.view_stack.setCurrentWidget(self.text_viewer)

    def _update_file_metadata_common(self, file_path: Path):
        """Populates the common file metadata fields in the info panel."""
        self.file_meta_name.setText(f"<b>{file_path.name}</b>")
        self.file_meta_path.setText(str(file_path))

        if not file_path.exists():
            self.file_preview.setPixmap(QPixmap())
            self.file_preview.setText("File not found!")
            self.file_meta_size.setText("N/A")
            self.file_meta_dims.setText("N/A")
            self._set_form_row_visibility(self.duration_row_index, False)
            self._set_form_row_visibility(self.fps_row_index, False)
            return

        self.file_preview.setText("")
        size = self._format_size(file_path.stat().st_size)
        self.file_meta_size.setText(size)

    def _create_metadata_panel(self):
        """Builds the metadata panel and its different view states."""
        self.metadata_panel = QGroupBox("Metadata")
        meta_layout = QVBoxLayout(self.metadata_panel)

        self.metadata_stack = QStackedWidget()
        meta_layout.addWidget(self.metadata_stack)

        # State 0: Initial placeholder (re-use the main one)
        placeholder = QLabel("Select an item from the list to see details.")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.metadata_stack.addWidget(placeholder)

        # State 1: Dataset Info
        dataset_info_widget = self._create_dataset_info_widget()
        self.metadata_stack.addWidget(dataset_info_widget)

        # State 2: File Info
        file_info_widget = self._create_file_info_widget()
        self.metadata_stack.addWidget(file_info_widget)

    def _create_dataset_info_widget(self) -> QWidget:
        """Creates the widget for displaying selected dataset metadata."""
        widget = QWidget()
        layout = QFormLayout(widget)
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.ds_meta_name = QLabel()
        self.ds_meta_type = QLabel()
        self.ds_meta_files = QLabel()
        layout.addRow("Name:", self.ds_meta_name)
        layout.addRow("Type:", self.ds_meta_type)
        layout.addRow("File Count:", self.ds_meta_files)
        return widget

    def _create_file_info_widget(self) -> QWidget:
        """Creates the widget for displaying selected file metadata."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.file_preview = ScalableImageLabel()
        self.file_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_preview.setStyleSheet("background-color: #222;")
        layout.addWidget(self.file_preview)

        self.file_info_form_layout = QFormLayout()
        self.file_info_form_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        self.file_meta_name = QLabel()
        self.file_meta_path = QLabel()
        self.file_meta_size = QLabel()
        self.file_meta_dims = QLabel()
        self.file_meta_duration = QLabel()
        self.file_meta_fps = QLabel()
        self.file_meta_path.setWordWrap(True)

        self.file_info_form_layout.addRow("Name:", self.file_meta_name)
        self.file_info_form_layout.addRow("Path:", self.file_meta_path)
        self.file_info_form_layout.addRow("Size:", self.file_meta_size)
        self.file_info_form_layout.addRow("Dimensions:", self.file_meta_dims)

        self.duration_row = self.file_info_form_layout.addRow(
            "Duration:", self.file_meta_duration
        )
        self.duration_row_index = self.file_info_form_layout.rowCount() - 1

        self.fps_row = self.file_info_form_layout.addRow(
            "Frame Rate:", self.file_meta_fps
        )
        self.fps_row_index = self.file_info_form_layout.rowCount() - 1

        layout.addLayout(self.file_info_form_layout)
        return widget

    def _set_form_row_visibility(self, row_index: int, is_visible: bool):
        """Sets the visibility of a row (both label and widget) in the form."""
        label_item = self.file_info_form_layout.itemAt(
            row_index, QFormLayout.ItemRole.LabelRole
        )
        if label_item and label_item.widget():
            label_item.widget().setVisible(is_visible)

        widget_item = self.file_info_form_layout.itemAt(
            row_index, QFormLayout.ItemRole.FieldRole
        )
        if widget_item and widget_item.widget():
            widget_item.widget().setVisible(is_visible)

    def _format_size(self, size_bytes: int) -> str:
        """Converts bytes into a human-readable string."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024**2:
            return f"{size_bytes / 1024:.2f} KB"
        elif size_bytes < 1024**3:
            return f"{size_bytes / 1024**2:.2f} MB"
        else:
            return f"{size_bytes / 1024**3:.2f} GB"

    def _update_image_metadata(self, path: Path):
        """Populates the metadata panel for an image file."""
        pixmap = QPixmap(str(path))
        self.file_preview.setPixmap(pixmap)
        self.file_meta_dims.setText(f"{pixmap.width()} x {pixmap.height()} px")
        self._set_form_row_visibility(self.duration_row_index, False)
        self._set_form_row_visibility(self.fps_row_index, False)

    def _update_video_metadata(self, path: Path):
        """Populates the metadata panel for a video file using OpenCV."""
        self._set_form_row_visibility(self.duration_row_index, True)
        self._set_form_row_visibility(self.fps_row_index, True)
        cap = cv2.VideoCapture(str(path))
        pixmap = QPixmap()
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ret, frame = cap.read()
            if ret:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h_img, w_img, ch = rgb_image.shape
                bytes_per_line = ch * w_img
                qt_image = QImage(
                    rgb_image.data,
                    w_img,
                    h_img,
                    bytes_per_line,
                    QImage.Format.Format_RGB888,
                )
                pixmap = QPixmap.fromImage(qt_image)
            cap.release()
            self.file_preview.setPixmap(pixmap)
            self.file_meta_dims.setText(f"{w} x {h} px")
            duration_sec = (frames / fps) if fps > 0 else 0
            self.file_meta_duration.setText(f"{duration_sec:.2f} s")
            self.file_meta_fps.setText(f"{fps:.2f} fps")
        else:
            self.file_preview.setText("Could not open video")
            self.file_meta_dims.setText("N/A")
            self.file_meta_duration.setText("N/A")
            self.file_meta_fps.setText("N/A")
