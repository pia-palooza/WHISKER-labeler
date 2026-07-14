import logging
import os
import shutil
import json
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QUrl, QThreadPool, QObject
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QWidget,
    QMessageBox,
    QInputDialog,
    QDialog,
    QMenu,
    QTreeWidgetItem,
    QTreeWidget,
    QFileDialog,
    QProgressDialog,
)

from whisker.core.workspace import Workspace, DatasetType, Project
from whisker.gui.dialogs import (
    CreateDatasetDialog,
    CreateDatasetTabbedDialog,
    WarnIfExistsDialog,
    ImportLabelsDialog,
    ExportAnnotationsDialog,
    ImportBundleDialog,
)
from whisker.core.workers.bundle_workers import ExportBundleJob, ImportBundleJob
from whisker.gui.signals import MessageBus
from whisker.gui.worker_wrapper import Worker
from whisker.gui.widgets.export_clip_dialog import ExportClipDialog
from whisker.gui.workflows.behavior_classification.widgets.export_dialogs import BoutExportDialog, ChartExportOptionsDialog
from whisker.gui.workflows.behavior_classification.utils.chart_rendering import (
    render_dataset_behavior_chart, gather_dataset_video_data
)
from .constants import ItemGroupEnum, ItemTypeEnum
from .selection_logic import infer_selection_item_type, get_item_hierarchy
from .tree_controller import (
    get_model_label,
    set_model_label,
    is_model_hidden,
    set_model_hidden,
    get_prediction_label,
    set_prediction_label,
    is_prediction_hidden,
    set_prediction_hidden,
)

class ActionHandler(QObject):
    def __init__(self, parent_widget: QWidget, thread_pool: QThreadPool):
        super().__init__(parent_widget)
        self.parent_widget = parent_widget
        self.thread_pool = thread_pool
        self._workspace: Optional[Workspace] = None
        self._current_model_run: Optional[str] = None
        self._active_project: Optional[Project] = None

        MessageBus.get().subscribe("selection/model_run/changed", lambda t, p: self._on_model_run_changed(p.get("name")))

    def update_workspace(self, workspace: Optional[Workspace]):
        self._workspace = workspace

    def set_active_project(self, project: Optional[Project]):
        self._active_project = project

    def _on_model_run_changed(self, run_name: str):
        self._current_model_run = run_name if run_name else None

    # --- Dialog Actions ---

    def show_create_project_dialog(self):
        if not self._workspace:
            logging.warning("No workspace set.")
            return

        project_name, ok = QInputDialog.getText(
            self.parent_widget, "Enter Project Name", "Project name:"
        )
        if ok and project_name:
            self._workspace.create_project(
                project_name, warn_if_exists=WarnIfExistsDialog.run
            )
            MessageBus.get().publish("request/workspace/projects/refresh")
            # Note: The widget listens to global signals to refresh the tree

    def show_create_dataset_dialog(self):
        if not self._workspace:
            return

        dialog = CreateDatasetTabbedDialog(self._workspace)
        dialog.exec()
        if dialog.result() != QDialog.DialogCode.Accepted:
            return

        result = dialog.get_result()
        if not result:
            return

        mode, info = result
        warn = lambda msg: WarnIfExistsDialog.run(msg, parent=self.parent_widget)

        if mode == "single":
            name, inferred_type, folder_path = info
            self._workspace.create_dataset(name, inferred_type, folder_path, warn_if_exists=warn)
            MessageBus.get().publish("request/workspace/datasets/refresh")
        elif mode == "multi":
            self._workspace.create_multi_arena_dataset(
                dataset_name=info["name"],
                dataset_data_dir=info["folder_path"],
                box_width=info["box_width"],
                box_height=info["box_height"],
                placements=info["placements"],
                warn_if_exists=warn,
            )
            MessageBus.get().publish("request/workspace/datasets/refresh")

    def _edit_arenas(self, dataset_name: str):
        """Open the arena placement editor pre-loaded with an existing
        multi-arena dataset, then persist the revised config after warning the
        user about any arenas whose existing artifacts the edit would invalidate."""
        if not self._workspace:
            return
        dataset = self._workspace.datasets.get(dataset_name)
        if not dataset or not getattr(dataset, "is_multi_arena", False):
            return

        from PyQt6.QtWidgets import QVBoxLayout
        from whisker.gui.dialogs.multi_arena import MultiArenaDatasetPanel

        dialog = QDialog(self.parent_widget)
        dialog.setWindowTitle(f"Edit Arenas — {dataset_name}")
        dialog.resize(1000, 720)
        layout = QVBoxLayout(dialog)
        panel = MultiArenaDatasetPanel()
        layout.addWidget(panel)
        panel.load_existing(
            name=dataset_name,
            folder_path=dataset.base_data_path,
            box_width=dataset.multi_arena.box_width,
            box_height=dataset.multi_arena.box_height,
            placements=dataset.multi_arena.placements,
        )
        panel.create_requested.connect(dialog.accept)
        panel.cancel_requested.connect(dialog.reject)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        info = panel.get_multi_arena_info()
        if info is None:
            return

        self._apply_arena_edit(dataset, info)

    def _apply_arena_edit(self, dataset, info: dict):
        """Validate the edit's impact against existing per-arena artifacts,
        confirm with the user, then persist."""
        from whisker.core.study.dataset import analyze_arena_edit

        impact = analyze_arena_edit(
            dataset.multi_arena, info["box_width"], info["box_height"], info["placements"]
        )

        if impact.has_risky_changes:
            affected = self._arena_stems_with_artifacts(
                dataset.name, impact.invalidated_stems
            )
            if affected:
                lines = []
                for stem in sorted(affected):
                    lines.append(f"  • {stem} — {', '.join(affected[stem])}")
                detail = "\n".join(lines)
                extra = (
                    "\n\nChanging the shared box size moves every arena, so all "
                    "arenas are affected."
                    if impact.box_size_changed else ""
                )
                proceed = QMessageBox.warning(
                    self.parent_widget,
                    "Existing arena data will be invalidated",
                    "This edit changes the box under arenas that already have "
                    "labels or predictions. Because arenas are identified by "
                    "position, their existing data no longer matches the new "
                    "box and pose predictions must be re-run:\n\n"
                    f"{detail}{extra}\n\n"
                    "The old artifacts are left on disk but will be mismatched. "
                    "Continue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if proceed != QMessageBox.StandardButton.Yes:
                    return

        try:
            self._workspace.update_multi_arena_dataset(
                dataset_name=dataset.name,
                box_width=info["box_width"],
                box_height=info["box_height"],
                placements=info["placements"],
            )
        except Exception as e:
            logging.error(f"Failed to update multi-arena dataset '{dataset.name}': {e}", exc_info=True)
            QMessageBox.critical(self.parent_widget, "Edit Failed", f"Could not save arena edits: {e}")
            return

        MessageBus.get().publish("request/workspace/datasets/refresh")

    def _arena_stems_with_artifacts(self, dataset_name: str, stems) -> dict:
        """For the given arena stems, return {stem: [artifact types present]},
        checking behavior labels, pose predictions, and behavior predictions.
        Stems with no existing artifacts are omitted."""
        stems = set(stems)
        found: dict[str, list[str]] = {}
        if not stems:
            return found

        def _add(stem: str, kind: str):
            found.setdefault(stem, [])
            if kind not in found[stem]:
                found[stem].append(kind)

        # 1. Behavior labels (single labels.h5 keyed by video_key == arena stem)
        try:
            labeled = self._workspace.get_behavior_labeled_video_keys(dataset_name)
            for stem in stems & set(labeled):
                _add(stem, "behavior labels")
        except Exception as e:
            logging.warning(f"Could not check behavior labels for arena edit: {e}")

        # 2. Pose predictions (per-run/dataset/stem directories on disk)
        try:
            pose_base = self._workspace.pose_predictions.base_dir
            if pose_base.is_dir():
                for run_dir in pose_base.iterdir():
                    ds_dir = run_dir / dataset_name
                    if not ds_dir.is_dir():
                        continue
                    for stem in stems:
                        if (ds_dir / stem / "predictions.h5").exists():
                            _add(stem, "pose predictions")
        except Exception as e:
            logging.warning(f"Could not check pose predictions for arena edit: {e}")

        # 3. Behavior predictions (per-run/dataset/stem directories on disk)
        try:
            bc_base = self._workspace.behavior_predictions.base_dir
            if bc_base.is_dir():
                for run_dir in bc_base.iterdir():
                    ds_dir = run_dir / dataset_name
                    if not ds_dir.is_dir():
                        continue
                    for stem in stems:
                        if (ds_dir / stem / "predictions.h5").exists():
                            _add(stem, "behavior predictions")
        except Exception as e:
            logging.warning(f"Could not check behavior predictions for arena edit: {e}")

        return found

    def show_import_pose_labels_dialog(self, button_reference=None):
        """
        Shows the import dialog.
        DEV_NOTE: button_reference is optional and used to manage UI state 
        (Enabled/Disabled) during background work if provided.
        """
        if not self._workspace:
            QMessageBox.warning(
                self.parent_widget,
                "Workspace Not Found",
                "Please open a workspace before importing labels.",
            )
            return

        dialog = ImportLabelsDialog(self._workspace, self.parent_widget)
        if not dialog.exec():
            return

        from whisker.core.workers.importer import LabelImporterJob
        worker = Worker(
            LabelImporterJob(
                workspace=self._workspace,
                dataset_name=dialog.selected_dataset_name,
                project_name=dialog.selected_project_name,
                import_path=Path(dialog.selected_path),
                data_type_str=str(dialog.selected_data_type.value),
                backend=dialog.selected_data_format,
                warn_if_exists=lambda msg: WarnIfExistsDialog.run(msg, parent=self.parent_widget),
            )
        )
        
        # UI State Management Closures
        def _on_finished(msg):
            if button_reference:
                button_reference.setEnabled(True)
                button_reference.setText("Import Pose Labels...")
            QMessageBox.information(self.parent_widget, "Import Successful", msg)
            MessageBus.get().publish("request/workspace/datasets/refresh")
            MessageBus.get().publish("request/workspace/labels/refresh")

        def _on_error(err):
            if button_reference:
                button_reference.setEnabled(True)
                button_reference.setText("Import Pose Labels...")
            QMessageBox.critical(
                self.parent_widget, "Import Failed", f"Error:\n{err}"
            )

        if button_reference:
            button_reference.setEnabled(False)
            button_reference.setText("Importing...")
        
        worker.signals.finished.connect(_on_finished)
        worker.signals.error.connect(_on_error)
        self.thread_pool.start(worker)

    # --- Annotation Bundle Export / Import ---

    def _export_annotations(self, dataset_name: str):
        """Export an image/frame dataset's annotations (project, manifest, label
        HDF5s and the frame images) into a tidy, self-describing bundle."""
        if not self._workspace:
            return

        default_project = self._active_project.name if self._active_project else None
        dialog = ExportAnnotationsDialog(
            self._workspace, dataset_name, default_project, self.parent_widget
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        plan = dialog.plan
        bundle_dir = dialog.bundle_dir
        if plan is None or bundle_dir is None:
            return

        overwrite = False
        if bundle_dir.exists():
            reply = QMessageBox.question(
                self.parent_widget,
                "Overwrite Existing Folder?",
                f"'{bundle_dir}' already exists.\n\nReplace it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            overwrite = True

        def _on_finished(result):
            msg = f"Exported '{dataset_name}' to:\n{result['bundle_dir']}\n\n"
            if result.get("media_included"):
                msg += (
                    f"{result['media_kind'].capitalize()} copied: "
                    f"{result['num_media_copied']}/{result['num_media']}"
                )
                if result.get("num_missing"):
                    msg += f"\nMissing/skipped: {result['num_missing']}"
            else:
                msg += (
                    f"{result['media_kind'].capitalize()} referenced "
                    f"(not copied): {result['num_media']}"
                )
            QMessageBox.information(self.parent_widget, "Export Complete", msg)

        self._run_bundle_job(
            ExportBundleJob(
                plan, bundle_dir, overwrite=overwrite, include_media=dialog.include_media
            ),
            "Exporting Annotations",
            _on_finished,
            "Export Failed",
        )

    def show_import_bundle_dialog(self):
        """Import an annotation bundle produced by 'Export Annotations...'."""
        if not self._workspace:
            QMessageBox.warning(
                self.parent_widget,
                "Workspace Not Found",
                "Please open a workspace before importing a bundle.",
            )
            return

        dialog = ImportBundleDialog(self._workspace, self.parent_widget)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        bundle_dir = dialog.bundle_dir
        if bundle_dir is None:
            return

        def _on_finished(result):
            # Rescan the workspace on the GUI thread now that files are written.
            self._workspace.scan_projects()
            self._workspace.scan_datasets()
            self._workspace.scan_labels()
            MessageBus.get().publish("request/workspace/projects/refresh")
            MessageBus.get().publish("request/workspace/datasets/refresh")
            MessageBus.get().publish("request/workspace/labels/refresh")

            msg = (
                f"Imported dataset '{result['dataset_name']}' "
                f"(project '{result['project_name']}').\n\n"
            )
            if result.get("media_included"):
                msg += (
                    f"{result['media_kind'].capitalize()} copied: "
                    f"{result['num_media_copied']}/{result['num_media']}"
                )
            else:
                msg += f"{result['media_kind'].capitalize()} referenced (not copied)."
            extras = []
            if result.get("pose_imported"):
                extras.append("pose labels")
            if result.get("behavior_imported"):
                extras.append("behavior labels")
            if extras:
                msg += "\nImported: " + ", ".join(extras)
            if result.get("num_missing"):
                msg += f"\nMedia not found: {result['num_missing']}"
            QMessageBox.information(self.parent_widget, "Import Complete", msg)

        self._run_bundle_job(
            ImportBundleJob(
                self._workspace,
                bundle_dir,
                overwrite=dialog.overwrite,
                media_source_dir=dialog.media_source_dir,
            ),
            "Importing Annotation Bundle",
            _on_finished,
            "Import Failed",
        )

    def _run_bundle_job(self, job, title: str, on_finished, error_title: str):
        """Run a bundle export/import job on the thread pool with a modal
        progress dialog. All overwrite decisions must already be resolved."""
        progress = QProgressDialog(title + "...", "Cancel", 0, 100, self.parent_widget)
        progress.setWindowTitle(title)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setAutoClose(True)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        worker = Worker(job)

        def _on_progress(message, percent):
            if not progress.wasCanceled():
                progress.setLabelText(message)
                progress.setValue(max(0, min(100, percent)))

        def _on_job_finished(result):
            progress.reset()
            on_finished(result)

        def _on_job_error(err):
            progress.reset()
            if progress.wasCanceled() or "cancel" in str(err).lower():
                return
            QMessageBox.critical(self.parent_widget, error_title, f"Error:\n{err}")

        worker.signals.progress.connect(_on_progress)
        worker.signals.finished.connect(_on_job_finished)
        worker.signals.error.connect(_on_job_error)
        progress.canceled.connect(worker.cancel)

        self.thread_pool.start(worker)

    # --- Context Menu Logic ---

    def handle_context_menu_request(self, pos, tree: QTreeWidget, group: ItemGroupEnum):
        item = tree.itemAt(pos)
        if not item:
            return

        match group.value:
            case ItemGroupEnum.DATASETS.value:
                self._handle_dataset_context_menu(item, pos, tree)
            case ItemGroupEnum.WORKSPACE_FILES.value:
                self._handle_workspace_file_context_menu(item, pos, tree)
            case ItemGroupEnum.MODELS.value:
                self._handle_models_context_menu(item, pos, tree)
            case ItemGroupEnum.PREDICTIONS.value:
                self._handle_predictions_context_menu(item, pos, tree)

    def _handle_dataset_context_menu(self, item, pos, tree):
        try:
            item_type = infer_selection_item_type(
                ItemGroupEnum.DATASETS, item, self._workspace
            )
            
            menu = QMenu(self.parent_widget)
            
            if item_type == ItemTypeEnum.DATASET_BASE:
                dataset_name = item.data(0, Qt.ItemDataRole.UserRole)
                dataset = self._workspace.datasets.get(dataset_name)
                
                refresh_action = menu.addAction("Refresh...")
                refresh_action.triggered.connect(
                    lambda: self._execute_dataset_refresh(dataset_name)
                )

                # Edit Arenas (multi-arena datasets only)
                if dataset and getattr(dataset, "is_multi_arena", False):
                    edit_arenas_action = menu.addAction("Edit Arenas...")
                    edit_arenas_action.triggered.connect(
                        lambda: self._edit_arenas(dataset_name)
                    )

                # Export Annotations (image/frame/video datasets) -> bundle
                if dataset and dataset.type in (
                    DatasetType.IMAGE_COLLECTION,
                    DatasetType.FRAME_SUBSET,
                    DatasetType.VIDEO_COLLECTION,
                ):
                    export_annotations_action = menu.addAction("Export Annotations...")
                    export_annotations_action.triggered.connect(
                        lambda: self._export_annotations(dataset_name)
                    )

                # Behavior Export Action (Legacy/Labels)
                if dataset and dataset.type == DatasetType.VIDEO_COLLECTION:
                    labels = self._workspace.behavior_labels.get_behavior_labels(dataset_name)
                    if labels and not labels.bouts.empty:
                        export_behavior_action = menu.addAction("Export Behavior Labels...")
                        export_behavior_action.triggered.connect(
                            lambda: self._export_behavior_labels(dataset_name)
                        )

                # --- Behavior Model Export Actions ---
                if self._current_model_run:
                    # Check if any predictions exist for this dataset in this run
                    run_preds = self._workspace.behavior_predictions.get_behavior_predictions_for_run(self._current_model_run)
                    if dataset_name in run_preds:
                        menu.addSeparator()
                        export_menu = menu.addMenu("Export Prediction Results")
                        
                        export_charts_action = export_menu.addAction("Export Charts (.png)")
                        export_charts_action.triggered.connect(
                            lambda: self._on_export_behavior_charts(dataset_name)
                        )
                        
                        jitter_action = export_menu.addAction("Export Jitter Analysis (.png)")
                        jitter_action.triggered.connect(
                            lambda: self._on_export_jitter_analysis(dataset_name)
                        )
                        
                        export_bouts_action = export_menu.addAction("Export Bouts (.json)")
                        export_bouts_action.triggered.connect(
                            lambda: self._on_export_bouts(dataset_name)
                        )

                menu.addSeparator()
                delete_action = menu.addAction("Delete...")
                delete_action.triggered.connect(
                    lambda: self._delete_dataset(dataset_name)
                )

            elif item_type == ItemTypeEnum.DATASET_VIDEO:
                # Video specific actions
                hierarchy = get_item_hierarchy(item)
                dataset_name = hierarchy[1]
                video_relative_path = hierarchy[-1]
                video_stem = Path(video_relative_path).stem
                
                if self._current_model_run:
                    # Check if predictions exist for this specific video
                    if self._workspace.behavior_predictions.has_behavior_prediction(
                        self._current_model_run, dataset_name, video_stem
                    ):
                        export_menu = menu.addMenu("Export Prediction Results")
                        
                        export_clip_action = export_menu.addAction("Export Clip (.mp4)")
                        export_clip_action.triggered.connect(
                            lambda: self._on_export_behavior_clip(dataset_name, video_relative_path)
                        )
                        
                        export_charts_action = export_menu.addAction("Export Charts (.png)")
                        export_charts_action.triggered.connect(
                            lambda: self._on_export_behavior_charts(dataset_name, video_relative_path)
                        )
                        
                        export_bouts_action = export_menu.addAction("Export Bouts (.json)")
                        export_bouts_action.triggered.connect(
                            lambda: self._on_export_bouts(dataset_name, video_relative_path)
                        )

            if not menu.isEmpty():
                menu.exec(tree.viewport().mapToGlobal(pos))
                
        except (ValueError, AttributeError) as e:
            logging.error(f"Error while bringing up context menu for item {item}: {str(e)}", exc_info=True)

    def _handle_workspace_file_context_menu(self, item, pos, tree):
        # Helper to reconstruct path from tree items
        def _get_full_path():
            full_path_parts = []
            path_item = item
            while path_item:
                full_path_parts.append(str(path_item.text(0)))
                path_item = path_item.parent()
            # Reverse, remove root (workspace name), join
            return os.path.join(
                self._workspace.base_dir, *full_path_parts[::-1][1:]
            )

        full_path = _get_full_path()

        menu = QMenu(self.parent_widget)
        
        delete_action = menu.addAction("Delete from Disk...")
        delete_action.triggered.connect(lambda: self._delete_path(full_path))
        
        open_browser_action = menu.addAction("Open in File Browser...")
        open_browser_action.triggered.connect(lambda: self._open_browser(full_path))
        
        menu.exec(tree.viewport().mapToGlobal(pos))

    # --- Worker Logic for Context Menus ---

    def _execute_dataset_refresh(self, dataset_name: str):
        if not self._workspace: return
        try:
            diff = self._workspace.datasets.refresh_dataset(dataset_name, dry_run=True)
            added, removed = diff['added'], diff['removed']

            if not added and not removed:
                QMessageBox.information(
                    self.parent_widget, "Refresh", "Dataset is up to date."
                )
                return

            if removed:
                msg = (
                    f"The following {len(removed)} items exist in the dataset "
                    "but were not found on disk:\n\n"
                )
                msg += "\n".join(f"- {f}" for f in removed[:10])
                if len(removed) > 10:
                    msg += f"\n...and {len(removed) - 10} more."
                msg += "\n\nRemove these entries from the dataset?"
                
                reply = QMessageBox.warning(
                    self.parent_widget, "Confirm Deletion", msg,
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.No:
                    return

            self._workspace.datasets.refresh_dataset(dataset_name, dry_run=False)
            
            summary = f"Refresh complete for '{dataset_name}'.\n"
            if added: summary += f"\nFound {len(added)} new items."
            if removed: summary += f"\nRemoved {len(removed)} missing items."
            
            QMessageBox.information(self.parent_widget, "Refresh Complete", summary)
            MessageBus.get().publish("workspace/datasets/refreshed")

        except Exception as e:
            QMessageBox.critical(
                self.parent_widget, "Error", f"Failed to refresh dataset: {e}"
            )

    def _delete_dataset(self, dataset_name: str):
        msg = QMessageBox(self.parent_widget)
        msg.setWindowTitle("Confirm Item Deletion")
        msg.setIcon(QMessageBox.Icon.Warning)
        highlighted_dataset_name = (
            f'<span style="color: #e74c3c; font-weight: bold;">{dataset_name}</span>'
        )
        msg.setText(
            "The selected action will permanently delete the following dataset:<br><br>"
            f"{highlighted_dataset_name}<br><br><b>Are you sure?</b>"
        )
        msg.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel
        )
        if msg.exec() == QMessageBox.StandardButton.Yes:
            try:
                self._workspace.delete_dataset(dataset_name)
            except Exception as e:
                logging.error(f"Failed to delete {dataset_name}: {str(e)}")

            logging.info(f"Deleted: {dataset_name}")
            self.parent_widget._update_data_tree()
            # Refresh tree
            MessageBus.get().publish("request/workspace/projects/refresh")
            if self.parent_widget.dropdown.currentText() == ItemGroupEnum.DATASETS.value:
                 MessageBus.get().publish("request/workspace/datasets/refresh")
            else:
                 # If in workspace files mode, we trigger a re-render manually
                 # or rely on the widget to catch the signal.
                 # The main widget should connect refresh_projects to its own update.
                 pass


    def _delete_path(self, path_str: str):
        msg = QMessageBox(self.parent_widget)
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
            except (FileNotFoundError, OSError) as e:
                logging.error(f"Failed to delete {path_str}: {str(e)}")

            logging.info(f"Deleted: {path_str}")
            self.parent_widget._update_data_tree()
            MessageBus.get().publish("workspace/files/refreshed")
            # Refresh tree
            MessageBus.get().publish("request/workspace/projects/refresh")
            if self.parent_widget.dropdown.currentText() == ItemGroupEnum.DATASETS.value:
                 MessageBus.get().publish("request/workspace/datasets/refresh")
            else:
                 # If in workspace files mode, we trigger a re-render manually
                 # or rely on the widget to catch the signal.
                 # The main widget should connect refresh_projects to its own update.
                 pass

    def _open_browser(self, path_str: str):
        url = QUrl.fromLocalFile(
            path_str if os.path.isdir(path_str) else os.path.dirname(path_str)
        )
        QDesktopServices.openUrl(url)

    def _export_behavior_labels(self, dataset_name: str):
        if not self._workspace:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self.parent_widget,
            "Export Behavior Labels",
            f"{dataset_name}_behavior_labels.json",
            "JSON Files (*.json)"
        )

        if not file_path:
            return

        try:
            labels = self._workspace.behavior_labels.get_behavior_labels(dataset_name)
            # Convert DataFrame to list of dicts for JSON export
            bouts_data = labels.bouts.to_dict(orient="records")

            # Also include the behaviors list for completeness
            export_data = {
                "dataset_name": dataset_name,
                "behaviors": labels.behaviors,
                "bouts": bouts_data
            }

            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=4)

            QMessageBox.information(
                self.parent_widget,
                "Export Successful",
                f"Behavior labels exported to:\n{file_path}"
            )
        except Exception as e:
            logging.error(f"Failed to export behavior labels: {e}")
            QMessageBox.critical(
                self.parent_widget,
                "Export Failed",
                f"An error occurred during export:\n{e}"
            )

    # --- Behavior Export Actions ---

    def _on_export_behavior_clip(self, dataset_name: str, video_relative_path: str):
        if not self._workspace or not self._current_model_run:
            return

        dataset = self._workspace.datasets.get(dataset_name)
        video_path = Path(dataset.base_data_path) / video_relative_path
        video_stem = video_path.stem

        import cv2
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        dlg = ExportClipDialog(self.parent_widget, total_frames, fps, 0)
        if dlg.exec():
            s_f, e_f = dlg.get_range()
            o_p = dlg.get_output_path()

            pb_ds = self._workspace.get_behavior_predictions(self._current_model_run, dataset_name, video_stem, False)
            gb_ds = self._workspace.behavior_labels.get_behavior_labels(dataset_name)

            sel_b = pb_ds.behaviors if pb_ds else []
            skeleton = self._active_project.skeleton if self._active_project else []
            body_parts = self._active_project.body_parts if self._active_project else []
            identities = self._active_project.identities if self._active_project else []

            from whisker.services.behavior_classification.internal.workers.behavior_export import BehaviorExportJob
            worker = Worker(BehaviorExportJob(
                source_video_path=video_path,
                output_video_path=o_p,
                start_frame=s_f,
                end_frame=e_f,
                pred_behavior_ds=pb_ds,
                gt_behavior_ds=gb_ds,
                project_skeleton=skeleton,
                project_bodyparts=body_parts,
                project_identities=identities,
                render_config={"prob_threshold": 0.5, "selected_behaviors": sel_b}
            ))
            worker.signals.finished.connect(lambda msg: QMessageBox.information(self.parent_widget, "Export Complete", msg))
            worker.signals.error.connect(lambda e: QMessageBox.critical(self.parent_widget, "Export Failed", e))

            self.thread_pool.start(worker)

    def _on_export_behavior_charts(self, dataset_name: str, video_relative_path: Optional[str] = None):
        if not self._workspace or not self._current_model_run:
            return

        if video_relative_path:
            QMessageBox.information(self.parent_widget, "Note", "Single video chart export currently requires opening the video in the Evaluation tab.")
            return

        behaviors = set()
        gt_ds = self._workspace.behavior_labels.get_behavior_labels(dataset_name)
        if gt_ds: behaviors.update(gt_ds.behaviors)

        run_preds = self._workspace.behavior_predictions.get_behavior_predictions_for_run(self._current_model_run)
        ds_vids = run_preds.get(dataset_name, {})
        if ds_vids:
            p_ds = self._workspace.get_behavior_predictions(self._current_model_run, dataset_name, next(iter(ds_vids)))
            if p_ds: behaviors.update(p_ds.behaviors)

        if not behaviors:
            QMessageBox.warning(self.parent_widget, "Export Error", "No behavior data found.")
            return

        video_data = gather_dataset_video_data(self._workspace, self._current_model_run, dataset_name, list(ds_vids.keys()))

        dlg = ChartExportOptionsDialog(
            sorted(list(behaviors)), 
            video_data, 
            render_dataset_behavior_chart,
            self._current_model_run,
            self.parent_widget
        )

        if dlg.exec():
            opts = dlg.get_options()
            def_n = f"dataset_behavior_{self._current_model_run}_{dataset_name}_{opts['behavior']}.png"
            path, _ = QFileDialog.getSaveFileName(self.parent_widget, "Save Dataset Chart", def_n, "PNG Images (*.png)")
            if path:
                try:
                    import matplotlib.pyplot as plt
                    fig = render_dataset_behavior_chart(opts, self._current_model_run, video_data)
                    fig.savefig(path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    QMessageBox.information(self.parent_widget, "Export Complete", f"Dataset behavior chart exported to {path}")
                except Exception as e:
                    QMessageBox.critical(self.parent_widget, "Export Failed", f"Error: {e}")

    def _on_export_jitter_analysis(self, dataset_name: str):
        if not self._workspace or not self._current_model_run:
            return

        def_n = f"jitter_analysis_{self._current_model_run}_{dataset_name}.png"
        path, _ = QFileDialog.getSaveFileName(self.parent_widget, "Save Jitter Plot", def_n, "PNG Images (*.png)")
        if not path: return

        from whisker.services.behavior_classification.internal.workers.analysis import BehaviorJitterAnalysisJob
        worker = Worker(BehaviorJitterAnalysisJob(
            workspace=self._workspace, 
            model_run_name=self._current_model_run, 
            dataset_name=dataset_name, 
            output_path=Path(path), 
            iou_threshold=0.1
        ))
        worker.signals.finished.connect(lambda m: QMessageBox.information(self.parent_widget, "Analysis Complete", m))
        worker.signals.error.connect(lambda e: QMessageBox.critical(self.parent_widget, "Analysis Failed", e))

        self.thread_pool.start(worker)

    def _on_export_bouts(self, dataset_name: str, video_relative_path: Optional[str] = None):
        if not self._workspace or not self._current_model_run:
            return

        dlg = BoutExportDialog(self.parent_widget)
        if dlg.exec():
            params = dlg.get_params()
            suffix = f"_{Path(video_relative_path).stem}" if video_relative_path else ""
            def_n = f"bouts_{self._current_model_run}_{dataset_name}{suffix}.json"
            save_p, _ = QFileDialog.getSaveFileName(self.parent_widget, "Save Bouts JSON", def_n, "JSON Files (*.json)")
            if not save_p: return

            from whisker.services.behavior_classification.internal.core.utils.bout_extraction import extract_bouts
            try:
                all_bouts = []
                fps = 30.0 

                if video_relative_path:
                    v_stem = Path(video_relative_path).stem
                    pred_ds = self._workspace.get_behavior_predictions(self._current_model_run, dataset_name, v_stem)
                    if pred_ds and not pred_ds.per_frame_probabilities.empty:
                        b_df = extract_bouts(pred_ds.per_frame_probabilities, pred_ds.behaviors, fps, params)
                        if not b_df.empty: 
                            b_df['video_key'] = v_stem
                            all_bouts.append(b_df)
                else:
                    run_preds = self._workspace.behavior_predictions.get_behavior_predictions_for_run(self._current_model_run)
                    ds_vids = run_preds.get(dataset_name, {})
                    for v_stem in ds_vids:
                        pred_ds = self._workspace.get_behavior_predictions(self._current_model_run, dataset_name, v_stem)
                        if pred_ds and not pred_ds.per_frame_probabilities.empty:
                            b_df = extract_bouts(pred_ds.per_frame_probabilities, pred_ds.behaviors, fps, params)
                            if not b_df.empty: 
                                b_df['video_key'] = v_stem
                                all_bouts.append(b_df)

                if not all_bouts:
                    QMessageBox.information(self.parent_widget, "Export", "No bouts found.")
                    return

                import pandas as pd
                final_df = pd.concat(all_bouts, ignore_index=True)

                export_data = {
                    "model_run": self._current_model_run,
                    "dataset": dataset_name,
                    "params": {
                        "probability_threshold": params.probability_threshold,
                        "min_bout_duration_sec": params.min_bout_duration_sec,
                        "max_gap_fill_sec": params.max_gap_fill_sec
                    },
                    "bouts": final_df.to_dict(orient='records')
                }

                with open(save_p, 'w') as f:
                    json.dump(export_data, f, indent=4)

                QMessageBox.information(self.parent_widget, "Export Complete", f"Bouts exported to {save_p}")
            except Exception as e:
                logging.error(f"Export bouts failed: {e}", exc_info=True)
                QMessageBox.critical(self.parent_widget, "Export Failed", f"Error: {e}")

    def _handle_models_context_menu(self, item: QTreeWidgetItem, pos, tree: QTreeWidget):
        if item.parent() is None:
            return

        model_name = item.data(0, Qt.ItemDataRole.UserRole)
        if not model_name or not self._workspace:
            return

        workspace_path = str(self._workspace.base_dir)

        menu = QMenu(self.parent_widget)

        label_menu = menu.addMenu("Label")
        
        none_action = label_menu.addAction("None")
        none_action.setCheckable(True)
        
        star_action = label_menu.addAction("⭐")
        star_action.setCheckable(True)

        emoji_actions = []
        for i in range(1, 10):
            emoji = f"{i}️⃣"
            action = label_menu.addAction(emoji)
            action.setCheckable(True)
            emoji_actions.append((emoji, action))

        current_label = get_model_label(workspace_path, model_name)

        none_action.setChecked(current_label is None)
        star_action.setChecked(current_label == "⭐")
        for emoji, action in emoji_actions:
            action.setChecked(current_label == emoji)

        none_action.triggered.connect(lambda: self._set_model_label_and_refresh(workspace_path, model_name, None))
        star_action.triggered.connect(lambda: self._set_model_label_and_refresh(workspace_path, model_name, "⭐"))
        for emoji, action in emoji_actions:
            action.triggered.connect(lambda checked, e=emoji: self._set_model_label_and_refresh(workspace_path, model_name, e))

        menu.addSeparator()

        hide_action = menu.addAction("Hide")
        hide_action.setCheckable(True)
        is_hidden = is_model_hidden(workspace_path, model_name)
        hide_action.setChecked(is_hidden)
        hide_action.triggered.connect(lambda checked: self._set_model_hidden_and_refresh(workspace_path, model_name, checked))

        menu.exec(tree.viewport().mapToGlobal(pos))

    def _set_model_label_and_refresh(self, workspace_path: str, model_name: str, label: Optional[str]):
        set_model_label(workspace_path, model_name, label)
        MessageBus.get().publish("workspace/models/refreshed")

    def _set_model_hidden_and_refresh(self, workspace_path: str, model_name: str, hidden: bool):
        set_model_hidden(workspace_path, model_name, hidden)
        MessageBus.get().publish("workspace/models/refreshed")

    def _handle_predictions_context_menu(self, item: QTreeWidgetItem, pos, tree: QTreeWidget):
        if item.parent() is None:
            return

        pred_name = item.data(0, Qt.ItemDataRole.UserRole)
        if not pred_name or not self._workspace:
            return

        workspace_path = str(self._workspace.base_dir)

        menu = QMenu(self.parent_widget)

        label_menu = menu.addMenu("Label")
        
        none_action = label_menu.addAction("None")
        none_action.setCheckable(True)
        
        star_action = label_menu.addAction("⭐")
        star_action.setCheckable(True)

        emoji_actions = []
        for i in range(1, 10):
            emoji = f"{i}️⃣"
            action = label_menu.addAction(emoji)
            action.setCheckable(True)
            emoji_actions.append((emoji, action))

        current_label = get_prediction_label(workspace_path, pred_name)

        none_action.setChecked(current_label is None)
        star_action.setChecked(current_label == "⭐")
        for emoji, action in emoji_actions:
            action.setChecked(current_label == emoji)

        none_action.triggered.connect(lambda: self._set_prediction_label_and_refresh(workspace_path, pred_name, None))
        star_action.triggered.connect(lambda: self._set_prediction_label_and_refresh(workspace_path, pred_name, "⭐"))
        for emoji, action in emoji_actions:
            action.triggered.connect(lambda checked, e=emoji: self._set_prediction_label_and_refresh(workspace_path, pred_name, e))

        menu.addSeparator()

        hide_action = menu.addAction("Hide")
        hide_action.setCheckable(True)
        is_hidden = is_prediction_hidden(workspace_path, pred_name)
        hide_action.setChecked(is_hidden)
        hide_action.triggered.connect(lambda checked: self._set_prediction_hidden_and_refresh(workspace_path, pred_name, checked))

        menu.exec(tree.viewport().mapToGlobal(pos))

    def _set_prediction_label_and_refresh(self, workspace_path: str, pred_name: str, label: Optional[str]):
        set_prediction_label(workspace_path, pred_name, label)
        MessageBus.get().publish("workspace/predictions/refreshed")

    def _set_prediction_hidden_and_refresh(self, workspace_path: str, pred_name: str, hidden: bool):
        set_prediction_hidden(workspace_path, pred_name, hidden)
        MessageBus.get().publish("workspace/predictions/refreshed")