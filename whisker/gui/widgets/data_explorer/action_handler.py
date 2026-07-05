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
)

from whisker.core.workspace import Workspace, DatasetType, Project
from whisker.gui.dialogs import (
    CreateDatasetDialog,
    WarnIfExistsDialog,
    ImportLabelsDialog,
)
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

        dialog = CreateDatasetDialog(self._workspace)
        dialog.exec()
        if dialog.result() == QDialog.DialogCode.Accepted:
            result = dialog.get_dataset_info()
            if result:
                name, inferred_type, folder_path = result
                self._workspace.create_dataset(
                    name,
                    inferred_type,
                    folder_path,
                    warn_if_exists=lambda msg: WarnIfExistsDialog.run(msg, parent=self.parent_widget),
                )
                MessageBus.get().publish("request/workspace/datasets/refresh")

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