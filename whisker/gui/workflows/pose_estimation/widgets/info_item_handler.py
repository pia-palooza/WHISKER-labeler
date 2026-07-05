# UPDATE_FILE: whisker/gui/workflows/pose_estimation/info_handler.py
# (Assuming standard filename based on your architecture, replace with actual)
import logging
import json
from pathlib import Path
from datetime import datetime

import yaml
import pandas as pd
from PyQt6.QtWidgets import (
    QTreeWidget, QWidget, QVBoxLayout, QTabWidget, QTextEdit, QLabel, 
    QPushButton, QHBoxLayout, QFileDialog, QMessageBox, QTableWidget, 
    QTableWidgetItem, QHeaderView, QTreeWidgetItem, QRadioButton, 
    QButtonGroup, QTreeWidgetItemIterator, QCheckBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from whisker.core.study.dataset import DatasetType
from whisker.services.pose_estimation.public.topics import PoseEvaluationParams
from whisker.core.workspace import Workspace
from whisker.gui.widgets import data_explorer
from whisker.gui.workflows.base_info_item_handler import BaseInfoItemHandlerWidget
from whisker.gui.job_manager import JobManager
from whisker.gui.worker_wrapper import Worker


class NumericTableWidgetItem(QTableWidgetItem):
    """Custom table item that sorts numerically if possible, otherwise falls back to string sorting."""
    def __lt__(self, other):
        try:
            return float(self.text()) < float(other.text())
        except ValueError:
            return self.text() < other.text()


class NumericTreeWidgetItem(QTreeWidgetItem):
    """Custom tree item that sorts numerically if possible, otherwise falls back to string sorting."""
    def __lt__(self, other):
        col = self.treeWidget().sortColumn()
        try:
            return float(self.text(col)) < float(other.text(col))
        except ValueError:
            return self.text(col) < other.text(col)


class ModelSummaryLoaderThread(QThread):
    """Background thread to load model metadata and history without freezing the GUI."""
    data_ready = pyqtSignal(int, list)
    history_ready = pyqtSignal(str, object)  # (model_name, history_dict | None)

    def __init__(self, workspace: Workspace, model_names: list, parent=None):
        super().__init__(parent)
        self.workspace = workspace
        self.model_names = model_names

    def run(self):
        for i, model_name in enumerate(self.model_names):
            model_dir = self.workspace.pose_models.base_dir / model_name
            metadata_path = model_dir / "metadata.yaml"

            c_date = "N/A"
            if metadata_path.exists():
                c_date = datetime.fromtimestamp(metadata_path.stat().st_ctime).strftime("%Y-%m-%d %I:%M:%S %p")

            settings = yaml.safe_load(metadata_path.read_text()) if metadata_path.exists() else {}
            base_model = f"{settings.get('backbone', 'N/A')} ({settings.get('model_architecture', 'N/A')})"

            rmse, pck, epochs = "N/A", "N/A", "N/A"
            history = self._get_history(model_dir)
            self.history_ready.emit(model_name, history)

            if history:
                epochs = str(len(history.get("epochs", [])))
                val_rmse = history.get("val_rmse")
                if val_rmse and val_rmse[-1] is not None: rmse = f"{val_rmse[-1]:.4f}"
                val_pck = history.get("val_pck")
                if val_pck and val_pck[-1] is not None: pck = f"{val_pck[-1]:.2f}"

            self.data_ready.emit(i, [model_name, c_date, base_model, epochs, rmse, pck])

    def _get_history(self, model_dir: Path) -> dict | None:
        """Retrieves training history from a CSV file or a PyTorch checkpoint."""
        for csv_path in [model_dir / "checkpoints" / "training_metrics.csv", model_dir / "training_metrics.csv"]:
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    return {**df.to_dict(orient='list'), "epochs": list(range(1, len(df) + 1))}
                except Exception: pass

        ckpt = model_dir / "best_model.pth"
        if not ckpt.exists(): ckpt = model_dir / "checkpoints" / "best_model.pth"

        if ckpt.exists():
            import torch
            try:
                return torch.load(str(ckpt), map_location='cpu', weights_only=False).get('history')
            except Exception: pass
        return None


class PoseEstimationInfoItemHandlerWidget(BaseInfoItemHandlerWidget):
    """
    Handles the display of metadata, training performance, and evaluation summaries 
    for Pose Estimation models within the workspace.
    """

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)
        self._history_cache: dict[str, dict | None] = {}

    def show_workflow_item_info(self, workspace: Workspace, selection: data_explorer.Selection) -> bool:
        """Routes the selection to the appropriate display method based on item group."""
        logging.debug(f"{self.__class__.__name__} received selection: {selection}")
        if selection.group == data_explorer.ItemGroupEnum.WORKSPACE_FILES:
            relative_path = Path(*selection.item[1:])
            file_path = workspace.base_dir / relative_path
            return self.show_workflow_file_info(workspace, file_path)
        elif selection.group == data_explorer.ItemGroupEnum.MODELS:
            return self.show_pose_estimation_model_selection_info(workspace, selection)
        return False

    def show_workflow_file_info(self, workspace: Workspace, file_path: Path) -> bool:
        """Displays detailed information (performance/params) for a specific model directory."""
        logging.debug(f"{self.__class__.__name__} handling file: {file_path}")
        current_tab_index = self.tabs.currentIndex()
        self._clear_layout()

        if not file_path.is_dir():
            return False

        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        self.tabs.addTab(self._create_performance_tab(file_path), "Training Performance")
        self.tabs.addTab(self._create_params_tab(file_path), "Training Parameters")

        if 0 <= current_tab_index < self.tabs.count():
            self.tabs.setCurrentIndex(current_tab_index)
        return True

    def show_pose_estimation_model_selection_info(self, workspace: Workspace, selection: data_explorer.Selection) -> bool:
        """Determines if a single model or an entire project summary should be shown."""
        if selection.type == data_explorer.ItemTypeEnum.POSE_ESTIMATION_MODEL:
            model_dir = workspace.pose_models.base_dir / selection.item[-1]
            if model_dir.exists():
                return self.show_workflow_file_info(workspace, model_dir)
            else:
                self._clear_layout()
                self.layout.addWidget(QLabel("Model directory not found."))
                return True
        elif selection.type == data_explorer.ItemTypeEnum.POSE_ESTIMATION_MODEL_PROJECT:
            self.show_models_summary(workspace)
            return True
        return False

    def show_models_summary(self, workspace: Workspace):
        """Creates a tabbed view summarizing all models within a specific project."""
        self._clear_layout()
        model_names = workspace.pose_models.get()
        
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        self.tabs.addTab(self._create_models_summary_tab(workspace, model_names), "Models Summary")
        self.tabs.addTab(self._create_models_evaluation_summary_tab(workspace, model_names), "Models Evaluation Summary")

    def _create_models_summary_tab(self, workspace: Workspace, model_names: list) -> QWidget:
        """Generates a table widget and pre-fills it with placeholders while loading metrics."""
        container = QWidget()
        tab_layout = QVBoxLayout(container)

        table_widget = QTableWidget()
        table_widget.setSortingEnabled(True)
        table_widget.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        
        def _on_double_clicked(index):
            model_dir = workspace.pose_models.base_dir / model_names[index.row()]
            if model_dir.exists():
                self.show_workflow_file_info(workspace, model_dir)

        table_widget.doubleClicked.connect(_on_double_clicked)
        headers = ["Model Name", "Creation Date", "Base Model", "Epochs", "Val RMSE (px)", "Val PCK (%)"]
        table_widget.setColumnCount(len(headers))
        table_widget.setRowCount(len(model_names))
        table_widget.setHorizontalHeaderLabels(headers)

        # 1. Pre-fill the table with known names and "Loading..." placeholders
        table_widget.setSortingEnabled(False) # Turn off sorting during initial fill
        for row, model_name in enumerate(model_names):
            table_widget.setItem(row, 0, NumericTableWidgetItem(model_name))
            for col in range(1, len(headers)):
                item = NumericTableWidgetItem("Loading...")
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                table_widget.setItem(row, col, item)
                
        table_widget.resizeColumnsToContents()
        table_widget.setSortingEnabled(True)

        tab_layout.addWidget(table_widget)

        # 2. Fire up the background thread to overwrite the placeholders
        self._summary_loader = ModelSummaryLoaderThread(workspace, model_names, self)
        self._summary_loader.data_ready.connect(lambda row, data: self._populate_table_row(table_widget, row, data))
        self._summary_loader.history_ready.connect(lambda name, hist: self._history_cache.__setitem__(name, hist))
        self._summary_loader.start()

        export_btn = QPushButton("Export to CSV")
        export_btn.clicked.connect(lambda: self._export_models_to_csv(model_names, table_widget))
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(export_btn)
        tab_layout.addLayout(btn_layout)

        return container

    def _finish_loading(self, loading_label: QLabel, table_widget: QTableWidget):
        """Swaps the loading placeholder for the populated table."""
        loading_label.hide()
        loading_label.deleteLater()
        table_widget.show()

    def _populate_table_row(self, table_widget: QTableWidget, row_idx: int, row_data: list):
        """Updates the table UI cleanly from the background thread's emitted signal."""
        try:
            is_sorting_enabled = table_widget.isSortingEnabled()
        except RuntimeError:
            logging.error("_populate_table_row: table widget was deleted before data arrived", exc_info=True)
            return
        table_widget.setSortingEnabled(False)

        for col, text in enumerate(row_data):
            item = NumericTableWidgetItem(text)
            if col in [1, 3, 4, 5]:
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            table_widget.setItem(row_idx, col, item)
            
        table_widget.resizeColumnsToContents()
        table_widget.setSortingEnabled(is_sorting_enabled)

    def _create_models_evaluation_summary_tab(self, workspace: Workspace, model_names: list) -> QWidget:
        container = QWidget()
        tab_layout = QVBoxLayout(container)
        
        # --- Radio Toggle Layout ---
        toggle_layout = QHBoxLayout()
        btn_group = QButtonGroup(container)
        radio_ds = QRadioButton("Group by Dataset")
        radio_mod = QRadioButton("Group by Model")
        radio_ds.setChecked(True)
        btn_group.addButton(radio_ds)
        btn_group.addButton(radio_mod)
        toggle_layout.addWidget(radio_ds)
        toggle_layout.addWidget(radio_mod)
        toggle_layout.addStretch()
        tab_layout.addLayout(toggle_layout)

        tree = QTreeWidget()
        tree.setSortingEnabled(True)
        tree.setAlternatingRowColors(True)
        tree.header().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        tab_layout.addWidget(tree)
        
        radio_ds.toggled.connect(lambda checked: self._populate_evaluation_tree(workspace, tree, "dataset") if checked else None)
        radio_mod.toggled.connect(lambda checked: self._populate_evaluation_tree(workspace, tree, "model") if checked else None)
        
        # Initial Population
        self._populate_evaluation_tree(workspace, tree, "dataset")

        # --- Action Buttons ---
        btn_layout = QHBoxLayout()
        self.purity_checkbox = QCheckBox("Calculate Purity")
        self.purity_checkbox.setChecked(False)
        fill_metrics_btn = QPushButton("Fill Missing Metrics")
        fill_metrics_btn.clicked.connect(lambda: self._fill_missing_metrics(workspace, tree))
        export_btn = QPushButton("Export Summary")
        export_btn.clicked.connect(lambda: self._export_evaluations_to_csv(tree))
        btn_layout.addWidget(self.purity_checkbox)
        btn_layout.addStretch()
        btn_layout.addWidget(fill_metrics_btn)
        btn_layout.addWidget(export_btn)
        tab_layout.addLayout(btn_layout)

        return container

    def _populate_evaluation_tree(self, workspace: Workspace, tree: QTreeWidget, group_by: str):
        """Flattens the data collection, then builds the tree based on the selected grouping."""
        tree.clear()
        
        headers = ["mAP", "mAP@0.50", "mAP@0.75", "mAR", "Purity", "Mean MPJPE", "P90 MPJPE", "Mean PCK"]
        if group_by == "dataset":
            tree.setHeaderLabels(["Dataset", "Model"] + headers)
        else:
            tree.setHeaderLabels(["Model", "Dataset"] + headers)

        eval_data = []

        for dataset in workspace.datasets.values():
            if dataset.type == DatasetType.VIDEO_COLLECTION: continue
            for model_name in workspace.pose_predictions.get_models_with_predictions_for_dataset(dataset.name):
                metrics = workspace.pose_predictions.get_evaluation_metrics(model_name, dataset.name)
                agg = metrics.get("aggregate_metrics", {}) if metrics else {}
                
                def get_met(k, fmt, mult=1.0):
                    val = agg.get(k)
                    return f"{(val * mult):{fmt}}" if isinstance(val, (int, float)) else "N/A"

                eval_data.append({
                    "model": model_name, "ds_type": dataset.type.value, "ds_name": dataset.name,
                    "map": get_met('mAP', '.3f'),
                    "map_50": get_met('mAP_50', '.3f'),
                    "map_75": get_met('mAP_75', '.3f'),
                    "mar": get_met('mAR', '.3f'),
                    "purity": get_met('purity', '.3f'),
                    "mpjpe": get_met('mean_mpjpe', '.2f'),
                    "p90": get_met('p90_mpjpe', '.2f'),
                    "pck": get_met('mean_pck', '.2f', 100),
                })

        nodes = {}
        for d in eval_data:
            row_data = [
                d["map"], d["map_50"], d["map_75"], d["mar"], 
                d["purity"], d["mpjpe"], d["p90"], d["pck"]
            ]
            
            if group_by == "dataset":
                if d["ds_type"] not in nodes:
                    nodes[d["ds_type"]] = NumericTreeWidgetItem(tree, [d["ds_type"]] + [""] * 10)
                    nodes[d["ds_type"]].setExpanded(True)
                ds_key = f"{d['ds_type']}_{d['ds_name']}"
                if ds_key not in nodes:
                    nodes[ds_key] = NumericTreeWidgetItem(nodes[d["ds_type"]], [d["ds_name"]] + [""] * 10)
                    nodes[ds_key].setExpanded(True)
                
                child = NumericTreeWidgetItem(nodes[ds_key], ["", d["model"]] + row_data)
            
            else:
                if d["model"] not in nodes:
                    nodes[d["model"]] = NumericTreeWidgetItem(tree, [d["model"]] + [""] * 10)
                    nodes[d["model"]].setExpanded(True)
                
                child = NumericTreeWidgetItem(nodes[d["model"]], ["", f'{d["ds_type"]} - {d["ds_name"]}'] + row_data)

            for col in range(2, 11):
                child.setTextAlignment(col, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            
            child.setData(0, Qt.ItemDataRole.UserRole, {"dataset": d["ds_name"], "model": d["model"]})

    def _fill_missing_metrics(self, workspace: Workspace, tree: QTreeWidget):
        iterator = QTreeWidgetItemIterator(tree)
        while iterator.value():
            item = iterator.value()
            meta = item.data(0, Qt.ItemDataRole.UserRole)
            
            if meta and item.text(3) == "N/A":
                logging.info(f"Reloading metrics for model '{meta['model']}' on dataset '{meta['dataset']}'")
                from whisker.services.pose_estimation.internal.workers.pose_evaluation import PoseEvaluationJob
                worker = Worker(
                    PoseEvaluationJob(
                        workspace=workspace,
                        params=PoseEvaluationParams(
                            run_name=meta['model'],
                            dataset_name=meta['dataset'],
                            pck_threshold=10.0,
                            data_split="all",
                            swap_identities=False,
                            calculate_purity=self.purity_checkbox.isChecked()
                        )
                    )
                )
                worker.signals.finished.connect(
                    lambda results, it=item: self._receive_pose_evaluation_results(it, results)
                )
                JobManager.get().submit_worker("Pose Estimation Evaluation", worker)
            
            iterator += 1

    def _receive_pose_evaluation_results(self, model_item: QTreeWidgetItem, results: dict):
        """Callback to receive updated evaluation results and update the tree widget accordingly."""
        agg = results.get("aggregate_metrics", {})
        
        def get_met(k, fmt, mult=1.0):
            val = agg.get(k)
            return f"{(val * mult):{fmt}}" if isinstance(val, (int, float)) else "N/A"

        model_item.setText(2, get_met('mAP', '.3f'))
        model_item.setText(3, get_met('mAP_50', '.3f'))
        model_item.setText(4, get_met('mAP_75', '.3f'))
        model_item.setText(5, get_met('mAR', '.3f'))
        model_item.setText(6, get_met('purity', '.3f'))
        model_item.setText(7, get_met('mean_mpjpe', '.2f'))
        model_item.setText(8, get_met('p90_mpjpe', '.2f'))
        model_item.setText(9, get_met('mean_pck', '.2f', 100))

    def _export_evaluations_to_csv(self, tree: QTreeWidget):
        path, _ = QFileDialog.getSaveFileName(self, "Save Export", f"eval_summary_{datetime.now().strftime('%Y%m%d')}.csv", "CSV Files (*.csv)")
        if not path: return

        rows = []
        iterator = QTreeWidgetItemIterator(tree)
        while iterator.value():
            item = iterator.value()
            meta = item.data(0, Qt.ItemDataRole.UserRole)
            
            if meta:
                rows.append({
                    "Dataset": meta["dataset"], "Model": meta["model"],
                    "mAP": item.text(2),
                    "mAP@0.50": item.text(3),
                    "mAP@0.75": item.text(4),
                    "mAR": item.text(5),
                    "Purity": item.text(6),
                    "Mean MPJPE (px)": item.text(7),
                    "P90 MPJPE (px)": item.text(8),
                    "Mean PCK (%)": item.text(9),
                })
            
            iterator += 1

        pd.DataFrame(rows).to_csv(path, index=False)

    def _export_models_to_csv(self, model_names: list, table_widget: QTableWidget):
        """Exports the QTableWidget data to a CSV file."""
        if not model_names:
            QMessageBox.warning(self, "Export Error", "No data available to export.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save Export", f"models_summary_{datetime.now().strftime('%Y%m%d')}.csv", "CSV Files (*.csv)")
        if not path: return

        try:
            headers = [table_widget.horizontalHeaderItem(i).text() for i in range(table_widget.columnCount())]
            data = [{headers[c]: table_widget.item(r, c).text() for c in range(table_widget.columnCount())} 
                    for r in range(table_widget.rowCount())]
            pd.DataFrame(data).to_csv(path, index=False)
            logging.info(f"Exported to {path}")
        except Exception as e:
            logging.error(f"Export failed: {e}")
            QMessageBox.critical(self, "Export Failed", str(e))

    def _clear_layout(self):
        """Removes all widgets from the main layout to prepare for new data."""
        while self.layout.count():
            child = self.layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()

    def _create_params_tab(self, model_dir: Path) -> QTextEdit:
        """Returns a read-only text editor displaying model metadata."""
        path = model_dir / "metadata.yaml"
        text_edit = QTextEdit(readOnly=True)
        text_edit.setText(path.read_text() if path.exists() else "No metadata found.")
        return text_edit

    def _get_history(self, model_dir: Path) -> dict | None:
        """Retrieves training history from a CSV file or a PyTorch checkpoint."""
        for csv_path in [model_dir / "checkpoints" / "training_metrics.csv", model_dir / "training_metrics.csv"]:
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    return {**df.to_dict(orient='list'), "epochs": list(range(1, len(df) + 1))}
                except Exception: pass

        ckpt = model_dir / "best_model.pth"
        if not ckpt.exists(): ckpt = model_dir / "checkpoints" / "best_model.pth"

        if ckpt.exists():
            import torch
            try:
                return torch.load(str(ckpt), map_location='cpu', weights_only=False).get('history')
            except Exception: pass
        return None
    
    def _get_evaluation_metrics(self, eval_path: Path) -> dict | None:
        """Loads evaluation metrics from a JSON file."""
        if eval_path.exists():
            try:
                with open(eval_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Failed to load evaluation metrics: {e}")
        return None

    def _create_performance_tab(self, model_dir: Path) -> QWidget:
        """Returns a monitor widget displaying training loss, RMSE, and PCK."""
        monitor = TrainingMonitorWidget()
        monitor.configure([
            {"key": "loss", "label": "Loss", "log_scale": True, "color_train": "m", "color_val": "g", "higher_is_better": False},
            {"key": "rmse", "label": "RMSE (px)", "color_train": "m", "color_val": "g", "higher_is_better": False},
            {"key": "pck", "label": "PCK (%)", "color_train": "m", "color_val": "g", "higher_is_better": True}
        ])
        monitor.stop_button.hide()
        monitor.new_run_button.hide()

        model_name = model_dir.name
        history = self._history_cache.get(model_name) or self._get_history(model_dir)
        if history:
            self._history_cache[model_name] = history
            logging.info(f"Loaded training history for {model_name}")
            monitor.update_data({"type": "history_restore", "history": history})
            return monitor
        return QLabel("No metrics found.")