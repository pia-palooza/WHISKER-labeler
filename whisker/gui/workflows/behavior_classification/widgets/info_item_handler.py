import logging
import json
import os
from pathlib import Path
from datetime import datetime

import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTabWidget, QTextEdit, QLabel, 
    QTableWidget, QTableWidgetItem, QHeaderView, QTreeWidget, 
    QTreeWidgetItem, QRadioButton, QHBoxLayout, QButtonGroup,
    QPushButton, QFileDialog, QMessageBox, QTreeWidgetItemIterator
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from whisker.core.workspace import Workspace
from whisker.gui.widgets import data_explorer
from whisker.gui.workflows.base_info_item_handler import BaseInfoItemHandlerWidget
from whisker.gui.job_manager import JobManager
from whisker.gui.worker_wrapper import Worker


class NumericTableWidgetItem(QTableWidgetItem):
    """Custom table item that sorts numerically if possible."""
    def __lt__(self, other):
        try:
            return float(self.text()) < float(other.text())
        except ValueError:
            return self.text() < other.text()


class NumericTreeWidgetItem(QTreeWidgetItem):
    """Custom tree item that sorts numerically if possible."""
    def __lt__(self, other):
        col = self.treeWidget().sortColumn()
        try:
            return float(self.text(col)) < float(other.text(col))
        except ValueError:
            return self.text(col) < other.text(col)


class BehaviorModelSummaryLoaderThread(QThread):
    """Background thread to load behavior model metadata and history."""
    data_ready = pyqtSignal(int, list)

    def __init__(self, workspace: Workspace, model_names: list, parent=None):
        super().__init__(parent)
        self.workspace = workspace
        self.model_names = model_names

    def run(self):
        for i, model_name in enumerate(self.model_names):
            model_dir = self.workspace.behavior_models.base_dir / model_name
            config_path = model_dir / "model_config.json"
            
            c_date = "N/A"
            if config_path.exists():
                c_date = datetime.fromtimestamp(config_path.stat().st_ctime).strftime("%Y-%m-%d %I:%M:%S %p")

            try:
                config = json.loads(config_path.read_text()) if config_path.exists() else {}
            except Exception:
                config = {}
            
            model_type = config.get('model_type', 'N/A')
            
            auc, precision, recall, epochs = "N/A", "N/A", "N/A", "N/A"
            history = self._get_history(model_dir) 
            
            if history:
                epochs = str(len(history.get("epochs", [])))
                val_auc = history.get("val_auc") or history.get("auc") or history.get("train_auc")
                if val_auc and val_auc[-1] is not None: auc = f"{val_auc[-1]:.4f}"
                val_prec = history.get("val_precision") or history.get("precision") or history.get("train_precision")
                if val_prec and val_prec[-1] is not None: precision = f"{val_prec[-1]:.4f}"
                val_rec = history.get("val_recall") or history.get("recall") or history.get("train_recall")
                if val_rec and val_rec[-1] is not None: recall = f"{val_rec[-1]:.4f}"

            self.data_ready.emit(i, [model_name, c_date, model_type, epochs, auc, precision, recall])

    def _get_history(self, model_dir: Path) -> dict | None:
        csv_path = model_dir / "training_metrics.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                return {**df.to_dict(orient='list'), "epochs": list(range(1, len(df) + 1))}
            except Exception: pass
        return None


class BehaviorClassificationInfoItemHandlerWidget(BaseInfoItemHandlerWidget):
    """
    Displays details for a selected Behavior Model directory.
    Provides visualization of training metrics, raw configuration view, 
    and evaluation summaries.
    """
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

    def show_workflow_item_info(self, workspace: Workspace, selection: data_explorer.Selection) -> bool:
        logging.debug(f"{self.__class__.__name__} received selection: {selection}")
        if selection.group == data_explorer.ItemGroupEnum.WORKSPACE_FILES:
            relative_path = Path(*selection.item[1:])
            file_path = workspace.base_dir / relative_path
            return self.show_workflow_file_info(workspace, file_path)
        elif selection.group == data_explorer.ItemGroupEnum.MODELS:
            return self.show_behavior_classification_model_selection_info(workspace, selection)
        return False

    def show_workflow_file_info(self, workspace: Workspace, file_path: Path) -> bool:
        logging.info(f"{self.__class__.__name__} handling file: {file_path}")
        self._clear_layout()

        if not file_path.is_dir():
            return False

        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        self._add_performance_tab(file_path)
        self._add_config_tab(file_path)
        return True

    def show_behavior_classification_model_selection_info(self, workspace: Workspace, selection: data_explorer.Selection) -> bool:
        if selection.type == data_explorer.ItemTypeEnum.BEHAVIOR_CLASSIFICATION_MODEL:
            model_dir = workspace.behavior_models.base_dir / selection.item[-1]
            if model_dir.exists():
                return self.show_workflow_file_info(workspace, model_dir)
            else:
                self._clear_layout()
                self.layout.addWidget(QLabel("Model directory not found."))
                return True
        elif selection.type == data_explorer.ItemTypeEnum.BEHAVIOR_CLASSIFICATION_MODEL_PROJECT:
            self.show_models_summary(workspace)
            return True
        return False

    def show_models_summary(self, workspace: Workspace):
        self._clear_layout()
        model_names = workspace.behavior_models.get()
        
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        self.tabs.addTab(self._create_models_summary_tab(workspace, model_names), "Models Summary")
        self.tabs.addTab(self._create_models_evaluation_summary_tab(workspace, model_names), "Models Evaluation Summary")

    def _clear_layout(self):
        while self.layout.count():
            child = self.layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.tabs = None

    def _add_performance_tab(self, model_dir: Path):
        metrics_path = model_dir / "training_metrics.csv"
        if not metrics_path.exists():
            self.tabs.addTab(QLabel("No training metrics found."), "Performance")
            return

        try:
            monitor = TrainingMonitorWidget()
            monitor.configure([
                {"key": "loss", "label": "Loss", "log_scale": True, "color": "blue"},
                {"key": "auc", "label": "AUC", "color": "green"},
                {"key": "precision", "label": "Precision", "color": "cyan"},
                {"key": "recall", "label": "Recall", "color": "magenta"}
            ])
            monitor.stop_button.hide()
            monitor.new_run_button.hide()

            df = pd.read_csv(metrics_path)
            history = {"epochs": list(range(1, len(df) + 1))}
            for col in df.columns:
                history[col] = df[col].tolist()

            history = self._sanitize_history(history)
            if not history:
                self.tabs.addTab(QLabel("Metrics file is empty or corrupted."), "Performance")
                return

            monitor.update_data({"type": "history_restore", "history": history})
            self.tabs.addTab(monitor, "Performance")
        except Exception as e:
            logging.error(f"Failed to load metrics visualization: {e}", exc_info=True)
            self.tabs.addTab(QLabel(f"Error loading metrics: {e}"), "Performance")

    def _sanitize_history(self, history: dict) -> dict:
        lengths = [len(v) for v in history.values() if isinstance(v, list) and len(v) > 0]
        if not lengths: return {}
        target_len = max(lengths)
        sanitized = {}
        if 'epoch' in history and isinstance(history['epoch'], list) and len(history['epoch']) == target_len:
            sanitized['epoch'] = history['epoch']
        else:
            sanitized['epoch'] = list(range(1, target_len + 1))
        for k, v in history.items():
            if k == 'epoch': continue 
            if isinstance(v, list):
                if len(v) >= target_len: sanitized[k] = v[:target_len]
                else: sanitized[k] = v + [None] * (target_len - len(v))
        return sanitized

    def _add_config_tab(self, model_dir: Path):
        config_path = model_dir / "model_config.json"
        editor = QTextEdit()
        editor.setReadOnly(True)
        editor.setStyleSheet("font-family: Consolas, Monaco, monospace;")
        if config_path.exists():
            try:
                raw_text = config_path.read_text()
                try:
                    data = json.loads(raw_text)
                    editor.setText(json.dumps(data, indent=4))
                except json.JSONDecodeError:
                    editor.setText(raw_text)
            except Exception as e:
                editor.setText(f"Error reading config: {e}")
        else:
            editor.setText("No configuration file found.")
        self.tabs.addTab(editor, "Configuration")

    def _create_models_summary_tab(self, workspace: Workspace, model_names: list) -> QWidget:
        container = QWidget()
        tab_layout = QVBoxLayout(container)
        table_widget = QTableWidget()
        table_widget.setSortingEnabled(True)
        table_widget.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        
        def _on_double_clicked(index):
            model_dir = workspace.behavior_models.base_dir / model_names[index.row()]
            if model_dir.exists(): self.show_workflow_file_info(workspace, model_dir)

        table_widget.doubleClicked.connect(_on_double_clicked)
        headers = ["Model Name", "Creation Date", "Type", "Epochs", "AUC", "Precision", "Recall"]
        table_widget.setColumnCount(len(headers))
        table_widget.setRowCount(len(model_names))
        table_widget.setHorizontalHeaderLabels(headers)

        table_widget.setSortingEnabled(False)
        for row, model_name in enumerate(model_names):
            table_widget.setItem(row, 0, NumericTableWidgetItem(model_name))
            for col in range(1, len(headers)):
                item = NumericTableWidgetItem("Loading...")
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                table_widget.setItem(row, col, item)
        table_widget.setSortingEnabled(True)
        tab_layout.addWidget(table_widget)

        self._summary_loader = BehaviorModelSummaryLoaderThread(workspace, model_names, self)
        self._summary_loader.data_ready.connect(lambda row, data: self._populate_table_row(table_widget, row, data))
        self._summary_loader.start()

        export_btn = QPushButton("Export to CSV")
        export_btn.clicked.connect(lambda: self._export_models_to_csv(model_names, table_widget))
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(export_btn)
        tab_layout.addLayout(btn_layout)
        return container

    def _populate_table_row(self, table_widget: QTableWidget, row_idx: int, row_data: list):
        is_sorting_enabled = table_widget.isSortingEnabled()
        table_widget.setSortingEnabled(False)
        for col, text in enumerate(row_data):
            item = NumericTableWidgetItem(text)
            if col in [1, 3, 4, 5, 6]:
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            table_widget.setItem(row_idx, col, item)
        table_widget.resizeColumnsToContents()
        table_widget.setSortingEnabled(is_sorting_enabled)

    def _create_models_evaluation_summary_tab(self, workspace: Workspace, model_names: list) -> QWidget:
        container = QWidget()
        tab_layout = QVBoxLayout(container)
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
        
        self._populate_evaluation_tree(workspace, tree, "dataset")

        btn_layout = QHBoxLayout()
        fill_metrics_btn = QPushButton("Fill Missing Metrics")
        fill_metrics_btn.clicked.connect(lambda: self._fill_missing_metrics(workspace, tree))
        export_btn = QPushButton("Export Summary")
        export_btn.clicked.connect(lambda: self._export_evaluations_to_csv(tree))
        btn_layout.addStretch()
        btn_layout.addWidget(fill_metrics_btn)
        btn_layout.addWidget(export_btn)
        tab_layout.addLayout(btn_layout)
        return container

    def _populate_evaluation_tree(self, workspace: Workspace, tree: QTreeWidget, group_by: str):
        tree.clear()
        # Behavior metrics are different from Pose
        headers = ["Accuracy", "F1 (Macro)", "Precision (Macro)", "Recall (Macro)"]
        if group_by == "dataset": tree.setHeaderLabels(["Dataset", "Model"] + headers)
        else: tree.setHeaderLabels(["Model", "Dataset"] + headers)

        eval_data = []
        base_pred_dir = workspace.behavior_predictions.base_dir
        if not base_pred_dir.exists(): return

        for model_name in os.listdir(base_pred_dir):
            model_pred_dir = base_pred_dir / model_name
            if not model_pred_dir.is_dir(): continue
            for dataset_name in os.listdir(model_pred_dir):
                dataset_pred_dir = model_pred_dir / dataset_name
                if not dataset_pred_dir.is_dir(): continue
                
                metrics_path = dataset_pred_dir / "evaluation_metrics.json"
                agg = {}
                if metrics_path.exists():
                    try:
                        agg = json.loads(metrics_path.read_text()).get("aggregate_metrics", {})
                    except Exception: pass
                
                def get_met(k, fmt):
                    val = agg.get(k)
                    return f"{val:{fmt}}" if isinstance(val, (int, float)) else "N/A"

                eval_data.append({
                    "model": model_name, "ds_name": dataset_name,
                    "acc": get_met('accuracy', '.3f'),
                    "f1": get_met('macro_f1', '.3f'),
                    "prec": get_met('macro_precision', '.3f'),
                    "rec": get_met('macro_recall', '.3f'),
                })

        nodes = {}
        for d in eval_data:
            row_data = [d["acc"], d["f1"], d["prec"], d["rec"]]
            if group_by == "dataset":
                if d["ds_name"] not in nodes:
                    nodes[d["ds_name"]] = NumericTreeWidgetItem(tree, [d["ds_name"]] + [""] * 10)
                    nodes[d["ds_name"]].setExpanded(True)
                child = NumericTreeWidgetItem(nodes[d["ds_name"]], ["", d["model"]] + row_data)
            else:
                if d["model"] not in nodes:
                    nodes[d["model"]] = NumericTreeWidgetItem(tree, [d["model"]] + [""] * 10)
                    nodes[d["model"]].setExpanded(True)
                child = NumericTreeWidgetItem(nodes[d["model"]], ["", d["ds_name"]] + row_data)

            for col in range(2, 6):
                child.setTextAlignment(col, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            child.setData(0, Qt.ItemDataRole.UserRole, {"dataset": d["ds_name"], "model": d["model"]})

    def _fill_missing_metrics(self, workspace: Workspace, tree: QTreeWidget):
        iterator = QTreeWidgetItemIterator(tree)
        while iterator.value():
            item = iterator.value()
            meta = item.data(0, Qt.ItemDataRole.UserRole)
            if meta and item.text(2) == "N/A":
                from whisker.services.behavior_classification.internal.workers.behavior_evaluation import BehaviorEvaluationJob
                worker = Worker(BehaviorEvaluationJob(
                    workspace=workspace, model_run_name=meta['model'],
                    dataset_name=meta['dataset'], iou_threshold=0.5, match_by_containment=True
                ))
                worker.signals.finished.connect(lambda res, it=item: self._receive_evaluation_results(it, res))
                JobManager.get().submit_worker("Behavior Evaluation", worker)
            iterator += 1

    def _receive_evaluation_results(self, item: QTreeWidgetItem, results: dict):
        agg = results.get("aggregate_metrics", {})
        def get_met(k, fmt):
            val = agg.get(k)
            return f"{val:{fmt}}" if isinstance(val, (int, float)) else "N/A"
        item.setText(2, get_met('accuracy', '.3f'))
        item.setText(3, get_met('macro_f1', '.3f'))
        item.setText(4, get_met('macro_precision', '.3f'))
        item.setText(5, get_met('macro_recall', '.3f'))

    def _export_evaluations_to_csv(self, tree: QTreeWidget):
        path, _ = QFileDialog.getSaveFileName(self, "Save Export", f"behavior_eval_summary_{datetime.now().strftime('%Y%m%d')}.csv", "CSV Files (*.csv)")
        if not path: return
        rows = []
        iterator = QTreeWidgetItemIterator(tree)
        while iterator.value():
            item = iterator.value()
            meta = item.data(0, Qt.ItemDataRole.UserRole)
            if meta:
                rows.append({
                    "Dataset": meta["dataset"], "Model": meta["model"],
                    "Accuracy": item.text(2), "F1 (Macro)": item.text(3),
                    "Precision (Macro)": item.text(4), "Recall (Macro)": item.text(5)
                })
            iterator += 1
        pd.DataFrame(rows).to_csv(path, index=False)

    def _export_models_to_csv(self, model_names: list, table_widget: QTableWidget):
        path, _ = QFileDialog.getSaveFileName(self, "Save Export", f"behavior_models_summary_{datetime.now().strftime('%Y%m%d')}.csv", "CSV Files (*.csv)")
        if not path: return
        try:
            headers = [table_widget.horizontalHeaderItem(i).text() for i in range(table_widget.columnCount())]
            data = [{headers[c]: table_widget.item(r, c).text() for c in range(table_widget.columnCount())} for r in range(table_widget.rowCount())]
            pd.DataFrame(data).to_csv(path, index=False)
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", str(e))
