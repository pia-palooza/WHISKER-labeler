"""Per-dataset summary for the ROI Tracking workflow: for the selected prediction run, each video's worst-arena
detection percentage and saved correction count, videos with the most gaps listed first (brief section 5.5 #7)."""
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView, QComboBox, QHBoxLayout, QHeaderView, QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget,
)

from whisker.core.workspace import Workspace
from whisker.services.roi_tracking.internal.verify_tracks import arena_tracks, detected_fraction

from .roi_verify_widget import arenas_for_video


class RoiSummaryWidget(QWidget):
    video_selected = pyqtSignal(Path)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._workspace: Optional[Workspace] = None
        self._dataset_name: Optional[str] = None

        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        header.addWidget(QLabel("Run:"))
        self.run_combo = QComboBox()
        self.run_combo.currentTextChanged.connect(self._refresh)
        header.addWidget(self.run_combo)
        header.addStretch()
        layout.addLayout(header)

        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Video", "Worst arena detected", "Corrections"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.doubleClicked.connect(self._on_row_double_clicked)
        layout.addWidget(self.table, 1)

    def set_workspace(self, workspace: Optional[Workspace]):
        self._workspace = workspace

    def set_dataset(self, dataset_name: str):
        self._dataset_name = dataset_name
        runs = self._workspace.list_roi_runs(dataset_name) if self._workspace else []
        self.run_combo.blockSignals(True)
        self.run_combo.clear()
        self.run_combo.addItems(runs)
        self.run_combo.blockSignals(False)
        self._refresh()

    def _refresh(self, *_args):
        self.table.setRowCount(0)
        if not (self._workspace and self._dataset_name):
            self.summary_label.setText("")
            return
        dataset = self._workspace.datasets.get(self._dataset_name)
        if not dataset:
            return
        run_name = self.run_combo.currentText()

        rows = []  # (video_name, worst_arena_pct_or_None, n_corrections)
        for video_file in dataset.files:
            video_name = Path(video_file).name
            stem = Path(video_file).stem
            worst_pct, n_corrections = None, 0
            prediction = self._workspace.load_roi_prediction(run_name, self._dataset_name, stem) if run_name else None
            if prediction:
                arenas = arenas_for_video(self._workspace, self._dataset_name, video_name)
                original_tracks = arena_tracks(prediction, arenas)
                if original_tracks:
                    edits = self._workspace.load_roi_corrections(run_name, self._dataset_name, stem)
                    n_corrections = len(edits)
                    worst_pct = min(100.0 * detected_fraction(edits.apply_to(t["xy"], k))
                                     for k, t in enumerate(original_tracks))
            rows.append((video_name, worst_pct, n_corrections))

        # Videos with the worst (lowest) detection fraction first; videos with no run/prediction yet last.
        rows.sort(key=lambda r: r[1] if r[1] is not None else 101.0)

        self.table.setRowCount(len(rows))
        for i, (video_name, worst_pct, n_corrections) in enumerate(rows):
            self.table.setItem(i, 0, QTableWidgetItem(video_name))
            pct_item = QTableWidgetItem(f"{worst_pct:.0f}%" if worst_pct is not None else "no prediction")
            pct_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(i, 1, pct_item)
            corr_item = QTableWidgetItem(str(n_corrections))
            corr_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(i, 2, corr_item)

        total = len(rows)
        with_gaps = sum(1 for _, pct, _ in rows if pct is not None and pct < 100.0)
        run_text = f"run '{run_name}'" if run_name else "no run selected"
        self.summary_label.setText(
            f"<b>{self._dataset_name}</b> ({total} video(s), {run_text})<br>"
            f"{with_gaps} video(s) have at least one frame without a detection in some arena.")

    def _on_row_double_clicked(self, index):
        if not (self._workspace and self._dataset_name):
            return
        dataset = self._workspace.datasets.get(self._dataset_name)
        if not dataset:
            return
        video_name = self.table.item(index.row(), 0).text()
        self.video_selected.emit(Path(dataset.base_data_path) / video_name)
