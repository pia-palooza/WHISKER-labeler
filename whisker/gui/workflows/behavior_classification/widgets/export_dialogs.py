import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer, QSettings
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QComboBox,
    QHBoxLayout, QCheckBox, QPushButton, QDialog,
    QFormLayout, QGroupBox, QDoubleSpinBox, QListWidget, QListWidgetItem
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from whisker.services.behavior_classification.public.data_structures import BoutExtractionParams

class BoutExportDialog(QDialog):
    """Dialog for selecting bout extraction parameters before exporting to JSON."""
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Export Bouts Configuration")
        
        layout = QVBoxLayout(self)
        
        group = QGroupBox("Bout Extraction Parameters")
        form = QFormLayout(group)
        
        self.prob_thresh_spin = QDoubleSpinBox()
        self.prob_thresh_spin.setRange(0.0, 1.0)
        self.prob_thresh_spin.setSingleStep(0.05)
        self.prob_thresh_spin.setValue(0.5)
        form.addRow("Probability Threshold:", self.prob_thresh_spin)

        self.min_bout_spin = QDoubleSpinBox()
        self.min_bout_spin.setRange(0.0, 60.0)
        self.min_bout_spin.setSingleStep(0.1)
        self.min_bout_spin.setValue(0.5)
        self.min_bout_spin.setSuffix(" s")
        form.addRow("Min Bout Duration:", self.min_bout_spin)

        self.max_gap_spin = QDoubleSpinBox()
        self.max_gap_spin.setRange(0.0, 60.0)
        self.max_gap_spin.setSingleStep(0.1)
        self.max_gap_spin.setValue(0.2)
        self.max_gap_spin.setSuffix(" s")
        form.addRow("Max Gap to Fill:", self.max_gap_spin)
        
        layout.addWidget(group)
        
        btns_layout = QHBoxLayout()
        self.export_btn = QPushButton("Export")
        self.export_btn.clicked.connect(self.accept)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        btns_layout.addStretch()
        btns_layout.addWidget(self.export_btn)
        btns_layout.addWidget(self.cancel_btn)
        layout.addLayout(btns_layout)

    def get_params(self) -> BoutExtractionParams:
        return BoutExtractionParams(
            min_bout_duration_sec=self.min_bout_spin.value(),
            probability_threshold=self.prob_thresh_spin.value(),
            max_gap_fill_sec=self.max_gap_spin.value(),
        )

class ChartExportOptionsDialog(QDialog):
    """Dialog for dataset stack chart export options with live preview and video selection."""
    def __init__(self, behaviors: List[str], video_data: List[Dict[str, Any]], render_callback, model_name: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.all_video_data = video_data
        self.render_callback = render_callback
        self.model_name = model_name
        self.setWindowTitle("Chart Export Options")
        self.resize(1100, 750)
        
        main_layout = QHBoxLayout(self)
        
        # --- Left: Controls ---
        controls_panel = QWidget()
        controls_layout = QVBoxLayout(controls_panel)
        form = QFormLayout()
        
        self.behavior_combo = QComboBox()
        self.behavior_combo.addItems(behaviors)
        self.behavior_combo.currentTextChanged.connect(self._queue_preview_update)
        form.addRow("Behavior:", self.behavior_combo)
        
        self.normalize_check = QCheckBox("Normalize X-axis (0.0 - 1.0)")
        self.normalize_check.setChecked(True)
        self.normalize_check.toggled.connect(self._queue_preview_update)
        form.addRow(self.normalize_check)

        self.hide_gt_check = QCheckBox("Hide Ground Truth")
        self.hide_gt_check.toggled.connect(self._queue_preview_update)
        form.addRow(self.hide_gt_check)

        self.hide_preds_check = QCheckBox("Hide Predicted Bouts")
        self.hide_preds_check.toggled.connect(self._queue_preview_update)
        form.addRow(self.hide_preds_check)
        
        self.hide_probs_check = QCheckBox("Hide Raw Probabilities")
        self.hide_probs_check.toggled.connect(self._queue_preview_update)
        form.addRow(self.hide_probs_check)
        
        # --- Color Customization ---
        tab_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
                      'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        
        settings = QSettings("Whisker", "ChartExport")
        
        self.gt_color_combo = QComboBox()
        self.gt_color_combo.addItems(tab_colors)
        self.gt_color_combo.setCurrentText(settings.value("gt_color", 'tab:orange'))
        self.gt_color_combo.currentTextChanged.connect(self._queue_preview_update)
        form.addRow("GT Color:", self.gt_color_combo)
        
        self.pred_color_combo = QComboBox()
        self.pred_color_combo.addItems(tab_colors)
        self.pred_color_combo.setCurrentText(settings.value("pred_color", 'tab:green'))
        self.pred_color_combo.currentTextChanged.connect(self._queue_preview_update)
        form.addRow("Pred Color:", self.pred_color_combo)
        
        self.prob_cmap_combo = QComboBox()
        self._setup_cmap_combo(self.prob_cmap_combo)
        
        saved_cmap = settings.value("prob_cmap", 'viridis')
        idx = self.prob_cmap_combo.findData(saved_cmap)
        if idx >= 0: self.prob_cmap_combo.setCurrentIndex(idx)
        
        self.prob_cmap_combo.currentTextChanged.connect(self._queue_preview_update)
        form.addRow("Prob Colormap:", self.prob_cmap_combo)

        controls_layout.addLayout(form)
        
        self.toggle_videos_btn = QPushButton("Select Videos >>")
        self.toggle_videos_btn.setCheckable(True)
        self.toggle_videos_btn.toggled.connect(self._on_toggle_videos)
        controls_layout.addWidget(self.toggle_videos_btn)

        controls_layout.addStretch()
        
        btns = QHBoxLayout()
        self.export_btn = QPushButton("Save Image...")
        self.export_btn.clicked.connect(self.accept)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        btns.addWidget(self.export_btn)
        btns.addWidget(self.cancel_btn)
        controls_layout.addLayout(btns)
        
        main_layout.addWidget(controls_panel, 0)

        # --- Center: Preview ---
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        self.loading_label = QLabel("Loading Preview...")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_layout.addWidget(self.loading_label)
        
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.hide()
        preview_layout.addWidget(self.canvas)
        
        main_layout.addWidget(preview_group, 1)

        # --- Right: Video Selection Panel ---
        self.video_panel = QGroupBox("Filter Videos")
        self.video_panel.setFixedWidth(250)
        v_panel_layout = QVBoxLayout(self.video_panel)
        
        sel_btns = QHBoxLayout()
        all_btn = QPushButton("All")
        none_btn = QPushButton("None")
        all_btn.clicked.connect(self._select_all_videos)
        none_btn.clicked.connect(self._select_no_videos)
        sel_btns.addWidget(all_btn)
        sel_btns.addWidget(none_btn)
        v_panel_layout.addLayout(sel_btns)

        filter_btns = QHBoxLayout()
        gt_only_btn = QPushButton("GT Only")
        pred_only_btn = QPushButton("Pred Only")
        both_btn = QPushButton("GT+Pred Only")
        gt_only_btn.clicked.connect(self._select_gt_only_videos)
        pred_only_btn.clicked.connect(self._select_pred_only_videos)
        both_btn.clicked.connect(self._select_gt_and_pred_videos)
        filter_btns.addWidget(gt_only_btn)
        filter_btns.addWidget(pred_only_btn)
        filter_btns.addWidget(both_btn)
        v_panel_layout.addLayout(filter_btns)

        self.video_list = QListWidget()
        for v in self.all_video_data:
            item = QListWidgetItem(v["stem"])
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            
            has_gt = len(v.get("gt_bouts", [])) > 0
            has_pred = len(v.get("pred_bouts", [])) > 0 or len(v.get("probs", {})) > 0
            
            if has_gt and has_pred:
                item.setCheckState(Qt.CheckState.Checked)
            else:
                item.setCheckState(Qt.CheckState.Unchecked)
                
            self.video_list.addItem(item)
        self.video_list.itemChanged.connect(self._queue_preview_update)
        v_panel_layout.addWidget(self.video_list)
        
        main_layout.addWidget(self.video_panel)
        self.video_panel.hide()

        # --- Debounce Timer for Updates ---
        self.preview_timer = QTimer()
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self._update_preview)
        
        # Initial trigger
        self._queue_preview_update()

    def accept(self):
        settings = QSettings("Whisker", "ChartExport")
        settings.setValue("gt_color", self.gt_color_combo.currentText())
        settings.setValue("pred_color", self.pred_color_combo.currentText())
        settings.setValue("prob_cmap", self.prob_cmap_combo.currentData())
        super().accept()

    def _setup_cmap_combo(self, combo: QComboBox):
        groups = {
            "Perceptually Uniform Sequential": ['viridis', 'plasma', 'inferno', 'magma', 'cividis'],
            "Sequential": ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                           'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                           'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'],
            "Sequential (2)": ['binary', 'gist_yarg', 'gist_gray', 'gray', 'bone',
                               'pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
                               'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper'],
            "Diverging": ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
                          'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
                          'berlin', 'managua', 'vanimo']
        }
        
        for group_name, cmaps in groups.items():
            combo.addItem(f"--- {group_name} ---")
            # Disable the header item
            combo.model().setData(combo.model().index(combo.count()-1, 0), 0, Qt.ItemDataRole.UserRole - 1)
            for cmap in cmaps:
                combo.addItem(f"  {cmap}", cmap)
        
        combo.setCurrentIndex(1) # Select 'viridis' (first after first header)

    def _on_toggle_videos(self, checked: bool):
        self.video_panel.setVisible(checked)
        self.toggle_videos_btn.setText("Select Videos <<" if checked else "Select Videos >>")

    def _select_all_videos(self):
        self.video_list.blockSignals(True)
        for i in range(self.video_list.count()):
            self.video_list.item(i).setCheckState(Qt.CheckState.Checked)
        self.video_list.blockSignals(False)
        self._queue_preview_update()

    def _select_no_videos(self):
        self.video_list.blockSignals(True)
        for i in range(self.video_list.count()):
            self.video_list.item(i).setCheckState(Qt.CheckState.Unchecked)
        self.video_list.blockSignals(False)
        self._queue_preview_update()

    def _select_gt_only_videos(self):
        self.video_list.blockSignals(True)
        for i in range(self.video_list.count()):
            v = self.all_video_data[i]
            has_gt = len(v.get("gt_bouts", [])) > 0
            state = Qt.CheckState.Checked if has_gt else Qt.CheckState.Unchecked
            self.video_list.item(i).setCheckState(state)
        self.video_list.blockSignals(False)
        self._queue_preview_update()

    def _select_pred_only_videos(self):
        self.video_list.blockSignals(True)
        for i in range(self.video_list.count()):
            v = self.all_video_data[i]
            has_pred = len(v.get("pred_bouts", [])) > 0 or len(v.get("probs", {})) > 0
            state = Qt.CheckState.Checked if has_pred else Qt.CheckState.Unchecked
            self.video_list.item(i).setCheckState(state)
        self.video_list.blockSignals(False)
        self._queue_preview_update()

    def _select_gt_and_pred_videos(self):
        self.video_list.blockSignals(True)
        for i in range(self.video_list.count()):
            v = self.all_video_data[i]
            has_gt = len(v.get("gt_bouts", [])) > 0
            has_pred = len(v.get("pred_bouts", [])) > 0 or len(v.get("probs", {})) > 0
            state = Qt.CheckState.Checked if (has_gt and has_pred) else Qt.CheckState.Unchecked
            self.video_list.item(i).setCheckState(state)
        self.video_list.blockSignals(False)
        self._queue_preview_update()

    def _queue_preview_update(self):
        self.loading_label.show()
        self.canvas.hide()
        self.preview_timer.start(300) # 300ms debounce

    def _update_preview(self):
        opts = self.get_options()
        try:
            self.render_callback(
                opts, self.model_name, self.all_video_data, fig=self.figure
            )
            self.canvas.draw()
            self.loading_label.hide()
            self.canvas.show()
        except Exception as e:
            self.loading_label.setText(f"Preview Error: {e}")
            logging.error(f"Preview failed: {e}")

    def get_options(self) -> Dict[str, Any]:
        selected_videos = []
        for i in range(self.video_list.count()):
            item = self.video_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                selected_videos.append(item.text())
        return {
            "behavior": self.behavior_combo.currentText(),
            "normalize": self.normalize_check.isChecked(),
            "hide_gt": self.hide_gt_check.isChecked(),
            "hide_preds": self.hide_preds_check.isChecked(),
            "hide_probs": self.hide_probs_check.isChecked(),
            "selected_videos": selected_videos,
            "gt_color": self.gt_color_combo.currentText(),
            "pred_color": self.pred_color_combo.currentText(),
            "prob_cmap": self.prob_cmap_combo.currentData()
        }
