import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union

from PyQt6.QtCore import Qt, pyqtSignal, QPointF, QRectF
from PyQt6.QtGui import QColor, QPen, QBrush
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QGroupBox, 
    QLabel, QComboBox, QPushButton
)

import pyqtgraph as pg

from whisker.gui.tabs.base_tab import BaseTab
from whisker.gui.widgets import data_explorer
from whisker.gui.widgets.media_viewer import MediaViewerWidget, OverlayPainter
from whisker.services.behavior_classification.internal.core.ml.features import (
    extract_multi_animal_features,
    BehaviorFeatureExtractionParams
)
from whisker.gui.signals import MessageBus

class DiagnosticOverlayPainter(OverlayPainter):
    """
    Advanced painter that supports Ghost Bones, Uncertainty Ellipses, 
    and Confidence Text labels.
    """
    def __init__(self, scene):
        super().__init__(scene)
        self._debug_meta = pd.DataFrame()
        self._raw_features = pd.DataFrame() 
        self._variances = {} # (frame, model_id, part) -> sigma^2
        self._show_ellipses = True
        self._show_ghost_bones = True
        self._show_conf_labels = True
        self._current_frame = 0

    def set_debug_data(self, debug_meta: pd.DataFrame, raw_features: pd.DataFrame, variances: Dict = None):
        self._debug_meta = debug_meta
        self._raw_features = raw_features
        if variances:
            self._variances = variances

    def set_current_frame(self, frame_idx: Union[int, str]):
        self._current_frame = frame_idx

    def _draw_points(self, data: List[Dict], style: str, offset_x: float = 0.0):
        # Draw base skeleton
        super()._draw_points(data, style, offset_x)
        
        # Draw diagnostic overlays
        if offset_x == 0.0:
            self._draw_diagnostics(data)

    def _draw_diagnostics(self, data: List[Dict]):
        if self._debug_meta.empty or self._current_frame not in self._debug_meta.index:
            return

        frame_debug = self._debug_meta.loc[self._current_frame]
        frame_raw = self._raw_features.loc[self._current_frame]
        
        for identity in data:
            model_id = identity.get("identity_id")
            points = identity.get("points", {})
            
            for name, pt in points.items():
                # 1. Confidence Label
                if self._show_conf_labels:
                    col_c = f"{model_id}_c_{name}"
                    if col_c in frame_raw:
                        c_smooth = frame_raw[col_c]
                        txt = pg.QtWidgets.QGraphicsSimpleTextItem(f"{name} c:{c_smooth:.2f}")
                        txt.setPos(pt.x() + 10, pt.y() - 15)
                        txt.setBrush(QBrush(QColor("yellow")))
                        txt.setScale(0.8)
                        self._pose_group.addToGroup(txt)

                # 2. Ghost Bones (Kinematic jumps)
                if self._show_ghost_bones:
                    col_rvx = f"{model_id}_rvx_{name}"
                    col_vx = f"{model_id}_vx_{name}"
                    if col_rvx in frame_debug and col_vx in frame_raw:
                        rv = np.sqrt(frame_debug[col_rvx]**2 + frame_debug[f"{model_id}_rvy_{name}"]**2)
                        v = np.sqrt(frame_raw[col_vx]**2 + frame_raw[f"{model_id}_vy_{name}"]**2)
                        
                        if rv > 20 and v == 0:
                            # Jump detected!
                            raw_pos_x = pt.x() + frame_debug[col_rvx]
                            raw_pos_y = pt.y() + frame_debug[f"{model_id}_rvy_{name}"]
                            
                            line = pg.QtWidgets.QGraphicsLineItem(pt.x(), pt.y(), raw_pos_x, raw_pos_y)
                            line.setPen(QPen(QColor("red"), 2, Qt.PenStyle.DashLine))
                            self._pose_group.addToGroup(line)
                            
                            dot = pg.QtWidgets.QGraphicsEllipseItem(raw_pos_x - 3, raw_pos_y - 3, 6, 6)
                            dot.setBrush(QBrush(QColor("red")))
                            self._pose_group.addToGroup(dot)

                # 3. Uncertainty Ellipse
                if self._show_ellipses:
                    var = self._variances.get((self._current_frame, model_id, name), 0.0)
                    if var > 0:
                        r = np.sqrt(var) * 100.0 # Scaling for visibility
                        ell = pg.QtWidgets.QGraphicsEllipseItem(pt.x() - r, pt.y() - r, r*2, r*2)
                        c = QColor("white")
                        c.setAlpha(80)
                        ell.setPen(QPen(c, 1))
                        self._pose_group.addToGroup(ell)

class PipelineDebugTab(BaseTab):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._connect_signals()
        
        self._current_pose_ds = None
        self._current_run_name = None
        self._current_selection = None
        self._video_stem = None
        self._debug_data = pd.DataFrame()
        self._raw_data = pd.DataFrame()
        self._variances = {} # (frame, model_id, part) -> sigma^2

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        
        # --- Top: Media Controls ---
        top_bar = QHBoxLayout()
        self.lbl_status = QLabel("Select a model run and dataset to begin.")
        self.btn_run_diag = QPushButton("Run Diagnostic Inference")
        self.btn_run_diag.clicked.connect(self._run_diagnostic)
        
        top_bar.addWidget(self.lbl_status, 1)
        top_bar.addWidget(self.btn_run_diag)
        main_layout.addLayout(top_bar)

        # --- Middle: Splitter for Viewer and Plots ---
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        
        # 1. Frame Viewer
        self.viewer = MediaViewerWidget(self)
        self.viewer.painter = DiagnosticOverlayPainter(self.viewer.scene)
        self.splitter.addWidget(self.viewer)
        
        # 2. Plot Dashboard
        plot_container = QWidget()
        plot_layout = QVBoxLayout(plot_container)
        
        self.velocity_plot = pg.PlotWidget(title="Kinematic Filter: Raw vs Filtered Velocity")
        self.velocity_plot.addLegend()
        self.velocity_plot.setLabel('left', 'Velocity (px/frame)')
        self.velocity_plot.setLabel('bottom', 'Frame Index')
        
        self.confidence_plot = pg.PlotWidget(title="Confidence Smoothing: Raw vs Smoothed")
        self.confidence_plot.addLegend()
        self.confidence_plot.setLabel('left', 'Confidence')
        self.confidence_plot.setLabel('bottom', 'Frame Index')
        
        plot_layout.addWidget(self.velocity_plot)
        plot_layout.addWidget(self.confidence_plot)
        
        self.splitter.addWidget(plot_container)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        
        main_layout.addWidget(self.splitter)

        # Vertical Cursor
        self.v_line_vel = pg.InfiniteLine(angle=90, movable=False, pen='y')
        self.v_line_conf = pg.InfiniteLine(angle=90, movable=False, pen='y')
        self.velocity_plot.addItem(self.v_line_vel)
        self.confidence_plot.addItem(self.v_line_conf)

    def _connect_signals(self):
        self.viewer.frame_changed.connect(self._on_frame_changed)
        MessageBus.get().subscribe("selection/model_run/changed", self._on_model_run_changed)

    def _on_model_run_changed(self, topic: str, payload: Dict):
        self._current_run_name = payload.get("name")
        self._update_status_label()

    def on_data_explorer_item_selected(self, selection: data_explorer.Selection):
        self._current_selection = selection
        self._update_status_label()

    def _update_status_label(self):
        run = self._current_run_name or "None"
        ds = self._current_selection.item[1] if self._current_selection else "None"
        self.lbl_status.setText(f"Model Run: <b>{run}</b> | Dataset: <b>{ds}</b>")

    def _run_diagnostic(self):
        if not self._workspace or not self._project or not self._current_selection or not self._current_run_name:
            return
        
        dataset_name = self._current_selection.item[1]
        video_relative_path = self._current_selection.item[-1] if self._current_selection.type == data_explorer.ItemTypeEnum.DATASET_VIDEO else None
        
        video_stem = None
        if video_relative_path:
             video_stem = Path(video_relative_path).stem
        
        self._video_stem = video_stem

        try:
            pose_ds = self._workspace.get_pose_predictions(
                self._current_run_name, dataset_name, video_stem=video_stem, raise_if_missing=True
            )
        except Exception as e:
            logging.error(f"Failed to load predictions for debug: {e}")
            return

        self._current_pose_ds = pose_ds

        # 1. Behavior Features
        params = BehaviorFeatureExtractionParams(
            project_bodyparts=self._project.body_parts,
            model_identities=self._project.identities,
            root_bodypart=self._project.body_parts[0],
            max_vel=50.0,
            confidence_smoothing_window=5
        )
        id_map = {i: i for i in self._project.identities}
        
        logging.info("Running diagnostic feature extraction on predictions...")
        features, debug_meta = extract_multi_animal_features(
            pose_ds, id_map, params, return_debug=True
        )
        self._debug_data = debug_meta
        self._raw_data = features
        
        # 2. Mock Variances for Ellipses (derived from confidence)
        self._variances = {}
        for (f, ind, bp), row in pose_ds.keypoint_data.iterrows():
            c = row['c']
            var = (1.0 / (c + 1e-6) - 1.0) / 1000.0
            self._variances[(f, ind, bp)] = var

        self.viewer.painter.set_debug_data(self._debug_data, self._raw_data, self._variances)

        dataset = self._workspace.datasets.get(dataset_name)
        if self._current_selection.type == data_explorer.ItemTypeEnum.DATASET_VIDEO:
            video_path = Path(dataset.base_data_path) / video_relative_path
            self.viewer.set_media(video_path)
        elif self._current_selection.type == data_explorer.ItemTypeEnum.DATASET_IMAGE:
             image_rel_path = self._current_selection.item[2]
             image_path = Path(dataset.base_data_path) / image_rel_path
             self.viewer.set_media(image_path)
        elif self._current_selection.type == data_explorer.ItemTypeEnum.DATASET_VIDEO_FRAME:
             image_key = self._current_selection.item[3]
             image_path = Path(dataset.base_data_path) / image_key
             self.viewer.set_media(image_path)
        
        self._update_plots()
        # Trigger initial frame render
        self._on_frame_changed(0)

    def _update_plots(self):
        self.velocity_plot.clear()
        self.confidence_plot.clear()
        self.velocity_plot.addItem(self.v_line_vel)
        self.confidence_plot.addItem(self.v_line_conf)
        
        if self._debug_data.empty: return

        ind0 = self._project.identities[0]
        # Find the first body part that isn't the root
        plot_bp = self._project.body_parts[0]
        root_bp = self._project.body_parts[0] # Usually the first one
        
        for bp in self._project.body_parts:
            if bp != root_bp:
                plot_bp = bp
                break
        
        logging.info(f"Plotting diagnostic velocity for {ind0} - {plot_bp} (Non-root: {plot_bp != root_bp})")

        col_rvx = f"{ind0}_rvx_{plot_bp}"
        col_rvy = f"{ind0}_rvy_{plot_bp}"
        col_vx = f"{ind0}_vx_{plot_bp}"
        col_vy = f"{ind0}_vy_{plot_bp}"
        
        if col_rvx in self._debug_data.columns:
            v_raw = np.sqrt(self._debug_data[col_rvx]**2 + self._debug_data[col_rvy]**2)
            # Use absolute filtered velocity (avx, avy) for true comparison
            col_avx = f"{ind0}_avx_{plot_bp}"
            col_avy = f"{ind0}_avy_{plot_bp}"
            
            if col_avx in self._debug_data.columns:
                v_filt = np.sqrt(self._debug_data[col_avx]**2 + self._debug_data[col_avy]**2)
            else:
                # Fallback to egocentric if avx not available yet
                v_filt = np.sqrt(self._raw_data[col_vx]**2 + self._raw_data[col_vy]**2)
                
            self.velocity_plot.plot(v_raw.values, pen=pg.mkPen('r', style=Qt.PenStyle.DashLine), name="Raw Velocity (World)")
            self.velocity_plot.plot(v_filt.values, pen=pg.mkPen('g', width=2), name="Filtered Velocity (World)")

        col_rc = f"raw_{ind0}_c_{plot_bp}"
        col_c = f"{ind0}_c_{plot_bp}"
        
        if col_rc in self._debug_data.columns:
            c_raw = self._debug_data[col_rc]
            c_smooth = self._raw_data[col_c]
            self.confidence_plot.plot(c_raw.values, pen=pg.mkPen('b', style=Qt.PenStyle.DotLine), name="Raw Confidence")
            self.confidence_plot.plot(c_smooth.values, pen=pg.mkPen('c', width=2), name="Smoothed Confidence")

    def _on_frame_changed(self, frame_idx: int):
        self.v_line_vel.setPos(frame_idx)
        self.v_line_conf.setPos(frame_idx)
        
        if self._current_pose_ds is not None:
            indices = self._current_pose_ds.frame_indices
            frame_key = None
            
            if self._video_stem:
                frame_key = f"{self._video_stem}/frame_{frame_idx:06d}"
            elif frame_idx < len(indices):
                frame_key = indices[frame_idx]
            
            if not frame_key:
                return

            if frame_idx % 100 == 0: # Avoid log spam
                logging.debug(f"DebugTab seeking frame_key: {frame_key} (Indices count: {len(indices)})")

            self.viewer.painter.set_current_frame(frame_key)

            try:
                # Using .xs or .loc[frame_key] on MultiIndex
                frame_data = self._current_pose_ds.keypoint_data.loc[frame_key]
                
                pose_list = []
                for ind in self._project.identities:
                    if ind in frame_data.index:
                        ind_pts = frame_data.loc[ind]
                        points_dict = {}
                        conf_dict = {}
                        for bp in self._project.body_parts:
                            if bp in ind_pts.index:
                                row = ind_pts.loc[bp]
                                col_c = f"{ind}_c_{bp}"
                                
                                if not self._raw_data.empty and frame_key in self._raw_data.index:
                                    c = self._raw_data.loc[frame_key, col_c]
                                else:
                                    c = row['c']
                                    
                                points_dict[bp] = QPointF(row['x'], row['y'])
                                conf_dict[bp] = c
                        
                        pose_list.append({
                            "identity_id": ind,
                            "points": points_dict,
                            "confidences": conf_dict,
                            "color": QColor("cyan")
                        })
                
                if frame_idx % 100 == 0:
                    logging.debug(f"Pushing {len(pose_list)} poses to viewer for frame {frame_key}")
                    
                self.viewer.set_pose_data(pose_list, [])
            except KeyError:
                if frame_idx % 100 == 0:
                    logging.warning(f"KeyError: Frame {frame_key} not found in PoseDataset. Example index: {indices[0] if indices else 'None'}")
                pass
