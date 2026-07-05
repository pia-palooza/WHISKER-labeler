import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import logging
import re

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from whisker.base.job import BaseJob
from whisker.services.pose_estimation.public.data_structures import PoseDataset
from whisker.services.behavior_classification.public.data_structures import BehaviorDataset
from whisker.services.behavior_classification.internal.core.ml.dataprep import create_frame_wise_labels
from whisker.core.constants import KEYPOINT_COLORS

class BehaviorExportJob(BaseJob):
    def __init__(
        self,
        source_video_path: Path,
        output_video_path: Path,
        start_frame: int,
        end_frame: int,
        # Data
        pred_behavior_ds: Optional[BehaviorDataset] = None,
        gt_behavior_ds: Optional[BehaviorDataset] = None,
        pred_pose_ds: Optional[PoseDataset] = None,
        # Context
        project_skeleton: List[Tuple[str, str]] = None,
        project_bodyparts: List[str] = None,
        project_identities: List[str] = None,
        # Config
        render_config: Dict = None,
        progress_callback=None
    ):
        super().__init__(progress_callback)
        self.src_path = str(source_video_path)
        self.out_path = str(output_video_path)
        self.start_f = start_frame
        self.end_f = end_frame
        
        self.pred_beh = pred_behavior_ds
        self.gt_beh = gt_behavior_ds
        self.pred_pose = pred_pose_ds
        
        self.skeleton = project_skeleton or []
        self.identities = project_identities or []
        self.cfg = render_config or {}
        
        # Cache for Pose Colors
        self.colors = {}
        n_colors = len(KEYPOINT_COLORS)
        for i, ident in enumerate(self.identities):
            hex_val = KEYPOINT_COLORS[i % n_colors].lstrip('#')
            # Slice hex to (R, G, B) then reverse for BGR
            rgb = tuple(int(hex_val[j:j+2], 16) for j in (0, 2, 4))
            self.colors[ident] = rgb[::-1]

        # --- 1. Prepare Behavior Data (Predictions) ---
        self.probs_df = None
        if self.pred_beh and not self.pred_beh.per_frame_probabilities.empty:
            df = self.pred_beh.per_frame_probabilities.copy()
            if not pd.api.types.is_numeric_dtype(df.index):
                try:
                    def extract_frame_num(idx_val):
                        s = str(idx_val)
                        matches = re.findall(r'\d+', s)
                        return int(matches[-1]) if matches else -1
                    new_index = df.index.map(extract_frame_num)
                    df.index = new_index
                except Exception: pass
            
            df = df[df.index >= 0].sort_index()
            df = df[~df.index.duplicated(keep='first')]
            self.probs_df = df

        # --- 2. Prepare GT Data (Ground Truth) ---
        self.gt_binary_df = None
        self.active_behaviors = self.cfg.get('selected_behaviors', [])
        
        # If no explicit selection, default to all columns in pred or GT
        if not self.active_behaviors:
            if self.probs_df is not None:
                self.active_behaviors = list(self.probs_df.columns)
            elif self.gt_beh and not self.gt_beh.bouts.empty:
                self.active_behaviors = list(self.gt_beh.bouts['behavior'].unique())

        if self.gt_beh and not self.gt_beh.bouts.empty:
            # We need to know the total video length to rasterize GT. 
            # We'll estimate it from the end_frame requested + margin, or max bout.
            max_bout_f = self.gt_beh.bouts['end_frame'].max()
            total_f = max(end_frame + 100, int(max_bout_f) + 100)
            
            # Robust Key Match (borrowed from existing logic)
            video_stem = Path(self.src_path).stem
            video_name = Path(self.src_path).name
            
            keys = self.gt_beh.bouts["video_key"].unique()
            target_key = None
            if video_name in keys: target_key = video_name
            elif video_stem in keys: target_key = video_stem
            
            if target_key:
                # Use shared helper to get a (Frame x Behavior) binary matrix
                self.gt_binary_df = create_frame_wise_labels(
                    self.gt_beh, target_key, self.active_behaviors, total_f
                )

        # --- 3. Prepare Pose Data Keys ---
        self.pose_keys = set()
        if self.pred_pose and not self.pred_pose.keypoint_data.empty:
            self.pose_keys = set(self.pred_pose.keypoint_data.index.get_level_values('frame_index').unique())

        self.threshold = self.cfg.get('prob_threshold', 0.5)

        # --- 4. Plotting Setup (Subplots) ---
        # Calculate height: Base + (Height per behavior)
        self.n_plots = len(self.active_behaviors)
        
        # If no behaviors to plot, we make a small dummy figure
        fig_h_inches = max(1.0, 0.8 * self.n_plots) 
        
        self.fig = Figure(figsize=(8, fig_h_inches), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        
        if self.n_plots > 0:
            self.axs = self.fig.subplots(nrows=self.n_plots, ncols=1, sharex=True, squeeze=False).flatten()
            self.fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15, hspace=0.1)
        else:
            self.axs = []

    def run(self) -> str:
        self.report_progress("Initializing Export...", 0)
        
        cap = cv2.VideoCapture(self.src_path)
        if not cap.isOpened():
            raise IOError(f"Could not open source: {self.src_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        
        # Calculate layout dimensions
        # Video Frame Height
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Plot Frame Height: We scale the matplotlib buffer to be exactly Width `w`
        # The height depends on the Aspect Ratio of the Figure we created.
        # Figure aspect = 8 / fig_h_inches.
        # Desired width = w.
        # Desired height = w * (fig_h_inches / 8).
        fig_w_inches, fig_h_inches = self.fig.get_size_inches()
        plot_h = int(w * (fig_h_inches / fig_w_inches))
        
        out_h = vid_h + plot_h
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.out_path, fourcc, fps, (w, out_h))
        
        total_frames_to_process = self.end_f - self.start_f + 1
        
        # Zoom Radius (+/- 5 seconds)
        radius_frames = int(fps * 5)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_f)
        current_f = self.start_f
        processed = 0
        
        video_stem = Path(self.src_path).stem
        video_name = Path(self.src_path).name
        
        while current_f <= self.end_f:
            if self.is_cancelled: break
            
            ret, frame = cap.read()
            if not ret: break
            
            # 1. Poses
            if self.pred_pose:
                candidates = [
                    f"{video_stem}/frame_{current_f:06d}",
                    f"frame_{current_f:06d}",
                    f"{video_name}/frame_{current_f:06d}"
                ]
                key = next((k for k in candidates if k in self.pose_keys), None)
                if key:
                    self._draw_pose(frame, self.pred_pose.keypoint_data, key)

            # 2. Probability Plot (Subplots)
            plot_img = self._render_plot_frame(current_f, w, plot_h, radius_frames, fps)
            
            # 3. Stack
            combined = np.vstack([frame, plot_img])
            
            # 4. Text
            self._draw_active_text(combined, current_f)

            out.write(combined)
            processed += 1
            
            if processed % 10 == 0:
                pct = int((processed / total_frames_to_process) * 100)
                self.report_progress(f"Exporting frame {current_f}...", pct)
            
            current_f += 1

        cap.release()
        out.release()
        self.fig.clear()
        
        return f"Exported {processed} frames to {Path(self.out_path).name}"

    def _render_plot_frame(self, current_f, target_w, target_h, radius, fps):
        # Reset Axes
        for ax in self.axs:
            ax.clear()
            ax.set_facecolor('#f9f9f9')
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.set_yticks([0, 1])
            ax.set_ylim(-0.05, 1.05)
            # Draw cursor
            ax.axvline(x=current_f, color='red', linestyle='-', linewidth=1.5, alpha=0.8)

        # Window
        start = max(0, current_f - radius)
        end = current_f + radius
        
        # Prepare Data Slice
        # We assume index alignment is close enough or use explicit index logic
        # Probs
        p_sub = None
        if self.probs_df is not None:
            try: p_sub = self.probs_df.loc[start:end]
            except: pass
            
        # GT
        g_sub = None
        if self.gt_binary_df is not None:
            try: g_sub = self.gt_binary_df.loc[start:end]
            except: pass

        # Plot Loop
        for i, beh in enumerate(self.active_behaviors):
            ax = self.axs[i]
            
            # 1. Ground Truth (Shaded)
            if g_sub is not None and beh in g_sub.columns:
                # fill_between is standard for binary GT visualization
                ax.fill_between(
                    g_sub.index, 0, g_sub[beh], 
                    color="tab:orange", alpha=0.3, step='mid'
                )

            # 2. Probability (Line)
            if p_sub is not None and beh in p_sub.columns:
                ax.plot(p_sub.index, p_sub[beh], color="black", linewidth=1.5)

            # Formatting
            ax.set_xlim(start, end)
            ax.set_ylabel(beh, rotation=0, ha='right', fontsize=8, va='center')
            
            # Only bottom axis gets labels
            if i < self.n_plots - 1:
                ax.set_xticklabels([])
            else:
                def time_fmt(x, pos):
                    m, s = divmod(int(x / fps) if fps else 0, 60)
                    return f"{m:02d}:{s:02d}"
                ax.xaxis.set_major_formatter(FuncFormatter(time_fmt))

        self.canvas.draw()
        
        img_rgba = np.asarray(self.canvas.buffer_rgba())
        img_resized = cv2.resize(img_rgba, (target_w, target_h))
        img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGBA2BGR)
        return img_bgr

    def _draw_pose(self, img, df: pd.DataFrame, frame_key: str):
        # (Same as before)
        try:
            frame_data = df.loc[frame_key]
        except KeyError: return

        for ident in self.identities:
            if ident not in frame_data.index: continue
            try:
                ind_data = frame_data.loc[ident]
                # Handle Series (1 row) vs DataFrame
                if isinstance(ind_data, pd.Series):
                    valid_rows = pd.DataFrame([ind_data])
                else:
                    valid_rows = ind_data[ind_data['c'] > 0.3]
                
                kps = {}
                for row_idx, row in valid_rows.iterrows():
                    bp_name = row.name if isinstance(row_idx, str) else row_idx
                    x, y = int(row['x']), int(row['y'])
                    kps[bp_name] = (x, y)
                    color = self.colors.get(ident, (255, 255, 255))
                    cv2.circle(img, (x, y), 4, color, -1)

                for bp1, bp2 in self.skeleton:
                    if bp1 in kps and bp2 in kps:
                        color = self.colors.get(ident, (255, 255, 255))
                        cv2.line(img, kps[bp1], kps[bp2], color, 2)
            except Exception: continue

    def _draw_active_text(self, img, current_f):
        # (Same as before, using selected active behaviors only?)
        if self.probs_df is None: return
        try:
            if current_f in self.probs_df.index:
                row = self.probs_df.loc[current_f]
                # Only check active behaviors for the text overlay
                cols_to_check = [c for c in self.active_behaviors if c in row.index]
                if not cols_to_check: return
                
                sub_row = row[cols_to_check]
                active = sub_row[sub_row > self.threshold].index.tolist()
                
                if active:
                    text = f"Active: {', '.join(active)}"
                    cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                                1.0, (0, 0, 0), 4)
                    cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                                1.0, (0, 255, 0), 2)
        except Exception: pass
