import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import logging

from whisker.base.job import BaseJob
from whisker.services.pose_estimation.public.data_structures import PoseDataset
from whisker.core.constants import KEYPOINT_COLORS

class PoseExportJob(BaseJob):
    def __init__(
        self,
        source_video_path: Path,
        output_video_path: Path,
        start_frame: int,
        end_frame: int,
        pred_data: Optional[PoseDataset],
        gt_data: Optional[PoseDataset],
        project_skeleton: List[Tuple[str, str]],
        project_bodyparts: List[str],
        project_identities: List[str],
        render_config: Dict,
        progress_callback=None
    ):
        super().__init__(progress_callback)
        self.src_path = str(source_video_path)
        self.out_path = str(output_video_path)
        self.start_f = start_frame
        self.end_f = end_frame
        
        # Pre-filter data to avoid lookups in the hot loop
        self.pred_df = pred_data.keypoint_data if pred_data else None
        self.gt_df = gt_data.keypoint_data if gt_data else None
        
        self.skeleton = project_skeleton
        self.bodyparts = project_bodyparts
        self.identities = project_identities
        self.cfg = render_config 

        # Pre-compute colors
        self.colors = {}
        n_colors = len(KEYPOINT_COLORS)

        for i, ident in enumerate(self.identities):
            hex_str = KEYPOINT_COLORS[i % n_colors].lstrip('#')
            # Convert hex pairs to ints and reverse for BGR (B, G, R)
            self.colors[ident] = tuple(int(hex_str[j:j+2], 16) for j in (4, 2, 0))

        # Pre-compute valid keys set for faster "contains" check
        self.pred_keys = set(self.pred_df.index.get_level_values('frame_index').unique()) if self.pred_df is not None else set()
        self.gt_keys = set(self.gt_df.index.get_level_values('frame_index').unique()) if self.gt_df is not None else set()

    def run(self) -> str:
        self.report_progress("Initializing Video Export...", 0)
        
        cap = cv2.VideoCapture(self.src_path)
        if not cap.isOpened():
            raise IOError(f"Could not open source: {self.src_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out_w = w * 2 if self.cfg.get('side_by_side') else w
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.out_path, fourcc, fps, (out_w, h))
        
        total_frames_to_process = self.end_f - self.start_f + 1
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_f)
        
        # Helper to find the matching key in the DF
        video_stem = Path(self.src_path).stem
        video_name = Path(self.src_path).name
        
        current_f = self.start_f
        processed = 0
        
        while current_f <= self.end_f:
            if self.is_cancelled: break
            
            ret, frame = cap.read()
            if not ret: break
            
            # Try multiple key formats
            # 1. Stem/frame_XXXXXX (Standard Internal Format)
            # 2. frame_XXXXXX (Flat Dataset)
            # 3. Name/frame_XXXXXX (If stem != name)
            candidates = [
                f"{video_stem}/frame_{current_f:06d}",
                f"frame_{current_f:06d}",
                f"{video_name}/frame_{current_f:06d}"
            ]
            
            # Canvas Setup
            if self.cfg.get('side_by_side'):
                canvas = np.hstack([frame, frame.copy()])
            else:
                canvas = frame

            # Draw Predictions
            if self.pred_df is not None:
                # Find matching key
                key = next((k for k in candidates if k in self.pred_keys), None)
                if key:
                    self._draw_pose(canvas, self.pred_df, key, offset_x=0)

            # Draw GT
            if self.cfg.get('show_gt') and self.gt_df is not None:
                key = next((k for k in candidates if k in self.gt_keys), None)
                if key:
                    offset = w if self.cfg.get('side_by_side') else 0
                    style = 'dashed' if not self.cfg.get('side_by_side') else 'solid'
                    self._draw_pose(canvas, self.gt_df, key, offset_x=offset, line_style=style)

            out.write(canvas)
            
            processed += 1
            if processed % 10 == 0:
                pct = int((processed / total_frames_to_process) * 100)
                self.report_progress(f"Exporting frame {current_f}...", pct)
            
            current_f += 1

        cap.release()
        out.release()
        
        return f"Exported {processed} frames to {Path(self.out_path).name}"

    def _draw_pose(self, img, df: pd.DataFrame, frame_key: str, offset_x: int, line_style='solid'):
        try:
            frame_data = df.loc[frame_key]
        except KeyError: return

        min_c = self.cfg.get('min_conf', 0.0)
        show_names = self.cfg.get('show_names', False)

        for ident in self.identities:
            if ident not in frame_data.index: continue
            
            try:
                ind_data = frame_data.loc[ident]
                
                # If the MultiIndex structure is (frame, id, bodypart), selecting loc[frame].loc[ident]
                # gives a Series or DataFrame indexed by bodypart.
                
                kps = {}
                # Handle Series vs DataFrame ambiguity if only 1 bodypart exists
                if isinstance(ind_data, pd.Series):
                    # Only one bodypart defined in project? unlikely but possible
                    if ind_data['c'] > min_c:
                        bp = ind_data.name # might be the bodypart name if index isn't dropped
                        x, y = int(ind_data['x']) + offset_x, int(ind_data['y'])
                        kps[bp] = (x, y)
                else:
                    valid_rows = ind_data[ind_data['c'] > min_c]
                    for bp_name, row in valid_rows.iterrows():
                        x, y = int(row['x']) + offset_x, int(row['y'])
                        kps[bp_name] = (x, y)
                        
                        # Draw Point
                        color = self.colors.get(ident, (255, 255, 255))
                        if line_style == 'dashed':
                            cv2.circle(img, (x, y), 3, color, 1) 
                        else:
                            cv2.circle(img, (x, y), 4, color, -1)
                            
                        if show_names:
                            cv2.putText(img, bp_name, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

                # Draw Skeleton
                for bp1, bp2 in self.skeleton:
                    if bp1 in kps and bp2 in kps:
                        pt1 = kps[bp1]
                        pt2 = kps[bp2]
                        color = self.colors.get(ident, (255, 255, 255))
                        
                        if line_style == 'dashed':
                            self._draw_dashed_line(img, pt1, pt2, color)
                        else:
                            cv2.line(img, pt1, pt2, color, 2)

            except Exception as e:
                logging.warning(f"Error drawing pose for {ident}: {e}")
                continue

    def _draw_dashed_line(self, img, pt1, pt2, color, thickness=1, gap=5):
        dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
        if dist == 0: return
        pts = np.linspace(pt1, pt2, int(dist / gap))
        for i in range(0, len(pts) - 1, 2):
            p_start = tuple(pts[i].astype(int))
            p_end = tuple(pts[i+1].astype(int))
            cv2.line(img, p_start, p_end, color, thickness)