import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, TYPE_CHECKING

import cv2
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from whisker.services.pose_estimation.public.data_structures import PoseDataset

from whisker.core.constants import KEYPOINT_COLORS

# Simple color palette for identities (BGR format for OpenCV)
COLORS = [
    tuple(int(c.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))
    for c in KEYPOINT_COLORS
]


def _get_affine_transform(
    center_x: float,
    center_y: float,
    angle_rad: float,
    scale: float,
    output_w: int,
    output_h: int,
) -> np.ndarray:
    """
    Constructs a 2x3 Affine Transformation Matrix to:
    1. Translate (center_x, center_y) to (0,0)
    2. Rotate by angle_rad
    3. Scale
    4. Translate to (output_w/2, output_h/2)
    """
    # 1. Translation to origin T1
    T1 = np.array([[1, 0, -center_x], [0, 1, -center_y], [0, 0, 1]])

    # 2. Rotation and Scale R (CCW rotation)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    # Standard 2D rotation matrix:
    # [ c  -s ]
    # [ s   c ]
    # Note: In image coords (y down), a positive angle usually rotates CW visually if we use standard math logic,
    # but here we follow the feature extraction logic: x_new = x*c - y*s.
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
    # 3. Scaling S
    S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])

    # 4. Translation to canvas center T2
    T2 = np.array([[1, 0, output_w / 2], [0, 1, output_h / 2], [0, 0, 1]])

    # Combined: M = T2 * S * R * T1
    M_full = T2 @ S @ R @ T1
    
    # Return top 2 rows for cv2.warpAffine
    return M_full[:2, :]


def create_feature_visualization_video(
    features_df: pd.DataFrame,
    identities: List[str],
    body_parts: List[str],
    skeleton: List[Tuple[str, str]],
    output_path: Path,
    fps: float = 30.0,
    canvas_size: int = 512,
    scale: float = 1.0,
    overlay_video_path: Optional[Path] = None,
    pose_dataset: Optional["PoseDataset"] = None,
    root_bodypart: Optional[str] = None,
    root_individual_id: Optional[str] = None,
    heading_axis: Optional[Tuple[str, str]] = None,
):
    if features_df.empty:
        logging.warning("Features DataFrame is empty. Cannot generate video.")
        return

    # Setup Video Reader
    cap = None
    if overlay_video_path:
        if overlay_video_path.exists() and pose_dataset and root_bodypart and root_individual_id:
            cap = cv2.VideoCapture(str(overlay_video_path))
            if not cap.isOpened():
                logging.error(f"Could not open video: {overlay_video_path}")
                cap = None
            else:
                logging.info(f"Overlaying video: {overlay_video_path.name}")
        else:
            logging.error("Overlay requested but parameters (path/data/root) are missing.")

    # Setup Video Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (canvas_size, canvas_size))
    center = canvas_size // 2

    # Pre-calc column names
    col_map = {
        idn: {
            bp: {
                "x": f"{idn}_x_{bp}",
                "y": f"{idn}_y_{bp}",
                "c": f"{idn}_c_{bp}"
            } for bp in body_parts
        } for idn in identities
    }

    logging.info(f"Rendering visualization to {output_path}...")

    for frame_idx_key, row in features_df.iterrows():
        canvas = None
        
        # --- Overlay Logic ---
        if cap:
            frame_num = -1
            
            # Fix 1: Robust frame index parsing (handles ints, strings, and filenames)
            if isinstance(frame_idx_key, int):
                frame_num = frame_idx_key
            else:
                # Handles 'path/img_0123.png' -> 123
                import re
                match = re.search(r'frame_(\d+)', str(frame_idx_key))
                if match:
                    frame_num = int(match.group(1))

            if frame_num >= 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if ret:
                    try:
                        # Fix 2: Ensure types match for lookup (MultiIndex is sensitive)
                        # If your dataset uses ints but iterator is string, this fails.
                        # We try the key as-is.
                        frame_raw_data = pose_dataset.keypoint_data.loc[(frame_idx_key, root_individual_id)]
                        
                        root_pt = frame_raw_data.loc[root_bodypart]
                        cx, cy = root_pt['x'], root_pt['y']

                        # Calculate Rotation
                        angle_rad = 0.0
                        if heading_axis and not (pd.isna(cx) or pd.isna(cy)):
                            try:
                                p_from = frame_raw_data.loc[heading_axis[0]]
                                p_to = frame_raw_data.loc[heading_axis[1]]
                                vec_x, vec_y = p_to['x'] - p_from['x'], p_to['y'] - p_from['y']
                                if (vec_x**2 + vec_y**2) > 25.0:
                                    angle_rad = (-np.pi / 2) - np.arctan2(vec_y, vec_x)
                            except KeyError:
                                pass 

                        # Warp Frame
                        if not (pd.isna(cx) or pd.isna(cy)):
                            M = _get_affine_transform(cx, cy, angle_rad, scale, canvas_size, canvas_size)
                            canvas = cv2.warpAffine(frame, M, (canvas_size, canvas_size))
                    
                    except KeyError:
                        # Fix 3: Log failure only once to avoid spam, or use debug
                        logging.debug(f"Pose data missing for {frame_idx_key}, skipping overlay.")
                        pass
                else:
                    logging.warning(f"Could not read frame {frame_num} from video.")

        # Fallback to black canvas
        if canvas is None:
            canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
            # Draw grid only on blank canvas
            cv2.line(canvas, (center, 0), (center, canvas_size), (50, 50, 50), 1)
            cv2.line(canvas, (0, center), (canvas_size, center), (50, 50, 50), 1)

        # Text overlay
        cv2.putText(canvas, f"Frame: {frame_idx_key}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # --- Draw Skeletons ---
        for i, identity in enumerate(identities):
            color = COLORS[i % len(COLORS)]
            kp_coords = {}

            # Draw Points
            for bp in body_parts:
                cols = col_map[identity][bp]
                nx, ny, conf = row.get(cols["x"]), row.get(cols["y"]), row.get(cols["c"])

                if pd.isna(nx) or pd.isna(ny) or not conf: 
                    continue

                px = int(center + nx * scale)
                py = int(center + ny * scale)
                kp_coords[bp] = (px, py)
                cv2.circle(canvas, (px, py), 3, color, -1)

            # Draw Connections
            for bp1, bp2 in skeleton:
                if bp1 in kp_coords and bp2 in kp_coords:
                    cv2.line(canvas, kp_coords[bp1], kp_coords[bp2], color, 1)

        out.write(canvas)

    out.release()
    if cap: cap.release()
    logging.info("Visualization complete.")