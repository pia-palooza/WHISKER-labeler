"""Turning one frame's predicted keypoint rows into media_viewer.set_pose_data's point format.

Shared by the Label task's "Primer Predictions" overlay (pose_labeling/top_controls.py) and the
Evaluate task's video overlay (tabs/evaluation.py) - same per-instance point/line shape, different
choice of identities and skeleton to draw.
"""
from typing import Callable, List, Sequence, Tuple

import pandas as pd
from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QColor

MIN_CONFIDENCE = 0.1


def build_pose_overlay_points(
    image_df: pd.DataFrame,
    identities: Sequence[str],
    skeleton: Sequence[Tuple[str, str]],
    color_for: Callable[[str], QColor],
    min_confidence: float = MIN_CONFIDENCE,
) -> List[dict]:
    """``image_df`` is one frame's keypoint rows, indexed by (individual_id, body_part) - i.e.
    ``PoseDataset.keypoint_data.loc[frame_key]``. Returns one dict per identity that has any
    keypoint above ``min_confidence``: ``{"name", "color", "points": {body_part: QPointF}, "lines":
    [(QPointF, QPointF), ...]}``, ready for ``MediaViewerWidget.set_pose_data``."""
    points_data = []
    for identity_id in identities:
        try:
            inst_df = image_df.loc[identity_id]
        except KeyError:
            continue

        points = {}
        if isinstance(inst_df, pd.Series):
            if pd.notna(inst_df.get("c")) and inst_df.get("c", 0) > min_confidence:
                points[inst_df.name] = QPointF(inst_df["x"], inst_df["y"])
        else:
            valid_rows = inst_df[(inst_df["c"] > min_confidence) & inst_df["c"].notna()]
            for bp_name, row in valid_rows.iterrows():
                points[str(bp_name)] = QPointF(row["x"], row["y"])

        lines = [
            (points[p1], points[p2])
            for p1, p2 in skeleton
            if p1 in points and p2 in points
        ]

        if points or lines:
            points_data.append({
                "name": identity_id,
                "color": color_for(identity_id),
                "points": points,
                "lines": lines,
            })

    return points_data
