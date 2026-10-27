"""Point-in-(possibly rotated)-zone geometry for ROI tracking.

Ported from full WHISKER's ``whisker/services/roi_tracking/internal/metrics.py``, trimmed to the geometry helpers the
labeler's Verify widget needs. ``compute_metrics`` (occupancy/transitions/distance) is not ported: the labeler never
recomputes those, it only ever writes the small corrections file (see ``internal/verify_edits.py``).
"""
from typing import Dict, List, Tuple

import numpy as np


def get_zone_bounds(zone_coords: Dict[str, float], is_normalized: bool = False, frame_w: int = 1,
                     frame_h: int = 1) -> Tuple[float, float, float, float]:
    """
    Extracts x_min, y_min, x_max, y_max from a zone coordinates dict (supporting both x_min/x_max and x/y/width/height
    formats).
    """
    if "x_min" in zone_coords or "y_min" in zone_coords:
        x_min = zone_coords.get("x_min", 0.0)
        y_min = zone_coords.get("y_min", 0.0)
        x_max = zone_coords.get("x_max", 1.0 if is_normalized else float(frame_w))
        y_max = zone_coords.get("y_max", 1.0 if is_normalized else float(frame_h))
    else:
        x = zone_coords.get("x", 0.0)
        y = zone_coords.get("y", 0.0)
        w = zone_coords.get("width") or zone_coords.get("w") or float(frame_w)
        h = zone_coords.get("height") or zone_coords.get("h") or float(frame_h)
        x_min = x
        y_min = y
        x_max = x + w
        y_max = y + h

    if is_normalized:
        x_min *= frame_w
        y_min *= frame_h
        x_max *= frame_w
        y_max *= frame_h

    return x_min, y_min, x_max, y_max


def rotate_point_about(px: float, py: float, cx: float, cy: float, angle_deg: float) -> Tuple[float, float]:
    """
    Rotates point (px, py) about center (cx, cy) by angle_deg degrees (counter-clockwise in standard math orientation;
    screen y-down makes it clockwise visually). Returns the rotated point. angle_deg == 0.0 returns the point unchanged.
    """
    if not angle_deg:
        return px, py
    theta = np.radians(angle_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    dx = px - cx
    dy = py - cy
    rx = cx + dx * cos_t - dy * sin_t
    ry = cy + dx * sin_t + dy * cos_t
    return rx, ry


def is_point_in_zone(point: Tuple[float, float], zone_coords: Dict[str, float], is_normalized: bool = False,
                      frame_w: int = 1, frame_h: int = 1, rotation: float = 0.0) -> bool:
    """
    Checks if a 2D point is inside a (possibly rotated) rectangular zone definition.

    When ``rotation`` is non-zero the zone rectangle is treated as rotated by ``rotation`` degrees about its own
    center. The test is exact: we rotate the query point by ``-rotation`` back into the zone's local (unrotated)
    frame and do the axis-aligned check.
    """
    px, py = point
    x_min, y_min, x_max, y_max = get_zone_bounds(zone_coords, is_normalized, frame_w, frame_h)
    if rotation:
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        px, py = rotate_point_about(px, py, cx, cy, -rotation)
    return bool((x_min <= px <= x_max) and (y_min <= py <= y_max))


def zones_for_video(template: Dict, video_config: Dict) -> List[Dict]:
    """The zones of one video in the shape full WHISKER's ``compute_metrics`` takes: the video's own zones when it has
    overrides, otherwise the dataset template's. Kept for parity with full WHISKER's config shape; the labeler itself
    only needs ``is_point_in_zone`` for editing, not this, but it is cheap and may be useful to summary/debug code."""
    if video_config.get("has_overrides", False):
        zones = []
        for z in video_config.get("rois", []):
            coords = z.get("coordinates", {})
            x, y = coords.get("x", 0.0), coords.get("y", 0.0)
            zones.append({
                "zone_name": z.get("name") or z.get("zone_name"),
                "coordinates": {"x_min": x, "y_min": y,
                                 "x_max": x + coords.get("width", 0.0), "y_max": y + coords.get("height", 0.0)},
                "rotation": z.get("rotation", 0.0),
            })
        return zones
    return [{
        "zone_name": z.get("zone_name") or z.get("name"),
        "normalized_coords": z.get("normalized_coords", {}),
        "coordinates": z.get("coordinates", {}),
        "rotation": z.get("rotation", 0.0),
    } for z in (template.get("zones", []) or template.get("rois", []))]
