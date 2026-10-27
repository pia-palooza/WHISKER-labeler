"""Per-arena track extraction and arena/subregion config parsing for ROI tracking.

Ported from full WHISKER's ``whisker/services/roi_tracking/internal/results_export.py``: only the two functions the
labeler's Verify widget needs, ``extract_arena_track`` and ``arenas_from_config`` (plus the private ``_point_list``
helper the first one calls). The rest of that file (occupancy/transitions export) is not ported.
"""
import math
from typing import Dict, List, Optional, Tuple

from .metrics import is_point_in_zone


def _point_list(frame) -> List[Tuple[float, float]]:
    """Normalises one trajectory frame entry into a list of (x, y) float tuples.

    A frame entry may already be a list of detections (each a ``(x, y)`` pair, a 2-element list, or a dict with
    ``x``/``y`` keys), a single ``(x, y)`` pair for a single-animal prediction, or ``None``/empty for no detection."""
    if not frame:
        return []
    if isinstance(frame, dict):
        return [(float(frame["x"]), float(frame["y"]))]
    if len(frame) == 2 and all(isinstance(v, (int, float)) and not math.isnan(v) for v in frame):
        return [(float(frame[0]), float(frame[1]))]
    points = []
    for item in frame:
        if item is None:
            continue
        if isinstance(item, dict):
            points.append((float(item["x"]), float(item["y"])))
        else:
            points.append((float(item[0]), float(item[1])))
    return points


def extract_arena_track(trajectory: list, coords: dict, rotation: float) -> List[Optional[Tuple[float, float]]]:
    """Reduces the (possibly multi-animal) trajectory to a single per-frame position
    for one arena: the detection falling inside the arena, choosing the one closest to
    the previous in-arena position when several are present (parallel-assay friendly).
    Frames with no detection inside the arena are ``None``."""
    track: List[Optional[Tuple[float, float]]] = []
    prev = None
    for frame in trajectory:
        pts = _point_list(frame)
        inside = [p for p in pts if is_point_in_zone(p, coords, rotation=rotation)]
        if not inside:
            track.append(None)
            continue
        if prev is not None and len(inside) > 1:
            chosen = min(inside, key=lambda q: (q[0] - prev[0]) ** 2 + (q[1] - prev[1]) ** 2)
        else:
            chosen = inside[0]
        track.append(chosen)
        prev = chosen
    return track


def arenas_from_config(config: dict) -> list:
    """Extracts the arena/subregion list from a saved ROI config.

    Prefers the ``arenas`` hierarchy; falls back to reconstructing arenas from the
    flattened ``rois``/``zones`` list (grouping subregions under their arena_name)."""
    if not config:
        return []
    arenas = config.get("arenas")
    if arenas:
        return arenas

    flat = config.get("rois") or config.get("zones") or []
    by_name: Dict[str, dict] = {}
    subs: Dict[str, list] = {}
    for r in flat:
        name = r.get("name") or r.get("zone_name") or ""
        arena_name = r.get("arena_name") or name
        if name == arena_name:
            by_name[arena_name] = {
                "arena_name": arena_name,
                "animal_id": r.get("animal_id", ""),
                "condition": r.get("condition", ""),
                "coordinates": r.get("coordinates", {}),
                "rotation": r.get("rotation", 0.0),
                "subregions": [],
            }
        else:
            short = name[len(arena_name) + 1:] if name.startswith(arena_name + "_") else name
            subs.setdefault(arena_name, []).append({
                "name": short,
                "contents": r.get("contents", ""),
                "coordinates": r.get("coordinates", {}),
                "rotation": r.get("rotation", 0.0),
            })
    for arena_name, arena in by_name.items():
        arena["subregions"] = subs.get(arena_name, [])
    return list(by_name.values())
