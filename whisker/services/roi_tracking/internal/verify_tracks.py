"""Per-animal tracks of an ROI prediction, for checking them against the video (the Verify widget).

An animal is an arena (every arena holds one animal) and its track is what the figures use:
:func:`results_export.extract_arena_track`, the detection inside the arena, nearest to the previous one when
there are several.

Ported from full WHISKER's ``whisker/services/roi_tracking/internal/verify_tracks.py``, trimmed to what the labeler's
Verify widget needs (the two-run comparison helpers used by full WHISKER's Evaluate tab are not ported).
"""
from typing import List, Optional, Tuple

import numpy as np

from .results_export import extract_arena_track

# Distinct dot colours for arenas that have none of their own (none is lavender: that is the path).
DOT_COLORS = ["#E74C3C", "#2ECC71", "#3498DB", "#F1C40F", "#E67E22", "#1ABC9C", "#FF6FB5", "#7F8C8D"]

# Per-frame status codes shared with DetectionStrip (DETECTED, MISSING, EDITED = AGREE, DISAGREE, ONLY_A).
AGREE, DISAGREE, ONLY_A, ONLY_B, NEITHER = 0, 1, 2, 3, 4


def trajectory_frames(trajectory) -> list:
    """The stored trajectory as one list of (x, y) detections per frame. Predictions of a single animal
    store one ``(x, y)`` or ``None`` per frame instead of a list of detections."""
    frames = []
    for frame in trajectory or []:
        if frame is None:
            frames.append([])
        elif len(frame) == 2 and all(isinstance(v, (int, float)) for v in frame):
            frames.append([tuple(frame)])
        else:
            frames.append(frame)
    return frames


def arena_tracks(prediction: dict, arenas: List[dict]) -> List[dict]:
    """``[{"name", "animal_id", "color", "xy", "coordinates", "rotation"}]``, one per arena: ``xy`` is an (n_frames, 2)
    array of video pixels, NaN where the arena has no detection; ``coordinates``/``rotation`` are the arena's box."""
    frames = trajectory_frames((prediction or {}).get("trajectory"))
    out = []
    for i, arena in enumerate(arenas or []):
        coords = arena.get("coordinates") or {}
        if not coords:
            continue
        track = extract_arena_track(frames, coords, arena.get("rotation", 0.0))
        xy = np.array([p if p is not None else (np.nan, np.nan) for p in track], dtype=float).reshape(-1, 2)
        name = arena.get("arena_name") or arena.get("name") or f"arena{i + 1}"
        out.append({"name": name, "animal_id": arena.get("animal_id", "") or "",
                    "color": arena.get("color") or DOT_COLORS[i % len(DOT_COLORS)], "xy": xy,
                    "coordinates": coords, "rotation": arena.get("rotation", 0.0)})
    return out


def detected_fraction(xy: np.ndarray) -> float:
    """Share of the frames in which the track has a position."""
    return float(np.mean(~np.isnan(xy[:, 0]))) if len(xy) else 0.0


def position_at(xy: np.ndarray, frame: int) -> Optional[Tuple[float, float]]:
    """The track's position at ``frame``, or None when there is none (or the frame is outside the track)."""
    if frame < 0 or frame >= len(xy) or np.isnan(xy[frame, 0]):
        return None
    return float(xy[frame, 0]), float(xy[frame, 1])


def missing_frames(xy: np.ndarray) -> np.ndarray:
    """Indices of the frames in which the track has no position."""
    return np.flatnonzero(np.isnan(xy[:, 0]))


# A gap this long or longer is filled by key frames (first, every KEYFRAME_INTERVAL'th counted from the first, and
# last), with the frames between them holding the preceding key frame's position/removal, instead of requiring
# every single frame by hand. User-adjustable (brief: "the two numbers (4 and 5) are user settings").
KEYFRAME_MIN_GAP = 4
KEYFRAME_INTERVAL = 5


def gap_runs(xy: np.ndarray) -> List[Tuple[int, int]]:
    """Each maximal run of consecutive missing frames in ``xy``, as ``(first, last)`` inclusive frame indices."""
    missing = np.isnan(xy[:, 0])
    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, m in enumerate(missing):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(missing) - 1))
    return runs


def key_frames(first: int, last: int, interval: int = KEYFRAME_INTERVAL) -> List[int]:
    """The frames of a gap ``[first, last]`` (inclusive) placed by hand when filling it by key frame: ``first``,
    every ``interval``'th frame counted from ``first`` (not at absolute multiples of ``interval``), and ``last``."""
    frames = list(range(first, last + 1, interval))
    if frames[-1] != last:
        frames.append(last)
    return frames


def required_frames(first: int, last: int, min_len: int = KEYFRAME_MIN_GAP,
                     interval: int = KEYFRAME_INTERVAL) -> List[int]:
    """Every frame the collaborator must place by hand to fill a gap ``[first, last]``: every frame of it when it is
    shorter than ``min_len``, else just its :func:`key_frames` (so a 500-frame gap takes ~100 clicks, not 500; the
    frames in between hold the preceding key frame's value -- see ``verify_edits.place_point``)."""
    if last - first + 1 < min_len:
        return list(range(first, last + 1))
    return key_frames(first, last, interval)


def pending_frames(orig_xy: np.ndarray, xy: np.ndarray, min_len: int = KEYFRAME_MIN_GAP,
                    interval: int = KEYFRAME_INTERVAL, skipped=frozenset()) -> List[Tuple[int, int, int]]:
    """``(frame, gap_first, gap_last)`` for each frame the collaborator still needs to set to fill every gap in
    ``orig_xy`` (the gap structure before any edits, so the schedule doesn't shift as frames fill in): every
    :func:`required_frames` entry of each gap run that is still missing in ``xy`` (the edits-applied track). A gap
    whose first frame is in ``skipped`` is left out entirely."""
    out: List[Tuple[int, int, int]] = []
    for first, last in gap_runs(orig_xy):
        if first in skipped:
            continue
        for frame in required_frames(first, last, min_len, interval):
            if np.isnan(xy[frame, 0]):
                out.append((frame, first, last))
    return out


def next_missing_frame(xys: List[np.ndarray], frame: int, backwards: bool = False) -> Optional[int]:
    """The nearest frame after (or, ``backwards``, before) ``frame`` in which any of the tracks has no position, or None."""
    best = None
    for xy in xys:
        gaps = missing_frames(xy)
        gaps = gaps[gaps < frame] if backwards else gaps[gaps > frame]
        if len(gaps):
            candidate = int(gaps[-1] if backwards else gaps[0])
            if best is None or (candidate > best if backwards else candidate < best):
                best = candidate
    return best


def path_segments(xy: np.ndarray, frame: int, max_gap: int = 10, max_points: int = 3000,
                   start: int = 0) -> List[List[Tuple[float, float]]]:
    """The track from frame ``start`` (default 0) to ``frame`` (inclusive) as polylines in video pixels.

    A run of more than ``max_gap`` frames without a detection starts a new polyline (shorter gaps are bridged), so the
    path shows where the animal was lost. A long path is thinned to about ``max_points`` points, always keeping the last."""
    n = min(len(xy), frame + 1)
    first = max(0, int(start))
    if n <= first:
        return []
    valid = first + np.flatnonzero(~np.isnan(xy[first:n, 0]))
    if len(valid) < 2:
        return []
    if len(valid) > max_points:
        stride = int(np.ceil(len(valid) / max_points))
        thinned = valid[::stride]
        valid = thinned if thinned[-1] == valid[-1] else np.append(thinned, valid[-1])
    segments, current = [], [(float(xy[valid[0], 0]), float(xy[valid[0], 1]))]
    for previous, index in zip(valid[:-1], valid[1:]):
        point = (float(xy[index, 0]), float(xy[index, 1]))
        if index - previous > max_gap + 1:
            if len(current) >= 2:
                segments.append(current)
            current = [point]
        else:
            current.append(point)
    if len(current) >= 2:
        segments.append(current)
    return segments
