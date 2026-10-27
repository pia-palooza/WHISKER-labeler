"""Hand corrections of an ROI prediction's tracks (the Verify widget).

An animal is an arena (see :mod:`verify_tracks`), so an edit is one arena's position in one frame: a point the user
placed or moved, or ``None`` for a detection the user removed, plus whether it was placed by hand or filled in
automatically while key-frame-filling a long gap (``place_point``, ``remove_point``).

Ported from full WHISKER's ``whisker/services/roi_tracking/internal/verify_edits.py``: ``TrackEdits`` (extended with
grouped undo/redo, ``clear``, the ``auto`` flag and ``to_records``/``from_records`` to match full WHISKER's own
implementation of this feature) and ``apply_edits``/``_bbox_frames``/``_median_box_size`` (verbatim; kept for parity
with the brief's port list, though the labeler itself doesn't currently call them -- it writes the small corrections
file, not a full corrected prediction). Full WHISKER's ``build_verified_prediction``/``save_verified_run`` are
**not** ported -- those recompute occupancy/transitions/distance and need zones, calibration, fps and mapping
settings the labeler never has. The labeler only ever writes the corrections JSON (see
``whisker/services/roi_tracking/public/correction_operations.py``); applying it to a full prediction run, as
``<run>_uservalidated``, is a full-WHISKER-side job.
"""
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import numpy as np

from .metrics import is_point_in_zone
from .verify_tracks import required_frames

Point = Tuple[float, float]
_ABSENT = object()


class TrackEdits:
    """The edits of one video's tracks, keyed ``(track index, frame)``, with grouped undo and redo.

    A value is the ``(x, y)`` the user put the animal at, or ``None`` for "no animal here" -- either way recorded as
    hand-set unless ``auto`` says otherwise. A frame without an edit keeps what the prediction had. ``auto``
    distinguishes a frame the collaborator placed themselves from one filled in automatically while key-frame-
    filling a long gap (see ``place_point``); it only affects the detection strip's colour and the pending-frames
    walk-through, never ``apply_to``."""

    def __init__(self):
        self._edits: Dict[Tuple[int, int], Tuple[Optional[Point], bool]] = {}
        self._undo: List[List[Tuple[Tuple[int, int], object, object]]] = []
        self._redo: List[List[Tuple[Tuple[int, int], object, object]]] = []
        self._current_group: Optional[List[Tuple[Tuple[int, int], object, object]]] = None

    def __len__(self) -> int:
        return len(self._edits)

    def items(self):
        """``[((track, frame), value), ...]`` sorted -- just the position/None, not the ``auto`` flag."""
        return [(key, value) for key, (value, _auto) in sorted(self._edits.items())]

    def get(self, track: int, frame: int, default=None):
        entry = self._edits.get((track, frame))
        return entry[0] if entry is not None else default

    def is_auto(self, track: int, frame: int) -> bool:
        entry = self._edits.get((track, frame))
        return bool(entry[1]) if entry is not None else False

    def has(self, track: int, frame: int) -> bool:
        return (track, frame) in self._edits

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    @contextmanager
    def group(self):
        """Groups every ``set``/``clear`` call made inside the block into a single undo/redo step (so placing a key
        frame and auto-filling the gap frames it governs undo together). Safe to nest -- an inner ``group()`` while
        one is already open just joins the outer group."""
        if self._current_group is not None:
            yield
            return
        self._current_group = []
        try:
            yield
        finally:
            recorded, self._current_group = self._current_group, None
            if recorded:
                self._undo.append(recorded)
                self._redo.clear()

    def _write(self, key, state):
        if state is _ABSENT:
            self._edits.pop(key, None)
        else:
            self._edits[key] = state

    def _record(self, key, old, new):
        if self._current_group is not None:
            self._current_group.append((key, old, new))
        else:
            self._undo.append([(key, old, new)])
            self._redo.clear()

    def set(self, track: int, frame: int, value: Optional[Point], auto: bool = False):
        """Places the animal of ``track`` at ``value`` in ``frame`` (hand-set unless ``auto``), or removes it there
        when ``value`` is None."""
        key = (track, frame)
        old = self._edits.get(key, _ABSENT)
        new = (None, auto) if value is None else ((float(value[0]), float(value[1])), auto)
        if old == new:
            return
        self._write(key, new)
        self._record(key, old, new)

    def clear(self, track: int, frame: int):
        """Removes the edit at ``(track, frame)`` entirely (undoable), so it reverts to whatever the original
        prediction had -- unlike ``set(..., None)``, which records an explicit "no animal here"."""
        key = (track, frame)
        old = self._edits.get(key, _ABSENT)
        if old is _ABSENT:
            return
        self._write(key, _ABSENT)
        self._record(key, old, _ABSENT)

    def undo(self) -> Optional[Tuple[int, int]]:
        """Undoes the last group of edits; returns its first ``(track, frame)`` (to seek to), or None when there is
        nothing to undo."""
        if not self._undo:
            return None
        group = self._undo.pop()
        for key, old, _new in reversed(group):
            self._write(key, old)
        self._redo.append(group)
        return group[0][0]

    def redo(self) -> Optional[Tuple[int, int]]:
        if not self._redo:
            return None
        group = self._redo.pop()
        for key, _old, new in group:
            self._write(key, new)
        self._undo.append(group)
        return group[0][0]

    def apply_to(self, xy: np.ndarray, track: int) -> np.ndarray:
        """A copy of the (n_frames, 2) positions ``xy`` of ``track`` with its edits in place (NaN where removed)."""
        out = np.array(xy, dtype=float)
        for (t, frame), (value, _auto) in self._edits.items():
            if t == track and 0 <= frame < len(out):
                out[frame] = (np.nan, np.nan) if value is None else value
        return out

    def edited_mask(self, track: int, n_frames: int, auto: Optional[bool] = None) -> np.ndarray:
        """Which of ``track``'s frames have an edit: every one (``auto=None``), only the auto-filled ones
        (``auto=True``), or only the hand-set ones (``auto=False``)."""
        mask = np.zeros(n_frames, dtype=bool)
        for (t, frame), (_value, is_auto) in self._edits.items():
            if t == track and 0 <= frame < n_frames and (auto is None or is_auto == auto):
                mask[frame] = True
        return mask

    def to_records(self) -> List[list]:
        """``[[track, frame, x, y, auto], ...]`` sorted by ``(track, frame)``; ``x``/``y`` are ``None`` for a
        removal. The format the labeler persists and full WHISKER's apply step reads."""
        records = []
        for (track, frame), (value, auto) in sorted(self._edits.items()):
            x, y = (None, None) if value is None else value
            records.append([track, frame, x, y, bool(auto)])
        return records

    @classmethod
    def from_records(cls, records) -> "TrackEdits":
        """The inverse of :meth:`to_records`. Undo/redo history is not restored -- it isn't persisted, only the net
        edit set is."""
        edits = cls()
        for track, frame, x, y, auto in records:
            value = None if x is None else (float(x), float(y))
            edits._edits[(int(track), int(frame))] = (value, bool(auto))
        return edits


# ------------------------------------------------------------------ the key-frame-fill gesture
def place_point(edits: TrackEdits, track: int, frame: int, point: Point, orig_xy: np.ndarray,
                min_len: int = 4, interval: int = 5) -> List[int]:
    """Places ``point`` at ``(track, frame)`` by hand and, when ``frame`` is a key frame of a long gap in
    ``orig_xy`` (and not that gap's last key frame), auto-fills the same point onto every frame up to (not
    including) the next key frame -- skipping any frame the collaborator already set by hand. One undo step.
    Returns the frames that were auto-filled (empty for a short gap or a standalone frame)."""
    filled: List[int] = []
    with edits.group():
        edits.set(track, frame, point, auto=False)
        for first, last in _gap_runs_containing(orig_xy, frame):
            keyframes = required_frames(first, last, min_len, interval)
            if frame not in keyframes:
                continue
            index = keyframes.index(frame)
            following = keyframes[index + 1] if index + 1 < len(keyframes) else last + 1
            for shadow in range(frame + 1, following):
                if edits.is_auto(track, shadow) or not edits.has(track, shadow):
                    edits.set(track, shadow, point, auto=True)
                    filled.append(shadow)
    return filled


def remove_point(edits: TrackEdits, track: int, frame: int, orig_xy: np.ndarray,
                  min_len: int = 4, interval: int = 5) -> List[int]:
    """Removes the detection at ``(track, frame)`` (hand-set) and, when ``frame`` is a key frame of a long gap,
    clears (not just sets to missing -- removes the edit entirely, so it goes back to needing an answer) every
    frame it had auto-filled. One undo step. Returns the frames that were cleared."""
    cleared: List[int] = []
    with edits.group():
        edits.set(track, frame, None, auto=False)
        for first, last in _gap_runs_containing(orig_xy, frame):
            keyframes = required_frames(first, last, min_len, interval)
            if frame not in keyframes:
                continue
            index = keyframes.index(frame)
            following = keyframes[index + 1] if index + 1 < len(keyframes) else last + 1
            for shadow in range(frame + 1, following):
                if edits.is_auto(track, shadow):
                    edits.clear(track, shadow)
                    cleared.append(shadow)
    return cleared


def _gap_runs_containing(orig_xy: np.ndarray, frame: int):
    from .verify_tracks import gap_runs
    return [(first, last) for first, last in gap_runs(orig_xy) if first <= frame <= last]


# ------------------------------------------------------------------ applying the edits to a prediction
def _bbox_frames(bboxes, frames: list) -> list:
    """The stored boxes as one list per frame, aligned with that frame's detections (``None`` where a box is missing)."""
    out = []
    for i, detections in enumerate(frames):
        boxes = bboxes[i] if bboxes and i < len(bboxes) else None
        if isinstance(boxes, dict):
            boxes = [boxes]
        boxes = list(boxes) if isinstance(boxes, (list, tuple)) else []
        out.append((boxes + [None] * len(detections))[:len(detections)])
    return out


def _median_box_size(frames, boxes, coords, rotation) -> Optional[Tuple[float, float]]:
    sizes = [(b["w"], b["h"]) for dets, bbs in zip(frames, boxes) for p, b in zip(dets, bbs)
             if b and "w" in b and "h" in b and is_point_in_zone(p, coords, rotation=rotation)]
    return (float(np.median([s[0] for s in sizes])), float(np.median([s[1] for s in sizes]))) if sizes else None


def apply_edits(prediction: dict, geometries: List[Tuple[dict, float]], edits: TrackEdits) -> Tuple[list, list]:
    """``(trajectory, bboxes)`` of ``prediction`` with ``edits`` applied: one list of detections (and of their boxes)
    per frame.

    ``geometries`` holds each track's arena as ``(coordinates, rotation)``. In an edited frame the detections inside
    the arena are replaced by the placed point (or dropped when it was removed); the other arenas' detections are
    kept. A placed point's box is the detection's box moved to the point, else a box of the arena's median size, else
    none."""
    from .verify_tracks import trajectory_frames

    frames = [list(f) for f in trajectory_frames((prediction or {}).get("trajectory"))]
    boxes = _bbox_frames((prediction or {}).get("bboxes"), frames)
    default_size = {}
    for (track, frame), value in edits.items():
        if not 0 <= frame < len(frames):
            continue
        coords, rotation = geometries[track]
        kept, removed_box = [], None
        for point, box in zip(frames[frame], boxes[frame]):
            if is_point_in_zone(point, coords, rotation=rotation):
                removed_box = removed_box or box
            else:
                kept.append((point, box))
        if value is not None:
            if track not in default_size:
                default_size[track] = _median_box_size(frames, boxes, coords, rotation)
            w, h = ((removed_box["w"], removed_box["h"]) if removed_box and "w" in removed_box and "h" in removed_box
                    else default_size[track] or (None, None))
            box = None if w is None else {"x": value[0] - w / 2.0, "y": value[1] - h / 2.0, "w": w, "h": h}
            kept.append(((value[0], value[1]), box))
        frames[frame] = [p for p, _ in kept]
        boxes[frame] = [b for _, b in kept]
    return frames, boxes
