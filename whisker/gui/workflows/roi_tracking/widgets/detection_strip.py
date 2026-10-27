"""Ported from full WHISKER's ``whisker/gui/workflows/roi_tracking/widgets/detection_strip.py``, extended with a
4th status, AUTO (light blue): a frame filled in automatically while key-frame-filling a long gap (see
``verify_edits.place_point``), as distinct from EDITED (a frame placed by hand, whether a key frame or a standalone
one). ``column_statuses`` is overridden because a pixel column's priority order differs from the base class's
(two-run) comparison: an unresolved gap outranks everything, and -- per spec -- a hand-set frame outranks an
auto-filled one."""
import numpy as np

from whisker.services.roi_tracking.internal.verify_tracks import AGREE, DISAGREE, ONLY_A, ONLY_B

from .agreement_strip import AgreementStrip

DETECTED, MISSING, EDITED, AUTO = AGREE, DISAGREE, ONLY_A, ONLY_B
DETECTION_COLORS = {DETECTED: "#2ECC71", MISSING: "#E74C3C", EDITED: "#3498DB", AUTO: "#A9D4F5"}
# Worst-first priority when a pixel column covers several frames: an unresolved gap is shown over everything else,
# then a hand-set correction over an auto-filled one, then a plain detection.
_PRIORITY = {MISSING: 3, EDITED: 2, AUTO: 1, DETECTED: 0}
_STATUS_BY_PRIORITY = {v: k for k, v in _PRIORITY.items()}


class DetectionStrip(AgreementStrip):
    """One colour strip per animal over the whole video: where its arena has a detection (green), where it has none
    (red), where the position was set by hand (blue), and where it was filled in automatically while key-frame-
    filling a long gap (light blue). A pixel column that covers several frames shows the most telling status, so a
    short gap is never hidden and a hand-set frame is never hidden behind an auto-filled one. Clicking a strip seeks
    the video to that frame."""

    STATUS_COLORS = DETECTION_COLORS

    @staticmethod
    def column_statuses(status: np.ndarray, columns: int) -> np.ndarray:
        n = len(status)
        if n == 0 or columns <= 0:
            return np.zeros(0, dtype=np.int8)
        if n <= columns:
            return status[(np.arange(columns) * n // columns)]
        priority = np.vectorize(_PRIORITY.get)(status)
        edges = np.linspace(0, n, columns + 1).astype(int)
        worst = np.maximum.reduceat(priority, edges[:-1])
        return np.array([_STATUS_BY_PRIORITY[int(w)] for w in worst], dtype=np.int8)
