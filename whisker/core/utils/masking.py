"""
Virtual clipping-mask frame source for multi-arena datasets.

Given a base video (or a single frame) and one axis-aligned ROI box, this module
produces frames with everything *outside* the box blacked out while the box
region itself is left untouched. The frame size is never changed and no files
are written — masking is applied on read. Because there is no crop, any
keypoints/labels produced against a masked frame stay in the original
full-frame pixel coordinate space.

Box convention: ``(x, y, w, h)`` — the top-left corner ``(x, y)`` plus size
``(w, h)``, all in full-frame pixels (the same space arena placements are stored
in).
"""

from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

import cv2
import numpy as np

# (x, y, w, h) in full-frame pixels.
Box = Tuple[int, int, int, int]


def clip_box_to_frame(box: Box, frame_width: int, frame_height: int) -> Box:
    """
    Clip ``box`` to the frame rectangle, returning ``(x, y, w, h)`` of the
    visible intersection. If the box does not overlap the frame at all, returns
    ``(0, 0, 0, 0)`` (an empty ROI).
    """
    x, y, w, h = box
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(int(frame_width), int(x) + int(w))
    y1 = min(int(frame_height), int(y) + int(h))
    if x1 <= x0 or y1 <= y0:
        return (0, 0, 0, 0)
    return (x0, y0, x1 - x0, y1 - y0)


def mask_frame(frame: np.ndarray, box: Box, in_place: bool = False) -> np.ndarray:
    """
    Return ``frame`` with everything outside ``box`` set to black (0).

    The returned array has the **same shape and dtype** as the input; pixels
    inside the (frame-clipped) box are byte-for-byte identical to the source.

    Args:
        frame: HxW or HxWxC image array.
        box: ``(x, y, w, h)`` ROI in full-frame pixels.
        in_place: When True, blacks out the input array's border regions and
            returns it (no allocation). When False (default), returns a fresh
            zero-filled array with only the ROI copied in.
    """
    if frame is None:
        raise ValueError("mask_frame received a None frame")

    h, w = frame.shape[:2]
    cx, cy, cw, ch = clip_box_to_frame(box, w, h)

    if in_place:
        if cw == 0 or ch == 0:
            frame[...] = 0
            return frame
        # Zero the four bands surrounding the ROI, leaving the ROI untouched.
        frame[:cy, :] = 0            # above
        frame[cy + ch:, :] = 0       # below
        frame[cy:cy + ch, :cx] = 0   # left
        frame[cy:cy + ch, cx + cw:] = 0  # right
        return frame

    out = np.zeros_like(frame)
    if cw > 0 and ch > 0:
        out[cy:cy + ch, cx:cx + cw] = frame[cy:cy + ch, cx:cx + cw]
    return out


class MaskedVideoReader:
    """
    Reads frames from a base video and returns them with everything outside a
    fixed ROI box blacked out. The full frame is preserved (only pixels outside
    the box are zeroed); no masked video file is written.

    This is the per-arena "pseudo-video" frame source: point it at a parent
    video plus one arena box and it behaves like a normal reader that happens to
    show only that arena.

    Usage::

        with MaskedVideoReader(path, box) as reader:
            for frame in reader:          # sequential masked frames
                ...
            ret, frame = reader.read_at(100)
    """

    def __init__(self, video_path: Union[str, Path], box: Box):
        self.video_path = str(video_path)
        self.box: Box = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        self._cap: Optional[cv2.VideoCapture] = None

    # -- lifecycle ---------------------------------------------------------
    def open(self) -> "MaskedVideoReader":
        if self._cap is None:
            self._cap = cv2.VideoCapture(self.video_path)
        return self

    def _ensure_open(self) -> cv2.VideoCapture:
        if self._cap is None:
            self.open()
        return self._cap

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "MaskedVideoReader":
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()

    # -- properties --------------------------------------------------------
    @property
    def frame_count(self) -> int:
        return int(self._ensure_open().get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def width(self) -> int:
        return int(self._ensure_open().get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self._ensure_open().get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def fps(self) -> float:
        return float(self._ensure_open().get(cv2.CAP_PROP_FPS))

    # -- reading -----------------------------------------------------------
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the next frame and return ``(ret, masked_frame_or_None)``."""
        cap = self._ensure_open()
        ret, frame = cap.read()
        if not ret or frame is None:
            return False, None
        return True, mask_frame(frame, self.box, in_place=True)

    def read_at(self, index: int) -> Tuple[bool, Optional[np.ndarray]]:
        """Seek to ``index`` and read that (masked) frame."""
        cap = self._ensure_open()
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        return self.read()

    def __iter__(self) -> Iterator[np.ndarray]:
        cap = self._ensure_open()
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            yield mask_frame(frame, self.box, in_place=True)
