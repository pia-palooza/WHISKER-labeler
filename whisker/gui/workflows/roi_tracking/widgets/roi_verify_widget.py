"""The per-video ROI tracking editor: step through gaps in a prediction run and hand-correct positions.

Ported from full WHISKER's ``VerifyRoiTab`` (``whisker/gui/workflows/roi_tracking/tabs/verification.py``), with:
- The STUDY-panel/``MessageBus`` run selection replaced by a local combo box (the labeler has neither).
- Saving replaced by autosaving a small corrections file, keyed by track index and frame (see
  ``whisker/services/roi_tracking/public/correction_operations.py``) -- the labeler can't recompute occupancy/
  transitions/distance, so unlike full WHISKER it never writes a full corrected prediction; applying the
  corrections file to a run, as ``<run>_uservalidated``, is a full-WHISKER-side job.
- Keyboard shortcuts (``N``/``M`` step a frame, ``G``/``Shift+G`` jump between gaps, ``E`` toggles edit mode,
  ``Ctrl+Z``/``Ctrl+Shift+Z`` undo/redo), so a long gap can be filled as click-then-tap-N without reaching for the
  mouse on every frame.
- A guided "Fill gaps" mode: auto-advances through every frame still required to fill every gap (every frame of a
  short gap, or just the first, last and every Nth frame of a long one -- see ``verify_tracks.pending_frames``),
  showing a status line and letting a gap be skipped for this pass. The frames a long gap doesn't require are
  filled in automatically with the preceding key frame's position (``verify_edits.place_point``).
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from whisker.core.workspace import Workspace
from whisker.gui.widgets.media_viewer import MediaViewerWidget
from whisker.services.roi_tracking.internal.metrics import is_point_in_zone
from whisker.services.roi_tracking.internal.results_export import arenas_from_config
from whisker.services.roi_tracking.internal.verify_edits import TrackEdits, place_point, remove_point
from whisker.services.roi_tracking.internal.verify_tracks import (
    KEYFRAME_INTERVAL, KEYFRAME_MIN_GAP, arena_tracks, detected_fraction, path_segments, pending_frames,
    position_at,
)
from .detection_strip import AUTO, DETECTED, EDITED, MISSING, DetectionStrip

PATH_COLOR = "#B497E0"          # lavender, matches full WHISKER's Verify tab
PATH_ALPHA = 0.9
PATH_WIDTH = 1.5
LAST_SECONDS = 10.0
TRACK_LAST, TRACK_ALL, TRACK_NONE = "last", "all", "none"
AUTOSAVE_DEBOUNCE_MS = 2000


def _make_track_mode_combo() -> QComboBox:
    combo = QComboBox()
    combo.addItem(f"Track: last {LAST_SECONDS:g} s", TRACK_LAST)
    combo.addItem("Track: entire path so far", TRACK_ALL)
    combo.addItem("Track: none (dot only)", TRACK_NONE)
    combo.setToolTip(
        "How much of each animal's track is drawn, as a thin semi-transparent line: the last "
        f"{LAST_SECONDS:g} seconds, the whole path from the start of the video to the current frame, or no line at "
        "all. A stretch of more than 10 frames without a detection breaks the line.")
    return combo


def _track_segments(mode: str, xy, frame: int, fps: float) -> list:
    if mode == TRACK_NONE:
        return []
    window = int(round(LAST_SECONDS * (fps if fps and fps > 0 else 30.0)))
    return path_segments(xy, frame, start=max(0, frame - window) if mode == TRACK_LAST else 0)


def arenas_for_video(workspace: Workspace, dataset_name: str, video_name: str) -> list:
    """Arena/subregion config for a video: its own override in ``rois.json`` if it has one, else the dataset's
    ``template.json`` (brief section 3.1). Read-only; never touches the prediction JSON."""
    template = workspace.get_roi_template(dataset_name) or {}
    overrides = workspace.get_roi_video_rois(dataset_name) or {}
    video_cfg = overrides.get(Path(video_name).name, {})
    cfg = video_cfg if video_cfg.get("has_overrides", False) else template
    return arenas_from_config(cfg)


class RoiVerifyWidget(QWidget):
    """One video's ROI tracking editor. See module docstring."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._workspace: Optional[Workspace] = None
        self._dataset_name: Optional[str] = None
        self._video_name: Optional[str] = None
        self._video_stem: Optional[str] = None
        self._run_name: Optional[str] = None
        self._arenas: List[dict] = []
        self._tracks: List[dict] = []
        self._loaded_run: Optional[str] = None
        self._original_prediction: Optional[dict] = None
        self._geometries: List[Tuple[dict, float]] = []
        self._orig_xy: List[np.ndarray] = []
        self._edits = TrackEdits()
        self._drag: Optional[tuple] = None

        self._min_len = KEYFRAME_MIN_GAP
        self._interval = KEYFRAME_INTERVAL
        self._skipped: Dict[int, Set[int]] = {}
        self._guided_active = False
        self._guided_target: Optional[Tuple[int, int, int, int]] = None

        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(AUTOSAVE_DEBOUNCE_MS)
        self._autosave_timer.timeout.connect(self._autosave)

        self._init_ui()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # --- UI ---
    def _init_ui(self):
        layout = QVBoxLayout(self)

        self.title_label = QLabel("")
        layout.addWidget(self.title_label)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Run:"))
        self.run_combo = QComboBox()
        self.run_combo.currentTextChanged.connect(self._on_run_changed)
        controls.addWidget(self.run_combo)
        self.show_dots_cb = QCheckBox("Dots")
        self.show_dots_cb.setChecked(True)
        self.show_dots_cb.toggled.connect(self._refresh_overlay)
        controls.addWidget(self.show_dots_cb)
        self.show_ids_cb = QCheckBox("Names")
        self.show_ids_cb.setChecked(True)
        self.show_ids_cb.toggled.connect(self._refresh_overlay)
        controls.addWidget(self.show_ids_cb)
        self.show_arenas_cb = QCheckBox("Arenas")
        self.show_arenas_cb.setChecked(True)
        self.show_arenas_cb.toggled.connect(self._apply_arena_outlines)
        controls.addWidget(self.show_arenas_cb)
        self.track_mode_combo = _make_track_mode_combo()
        self.track_mode_combo.currentIndexChanged.connect(self._refresh_overlay)
        controls.addWidget(self.track_mode_combo)
        controls.addStretch()
        layout.addLayout(controls)

        edit_controls = QHBoxLayout()
        self.edit_cb = QCheckBox("Edit tracking (E)")
        self.edit_cb.setToolTip(
            "Click inside an arena to place/move its animal in the current frame, drag to fine-tune, right-click to "
            "remove. Playback pauses while this is on.")
        self.edit_cb.toggled.connect(self._on_edit_mode_toggled)
        edit_controls.addWidget(self.edit_cb)
        self.undo_btn = QPushButton("Undo (Ctrl+Z)")
        self.undo_btn.clicked.connect(self._undo)
        edit_controls.addWidget(self.undo_btn)
        self.redo_btn = QPushButton("Redo (Ctrl+Shift+Z)")
        self.redo_btn.clicked.connect(self._redo)
        edit_controls.addWidget(self.redo_btn)
        self.prev_gap_btn = QPushButton("◀ Gap (Shift+G)")
        self.prev_gap_btn.clicked.connect(lambda: self._go_to_gap(backwards=True))
        edit_controls.addWidget(self.prev_gap_btn)
        self.next_gap_btn = QPushButton("Gap ▶ (G)")
        self.next_gap_btn.clicked.connect(lambda: self._go_to_gap(backwards=False))
        edit_controls.addWidget(self.next_gap_btn)
        edit_controls.addStretch()
        layout.addLayout(edit_controls)

        guided_controls = QHBoxLayout()
        self.fill_gaps_cb = QCheckBox("Guided fill gaps")
        self.fill_gaps_cb.setToolTip(
            "Walks through every frame still needed to fill every gap, closest first. A gap of "
            f"{self._min_len}+ frames only needs its first, last and every {self._interval}th frame placed; the "
            "rest fill in automatically with that position.")
        self.fill_gaps_cb.toggled.connect(self._on_fill_gaps_toggled)
        guided_controls.addWidget(self.fill_gaps_cb)
        self.skip_gap_btn = QPushButton("Skip gap")
        self.skip_gap_btn.setEnabled(False)
        self.skip_gap_btn.clicked.connect(self._skip_current_gap)
        guided_controls.addWidget(self.skip_gap_btn)
        guided_controls.addWidget(QLabel("Min. gap length:"))
        self.min_len_spin = QSpinBox()
        self.min_len_spin.setRange(1, 9999)
        self.min_len_spin.setValue(self._min_len)
        self.min_len_spin.valueChanged.connect(self._on_gap_settings_changed)
        guided_controls.addWidget(self.min_len_spin)
        guided_controls.addWidget(QLabel("Key-frame interval:"))
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 9999)
        self.interval_spin.setValue(self._interval)
        self.interval_spin.valueChanged.connect(self._on_gap_settings_changed)
        guided_controls.addWidget(self.interval_spin)
        guided_controls.addStretch()
        layout.addLayout(guided_controls)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.detection_strip = DetectionStrip()
        layout.addWidget(self.detection_strip)

        self.media_viewer = MediaViewerWidget()
        layout.addWidget(self.media_viewer, 1)
        self.media_viewer.frame_changed.connect(self._on_frame_changed)
        self.media_viewer.scene_clicked.connect(self._on_scene_pressed)
        self.media_viewer.scene_dragged.connect(self._on_scene_dragged)
        self.media_viewer.scene_released.connect(self._on_scene_released)
        self.media_viewer.scene_right_clicked.connect(self._on_scene_right_clicked)
        self.detection_strip.frame_selected.connect(self.media_viewer.seek_to_frame)

        self._update_edit_buttons()

    # --- Keyboard shortcuts ---
    def keyPressEvent(self, event: QKeyEvent):
        key, mods = event.key(), event.modifiers()
        if mods == Qt.KeyboardModifier.ControlModifier and key == Qt.Key.Key_Z:
            self._undo()
            return
        if mods == (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier) and key == Qt.Key.Key_Z:
            self._redo()
            return
        if mods == Qt.KeyboardModifier.ShiftModifier and key == Qt.Key.Key_G:
            self._go_to_gap(backwards=True)
            return
        if mods == Qt.KeyboardModifier.NoModifier:
            if key == Qt.Key.Key_N:
                self._step_frame(1)
                return
            if key == Qt.Key.Key_M:
                self._step_frame(-1)
                return
            if key == Qt.Key.Key_G:
                self._go_to_gap(backwards=False)
                return
            if key == Qt.Key.Key_E:
                self.edit_cb.toggle()
                return
        super().keyPressEvent(event)

    def _step_frame(self, delta: int):
        if not self._tracks:
            return
        self.media_viewer.seek_to_frame(self.media_viewer.get_current_frame() + delta)

    # --- Loading media / predictions ---
    def set_media(self, workspace: Workspace, dataset_name: str, video_path: Path):
        self._flush_pending_autosave()
        self._workspace = workspace
        self._dataset_name = dataset_name
        self._video_name = video_path.name
        self._video_stem = video_path.stem
        self.title_label.setText(f"{dataset_name} / {self._video_name}")
        self.edit_cb.setChecked(False)
        self.fill_gaps_cb.setChecked(False)
        self._skipped = {}
        self.media_viewer.set_media(video_path)
        self._populate_run_combo()

    def _populate_run_combo(self):
        runs = self._workspace.list_roi_runs(self._dataset_name) if self._workspace and self._dataset_name else []
        self.run_combo.blockSignals(True)
        self.run_combo.clear()
        self.run_combo.addItems(runs)
        self.run_combo.blockSignals(False)
        if runs:
            self._on_run_changed(runs[0])
        else:
            self._clear_predictions()
            self.status_label.setText("No ROI prediction run has a video for this dataset.")

    def _on_run_changed(self, run_name: str):
        self._flush_pending_autosave()
        self._run_name = run_name or None
        self._load_predictions()

    @staticmethod
    def _track_name(track: dict) -> str:
        return track["animal_id"] or track["name"]

    def _load_predictions(self):
        if not (self._workspace and self._run_name and self._dataset_name and self._video_name):
            self._clear_predictions()
            return
        original_prediction = self._workspace.load_roi_prediction(self._run_name, self._dataset_name,
                                                                    self._video_stem)
        if not original_prediction:
            self._clear_predictions()
            self.status_label.setText(f"Run '{self._run_name}' has no prediction for this video.")
            return
        arenas = arenas_for_video(self._workspace, self._dataset_name, self._video_name)
        original_tracks = arena_tracks(original_prediction, arenas)
        if not original_tracks:
            self._clear_predictions()
            self.status_label.setText(
                "This video has no arenas drawn (set up in full WHISKER), so there is no animal to follow.")
            return

        self._arenas, self._tracks, self._loaded_run = arenas, original_tracks, self._run_name
        self._original_prediction = original_prediction
        self._orig_xy = [t["xy"].copy() for t in original_tracks]
        self._geometries = [(t["coordinates"], t["rotation"]) for t in original_tracks]
        self._skipped = {}
        self._guided_target = None

        self._edits = self._workspace.load_roi_corrections(self._run_name, self._dataset_name, self._video_stem)
        logging.info(
            f"ROI verify: loaded run '{self._run_name}' for {self._video_name} "
            f"({len(original_tracks)} animal(s), {len(self._edits)} saved correction(s)).")

        self._apply_arena_outlines()
        self._after_edit()

    def _clear_predictions(self):
        self._arenas, self._tracks, self._loaded_run = [], [], None
        self._original_prediction, self._orig_xy, self._drag = None, [], None
        self._geometries = []
        self._edits = TrackEdits()
        self._skipped, self._guided_target = {}, None
        self.media_viewer.set_track_overlay([])
        self.media_viewer.clear_arena_boxes()
        self.detection_strip.set_rows([], [])
        self._update_edit_buttons()

    # --- Rebuilding derived state after an edit ---
    def _after_edit(self):
        statuses = []
        for k, track in enumerate(self._tracks):
            track["xy"] = xy = self._edits.apply_to(self._orig_xy[k], k)
            status = np.full(len(xy), DETECTED, dtype=np.int8)
            status[np.isnan(xy[:, 0])] = MISSING
            detected = ~np.isnan(xy[:, 0])
            status[self._edits.edited_mask(k, len(xy), auto=True) & detected] = AUTO
            status[self._edits.edited_mask(k, len(xy), auto=False) & detected] = EDITED
            statuses.append(status)
        self.detection_strip.set_rows([self._track_name(t) for t in self._tracks], statuses)
        self.detection_strip.set_current_frame(self.media_viewer.get_current_frame())
        self._update_status()
        self._update_edit_buttons()
        self._refresh_overlay()
        if self._tracks:
            self._autosave_timer.start()

    def _update_status(self):
        if not self._tracks:
            if not self.status_label.text():
                self.status_label.setText("")
            return
        if self._guided_active:
            self._update_guided_status()
            return
        parts = [f"{self._track_name(t)}: detected in {100 * detected_fraction(t['xy']):.0f}% of frames"
                 for t in self._tracks]
        text = f"Run '{self._loaded_run}' - " + "; ".join(parts)
        if len(self._edits):
            text += f"; {len(self._edits)} correction(s) saved"
        self.status_label.setText(text)

    def _update_edit_buttons(self):
        loaded = bool(self._tracks)
        self.edit_cb.setEnabled(loaded)
        self.fill_gaps_cb.setEnabled(loaded)
        self.undo_btn.setEnabled(loaded and self._edits.can_undo())
        self.redo_btn.setEnabled(loaded and self._edits.can_redo())
        self.prev_gap_btn.setEnabled(loaded)
        self.next_gap_btn.setEnabled(loaded)

    # --- Autosave (the collaborator never has to press Save for work to survive) ---
    def _autosave(self):
        if not (self._workspace and self._tracks and self._loaded_run and self._original_prediction is not None):
            return
        if not len(self._edits):
            # Back to untouched: drop the corrections file rather than keep an empty one around.
            self._workspace.discard_roi_corrections(self._loaded_run, self._dataset_name, self._video_stem)
            return
        total_frames = len(self._tracks[0]["xy"])
        prediction_sha256 = self._workspace.get_roi_prediction_sha256(
            self._loaded_run, self._dataset_name, self._video_stem)
        self._workspace.save_roi_corrections(
            self._loaded_run, self._dataset_name, self._video_name, self._video_stem, self._edits,
            total_frames, self._loaded_run, prediction_sha256)

    def _flush_pending_autosave(self):
        if self._autosave_timer.isActive():
            self._autosave_timer.stop()
            self._autosave()

    # --- Gap navigation ("G"/"Shift+G" -- the nearest frame still required to fill a gap, i.e. every frame of a
    # short gap but only the key frames of a long one, same set the guided mode walks through). ---
    def _go_to_gap(self, backwards: bool):
        if not self._tracks:
            return
        current = max(0, self.media_viewer.get_current_frame())
        frames = sorted({frame for frame, _k, _first, _last in self._compute_pending()})
        candidates = [f for f in frames if (f < current if backwards else f > current)]
        frame = (max if backwards else min)(candidates) if candidates else None
        if frame is None:
            self.status_label.setText("No more frames without a detection in that direction.")
            return
        self.media_viewer.seek_to_frame(frame)

    def _on_edit_mode_toggled(self, checked: bool):
        if not checked and self.fill_gaps_cb.isChecked():
            self.fill_gaps_cb.setChecked(False)  # guided mode needs edit mode on; turning it off ends guided mode
            return
        if checked:
            self.media_viewer.pause()
            self.status_label.setText(
                "Edit tracking is on: click inside an arena to place or move its animal, drag to fine-tune, "
                "right-click to remove. Tap N to step forward, M back, G / Shift+G to jump between gaps. A gap of "
                f"{self._min_len}+ frames only needs its first, last and every {self._interval}th frame placed -- "
                "the rest fill in automatically.")
        else:
            self._update_status()

    # --- Guided "Fill gaps" mode ---
    def _compute_pending(self) -> List[Tuple[int, int, int, int]]:
        """``[(frame, track, gap_first, gap_last), ...]`` sorted by frame, then track: every frame still needed to
        fill every gap in every track, per :func:`verify_tracks.pending_frames`."""
        out: List[Tuple[int, int, int, int]] = []
        for k, track in enumerate(self._tracks):
            skipped = self._skipped.get(k, frozenset())
            for frame, first, last in pending_frames(self._orig_xy[k], track["xy"], self._min_len, self._interval,
                                                       skipped):
                out.append((frame, k, first, last))
        out.sort(key=lambda t: (t[0], t[1]))
        return out

    def _on_fill_gaps_toggled(self, checked: bool):
        self._guided_active = checked
        self.skip_gap_btn.setEnabled(checked)
        if checked:
            self.edit_cb.setChecked(True)
            self.media_viewer.pause()
            self._advance_guided()
        else:
            self._guided_target = None
            self._update_status()

    def _advance_guided(self):
        if not self._guided_active:
            return
        pending = self._compute_pending()
        if not pending:
            self._guided_target = None
            self.status_label.setText("All gaps are filled.")
            self.fill_gaps_cb.setChecked(False)
            return
        frame, track, first, last = pending[0]
        self._guided_target = (frame, track, first, last)
        self._update_guided_status(len(pending))
        self.media_viewer.seek_to_frame(frame)

    def _update_guided_status(self, n_pending: Optional[int] = None):
        if not self._guided_target:
            return
        frame, track, first, last = self._guided_target
        if n_pending is None:
            n_pending = len(self._compute_pending())
        name = self._track_name(self._tracks[track])
        self.status_label.setText(
            f"Guided fill -- {name}: gap frames {first}-{last} ({last - first + 1} frames), now at frame {frame} "
            f"({n_pending} frame(s) left to place). Click inside {name}'s arena, or 'Skip gap' to leave this gap "
            "for later.")

    def _skip_current_gap(self):
        if not self._guided_target:
            return
        _frame, track, first, _last = self._guided_target
        self._skipped.setdefault(track, set()).add(first)
        self._advance_guided()

    def _on_gap_settings_changed(self, *_args):
        self._min_len = self.min_len_spin.value()
        self._interval = self.interval_spin.value()
        self.fill_gaps_cb.setToolTip(
            "Walks through every frame still needed to fill every gap, closest first. A gap of "
            f"{self._min_len}+ frames only needs its first, last and every {self._interval}th frame placed; the "
            "rest fill in automatically with that position.")
        if self._guided_active:
            self._advance_guided()

    # --- Editing gestures ---
    def _edit_target(self, x: float, y: float) -> Optional[tuple]:
        """``(track index, frame)`` a press at video pixel (x, y) edits, or None (with the reason in the status line)."""
        if not (self.edit_cb.isChecked() and self._tracks):
            return None
        self.media_viewer.pause()
        frame = self.media_viewer.get_current_frame()
        if frame < 0 or frame >= len(self._tracks[0]["xy"]):
            self.status_label.setText("This frame is beyond the end of the loaded predictions.")
            return None
        for k, track in enumerate(self._tracks):
            if is_point_in_zone((x, y), track["coordinates"], rotation=track["rotation"]):
                return k, frame
        self.status_label.setText("Press inside an arena: each arena holds one animal.")
        return None

    def _commit_edit(self, track: int, frame: int, value: Optional[Tuple[float, float]]):
        """Sets one edit via :func:`verify_edits.place_point`/:func:`verify_edits.remove_point`, which auto-fills
        (or clears) the frames a long gap's key frame governs."""
        if value is None:
            remove_point(self._edits, track, frame, self._orig_xy[track], self._min_len, self._interval)
        else:
            place_point(self._edits, track, frame, value, self._orig_xy[track], self._min_len, self._interval)

    def _on_scene_pressed(self, x: float, y: float):
        target = self._edit_target(x, y)
        if target is not None:
            self._drag = (*target, (x, y))
            self._move_drag(x, y)
        self.setFocus()  # so N/M/G/E/Ctrl+Z keep working right after a click, without reaching for the mouse again

    def _move_drag(self, x: float, y: float):
        k, frame, _ = self._drag
        track = self._tracks[k]
        if is_point_in_zone((x, y), track["coordinates"], rotation=track["rotation"]):  # it stays in its own arena
            self._drag = (k, frame, (x, y))
            track["xy"][frame] = (x, y)
            self._refresh_overlay()

    def _on_scene_dragged(self, x: float, y: float):
        if self._drag is not None:
            self._move_drag(x, y)

    def _on_scene_released(self, x: float, y: float):
        if self._drag is None:
            return
        self._move_drag(x, y)
        k, frame, point = self._drag
        self._drag = None
        self._commit_edit(k, frame, point)
        self._after_edit()
        if self._guided_active:
            self._advance_guided()

    def _on_scene_right_clicked(self, x: float, y: float):
        target = self._edit_target(x, y)
        if target is None:
            return
        k, frame = target
        if position_at(self._tracks[k]["xy"], frame) is None:
            self.status_label.setText("This animal has no detection in this frame.")
            return
        self._commit_edit(k, frame, None)
        self._after_edit()
        if self._guided_active:
            self._advance_guided()

    # --- Undo / redo ---
    def _undo(self):
        self._step_history(self._edits.undo())

    def _redo(self):
        self._step_history(self._edits.redo())

    def _step_history(self, key: Optional[tuple]):
        if key is not None:
            self.media_viewer.seek_to_frame(key[1])
        self._after_edit()

    # --- Overlay ---
    def _on_frame_changed(self, frame: int):
        self.detection_strip.set_current_frame(frame)
        self._refresh_overlay()

    def _refresh_overlay(self, *_args):
        viewer = self.media_viewer
        if not self._tracks:
            viewer.set_track_overlay([])
            return
        frame = max(0, viewer.get_current_frame())
        radius = max(4.0, float(viewer.video_width or 0.0) / 150.0)
        mode = self.track_mode_combo.currentData()
        items = []
        for track in self._tracks:
            xy = track["xy"]
            position = position_at(xy, frame)
            items.append({
                "segments": _track_segments(mode, xy, frame, viewer.fps),
                "path_color": PATH_COLOR, "path_alpha": PATH_ALPHA, "path_width": PATH_WIDTH,
                "dot": position if self.show_dots_cb.isChecked() else None,
                "dot_color": track["color"], "dot_radius": radius,
                "label": self._track_name(track) if self.show_ids_cb.isChecked() else "",
            })
        viewer.set_track_overlay(items)

    def _apply_arena_outlines(self, *_args):
        viewer = self.media_viewer
        if not (self._tracks and self.show_arenas_cb.isChecked()):
            viewer.clear_arena_boxes()
            return
        boxes = []
        for arena in self._arenas:
            c = arena.get("coordinates") or {}
            if c:
                boxes.append((float(c.get("x", 0.0)), float(c.get("y", 0.0)),
                              float(c.get("width", 0.0)), float(c.get("height", 0.0))))
        viewer.set_arena_boxes(boxes)
