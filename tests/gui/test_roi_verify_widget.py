import json

import pytest
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QKeyEvent

from whisker.gui.workflows.roi_tracking.widgets.detection_strip import AUTO, DETECTED, EDITED, MISSING
from whisker.gui.workflows.roi_tracking.widgets.roi_verify_widget import RoiVerifyWidget


def _fake_frame_cursor(widget: RoiVerifyWidget, total_frames: int = 5, start: int = 0) -> dict:
    """Replaces the widget's QMediaPlayer-backed frame cursor with a plain in-memory one, so editor logic can be
    tested without depending on (flaky, headless) real video playback/seeking."""
    state = {"frame": start}
    widget.media_viewer.get_current_frame = lambda: state["frame"]
    widget.media_viewer.seek_to_frame = lambda frame: state.__setitem__("frame", max(0, min(frame, total_frames - 1)))
    widget.media_viewer.pause = lambda: None
    return state


@pytest.fixture
def verify_widget(roi_workspace):
    ws, video_path = roi_workspace
    w = RoiVerifyWidget()
    w.set_media(ws, "ds1", video_path)
    yield w, ws, video_path
    w._autosave_timer.stop()


def test_loading_predictions_builds_one_track_per_arena(verify_widget):
    w, _ws, _video_path = verify_widget
    assert len(w._tracks) == 2
    assert {t["name"] for t in w._tracks} == {"Arena1", "Arena2"}


def test_the_strip_shows_detected_and_missing(verify_widget):
    w, _ws, _video_path = verify_widget
    assert w.detection_strip._names == ["Arena1", "Arena2"]
    arena1_status, arena2_status = w.detection_strip._rows
    assert list(arena1_status) == [DETECTED] * 5
    assert arena2_status[2] == MISSING
    assert arena2_status[0] == DETECTED


def test_pressing_outside_every_arena_is_rejected(verify_widget):
    w, _ws, _video_path = verify_widget
    _fake_frame_cursor(w)
    w.edit_cb.setChecked(True)
    w._on_scene_pressed(1000.0, 1000.0)
    assert len(w._edits) == 0


def test_nothing_is_edited_unless_edit_tracking_is_on(verify_widget):
    w, _ws, _video_path = verify_widget
    _fake_frame_cursor(w)
    w._on_scene_pressed(45.0, 6.0)
    w._on_scene_released(45.0, 6.0)
    assert len(w._edits) == 0


def test_pressing_in_an_arena_fills_a_gap(verify_widget):
    w, _ws, _video_path = verify_widget
    state = _fake_frame_cursor(w)
    state["frame"] = 2  # Arena2's gap frame
    w.edit_cb.setChecked(True)
    w._on_scene_pressed(45.0, 6.0)   # inside Arena2
    w._on_scene_released(45.0, 6.0)
    assert w._edits.get(1, 2) == (45.0, 6.0)
    assert w.detection_strip._rows[1][2] == EDITED


def test_pressing_on_a_detected_frame_moves_it(verify_widget):
    w, _ws, _video_path = verify_widget
    state = _fake_frame_cursor(w)
    state["frame"] = 0  # Arena1 already has a detection here
    w.edit_cb.setChecked(True)
    w._on_scene_pressed(8.0, 8.0)
    w._on_scene_released(8.0, 8.0)
    assert w._edits.get(0, 0) == (8.0, 8.0)


def test_drag_fine_tunes_and_commits_only_on_release(verify_widget):
    w, _ws, _video_path = verify_widget
    state = _fake_frame_cursor(w)
    state["frame"] = 2
    w.edit_cb.setChecked(True)
    w._on_scene_pressed(45.0, 6.0)
    assert len(w._edits) == 0  # press alone doesn't commit
    w._on_scene_dragged(50.0, 10.0)
    assert len(w._edits) == 0  # neither does a drag
    w._on_scene_released(50.0, 10.0)
    assert w._edits.get(1, 2) == (50.0, 10.0)


def test_a_drag_cannot_leave_the_arena_it_started_in(verify_widget):
    w, _ws, _video_path = verify_widget
    state = _fake_frame_cursor(w)
    state["frame"] = 2
    w.edit_cb.setChecked(True)
    w._on_scene_pressed(45.0, 6.0)   # starts in Arena2
    w._on_scene_dragged(5.0, 5.0)    # Arena1's territory: rejected, stays put
    w._on_scene_released(5.0, 5.0)
    assert w._edits.get(1, 2) == (45.0, 6.0)


def test_right_click_removes_a_detection(verify_widget):
    w, _ws, _video_path = verify_widget
    state = _fake_frame_cursor(w)
    state["frame"] = 0
    w.edit_cb.setChecked(True)
    w._on_scene_right_clicked(5.0, 5.0)
    assert w._edits.has(0, 0)
    assert w._edits.get(0, 0) is None


def test_right_click_on_an_already_missing_detection_is_a_no_op(verify_widget):
    w, _ws, _video_path = verify_widget
    state = _fake_frame_cursor(w)
    state["frame"] = 2
    w.edit_cb.setChecked(True)
    w._on_scene_right_clicked(45.0, 6.0)  # already missing
    assert len(w._edits) == 0


def test_undo_redo_restore_an_edit_and_seek_to_its_frame(verify_widget):
    w, _ws, _video_path = verify_widget
    state = _fake_frame_cursor(w)
    state["frame"] = 2
    w.edit_cb.setChecked(True)
    w._on_scene_pressed(45.0, 6.0)
    w._on_scene_released(45.0, 6.0)
    assert len(w._edits) == 1

    state["frame"] = 0  # move away, then undo should seek back
    w._undo()
    assert len(w._edits) == 0
    assert state["frame"] == 2

    state["frame"] = 0
    w._redo()
    assert len(w._edits) == 1
    assert state["frame"] == 2


def test_go_to_gap_finds_the_next_and_previous_missing_frame(verify_widget):
    w, _ws, _video_path = verify_widget
    state = _fake_frame_cursor(w)
    w._go_to_gap(backwards=False)
    assert state["frame"] == 2
    state["frame"] = 4
    w._go_to_gap(backwards=True)
    assert state["frame"] == 2


def test_n_and_m_step_one_frame(verify_widget):
    w, _ws, _video_path = verify_widget
    state = _fake_frame_cursor(w, start=2)
    w._step_frame(1)
    assert state["frame"] == 3
    w._step_frame(-1)
    assert state["frame"] == 2


def test_keyboard_shortcuts_dispatch_to_the_right_actions(verify_widget):
    w, _ws, _video_path = verify_widget
    state = _fake_frame_cursor(w, start=0)

    def press(key, modifiers=Qt.KeyboardModifier.NoModifier):
        event = QKeyEvent(QEvent.Type.KeyPress, key, modifiers)
        w.keyPressEvent(event)

    press(Qt.Key.Key_N)
    assert state["frame"] == 1
    press(Qt.Key.Key_M)
    assert state["frame"] == 0
    press(Qt.Key.Key_G)
    assert state["frame"] == 2  # jumped to the gap
    press(Qt.Key.Key_G, Qt.KeyboardModifier.ShiftModifier)
    assert state["frame"] == 2  # no gap earlier than frame 2 exists in this fixture, so nothing to jump to
    assert not w.edit_cb.isChecked()
    press(Qt.Key.Key_E)
    assert w.edit_cb.isChecked()


def test_first_edit_autosaves_a_corrections_file_and_resumes_on_reopen(roi_workspace):
    ws, video_path = roi_workspace
    w = RoiVerifyWidget()
    w.set_media(ws, "ds1", video_path)
    state = _fake_frame_cursor(w)
    state["frame"] = 2
    w.edit_cb.setChecked(True)
    w._on_scene_pressed(45.0, 6.0)
    w._on_scene_released(45.0, 6.0)

    w._flush_pending_autosave()  # normally debounced; force it now
    corrections_path = ws.roi_corrections.get_corrections_path("run1", "ds1", "vid1")
    assert corrections_path.exists()
    assert ws.roi_corrections.has_corrections("run1", "ds1", "vid1")
    # the original prediction file is never modified in place
    original = json.loads(ws.roi_predictions.get_prediction_path("run1", "ds1", "vid1").read_text())
    assert original["trajectory"][2] == [[5.0, 7.0]]  # Arena2 still missing in the original

    w2 = RoiVerifyWidget()
    w2.set_media(ws, "ds1", video_path)
    assert w2._edits.get(1, 2) == (45.0, 6.0)
    w2._autosave_timer.stop()


def test_undoing_every_edit_discards_the_corrections_file(roi_workspace):
    ws, video_path = roi_workspace
    w = RoiVerifyWidget()
    w.set_media(ws, "ds1", video_path)
    state = _fake_frame_cursor(w)
    state["frame"] = 2
    w.edit_cb.setChecked(True)
    w._on_scene_pressed(45.0, 6.0)
    w._on_scene_released(45.0, 6.0)
    w._flush_pending_autosave()
    assert ws.roi_corrections.has_corrections("run1", "ds1", "vid1")

    w._undo()
    w._flush_pending_autosave()
    assert not ws.roi_corrections.has_corrections("run1", "ds1", "vid1")
    w._autosave_timer.stop()


def test_a_video_with_no_edits_keeps_only_its_original_prediction(roi_workspace):
    ws, video_path = roi_workspace
    w = RoiVerifyWidget()
    w.set_media(ws, "ds1", video_path)
    w._flush_pending_autosave()
    assert not ws.roi_corrections.has_corrections("run1", "ds1", "vid1")
    w._autosave_timer.stop()


def test_gap_button_only_stops_at_keyframes_of_a_long_gap(long_gap_workspace):
    ws, video_path = long_gap_workspace
    w = RoiVerifyWidget()
    w.set_media(ws, "ds1", video_path)
    state = _fake_frame_cursor(w, total_frames=15)

    w._go_to_gap(backwards=False)
    assert state["frame"] == 2
    w._go_to_gap(backwards=False)
    assert state["frame"] == 7   # not 3, 4, 5 or 6: those are shadow frames of keyframe 2
    w._go_to_gap(backwards=False)
    assert state["frame"] == 12
    w._go_to_gap(backwards=False)
    assert state["frame"] == 12  # nothing left to jump to
    w._autosave_timer.stop()


def test_placing_a_keyframe_fills_its_shadow_frames_with_the_same_position(long_gap_workspace):
    ws, video_path = long_gap_workspace
    w = RoiVerifyWidget()
    w.set_media(ws, "ds1", video_path)
    state = _fake_frame_cursor(w, total_frames=15)
    w.edit_cb.setChecked(True)

    state["frame"] = 2
    w._on_scene_pressed(9.0, 9.0)
    w._on_scene_released(9.0, 9.0)

    # frames 3-6 (shadow of keyframe 2) now hold keyframe 2's position; frame 7 (the next keyframe) is still a gap
    for frame in range(3, 7):
        assert w._edits.get(0, frame) == (9.0, 9.0)
    assert not w._edits.has(0, 7)
    assert w.detection_strip._rows[0][5] == AUTO

    # the next Gap press skips straight to keyframe 7 -- the shadow frames are no longer missing
    w._go_to_gap(backwards=False)
    assert state["frame"] == 7


def test_placing_the_last_keyframe_has_no_shadow_to_fill(long_gap_workspace):
    ws, video_path = long_gap_workspace
    w = RoiVerifyWidget()
    w.set_media(ws, "ds1", video_path)
    state = _fake_frame_cursor(w, total_frames=15)
    w.edit_cb.setChecked(True)

    state["frame"] = 12
    w._on_scene_pressed(20.0, 20.0)
    w._on_scene_released(20.0, 20.0)
    assert len(w._edits) == 1  # only frame 12 itself -- it governs nothing after it


def test_right_click_on_a_keyframe_clears_its_shadow_frames_too(long_gap_workspace):
    ws, video_path = long_gap_workspace
    w = RoiVerifyWidget()
    w.set_media(ws, "ds1", video_path)
    state = _fake_frame_cursor(w, total_frames=15)
    w.edit_cb.setChecked(True)

    state["frame"] = 2
    w._on_scene_pressed(9.0, 9.0)
    w._on_scene_released(9.0, 9.0)
    assert len(w._edits) == 5  # keyframe 2 + shadow frames 3,4,5,6

    w._on_scene_right_clicked(9.0, 9.0)  # remove the keyframe itself
    assert w._edits.get(0, 2) is None
    for frame in range(3, 7):
        assert w._edits.get(0, frame) is None
    w._autosave_timer.stop()


def test_guided_fill_gaps_advances_through_required_frames(long_gap_workspace):
    ws, video_path = long_gap_workspace
    w = RoiVerifyWidget()
    w.set_media(ws, "ds1", video_path)
    state = _fake_frame_cursor(w, total_frames=15)

    w.fill_gaps_cb.setChecked(True)
    assert w.edit_cb.isChecked()  # guided mode turns edit mode on
    assert state["frame"] == 2  # the gap's first required frame

    w._on_scene_pressed(9.0, 9.0)
    w._on_scene_released(9.0, 9.0)
    assert state["frame"] == 7  # auto-advanced past the shadow frames straight to the next keyframe

    w._on_scene_pressed(9.0, 9.0)
    w._on_scene_released(9.0, 9.0)
    assert state["frame"] == 12  # the gap's last required frame

    w._on_scene_pressed(9.0, 9.0)
    w._on_scene_released(9.0, 9.0)
    assert not w.fill_gaps_cb.isChecked()  # nothing left to place -- guided mode turns itself off
    w._autosave_timer.stop()


def test_skip_gap_button_leaves_the_gap_for_later(long_gap_workspace):
    ws, video_path = long_gap_workspace
    w = RoiVerifyWidget()
    w.set_media(ws, "ds1", video_path)
    state = _fake_frame_cursor(w, total_frames=15)
    w.fill_gaps_cb.setChecked(True)
    assert state["frame"] == 2

    w._skip_current_gap()
    assert not w.fill_gaps_cb.isChecked()  # this was the only gap, so guided mode has nothing left and turns off
    assert len(w._edits) == 0  # skipping placed nothing
    w._autosave_timer.stop()


def test_changing_the_key_frame_interval_changes_the_required_frames(long_gap_workspace):
    ws, video_path = long_gap_workspace
    w = RoiVerifyWidget()
    w.set_media(ws, "ds1", video_path)
    state = _fake_frame_cursor(w, total_frames=15)

    w.interval_spin.setValue(3)
    w.fill_gaps_cb.setChecked(True)
    assert state["frame"] == 2
    w._on_scene_pressed(9.0, 9.0)
    w._on_scene_released(9.0, 9.0)
    assert state["frame"] == 5  # keyframes are now 2, 5, 8, 11, 12 instead of 2, 7, 12
    w._autosave_timer.stop()


def test_auto_filled_shadow_frames_are_distinguished_from_hand_set_ones(long_gap_workspace):
    ws, video_path = long_gap_workspace
    w = RoiVerifyWidget()
    w.set_media(ws, "ds1", video_path)
    state = _fake_frame_cursor(w, total_frames=15)
    w.edit_cb.setChecked(True)

    state["frame"] = 2
    w._on_scene_pressed(9.0, 9.0)
    w._on_scene_released(9.0, 9.0)
    assert w.detection_strip._rows[0][2] == EDITED  # the key frame itself was placed by hand
    assert w.detection_strip._rows[0][4] == AUTO     # a shadow frame was filled automatically
    w._autosave_timer.stop()


def test_a_short_gap_still_requires_every_frame_by_hand(roi_workspace):
    """The original (short-gap) fixture's 1-frame gap behaves exactly as before: no keyframe shortcut applies."""
    ws, video_path = roi_workspace
    w = RoiVerifyWidget()
    w.set_media(ws, "ds1", video_path)
    state = _fake_frame_cursor(w)
    state["frame"] = 2
    w.edit_cb.setChecked(True)
    w._on_scene_pressed(45.0, 6.0)
    w._on_scene_released(45.0, 6.0)
    assert len(w._edits) == 1  # no shadow frames for a gap this short
    w._autosave_timer.stop()


def test_a_single_animal_xy_or_none_prediction_loads(roi_workspace):
    ws, video_path = roi_workspace
    pred_path = ws.roi_predictions.get_prediction_path("run1", "ds1", "vid1")
    pred_path.write_text(json.dumps({"trajectory": [[5.0, 5.0], None, [5.0, 7.0]], "total_frames": 3}))
    ws.scan_predictions()

    w = RoiVerifyWidget()
    w.set_media(ws, "ds1", video_path)
    arena1 = next(t for t in w._tracks if t["name"] == "Arena1")
    assert arena1["xy"].shape == (3, 2)
    w._autosave_timer.stop()
