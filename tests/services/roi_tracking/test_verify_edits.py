import numpy as np
import pytest

from whisker.services.roi_tracking.internal.verify_edits import TrackEdits, apply_edits, place_point, remove_point

ARENA1_COORDS = {"x": 0.0, "y": 0.0, "width": 10.0, "height": 10.0}
ARENA2_COORDS = {"x": 100.0, "y": 100.0, "width": 10.0, "height": 10.0}


def test_set_then_undo_then_redo():
    edits = TrackEdits()
    edits.set(0, 5, (1.0, 2.0))
    assert edits.get(0, 5) == (1.0, 2.0)
    assert len(edits) == 1

    key = edits.undo()
    assert key == (0, 5)
    assert edits.get(0, 5) is None
    assert not edits.has(0, 5)
    assert len(edits) == 0

    key = edits.redo()
    assert key == (0, 5)
    assert edits.get(0, 5) == (1.0, 2.0)


def test_setting_the_same_value_again_is_a_no_op():
    edits = TrackEdits()
    edits.set(0, 5, (1.0, 2.0))
    assert edits.can_undo()
    edits.undo()
    assert not edits.can_undo()

    edits.set(0, 5, (1.0, 2.0))
    edits.set(0, 5, (1.0, 2.0))  # identical value again: no new undo step
    assert edits.undo() == (0, 5)
    assert not edits.can_undo()


def test_a_new_edit_clears_the_redo_stack():
    edits = TrackEdits()
    edits.set(0, 5, (1.0, 2.0))
    edits.undo()
    assert edits.can_redo()

    edits.set(0, 6, (3.0, 4.0))
    assert not edits.can_redo()


def test_setting_none_removes_a_detection():
    edits = TrackEdits()
    edits.set(0, 5, None)
    assert edits.get(0, 5, "missing") is None
    assert edits.has(0, 5)


def test_apply_to_writes_nan_for_removed_frames():
    edits = TrackEdits()
    edits.set(1, 2, None)
    edits.set(1, 3, (9.0, 9.0))
    xy = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

    out_track_1 = edits.apply_to(xy, track=1)
    assert np.isnan(out_track_1[2]).all()
    assert tuple(out_track_1[3]) == (9.0, 9.0)
    assert tuple(out_track_1[0]) == (0.0, 0.0)  # untouched frame unchanged

    out_track_0 = edits.apply_to(xy, track=0)
    assert np.array_equal(out_track_0, xy)  # edits on a different track don't apply


def test_edited_mask_marks_only_edited_frames_for_that_track():
    edits = TrackEdits()
    edits.set(0, 1, (1.0, 1.0))
    edits.set(1, 2, (2.0, 2.0))
    mask = edits.edited_mask(0, n_frames=4)
    assert list(mask) == [False, True, False, False]


def test_group_makes_several_sets_one_undo_step():
    edits = TrackEdits()
    with edits.group():
        edits.set(0, 5, (1.0, 1.0))
        edits.set(0, 6, (2.0, 2.0))
        edits.set(0, 7, (3.0, 3.0))
    assert len(edits) == 3

    key = edits.undo()
    assert key == (0, 5)       # the first key of the group
    assert len(edits) == 0     # all three went in one undo step

    key = edits.redo()
    assert key == (0, 5)
    assert len(edits) == 3


def test_group_clears_redo_once_for_the_whole_group():
    edits = TrackEdits()
    edits.set(0, 1, (0.0, 0.0))
    with edits.group():
        edits.set(0, 5, (1.0, 1.0))
        edits.set(0, 6, (2.0, 2.0))
    assert len(edits) == 3
    edits.undo()  # undoes the group
    assert edits.can_redo()
    edits.set(0, 9, (9.0, 9.0))  # a fresh edit clears the pending redo
    assert not edits.can_redo()


def test_clear_removes_an_edit_without_recording_a_removal():
    edits = TrackEdits()
    edits.set(0, 5, (1.0, 2.0))
    edits.clear(0, 5)
    assert not edits.has(0, 5)
    assert edits.get(0, 5) is None
    assert len(edits) == 0

    key = edits.undo()
    assert key == (0, 5)
    assert edits.has(0, 5)  # undoing the clear restores the original edit
    assert edits.get(0, 5) == (1.0, 2.0)


def test_clear_on_an_unedited_frame_is_a_no_op():
    edits = TrackEdits()
    edits.clear(0, 5)
    assert not edits.can_undo()


def test_auto_flag_is_stored_and_reported():
    edits = TrackEdits()
    edits.set(0, 1, (1.0, 1.0), auto=False)
    edits.set(0, 2, (2.0, 2.0), auto=True)
    assert edits.is_auto(0, 1) is False
    assert edits.is_auto(0, 2) is True
    assert edits.is_auto(0, 99) is False  # no edit there at all


def test_edited_mask_filters_by_auto():
    edits = TrackEdits()
    edits.set(0, 1, (1.0, 1.0), auto=False)
    edits.set(0, 2, (2.0, 2.0), auto=True)
    assert list(edits.edited_mask(0, 4)) == [False, True, True, False]
    assert list(edits.edited_mask(0, 4, auto=True)) == [False, False, True, False]
    assert list(edits.edited_mask(0, 4, auto=False)) == [False, True, False, False]


def test_to_records_and_from_records_round_trip():
    edits = TrackEdits()
    edits.set(0, 1, (1.5, 2.5), auto=False)
    edits.set(1, 2, None, auto=False)
    edits.set(0, 3, (4.0, 5.0), auto=True)

    records = edits.to_records()
    assert records == sorted(records)  # sorted by (track, frame)
    assert [0, 1, 1.5, 2.5, False] in records
    assert [1, 2, None, None, False] in records
    assert [0, 3, 4.0, 5.0, True] in records

    restored = TrackEdits.from_records(records)
    assert restored.get(0, 1) == (1.5, 2.5)
    assert restored.is_auto(0, 3) is True
    assert restored.has(1, 2) and restored.get(1, 2) is None
    assert len(restored) == 3
    assert not restored.can_undo()  # history is not restored, only the net edit set


def test_place_point_on_a_short_gap_has_no_shadow():
    orig_xy = np.full((5, 2), np.nan)
    edits = TrackEdits()
    filled = place_point(edits, 0, 1, (9.0, 9.0), orig_xy, min_len=4, interval=5)
    assert filled == []
    assert len(edits) == 1
    assert edits.is_auto(0, 1) is False


def test_place_point_on_a_key_frame_fills_its_shadow_span():
    orig_xy = np.full((11, 2), np.nan)  # one gap, frames 0-10: key frames 0, 5, 10
    edits = TrackEdits()
    filled = place_point(edits, 0, 0, (9.0, 9.0), orig_xy, min_len=4, interval=5)
    assert filled == [1, 2, 3, 4]
    for frame in range(1, 5):
        assert edits.get(0, frame) == (9.0, 9.0)
        assert edits.is_auto(0, frame) is True
    assert not edits.has(0, 5)  # the next key frame is untouched
    assert edits.is_auto(0, 0) is False


def test_place_point_skips_frames_already_set_by_hand():
    orig_xy = np.full((11, 2), np.nan)
    edits = TrackEdits()
    edits.set(0, 3, (7.0, 7.0), auto=False)  # the collaborator hand-set frame 3 directly
    filled = place_point(edits, 0, 0, (9.0, 9.0), orig_xy, min_len=4, interval=5)
    assert 3 not in filled
    assert edits.get(0, 3) == (7.0, 7.0)  # untouched by the auto-fill
    assert edits.get(0, 1) == (9.0, 9.0)  # the rest still filled


def test_moving_a_key_frame_later_refills_its_span():
    orig_xy = np.full((11, 2), np.nan)
    edits = TrackEdits()
    place_point(edits, 0, 0, (1.0, 1.0), orig_xy, min_len=4, interval=5)
    place_point(edits, 0, 0, (2.0, 2.0), orig_xy, min_len=4, interval=5)  # moved
    for frame in range(1, 5):
        assert edits.get(0, frame) == (2.0, 2.0)


def test_place_point_on_the_last_key_frame_has_no_shadow():
    orig_xy = np.full((11, 2), np.nan)
    edits = TrackEdits()
    filled = place_point(edits, 0, 10, (9.0, 9.0), orig_xy, min_len=4, interval=5)
    assert filled == []


def test_remove_point_clears_the_shadow_it_had_filled():
    orig_xy = np.full((11, 2), np.nan)
    edits = TrackEdits()
    place_point(edits, 0, 0, (9.0, 9.0), orig_xy, min_len=4, interval=5)
    assert len(edits) == 5  # keyframe + 4 shadow frames

    cleared = remove_point(edits, 0, 0, orig_xy, min_len=4, interval=5)
    assert cleared == [1, 2, 3, 4]
    assert edits.has(0, 0) and edits.get(0, 0) is None  # the keyframe itself is an explicit removal
    for frame in range(1, 5):
        assert not edits.has(0, frame)  # the shadow frames are cleared entirely, not set to None


def test_remove_point_is_one_undo_step():
    orig_xy = np.full((11, 2), np.nan)
    edits = TrackEdits()
    place_point(edits, 0, 0, (9.0, 9.0), orig_xy, min_len=4, interval=5)
    remove_point(edits, 0, 0, orig_xy, min_len=4, interval=5)
    assert len(edits) == 1  # just the explicit removal at frame 0

    edits.undo()
    assert len(edits) == 5  # back to keyframe + 4 shadow frames, all at once


def test_apply_edits_replaces_detections_in_the_edited_arena_only():
    prediction = {"trajectory": [[(5.0, 5.0), (105.0, 105.0)]]}  # frame 0: one point per arena
    edits = TrackEdits()
    edits.set(0, 0, (7.0, 7.0))  # move Arena1's animal
    geometries = [(ARENA1_COORDS, 0.0), (ARENA2_COORDS, 0.0)]

    frames, boxes = apply_edits(prediction, geometries, edits)
    assert len(frames) == 1
    assert set(frames[0]) == {(7.0, 7.0), (105.0, 105.0)}  # Arena2's detection is untouched


def test_apply_edits_removes_a_detection():
    prediction = {"trajectory": [[(5.0, 5.0), (105.0, 105.0)]]}
    edits = TrackEdits()
    edits.set(0, 0, None)
    geometries = [(ARENA1_COORDS, 0.0), (ARENA2_COORDS, 0.0)]

    frames, _boxes = apply_edits(prediction, geometries, edits)
    assert frames[0] == [(105.0, 105.0)]


def test_apply_edits_handles_single_animal_xy_or_none_trajectory():
    prediction = {"trajectory": [(5.0, 5.0), None]}
    edits = TrackEdits()
    edits.set(0, 1, (6.0, 6.0))  # fill the gap in frame 1
    geometries = [(ARENA1_COORDS, 0.0)]

    frames, _boxes = apply_edits(prediction, geometries, edits)
    assert frames[0] == [(5.0, 5.0)]
    assert frames[1] == [(6.0, 6.0)]


def test_apply_edits_ignores_out_of_range_frames():
    prediction = {"trajectory": [[(5.0, 5.0)]]}
    edits = TrackEdits()
    edits.set(0, 50, (1.0, 1.0))  # beyond the end of the trajectory
    geometries = [(ARENA1_COORDS, 0.0)]

    frames, boxes = apply_edits(prediction, geometries, edits)
    assert len(frames) == 1  # the out-of-range edit is silently skipped, not appended
