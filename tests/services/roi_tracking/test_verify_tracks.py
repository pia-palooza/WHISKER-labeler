import numpy as np

from whisker.services.roi_tracking.internal.verify_tracks import (
    arena_tracks, detected_fraction, gap_runs, key_frames, missing_frames, next_missing_frame, path_segments,
    pending_frames, position_at, required_frames, trajectory_frames,
)

ARENA1 = {"arena_name": "Arena1", "coordinates": {"x": 0.0, "y": 0.0, "width": 10.0, "height": 10.0}}
ARENA2 = {"arena_name": "Arena2", "coordinates": {"x": 100.0, "y": 100.0, "width": 10.0, "height": 10.0}}


def test_trajectory_frames_normalizes_multi_animal_frames():
    trajectory = [[(1.0, 2.0), (3.0, 4.0)], []]
    assert trajectory_frames(trajectory) == [[(1.0, 2.0), (3.0, 4.0)], []]


def test_trajectory_frames_normalizes_single_animal_xy_or_none():
    trajectory = [(1.0, 2.0), None, (3.0, 4.0)]
    assert trajectory_frames(trajectory) == [[(1.0, 2.0)], [], [(3.0, 4.0)]]


def test_arena_tracks_one_per_arena_with_gaps():
    prediction = {"trajectory": [[(5.0, 5.0)], [(200.0, 200.0)], [(6.0, 6.0)]]}
    tracks = arena_tracks(prediction, [ARENA1, ARENA2])
    assert [t["name"] for t in tracks] == ["Arena1", "Arena2"]
    arena1_xy = tracks[0]["xy"]
    assert arena1_xy.shape == (3, 2)
    assert not np.isnan(arena1_xy[0, 0])
    assert np.isnan(arena1_xy[1, 0])  # the detection in frame 1 is outside Arena1
    assert not np.isnan(arena1_xy[2, 0])


def test_arena_tracks_skips_arenas_without_coordinates():
    prediction = {"trajectory": [[(5.0, 5.0)]]}
    no_coords_arena = {"arena_name": "Arena3", "coordinates": {}}
    tracks = arena_tracks(prediction, [ARENA1, no_coords_arena])
    assert [t["name"] for t in tracks] == ["Arena1"]


def test_detected_fraction():
    xy = np.array([[1.0, 1.0], [np.nan, np.nan], [2.0, 2.0], [np.nan, np.nan]])
    assert detected_fraction(xy) == 0.5
    assert detected_fraction(np.zeros((0, 2))) == 0.0


def test_position_at_returns_none_for_missing_or_out_of_range():
    xy = np.array([[1.0, 2.0], [np.nan, np.nan]])
    assert position_at(xy, 0) == (1.0, 2.0)
    assert position_at(xy, 1) is None
    assert position_at(xy, 5) is None
    assert position_at(xy, -1) is None


def test_missing_frames_returns_indices_without_a_position():
    xy = np.array([[1.0, 1.0], [np.nan, np.nan], [2.0, 2.0], [np.nan, np.nan]])
    assert list(missing_frames(xy)) == [1, 3]


def test_next_missing_frame_looks_in_both_directions():
    xy_a = np.array([[1.0, 1.0], [np.nan, np.nan], [2.0, 2.0], [3.0, 3.0], [np.nan, np.nan]])
    xy_b = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]])
    assert next_missing_frame([xy_a, xy_b], frame=2, backwards=False) == 4
    assert next_missing_frame([xy_a, xy_b], frame=2, backwards=True) == 1
    assert next_missing_frame([xy_b], frame=2, backwards=False) is None


def test_path_segments_breaks_on_long_gaps_and_bridges_short_ones():
    xy = np.full((20, 2), np.nan)
    xy[0] = (0.0, 0.0)
    xy[1] = (1.0, 1.0)
    # a short (bridged) gap of 2 frames
    xy[4] = (4.0, 4.0)
    # a long gap of 12 frames before the next detection
    xy[17] = (17.0, 17.0)
    xy[18] = (18.0, 18.0)
    segments = path_segments(xy, frame=19, max_gap=10)
    assert len(segments) == 2
    assert segments[0] == [(0.0, 0.0), (1.0, 1.0), (4.0, 4.0)]
    assert segments[1] == [(17.0, 17.0), (18.0, 18.0)]


def test_path_segments_respects_start_frame():
    xy = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    assert path_segments(xy, frame=2, start=1) == [[(1.0, 1.0), (2.0, 2.0)]]


def _xy_with_gap(total: int, gap_start: int, gap_end: int) -> np.ndarray:
    xy = np.tile(np.array([1.0, 1.0]), (total, 1))
    xy[gap_start:gap_end + 1] = np.nan
    return xy


def test_gap_runs_finds_every_maximal_missing_run():
    xy = np.array([[0, 0], [np.nan, np.nan], [np.nan, np.nan], [1, 1], [np.nan, np.nan]], dtype=float)
    assert gap_runs(xy) == [(1, 2), (4, 4)]


def test_gap_runs_handles_no_gaps_and_all_gap():
    assert gap_runs(np.array([[0.0, 0.0], [1.0, 1.0]])) == []
    assert gap_runs(np.full((3, 2), np.nan)) == [(0, 2)]


def test_key_frames_counts_the_interval_from_the_gap_start():
    assert key_frames(0, 10) == [0, 5, 10]
    assert key_frames(0, 3) == [0, 3]           # no room for a 5th before the end
    assert key_frames(0, 4) == [0, 4]           # 0+5 would equal end anyway
    assert key_frames(0, 6) == [0, 5, 6]        # last doesn't land on the interval
    assert key_frames(10, 20) == [10, 15, 20]   # counted from the gap's own first frame, not absolute multiples


def test_required_frames_short_gap_is_every_frame():
    assert required_frames(10, 12) == [10, 11, 12]  # length 3 < min_len 4


def test_required_frames_long_gap_is_key_frames_only():
    assert required_frames(0, 10) == [0, 5, 10]


def test_pending_frames_lists_every_required_frame_still_missing():
    xy = _xy_with_gap(15, gap_start=2, gap_end=4)   # short gap (3 frames): every frame required
    xy[8:15] = np.nan                               # long gap of 7 frames (8..14): required 8, 13, 14
    pending = pending_frames(xy, xy.copy())
    assert [p[0] for p in pending] == [2, 3, 4, 8, 13, 14]
    assert pending[0] == (2, 2, 4)
    assert pending[3] == (8, 8, 14)


def test_pending_frames_only_lists_frames_still_missing_in_the_current_track():
    xy = _xy_with_gap(11, gap_start=0, gap_end=10)   # required: 0, 5, 10
    current = xy.copy()
    current[0] = (1.0, 1.0)                          # frame 0 has been placed
    pending = pending_frames(xy, current)
    assert [p[0] for p in pending] == [5, 10]


def test_pending_frames_skips_a_gap_whose_first_frame_is_skipped():
    xy = _xy_with_gap(15, gap_start=2, gap_end=4)
    xy[8:15] = np.nan
    pending = pending_frames(xy, xy.copy(), skipped={2})
    assert [p[0] for p in pending] == [8, 13, 14]
