from whisker.services.roi_tracking.internal.results_export import arenas_from_config, extract_arena_track

ARENA = {"x": 0.0, "y": 0.0, "width": 10.0, "height": 10.0}
OTHER_ARENA = {"x": 100.0, "y": 100.0, "width": 10.0, "height": 10.0}


def test_extract_arena_track_single_detection_per_frame():
    trajectory = [[(5.0, 5.0)], [(6.0, 6.0)], []]
    track = extract_arena_track(trajectory, ARENA, rotation=0.0)
    assert track == [(5.0, 5.0), (6.0, 6.0), None]


def test_extract_arena_track_picks_the_detection_nearest_the_previous_one():
    # Two in-arena detections in the second frame: must pick the one closest to the previous position.
    trajectory = [[(1.0, 1.0)], [(1.5, 1.5), (9.0, 9.0)]]
    track = extract_arena_track(trajectory, ARENA, rotation=0.0)
    assert track == [(1.0, 1.0), (1.5, 1.5)]


def test_extract_arena_track_ignores_detections_outside_the_arena():
    trajectory = [[(200.0, 200.0)]]
    track = extract_arena_track(trajectory, ARENA, rotation=0.0)
    assert track == [None]


def test_extract_arena_track_handles_single_animal_xy_or_none_frames():
    trajectory = [(5.0, 5.0), None, (6.0, 6.0)]
    track = extract_arena_track(trajectory, ARENA, rotation=0.0)
    assert track == [(5.0, 5.0), None, (6.0, 6.0)]


def test_arenas_from_config_prefers_the_arenas_key():
    config = {"arenas": [{"arena_name": "Arena1", "coordinates": ARENA}]}
    assert arenas_from_config(config) == config["arenas"]


def test_arenas_from_config_reconstructs_from_flat_rois_with_subregions():
    config = {
        "rois": [
            {"name": "Arena1", "arena_name": "Arena1", "coordinates": ARENA, "animal_id": "mouse1"},
            {"name": "Arena1_center", "arena_name": "Arena1", "coordinates": ARENA, "contents": "center zone"},
            {"name": "Arena2", "arena_name": "Arena2", "coordinates": OTHER_ARENA},
        ]
    }
    arenas = {a["arena_name"]: a for a in arenas_from_config(config)}
    assert set(arenas) == {"Arena1", "Arena2"}
    assert arenas["Arena1"]["animal_id"] == "mouse1"
    assert [s["name"] for s in arenas["Arena1"]["subregions"]] == ["center"]
    assert arenas["Arena2"]["subregions"] == []


def test_arenas_from_config_empty_config_returns_empty_list():
    assert arenas_from_config({}) == []
    assert arenas_from_config(None) == []
