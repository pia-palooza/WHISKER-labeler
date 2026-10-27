import math

from whisker.services.roi_tracking.internal.metrics import get_zone_bounds, is_point_in_zone, rotate_point_about


def test_get_zone_bounds_from_x_y_width_height():
    bounds = get_zone_bounds({"x": 10.0, "y": 20.0, "width": 30.0, "height": 40.0})
    assert bounds == (10.0, 20.0, 40.0, 60.0)


def test_get_zone_bounds_from_x_min_x_max():
    bounds = get_zone_bounds({"x_min": 1.0, "y_min": 2.0, "x_max": 5.0, "y_max": 6.0})
    assert bounds == (1.0, 2.0, 5.0, 6.0)


def test_is_point_in_zone_axis_aligned():
    coords = {"x": 0.0, "y": 0.0, "width": 10.0, "height": 10.0}
    assert is_point_in_zone((5.0, 5.0), coords)
    assert not is_point_in_zone((15.0, 5.0), coords)


def test_is_point_in_zone_rotation_aware():
    # A 10x10 box centered at (50, 50), rotated 45 degrees: its corner points are no longer
    # inside the axis-aligned bounds, but a point near the center is inside regardless of rotation.
    coords = {"x": 45.0, "y": 45.0, "width": 10.0, "height": 10.0}
    assert is_point_in_zone((50.0, 50.0), coords, rotation=45.0)
    # A point just outside the unrotated box on the x-axis is swept into the rotated box.
    assert is_point_in_zone((56.0, 50.0), coords, rotation=45.0)
    assert not is_point_in_zone((56.0, 50.0), coords, rotation=0.0)


def test_rotate_point_about_zero_angle_is_identity():
    assert rotate_point_about(3.0, 4.0, 0.0, 0.0, 0.0) == (3.0, 4.0)


def test_rotate_point_about_ninety_degrees():
    x, y = rotate_point_about(1.0, 0.0, 0.0, 0.0, 90.0)
    assert math.isclose(x, 0.0, abs_tol=1e-9)
    assert math.isclose(y, 1.0, abs_tol=1e-9)
