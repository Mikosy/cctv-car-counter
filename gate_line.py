"""
Determines which side of the virtual gate line a point is on, and whether a
vehicle has crossed it between two frames. Pure math, no video/model
dependencies -- easy to unit test on its own.
"""

import config

_P1, _P2 = config.GATE_LINE


def side_of_line(point) -> float:
    """
    Returns the signed cross product of (P2-P1) x (point-P1). Sign tells you
    which side of the line the point is on; magnitude is proportional to
    distance from the line (not normalized, but we only care about the sign).
    """
    x, y = point
    x1, y1 = _P1
    x2, y2 = _P2
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def box_leading_edge(box):
    """
    Horizontal center, but the BOTTOM edge of the box rather than its
    vertical center. Used for gate-crossing checks instead of box_center()
    because a vehicle driving straight at the camera fills more of the
    frame over time, cropping its box top -- this keeps the centroid
    artificially high even as the vehicle (and its plate) is already well
    past the line. The bottom edge tracks the vehicle's actual leading edge
    and isn't fooled by this.
    """
    x1, _, x2, y2 = box
    return ((x1 + x2) / 2, y2)


def direction_from_crossing(prev_side: float, curr_side: float):
    """
    Returns "entry", "exit", or None (no crossing) based on the sign flip
    between two consecutive frames' side_of_line() values.
    """
    if prev_side == 0 or curr_side == 0:
        return None  # sitting exactly on the line; wait for a clearer frame
    if (prev_side > 0) == (curr_side > 0):
        return None  # same side, no crossing yet

    positive_meaning = config.POSITIVE_SIDE_MEANS
    crossed_to_positive = curr_side > 0
    if crossed_to_positive:
        return positive_meaning
    else:
        return "entry" if positive_meaning == "exit" else "exit"