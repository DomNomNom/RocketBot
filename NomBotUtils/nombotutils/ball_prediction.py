from typing import Callable
from rlbot.utils.structures.ball_prediction_struct import Slice, BallPrediction

from nombotutils.vector_math import Vec3

def bisect_ball_prediction(ball_prediction: BallPrediction, is_too_early: Callable[[Slice], bool]) -> Slice:
    """
    Returns the first slice that is not deemed too early.
    """
    assert ball_prediction.num_slices
    lo = 0
    hi = ball_prediction.num_slices - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if is_too_early(ball_prediction.slices[mid]):
            lo = mid + 1
        else:
            hi = mid
    return ball_prediction.slices[lo]

def ball_pos(slice: Slice) -> Vec3:
    return Vec3(slice.physics.location.x, slice.physics.location.y, slice.physics.location.z)
