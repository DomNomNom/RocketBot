from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlbot.utils.structures.ball_prediction_struct import Slice, BallPrediction
from rlbot.utils.game_state_util import GameState, BoostState, BallState, CarState, Physics, Vector3, Rotator

from math import atan2, asin, acos
import numpy as np

try:  # NomBotUtils Bootstrap
    import nombotutils
except ImportError:
    from pathlib import Path
    from pip._internal import main as pipmain
    import sys
    utils_path = Path(__file__).absolute().parent.parent / 'NomBotUtils'
    pipmain(['install', '-e', str(utils_path)])
    sys.path.append(str(utils_path))
from nombotutils.quick_check import quick_bot_check
from nombotutils.game_state import EasyGameState, Ball
from nombotutils.constants import TO_STATUE, TO_ORANGE, TO_CEILING, UP, MAX_CAR_SPEED, BALL_RADIUS
from nombotutils.vector_math import Vec2, Vec3, mag, dist, normalize, clamp, clamp01, clamp11, lerp, is_close, xy_only, z0, cross
from nombotutils.nonlinear_math import solve_quadratic
from nombotutils.rendering import trace, magic_renderer, render_ball_circle
from nombotutils.movement import get_pitch_yaw_roll
from nombotutils.ball_prediction import bisect_ball_prediction, ball_pos


class Defendinator(BaseAgent):

    """
    Gets shot out of a cannon.
    """

    def initialize_agent(self):
        self.controller_state = SimpleControllerState()


    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:

        s = EasyGameState(packet, self.team, self.index)
        out = self.controller_state

        # Sanitize our output
        out.steer = clamp11(out.steer)
        out.throttle = clamp11(out.throttle)
        out.pitch = clamp11(out.pitch)
        out.yaw = clamp11(out.yaw)
        out.roll = clamp11(out.roll)
        out.jump = bool(out.jump)
        out.boost = bool(out.boost)
        out.handbrake = bool(out.handbrake)
        return out




if __name__ == '__main__':
    quick_bot_check(Defendinator('blah', 0, 0))
