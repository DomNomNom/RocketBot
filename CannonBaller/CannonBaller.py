from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlbot.utils.structures.ball_prediction_struct import Slice, BallPrediction
from rlbot.utils.logging_utils import get_logger
from rlbot.utils.game_state_util import GameState, BoostState, BallState, CarState, Physics, Vector3, Rotator

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
from nombotutils.game_state import EasyGameState
from nombotutils.constants import TO_STATUE, TO_ORANGE, TO_CEILING
from nombotutils.vector_math import mag, dist, normalize, clamp, clamp01, clamp11, lerp, is_close, xy_only, z0
from nombotutils.rendering import trace

logger = get_logger('CannonBaller')

class CannonBaller(BaseAgent):

    """
    Gets shot out of a cannon.
    """

    def initialize_agent(self):
        self.controller_state = SimpleControllerState()

        # Variables about previous states.

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:

        s = EasyGameState(packet, self.team, self.index)
        # trace(mag(s.car.vel))

        if s.car.pos[TO_CEILING] < 100:
            self.set_game_state(GameState(
                cars={
                    self.index: CarState(
                        physics=Physics(
                            # rotation=Rotator(0, pi / 2, 0),
                            velocity=Vector3(
                                0,
                                10000 if s.car.pos[TO_ORANGE] < 0 else -10000,
                                1000),
                            # angular_velocity=Vector3(0, 0, 0)
                            ),
                        jumped=False,
                        double_jumped=False,
                        boost_amount=100)
                },
            ))

        self.controller_state.roll = 1
        self.controller_state.boost = False

        # Sanitize our output
        out = self.controller_state
        out.steer = clamp11(out.steer)
        out.throttle = clamp11(out.throttle)
        out.pitch = clamp11(out.pitch)
        out.yaw = clamp11(out.yaw)
        out.roll = clamp11(out.roll)
        out.jump = bool(out.jump)
        out.boost = bool(out.boost)
        out.handbrake = bool(out.handbrake)
        return out

    def get_target_pos(self, s):
        '''
        Returns a position that seems good to go towards right now.
        '''

        # TODO: maybe adjust this as we get closer/further away
        underestimated_time_to_ball = dist(s.car.pos, s.ball.pos) / (1.5 * MAX_CAR_SPEED)
        prediction_duration = clamp(underestimated_time_to_ball, 0.02, 2.0) #0.2
        # prediction_duration = 0.2
        ball_prediction = self.get_ball_prediction_struct()

        predicted_slice = bisect_ball_prediction(
            ball_prediction,
            lambda slice: not could_reach_target_in_time(
                s.car,
                Vec3(slice.physics.location.x, slice.physics.location.y, slice.physics.location.z),
                slice.game_seconds - s.time + 300.0 / dist(s.car.pos, Vec3(slice.physics.location.x, slice.physics.location.y, slice.physics.location.z))
            )
        )
        future_ball = Ball(predicted_slice.physics)
        self.render_ball(future_ball.pos)
        # future_ball = s.ball
        target_ball_pos = s.enemy_goal_center
        to_goal_dir = normalize(z0(target_ball_pos - future_ball.pos))

        # DONE: predict the ball by some small amount.
        # DONE: avoid ball when coming back
        # DONE: hit at an angle to change ball velocity
        desired_ball_speed = MAX_CAR_SPEED
        desired_ball_vel = MAX_CAR_SPEED * to_goal_dir
        desired_ball_vel_change = desired_ball_vel - future_ball.vel
        normal_dir = -normalize(z0(desired_ball_vel))  # center of ball to hit point
        # alignment = dot(normal_dir, normalize(z0(s.car.pos - future_ball.pos)))
        # hit_radius = (0.9 if alignment > 0.7 else 1.5) * BALL_RADIUS
        hit_radius = 1.0 * BALL_RADIUS
        ball_hit_offset = hit_radius * normal_dir
        # ball_hit_offset = -0.8 * BALL_RADIUS * to_goal_dir
        target_pos = future_ball.pos + ball_hit_offset

        # Avoid the ball when coming back
        if dist(s.car.pos, target_ball_pos) < dist(future_ball.pos, target_ball_pos):
            # TODO: make sure options are in bounds
            avoid_back_ball_radius = BALL_RADIUS * 5.0
            options = [
                future_ball.pos + avoid_back_ball_radius * normalize(Vec3( 1, -s.enemy_goal_dir, 0)),
                future_ball.pos + avoid_back_ball_radius * normalize(Vec3(-1, -s.enemy_goal_dir, 0)),
            ]
            best_avoid_option = min(options, key=lambda avoid_ball_pos: dist(s.car.pos, avoid_ball_pos))
            # TODO: factor in current velocity maybe
            target_pos = best_avoid_option

        return target_pos



if __name__ == '__main__':
    quick_bot_check(CannonBaller('blah', 0, 0))
