from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlbot.utils.structures.ball_prediction_struct import Slice, BallPrediction
from rlbot.utils.logging_utils import get_logger
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

logger = get_logger('CannonBaller')

def get_cannon_vel(start: Vec3, target: Vec3, flight_duration: float, gravity_z: float):
    # https://www.desmos.com/calculator/ceuii0dldg
    to_target = target - start
    with magic_renderer() as renderer:
        render_ball_circle(renderer, target)

    vel_xy = z0(to_target) / flight_duration
    vel_z = Vec3(0, 0, to_target[TO_CEILING] / flight_duration + -.5*gravity_z*flight_duration)
    return vel_xy + vel_z


class CannonBaller(BaseAgent):

    """
    Gets shot out of a cannon.
    """

    min_replan_period = 0.5
    min_time_on_ground = 0.3
    prepare_to_land_period = .6

    def initialize_agent(self):
        self.controller_state = SimpleControllerState()

        # Variables about previous states.
        self.target_pos = Vec3(0,0,100)
        self.last_replan_time = 0
        self.last_time_in_air = 0

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:

        s = EasyGameState(packet, self.team, self.index)
        gravity_z = packet.game_info.world_gravity_z
        # trace(mag(s.car.vel))

        # Re plan our trajectory.
        if not s.car.on_ground:
            self.last_time_in_air = s.time
        if s.time - self.last_time_in_air > self.min_time_on_ground and s.time - self.last_replan_time > self.min_replan_period:
            self.last_replan_time = s.time
            # self.target_pos = self.get_target_pos(s)
            def pick_offset_target(slice):
                pos = ball_pos(slice)
                pos += + 50 * UP
                return pos

            time_offset = 0.0
            predicted_slice = bisect_ball_prediction(
                self.get_ball_prediction_struct(),
                lambda slice: mag(get_cannon_vel(
                    s.car.pos,
                    pick_offset_target(slice),
                    slice.game_seconds-s.time - time_offset,
                    s.gravity_z
                )) > MAX_CAR_SPEED
            )
            vel = get_cannon_vel(
                s.car.pos,
                pick_offset_target(predicted_slice),
                predicted_slice.game_seconds - s.time,
                s.gravity_z
            )
            # if mag(vel) > MAX_CAR_SPEED:
            #     print(f'cannon was not powerful enough: {mag(vel)} > {MAX_CAR_SPEED}')
            self.set_game_state(GameState(
                cars={
                    self.index: CarState(
                        physics=Physics(
                            location=Vector3(*s.car.pos),
                            rotation=Rotator(
                                acos(mag(z0(vel)) / mag(vel)),
                                atan2(vel[1], vel[0]),
                                0,
                            ),
                            velocity=Vector3(*vel),
                            angular_velocity=Vector3(*(vel)),
                        ),
                        jumped=False,
                        double_jumped=False,
                        boost_amount=100)
                },
            ))

        out = self.controller_state
        forward = s.car.vel + Vec3(0, 0, 0.1*gravity_z)
        time_to_touchdown = max(solve_quadratic(.5*gravity_z, s.car.vel[TO_CEILING], s.car.pos[TO_CEILING]), default=0)
        is_landing = time_to_touchdown < self.prepare_to_land_period or (s.car.pos[TO_CEILING] < 100 and not s.car.vel[TO_CEILING] > 100)
        out.pitch, out.yaw, out.roll = get_pitch_yaw_roll(
            s.car,
            forward,
            UP if is_landing else normalize(lerp(s.car.up, cross(forward, s.car.up), 0.1)),
        )
        out.boost = False
        out.throttle = 0
        out.throttle = 1 if is_landing else 0

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

    def get_target_pos(self, s):
        '''
        Returns a position that seems good to go towards right now.
        '''
        # out = Vec3(2000, 1000, 200)
        # if s.car.pos[TO_ORANGE] > 0:
        #     out[TO_ORANGE] *= -1
        # return out

        # TODO: maybe adjust this as we get closer/further away
        underestimated_time_to_ball = dist(s.car.pos, s.ball.pos) / (1.5 * MAX_CAR_SPEED)
        prediction_duration = clamp(underestimated_time_to_ball, 0.02, 2.0) #0.2
        # prediction_duration = 0.2
        ball_prediction = self.get_ball_prediction_struct()

        predicted_slice = bisect_ball_prediction(
            ball_prediction,
            lambda slice: mag(get_cannon_vel(
                s.car.pos,
                ball_pos(slice),
                slice.game_seconds-s.time,
                s.gravity_z
            )) > MAX_CAR_SPEED
        )

        future_ball = Ball(predicted_slice.physics)
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
