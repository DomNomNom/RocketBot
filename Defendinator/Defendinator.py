import ctypes
from typing import Tuple
from enum import Enum
from collections import deque, Counter
from contextlib import contextmanager
from math import pi


from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.game_state_util import GameState, BoostState, BallState, CarState, Physics, Vector3, Rotator
from rlbot.utils.structures.ball_prediction_struct import BallPrediction, Slice as BallPredictionSlice
from rlbot.utils.structures.ball_prediction_struct import Slice, BallPrediction
from rlbot.utils.structures.game_data_struct import GameTickPacket, Vector3

from math import tau, sin, cos, tan, copysign, sqrt
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
from nombotutils.constants import TO_STATUE, TO_ORANGE, TO_CEILING, UP, MAX_CAR_SPEED, BALL_RADIUS, GOAL_CENTER_TO_POST
from nombotutils.vector_math import Vec2, Vec3, mag, dist, normalize, clamp, clamp01, clamp11, lerp, is_close, xy_only, z0, cross, struct_vector3_to_numpy, dot, vec2angle
from nombotutils.nonlinear_math import solve_quadratic
from nombotutils.rendering import trace, magic_renderer, render_ball_circle
from nombotutils.movement import get_pitch_yaw_roll, flip_towards
from nombotutils.ball_prediction import bisect_ball_prediction, ball_pos


def zero_centered_angle(theta:float) -> float:
    while theta > tau/2:
        theta -= tau
    return theta

def clamp(x, minimum=0, maximum=1):
    return min(maximum, max(minimum, x))

def distance(a: Vector3, b: Vector3):
    """
    Returns the euclidian distance between @a and @b
    TODO: use a shared library for this.
    """
    return sqrt(
        (a.x - b.x)**2 +
        (a.y - b.y)**2 +
        (a.z - b.z)**2
    )

def obj_distance(a, b):
    """
    Returns the distance between two objects which have a "physics" (type Physics) property.
    """
    return distance(a.physics.location, b.physics.location)

def get_steer_towards(s, target_pos):
    towards_target = target_pos - s.car.pos
    target_on_car_plane = np.array([
        dot(towards_target, s.car.forward),
        dot(towards_target, s.car.right),
    ])
    angle = vec2angle(target_on_car_plane)
    steer = angle*2.0
    return steer



class Defendinator(BaseAgent):
    """
    A goalie which tries to block the ball bymoving along a line parallel to the goals.
    A known weakness of this goalie bot is that it never comes out to challange the ball.
    """

    # How far away from the center of the car we should aim to intercept the ball.
    AIM_Z_DIST = 20

    # Don't go too far outside the width of the goal.
    MAX_X = 4000
    MIN_X = -MAX_X

    MAX_Z = 800
    MIN_Z = 0

    JUMP_BEFORE_INTERCEPT_SECONDS = .55
    DODGE_BEFORE_INTERCEPT_SECONDS = .45
    HEIGHT_BEFORE_DODGING = 50

    MIN_HIGH_JUMP_Z = 200  # minumum height of the predicted ball to get into the HIGH_JUMP state.
    MAX_HIGH_JUMP_Z = 630  # When not to jump at all because the ball it soo hightg

    DESIRED_OFFSET_FROM_OWN_GOAL_Y = 100

    MAX_CONVENIENT_JUMP_HEIGHT = 3*BALL_RADIUS
    MAX_CONVENIENT_JUMP_DISTANCE = 3*BALL_RADIUS

    class State(Enum):
        GROUND = 1
        JUMPING = 2
        DODGING = 3
        IDLE = 4
        HIGH_JUMP = 5
        HIGH_JUMP_GROUND = 6
        KICKOFF = 7
    assert len(State) == len(State.__members__)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ticks_in_dodge_state = 0
        # State de-noising:
        self.state_history = deque([self.State.GROUND] * 5)
        self.state_history_counter = Counter(self.state_history)

    def high_jump_before_intercept(self, predicted_ball_z: float) -> float:
        """
        Returns how long the high jump maneuver needs to be triggered before the intercept
        """
        return min(.99, predicted_ball_z/600.)

    def get_output(self, game_tick_packet: GameTickPacket) -> SimpleControllerState:
        s = EasyGameState(game_tick_packet, self.team, self.index)

        car_obj = game_tick_packet.game_cars[self.index]
        car = car_obj.physics

        # Find the time/position where we should intercept the ball.
        car_y = car.location.y
        to_ball_y = game_tick_packet.game_ball.physics.location.y - car_y
        intercept_y = car_y + copysign(self.AIM_Z_DIST, to_ball_y)
        prediction_struct = self.get_ball_prediction_struct()
        ball_intercept = prediction_struct.slices[0]
        to_enemy_goal_y = 1 if self.team == 0 else -1
        desired_car_y = -5200 * to_enemy_goal_y + self.DESIRED_OFFSET_FROM_OWN_GOAL_Y * to_enemy_goal_y
        self.render_horizontal_line(desired_car_y, self.MIN_HIGH_JUMP_Z)

        for i, next_intercept in zip(range(prediction_struct.num_slices), prediction_struct.slices):
            if i == 0 or i == 1:
                continue
            if (
                abs(next_intercept.physics.location.y - intercept_y) <
                abs(ball_intercept.physics.location.y - intercept_y)
            ):
                ball_intercept = next_intercept
            else:
                # Do not search further: we do not care about backboard bounces which might come closer.
                break
        ball_is_travelling_towards_line = ball_intercept.physics.velocity.y * to_ball_y < 0
        ball_intercept.physics.location.y = intercept_y

        other_defenders = [
            car for car in s.allies if (
                abs(car.pos[TO_ORANGE] - intercept_y) < 5.0 * BALL_RADIUS
                and abs(car.pos[TO_STATUE]) < 1.2*GOAL_CENTER_TO_POST
            )
        ]

        seconds_until_intercept = ball_intercept.game_seconds - game_tick_packet.game_info.seconds_elapsed
        seconds_until_jump = seconds_until_intercept - self.JUMP_BEFORE_INTERCEPT_SECONDS
        seconds_until_doge = seconds_until_intercept - self.DODGE_BEFORE_INTERCEPT_SECONDS

        state = self.State.GROUND
        if game_tick_packet.game_info.is_kickoff_pause and abs(car_y - desired_car_y) > BALL_RADIUS:
            state = self.State.KICKOFF
        elif car_obj.has_wheel_contact:
            if ball_is_travelling_towards_line:
                if abs(ball_intercept.physics.location.x) > GOAL_CENTER_TO_POST:
                    # TODO: Side defence.
                    state = self.State.IDLE
                elif ball_intercept.physics.location.z > self.MAX_HIGH_JUMP_Z:
                    state = self.State.IDLE
                elif ball_intercept.physics.location.z > self.MIN_HIGH_JUMP_Z:
                    if seconds_until_intercept < self.high_jump_before_intercept(ball_intercept.physics.location.z):
                        state = self.State.HIGH_JUMP
                    else:
                        state = self.State.HIGH_JUMP_GROUND
                else:
                    if seconds_until_intercept < self.JUMP_BEFORE_INTERCEPT_SECONDS:
                        state = self.State.JUMPING
                    else:
                        state = self.State.GROUND
            else:
                state = self.State.IDLE
        else:
            if not ball_is_travelling_towards_line:
                state = self.State.IDLE
            elif ball_intercept.physics.location.z > self.MIN_HIGH_JUMP_Z:
                state = self.State.HIGH_JUMP
            else:
                if car.location.z <= self.HEIGHT_BEFORE_DODGING:
                    state = self.State.JUMPING
                elif car_obj.double_jumped:
                    state = self.State.IDLE
                else:
                    state = self.State.DODGING

        # Flip to the ball if convenient
        if state == self.State.IDLE:
            near_future_ball = ball_pos(bisect_ball_prediction(prediction_struct, lambda slice: (
                slice.game_seconds - s.time < self.JUMP_BEFORE_INTERCEPT_SECONDS
            )))
            if (
                    dist(near_future_ball, s.car.pos) < self.MAX_CONVENIENT_JUMP_DISTANCE
                    and near_future_ball[TO_CEILING] < self.MAX_CONVENIENT_JUMP_HEIGHT
                    and abs(near_future_ball[TO_ORANGE]) < abs(s.car.pos[TO_ORANGE])
                    ):
                state = self.State.JUMPING

        # Don't over commit. Rely on team mates to jump for it if they're closer and in line wit the ball.
        if state in [self.State.HIGH_JUMP, self.State.JUMPING, self.State.HIGH_JUMP_GROUND]:
            if any((
                        dist(other.pos, s.ball.pos) < dist(s.ball.pos, s.car.pos)
                         and 0.0 < dot(normalize(s.ball.pos - s.car.pos), other.pos - s.car.pos) < dist(s.ball.pos, s.car.pos)
                    ) for other in other_defenders):
                state = self.State.IDLE

        assert state


        # De-noise state
        self.state_history_counter[self.state_history.popleft()] -= 1
        self.state_history_counter[state] += 1
        self.state_history.append(state)
        state = self.state_history_counter.most_common(1)[0][0]

        if state != self.State.DODGING:
            self.ticks_in_dodge_state = 0
        out = SimpleControllerState()

        def clamp_into_goal(desired_x):
            max_abs_x = GOAL_CENTER_TO_POST - 1.5*BALL_RADIUS
            return clamp(desired_x, -max_abs_x, max_abs_x)
        def forward_adjust(desired_x):
            return (
                # PD-controller
                1.0 * (desired_x - car.location.x) +
                0.3 * (0 - car.velocity.x)
            )

        def avoid_others_x(desired_x):
            for other in other_defenders:
                other_x = clamp_into_goal(other.pos[TO_STATUE])*1.001
                self_x = s.car.pos[TO_STATUE]
                if abs(other_x - desired_x) < abs(self_x - desired_x):
                    desired_x = other_x + copysign(BALL_RADIUS*4.0, self_x-other_x)

                # other_ratio = (desired_x - self_x) / (desired_x - other_x)
                # # if there is a car between us and the desired_x, then don't go through that car.
                # if 0 < other_ratio < 1:
                #     desired_x = other_x + copysign(BALL_RADIUS*4.0, self_x-other_x)
            return desired_x

        if state == self.State.GROUND:
            # Drive to ball_intercept.physics.location.x
            out.throttle = forward_adjust(clamp_into_goal(avoid_others_x(ball_intercept.physics.location.x)))
            out.boost = out.throttle > 200.

        elif state == self.State.JUMPING:
            out.jump = True

        elif state == self.State.DODGING:
            self.ticks_in_dodge_state += 1
            if self.ticks_in_dodge_state < 2:
                out.jump = False
            elif self.ticks_in_dodge_state == 3:
                dodge_point = lerp(
                    struct_vector3_to_numpy(ball_intercept.physics.location),
                    s.ball.pos,
                    .5  # fudge
                )
                out.pitch, out.yaw, out.roll = flip_towards(
                    s,
                    dodge_point
                )
                # out.roll = 1
                # out.pitch = -0.008*forward_adjust(ball_intercept.physics.location.x)
                # length = 1/sqrt(out.roll**2 + out.pitch**2)
                # out.roll *= length
                # out.pitch *= length
                # if out.roll < .4:
                #     # if we really need to go forwards/back, don't go sideways at all
                #      out.roll = 0
                out.jump = True
            else:
                out.jump = False

        elif state == self.State.KICKOFF:
            target_pos = Vec3(copysign(GOAL_CENTER_TO_POST*0.8, s.car.pos[0]), desired_car_y, 0)  # corner boost
            # self.renderer.draw_rect_3d(target_pos, 100, 100, True, self.renderer.pink(), True)
            forward = True
            out.steer = get_steer_towards(s, target_pos)
            out.throttle = -(s.car.pos[TO_ORANGE] - .99 * desired_car_y) + .1 * s.car.vel[TO_ORANGE]

        elif state == self.State.IDLE:
            desired_car_x = 0.4 * (
                    game_tick_packet.game_ball.physics.location.x +
                    .5 * game_tick_packet.game_ball.physics.velocity.x
                )

            # Back and forth to go to goal
            time_ratio = (game_tick_packet.game_info.seconds_elapsed) % 2
            is_close_x = abs(car.location.x - desired_car_x) < 50
            if time_ratio < .5:
                desired_car_x *= -1
            desired_car_x = avoid_others_x(desired_car_x)
            desired_car_x = clamp_into_goal(desired_car_x)
            out.throttle = forward_adjust(desired_car_x)


            # Try to stay aligned with the x axis.
            car_yaw = zero_centered_angle(car.rotation.yaw)
            to_desired_car_y = desired_car_y - car.location.y
            desired_yaw = min(pi/3, max(-pi/3, 0.004 * to_desired_car_y))
            if desired_car_x < car.location.x:
                desired_yaw *= -1

            car_yaw_vel = car.angular_velocity.z
            out.steer = ( # PD-controller
                5.0 * (desired_yaw - car_yaw) +
                0.1 * (0 - car_yaw_vel)
            )
            if out.throttle < 1:
                out.steer *= -1
            if not car_obj.has_wheel_contact:
                desired_forward = normalize(Vec3(
                    1,
                    .001 * (desired_car_y - car.location.y),
                    0
                ))
                out.pitch, out.yaw, out.roll = get_pitch_yaw_roll(
                    s.car,
                    desired_forward,
                )

            # asign stuff just for rendering.
            seconds_until_intercept = 2.5
            ball_intercept.physics.location.x = desired_car_x
            ball_intercept.physics.location.z = 100

        elif state == self.State.HIGH_JUMP:
            z = car.location.z
            out.jump = not (110 < z < 120)
            if car_obj.double_jumped:
                out.pitch = min(1, max(-1, 5*(.9-car.rotation.pitch)))

        elif state == self.State.HIGH_JUMP_GROUND:
            if seconds_until_intercept == 0:
                desired_vel_x = 0  # avoid division by zero
            else:
                desired_vel_x = (ball_intercept.physics.location.x - car.location.x) / seconds_until_intercept
                desired_vel_x *= 1.1
            out.throttle = (desired_vel_x - car.velocity.x)

        else:
            self.logger.warn(f'invalid state {state}')

        self.render_ball_intercept(ball_intercept, seconds_until_intercept, state)

        out.throttle = clamp(out.throttle, -1, 1)
        out.steer = clamp(out.steer, -1, 1)

        out.steer = clamp11(out.steer)
        out.throttle = clamp11(out.throttle)
        out.pitch = clamp11(out.pitch)
        out.yaw = clamp11(out.yaw)
        out.roll = clamp11(out.roll)
        out.jump = bool(out.jump)
        out.boost = bool(out.boost)
        out.handbrake = bool(out.handbrake)
        return out

    def render_ball_intercept(self, ball_intercept: BallPredictionSlice, seconds_until_intercept: float, state):
        self.renderer.begin_rendering('block intercept')
        location = ball_intercept.physics.location
        def vec3(x,z):
            return (
                location.x + x,
                location.y,
                location.z + z
            )

        # Draw a stop-sign octagon with an indication of time.
        sides = 8
        r = 40
        for i in range(sides):
            theta = tau*i/sides
            x = r * cos(theta) # center of polygon edge
            z = r * sin(theta)
            tx = -z * tan(.5*tau/sides) # tangent of edge
            tz = +x * tan(.5*tau/sides)
            outwards = 1 + max(0, 2 * (seconds_until_intercept - self.JUMP_BEFORE_INTERCEPT_SECONDS))
            x *= outwards
            z *= outwards
            self.renderer.draw_line_3d(
                vec3(x+tx, z+tz),
                vec3(x-tx, z-tz),
                self.renderer.red()
            )
            s = 1.1  # draw a slightly larger one as well.
            self.renderer.draw_line_3d(
                vec3(s*(x+tx), s*(z+tz)),
                vec3(s*(x-tx), s*(z-tz)),
                self.renderer.red()
            )

        # draw something in the center
        if state == self.State.GROUND: color_func = self.renderer.grey
        elif state == self.State.JUMPING: color_func = self.renderer.red
        elif state == self.State.DODGING: color_func = self.renderer.white
        elif state == self.State.IDLE: color_func = self.renderer.green
        elif state == self.State.HIGH_JUMP_GROUND: color_func = lambda: self.renderer.create_color(255, 70, 70, 255)
        elif state == self.State.HIGH_JUMP: color_func = self.renderer.blue
        else: color_func = self.renderer.orange
        self.renderer.draw_rect_3d(location, 10, 10, True, color_func(), True)
        self.renderer.end_rendering()

    def render_horizontal_line(self, y, z, x_min=-900, x_max=900):
        self.renderer.begin_rendering()
        self.renderer.draw_line_3d([x_min, y+ 0, z], [x_max, y+ 0, z], self.renderer.create_color(200, 10, 255, 10))
        self.renderer.draw_line_3d([x_min, y+20, z], [x_max, y+20, z], self.renderer.create_color(200, 10, 255, 10))
        self.renderer.draw_line_3d([x_min, y-20, z], [x_max, y-20, z], self.renderer.create_color(200, 10, 255, 10))
        self.renderer.end_rendering()


if __name__ == '__main__':
    quick_bot_check(Defendinator('blah', 0, 0))
