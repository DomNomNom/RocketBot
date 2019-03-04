from typing import Callable

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlbot.utils.structures.ball_prediction_struct import Slice, BallPrediction

from NomBot.utils import EasyGameState, Ball
from NomBot.vector_math import *
from NomBot.constants import *


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


class NomBot_1_5(BaseAgent):

    """
    Always filps.
    """

    def initialize_agent(self):
        self.controller_state = SimpleControllerState()

        # Variables about previous states.
        self.jumped_last_frame = False
        self.last_time_of_double_jump = 0.0
        self.last_time_of_jump_not_pressed = 0.0
        self.last_time_in_air = 0.0
        self.last_time_on_ground = 0.0


    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:

        s = EasyGameState(packet, self.team, self.index)

        self.maintain_historic_variables_pre(s)

        self.state_flip_towards_ball(s)

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
        self.maintain_historic_variables_post(s)
        return out

    def maintain_historic_variables_pre(self, s: EasyGameState):
        if s.car.on_ground: self.last_time_on_ground = s.time
        else:               self.last_time_in_air    = s.time

    def maintain_historic_variables_post(self, s: EasyGameState):
        self.jumped_last_frame = self.controller_state.jump
        if not self.controller_state.jump:
            self.last_time_of_jump_not_pressed = s.time



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
            lambda slice: slice.game_seconds < s.time+2.0
        )
        future_ball = Ball(predicted_slice.physics)
        predicted_ball_pos = predicted_slice.physics.location
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


        # trace(future_ball.pos, view_box='game')
        # trace(s.ball.pos, view_box='game')
        # trace(s.car.pos, view_box='game')
        # trace(s.enemy_goal_center, view_box='game')
        # trace(s.own_goal_center, view_box='game')
        # trace(100 * -normalize(desired_ball_vel_change), view_box='game')
        # trace(100 * -to_goal_dir, view_box='game')
        # trace(underestimated_time_to_ball)

        return target_pos


    def state_flip_towards_ball(self, s: EasyGameState):
        target_pos = self.get_target_pos(s)
        dir_to_target = normalize(target_pos - s.car.pos)

        out = self.controller_state
        out.jump = False
        out.boost = False
        out.handbrake = False

        vertical_to_ball = dot(target_pos - s.car.pos, UP)
        if s.car.on_ground:
            WAIT_ON_GROUND = 0.40 # Wait a bit to stabilize on the ground
            if s.time - self.last_time_in_air > WAIT_ON_GROUND:
                if s.time - self.last_time_of_ground_non_jump > 0.5:  # avoid holding jump (note: delays jumping by one frame)
                    self.last_time_of_ground_non_jump = s.time
                else:
                    out.jump = True
            else:
                # Drive to ball
                is_forward = dot(s.car.forward, dir_to_target)
                out.throttle = 6.*is_forward
                if is_forward > 0.88: out.boost = True
                out.steer = get_steer_towards(s, target_pos)
        else:
            out.throttle = 1.0  # recovery from turtling
            if s.car.double_jumped:
                desired_forward = z0(dir_to_target)
                if s.time - self.last_time_of_double_jump > 0.2:  # wait for the flip to mostly complete
                    (
                        out.pitch,
                        out.yaw,
                        out.roll,
                    ) = get_pitch_yaw_roll(s, desired_forward)
                if s.car.boost > 50 and dot(s.car.forward, desired_forward) > 0.95 and dist(target_pos, s.car.pos) > 500:
                    out.boost = True
            else:
                WAIT_ALTITUDE = 0.1
                if s.time - self.last_time_on_ground > WAIT_ALTITUDE:  # Wait for the car to have some altitude
                    (
                        out.pitch,
                        out.yaw,
                        out.roll,
                    ) = flip_towards(s, target_pos) #flip_in_direction(s, target_pos - s.car.pos)
                    out.jump = True
                    self.last_time_of_double_jump = s.time
                elif s.time - self.last_time_on_ground < 0.5*WAIT_ALTITUDE:
                    out.jump = True

        self.jumped_last_frame = out.jump
        return out


def quick_self_check():
    bot = NomBot_1_5('bot name', 0, 0)
    bot.initialize_agent()

    # Stub out a few things
    packet = GameTickPacket()
    packet.game_cars[0].physics.location.x = 100
    def get_ball_prediction_struct():
        ball_prediction = BallPrediction()
        ball_prediction.num_slices = 7
        return ball_prediction
    bot.get_ball_prediction_struct = get_ball_prediction_struct

    bot.get_output(packet)
    print ('quick_self_check passed.')


if __name__ == '__main__':
    quick_self_check()
