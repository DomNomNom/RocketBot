from typing import Callable

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlbot.utils.structures.ball_prediction_struct import Slice, BallPrediction

from NomBot.utils import EasyGameState, Ball
from NomBot.vector_math import *
from NomBot.constants import *

def get_steer_towards(s, target_pos):
    towards_target = target_pos - s.car.pos
    target_on_car_plane = np.array([
        dot(towards_target, s.car.forward),
        dot(towards_target, s.car.right),
    ])
    angle = vec2angle(target_on_car_plane)
    steer = angle*2.0
    return steer

def get_pitch_yaw_roll(s, forward, up=UP):
    car = s.car
    forward = normalize(forward)
    desired_facing_angular_vel = -cross(car.forward, forward)
    desired_up_angular_vel = -cross(car.up, up)

    pitch = dot(desired_facing_angular_vel, car.right)
    yaw = -dot(desired_facing_angular_vel, car.up)
    roll = dot(desired_up_angular_vel, car.forward)

    pitch_vel =  dot(car.angular_vel, car.right)
    yaw_vel   = -dot(car.angular_vel, car.up)
    roll_vel  =  dot(car.angular_vel, car.forward)

    # avoid getting stuck in directly-opposite states
    if dot(car.up, up) < -.8 and dot(car.forward, forward) > .8:#abs(roll_vel) < .3 and dot(car.up, up) < -.98 and dot(car.forward, forward) > .8:
        if roll == 0:
            roll = 1
        roll *= 1e10
    if dot(car.forward, forward) < -.8:
        if pitch == 0:
            pitch = 1
        pitch *= 1e10
    # TODO: do this for pitch too. (yaw not required)
    # trace(dot(car.up, up))
    # trace(pitch)
    # trace(roll)

    if dot(car.forward, forward) < 0.0:
        pitch_vel *= -1

    # PID control to stop overshooting.
    roll  = 3*roll  + 0.30*roll_vel
    yaw   = 3*yaw   + 0.70*yaw_vel
    pitch = 3*pitch + 0.90*pitch_vel

    # only start adjusting roll once we're roughly facing the right way
    if dot(car.forward, forward) < 0:
        roll = 0

    # To debug a single-axis
    # pitch = 0
    # yaw = 0
    # roll = 0
    return (pitch, yaw, roll)


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

def flip_in_direction(s, target_direction):
    '''
    returns a (pitch, yaw, roll) tuple which will make the car flip in the @target_direction

    Flip testing notes:
    I call it flipping, others call it air-rolling, but I dislike "rolling" in this context as it implies it's slow.
    Flipping will set your vertical speed to 0.
    "flip_forward" is the normalized direction for car.forward projected onto the horizontal plane.
    I define a "front flip" to be the flip where the cars nose turns towards the wheels.
    When front flipping, velocity is added in the flip_forward direction.
    This implies that if you pich your car up slightly more than 90degrees and front flip, you'll gain speed in the direction what used to be your backwards (before pitching up).
    car-orientation-roll does not affect flip_forward.
    '''
    target_direction = normalize(target_direction)
    flip_forward = normalize(z0(s.car.forward))
    flip_right = cross(flip_forward, UP)

    yaw = 0.0
    pitch = -dot(target_direction, flip_forward)
    roll = -dot(target_direction, flip_right)
    # pitch = 0
    return (pitch, yaw, roll)

def flip_towards(s, target_pos):
    '''
    Takes into account velocity
    '''
    towards_target_dir = normalize(target_pos - s.car.pos)
    # desired_speed =
    desired_vel = towards_target_dir * min(MAX_CAR_SPEED*1.05, s.car.speed + FLIP_SPEED_CHANGE*1.0)
    acceleration_dir = normalize(desired_vel - s.car.vel)
    return flip_in_direction(s, acceleration_dir)


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
                if s.time - self.last_time_of_jump_not_pressed > 0.5:  # avoid holding jump (note: delays jumping by one frame)
                    self.last_time_of_jump_not_pressed = s.time
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
