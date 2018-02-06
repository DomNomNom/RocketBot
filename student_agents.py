from utils import * #main, mag, normalize, vec2angle, rotate90degrees, closest180, clamp, clamp01, clamp11, lerp, tau, URotationToRadians, cross, UP, estimate_turn_radius
if __name__ == '__main__':
    main()

import numpy as np
from collections import deque, namedtuple

from quicktracer import trace

from tangents import get_tangent_paths, get_length_of_tangent_path
from tangents_visualizer import TangentVisualizer
from scorer import rms_deviation_from_diffs
# import basic_physics
import marvin_atbab
from marvin_atbab import BALL_STATE_POS, BALL_STATE_VEL, BALL_STATE_ANGULAR_VEL, BALL_STATE_TIME
import time


# Note: the variable `s` always refers to a EasyGameState here.

class StudentAgent(object):
    def get_output_vector(self, s):  # s - EasyGameState
        return [0]*8

def stop_if_close(s, target_pos, r=1000):
    if mag(s.car.pos - target_pos) > 1000:
        return None
    towards_target = target_pos - s.car.pos
    forward_speed = s.car.forward.dot(s.car.vel)
    tangent_forward_amount = towards_target.dot(s.car.forward)

    fThrottle = 0
    if forward_speed > 300:
        fThrottle = -1
    elif forward_speed < -300:
        fThrottle = 1
    else:
        # slowspeed
        fThrottle = tangent_forward_amount / 4000
    return [
        fThrottle,  # fThrottle
        0,  # fSteer
        0,  # fPitch
        0,  # fYaw
        0,  # fRoll
        0,  # bJump
        0,  # bBoost
        0,  # bHandbrake
    ]

def get_steer_towards(s, target_pos):
    towards_target = target_pos - s.car.pos
    target_on_car_plane = np.array([
        dot(towards_target, s.car.forward),
        dot(towards_target, s.car.right),
    ])
    angle = vec2angle(target_on_car_plane)
    steer = angle*2.0
    return steer


def drive_to_pos(s, target_pos):
    steer = get_steer_towards(s, target_pos)
    out_vec = [
        1,  # fThrottle
        steer,  # fSteer
        0,  # fPitch
        0,  # fYaw
        0,  # fRoll
        0,  # bJump
        1,  # bBoost
        0,  # bHandbrake
    ]
    return  out_vec

def steer_and_speed(s, steer, target_speed):
    # TODO: speed adjustment
    if s.car.speed < target_speed:
        return [
            1,  # fThrottle
            steer,  # fSteer
            0,  # fPitch
            0,  # fYaw
            0,  # fRoll
            0,  # bJump
            1,  # bBoost
            0,  # bHandbrake
        ]
    return [
        0,  # fThrottle
        steer,  # fSteer
        0,  # fPitch
        0,  # fYaw
        0,  # fRoll
        0,  # bJump
        0,  # bBoost
        0,  # bHandbrake
    ]


def execute_tangent_path(s, path, target_speed):
    pos = xy_only(s.car.pos)
    state = 0
    lookahead_time = 0.15  # seconds
    lookahead_dist = s.car.speed * lookahead_time
    if dist(pos, path.tangent_1) < lookahead_dist:
        state = 1
        steer = STEER_R if path.clockwise_1 else STEER_L
    elif dist(pos, path.tangent_0) < lookahead_dist:
        state = 2
        steer = get_steer_towards(s, z0(path.tangent_1))
    else:
        state = 3
        steer = STEER_R if path.clockwise_0 else STEER_L
        # target_speed = mag(s.car.vel)
        # target_speed = clamp(target_speed, 500, 999999)  # Kinda arbitrary values
    return steer_and_speed(s, steer, target_speed)  # TODO: maybe go faster in the mean time?


    # tangent_connection = path.tangent_1 - path.tangent_0
    # car_on_tangent = s.car.pos - path.tangent_0
    # tangent_forward_amount =

def failsafe_output_vector(s):
    return [
        1,  # fThrottle
        0,  # fSteer
        0,  # fPitch
        0,  # fYaw
        0,  # fRoll
        0,  # bJump
        0,  # bBoost
        0,  # bHandbrake
    ]

def get_best_tangent_path(s, target_pos, target_vel):
    turn_radius_0 = estimate_turn_radius(mag(s.car.vel))
    turn_radius_1 = estimate_turn_radius(mag(target_vel))
    right_0 = normalize(clockwise90degrees(xy_only(s.car.vel)))
    right_1 = normalize(clockwise90degrees(xy_only(target_vel)))
    pos_0 = xy_only(s.car.pos)
    pos_1 = xy_only(target_pos)
    paths = get_tangent_paths(pos_0, turn_radius_0, right_0, pos_1, turn_radius_1, right_1)
    if not len(paths):
        return None
    # paths.sort(key=get_length_of_tangent_path)
    path = min(paths, key=get_length_of_tangent_path)
    return path

def drive_to_pos_vel(s, target_pos, target_vel):
    boost = 1 # TODO

    path = get_best_tangent_path(s, target_pos, target_vel)
    if path is None:
        print ('omg no tangent path! wtf')
        return failsafe_output_vector(s)

    target_speed = mag(target_vel)  # TODO: Can we go that fast?
    return execute_tangent_path(s, path, target_speed)

def estimate_tangent_path_execution_time(s, path, target_speed):
    return get_length_of_tangent_path(path) / target_speed

BallInterceptPlan = namedtuple(
    'BallInterceptPlan',
    'start_time tangent_path tangent_path_duration ball_time ball_pos ball_vel ball_angular_vel target_vel'
)

def plan_from_ball_state(s, ball_state, target_vel):
    ball_pos, ball_vel, ball_angular_vel, ball_time = ball_state
    target_pos = ball_pos - 1.8*BALL_RADIUS * normalize(target_vel)
    path = get_best_tangent_path(s, target_pos, target_vel)
    if path is None:
        return None

    tangent_path_duration = estimate_tangent_path_execution_time(s, path, mag(target_vel))
    return BallInterceptPlan(
        start_time=s.time,
        tangent_path=path,
        tangent_path_duration=tangent_path_duration,
        ball_time=s.time+ball_time,
        ball_pos=ball_pos,
        ball_vel=ball_vel,
        ball_angular_vel=ball_angular_vel,
        target_vel=target_vel,
    )

def get_ball_path(s, prediction_duration):
    return marvin_atbab.predict_b(
        s.ball.pos,
        s.ball.vel,
        s.ball.angular_vel,
        prediction_duration,
    )

def plan_score(plan):
    return -abs(plan.start_time + plan.tangent_path_duration - plan.ball_time) #- 0.9* plan.tangent_path_duration
def get_ball_intercept_plan(s, get_target_vel, previous_plan=None):
    predicted_ball = Ball()
    plans = []
    intercept_time = 0
    time_0 = time.clock()
    ball_predict_duration = 4.0
    ball_path = get_ball_path(s, ball_predict_duration)
    time_1 = time.clock()

    ball_path_samples = []
    num_random_samples = 0
    if previous_plan is None:
        num_random_samples = 24
    else:
        num_random_samples = 5
        num_close_samples = 10
        prev_time = previous_plan.ball_time# - previous_plan.start_time
        closest_ball_states = sorted(ball_path, key=lambda ball_state: abs(s.time+ball_state[BALL_STATE_TIME] - prev_time))
        ball_path_samples.extend(closest_ball_states[:num_close_samples])
    ball_path_samples.extend([
        ball_path[i]
        for i in np.random.choice(len(ball_path), num_random_samples, replace=False)
    ])
    # trace(len(ball_path_samples))
    ball_path_samples = [
        ball_state for ball_state in ball_path_samples
        if ball_state[BALL_STATE_POS][-1] < 4 * BALL_RADIUS
    ]

    for ball_state in ball_path_samples:
        plans.append(plan_from_ball_state(s, ball_state, get_target_vel(s, ball_state)))
    # for ball_state in ball_path[::10]:
    #     plans.append(plan_from_ball_state(s, ball_state, target_vel))

    plans = [p for p in plans if p is not None]
    if not plans:
        return None

    best_plan = max(plans, key=plan_score)
    time_2 = time.clock()
    # trace(best_plan.ball_time - best_plan.start_time)
    # trace(time_diff)
    # trace(intercept_time)
    # trace(best_plan.start_time + best_plan.tangent_path_duration - best_plan.ball_time)
    # trace(best_plan.ball_time)
    # trace(time_1 - time_0)
    # trace(time_2 - time_1)
    return best_plan

def execute_intercept_plan(s, intercept_plan):
    return execute_tangent_path(s, intercept_plan.tangent_path, mag(intercept_plan.target_vel))


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
    if abs(roll_vel) < .3 and dot(car.up, up) < -.98 and dot(car.forward, forward) > .8:
        roll = 1.0
    # TODO: do this for pitch too. (yaw not required)
    # trace(dot(car.up, up))
    # trace(roll_vel)
    # trace(roll)

    # PID control to stop overshooting.
    roll  = 3*roll  + 0.25*roll_vel
    yaw   = 3*yaw   + 0.75*yaw_vel
    pitch = 3*pitch + 0.55*pitch_vel

    # only start adjusting roll once we're roughly facing the right way
    if dot(car.forward, forward) < 0:
        roll = 0

    # To debug a single-axis
    # pitch = 0
    # yaw = 0
    # roll = 0
    return (pitch, yaw, roll)

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
    desired_vel = towards_target_dir * min(MAX_CAR_SPEED, s.car.speed + FLIP_SPEED_CHANGE*1.0)
    acceleration_dir = normalize(desired_vel - s.car.vel)
    return flip_in_direction(s, acceleration_dir)

def useful_target_pos_v1(s):
    '''
    Returns a position that seems good to go towards right now.
    '''

    prediction_duration = 0.2
    ball_path = get_ball_path(s, prediction_duration)
    predicted_ball = ball_path[-1]
    predicted_ball_pos = predicted_ball[BALL_STATE_POS]
    target_ball_pos = s.enemy_goal_center
    # predicted_ball_pos = s.ball.pos
    # predicted_ball_vel = predicted_ball[BALL_STATE_VEL]
    to_goal_dir = normalize(target_ball_pos - predicted_ball_pos)

    # DONE: predict the ball by some small amount.
    # DONE: avoid ball when coming back
    # TODO: hit at an angle to change ball velocity
    target_pos = predicted_ball_pos - 0.8 * BALL_RADIUS * to_goal_dir

    avoid = 0
    if dist(s.car.pos, target_ball_pos) < dist(predicted_ball_pos, target_ball_pos):
        avoid = 1
        # TODO: make sure options are in bounds
        avoid_back_ball_radius = BALL_RADIUS * 5.0
        options = [
            predicted_ball_pos + avoid_back_ball_radius * normalize(Vec3( 1, -s.enemy_goal_dir, 0)),
            predicted_ball_pos + avoid_back_ball_radius * normalize(Vec3(-1, -s.enemy_goal_dir, 0)),
        ]
        best_avoid_option = min(options, key=lambda avoid_ball_pos: dist(s.car.pos, avoid_ball_pos))
        # TODO: factor in current velocity maybe
        target_pos = best_avoid_option

    # trace(avoid)
    # trace(s.ball.pos, view_box='game')
    # trace(s.car.pos, view_box='game')
    # trace(predicted_ball_pos, view_box='game')
    # trace(s.enemy_goal_center, view_box='game')
    # trace(s.own_goal_center, view_box='game')
    return target_pos

############################################################
# pure functions above.
# stateful classes below.
############################################################

class FlipTowardsBall(StudentAgent):
    def __init__(self):
        self.jumped_last_frame = False
        self.last_time_of_double_jump = 0.0
        self.last_time_of_ground_non_jump = 0.0
        self.last_time_in_air = 0.0
        self.last_time_on_ground = 0.0

    def get_target_pos(self, s):
        # return s.ball.pos
        return useful_target_pos_v1(s)

    def get_output_vector(self, s):

        if s.car.on_ground: self.last_time_on_ground = s.time
        else:               self.last_time_in_air    = s.time

        target_pos = self.get_target_pos(s)
        # target_pos = s.ball.pos

        out = [0]*8
        dir_to_target = normalize(target_pos - s.car.pos)
        # trace(mag(z0(s.car.vel)))


        vertical_to_ball = dot(target_pos - s.car.pos, UP)
        if s.car.on_ground:
            WAIT_ON_GROUND = 0.40 # Wait a bit to stabilize on the ground
            if s.time - self.last_time_in_air > WAIT_ON_GROUND:
                if s.time - self.last_time_of_ground_non_jump > 0.5:  # avoid holding jump (note: delays jumping by one frame)
                    self.last_time_of_ground_non_jump = s.time
                else:
                    out[OUT_VEC_JUMP] = 1
            else:
                # Drive to ball
                is_forward = dot(s.car.forward, dir_to_target)
                out[OUT_VEC_THROTTLE] = 6*is_forward
                if is_forward > 0.88: out[OUT_VEC_BOOST] = 1
                out[OUT_VEC_STEER] = get_steer_towards(s, target_pos)
        else:
            out[OUT_VEC_THROTTLE] = 1  # recovery from turtling
            if s.car.double_jumped:
                # pass
                if s.time - self.last_time_of_double_jump > 0.5:  # wait for the flip to mostly complete
                    (
                        out[OUT_VEC_PITCH],
                        out[OUT_VEC_YAW],
                        out[OUT_VEC_ROLL],
                    ) = get_pitch_yaw_roll(s, z0(dir_to_target))
            else:
                WAIT_ALTITUDE = 0.1
                if s.time - self.last_time_on_ground > WAIT_ALTITUDE:  # Wait for the car to have some altitude
                    (
                        out[OUT_VEC_PITCH],
                        out[OUT_VEC_YAW],
                        out[OUT_VEC_ROLL],
                    ) = flip_towards(s, target_pos) #flip_in_direction(s, target_pos - s.car.pos)
                    out[OUT_VEC_JUMP] = 1
                    self.last_time_of_double_jump = s.time
                elif s.time - self.last_time_on_ground < 0.5*WAIT_ALTITUDE:
                    out[OUT_VEC_JUMP] = 1

        self.jumped_last_frame = out[OUT_VEC_JUMP]
        return out

class AirStabilizerTowardsBall(StudentAgent):
    def __init__(self):
        pass
    def get_output_vector(self, s):
        target_pos = s.ball.pos
        out = [0]*8
        (
            out[OUT_VEC_PITCH],
            out[OUT_VEC_YAW],
            out[OUT_VEC_ROLL],
        ) = get_pitch_yaw_roll(s, normalize(target_pos - s.car.pos))
        return out
class AirStabilizerTowardsOwnGoal(StudentAgent):
    def __init__(self):
        self.jumped_last_frame = False
    def get_output_vector(self, s):
        target_pos = (s.own_goal_center + s.ball.pos) / 2.
        should_jump = s.car.on_ground and not self.jumped_last_frame
        self.jumped_last_frame = should_jump
        pitch, yaw, roll = get_pitch_yaw_roll(s, normalize(target_pos - s.car.pos))
        return [
            1,  # fThrottle
            0,  # fSteer
            pitch,  # fPitch
            yaw,  # fYaw
            roll,  # fRoll
            should_jump,  # bJump
            0,  # bBoost
            0,  # bHandbrake
        ]
        return output_vector

class HumanStudent(StudentAgent):
    def __init__(self):
        from controller_input import controller
        self.controller = controller
    def get_output_vector(self, s):
        return (
            round(self.controller.fThrottle),
            round(self.controller.fSteer),
            round(self.controller.fPitch),
            round(self.controller.fYaw),
            round(self.controller.fRoll),
            round(self.controller.bJump),
            round(self.controller.bBoost),
            round(self.controller.bHandbrake),
        )

class CompositeStudent(StudentAgent):
    def __init__(self):
        self.stablizer = AirStabilizerTowardsOwnGoal()
        self.hit_ball_into_goal = InterceptBallTowardsEnemyGoal()

    def get_sub_student(self, s):
        if not s.car.on_ground or dot(s.car.pos, UP) > 3.0*BALL_RADIUS:
            return self.stablizer
        else:
            return self.hit_ball_into_goal
    def get_output_vector(self, s):
        return self.get_sub_student(s).get_output_vector(s)

class DriveToPosAndVel(StudentAgent):
    def __init__(self, target_pos, target_vel):
        self.target_pos = target_pos
        self.target_vel = target_vel
    def get_output_vector(self, s):
        return drive_to_pos_vel(s, self.target_pos, self.target_vel)


class InterceptBallWithVel(StudentAgent):
    def __init__(self, target_vel):
        self.target_vel = target_vel
        self.best_plan = None
        # TODO: offset
    def get_target_vel(self, s, ball_state):
        return self.target_vel
    def get_output_vector(self, s):
        # trace(s.car.pos , view_box='game')
        # trace(s.ball.pos, view_box='game')

        # trace(mag(s.ball.angular_vel * BALL_RADIUS), view_box='vel')
        # trace(mag(s.ball.vel), view_box='vel')
        # trace(s.ball.vel)

        if not self.best_plan or self.should_recompute_plan(s, self.best_plan):
            self.best_plan = get_ball_intercept_plan(s, self.get_target_vel, previous_plan=self.best_plan)
            if not self.best_plan:
                return failsafe_output_vector(s)
            trace(self.best_plan.tangent_path, custom_display=TangentVisualizer, key='tangent', view_box='game')
            return execute_intercept_plan(s, self.best_plan)
        else:
            # Don't change plans if we're close.
            path = get_best_tangent_path(
                s,
                self.best_plan.tangent_path.pos_1,
                self.best_plan.target_vel
            )
            trace(path, custom_display=TangentVisualizer, key='tangent', view_box='game')
            return execute_tangent_path(s, path, mag(self.best_plan.target_vel))
            # trace(self.best_plan.tangent_path, custom_display=TangentVisualizer, view_box='game')
            # return drive_to_pos_vel(s, self.best_plan.tangent_path.pos_1, self.target_vel)

    def should_recompute_plan(self, s, plan):
        duration_until_hit =  plan.ball_time - s.time  #estimate_tangent_path_execution_time(s, self.best_plan.tangent_path, mag(self.target_vel))
        should_recompute = not (-.1 < duration_until_hit < 1.0)
        # trace(duration_until_hit, view_box='ETA')
        # trace(should_recompute, view_box='ETA')
        return should_recompute
class InterceptBallTowardsEnemyGoal(InterceptBallWithVel):
    def __init__(self):
        self.best_plan = None
    def get_target_vel(self, s, ball_state):
        return MAX_CAR_SPEED * normalize(s.enemy_goal_center - ball_state[BALL_STATE_POS])


class TheoreticalPhysicist(StudentAgent):
    ''' Just sits in an armchair and tries to predict the ball and figure out how good the prediction is '''
    def __init__(self):
        self.predictions = deque()
        self.prediction_duration = 0.5 # s

    def get_output_vector(self, s):
        ball_path = marvin_atbab.predict_b(s.ball.pos, s.ball.vel, s.ball.angular_vel, self.prediction_duration)
        predicted_ball = Ball()
        prediction = ball_path[-2]
        predicted_ball.pos, predicted_ball.vel, predicted_ball.angular_vel, _ = prediction

        # trace(s.time)
        self.predictions.append((
            s.time + self.prediction_duration,
            predicted_ball,
            # basic_physics.predict_ball(s.ball, self.prediction_duration),
        ))


        # evaluate predictions
        pred = None
        while len(self.predictions) and s.time >= self.predictions[0][0]:
            predict_time, predicted_ball = self.predictions.popleft()
            pred = predicted_ball
        if pred is not None:
            diff_pos = s.ball.pos - pred.pos
            diff_vel = s.ball.vel - pred.vel
            error = rms_deviation_from_diffs([
                diff_pos,
                diff_vel,
            ])
            # trace(error)
            trace(diff_pos[-1])
            trace(diff_vel[-1])

        return [0]*8

