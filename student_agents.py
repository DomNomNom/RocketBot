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
import time

TURN_RADIUS_WHILE_BOOSTING = 1100
TURN_RADIUS_WHILE_NON_BOOSTING = 550

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
        s.car.forward.dot(towards_target),
        s.car.right.dot(towards_target),
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
    trace(path, custom_display=TangentVisualizer)

    target_speed = mag(target_vel)  # TODO: Can we go that fast?
    return execute_tangent_path(s, path, target_speed)

def estimate_tangent_path_execution_time(s, path, target_speed):
    return get_length_of_tangent_path(path) / target_speed

BallInterceptPlan = namedtuple(
    'BallInterceptPlan',
    'start_time tangent_path tangent_path_duration ball_time ball_pos ball_vel ball_angular_vel target_speed'
)

def plan_from_prediction(s, prediction, target_vel):
    ball_pos, ball_vel, ball_angular_vel, ball_time = prediction
    # print (dt_milis)
    # intercept_time += dt_milis
    # print (intercept_time)
    # print (intercept_time)

    path = get_best_tangent_path(s, ball_pos, target_vel)
    if path is None:
        return None

    target_speed = mag(target_vel)
    tangent_path_duration = estimate_tangent_path_execution_time(s, path, target_speed)
    # tangent_path_duration *= 0.8
    return BallInterceptPlan(
        start_time=s.time,
        tangent_path=path,
        tangent_path_duration=tangent_path_duration,
        ball_time=s.time+ball_time,
        ball_pos=ball_pos,
        ball_vel=ball_vel,
        ball_angular_vel=ball_angular_vel,
        target_speed=target_speed,
    )

def plan_score(plan):
    return abs(plan.start_time + plan.tangent_path_duration - plan.ball_time)# + 7.* plan.tangent_path_duration
def get_ball_intercept_plan(s, target_vel, previous_plan=None):
    predicted_ball = Ball()
    plans = []
    intercept_time = 0
    time_0 = time.clock()
    ball_predict_duration = 4.0
    ball_path = marvin_atbab.predict_b(s.ball.pos, s.ball.vel, s.ball.angular_vel, ball_predict_duration)
    time_1 = time.clock()

    # num_random_samples = 24
    # prediction_sample_indexes = np.random.choice(len(ball_path), num_random_samples, replace=False)
    # prediction_sample_indexes = random_indexes + explore_close_indexes
    for prediction in ball_path[::10]:
        plans.append(plan_from_prediction(s, prediction, target_vel))

    plans = [p for p in plans if p is not None]
    if not plans:
        return None

    best_plan = min(plans, key=plan_score)
    time_2 = time.clock()
    # trace(time_diff)
    # trace(intercept_time)
    # trace(best_plan.start_time + best_plan.tangent_path_duration - best_plan.ball_time)
    # trace(best_plan.ball_time)
    # trace(time_1 - time_0)
    # trace(time_2 - time_1)
    return best_plan

def execute_intercept_plan(s, intercept_plan):
    # trace(intercept_plan.tangent_path, custom_display=TangentVisualizer)
    return execute_tangent_path(s, intercept_plan.tangent_path, intercept_plan.target_speed)


class DriveToPosAndVel(StudentAgent):
    def __init__(self, target_pos, target_vel):
        self.target_pos = target_pos
        self.target_vel = target_vel
    def get_output_vector(self, s):
        return drive_to_pos_vel(s, self.target_pos, self.target_vel)

class InterceptBallWithVel(StudentAgent):
    def __init__(self, target_vel):
        self.target_vel = target_vel
        # TODO: offset
    def get_output_vector(self, s):
        best_plan = get_ball_intercept_plan(s, self.target_vel)
        if not best_plan:
            return failsafe_output_vector(s)
        return execute_intercept_plan(s, best_plan)


class TheoreticalPhysicist(StudentAgent):
    ''' Just sits in an armchair and tries to predict the ball and figure out how good the prediction is '''
    def __init__(self):
        self.predictions = deque()
        self.predition_duration = 0.5 # s

    def get_output_vector(self, s):
        ball_path = marvin_atbab.predict_b(s.ball.pos, s.ball.vel, s.ball.angular_vel, self.predition_duration)
        predicted_ball = Ball()
        prediction = ball_path[-2]
        predicted_ball.pos, predicted_ball.vel, predicted_ball.angular_vel, _ = prediction

        # trace(s.time)
        self.predictions.append((
            s.time + self.predition_duration,
            predicted_ball,
            # basic_physics.predict_ball(s.ball, self.predition_duration),
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

