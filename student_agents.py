from utils import * #main, mag, normalize, vec2angle, rotate90degrees, closest180, clamp, clamp01, clamp11, lerp, tau, URotationToRadians, cross, UP, estimate_turn_radius
if __name__ == '__main__':
    main()

import numpy as np

from quicktracer import trace

from tangents import get_tangent_paths, get_length_of_tangent_path
from tangents_visualizer import TangentVisualizer

TURN_RADIUS_WHILE_BOOSTING = 1100
TURN_RADIUS_WHILE_NON_BOOSTING = 550

class StudentAgent(object):
    def get_output_vector(self, game_tick_packet):
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

def drive_to_pos_vel(s, target_pos, target_vel):
    boost = 1 # TODO
    # car_to_target = target_pos - s.car.pos
    # target_facing = normalize(target_vel)
    # if car_to_target.dot(s.car.forward) <= 0:
    #     # trace(car_to_target.dot(s.car.forward))
    #     trace(target_vel.dot(s.car.right))
    #     steer = STEER_R if target_vel.dot(s.car.right) > 0 else STEER_L
    #     # steer = STEER_R
    #     # target is behind us. turn towards the opposite side of target_vel
    #     out_vec = [
    #         1,  # fThrottle
    #         steer,  # fSteer
    #         0,  # fPitch
    #         0,  # fYaw
    #         0,  # fRoll
    #         0,  # bJump
    #         boost,  # bBoost
    #         0,  # bHandbrake
    #     ]
    #     return out_vec
    turn_radius_0 = estimate_turn_radius(mag(s.car.vel))
    turn_radius_1 = estimate_turn_radius(mag(target_vel))
    right_0 = normalize(clockwise90degrees(xy_only(s.car.vel)))
    right_1 = normalize(clockwise90degrees(xy_only(target_vel)))
    pos_0 = xy_only(s.car.pos)
    pos_1 = xy_only(target_pos)
    paths = get_tangent_paths(pos_0, turn_radius_0, right_0, pos_1, turn_radius_1, right_1)
    if not len(paths):
        print ('omg no tangent path! wtf')
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
    paths.sort(key=get_length_of_tangent_path)
    path = paths[0]
    # trace(2)
    trace(path, custom_display=TangentVisualizer)

    target_speed = mag(target_vel)  # TODO: Can we go that fast?
    # stop_if_close(s, Vec3(0,0,0))
    return execute_tangent_path(s, path, target_speed)


    target_right = cross(target_facing, UP)
    turn_radius = estimate_turn_radius(approach_speed)
    turn_center_right = target_pos + target_right * turn_radius
    turn_center_left = target_pos + target_right * turn_radius
    # Try to face in target_vel when there
    return [0]*8
    return stop_if_close(s, target_pos) or drive_to_pos(s, target_pos)

class DriveToPosAndVel(StudentAgent):
    def __init__(self, target_pos, target_vel):
        self.target_pos = target_pos
        self.target_vel = target_vel
    def get_output_vector(self, s):
        return drive_to_pos_vel(s, self.target_pos, self.target_vel)


