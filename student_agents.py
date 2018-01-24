from utils import * #main, mag, normalize, vec2angle, rotate90degrees, closest180, clamp, clamp01, clamp11, lerp, tau, URotationToRadians, cross, UP, estimate_turn_radius
if __name__ == '__main__':
    main()

import numpy as np

# from quicktracer import trace

from importlib.machinery import SourceFileLoader
quicktracer = SourceFileLoader("module.name", r"C:\Users\dom\Documents\GitHub\quicktracer\quicktracer\__init__.py").load_module()
trace = quicktracer.trace




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
    trace(tangent_forward_amount)

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


def turn_to_pos(s, target_pos):
    towards_target = target_pos - s.car.pos
    target_on_car_plane = np.array([
        s.car.forward.dot(towards_target),
        s.car.right.dot(towards_target),
    ])
    angle = vec2angle(target_on_car_plane)
    steer = angle*2.0
    trace(towards_target.dot(s.car.up))
    trace(steer)
    # steer = clamp11(steer)
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

def drive_to_pos_vel(s, target_pos, target_vel):
    boost = 1 # TODO
    car_to_target = target_pos - s.car.pos
    target_facing = normalize(target_vel)
    if car_to_target.dot(s.car.forward) <= 0:
        # trace(car_to_target.dot(s.car.forward))
        trace(target_vel.dot(s.car.right))
        steer = STEER_R if target_vel.dot(s.car.right) > 0 else STEER_L
        # steer = STEER_R
        # target is behind us. turn towards the opposite side of target_vel
        out_vec = [
            1,  # fThrottle
            steer,  # fSteer
            0,  # fPitch
            0,  # fYaw
            0,  # fRoll
            0,  # bJump
            boost,  # bBoost
            0,  # bHandbrake
        ]
        return out_vec

    target_right = cross(target_facing, UP)
    approach_speed = mag(target_vel)  # TODO: Can we go that fast?
    turn_radius = estimate_turn_radius(approach_speed)
    turn_center_right = target_pos + target_right * turn_radius
    turn_center_left = target_pos + target_right * turn_radius
    # Try to face in target_vel when there
    return [0]*8
    return stop_if_close(s, target_pos) or turn_to_pos(s, target_pos)

class DriveToPosAndVel(StudentAgent):
    def __init__(self, target_pos, target_facing_dir):
        self.target_pos = target_pos
        self.target_facing_dir = target_facing_dir
    def get_output_vector(self, s):
        return drive_to_pos_vel(s, np.array([-1000,0,10]), np.array([1000, 0, 0]))
        return turn_to_pos(s, np.array([-3600,0,0]))
        return [-1] + [0]*7




def tangentPointLineCircles(c, p, side):
    #************************************************
    # Input - c circle object
    #         p point object of focus tangent line
    #         side which side
    # Return  tangent point on the circle 0 or 1
    # http://www.ambrsoft.com/TrigoCalc/Circles2/Circles2Tangent_.htm
    #************************************************
    pTangent = new point(0, 0)
    dis = (p.x - c.a)**2 + (p.y - c.b)**2 - c.r**2  # point to circle surface distance

    if dis >= 0:
        dis = Math.sqrt(dis)

        sign = 1 if side == 0 else -1
        pTangent.x = (c.r**2 * (p.x - c.a) + sign * c.r * (p.y - c.b) * dis) / ((p.x - c.a)**2 + (p.y - c.b)**2) + c.a
        pTangent.y = (c.r**2 * (p.y - c.b) - sign * c.r * (p.x - c.a) * dis) / ((p.x - c.a)**2 + (p.y - c.b)**2) + c.b
        return pTangent
    else
        return None
