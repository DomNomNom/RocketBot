from utils import main, mag, normalize, vec2angle, rotate90degrees, closest180, clamp, clamp01, clamp11, lerp, tau, URotationToRadians, cross
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

def turn_to_pos(state, target_pos):
    towards_ball = target_pos - state.car_pos
    ball_on_car_plane = np.array([
        state.car_forward.dot(towards_ball),
        state.car_right.dot(towards_ball),
    ])
    angle = vec2angle(ball_on_car_plane)
    steer = angle*2.0
    trace(towards_ball.dot(state.car_up))
    trace(steer)
    # steer = clamp11(steer)
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
    return [0]*8
class DriveToPosAndVel(StudentAgent):
    def __init__(self, target_pos, target_facing_dir):
        self.target_pos = target_pos
        self.target_facing_dir = target_facing_dir
    def get_output_vector(self, state):
        return turn_to_pos(state, [-3600,0,0])
        return [-1] + [0]*7

