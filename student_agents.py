from utils import main, mag, normalize, vec2angle, rotate90degrees, closest180, clamp, clamp01, clamp11, lerp, tau, URotationToRadians

from quicktracer import trace

TURN_RADIUS_WHILE_BOOSTING = 1100
TURN_RADIUS_WHILE_NON_BOOSTING = 550

class StudentAgent(object):
    def get_output_vector(self, game_tick_packet):
        return [0]*8

def turn_to_pos(state, target_pos):
    return [0]*8
class DriveToPosAndVel(StudentAgent):
    def __init__(self, target_pos, target_facing_dir):
        self.target_pos = target_pos
        self.target_facing_dir = target_facing_dir
    def get_output_vector(self, state):
        return turn_to_pos(state, [0,0,0])
        return [-1] + [0]*7

