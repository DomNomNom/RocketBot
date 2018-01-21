
import math
import numpy as np
from functools import reduce

UCONST_Pi = 3.1415926
URotation180 = float(32768)
URotationToRadians = UCONST_Pi / URotation180
tau = 2*UCONST_Pi


cross = np.cross
def mag(vec):
    ''' magnitude/length of a vector '''
    return np.linalg.norm(vec)
def normalize(vec):
    magnitude = mag(vec)
    if not magnitude: return vec
    return vec / magnitude
def vec2angle(vec):
    return math.atan2(vec[1], vec[0])
def rotate90degrees(xy):
    return np.array([xy[1], -xy[0]])
def closest180(angle):
    return ((angle+UCONST_Pi) % (2*UCONST_Pi)) - UCONST_Pi
def clamp(x, bot, top):
    return min(top, max(bot, x))
def clamp01(x):
    return clamp(x, 0.0, 1.0)
def clamp11(x):
    return clamp(x, -1.0, 1.0)
def lerp(v0, v1, t):  # linear interpolation
  return (1 - t) * v0 + t * v1;

def sanitize_output_vector(output_vector):
    return [
        clamp11(output_vector[0]),  # fThrottle
        clamp11(output_vector[1]),  # fSteer
        clamp11(output_vector[2]),  # fPitch
        clamp11(output_vector[3]),  # fYaw
        clamp11(output_vector[4]),  # fRoll
        clamp01(output_vector[5]),  # bJump
        clamp01(output_vector[6]),  # bBoost
        clamp01(output_vector[7]),  # bHandbrake
    ]

def stuct_vector3_to_numpy(vec):
    return np.array([vec.X, vec.Y, vec.Z])

def rotation_to_mat(rotator):
    return to_rotation_matrix(
        URotationToRadians * rotator.Pitch,
        URotationToRadians * rotator.Yaw,
        URotationToRadians * rotator.Roll
    )

def to_rotation_matrix(pitch, yaw, roll):

    y=pitch
    cosy = math.cos(y)
    siny = math.sin(y)
    mat_pitch = np.array(
            [[cosy, 0, -siny],
             [0, 1, 0],
             [siny, 0, cosy]])

    z=yaw
    cosz = math.cos(z)
    sinz = math.sin(z)
    mat_yaw = np.array(
            [[cosz, -sinz, 0],
             [sinz, cosz, 0],
             [0, 0, 1]])

    x=roll
    cosx = math.cos(x)
    sinx = math.sin(x)
    mat_roll = np.array(
            [[1, 0, 0],
             [0, cosx, sinx],
             [0, -sinx, cosx]])

    return reduce(np.dot, [mat_yaw, mat_roll, mat_pitch])


# A wrapper for the game_tick_packet
class EasyGameState(object):
    def __init__(self, game_tick_packet, car_index):
        car = game_tick_packet.gamecars[car_index]
        self.car_pos = stuct_vector3_to_numpy(car.Location)
        self.car_vel = stuct_vector3_to_numpy(car.Velocity)
        self.car_to_global_matrix = rotation_to_mat(car.Rotation)
        self.car_forward = self.car_to_global_matrix.dot(np.array([1.0, 0.0, 0.0]))
        self.car_right   = self.car_to_global_matrix.dot(np.array([0.0, 1.0, 0.0]))
        self.car_up      = self.car_to_global_matrix.dot(np.array([0.0, 0.0, 1.0]))
        ball = game_tick_packet.gameball
        self.ball_pos = stuct_vector3_to_numpy(ball.Location)
        self.ball_vel = stuct_vector3_to_numpy(ball.Velocity)

def main():
    import sys
    import os
    os.chdir(r'C:\Users\dom\Documents\GitHub\NomBot')
    os.system('python doms_runner.py')
    sys.exit()

if __name__ == '__main__':
    main()

def to_left_handedness(rotation_matrix):
    return rotation_matrix
    # To change it from left to right or right to left, flip it like this:
    (( rx, ry, rz ),    ( ux, uy, uz ),    ( lx, ly, lz ),) = rotation_matrix
    return np.array([
        [rx, rz, ry],
        [lx, lz, ly],
        [ux, uz, uy],
    ])
