
import math
import numpy as np
from functools import reduce

from vector_math import *


UP = np.array([0.0, 0.0, 1.0])
UP.flags.writeable = False
STEER_R = +1
STEER_L = -1

MAX_CAR_TURN_SPEED = 2330.0

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

def estimate_turn_radius(car_speed):
    # https://docs.google.com/spreadsheets/d/1Hhg1TJqVUCcKIRmwvO2KHnRZG1z8K4Qn-UnAf5-Pt64/edit?usp=sharing
    # TODO: reverse speed?
    car_speed = clamp(car_speed, 0.0, MAX_CAR_TURN_SPEED)
    return (
        +156
        +0.1         * car_speed
        +0.000069    * car_speed**2
        +0.000000164 * car_speed**3
        -5.62E-11    * car_speed**4
    )

class Car(object):
    def __init__(self, gamecar):
        self.pos = stuct_vector3_to_numpy(gamecar.Location)
        self.vel = stuct_vector3_to_numpy(gamecar.Velocity)
        self.to_global_matrix = rotation_to_mat(gamecar.Rotation)
        self.forward = self.to_global_matrix.dot(np.array([1.0, 0.0, 0.0]))
        self.right   = self.to_global_matrix.dot(np.array([0.0, 1.0, 0.0]))
        self.up      = self.to_global_matrix.dot(np.array(UP))

# A wrapper for the game_tick_packet
class EasyGameState(object):
    def __init__(self, game_tick_packet, car_index):
        self.car = Car(game_tick_packet.gamecars[car_index])
        ball = game_tick_packet.gameball
        self.ball_pos = stuct_vector3_to_numpy(ball.Location)
        self.ball_vel = stuct_vector3_to_numpy(ball.Velocity)
        self.time = game_tick_packet.gameInfo.TimeSeconds

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
