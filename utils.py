
import math
import numpy as np
from functools import reduce

from vector_math import *


UP = np.array([0.0, 0.0, 1.0])
UP.flags.writeable = False
STEER_R = +1
STEER_L = -1


BALL_RADIUS = 92.
MAX_CAR_SPEED = 2300.005

OUT_VEC_THROTTLE = 0
OUT_VEC_STEER = 1
OUT_VEC_PITCH = 2
OUT_VEC_YAW = 3
OUT_VEC_ROLL = 4
OUT_VEC_JUMP = 5
OUT_VEC_BOOST = 6
OUT_VEC_HANDBRAKE = 7

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
    car_speed = clamp(car_speed, 0.0, MAX_CAR_SPEED)
    return (
        # -10
        +156
        +0.1         * car_speed
        +0.000069    * car_speed**2
        +0.000000164 * car_speed**3
        -5.62E-11    * car_speed**4
    )

class Car(object):
    def __init__(self, gamecar):
        self.pos = struct_vector3_to_numpy(gamecar.Location)
        self.vel = struct_vector3_to_numpy(gamecar.Velocity)
        self.angular_vel = struct_vector3_to_numpy(gamecar.AngularVelocity)
        self.speed = mag(self.vel)
        self.to_global_matrix = rotation_to_mat(gamecar.Rotation)
        self.forward = self.to_global_matrix.dot(np.array([1.0, 0.0, 0.0]))
        self.right   = self.to_global_matrix.dot(np.array([0.0, 1.0, 0.0]))
        self.up      = self.to_global_matrix.dot(np.array(UP))
        self.on_ground = gamecar.bOnGround
        self.jumped = gamecar.bJumped
        self.double_jumped = gamecar.bDoubleJumped

class Ball(object):
    def __init__(self, ball=None):
        self.pos = Vec3(0,0,0)
        self.vel = Vec3(0,0,0)
        self.angular_vel = Vec3(0,0,0)
        if ball is None:
            return
        if isinstance(ball, Ball):
            self.pos = ball.pos
            self.vel = ball.vel
            return
        # c-struct
        self.pos = struct_vector3_to_numpy(ball.Location)
        self.vel = struct_vector3_to_numpy(ball.Velocity)
        self.angular_vel = struct_vector3_to_numpy(ball.AngularVelocity)

# A wrapper for the game_tick_packet
class EasyGameState(object):
    def __init__(self, game_tick_packet, team, car_index):
        self.car = Car(game_tick_packet.gamecars[car_index])
        self.ball = Ball(game_tick_packet.gameball)
        self.time = game_tick_packet.gameInfo.TimeSeconds
        enemy_goal_dir = 1.0 if team==0 else -1.0
        self.enemy_goal_center = Vec3(0,  enemy_goal_dir*5350, 200)
        self.own_goal_center   = Vec3(0, -enemy_goal_dir*5350, 200)

def main():
    import sys
    import os
    os.chdir(r'C:\Users\dom\Documents\GitHub\NomBot')
    os.system('python doms_runner.py')
    sys.exit()

if __name__ == '__main__':
    main()
