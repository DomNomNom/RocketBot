
import math
import numpy as np
from functools import reduce

from .vector_math import *



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
    def __init__(self, game_car):
        physics = game_car.physics
        self.pos = struct_vector3_to_numpy(physics.location)
        self.vel = struct_vector3_to_numpy(physics.velocity)
        self.angular_vel = struct_vector3_to_numpy(physics.angular_velocity)
        self.boost = game_car.boost
        self.speed = mag(self.vel)
        self.to_global_matrix = rotation_to_mat(physics.rotation)
        self.forward = self.to_global_matrix.dot(np.array([1.0, 0.0, 0.0]))
        self.right   = self.to_global_matrix.dot(np.array([0.0, 1.0, 0.0]))
        self.up      = self.to_global_matrix.dot(np.array(UP))
        self.on_ground = game_car.has_wheel_contact
        self.jumped = game_car.jumped
        self.double_jumped = game_car.double_jumped

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
        self.pos = struct_vector3_to_numpy(ball.physics.location)
        self.vel = struct_vector3_to_numpy(ball.physics.velocity)
        self.angular_vel = struct_vector3_to_numpy(ball.physics.angular_velocity)

# A wrapper for the game_tick_packet
class EasyGameState(object):
    def __init__(self, game_tick_packet, team, car_index):
        game_cars = game_tick_packet.game_cars[:game_tick_packet.num_cars]
        self.car = Car(game_tick_packet.game_cars[car_index])
        self.opponents = [ Car(c) for c in game_cars if c.Team != team]
        self.allies = [ Car(c) for i,c in enumerate(game_cars) if c.Team == team and i!=car_index]
        self.ball = Ball(game_tick_packet.game_ball)
        self.time = game_tick_packet.game_info.seconds_elapsed
        self.enemy_goal_dir = 1.0 if team==0 else -1.0  # Which side of the Y axis the goal is.
        self.enemy_goal_center = Vec3(0,  self.enemy_goal_dir*5350, 200)
        self.own_goal_center   = Vec3(0, -self.enemy_goal_dir*5350, 200)
        self.is_kickoff_time = not game_tick_packet.game_info.is_kickoff_pause
