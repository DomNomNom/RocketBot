
import math
import numpy as np

from .vector_math import *

from rlbot.utils.structures.game_data_struct import Physics

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
    def __init__(self, physics: Physics):
        self.pos = struct_vector3_to_numpy(physics.location)
        self.vel = struct_vector3_to_numpy(physics.velocity)
        self.angular_vel = struct_vector3_to_numpy(physics.angular_velocity)

# A wrapper for the game_tick_packet
class EasyGameState(object):
    def __init__(self, game_tick_packet, team, car_index):
        game_cars = game_tick_packet.game_cars[:game_tick_packet.num_cars]
        self.car = Car(game_tick_packet.game_cars[car_index])
        self.opponents = [ Car(c) for c in game_cars if c.team != team]
        self.allies = [ Car(c) for i,c in enumerate(game_cars) if c.team == team and i!=car_index]
        self.ball = Ball(game_tick_packet.game_ball.physics)
        self.time = game_tick_packet.game_info.seconds_elapsed
        self.enemy_goal_dir = 1.0 if team==0 else -1.0  # Which side of the Y axis the goal is.
        self.enemy_goal_center = Vec3(0,  self.enemy_goal_dir*5350, 200)
        self.own_goal_center   = Vec3(0, -self.enemy_goal_dir*5350, 200)
        self.is_kickoff_time = not game_tick_packet.game_info.is_kickoff_pause
        self.is_round_active = game_tick_packet.game_info.is_round_active
