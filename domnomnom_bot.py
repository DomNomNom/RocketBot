if __name__ == '__main__':
    import sys
    import os
    os.chdir(r'C:\Users\dom\Documents\GitHub\RLBot\\')
    os.system('python runner.py')
    sys.exit()


import math
import numpy as np
from controller_input import controller
# from quicktracer import trace
from importlib.machinery import SourceFileLoader
quicktracer = SourceFileLoader("module.name", r"C:\Users\dom\Documents\GitHub\quicktracer\quicktracer\__init__.py").load_module()
trace = quicktracer.trace


def rotate90degrees(xy):
    return np.array([-xy[1], xy[0]])

UCONST_Pi = 3.1415926
URotation180 = float(32768)
URotationToRadians = UCONST_Pi / URotation180



class Agent:
    def __init__(self, name, team, index):
        self.name = name
        self.team = team  # 0 towards positive goal, 1 towards negative goal.
        self.index = index


    def get_output_vector(self, game_tick_packet):

        # return [
        #     1.0,    # fThrottle
        #     1,   # fSteer
        #     0.0,    # fPitch
        #     0.0,    # fYaw
        #     0.0,    # fRoll
        #     0,      # bJump
        #     0,      # bBoost
        #     0       # bHandbrake
        # ]


        # trace(controller.fThrottle)
        # trace(controller.fSteer)
        # trace(controller.fYaw)

        player_pos = np.array([
            game_tick_packet.gamecars[self.index].Location.X,
            game_tick_packet.gamecars[self.index].Location.Y,
        ])
        player_vel = np.array([
            game_tick_packet.gamecars[self.index].Velocity.X,
            game_tick_packet.gamecars[self.index].Velocity.Y,
        ])
        pitch = URotationToRadians * float(game_tick_packet.gamecars[self.index].Rotation.Pitch)
        yaw = URotationToRadians * float(game_tick_packet.gamecars[self.index].Rotation.Yaw)
        player_facing = np.array([
            math.cos(pitch) * math.cos(yaw),
            math.cos(pitch) * math.sin(yaw)
        ])
        player_right = -rotate90degrees(player_facing)

        # score should be positive if going counter clockwise
        target_pos = np.array([0, 0])
        circle_forward = rotate90degrees(player_pos - target_pos)
        drifting_score = player_vel.dot(player_right)
        going_around_target_score = circle_forward.dot(player_vel)
        score = going_around_target_score * drifting_score

        steer = controller.fSteer
        steer = round(steer)

        trace(drifting_score)
        trace(-steer)
        trace(player_pos)

        return [
            controller.fThrottle,
            steer,
            controller.fPitch,
            controller.fYaw,
            controller.fRoll,
            controller.bJump,
            controller.bBoost,
            controller.bHandbrake,
        ]

