from utils import main, mag, normalize, vec2angle, rotate90degrees, closest180, clamp, clamp01, clamp11, lerp, tau, URotationToRadians
if __name__ == '__main__':
    main()  # blocking

import os
import math
import time
from collections import deque
from importlib.machinery import SourceFileLoader
import numpy as np

bakkes = SourceFileLoader("module.name", os.path.dirname(os.path.realpath(__file__)) + "\\bakkes.py").load_module()

from controller_input import controller
# from quicktracer import trace
quicktracer = SourceFileLoader("module.name", r"C:\Users\dom\Documents\GitHub\quicktracer\quicktracer\__init__.py").load_module()
trace = quicktracer.trace



STATE_RECORD = 'record'
STATE_MIMIC = 'mimic'

class Agent:
    def __init__(self, name, team, index):
        self.name = name
        self.team = team
        self.index = index
        self.state = STATE_MIMIC
        self.clear_recording(0.0)
        self.first_time = True

    def clear_recording(self, time):
        self.actions = {}  # time since recording begin -> controller_state
        self.record_start_time = time
        self.bakkes_reset_command = ''

    def get_output_vector(self, game_tick_packet):
        time = game_tick_packet.gameInfo.TimeSeconds
        controller_state = (
            round(controller.fThrottle),
            round(controller.fSteer),
            round(controller.fPitch),
            round(controller.fYaw),
            round(controller.fRoll),
            round(controller.bJump),
            round(controller.bBoost),
            round(controller.bHandbrake),
        )

        if self.state == STATE_MIMIC and controller.hat_toggle_west:  # mimic -> record
            self.clear_recording(time)
            self.bakkes_reset_command = bakkes.convert_tick_packet_to_command(game_tick_packet)
            self.state = STATE_RECORD
        if self.state == STATE_RECORD and not controller.hat_toggle_west:  # record -> mimic
            self.actions[time - self.record_start_time] = controller_state  # put in an end token
            self.state = STATE_MIMIC
            self.mimic_start_time = -1

            print ('recording finished: ')
            for key in self.get_keyframe_timestamps():
                print('{:02.2f}: {}'.format(key, repr(self.actions[key])))


        if self.state == STATE_MIMIC:
            return self.mimic(time)
        elif self.state == STATE_RECORD:
            return self.record(time, controller_state)

    def get_keyframe_timestamps(self):
        # TODO: cache this if I feel like it
        return sorted(self.actions.keys())

    def mimic(self, time):
        keyframe_timestamps = self.get_keyframe_timestamps()
        if len(keyframe_timestamps) < 2:
            return [0] * 8  # No action
        replay_duration = max(keyframe_timestamps)
        if time - self.mimic_start_time > replay_duration:
            bakkes.rcon(self.bakkes_reset_command)
            self.mimic_start_time = time
        replay_time = time - self.mimic_start_time
        key = max([keyframe_timestamps[0]] + [ t for t in keyframe_timestamps if t <= replay_time ])
        return self.actions[key]

    def record(self, time, controller_state):
        if self.first_time:
            print (self.bakkes_reset_command.replace(';', ';\n'))
            self.first_time = False


        last_controller_state = None
        if self.actions: last_controller_state = self.actions[max(self.get_keyframe_timestamps())]
        if controller_state != last_controller_state:
            self.actions[time - self.record_start_time] = controller_state
        return controller_state


