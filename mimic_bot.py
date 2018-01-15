from utils import main, mag, normalize, vec2angle, rotate90degrees, closest180, clamp, clamp01, clamp11, lerp, tau, URotationToRadians
if __name__ == '__main__':
    main()  # blocking

import math
import time
from collections import deque

import numpy as np

from controller_input import controller
# from quicktracer import trace
from importlib.machinery import SourceFileLoader
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

    def clear_recording(self, time):
        self.actions = {}  # time since recording begin -> controller_state
        self.record_start_time = time

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

        # TODO: cache the fuck outta this
        keyframe_timestamps = sorted(self.actions.keys())

        if self.state == STATE_MIMIC and controller.hat_toggle_west:
            self.clear_recording(time)
            self.state = STATE_RECORD
        if self.state == STATE_RECORD and not controller.hat_toggle_west:
            self.actions[time - self.record_start_time] = controller_state  # put in an end token
            self.state = STATE_MIMIC

            print ('recording finished: ')
            for key in keyframe_timestamps:
                print('{:02.2f}: {}'.format(key, repr(self.actions[key])))

        if self.state == STATE_MIMIC:
            if len(keyframe_timestamps) < 2:
                return [0] * 8  # No action
            replay_time = time % max(keyframe_timestamps)
            key = max([keyframe_timestamps[0]] + [ t for t in keyframe_timestamps if t <= replay_time ])
            return self.actions[key]


        elif self.state == STATE_RECORD:
            last_controller_state = None
            if self.actions: last_controller_state = self.actions[max(keyframe_timestamps)]
            if controller_state != last_controller_state:
                self.actions[time - self.record_start_time] = controller_state
            return controller_state


