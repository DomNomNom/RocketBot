from utils import main, mag, normalize, vec2angle, rotate90degrees, closest180, clamp, clamp01, clamp11, lerp, tau, URotationToRadians

if __name__ == '__main__':
    main()  # blocking

import os
import sys
import math
import time
from collections import deque
from importlib.machinery import SourceFileLoader
import numpy as np

# bakkes = SourceFileLoader("module.name", os.path.dirname(os.path.realpath(__file__)) + "\\bakkes.py").load_module()

from quicktracer import trace
# quicktracer = SourceFileLoader("module.name", r"C:\Users\dom\Documents\GitHub\quicktracer\quicktracer\__init__.py").load_module()
# trace = quicktracer.trace


import historian
import bakkes
from controller_input import controller


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
        self.action_dict = {}

    def clear_recording(self, time):
        # self.actions = {}  # time since recording begin -> controller_state
        self.record_start_time = time
        self.history = []

    def get_output_vector(self, game_tick_packet):
        time = game_tick_packet.gameInfo.TimeSeconds

        # State transition
        if self.state == STATE_MIMIC and controller.hat_toggle_west:  # mimic -> record
            self.clear_recording(time)
            self.bakkes_reset_command = bakkes.convert_tick_packet_to_command(game_tick_packet)
            self.state = STATE_RECORD
        if self.state == STATE_RECORD and not controller.hat_toggle_west:  # record -> mimic
            self.state = STATE_MIMIC
            self.mimic_start_time = -1
            self.action_dict = historian.to_action_dict(self.history)

            print ('recording finished. ')


        if self.state == STATE_MIMIC:
            return self.mimic(time)
        elif self.state == STATE_RECORD:
            return self.record(time, game_tick_packet)

    def mimic(self, time):
        keyframe_timestamps = sorted(self.action_dict.keys())
        if len(keyframe_timestamps) < 2:
            return [0] * 8  # No action
        replay_duration = max(keyframe_timestamps)
        if time - self.mimic_start_time > replay_duration:
            bakkes.rcon(self.bakkes_reset_command)
            self.mimic_start_time = time
        replay_time = time - self.mimic_start_time
        key = max([keyframe_timestamps[0]] + [ t for t in keyframe_timestamps if t <= replay_time ])
        player_input = self.action_dict[key]
        return [
            player_input.fThrottle,
            player_input.fSteer,
            player_input.fPitch,
            player_input.fYaw,
            player_input.fRoll,
            player_input.bJump,
            player_input.bBoost,
            player_input.bHandbrake,
        ]


    def record(self, time, game_tick_packet):
        time = time - self.record_start_time

        output_vector = (
            round(controller.fThrottle),
            round(controller.fSteer),
            round(controller.fPitch),
            round(controller.fYaw),
            round(controller.fRoll),
            round(controller.bJump),
            round(controller.bBoost),
            round(controller.bHandbrake),
        )

        history_item = historian.HistoryItem(
            float(time),
            # game_tick_packet,
            # output_vector,
        )
        history_item.output_vector = output_vector
        history_item.game_tick_packet = game_tick_packet
        self.history.append(history_item)
        # if self.first_time:
        #     print (history_item.encode())
        #     self.first_time = False

        return output_vector


