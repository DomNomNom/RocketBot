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
import imp

# bakkes = SourceFileLoader("module.name", os.path.dirname(os.path.realpath(__file__)) + "\\bakkes.py").load_module()

# from quicktracer import trace
quicktracer = SourceFileLoader("module.name", r"C:\Users\dom\Documents\GitHub\quicktracer\quicktracer\__init__.py").load_module()
trace = quicktracer.trace


import historian
import bakkes
# import reloader
from controller_input import controller
import ctype_utils

imp.reload(ctype_utils)
imp.reload(historian)
imp.reload(bakkes)

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
        self.last_time = None
        self.history = historian.History()
        self.history.load(descriptor_file_path=None)  # latest
        self.mimic_start_time = -1



    def clear_recording(self, time):
        # self.actions = {}  # time since recording begin -> controller_state
        self.record_start_time = time
        self.history = historian.History()

    def get_output_vector(self, game_tick_packet):
        time = game_tick_packet.gameInfo.TimeSeconds

        if self.last_time is not None:
            trace(time - self.last_time)
        self.last_time = time

        # State transition
        if self.state == STATE_MIMIC and controller.hat_toggle_west:  # mimic -> record
            self.clear_recording(time)
            self.state = STATE_RECORD
        if self.state == STATE_RECORD and not controller.hat_toggle_west:  # record -> mimic
            self.history.save()
            self.mimic_start_time = -1
            print('recording finished. ')
            self.state = STATE_MIMIC


        if self.state == STATE_MIMIC:
            return self.mimic(time, game_tick_packet)
        elif self.state == STATE_RECORD:
            return self.record(time, game_tick_packet)


    def mimic(self, time, game_tick_packet):
        action_dict = self.history.get_action_dict()
        keyframe_timestamps = sorted(action_dict.keys())
        if len(keyframe_timestamps) < 2:
            return [0] * 8  # No action
        replay_duration = max(keyframe_timestamps)
        if time - self.mimic_start_time > replay_duration:
            bakkes_reset_command = bakkes.convert_tick_packet_to_command(
                self.history.get_closest_game_tick_packet(self.history.start_time)
            )
            bakkes.rcon(bakkes_reset_command)
            self.mimic_start_time = time
        replay_time = time - self.mimic_start_time
        key = max([keyframe_timestamps[0]] + [ t for t in keyframe_timestamps if t <= replay_time ])
        player_input = action_dict[key]

        expected_state = self.history.get_closest_game_tick_packet(replay_time)
        packet_diff = ctype_utils.struct_rms_deviation(expected_state, game_tick_packet, mask={
            'gameball': all,
            'gamecars': all,
        })
        # def diff(mask):
        #     deviation = ctype_utils.struct_rms_deviation(expected_state, game_tick_packet, mask=mask)
        #     return min(deviation, 1000000)
        # trace(diff({'gamecars': {0: {'AngularVelocity': all, }}}))
        # trace(diff({'gamecars': {0: {'Velocity': all, }}}))
        # trace(diff({'gamecars': {0: {'Location': all, }}}))
        # trace(packet_diff)

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


