from utils import main, EasyGameState, clamp01, clamp11
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
import slicer

imp.reload(ctype_utils)
imp.reload(historian)
imp.reload(bakkes)

STATE_RECORD = 'record'
STATE_MIMIC = 'mimic'

def player_input_to_vector(player_input):
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

class Agent:
    def __init__(self, name, team, index):
        self.name = name
        self.team = team
        self.index = index
        self.state = STATE_MIMIC
        # controller.hat_toggle_west = True  # self.state = STATE_RECORD
        self.clear_recording(0.0)
        self.first_time = True
        self.last_time = None
        self.should_reset_mimic = True
        self.history = historian.History()
        self.history.load(descriptor_file_path=None)  # latest
        self.mimic_start_time = -1
        self.slicer = slicer.Slicer()
        self.init_slicer(self.slicer, self.history)

    def init_slicer(self, slicer, history):
        actions = history.get_action_dict()
        def diff_index(player_input_1, player_input_2):
            player_input_1 = player_input_to_vector(player_input_1)
            player_input_2 = player_input_to_vector(player_input_2)
            for i,(v1, v2) in enumerate(zip(player_input_1, player_input_2)):
                if v1 != v2:
                    return i
            return -1
        keys = sorted(actions.keys())
        positions = [[keys[0],-1]] + [
            [key, diff_index(actions[key], actions[keys[i_minus_one]])]
            for i_minus_one, key in enumerate(keys[1:])
        ]

        def set_history_start_end(slicer_min, slicer_max):
            self.history.start_time = slicer_min
            self.history.end_time = slicer_max
            self.mimic_start_time = -1


        slicer.set_positions(positions)
        slicer.set_min_max(history.start_time, history.end_time)
        slicer.register_min_max_callback(set_history_start_end)

    def retire(self):
        self.slicer.close_window()

    def clear_recording(self, time):
        # self.actions = {}  # time since recording begin -> controller_state
        self.record_start_time = time
        self.history = historian.History()

    def get_output_vector(self, game_tick_packet):
        time = game_tick_packet.gameInfo.TimeSeconds


        # if self.last_time is not None:
        #     trace(time - self.last_time)
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
        if len(action_dict) < 2:
            return [0] * 8  # No action
        replay_duration = self.history.end_time - self.history.start_time

        time_in_history = time - self.mimic_start_time + self.history.start_time
        if time - self.mimic_start_time > replay_duration or self.should_reset_mimic:
            self.should_reset_mimic = False
            bakkes_reset_command = bakkes.convert_tick_packet_to_command(
                self.history.get_closest_game_tick_packet(self.history.start_time)
            )
            # bakkes.rcon(bakkes_reset_command)
            bakkes.rcon(';'.join([
                bakkes_reset_command,
                'ball location 0 0 0',
                'ball velocity -500 500 0',
                'ball angularvelocity 0 0 0',
            ]))  ## HAAX
            print('ball reset')
            self.mimic_start_time = time
            self.on_mimic_reset()
        return self.decide_on_action(action_dict, time_in_history, game_tick_packet)

    def on_mimic_reset(self):
        pass

    def decide_on_action(self, action_dict, time_in_history, game_tick_packet):
        keyframe_timestamps = sorted(action_dict.keys())
        key = max([keyframe_timestamps[0]] + [ t for t in keyframe_timestamps if t <= time_in_history ])
        trace(key)
        player_input = action_dict[key]
        return player_input_to_vector(player_input)


    def record(self, time, game_tick_packet):

        state = EasyGameState(game_tick_packet, self.index)
        trace(state.car.pos)

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


