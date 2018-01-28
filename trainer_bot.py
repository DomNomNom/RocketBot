from utils import main, sanitize_output_vector, EasyGameState, MAX_CAR_SPEED
if __name__ == '__main__':
    main()  # blocking

import imp
from quicktracer import trace

from vector_math import *
import mimic_bot
import student_agents
import scorer

NUM_FRAMES_TO_WAIT_FOR_BAKKES = 5

class Agent(mimic_bot.Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frames_until_scoring = NUM_FRAMES_TO_WAIT_FOR_BAKKES

    def on_mimic_reset(self):
        self.frames_until_scoring = NUM_FRAMES_TO_WAIT_FOR_BAKKES
        target_pos = Vec3(0,0,0)
        target_vel = Vec3(0,1,0) * MAX_CAR_SPEED
        target_pos -= target_vel * 0.05
        self.student = student_agents.DriveToPosAndVel(target_pos, target_vel)
        self.scorer = scorer.PosVelScorer(target_pos, target_vel)

    # Override
    def decide_on_action(self, action_dict, time_in_history, game_tick_packet):
        if self.frames_until_scoring:
            self.frames_until_scoring -= 1
            return [0]*8
        s = EasyGameState(game_tick_packet, self.index)
        self.scorer.update(s)
        trace(self.scorer.get_score())
        return sanitize_output_vector(self.student.get_output_vector(s))

