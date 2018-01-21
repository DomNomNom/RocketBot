from utils import main, sanitize_output_vector, EasyGameState
if __name__ == '__main__':
    main()  # blocking

import mimic_bot
import imp

def get_student():
    import student_agents
    imp.reload(student_agents)
    return student_agents.DriveToPosAndVel(
        target_pos=[0,0],
        target_facing_dir=[1,0],
    )

class Agent(mimic_bot.Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.student = get_student()
        self.frames_until_scoring = 1

    def on_mimic_reset(self):
        self.frames_until_scoring = 1

    # Override
    def decide_on_action(self, action_dict, time_in_history, game_tick_packet):
        if self.frames_until_scoring:
            self.frames_until_scoring -= 1
            return [0]*8

        return sanitize_output_vector(self.student.get_output_vector(EasyGameState(game_tick_packet, self.index)))
