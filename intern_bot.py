from utils import main, sanitize_output_vector, EasyGameState, MAX_CAR_SPEED, BALL_RADIUS
if __name__ == '__main__':
    main()  # blocking

from quicktracer import trace

import student_agents


class Agent(object):
    def __init__(self, name, team, index):
        self.name = name
        self.team = team
        self.index = index
        self.student = student_agents.CompositeStudent()

    def get_output_vector(self, game_tick_packet):
        s = EasyGameState(game_tick_packet, self.team, self.index)
        return sanitize_output_vector(self.student.get_output_vector(s))
