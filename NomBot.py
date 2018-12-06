from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

from RocketBot.utils import main, graduate_student_into_agent
from RocketBot.student_agents import NomBot_v1

class PythonExample(BaseAgent):
    agent = None

    def initialize_agent(self):
        self.controller_state = SimpleControllerState()
        self.agent = graduate_student_into_agent(NomBot_v1, self.team, self.index)

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        game_tick_packet = self.convert_packet_to_v3(packet)
        self.controller_state = self.convert_output_to_v4(self.agent.get_output_vector(game_tick_packet))
        return self.controller_state
