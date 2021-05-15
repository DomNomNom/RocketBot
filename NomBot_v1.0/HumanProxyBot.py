from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

from NomBot.utils import main, graduate_student_into_agent

from inputs import get_gamepad
import threading

def deadzone(normalized_axis):
    if abs(normalized_axis) < 0.1: return 0.0
    return normalized_axis

class ControllerInput:
    def __init__(self):
        # self.fThrottle = 0.0
        self._gas_pedal = 0.0
        self._brake_pedal = 0.0

        self.fSteer = 0.0
        self.fPitch = 0.0
        self.fYaw = 0.0
        self.fRoll = 0.0
        self.bJump = 0
        self.bBoost = 0
        self.bHandbrake = 0

        self.hat_toggle_north = False
        self.hat_toggle_south = False
        self.hat_toggle_west = False
        self.hat_toggle_east = False
    @property
    def fThrottle(self):
        return self._gas_pedal - self._brake_pedal

    def main_poll_loop(self):
        while 1:
            events = get_gamepad()  # Blocking
            for event in events:
                if False: pass
                elif event.ev_type=='Absolute' and event.code=='ABS_RZ': self._gas_pedal = deadzone(event.state / 255.0)
                elif event.ev_type=='Absolute' and event.code=='ABS_Z': self._brake_pedal = deadzone(event.state / 255.0)
                elif event.ev_type=='Absolute' and event.code=='ABS_X': self.fSteer = self.fYaw = deadzone(event.state / 32768.0)
                elif event.ev_type=='Absolute' and event.code=='ABS_Y': self.fPitch = deadzone(-event.state / 32768.0)
                elif event.ev_type=='Key' and event.code=='BTN_THUMBL': self.fRoll = -event.state
                elif event.ev_type=='Key' and event.code=='BTN_SOUTH': self.bJump = event.state
                elif event.ev_type=='Key' and event.code=='BTN_TR': self.bBoost = event.state
                elif event.ev_type=='Key' and event.code=='BTN_WEST': self.bHandbrake = event.state
                elif event.ev_type=='Absolute' and event.code=='ABS_HAT0Y' and event.state==-1: self.hat_toggle_north = not self.hat_toggle_north
                elif event.ev_type=='Absolute' and event.code=='ABS_HAT0Y' and event.state==+1: self.hat_toggle_south = not self.hat_toggle_south
                elif event.ev_type=='Absolute' and event.code=='ABS_HAT0X' and event.state==-1: self.hat_toggle_west = not self.hat_toggle_west
                elif event.ev_type=='Absolute' and event.code=='ABS_HAT0X' and event.state==+1: self.hat_toggle_east = not self.hat_toggle_east

controller = ControllerInput()
threading.Thread(target=controller.main_poll_loop, daemon=True).start()

class HumanStudent:
    def __init__(self):
        self.controller = controller
    def get_output_vector(self, s):
        return (
            round(self.controller.fThrottle),
            round(self.controller.fSteer),
            round(self.controller.fPitch),
            round(self.controller.fYaw),
            round(self.controller.fRoll),
            round(self.controller.bJump),
            round(self.controller.bBoost),
            round(self.controller.bHandbrake),
        )



class NomBot(BaseAgent):
    v3_agent = None

    def initialize_agent(self):
        self.controller_state = SimpleControllerState()
        self.v3_agent = graduate_student_into_agent(HumanStudent, self.team, self.index)

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        game_tick_packet = self.convert_packet_to_v3(packet)
        self.controller_state = self.convert_output_to_v4(self.v3_agent.get_output_vector(game_tick_packet))
        return self.controller_state
