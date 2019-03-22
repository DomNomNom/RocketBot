from typing import List, Callable
from dataclasses import dataclass, field
import inspect
import re

from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

def trace(message: str):
    '''
    Puts the message on the screen MAGIC STACK HACKING >:D
    Kinda like quicktracer but RLBot specific.
    '''

    # Hijack render call to find the frame where get_output() is being called.
    # Then store messages in a wrapper for get_output().
    outer_frames = inspect.getouterframes(inspect.currentframe())
    trace_wrapper = None
    for frame in outer_frames:
        f_locals = frame.frame.f_locals
        agent = f_locals.get('self', None)
        if not isinstance(agent, BaseAgent):
            continue
        if not isinstance(agent.get_output, TraceWrapper):
            agent.get_output = TraceWrapper(agent, agent.get_output)
        trace_wrapper = agent.get_output
    assert trace_wrapper, 'trace() must be called from within a bot.'

    code = inspect.getframeinfo(outer_frames[1].frame).code_context[0].strip()
    match = re.search(r'trace\((.*)\)', code)
    if match:
        code = match.group(1)
    trace_wrapper.messages.append(f'{code} {message}')


@dataclass
class TraceWrapper:
    agent: BaseAgent
    real_get_output_func: Callable[[GameTickPacket], SimpleControllerState]
    messages: List[str] = field(default_factory=list)

    def __call__(self, packet: GameTickPacket) -> SimpleControllerState:
        agent = self.agent
        out = self.real_get_output_func(packet)
        agent.renderer.begin_rendering(group_id=f'nombotutils_trace_{str(agent.index)}')
        draw_pos = packet.game_cars[agent.index].physics.location
        for i, message in enumerate(self.messages):
            agent.renderer.draw_string_2d(300, 100 + 60 * i, 3, 3, message, agent.renderer.blue())
        self.messages = []
        agent.renderer.end_rendering()
        return out
