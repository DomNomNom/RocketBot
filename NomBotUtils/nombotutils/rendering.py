from threading import Thread
from typing import List, Callable
from dataclasses import dataclass, field
import inspect
import re
import time

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
            trace_wrapper = TraceWrapper(agent, agent.get_output)
            agent.get_output = trace_wrapper
            def clear_func():
                nonlocal trace_wrapper
                while time.time() - trace_wrapper.last_call_time < .1:
                    should_clear = True
                    time.sleep(0.1)
                trace_wrapper.agent.renderer.clear_screen(group_id=trace_wrapper.get_render_group())

            clear_thread = Thread(target=clear_func)
            clear_thread.start()
        else:
            trace_wrapper = agent.get_output
        break
    assert trace_wrapper, 'trace() must be called while having a bot on the stack.'

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
    last_call_time: float = field(default_factory=time.time)
    first_call: bool = True

    def __call__(self, packet: GameTickPacket) -> SimpleControllerState:
        self.last_call_time = time.time()
        agent = self.agent
        out = self.real_get_output_func(packet)

        # On first call, we accumulated messages twice as we the TraceWrapper
        # was not called since the original call webt straight into the real_get_output_func
        if self.first_call:
            self.messages = self.messages[:len(self.messages)//2]
            self.first_call = False

        agent.renderer.begin_rendering(group_id=self.get_render_group())
        draw_pos = packet.game_cars[agent.index].physics.location
        for i, message in enumerate(self.messages):
            agent.renderer.draw_string_2d(300, 100 + 60 * i, 3, 3, message, agent.renderer.blue())
        self.messages = []
        agent.renderer.end_rendering()

        return out

    def get_render_group(self) -> str:
        return f'nombotutils_trace_{str(self.agent.index)}'
