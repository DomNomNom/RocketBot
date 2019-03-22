from rlbot.agents.base_agent import BaseAgent
from rlbot.utils.structures.game_data_struct import GameTickPacket
from unittest.mock import MagicMock

def quick_bot_check(bot: BaseAgent):
    bot.initialize_agent()

    # Stub out a few things
    packet = GameTickPacket()
    packet.game_cars[0].physics.location.x = 100
    def get_ball_prediction_struct():
        ball_prediction = BallPrediction()
        ball_prediction.num_slices = 7
        return ball_prediction
    bot.get_ball_prediction_struct = get_ball_prediction_struct
    bot.renderer = MagicMock()
    bot.set_game_state = MagicMock()

    bot.get_output(packet)
    print('quick_bot_check() passed.')

