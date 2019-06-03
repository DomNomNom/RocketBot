from pathlib import Path
from dataclasses import dataclass, field
from math import pi

from rlbot.utils.game_state_util import GameState, BoostState, BallState, CarState, Physics, Vector3, Rotator
from rlbot.matchconfig.match_config import MatchConfig, PlayerConfig, Team
from rlbottraining.common_exercises.common_base_exercises import StrikerExercise
from rlbottraining.rng import SeededRandomNumberGenerator
from rlbottraining.match_configs import make_empty_match_config
from rlbottraining.grading.grader import Grader
from rlbottraining.training_exercise import TrainingExercise, Playlist
from rlbottraining.paths import BotConfigs
from rlbottraining.common_exercises.versus_line_goalie import VersusLineGoalie, SecondShot, versus_line_goalie_match_config
from rlbottraining.paths import BotConfigs

# def make_match_config_with_my_bot() -> MatchConfig:
#     # Makes a config which only has our bot in it for now.
#     # For more defails: https://youtu.be/uGFmOZCpel8?t=375
#     match_config = make_empty_match_config()
#     match_config.player_configs = [
#         PlayerConfig.bot_config(
#             Path(__file__).absolute().parent.parent / 'Defendinator.cfg',
#             Team.BLUE
#         ),
#     ]
#     return match_config


def make_default_playlist() -> Playlist:
    exercises = [
        VersusLineGoalie('VersusLineGoalie'),
        # SecondShot('SecondShot'),
    ]
    for exercise in exercises:
        exercise.match_config = versus_line_goalie_match_config(
            attacker=BotConfigs.simple_bot,
            goalie=Path(__file__).absolute().parent.parent / 'Defendinator.cfg',
        )
    return exercises
