from pathlib import Path
from rlbot.matchconfig.match_config import MatchConfig, PlayerConfig, Team
from rlbottraining.match_configs import make_empty_match_config


def make_match_config_with_nombot() -> MatchConfig:
    # Makes a config which only has our bot in it.
    match_config = make_empty_match_config()
    match_config.player_configs = [
        PlayerConfig.bot_config(
            Path(__file__).absolute().parent.parent / 'NomBot_v1_5.cfg',
            Team.BLUE
        ),
    ]
    return match_config
