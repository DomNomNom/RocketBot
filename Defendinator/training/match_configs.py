from pathlib import Path
from rlbot.matchconfig.match_config import MatchConfig, PlayerConfig, Team
from rlbottraining.match_configs import make_empty_match_config


def make_match_config_with_nombot() -> MatchConfig:
    # Makes a config which only has our bot in it.
    match_config = make_empty_match_config()
    match_config.player_configs = [
        make_defendinator_player_config(Team.BLUE),
    ]
    return match_config

def make_defendinator_player_config(team: Team) -> PlayerConfig:
    return PlayerConfig.bot_config(
        Path(__file__).absolute().parent.parent / 'Defendinator.cfg',
        team
    )

def make_kickoff_config(other_bot: PlayerConfig, bot_thats_supposed_to_win: PlayerConfig=None) -> MatchConfig:
    if bot_thats_supposed_to_win is None:
        bot_thats_supposed_to_win = make_defendinator_player_config(Team.BLUE)
    bot_thats_supposed_to_win.team = Team.BLUE.value
    other_bot.team = Team.ORANGE.value

    match_config = make_empty_match_config()
    match_config.player_configs = [
        bot_thats_supposed_to_win,
        other_bot,
    ]
    return match_config


def make_allstar_player_config(team: Team) -> PlayerConfig:
    config = PlayerConfig()
    config.bot = True
    config.rlobt_controlled = False
    config.bot_skill = 1.0
    config.name = 'Psyonix Bot'
    return config
