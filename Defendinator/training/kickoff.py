
from dataclasses import dataclass, field
from math import pi
from typing import Optional

from rlbot.utils.game_state_util import GameState, BoostState, BallState, CarState, Physics, Vector3, Rotator, GameInfoState
from rlbot.matchconfig.match_config import MatchConfig, PlayerConfig, Team
from rlbot.training.training import Pass, Fail

from rlbottraining.common_graders.compound_grader import CompoundGrader
from rlbottraining.common_graders.timeout import FailOnTimeout
from rlbottraining.grading.grader import Grader, Grade
from rlbottraining.grading.training_tick_packet import TrainingTickPacket
from rlbottraining.rng import SeededRandomNumberGenerator
from rlbottraining.training_exercise import Playlist, TrainingExercise
from rlbottraining.match_configs import make_empty_match_config
from rlbottraining.paths import BotConfigs

import match_configs

class KickoffGrader(CompoundGrader):
    """
    Checks that the car gets to the ball in a reasonable amount of time.
    """
    def __init__(self, timeout_seconds=25.0):
        super().__init__([
            GradeBasedOnKickoff(),
            FailOnTimeout(timeout_seconds),
        ])

class WrongGoalFail(Fail):
    def __repr__(self):
        return f'{super().__repr__()}: Ball went into the wrong goal.'

@dataclass
class GradeBasedOnKickoff(Grader):
    """
    Returns a Pass grade once the car is sufficiently close to the ball.
    """

    car_index: int = 0
    init_team_score: int = None
    goal_time: float = None
    last_packet: float = None
    first_time: float = None

    def get_posession(self, s):
        """
        A value >= 1 means clearly posessed by the player
        """
        return 1

    def on_tick(self, tick: TrainingTickPacket) -> Optional[Grade]:
        packet = tick.game_tick_packet
        if self.first_time is None:
            self.first_time = packet.game_info.seconds_elapsed
        self.last_packet = packet
        car = packet.game_cars[self.car_index]
        if self.init_team_score is None:
            self.init_team_score = packet.teams[car.team].score
        # Wait for new goal
        if packet.teams[car.team].score == self.init_team_score:
            return None

        # Are we in kickoff countdown?
        if self.goal_time is None:
            self.goal_time = packet.game_info.seconds_elapsed
        # if packet.game_info.seconds_elapsed - self.goal_time > 5.0:
        #     return GotSpawnLocation(spawn_location=car.physics.location, yaw_over_pi=car.physics.rotation.yaw / pi)

        return None


@dataclass
class NaturalKickoffExercise(TrainingExercise):
    """
    Lets Rocket League decise where to spawn cars, and makes
    """
    grader: Grader = field(default_factory=lambda: KickoffGrader())

    def make_game_state(self, rng: SeededRandomNumberGenerator) -> GameState:
        return GameState(
            ball=BallState(Physics(
                location=Vector3(0, 5000, 100),
                velocity=Vector3(0, 2000, 500)
            )),
            cars={
                0: CarState(boost_amount=95, physics=Physics(
                    location=Vector3(200, -5400, 100),
                    rotation=Rotator(0, 3, 0),
                    velocity=Vector3(0, 0, 0),
                    angular_velocity=Vector3(0, 0, 0),
                )),
                1: CarState(boost_amount=95, physics=Physics(
                    location=Vector3(-200, -5400, 100),
                    rotation=Rotator(0, 3, 0),
                    velocity=Vector3(0, 0, 0),
                    angular_velocity=Vector3(0, 0, 0),
                )),
            },
            # game_info=GameInfoState(game_speed=0.1),
        )



def make_default_playlist():
    exercises = [
        NaturalKickoffExercise(name="kickoff"),
        NaturalKickoffExercise(name="kickoff"),
        NaturalKickoffExercise(name="kickoff"),
    ]
    for exercise in exercises:
        exercise.match_config = match_configs.make_kickoff_config(
            match_configs.make_allstar_player_config(Team.ORANGE)
        )
        exercise.match_config.mutators.respawn_time = '2 Seconds'
        exercise.match_config.instant_start = False

    return exercises
