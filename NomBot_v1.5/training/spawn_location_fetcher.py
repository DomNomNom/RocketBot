from dataclasses import dataclass, field
from math import pi
from typing import Optional

from rlbot.utils.game_state_util import GameState, BoostState, BallState, CarState, Physics, Vector3, Rotator
from rlbot.matchconfig.match_config import MatchConfig, PlayerConfig, Team
from rlbot.training.training import Pass

from rlbottraining.common_graders.compound_grader import CompoundGrader
from rlbottraining.common_graders.timeout import FailOnTimeout
from rlbottraining.grading.grader import Grader, Grade
from rlbottraining.grading.training_tick_packet import TrainingTickPacket
from rlbottraining.rng import SeededRandomNumberGenerator
from rlbottraining.training_exercise import Playlist, TrainingExercise
from rlbottraining.match_configs import make_empty_match_config
from rlbottraining.paths import BotConfigs

import match_configs

class SpawnLocationGrader(CompoundGrader):
    """
    Checks that the car gets to the ball in a reasonable amount of time.
    """
    def __init__(self, timeout_seconds=8.0):
        super().__init__([
            PassOnSpawn(),
            FailOnTimeout(timeout_seconds),
        ])

@dataclass
class GotSpawnLocation(Pass):
    spawn_location: Vector3
    yaw_over_pi: Rotator

    def __repr__(self):
        return f'Pass: loc: {self.spawn_location}, yaw: {self.yaw_over_pi}'

@dataclass
class PassOnSpawn(Grader):
    """
    Returns a Pass grade once the car is sufficiently close to the ball.
    """

    min_dist_to_pass: float = 200
    car_index: int = 0
    init_team_score: int = None
    goal_time: float = None

    def on_tick(self, tick: TrainingTickPacket) -> Optional[Grade]:
        packet = tick.game_tick_packet
        car = packet.game_cars[self.car_index]
        if self.init_team_score is None:
            self.init_team_score = packet.teams[car.team].score
        # Wait for new goal
        if packet.teams[car.team].score == self.init_team_score:
            return None

        # Are we in kickoff countdown?
        if self.goal_time is None:
            self.goal_time = packet.game_info.seconds_elapsed
        if packet.game_info.seconds_elapsed - self.goal_time > 5.0:
            return GotSpawnLocation(spawn_location=car.physics.location, yaw_over_pi=car.physics.rotation.yaw / pi)
        return None


# The ball is rolling towards goal but you still need to put it in
@dataclass
class SpawnLocationExercise(TrainingExercise):
    grader: Grader = field(default_factory=lambda: SpawnLocationGrader())

    def make_game_state(self, rng: SeededRandomNumberGenerator) -> GameState:
        return GameState(
            ball=BallState(Physics(
                location=Vector3(0, 2500, 100),
                velocity=Vector3(0, 2000, 500)
            )),
            cars={0: CarState(boost_amount=87, jumped=True, double_jumped=True, physics=Physics(
                location=Vector3(300, 0, 100),
                rotation=Rotator(0, 0, 0),
                velocity=Vector3(0, 0, 0),
                angular_velocity=Vector3(0, 0, 0),
            ))}
        )



def make_default_playlist():
    exercises = (
        [
            SpawnLocationExercise(name="SpawnLocationExercise"),
            SpawnLocationExercise(name="SpawnLocationExercise"),
            SpawnLocationExercise(name="SpawnLocationExercise"),
            # SpawnLocationExercise(name="SpawnLocationExercise"),
            # SpawnLocationExercise(name="SpawnLocationExercise"),
        ]
    )
    for exercise in exercises:
        match_config = make_empty_match_config()
        match_config.player_configs = [
            # RocketLeague doesn't like being started without any players.
            PlayerConfig.bot_config(BotConfigs.brick_bot, Team.BLUE),
        ]
        match_config.mutators.respawn_time = '3 Seconds'
        match_config.instant_start = False
        exercise.match_config = match_config

    return exercises
