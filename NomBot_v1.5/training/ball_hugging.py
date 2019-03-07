from dataclasses import dataclass, field
from math import pi
from typing import Optional

from rlbot.training.training import Pass
from rlbot.utils.game_state_util import GameState, BoostState, BallState, CarState, Physics, Vector3, Rotator, GameInfoState

from rlbottraining.common_graders.compound_grader import CompoundGrader
from rlbottraining.common_graders.timeout import PassOnTimeout
from rlbottraining.grading.grader import Grader, Grade
from rlbottraining.grading.training_tick_packet import TrainingTickPacket
from rlbottraining.rng import SeededRandomNumberGenerator
from rlbottraining.training_exercise import Playlist, TrainingExercise
from rlbottraining.match_configs import make_empty_match_config
from rlbottraining.paths import BotConfigs

import match_configs


# The ball is rolling towards goal but you still need to put it in
@dataclass
class BallHugExercise(TrainingExercise):
    grader: Grader = field(default_factory=lambda: PassOnTimeout(4.1))

    def make_game_state(self, rng: SeededRandomNumberGenerator) -> GameState:
        # return GameState(
        #     ball=BallState(Physics(
        #         location=Vector3(0, 0, 70),
        #         velocity=Vector3(0, -300, 0)
        #     )),
        #     cars={0: CarState(boost_amount=87, jumped=True, double_jumped=True, physics=Physics(
        #         location=Vector3(0, -180, 130),
        #         rotation=Rotator(.48*pi, .5*pi, 0),
        #         velocity=Vector3(0, 100, -100),
        #         angular_velocity=Vector3(0, 0, 0),
        #     ))},
        #     game_info=GameInfoState(game_speed=0.5),
        # )
        return GameState(
            ball=BallState(Physics(
                location=Vector3(0, 0, 1600),
                velocity=Vector3(0, 0, -300),
                angular_velocity=Vector3(0, 0, 0),
            )),
            cars={0: CarState(boost_amount=87, jumped=False, double_jumped=False, physics=Physics(
                location=Vector3(-1.3, 0, 1400),
                rotation=Rotator(pi, 0, 0),
                velocity=Vector3(0, 0, 100),
                angular_velocity=Vector3(0, 0, 0),
            ))},
            game_info=GameInfoState(game_speed=1.0),
        )



def make_default_playlist():
    exercises = [
        BallHugExercise(name="BallHugExercise"),
        BallHugExercise(name="BallHugExercise"),
        BallHugExercise(name="BallHugExercise"),
        BallHugExercise(name="BallHugExercise"),
    ]
    for exercise in exercises:
        exercise.match_config = match_configs.make_match_config_with_nombot()

    return exercises
