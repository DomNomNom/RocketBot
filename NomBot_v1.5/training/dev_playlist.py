from dataclasses import dataclass, field
from math import pi

from rlbottraining.common_exercises.common_base_exercises import StrikerExercise
from rlbottraining.common_graders.goal_grader import StrikerGrader
from rlbottraining.rng import SeededRandomNumberGenerator
from rlbottraining.training_exercise import Playlist

from rlbot.utils.game_state_util import GameState, BoostState, BallState, CarState, Physics, Vector3, Rotator

import match_configs

# The ball is rolling towards goal but you still need to put it in
@dataclass
class WideRollingTowardsGoalShot(StrikerExercise):
    grader: StrikerGrader = field(default_factory=lambda: StrikerGrader(timeout_seconds=6.))

    def make_game_state(self, rng: SeededRandomNumberGenerator) -> GameState:
        return GameState(
            ball=BallState(Physics(
                location=Vector3(rng.uniform(900, 1300), rng.uniform(0, 1500), 100),
                velocity=Vector3(0, 550, 0)
            )),
            cars={0: CarState(boost_amount=87, jumped=True, double_jumped=True, physics=Physics(
                location=Vector3(2000, -2500, 25),
                rotation=Rotator(0, pi / 2, 0),
                velocity=Vector3(0, 0, 0),
                angular_velocity=Vector3(0, 0, 0),
            ))}
        )


class FastRollingAcrossGoalShot(StrikerExercise):
    def make_game_state(self, rng: SeededRandomNumberGenerator) -> GameState:
        return GameState(
            ball=BallState(Physics(
                location=Vector3(3000, 4000, 100),
                velocity=Vector3(rng.uniform(-1000,-2000), 0, 0)
            )),
            cars={0: CarState(boost_amount=87, jumped=True, double_jumped=True, physics=Physics(
                location=Vector3(0, 1000, 25),
                rotation=Rotator(0, pi / 2, 0),
                velocity=Vector3(0, 0, 0),
                angular_velocity=Vector3(0, 0, 0),
            ))}
        )


def make_default_playlist():
    exercises = (
        # bronze_striker.make_default_playlist()
        [WideRollingTowardsGoalShot(name="rolley polley")]
        # [FastRollingAcrossGoalShot('wheee')]
    )
    for exercise in exercises:
        exercise.match_config = match_configs.make_match_config_with_nombot()

    return exercises
