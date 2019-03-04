import match_configs
import rlbottraining.common_exercises.bronze_striker as bronze_striker

def make_default_playlist():
    exercises = (
        bronze_striker.make_default_playlist()
    )
    for exercise in exercises:
        exercise.match_config = match_configs.make_match_config_with_nombot()

    return exercises
