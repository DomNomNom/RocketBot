import numpy as np

# https://github.com/RLBot/RLBot/wiki/Useful-Game-Values

UP = np.array([0.0, 0.0, 1.0])
UP.flags.writeable = False
STEER_R = +1
STEER_L = -1

# Indexes into pos/vel
TO_STATUE = 0   # X - There is a statue outside the playing fields. The direction the default observer cam faces.
TO_ORANGE = 1   # Y - Towards the orange goal from the midpoint.
TO_CEILING = 2  # Z - UP is already taken

# Physics constants
BALL_RADIUS = 92.
MAX_CAR_SPEED = 2300.005
FLIP_SPEED_CHANGE = 500.0  # TODO: refine constant
BOOST_ACCELERATION = 991.666
