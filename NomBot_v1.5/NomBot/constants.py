import numpy as np

UP = np.array([0.0, 0.0, 1.0])
UP.flags.writeable = False
STEER_R = +1
STEER_L = -1

# Indexes into pos/vel
TO_STATUE = 0  # There is a statue outside the playing fields. The direction the default observer cam faces.
TO_ORANGE = 1   # towards
TO_CEILING = 2  # UP is already taken

# Physics constants
BALL_RADIUS = 92.
MAX_CAR_SPEED = 2300.005
FLIP_SPEED_CHANGE = 500.0  # TODO: refine constant

# indexes into the output vector
OUT_VEC_THROTTLE = 0
OUT_VEC_STEER = 1
OUT_VEC_PITCH = 2
OUT_VEC_YAW = 3
OUT_VEC_ROLL = 4
OUT_VEC_JUMP = 5
OUT_VEC_BOOST = 6
OUT_VEC_HANDBRAKE = 7
