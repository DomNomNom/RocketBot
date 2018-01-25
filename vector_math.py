import math
import numpy as np
from functools import reduce

#  A utility wrapper around numpy.array
# mainly to name things how I like them

def Vec2(x, y):
    return np.array([x,y])
def Vec3(x, y, z):
    return np.array([x,y,z])

UCONST_Pi = 3.1415926
URotation180 = float(32768)
URotationToRadians = UCONST_Pi / URotation180
tau = 2*UCONST_Pi


UP = np.array([0.0, 0.0, 1.0])
UP.flags.writeable = False
cross = np.cross
sqrt = np.sqrt
equal = np.array_equal
def mag(vec):
    ''' magnitude/length of a vector '''
    return np.linalg.norm(vec)
def dist(vec1, vec2):
    return mag(vec2 - vec1)
def normalize(vec):
    magnitude = mag(vec)
    if not magnitude: return vec
    return vec / magnitude
def vec2angle(vec):
    return math.atan2(vec[1], vec[0])
def rotate90degrees(vec2):
    # Clockwise
    return Vec2(vec2[1], -vec2[0])
def closest180(angle):
    return ((angle+UCONST_Pi) % (2*UCONST_Pi)) - UCONST_Pi
def clamp(x, bot, top):
    return min(top, max(bot, x))
def clamp01(x):
    return clamp(x, 0.0, 1.0)
def clamp11(x):
    return clamp(x, -1.0, 1.0)
def lerp(v0, v1, t):  # linear interpolation
  return (1 - t) * v0 + t * v1;





def stuct_vector3_to_numpy(vec):
    return np.array([vec.X, vec.Y, vec.Z])

def rotation_to_mat(rotator):
    return to_rotation_matrix(
        URotationToRadians * rotator.Pitch,
        URotationToRadians * rotator.Yaw,
        URotationToRadians * rotator.Roll
    )

def to_rotation_matrix(pitch, yaw, roll):
    # Note: Unreal engine coordinate system
    y=pitch
    cosy = math.cos(y)
    siny = math.sin(y)
    mat_pitch = np.array(
            [[cosy, 0, -siny],
             [0, 1, 0],
             [siny, 0, cosy]])

    z=yaw
    cosz = math.cos(z)
    sinz = math.sin(z)
    mat_yaw = np.array(
            [[cosz, -sinz, 0],
             [sinz, cosz, 0],
             [0, 0, 1]])

    x=roll
    cosx = math.cos(x)
    sinx = math.sin(x)
    mat_roll = np.array(
            [[1, 0, 0],
             [0, cosx, sinx],
             [0, -sinx, cosx]])

    return reduce(np.dot, [mat_yaw, mat_roll, mat_pitch])

