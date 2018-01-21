
import math
import numpy as np
from functools import reduce

UCONST_Pi = 3.1415926
URotation180 = float(32768)
URotationToRadians = UCONST_Pi / URotation180
tau = 2*UCONST_Pi



def mag(vec):
    ''' magnitude/length of a vector '''
    return np.linalg.norm(vec)
def normalize(vec):
    magnitude = mag(vec)
    if not magnitude: return vec
    return vec / magnitude
def vec2angle(vec):
    return math.atan2(vec[1], vec[0])
def rotate90degrees(xy):
    return np.array([xy[1], -xy[0]])
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
    # return to_left_handedness(euler2mat(
    #         URotationToRadians * rotator.Roll,
    #         URotationToRadians * rotator.Yaw,
    #         URotationToRadians * rotator.Pitch,
    #     ))

def to_rotation_matrix(pitch, yaw, roll):

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

    # return mat_roll
    # return reduce(np.dot, [mat_yaw, mat_pitch])
    return reduce(np.dot, [mat_yaw, mat_roll, mat_pitch])


# A wrapper for the game_tick_packet
class EasyGameState(object):
    def __init__(self, game_tick_packet, car_index):
        car = game_tick_packet.gamecars[car_index]
        self.car_pos = stuct_vector3_to_numpy(car.Location)
        self.car_vel = stuct_vector3_to_numpy(car.Velocity)
        self.car_rotation_matrix = rotation_to_mat(car.Rotation)
        self.car_forward = self.car_rotation_matrix.dot(np.array([1.0, 0.0, 0.0]))
        self.car_right = self.car_rotation_matrix.dot(np.array([0.0, 1.0, 0.0]))
        self.car_up = self.car_rotation_matrix.dot(np.array([0.0, 0.0, 1.0]))
        self.pyr = np.array([
            URotationToRadians * car.Rotation.Pitch,
            URotationToRadians * car.Rotation.Yaw,
            URotationToRadians * car.Rotation.Roll,
        ])


def main():
    import sys
    import os
    os.chdir(r'C:\Users\dom\Documents\GitHub\NomBot')
    os.system('python doms_runner.py')
    sys.exit()

if __name__ == '__main__':
    main()

def to_left_handedness(rotation_matrix):
    return rotation_matrix
    # To change it from left to right or right to left, flip it like this:
    (( rx, ry, rz ),    ( ux, uy, uz ),    ( lx, ly, lz ),) = rotation_matrix
    return np.array([
        [rx, rz, ry],
        [lx, lz, ly],
        [ux, uz, uy],
    ])


def euler2mat(z=0, y=0, x=0):
    ''' Return matrix for rotations around z, y and x axes

    Uses the z, then y, then x convention above

    Parameters
    ----------
    z : scalar
       Rotation angle in radians around z-axis (performed first)
    y : scalar
       Rotation angle in radians around y-axis
    x : scalar
       Rotation angle in radians around x-axis (performed last)

    Returns
    -------
    M : array shape (3,3)
       Rotation matrix giving same rotation as for given angles

    Examples
    --------
    >>> zrot = 1.3 # radians
    >>> yrot = -0.1
    >>> xrot = 0.2
    >>> M = euler2mat(zrot, yrot, xrot)
    >>> M.shape == (3, 3)
    True

    The output rotation matrix is equal to the composition of the
    individual rotations

    >>> M1 = euler2mat(zrot)
    >>> M2 = euler2mat(0, yrot)
    >>> M3 = euler2mat(0, 0, xrot)
    >>> composed_M = np.dot(M3, np.dot(M2, M1))
    >>> np.allclose(M, composed_M)
    True

    You can specify rotations by named arguments

    >>> np.all(M3 == euler2mat(x=xrot))
    True

    When applying M to a vector, the vector should column vector to the
    right of M.  If the right hand side is a 2D array rather than a
    vector, then each column of the 2D array represents a vector.

    >>> vec = np.array([1, 0, 0]).reshape((3,1))
    >>> v2 = np.dot(M, vec)
    >>> vecs = np.array([[1, 0, 0],[0, 1, 0]]).T # giving 3x2 array
    >>> vecs2 = np.dot(M, vecs)

    Rotations are counter-clockwise.

    >>> zred = np.dot(euler2mat(z=np.pi/2), np.eye(3))
    >>> np.allclose(zred, [[0, -1, 0],[1, 0, 0], [0, 0, 1]])
    True
    >>> yred = np.dot(euler2mat(y=np.pi/2), np.eye(3))
    >>> np.allclose(yred, [[0, 0, 1],[0, 1, 0], [-1, 0, 0]])
    True
    >>> xred = np.dot(euler2mat(x=np.pi/2), np.eye(3))
    >>> np.allclose(xred, [[1, 0, 0],[0, 0, -1], [0, 1, 0]])
    True

    Notes
    -----
    The direction of rotation is given by the right-hand rule (orient
    the thumb of the right hand along the axis around which the rotation
    occurs, with the end of the thumb at the positive end of the axis;
    curl your fingers; the direction your fingers curl is the direction
    of rotation).  Therefore, the rotations are counterclockwise if
    looking along the axis of rotation from positive to negative.
    '''
    rotations = []
    if z:
        cosz = math.cos(z)
        sinz = math.sin(z)
        rotations.append(np.array(
                [[cosz, -sinz, 0],
                 [sinz, cosz, 0],
                 [0, 0, 1]]))
    if y:
        cosy = math.cos(y)
        siny = math.sin(y)
        rotations.append(np.array(
                [[cosy, 0, siny],
                 [0, 1, 0],
                 [-siny, 0, cosy]]))
    if x:
        cosx = math.cos(x)
        sinx = math.sin(x)
        rotations.append(np.array(
                [[1, 0, 0],
                 [0, cosx, -sinx],
                 [0, sinx, cosx]]))
    if rotations:
        return reduce(np.dot, rotations[::-1])
    return np.eye(3)
