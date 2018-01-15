
import math
import numpy as np

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

def main():
    import sys
    import os
    os.chdir(r'C:\Users\dom\Documents\GitHub\RLBot\\')
    os.system('python runner.py')
    sys.exit()

if __name__ == '__main__':
    main()
