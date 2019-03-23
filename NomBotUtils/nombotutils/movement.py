from nombotutils.vector_math import normalize, dot, cross
from nombotutils.constants import UP


def get_pitch_yaw_roll(car, forward, up=UP):
    """
    Returns a (pitch, yaw, roll) tuple which try to orient the car with the nose pointing
    in the @forward direction and with the roof pointing in the @up direction
    """
    forward = normalize(forward)
    desired_facing_angular_vel = -cross(car.forward, forward)
    desired_up_angular_vel = -cross(car.up, up)

    pitch = dot(desired_facing_angular_vel, car.right)
    yaw = -dot(desired_facing_angular_vel, car.up)
    roll = dot(desired_up_angular_vel, car.forward)

    pitch_vel =  dot(car.angular_vel, car.right)
    yaw_vel   = -dot(car.angular_vel, car.up)
    roll_vel  =  dot(car.angular_vel, car.forward)

    # avoid getting stuck in directly-opposite states
    if dot(car.up, up) < -.8 and dot(car.forward, forward) > .8:#abs(roll_vel) < .3 and dot(car.up, up) < -.98 and dot(car.forward, forward) > .8:
        if roll == 0:
            roll = 1
        roll *= 1e10
    if dot(car.forward, forward) < -.8:
        if pitch == 0:
            pitch = 1
        pitch *= 1e10
    # TODO: do this for pitch too. (yaw not required)

    if dot(car.forward, forward) < 0.0:
        pitch_vel *= -1

    # PID control to stop overshooting.
    roll  = 3*roll  + 0.30*roll_vel
    yaw   = 3*yaw   + 0.70*yaw_vel
    pitch = 3*pitch + 0.90*pitch_vel

    # only start adjusting roll once we're roughly facing the right way
    if dot(car.forward, forward) < 0:
        roll = 0

    # To debug a single-axis
    # pitch = 0
    # yaw = 0
    # roll = 0
    return (pitch, yaw, roll)
