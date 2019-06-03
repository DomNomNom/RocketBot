from nombotutils.vector_math import normalize, dot, cross, z0
from nombotutils.constants import UP, MAX_CAR_SPEED, FLIP_SPEED_CHANGE


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

def flip_in_direction(s, target_direction):
    '''
    returns a (pitch, yaw, roll) tuple which will make the car flip in the @target_direction

    Flip testing notes:
    I call it flipping, others call it air-rolling, but I dislike "rolling" in this context as it implies it's slow.
    Flipping will set your vertical speed to 0.
    "flip_forward" is the normalized direction for car.forward projected onto the horizontal plane.
    I define a "front flip" to be the flip where the cars nose turns towards the wheels.
    When front flipping, velocity is added in the flip_forward direction.
    This implies that if you pich your car up slightly more than 90degrees and front flip, you'll gain speed in the direction what used to be your backwards (before pitching up).
    car-orientation-roll does not affect flip_forward.
    '''
    target_direction = normalize(target_direction)
    flip_forward = normalize(z0(s.car.forward))
    flip_right = cross(flip_forward, UP)

    yaw = 0.0
    pitch = -dot(target_direction, flip_forward)
    roll = -dot(target_direction, flip_right)
    # pitch = 0
    return (pitch, yaw, roll)

def flip_towards(s, target_pos):
    '''
    Takes into account velocity
    '''
    towards_target_dir = normalize(target_pos - s.car.pos)
    desired_vel = towards_target_dir * min(MAX_CAR_SPEED*1.05, s.car.speed + FLIP_SPEED_CHANGE*1.0)
    acceleration_dir = normalize(desired_vel - s.car.vel)
    return flip_in_direction(s, acceleration_dir)
