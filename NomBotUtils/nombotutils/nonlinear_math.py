from typing import List
from math import sqrt

def solve_quadratic(a,b,c) -> List[float]:
    # Returns solutions for `x` in a quadratic equation of the form
    # ax^2 + bx + c = 0
    if a == 0:
        if c == 0:
            return []
        return [-c / b]

    determinant = b*b - 4*a*c
    if determinant < 0:
        return []
    return [ (-b + det)/(2*a) for det in [-sqrt(determinant), sqrt(determinant)] ]
