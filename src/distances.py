import numpy as np
import numpy.typing as npt
from numba import cuda, float32, float64
import math
@cuda.jit
def clamp(n, min, max): 
    if n < min: 
        return min
    elif n > max: 
        return max
    else: 
        return n
    
@cuda.jit
def sign(x):
    if x < 0:
        return -1
    elif x > 0:
        return 1
    else:
        return 0
    
@cuda.jit(device=True)
def rotate_point(p_x, p_y, p_z, dir_x, dir_y, dir_z, angle):
    angle = math.radians(angle)
    c = math.cos(angle)
    s = math.sin(angle)
    q_x = dir_x * (dir_x * p_x + dir_y * p_y + dir_z * p_z) * (1 - c) + p_x * c + (-dir_z * p_y + dir_y * p_z) * s
    q_y = dir_y * (dir_x * p_x + dir_y * p_y + dir_z * p_z) * (1 - c) + p_y * c + (dir_z * p_x - dir_x * p_z) * s
    q_z = dir_z * (dir_x * p_x + dir_y * p_y + dir_z * p_z) * (1 - c) + p_z * c + (-dir_y * p_x + dir_x * p_y) * s
    return q_x, q_y, q_z

@cuda.jit(device=True)
def normalize(x, y, z):
    norm = (x ** 2 + y ** 2 + z ** 2) ** 0.5
    return norm

@cuda.jit(device=True)
def mod(x, y):
    return x % y

@cuda.jit(device=True)
def distance_from_sphere(origin_x, origin_y, origin_z, radius, center_x, center_y, center_z, multiplier=1.0):
    origin_x = origin_x - center_x
    origin_y = origin_y - center_y
    origin_z = origin_z - center_z

    if multiplier != 0.0:
        a = multiplier
        origin_x = -a + mod(origin_x + a, 2*a)
        origin_y = -a + mod(origin_y + a, 2*a)
        origin_z = -a + mod(origin_z + a, 2*a)

    norm = 0.0
    norm += (origin_x) ** 2
    norm += (origin_y) ** 2
    norm += (origin_z) ** 2
    norm = norm ** 0.5
    return norm - radius

@cuda.jit(device=True)
def distance_from_fuzzy_sphere(origin_x, origin_y, origin_z, r, center_x, center_y, center_z, scaler, mult_x, mult_y, mult_z):
    displacement = math.sin(scaler * origin_x) * math.sin(scaler * origin_y) * math.sin(scaler * origin_z) * 0.25 * r

    origin_x = origin_x - center_x
    origin_y = origin_y - center_y
    origin_z = origin_z - center_z

    if mult_x != 0.0:
        origin_x = -mult_x + mod(origin_x + mult_x, 2*mult_x)
    if mult_y != 0.0:
        origin_y = -mult_y + mod(origin_y + mult_y, 2*mult_y)
    if mult_z != 0.0:
        origin_z = -mult_z + mod(origin_z + mult_z, 2*mult_z)

    return math.sqrt((origin_x) ** 2 + (origin_y) ** 2 + (origin_z) ** 2) - r + displacement

@cuda.jit(device=True)
def distance_from_planey(origin_x, origin_y, origin_z, center_y):
    return abs(origin_y - center_y)

@cuda.jit(device=True)
def distance_from_box(point_x, point_y, point_z, half_sides_x, half_sides_y, half_sides_z, dir_x, dir_y, dir_z, angle):
    if angle != 0:
        point_x, point_y, point_z = rotate_point(point_x, point_y, point_z, dir_x, dir_y, dir_z, angle)

    q_x = abs(point_x) - half_sides_x
    q_y = abs(point_y) - half_sides_y
    q_z = abs(point_z) - half_sides_z
    
    max_q = 0.0
    max_q = max(max_q, q_x)
    max_q = max(max_q, q_y)
    max_q = max(max_q, q_z)

    norm = normalize(max(0, q_x), max(0, q_y), max(0, q_z))
    return norm + min(max_q, 0)

@cuda.jit(device=True)
def distance_from_frame_box(point_x, point_y, point_z, half_side_x, half_side_y, half_side_z, thickness, dir_x, dir_y, dir_z, angle):
    if angle != 0:
        point_x, point_y, point_z = rotate_point(point_x, point_y, point_z, dir_x, dir_y, dir_z, angle)
    point_x = abs(point_x) - half_side_x
    point_y = abs(point_y) - half_side_y
    point_z = abs(point_z) - half_side_z
    q_x = abs(point_x + thickness) - thickness
    q_y = abs(point_y + thickness) - thickness
    q_z = abs(point_z + thickness) - thickness

    result_0 = (max(0, point_x) ** 2 + max(0, q_y) ** 2 + max(0, q_z) ** 2) ** 0.5 + min(max(point_x, q_y, q_z), 0)
    result_1 = (max(0, q_x) ** 2 + max(0, point_y) ** 2 + max(0, q_z) ** 2) ** 0.5 + min(max(q_x, point_y, q_z), 0)
    result_2 = (max(0, q_x) ** 2 + max(0, q_y) ** 2 + max(0, point_z) ** 2) ** 0.5 + min(max(q_x, q_y, point_z), 0)
    
    return min(result_0, result_1, result_2)

@cuda.jit(device=True)
def distance_from_round_box(point_x, point_y, point_z, half_side_x, half_side_y, half_side_z, rounding):
    q_x = abs(point_x) - half_side_x + rounding
    q_y = abs(point_y) - half_side_y + rounding
    q_z = abs(point_z) - half_side_z + rounding
    max_q = max(q_x, q_y, q_z)
    norm = (max(0, q_x) ** 2 + max(0, q_y) ** 2 + max(0, q_z) ** 2) ** 0.5
    return norm + min(max_q, 0) - rounding

@cuda.jit(device=True)
def distance_from_torus(point_x, point_y, point_z, radi_x, radi_y):
    q_x = (point_x ** 2 + point_z ** 2) ** 0.5 - radi_x
    q_y = point_y
    norm = (q_x ** 2 + q_y ** 2) ** 0.5
    return norm - radi_y

@cuda.jit(device=True)
def distance_from_cylinder(point_x, point_y, point_z, radius, height, dir_x, dir_y, dir_z, angle):
    if angle != 0:
        point_x, point_y, point_z = rotate_point(point_x, point_y, point_z, dir_x, dir_y, dir_z, angle)
    d_x = (point_x ** 2 + point_z ** 2) ** 0.5 - radius
    d_y = abs(point_y) - height
    max_d = max(d_x, d_y)
    norm = (max(0, d_x) ** 2 + max(0, d_y) ** 2) ** 0.5
    return min(max_d, 0) + norm

@cuda.jit(device=True)
def distance_from_cone(point_x, point_y, point_z, c_x, c_y, height, angle, dir_x, dir_y, dir_z):
    if angle != 0:
        point_x, point_y, point_z = rotate_point(point_x, point_y, point_z, dir_x, dir_y, dir_z, angle)
    q_x = height * c_x / c_y
    q_y = -height
    w_x = (point_x ** 2 + point_z ** 2) ** 0.5
    w_y = point_y
    dot_wq = w_x * q_x + w_y * q_y
    dot_qq = q_x ** 2 + q_y ** 2
    a_x = w_x - q_x * clamp(dot_wq / dot_qq, 0, 1)
    a_y = w_y - q_y * clamp(dot_wq / dot_qq, 0, 1)
    b_x = w_x - q_x * clamp(w_x / q_x, 0, 1)
    b_y = w_y - q_y
    k = sign(q_y)
    d = min(a_x ** 2 + a_y ** 2, b_x ** 2 + b_y ** 2)
    s = max(k * (w_x * q_y - w_y * q_x), k * (w_y - q_y))
    return math.sqrt(d) * sign(s)
