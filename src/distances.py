import numpy as np
import numpy.typing as npt
from numba import cuda, float32, float64
import math

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
def distance_from_fuzzy_sphere(origin_x, origin_y, origin_z, r, center_x, center_y, center_z, scaler):
    displacement = math.sin(scaler * origin_x) * math.sin(scaler * origin_y) * math.sin(scaler * origin_z) * 0.25 * r
    return math.sqrt((origin_x - center_x) ** 2 + (origin_y - center_y) ** 2 + (origin_z - center_z) ** 2) - r + displacement

@cuda.jit(device=True)
def distance_from_planey(origin_x, origin_y, origin_z, center_y):
    return abs(origin_y - center_y)

@cuda.jit(device=True)
def distance_from_box(point_x, point_y, point_z, half_sides_x, half_sides_y, half_sides_z, dir_x, dir_y, dir_z, angle):
    # for i in range(3):
    #     q[i] = abs(point[i]) - half_sides[i]
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
def distance_from_frame_box(point, half_sides, thickness):
    for i in range(3):
        point[i] = abs(point[i]) - half_sides[i]
    q = cuda.local.array(3, float32)
    for i in range(3):
        q[i] = abs(point[i] + thickness) - thickness
    result = cuda.local.array(3, float32)
    for i in range(3):
        result[i] = (np.linalg.norm([point[i], q[(i+1)%3], q[(i+2)%3]]) + 
                     min(max(point[i], q[(i+1)%3], q[(i+2)%3]), 0))
    return min(result[0], result[1], result[2])

@cuda.jit(device=True)
def distance_from_round_box(point, half_sides, dir_x, dir_y, dir_z, angle, rounding):
    if angle != 0:
        point[0], point[1], point[2] = rotate_point(point[0], point[1], point[2], dir_x, dir_y, dir_z, angle)
    
    q = cuda.local.array(3, float32)
    for i in range(3):
        q[i] = abs(point[i]) - half_sides[i] + rounding

    # q_x = abs(point_x) - half_sides_x + rounding
    # q_y = abs(point_y) - half_sides_y + rounding
    # q_z = abs(point_z) - half_sides_z + rounding
    max_q = 0.0
    for i in range(3):
        max_q = max(max_q, q[i])
    norm = 0.0
    for i in range(3):
        norm += max(0, q[i]) ** 2
    norm = norm ** 0.5
    return norm + min(max_q, 0) - rounding

@cuda.jit(device=True)
def distance_from_torus(point, radi):
    q = cuda.local.array(2, float32)
    q[0] = np.linalg.norm([point[0], point[2]]) - radi[0]
    q[1] = point[1]
    norm = 0.0
    for i in range(2):
        norm += q[i] ** 2
    norm = norm ** 0.5
    return norm - radi[1]

@cuda.jit(device=True)
def distance_from_cylinder(point, radius, height):
    d = cuda.local.array(2, float32)
    d[0] = np.linalg.norm([point[0], point[2]]) - radius
    d[1] = abs(point[1]) - height
    max_d = max(d[0], d[1])
    norm = 0.0
    for i in range(2):
        norm += max(0, d[i]) ** 2
    norm = norm ** 0.5
    return min(max_d, 0) + norm

@cuda.jit(device=True)
def distance_from_cone(point, c, height):
    q = height * cuda.local.array(2, float32)
    q[0] = c[0] / c[1]
    q[1] = -1
    w = cuda.local.array(2, float32)
    w[0] = np.linalg.norm([point[0], point[2]])
    w[1] = point[1]
    a = w - q * np.clip(np.dot(w, q) / np.dot(q, q), 0, 1)
    b = w - q * cuda.local.array(2, float32)
    b[0] = np.clip(w[0] / q[0], 0, 1)
    b[1] = 1
    k = np.sign(q[1])
    d = min(np.dot(a, a), np.dot(b, b))
    s = max(k * (w[0] * q[1] - w[1] * q[0]), k * (w[1] - q[1]))
    return np.sqrt(d) * np.sign(s)

