import numpy as np
from numba import cuda, float32

@cuda.jit(device=True)
def distance_from_sphere(point_x, point_y, point_z, radius):
    norm = (point_x ** 2 + point_y ** 2 + point_z ** 2) ** 0.5
    return norm - radius

@cuda.jit(device=True)
def distance_from_box(point, half_sides):
    q = cuda.local.array(3, float32)
    for i in range(3):
        q[i] = abs(point[i]) - half_sides[i]
    max_q = 0.0
    for i in range(3):
        max_q = max(max_q, q[i])
    norm = 0.0
    for i in range(3):
        norm += max(0, q[i]) ** 2
    norm = norm ** 0.5
    return norm + min(max_q, 0)

@cuda.jit(device=True)
def distance_from_frame_box(point_x, point_y, point_z, half_side_x, half_side_y, half_side_z, thickness):
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
def distance_from_cylinder(point_x, point_y, point_z, radius, height):
    d_x = (point_x ** 2 + point_z ** 2) ** 0.5 - radius
    d_y = abs(point_y) - height
    max_d = max(d_x, d_y)
    norm = (max(0, d_x) ** 2 + max(0, d_y) ** 2) ** 0.5
    return min(max_d, 0) + norm

@cuda.jit(device=True)
def distance_from_cone(point_x, point_y, point_z, c_x, c_y, height):
    q_x = height * c_x / c_y
    q_y = -height
    w_x = (point_x ** 2 + point_z ** 2) ** 0.5
    w_y = point_y
    dot_wq = w_x * q_x + w_y * q_y
    dot_qq = q_x ** 2 + q_y ** 2
    a_x = w_x - q_x * np.clip(dot_wq / dot_qq, 0, 1)
    a_y = w_y - q_y * np.clip(dot_wq / dot_qq, 0, 1)
    b_x = w_x - q_x * np.clip(w_x / q_x, 0, 1)
    b_y = w_y - q_y
    k = np.sign(q_y)
    d = min(a_x ** 2 + a_y ** 2, b_x ** 2 + b_y ** 2)
    s = max(k * (w_x * q_y - w_y * q_x), k * (w_y - q_y))
    return np.sqrt(d) * np.sign(s)
