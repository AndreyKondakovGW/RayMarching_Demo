import numpy as np
from numba import cuda
import math
from typing import List, Tuple
from src.distances import *

number_of_steps = 64
min_hit_distance = 0.01
max_trace_distance = 100
background_color = np.array([0, 0, 0])

@cuda.jit
def clamp(n: float, min: float, max: float) -> float:
    if n < min:
        return min
    elif n > max:
        return max
    else:
        return n

@cuda.jit
def mix(a: float, b: float, k: float) -> float:
    return a * (1.0 - k) + b * k

@cuda.jit
def smin(a: float, b: float, k: float) -> float:
    k *= 1.0 / (1.0 - math.sqrt(0.5))
    h = max(k - abs(a - b), 0.0) / k
    return min(a, b) - k * 0.5 * (1.0 + h - math.sqrt(1.0 - h * (h - 2.0)))

@cuda.jit
def find_closest_obj(
    pos_x: float, pos_y: float, pos_z: float, object_buffer: List[List[float]]
) -> Tuple[float, float, float, float]:
    """
    Function which iterates over all objects in object_buffer and finds the closest object to the point pos_x, pos_y, pos_z.
    Returns distance to this object and its color.
    """
    closest_dist = np.inf
    r, g, b = 0.0, 0.0, 0.0

    for obj in object_buffer:
        if obj[0] == 0:
            dist = distance_from_sphere(pos_x, pos_y, pos_z, obj[4], obj[5], obj[6], obj[7], obj[8] * obj[9])
        elif obj[0] == 1:
            dist = distance_from_fuzzy_sphere(pos_x, pos_y, pos_z, obj[4], obj[5], obj[6], obj[7], obj[8], obj[9] * obj[12], obj[10]* obj[12], obj[11]* obj[12])
        elif obj[0] == 8:
            dist = distance_from_planey(pos_x, pos_y, pos_z, obj[4])
        elif obj[0] == 2:
            dist = distance_from_box(pos_x - obj[7], pos_y - obj[8], pos_z - obj[9], obj[4], obj[5], obj[6], obj[10], obj[11], obj[12], obj[13])
        elif obj[0] == 3:
            dist = distance_from_round_box(pos_x - obj[8], pos_y - obj[9], pos_z - obj[10], obj[5], obj[6], obj[7], obj[4])
        elif obj[0] == 4:
            dist = distance_from_frame_box(pos_x - obj[8], pos_y - obj[9], pos_z - obj[10], obj[5], obj[6], obj[7], obj[4], obj[11], obj[12], obj[13], obj[14])
        elif obj[0] == 5:
            dist = distance_from_torus(pos_x - obj[6], pos_y - obj[7], pos_z - obj[8], obj[4], obj[5])
        elif obj[0] == 6:
            dist = distance_from_cylinder(pos_x - obj[6], pos_y - obj[7], pos_z - obj[8], obj[4], obj[5], obj[9], obj[10], obj[11], obj[12])
        elif obj[0] == 7:
            dist = distance_from_cone(pos_x - obj[7], pos_y - obj[8], pos_z - obj[9], obj[5], obj[6], obj[4], obj[10], obj[11], obj[12], obj[13])

        if abs(dist - closest_dist) < 0.1:
            closest_dist = smin(closest_dist, dist, 0.01)
            r = r #smin(r, obj[1], 0.1)
            g = g #smin(g, obj[2], 0.1)
            b = b #smin(b, obj[3], 0.1)
        elif dist < closest_dist:
            closest_dist = dist
            r, g, b = obj[1], obj[2], obj[3]

    return closest_dist, r, g, b

@cuda.jit
def calculate_normal(
    pos_x: float, pos_y: float, pos_z: float, object_buffer: List[List[float]]
) -> Tuple[float, float, float]:
    """
    Function which calculates the normal to the object at the point pos_x, pos_y, pos_z.
    """
    eps = 0.01
    normal_x = (
        find_closest_obj(pos_x + eps, pos_y, pos_z, object_buffer)[0]
        - find_closest_obj(pos_x - eps, pos_y, pos_z, object_buffer)[0]
    )
    normal_y = (
        find_closest_obj(pos_x, pos_y + eps, pos_z, object_buffer)[0]
        - find_closest_obj(pos_x, pos_y - eps, pos_z, object_buffer)[0]
    )
    normal_z = (
        find_closest_obj(pos_x, pos_y, pos_z + eps, object_buffer)[0]
        - find_closest_obj(pos_x, pos_y, pos_z - eps, object_buffer)[0]
    )

    normal_norm = math.sqrt(normal_x ** 2 + normal_y ** 2 + normal_z ** 2)
    if normal_norm == 0:
        normal_norm = 1

    normal_x /= normal_norm
    normal_y /= normal_norm
    normal_z /= normal_norm

    return normal_x, normal_y, normal_z

@cuda.jit
def calculate_lighting(
    pos_x: float, pos_y: float, pos_z: float, object_buffer: List[List[float]], light_pos_x: float, light_pos_y: float, light_pos_z: float
) -> float:
    """
    Calculate the lighting level at the point pos_x, pos_y, pos_z.
    """
    normal_x, normal_y, normal_z = calculate_normal(pos_x, pos_y, pos_z, object_buffer)
    light_dir_x = light_pos_x - pos_x
    light_dir_y = light_pos_y - pos_y
    light_dir_z = light_pos_z - pos_z

    light_dir_norm = math.sqrt(light_dir_x ** 2 + light_dir_y ** 2 + light_dir_z ** 2)
    if light_dir_norm == 0:
        light_dir_norm = 1

    light_dir_x /= light_dir_norm
    light_dir_y /= light_dir_norm
    light_dir_z /= light_dir_norm

    return max(0.0, normal_x * light_dir_x + normal_y * light_dir_y + normal_z * light_dir_z)

@cuda.jit
def ray_march_kernel(
    result: np.ndarray, origin_x: float, origin_y: float, origin_z: float, directions: np.ndarray, object_buffer: List[List[float]]
) -> None:
    x, y = cuda.grid(2)
    total_distance_traveled = 0.0
    bump_obj = 0
    if x < result.shape[0] and y < result.shape[1]:
        for _ in range(number_of_steps):
            current_position_x = origin_x + total_distance_traveled * directions[x, y, 0]
            current_position_y = origin_y + total_distance_traveled * directions[x, y, 1]
            current_position_z = origin_z + total_distance_traveled * directions[x, y, 2]

            cdist, color_r, color_g, color_b = find_closest_obj(current_position_x, current_position_y, current_position_z, object_buffer)
            if cdist < min_hit_distance:
                l = calculate_lighting(current_position_x, current_position_y, current_position_z, object_buffer, 0, -5, 0)
                result[x, y, 0] = color_r * l
                result[x, y, 1] = color_g * l
                result[x, y, 2] = color_b * l
                bump_obj = 1
                break

            if total_distance_traveled > max_trace_distance:
                break

            total_distance_traveled += cdist

        if bump_obj == 0:
            result[x, y, 0] = background_color[0]
            result[x, y, 1] = background_color[1]
            result[x, y, 2] = background_color[2]
