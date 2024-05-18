import numpy as np
from numba import cuda
import math

number_of_steps = 32
min_hit_distance = 0.1
max_trace_distabce = 100
background_color = np.array([0,0,0])

@cuda.jit(device=True)
def sphere(origin_x, origin_y, origin_z, r, center_x, center_y, center_z):
    return math.sqrt((origin_x - center_x) ** 2 + (origin_y - center_y) ** 2 + (origin_z - center_z) ** 2) - r


@cuda.jit
def ray_march_kernel(result, origin_x, origin_y, origin_z, directions):
    x, y = cuda.grid(2)
    total_distance_traveled = 0.0
    bump_obj = 0
    if x < result.shape[0] and y < result.shape[1]:
        for i in range(32):
            current_position_x = origin_x + total_distance_traveled * directions[x, y, 0]
            current_position_y = origin_y + total_distance_traveled * directions[x, y, 1]
            current_position_z = origin_z + total_distance_traveled * directions[x, y, 2]
            dist_sphere = sphere(current_position_x, current_position_y, current_position_z, 1, 5, 0, 0)
            
            if (dist_sphere < 0.1):
                result[x, y, 0] = 1
                result[x, y, 1] = 0
                result[x, y, 2] = 0
                bump_obj = 1
                break
            
            if total_distance_traveled > 100:
                break

            total_distance_traveled += dist_sphere

        if bump_obj == 0:
            result[x, y, 0] = 0#directions[x, y, 0]
            result[x, y, 1] = 0#directions[x, y, 1]
            result[x, y, 2] = 0#directions[x, y, 2]

def calculate_normal(point, world):
    eps = 0.01
    dx = np.array([eps, 0, 0])
    dy = np.array([0, eps, 0])
    dz = np.array([0, 0, eps])

    normal = np.array([
        closest_dist_in_world(point + dx, world)[0] - closest_dist_in_world(point - dx, world)[0],
        closest_dist_in_world(point + dy, world)[0] - closest_dist_in_world(point - dy, world)[0],
        closest_dist_in_world(point + dz, world)[0] - closest_dist_in_world(point - dz, world)[0]
    ])
    return normal / np.linalg.norm(normal)

def calculate_lighting(point, world, light_pos = np.array([0, -5, 0])):
    normal = calculate_normal(point, world)
    light_dir = light_pos - point
    light_dir = light_dir / np.linalg.norm(light_dir)
    return max(0.0, np.dot(normal, light_dir))

def closest_dist_in_world(point, world):
    dist = np.inf
    closest_color = np.array([0,0,0])
    for obj in world:
        if obj.distance2point(point) < dist:
            dist = obj.distance2point(point)
            closest_color = obj.get_color
    return dist, closest_color

def ray_march(origin, direction, world) -> np.array:
    #return color of object in this direction
    total_distance_traveled = 0.0
    for i in range(number_of_steps):
        current_position = origin + total_distance_traveled * direction
        distance_to_closest = 0
        distance_to_closest, color = closest_dist_in_world(current_position, world)

        if distance_to_closest < min_hit_distance:
            return color * calculate_lighting(current_position, world)
        
        if distance_to_closest > max_trace_distabce:
            break

        total_distance_traveled += distance_to_closest

    return background_color
