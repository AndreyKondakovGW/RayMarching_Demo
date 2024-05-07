import numpy as np
number_of_steps = 32
min_hit_distance = 0.1
max_trace_distabce = 100
background_color = np.array([0,0,0])


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
            closest_color = obj.get_color()
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
