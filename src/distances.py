import numpy as np
import numpy.typing as npt
from numba import cuda, float32, float64

@cuda.jit(device=True)
def distance_from_sphere(point, radius):
    norm = 0.0
    for i in range(point.shape[0]):
        norm += point[i] ** 2
    norm = norm ** 0.5
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
def distance_from_round_box(point, half_sides, rounding):
    q = cuda.local.array(3, float32)
    for i in range(3):
        q[i] = abs(point[i]) - half_sides[i] + rounding
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

@cuda.jit
def compute_distances(points, radii, results, shape_type):
    i = cuda.grid(1)
    if i < points.shape[0]:
        if shape_type[i] == 0:
            results[i] = distance_from_sphere(points[i], radii[i])
        elif shape_type[i] == 1:
            results[i] = distance_from_box(points[i], radii[i])
        elif shape_type[i] == 2:
            results[i] = distance_from_frame_box(points[i], radii[i][0], radii[i][1])
        elif shape_type[i] == 3:
            results[i] = distance_from_round_box(points[i], radii[i][0], radii[i][1])
        elif shape_type[i] == 4:
            results[i] = distance_from_torus(points[i], radii[i])
        elif shape_type[i] == 5:
            results[i] = distance_from_cylinder(points[i], radii[i][0], radii[i][1])
        elif shape_type[i] == 6:
            results[i] = distance_from_cone(points[i], radii[i][0], radii[i][1])

# Example usage:
points = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
radii = np.array([1.0, 2.0], dtype=np.float32)
results = np.zeros(points.shape[0], dtype=np.float32)
shape_type = np.array([0, 1], dtype=np.int32)  # 0 for sphere, 1 for box, etc.

threads_per_block = 256
blocks_per_grid = (points.shape[0] + (threads_per_block - 1)) // threads_per_block

compute_distances[blocks_per_grid, threads_per_block](points, radii, results, shape_type)
print(results)
