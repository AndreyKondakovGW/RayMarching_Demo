import numpy as np
from numba import cuda
import math

#unfortunatly there is no way to use jitclass with cuda.jit so we need to define dist functions here

#object type 1
@cuda.jit(device=True)
def sphere(origin_x, origin_y, origin_z, r, center_x, center_y, center_z):
    return math.sqrt((origin_x - center_x) ** 2 + (origin_y - center_y) ** 2 + (origin_z - center_z) ** 2) - r

#object type 2
@cuda.jit(device=True)
def fuzzy_sphere(origin_x, origin_y, origin_z, r, center_x, center_y, center_z, scaler):
    displacement = math.sin(scaler * origin_x) * math.sin(scaler * origin_y) * math.sin(scaler * origin_z) * 0.25 * r
    return math.sqrt((origin_x - center_x) ** 2 + (origin_y - center_y) ** 2 + (origin_z - center_z) ** 2) - r + displacement

@cuda.jit(device=True)
def planey(origin_x, origin_y, origin_z, center_y):
    return origin_y - center_y