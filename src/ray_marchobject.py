from abc import ABC, abstractmethod
from numba import int32, float64
from numba.experimental import jitclass
import numpy as np

class RayMarchObject(ABC):
    @abstractmethod
    def distance2point(self, point):
        pass

    @abstractmethod
    def get_color(self):
        pass

spec = [
    ('center', float64[:]),
    ('radius', float64),             
    ('color', float64[:]),
    ('id', int32),
]


max_figure_params = 4

@jitclass(spec)
class Sphere():
    def __init__(self, center, radius, color = np.array([1,0,0])):
        self.id = 0
        self.center = center
        self.radius = radius
        self.color = color

    def distance2point(self, point: np.array):
        return np.linalg.norm(self.center - point) - self.radius
    
    @property
    def get_color(self):
        return self.color
    
    def to_array(self):
        # every sphere encoded with [id, color_r, color_g, color_b, radius, center_x, center_y, center_z]
        res = np.zeros(4 + max_figure_params)
        res[0] = self.id
        res[1:4] = self.color
        res[4] = self.radius
        res[5:8] = self.center
        return res
    

class FuzzySphere(RayMarchObject):
    def __init__(self, center, radius, color = np.array([1,0,0])):
        self.center = center
        self.radius = radius
        self.color = color

    def distance2point(self, point):
        scaler = 5.0 
        displacement = np.sin(scaler * point[0]) * np.sin(scaler* point[1]) * np.sin(scaler* point[2]) * 0.25 * self.radius
        return np.linalg.norm(self.center - point) - self.radius + displacement
    def get_color(self):
        return self.color
    
    def to_array(self):
        # every sphere encoded with [id, color_r, color_g, color_b, radius, center_x, center_y, center_z]
        res = np.zeros(4 + max_figure_params)
        res[0] = self.id
        res[1:4] = self.color
        res[4] = self.radius
        res[5:8] = self.center
        return res