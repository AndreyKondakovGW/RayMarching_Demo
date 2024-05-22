from abc import ABC, abstractmethod
import numpy as np

class RayMarchObject(ABC):
    @abstractmethod
    def distance2point(self, point):
        pass

    @abstractmethod
    def get_color(self):
        pass

max_figure_params = 5

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
    

class FuzzySphere():
    def __init__(self, center, radius, color = np.array([1,0,0])):
        self.id = 1
        self.center = center
        self.radius = radius
        self.color = color
        self.scaler = 5.0

    def distance2point(self, point): 
        displacement = np.sin(self.scaler * point[0]) * np.sin(self.scaler* point[1]) * np.sin(self.scaler* point[2]) * 0.25 * self.radius
        return np.linalg.norm(self.center - point) - self.radius + displacement
    def get_color(self):
        return self.color
    
    def to_array(self):
        # every sphere encoded with [id, color_r, color_g, color_b, radius, center_x, center_y, center_z, scaler]
        res = np.zeros(4 + max_figure_params)
        res[0] = self.id
        res[1:4] = self.color
        res[4] = self.radius
        res[5:8] = self.center
        res[8] = self.scaler
        return res
    
class PlaneY:
    def __init__(self, y, color = np.array([0,1,0])):
        self.id = 2
        self.y = y
        self.color = color

    def distance2point(self, point):
        return point[1] - self.y
    
    def get_color(self):
        return self.color
    
    def to_array(self):
        # every sphere encoded with [id, color_r, color_g, color_b, y]
        res = np.zeros(4 + max_figure_params)
        res[0] = self.id
        res[1:4] = self.color
        res[4] = self.y
        return res