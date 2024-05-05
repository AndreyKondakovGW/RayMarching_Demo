from abc import ABC, abstractmethod
import numpy as np

class RayMarchObject(ABC):
    @abstractmethod
    def distance2point(self, point):
        pass

    @abstractmethod
    def get_color(self):
        pass

class Sphere(RayMarchObject):
    def __init__(self, center, radius, color = np.array([1,0,0])):
        self.center = center
        self.radius = radius
        self.color = color

    def distance2point(self, point):
        return np.linalg.norm(self.center - point) - self.radius
    
    def get_color(self):
        return self.color
    

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