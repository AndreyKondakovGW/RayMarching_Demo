from abc import ABC, abstractmethod
import numpy as np
import src.distances as distances

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
        return distances.distance_from_sphere(self.center-point,self.radius)

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


class Box(RayMarchObject):
    def __init__(self, center, half_sides, color = np.array([1,0,0])):
        self.center = center
        self.half_sides = half_sides
        self.color = color

    def distance2point(self, point):
        d=distances.distance_from_box(self.center-point,self.half_sides)
        #print(d)
        return d

    def get_color(self):
        return self.color


class RoundBox(RayMarchObject):
    def __init__(self, center, half_sides, rounding, color = np.array([1,0,0])):
        self.center = center
        self.half_sides = half_sides
        self.roundig=rounding
        self.color = color

    def distance2point(self, point):
        return distances.distance_from_round_box(self.center-point,self.half_sides,self.roundig)

    def get_color(self):
        return self.color

class FrameBox(RayMarchObject):
    def __init__(self, center, half_sides, thickness, color = np.array([1,0,0])):
        self.center = center
        self.half_sides = half_sides
        self.thickness=thickness
        self.color = color

    def distance2point(self, point):
        return distances.distance_from_frame_box(self.center-point,self.half_sides,self.thickness)

    def get_color(self):
        return self.color

class Torus(RayMarchObject):
    def __init__(self, center, radi, color = np.array([1,0,0])):
        self.center = center
        self.radi = radi
        self.color = color

    def distance2point(self, point):
        return distances.distance_from_torus(self.center-point,self.radi)

    def get_color(self):
        return self.color

class Cylinder(RayMarchObject):
    def __init__(self, center, radius,height ,color = np.array([1,0,0])):
        self.center = center
        self.radius = radius
        self.height = height
        self.color = color

    def distance2point(self, point):
        return distances.distance_from_cylinder(self.center-point,self.radius,self.height)

    def get_color(self):
        return self.color

class Cone(RayMarchObject):
    def __init__(self, center, c, height, color = np.array([1,0,0])):
        self.center = center
        self.c = c
        self.height = height
        self.color = color

    def distance2point(self, point):
        return distances.distance_from_cone(self.center-point,self.c,self.height)

    def get_color(self):
        return self.color

