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

max_figure_params = 14

class Sphere():
    def __init__(self, center, radius, color = np.array([1,0,0]), multipy=0, multipy_dist=5):
        self.id = 0
        self.center = center
        self.radius = radius
        self.color = color
        self.multipy = multipy
        self.multipy_dist = multipy_dist

    def distance2point(self, point: np.array):
        return distances.distance_from_sphere(self.center-point,self.radius)

    @property
    def get_color(self):
        return self.color
    
    def to_array(self):
        # every sphere encoded with [id, color_r, color_g, color_b, radius, center_x, center_y, center_z, multipy, multipy_dist]
        res = np.zeros(4 + max_figure_params)
        res[0] = self.id
        res[1:4] = self.color
        res[4] = self.radius
        res[5:8] = self.center
        res[8] = self.multipy
        res[9] = self.multipy_dist
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
        self.id = 8
        self.y = y
        self.color = color

    def distance2point(self, point):
        return point[1] - self.y
    
    def to_array(self):
        # every sphere encoded with [id, color_r, color_g, color_b, y]
        res = np.zeros(4 + max_figure_params)
        res[0] = self.id
        res[1:4] = self.color
        res[4] = self.y
        return res
    


class Box(RayMarchObject):
    def __init__(self, center, half_sides, rotation_dir = np.array([0,0,0]), rotation_angle = 0 ,color = np.array([1,0,0])):
        self.id=2
        self.center = center
        self.half_sides = half_sides
        self.color = color
        self.rotation_dir = rotation_dir
        self.rotation_angle = rotation_angle

    def distance2point(self, point):
        d=distances.distance_from_box(self.center-point,self.half_sides)
        #print(d)
        return d

    def get_color(self):
        return self.color
    def to_array(self):
        # every sphere encoded with [id, color_r, color_g, color_b, half_side_x, half_side_y, half_side_z, center_x, center_y, center_z, rotation_x, rotation_y, rotation_z, angle]
        res = np.zeros(4 + max_figure_params)
        res[0] = self.id
        res[1:4] = self.color
        res[4:7] = self.half_sides
        res[7:10] = self.center
        res[10:13] = self.rotation_dir
        res[13] = self.rotation_angle
        return res


class RoundBox(RayMarchObject):
    def __init__(self, center, half_sides, rounding, color = np.array([1,0,0]),rotation_dir = np.array([0,0,0]), rotation_angle = 0):
        self.id=3
        self.center = center
        self.half_sides = half_sides
        self.roundig = rounding
        self.color = color
        self.rotation_dir = rotation_dir
        self.rotation_angle = rotation_angle

    def distance2point(self, point):
        return distances.distance_from_round_box(self.center-point,self.half_sides,self.roundig)

    def get_color(self):
        return self.color

    def to_array(self):
        # every sphere encoded with [id, color_r, color_g, color_b, rounding, half_side_x, half_side_y, half_side_z, center_x, center_y, center_z, rotation_x, rotation_y, rotation_z, angle]
        res = np.zeros(4 + max_figure_params)
        res[0] = self.id
        res[1:4] = self.color
        res[4] = self.roundig
        res[5:8] = self.half_sides
        res[8:11] = self.center
        res[11:14] = self.rotation_dir
        res[14] = self.rotation_angle
        return res


class FrameBox(RayMarchObject):
    def __init__(self, center, half_sides, thickness, color = np.array([1,0,0])):
        self.id=4
        self.center = center
        self.half_sides = half_sides
        self.thickness=thickness
        self.color = color

    def distance2point(self, point):
        return distances.distance_from_frame_box(self.center-point,self.half_sides,self.thickness)

    def get_color(self):
        return self.color
    
    def to_array(self):
        res = np.zeros(4 + max_figure_params)
        res[0] = self.id
        res[1:4] = self.color
        res[4] = self.thickness
        res[5:8] = self.half_sides
        res[8:11] = self.center
        return res

class Torus(RayMarchObject):
    def __init__(self, center, radi = np.array([1,1]), color = np.array([1,0,0])):
        self.id=5
        self.center = center
        self.radi = radi
        self.color = color

    def distance2point(self, point):
        return distances.distance_from_torus(self.center-point,self.radi)

    def get_color(self):
        return self.color
    
    def to_array(self):
        res = np.zeros(4 + max_figure_params)
        res[0] = self.id
        res[1:4] = self.color
        res[4:6] = self.radi
        res[6:9] = self.center
        return res

class Cylinder(RayMarchObject):
    def __init__(self, center, radius,height ,color = np.array([1,0,0])):
        self.id=6
        self.center = center
        self.radius = radius
        self.height = height
        self.color = color

    def distance2point(self, point):
        return distances.distance_from_cylinder(self.center-point,self.radius,self.height)

    def get_color(self):
        return self.color
    
    def to_array(self):
        res = np.zeros(4 + max_figure_params)
        res[0] = self.id
        res[1:4] = self.color
        res[4] = self.height
        res[5] = self.radius
        res[6:9] = self.center
        return res

class Cone(RayMarchObject):
    def __init__(self, center, c, height, color = np.array([1,0,0])):
        self.id=7
        self.center = center
        self.c = c
        self.height = height
        self.color = color

    def distance2point(self, point):
        return distances.distance_from_cone(self.center-point,self.c,self.height)

    def get_color(self):
        return self.color
    
    def to_array(self):
        res = np.zeros(4 + max_figure_params)
        res[0] = self.id
        res[1:4] = self.color
        res[4] = self.height
        res[5:7] = self.c
        res[7:10] = self.center
        return res
