import numpy as np
from src.ray_march import ray_march
from math import pi, tan, cos, sin, sqrt, asin, acos
class RayMarchCamera:
    def __init__(self, pos = np.array([0,0,0]), target = np.array([1,0,0]), fov = 90, aspect=1):
        self.pos = pos
        dir = target - pos
        self.dir = dir
        self.dir = self.dir / np.linalg.norm(self.dir)
        self.fov = fov
        self.aspect_ratio = aspect
        self.near = 1
        self.far = 1000

        dist_to_screen = 1
        self.screne_center = self.pos + self.dir * dist_to_screen
        self.screne_width = 2 * dist_to_screen * np.tan(np.radians(self.fov/2))
        self.screne_height = self.screne_width

        self.projection_matrix = self.get_projection()
        self.lookAtMatrix = self.count_lookAtMatrix()

    def count_axis_param(self):
        self.right = np.cross(self.dir, np.array([0,1,0]))
        if np.linalg.norm(self.right) != 0:
            self.right = self.right / np.linalg.norm(self.right)
        self.up = np.cross(self.right, self.dir)

    def count_lookAtMatrix(self):
        self.count_axis_param()
        axis_matrix = np.array([
            [self.right[0], self.right[1], self.right[2], 0],
            [self.up[0], self.up[1], self.up[2], 0],
            [self.dir[0], self.dir[1], self.dir[2], 0],
            [0, 0, 0, 1]
        ])

        position_matrix = np.array([
            [1, 0, 0, -self.pos[0]],
            [0, 1, 0, -self.pos[1]],
            [0, 0, 1, -self.pos[2]],
            [0, 0, 0, 1]
        ])

        return axis_matrix.dot(position_matrix)

    def get_projection(self):
        f = 1 / np.tan(np.radians(self.fov/2))
        aspect = self.aspect_ratio
        a = (self.far + self.near) / (self.far - self.near)
        b = (2 * self.far * self.near) / (self.far - self.near)

        return np.array([
            [f * aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, -a, -1],
            [0, 0, b, 0]
        ]).T
    

    def get_window_content(self, pixel_width, pixel_height, world):
        output = np.zeros((pixel_width, pixel_height, 3))
        for x in range(pixel_width):
            print(f"Rendering {x}/{pixel_width}")
            for y in range(pixel_height):
                output[x, y] = (ray_march(self.pos, self.all_rays[x,y], world) * 255).astype(int)
        return output


    def set_all_rays(self, pixel_width, pixel_height):
        #calculate all rays directions
        self.all_rays = np.zeros((pixel_width, pixel_height, 3))
        for x in range(pixel_width):
            for y in range(pixel_height):
                self.all_rays[x, y] = self.get_ray_direction(x/pixel_width, y/pixel_height)

    def get_ray_direction(self, x, y):
        #It just linear algebra and it just works
        x = x * self.screne_width - self.screne_width / 2
        y = y * self.screne_height - self.screne_height / 2
        direction = np.array([x, y, -1, 1])
        direction = self.projection_matrix.dot(direction)
        direction = self.lookAtMatrix.dot(direction)

        return direction[:3] / np.linalg.norm(direction[:3])

