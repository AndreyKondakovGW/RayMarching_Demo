import numpy as np
from src.ray_march import ray_march, ray_march_kernel
from numba import jit
from numba import int32, float64
from numba.experimental import jitclass
from numba import cuda

spec = [
    ('pos', float64[:]),
    ('dir', float64[:]),
    ('fov', int32),
    ('aspect_ratio', float64),
    ('near', float64),
    ('far', float64),
    ('screne_center', float64[:]),
    ('screne_width', float64),
    ('screne_height', float64),
    ('projection_matrix', float64[:,:]),
    ('lookAtMatrix', float64[:,:]),
    ('all_rays', float64[:,:,:]),
    ('right', float64[:]),
    ('up', float64[:]),
]

#Cuda kernel for get_all_rays
@cuda.jit
def get_all_rays_kernel(all_rays, pixel_width, pixel_height, projection_matrix, lookAtMatrix):
    x, y = cuda.grid(2)
    if x < all_rays.shape[0] and y < all_rays.shape[1]:
        ndc_x = (2.0 * x) / pixel_width - 1.0
        ndc_y = 1.0 - (2.0 * y) / pixel_height
        ndc_z = 1

        #I hate my life
        #direction = projection_matrix.dot(direction)
        dir_x = projection_matrix[0,0] * ndc_x + projection_matrix[0,1] * ndc_y + projection_matrix[0,2] * ndc_z + projection_matrix[0,3] * 1
        dir_y = projection_matrix[1,0] * ndc_x + projection_matrix[1,1] * ndc_y + projection_matrix[1,2] * ndc_z + projection_matrix[1,3] * 1
        dir_z = projection_matrix[2,0] * ndc_x + projection_matrix[2,1] * ndc_y + projection_matrix[2,2] * ndc_z + projection_matrix[2,3] * 1
        dir_w = projection_matrix[3,0] * ndc_x + projection_matrix[3,1] * ndc_y + projection_matrix[3,2] * ndc_z + projection_matrix[3,3] * 1

        dir_x = dir_x / dir_w
        dir_y = dir_y / dir_w
        dir_z = dir_z / dir_w
        dir_w = dir_w / dir_w

        dir_z = 1
        dir_w = 0.0


        #direction = lookAtMatrix.dot(direction)
        w_x = lookAtMatrix[0,0] * dir_x + lookAtMatrix[0,1] * dir_y + lookAtMatrix[0,2] * dir_z + lookAtMatrix[0,3] * dir_w
        w_y = lookAtMatrix[1,0] * dir_x + lookAtMatrix[1,1] * dir_y + lookAtMatrix[1,2] * dir_z + lookAtMatrix[1,3] * dir_w
        w_z = lookAtMatrix[2,0] * dir_x + lookAtMatrix[2,1] * dir_y + lookAtMatrix[2,2] * dir_z + lookAtMatrix[2,3] * dir_w
        w_w = lookAtMatrix[3,0] * dir_x + lookAtMatrix[3,1] * dir_y + lookAtMatrix[3,2] * dir_z + lookAtMatrix[3,3] * dir_w


        cam_pos_x = lookAtMatrix[0,3]
        cam_pos_y = lookAtMatrix[1,3]
        cam_pos_z = lookAtMatrix[2,3]

        dir_x = w_x - cam_pos_x
        dir_y = w_y - cam_pos_y
        dir_z = w_z - cam_pos_z

        #np.linalg.norm(direction[:3])
        dir_norm = 0.0
        dir_norm += dir_x ** 2
        dir_norm += dir_y ** 2
        dir_norm += dir_z ** 2
        dir_norm = dir_norm ** 0.5

        #direction[:3] / dir_norm
        if dir_norm == 0:
            dir_norm = 1
        dir_x = dir_x / dir_norm
        dir_y = dir_y / dir_norm
        dir_z = dir_z / dir_norm

        all_rays[x, y, 0] = dir_x
        all_rays[x, y, 1] = dir_y
        all_rays[x, y, 2] = dir_z



#Jitclass only support static methods
@jit(nopython=True)
def get_all_rays_satatic(pixel_width, pixel_height, screne_width, screne_height, projection_matrix, lookAtMatrix):
    all_rays = np.zeros((pixel_width, pixel_height, 3), dtype=np.float64)
    for x in range(pixel_width):
        for y in range(pixel_height):
            x_loc = x * screne_width - screne_width / 2
            y_loc = y * screne_height - screne_height / 2
            direction = np.array([x_loc, y_loc, -1, 1], dtype=np.float64)
            direction = projection_matrix.dot(direction)
            direction = lookAtMatrix.dot(direction)
            all_rays[x, y] = direction[:3] / np.linalg.norm(direction[:3])
    return all_rays

#@jitclass(spec)
class RayMarchCamera:
    def __init__(self, pos = np.array([0,0,0]), target = np.array([1,0,0]), fov = 90, screne_width = 1, screne_height = 1):
        self.pos = pos
        dir = target - pos
        self.dir = dir
        self.dir = self.dir / np.linalg.norm(self.dir)
        self.fov = fov
        self.aspect_ratio =  screne_height / screne_width
        self.near = 1.0
        self.far = 1000.0

        self.projection_matrix = self.get_projection()
        self.count_axis_param()
        self.lookAtMatrix = self.count_lookAtMatrix()

    def count_axis_param(self):
        self.right = np.cross(self.dir, np.array([0,1,0], dtype=np.float64))
        if np.linalg.norm(self.right) != 0:
            self.right = self.right / np.linalg.norm(self.right)
        self.up = np.cross(self.right, self.dir)

    def count_lookAtMatrix(self):
        axis_matrix = np.array([
            [self.right[0], self.right[1], self.right[2], 0.0],
            [self.up[0], self.up[1], self.up[2], 0.0],
            [self.dir[0], self.dir[1], self.dir[2], 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

        position_matrix = np.array([
            [1.0, 0.0, 0.0, -self.pos[0]],
            [0.0, 1.0, 0.0, -self.pos[1]],
            [0.0, 0.0, 1.0, -self.pos[2]],
            [0.0, 0.0, 0.0, 1.0]
        ])

        return axis_matrix.dot(position_matrix)
        #return np.zeros((4,4))

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
        ], dtype=np.float64)
    
    def get_window_content(self, pixel_width, pixel_height, world):
        output = np.zeros((pixel_width, pixel_height, 3), dtype=np.float64)
        objects_buffer = np.array([obj.to_array() for obj in world], dtype=np.float64)


        threadsperblock = (16, 16)
        blockspergrid_x = int(np.ceil(pixel_width / threadsperblock[0]))
        blockspergrid_y = int(np.ceil(pixel_height / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        stream = cuda.stream()
        d_all_rays = cuda.to_device(self.all_rays, stream)
        d_output = cuda.to_device(output, stream)
        d_objects_buffer = cuda.to_device(objects_buffer, stream)
        ray_march_kernel[blockspergrid, threadsperblock](d_output, 0,0,0, d_all_rays, d_objects_buffer)
        output = d_output.copy_to_host(stream=stream)
        return (output * 255).astype(np.uint8)

    def set_all_rays(self, pixel_width, pixel_height):
        #calculate all rays directions
        threadsperblock = (16, 16)
        blockspergrid_x = int(np.ceil(pixel_width / threadsperblock[0]))
        blockspergrid_y = int(np.ceil(pixel_height / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        self.all_rays = np.zeros((pixel_width, pixel_height, 3), dtype=np.float64)
        stream = cuda.stream()
        d_all_rays = cuda.to_device(self.all_rays, stream)
        d_projetion_matrix = cuda.to_device(np.linalg.inv(self.projection_matrix), stream)
        d_lookAtMatrix = cuda.to_device(np.linalg.inv(self.lookAtMatrix), stream)
        get_all_rays_kernel[blockspergrid, threadsperblock](d_all_rays,
                                                             pixel_width, pixel_height,
                                                             d_projetion_matrix, d_lookAtMatrix)
        self.all_rays = d_all_rays.copy_to_host(stream=stream)
        #self.all_rays = get_all_rays_satatic(pixel_width, pixel_height, self.screne_width, self.screne_height, self.projection_matrix, self.lookAtMatrix)

    def get_ray_direction(self, x, y):
        #It just linear algebra and it just works
        x = x * self.screne_width - self.screne_width / 2
        y = y * self.screne_height - self.screne_height / 2
        direction = np.array([x, y, -1, 1])
        direction = self.projection_matrix.dot(direction)
        direction = self.lookAtMatrix.dot(direction)
        return direction[:3] / np.linalg.norm(direction[:3])

