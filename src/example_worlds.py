from src.ray_marchobject import *
import time 
class World:
    '''
    Class to store world object and animation
    '''
    def __init__(self, objects: RayMarchObject, animation):
        self.objects = objects
        self.animation = animation

#Example 1
object_list_1 = [Sphere(center = np.array([3, 0, 0]),
                        radius =  0.5,
                        color = np.array([1, 0, 0])),
                Box(center = np.array([3, 1, 0]),
                        half_sides=np.array([1, 0.1, 1]),
                        color = np.array([1, 1, 0]),
                        rotation_dir = np.array([0, 1, 0]),
                        rotation_angle = 45), 
                ]
def animation_1(objects):
    objects[0].center = np.array([3, 0 + 1.5*np.sin(time.time()), 0], dtype=np.float64)
    objects[1].rotation_angle += 1
    if objects[1].rotation_angle >= 360:
        objects[1].rotation_angle = 0
    return objects

world_1 = World(object_list_1, animation_1)


#Example 2
object_list_2 = [PlaneY(y = 1.5,
                        color = np.array([0, 204/255, 204/255])),
                FuzzySphere(center = np.array([3, 0, 0]),
                        radius =  0.5,
                        color = np.array([1, 0, 0]),
                        multipy_x = 1,
                        multipy_y = 1,
                        multipy_z = 1,),
                FrameBox(center = np.array([5, 1.5, 0]),
                        half_sides=np.array([2, 2, 2]),
                        color = np.array([1, 1, 0]),
                        thickness=0.5,
                        rotation_dir = np.array([1, 1, 0]),
                        rotation_angle = 45),
                 ]

def animation_2(objects):
    objects[1].center = np.array([3, 0 + 3*np.sin(time.time()), 1*np.sin(time.time())], dtype=np.float64)
    objects[1].scaler = 10 + 5.0*np.cos(time.time())
    objects[1].color = np.array([1, 0, 0]) + np.array([0, 0, 1])*np.sin(time.time())
    return objects

world_2 = World(object_list_2, animation_2)

#Example 3
object_list_3 = [Cylinder(center = np.array([3, 3, 0]),
                        radius = 5,
                        height = 0.1,
                        color = np.array([1, 0, 127/255]),
                        rotation_dir=np.array([0, 1, 0])),
                FuzzySphere(center = np.array([3, 0, 0]),
                        radius =  0.3,
                        color = np.array([1, 0, 0]),
                        multipy_x = 0,
                        multipy_y = 1,
                        multipy_z = 1,
                        multipy_dist=0.35,
                        ),
                ]

def animation_3(objects):
    objects[0].rotation_angle += 1
    if objects[0].rotation_angle >= 360:
        objects[0].rotation_angle = 0

    objects[1].center = np.array([3, 0 + 5*np.sin(time.time()/ 10), 0], dtype=np.float64)
    objects[1].scaler = 10 + 5.0*np.cos(time.time()/ 8)
    objects[1].color = np.array([1, 0, 0]) + np.array([0, 0, 1])*np.sin(time.time() / 15) + np.array([0, 1, 0])*np.cos(time.time()/ 15)
    return objects

world_3 = World(object_list_3, animation_3)

#Example 4

object_list_4 = [Cone(center=np.array([3, 1, 0]),
                        c = np.array([np.cos(np.pi/6), np.sin(np.pi/6)]),
                        height = 1,
                        rotation_dir = np.array([0, 1, 0]),
                        rotation_angle = 0,
                        color = np.array([1, 0, 0]),),]

def animation_4(objects):
    objects[0].rotation_angle += 1
    if objects[0].rotation_angle >= 360:
        objects[0].rotation_angle = 0
    return objects

world_4 = World(object_list_4, animation_4)
