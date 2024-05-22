import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from src.camera import RayMarchCamera
from src.ray_marchobject import Sphere, FuzzySphere, PlaneY
import time
from matplotlib import transforms

class PyPlotRenderer:
    '''
    class to generate gif from matplotlib
    '''
    def __init__(self, width: int, height: int):
        self.screen_width = width
        self.screen_height = height
        self.world = [PlaneY(1, np.array([1,1,1])),
                      FuzzySphere(np.array([5, 0, 0], dtype=np.float64), 1, np.array([1, 1, 0], dtype=np.float64)),
                      Sphere(np.array([5, 0, 3], dtype=np.float64), 0.5, np.array([1, 0, 0], dtype=np.float64)),
                      Sphere(np.array([5, 0, -3], dtype=np.float64), 0.5, np.array([0, 0, 1], dtype=np.float64))]

        self.camera = RayMarchCamera(pos = np.array([0,0,0], dtype=np.float64), target=np.array([10,0,0], dtype=np.float64), screne_width=width, screne_height=height)
        
        start = time.time()
        self.camera.set_all_rays(self.screen_width, self.screen_height)
        self.window_content = self.camera.get_window_content(self.screen_width, self.screen_height, self.world)
        print(f"Time to render {width}x{height}: {time.time() - start:.2f} sec", )
        self.figure  = plt.figure(figsize=(15, 15))
        self.ax = self.figure.add_subplot(111)
        #plt.axis('off')

    def render_gif(self, seconds: int):
        tr = transforms.Affine2D().rotate_deg(90)
        def update(i):
            center_pos = self.world[0].center
            self.world[0].scaler = 2 + 5 * np.sin(np.pi *2 * i / seconds)
            self.world[1].center = np.array([center_pos[0] + 3 * np.sin(np.pi *2 * i / seconds), center_pos[1], center_pos[2] + 3 * np.cos(np.pi *2 * i / seconds)], dtype=np.float64)
            self.world[2].center = np.array([center_pos[0] + 3 * np.sin(np.pi *2 * i / seconds), center_pos[1], center_pos[2] - 3 * np.cos(np.pi *2 * i / seconds)], dtype=np.float64)
            self.window_content = self.camera.get_window_content(self.screen_width, self.screen_height, self.world)
            img = plt.imshow(np.rot90(self.window_content))
            plt.axis('off')
            plt.tight_layout()
            return img,
        
        ani = FuncAnimation(self.figure, update, frames=seconds, blit=True)
        ani.save('./animation.gif', writer='imagemagick', fps=5)