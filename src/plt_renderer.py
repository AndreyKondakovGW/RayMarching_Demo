import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from src.camera import RayMarchCamera
from src.ray_marchobject import *
import time
from src.example_worlds import *

class PyPlotRenderer:
    '''
    class to generate gif from matplotlib
    '''
    def __init__(self, width: int, height: int, world: World):
        self.screen_width = width
        self.screen_height = height
        self.world = world

        self.camera = RayMarchCamera(pos = np.array([0,0,0], dtype=np.float64), target=np.array([10,0,0], dtype=np.float64), screne_width=width, screne_height=height)
        
        start = time.time()
        self.camera.set_all_rays(self.screen_width, self.screen_height)
        self.window_content = self.camera.get_window_content(self.screen_width, self.screen_height, self.world.objects)
        print(f"Time to render {width}x{height}: {time.time() - start:.2f} sec", )
        self.figure  = plt.figure(figsize=(15, 15))
        self.ax = self.figure.add_subplot(111)
        #plt.axis('off')

    def render_gif(self, seconds: int):
        def update(i):
            self.world.objects = self.world.animation(self.world.objects)
            self.window_content = self.camera.get_window_content(self.screen_width, self.screen_height, self.world.objects)
            img = plt.imshow(np.rot90(np.rot90(np.rot90(self.window_content))))
            plt.axis('off')
            #plt.tight_layout()
            return img,
        
        ani = FuncAnimation(self.figure, update, frames=seconds, blit=True)
        ani.save('./animation.gif', writer='imagemagick', fps=5)