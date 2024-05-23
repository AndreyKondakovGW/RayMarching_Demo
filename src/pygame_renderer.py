import pygame
import numpy as np
from src.camera import RayMarchCamera
from src.ray_marchobject import *
import time
from src.example_worlds import *

class PyGameWindowRenderer:
    def __init__(self, width: int, height: int, world: World) -> None:
        self.screen_width = width
        self.screen_height = height
        self.screen = pygame.display.set_mode((width, height))
        
        #initial set of objects
        self.world = world

        self.camera = RayMarchCamera(pos = np.array([0,0,0], dtype=np.float64), target=np.array([10,0,0], dtype=np.float64), screne_width=width, screne_height=height)
        
        #calculate time for inital rendering
        start = time.time()
        self.camera.set_all_rays(self.screen_width, self.screen_height)
        self.window_content = self.camera.get_window_content(self.screen_width, self.screen_height, self.world.objects)
        print(f"Time to render {width}x{height}: {time.time() - start:.2f} sec")

    def mainloop(self) -> None:
        running = True
        num_frames = 0
        while running:
            if num_frames % 60 == 0:
                self.screen.fill((255, 255, 255))
                self.world.objects = self.world.animation(self.world.objects)
                self.window_content = self.camera.get_window_content(self.screen_width, self.screen_height, self.world.objects)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Exiting")
                    running = False

            num_frames += 1
            self.screen.blit(pygame.surfarray.make_surface(self.window_content), (0, 0))
            pygame.display.flip()
