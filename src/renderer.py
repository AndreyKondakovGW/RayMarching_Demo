import pygame
import numpy as np
from src.camera import RayMarchCamera
from src.ray_marchobject import Sphere, FuzzySphere
import time

class PyGameWindowRenderer:
    #simple pygame logic
    #simple class to render the game state render window from np array
    def __init__(self, width: int, height: int):
        self.screen_width = width
        self.screen_height = height
        self.screen = pygame.display.set_mode((width, height))
        
        self.world = [Sphere(np.array([5, 0, 0], dtype=np.float64), 1, np.array([1, 1, 0], dtype=np.float64)),
                      FuzzySphere(np.array([7, 0, 5], dtype=np.float64), 1, np.array([1, 0, 0], dtype=np.float64))]

        self.camera = RayMarchCamera(pos = np.array([0,0,0], dtype=np.float64), target=np.array([10,0,0], dtype=np.float64), screne_width=width, screne_height=height)
        start = time.time()
        self.camera.set_all_rays(self.screen_width, self.screen_height)
        self.window_content = self.camera.get_window_content(self.screen_width, self.screen_height, self.world)
        print(f"Time to render {width}x{height}: {time.time() - start:.2f} sec", )

    def mainloop(self):
        running = True
        while running:
            self.screen.fill((255, 255, 255))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Exiting")
                    running = False
            self.screen.blit(pygame.surfarray.make_surface(self.window_content), (0, 0))
            pygame.display.flip()
