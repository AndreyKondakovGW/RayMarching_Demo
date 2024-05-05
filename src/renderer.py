import pygame
import numpy as np
from src.camera import RayMarchCamera
from src.ray_marchobject import Sphere, FuzzySphere

class PyGameWindowRenderer:
    #simple class to render the game state render window from np array
    def __init__(self, width: int, height: int):
        self.screen_width = width
        self.screen_height = height
        self.screen = pygame.display.set_mode((width, height))
        
        self.world = [FuzzySphere(np.array([10, 0, 0]), 1, np.array([0, 1, 0])), Sphere(np.array([5, 1, 0]), 0.1, np.array([1, 1, 0])), Sphere(np.array([5, -1, 0]), 0.1, np.array([1, 0, 1]))]

        self.camera = RayMarchCamera(pos = np.array([0,0,0]), target=np.array([10,0,0]))
        self.camera.set_all_rays(self.screen_width, self.screen_height)
        self.window_content = self.camera.get_window_content(self.screen_width, self.screen_height, self.world)

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
