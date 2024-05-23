from src.pygame_renderer import PyGameWindowRenderer
from src.example_worlds import *

if __name__=="__main__":
    renderer = PyGameWindowRenderer(800, 600, world=world_3)
    renderer.mainloop()
