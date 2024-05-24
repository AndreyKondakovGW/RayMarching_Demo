from src.pygame_renderer import PyGameWindowRenderer
from src.example_worlds import *

if __name__=="__main__":
    renderer = PyGameWindowRenderer(1000, 800, world=world_2)
    renderer.mainloop()
