from src.renderer import PyGameWindowRenderer
from src.ray_marchobject import Sphere
import numpy as np
if __name__=="__main__":
    renderer = PyGameWindowRenderer(700, 500)
    renderer.mainloop()
