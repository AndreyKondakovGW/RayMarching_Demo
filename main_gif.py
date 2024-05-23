from src.plt_renderer import PyPlotRenderer
from src.example_worlds import *

if __name__=="__main__":
    renderer = PyPlotRenderer(800, 600, world=world_3)
    renderer.render_gif(80)