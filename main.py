from src.pygame_renderer import PyGameWindowRenderer
#from src.plt_renderer import PyPlotRenderer

if __name__=="__main__":
    renderer = PyGameWindowRenderer(800, 600)
    renderer.mainloop()

    # renderer = PyPlotRenderer(800, 600)
    # renderer.render_gif(60)
