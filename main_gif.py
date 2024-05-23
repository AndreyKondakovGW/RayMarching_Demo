from src.plt_renderer import PyPlotRenderer

if __name__=="__main__":
    renderer = PyPlotRenderer(800, 600)
    renderer.render_gif(60)