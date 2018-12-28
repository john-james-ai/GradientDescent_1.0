
# %%
# =========================================================================== #
#                             UTILITY FUNCTIONS                               #
# =========================================================================== #
import os

from matplotlib import animation, rc
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import pandas as pd

def save_fig(fig, directory, filename):
    if os.path.exists(directory):
        path = os.path.join(os.path.abspath(directory), filename)
        fig.savefig(path, facecolor='w')
    else:
        os.makedirs(directory)
        path = os.path.join(os.path.abspath(directory),filename)
        fig.savefig(path, facecolor='w')

def save_gif(ani, directory, filename, fps):
    face_edge_colors = {'facecolor': 'w', 'edgecolor': 'w'}
    path = os.path.join(directory, filename)
    if os.path.exists(directory):
        ani.save(path, writer='imagemagick', fps=fps, savefig_kwargs = face_edge_colors)
    else:
        os.makedirs(directory)                
        ani.save(path, writer='imagemagick', fps=fps, savefig_kwargs = face_edge_colors)

def save_csv(df, directory, filename):
    path = os.path.join(directory, filename)
    if os.path.exists(directory):
        df.to_csv(path)
    else:
        os.makedirs(directory)                
        df.to_csv(path)
