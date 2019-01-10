# =========================================================================== #
#                   BATCH GRADIENT DESCENT DEMO TEST                          #
# =========================================================================== #
# %%
import inspect
import os
import sys

from IPython.display import HTML
from matplotlib import animation, rc
from matplotlib import rcParams
import numpy as np
import pandas as pd

src = "c:\\Users\\John\\Documents\\Data Science\\Libraries\\GradientDescent\\src"
sys.path.append(src)
from GradientDemo import BGDDemo
import data

# Data
X, X_val, y, y_val = data.demo(n=500)

# Parameters
theta = np.array([-1,-1]) 
learning_rate = 1
precision = 0.001
maxiter = 10000
no_improvement_stop=5
directory = "./test/figures/BGD/"

# Run experiment
demo = BGDDemo()
demo.fit(X=X,y=y, theta=theta, learning_rate=learning_rate, precision=precision,
           maxiter=maxiter, no_improvement_stop=no_improvement_stop)
#%%           
demo.show_search(directory=directory, fps=1)
#demo.show_fit(directory=directory, fps=1)

