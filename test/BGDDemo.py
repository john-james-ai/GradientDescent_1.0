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

# Parameters
theta = np.array([-1,-1]) 
alpha = 0.01
precision = 0.0001
maxiter = 10000
stop_measure = 'j'
stop_metric = 'a'
path_search = "./test/figures/BGD/BGDDemoSearch.gif"
path_fit = "./test/figures/BGD/BGDDemoFit.gif"

# Run experiment
demo = BGDDemo()
demo.fit(n=500, theta=theta, alpha=alpha, precision=precision,
           maxiter=maxiter, stop_measure=stop_measure, stop_metric=stop_metric)
#%%           
demo.show_search(path=path_search, maxframes=100)
demo.show_fit(path=path_fit)

