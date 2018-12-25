# =========================================================================== #
#                    BATCH GRADIENT DESCENT LAB TEST                          #
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
from GradientLab import BGDLab
import data

# Data
X, X_val, y, y_val = data.ames()

# Parameters
theta = np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]) 
alpha = [0.01, 0.03]
precision = [0.1, 0.01]
maxiter = 5000
stop_parameter = ['t', 'v']
stop_metric = ['a', 'r']
directory = "./test/figures/BGD/Lab/"

#%%
# Run experiment
lab = BGDLab()
lab.gridsearch(X=X, y=y, X_val=X_val, y_val=y_val, theta=theta, alpha=alpha, precision=precision,
           maxiter=maxiter, stop_parameter=stop_parameter, stop_metric=stop_metric)
#%%%           
lab.plot_costs(x='alpha', z='precision', fig_key='stop_parameter', 
               row_key='stop_condition',  directory=directory, show=False)
# lab.plot_curves(directory=directory)
# lab.plot_times(directory=directory)
# lab.report(n=5)