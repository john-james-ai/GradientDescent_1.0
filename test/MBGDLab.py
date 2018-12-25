# =========================================================================== #
#                    MINI-BATCH GRADIENT DESCENT LAB TEST                     #
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
from GradientLab import MBGDLab
import data

# Data
X, X_val, y, y_val = data.ames()

# Parameters
theta = np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]) 
alpha = [0.01]
precision = [0.01]
maxiter = 5000
stop_parameter = ['g']
stop_metric = ['r']
batch_size = [0.01, 0.05, 0.1, 0.2]
directory = "./test/figures/MBGD/Lab/"

#%%
# Run experiment
lab = MBGDLab()
lab.gridsearch(X=X, y=y, X_val=X_val, y_val=y_val, theta=theta, alpha=alpha, precision=precision,
           maxiter=maxiter, stop_parameter=stop_parameter, stop_metric=stop_metric, batch_size=batch_size)
#%%%           
lab.plot_costs(directory=directory)
lab.plot_curves(directory=directory)
lab.plot_times(directory=directory)
lab.report(n=5)