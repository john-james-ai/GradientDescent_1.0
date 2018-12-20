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
X, X_val, y, y_val = data.demo()

# Parameters
theta = np.array([-1,-1]) 
alpha = [0.01, 0.03, 0.05, 0.08, 0.1, 0.3, 0.5, 0.8, 1]
precision = [0.01, 0.001, 0.0001, 0.00001]
maxiter = 10000
stop_measure = 'j'
stop_metric = 'a'
path_plot = "./test/figures/BGD/bgd_lab_abs_cost.png"

# Run experiment
lab = BGDLab()
lab.fit(X=X, y=y, X_val=X_val, y_val=y_val, theta=theta, alpha=alpha, precision=precision,
           maxiter=maxiter, stop_measure=stop_measure, stop_metric=stop_metric)
rpt = lab.report()
print(rpt)
lab.plot(path=path_plot)
