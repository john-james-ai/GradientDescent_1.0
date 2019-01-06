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
X, X_val, y, y_val = data.demo(n=500)

# Parameters
theta = np.array([-1,-1]) 
alpha = [0.01, 0.03, 0.1, 0.8]
precision = [0.1, 0.01, 0.001, 0.0001]
maxiter = 5000
improvement=[5,10]
directory = "./test/figures/BGD/Lab/"

#%%
# Run experiment
lab = BGDLab()
lab.gridsearch(X=X, y=y, X_val=X_val, y_val=y_val, theta=theta, alpha=alpha, precision=precision,
           maxiter=maxiter, improvement=improvement)
#%%%           
# Render plots
dfs = lab.summary()
dfd = lab.get_detail()
lab.figure(data=dfs, x='precision', y='duration', z='alpha',
           func=lab.barplot, directory=directory, show=True)
report = lab.report(directory=directory)
