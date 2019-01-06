# =========================================================================== #
#                    STOCHASTIC GRADIENT DESCENT LAB TEST                     #
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
from GradientLab import SGDLab
import data

# Data
X, X_val, y, y_val = data.demo(n=500)

# Parameters
theta = np.array([-1,-1]) 
alpha = [0.01, 0.05, 0.1, 1]
improvement = [5,10,15]
precision = [0.1, 0.01]
maxiter = 5000
directory = "./test/figures/SGD/Lab/"

#%%
# Run experiment
lab = SGDLab()
lab.gridsearch(X=X, y=y, X_val=X_val, y_val=y_val, theta=theta, alpha=alpha, precision=precision,
           maxiter=maxiter, improvement=improvement)
#%%%           
dfs = lab.summary()
dfd = lab.get_detail()
lab.figure(data=dfs, x='precision', y='duration', z='alpha',
           func=lab.barplot, directory=directory, show=True)
report = lab.report(directory=directory, filename='Stochastic Gradient Descent Report.csv')