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
learning_rate = [1.6]
stop_metric=['j']
precision = [0.001]
maxiter = 5000
learning_rate_sched = ['t']
time_decay = np.arange(0.01,1,0.01)
step_decay = [0.1, 0.01, 0.001]
step_epochs = [2,5, 10]
exp_decay = [0.1, 0.01, 0.001]
i_s=[10]
directory = "./test/figures/BGD/Lab/"

#%%
# Run experiment
lab = BGDLab()
lab.gridsearch(X=X, y=y, X_val=X_val, y_val=y_val, theta=theta, learning_rate=learning_rate, 
               learning_rate_sched=learning_rate_sched, time_decay=time_decay, 
               step_decay=step_decay, step_epochs=step_epochs, exp_decay=exp_decay,
               stop_metric=stop_metric, precision=precision, maxiter=maxiter, i_s=i_s)
#%%%           
# Render plots
dfs = lab.summary()
dfd = lab.detail()

print(dfd)