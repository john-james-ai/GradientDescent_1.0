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
precision = [0.001]
maxiter = [5000]
learning_rate_sched = ['t']
time_decay = [0.01]
step_decay = [0.1, 0.01, 0.001]
step_epochs = [2,4]
exp_decay = [0.1, 0.01, 0.001]
no_improvement_stop=[20]
directory = "./test/figures/BGD/Lab/"

#%%
# Run experiment
lab = BGDLab()
lab.gridsearch(X=X, y=y, X_val=X_val, y_val=y_val, theta=theta, learning_rate=learning_rate, 
               learning_rate_sched=learning_rate_sched, time_decay=time_decay, 
               step_decay=step_decay, step_epochs=step_epochs, exp_decay=exp_decay,
               precision=precision, maxiter=maxiter, no_improvement_stop=no_improvement_stop)
#%%%           
# Render plots
dfs = lab.summary()
dfd = lab.detail()
print(dfd.head())
lab.figure(data=dfd, x='iterations', y='learning_rates', 
           func=lab.lineplot, directory=directory, show=True)
report = lab.report(directory=directory)
print(report)
