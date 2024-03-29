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
X, X_val, y, y_val = data.demo()

# Parameters
theta = np.array([-1,-1]) 
batch_size=10
learning_rate = [0.01, 0.03, 0.1, 0.8]
precision = [0.1, 0.01, 0.001]
maxiter = 5000
learning_rate_sched = ['c', 't', 's', 'e']
time_decay = [0.1, 0.01]
step_decay = [0.1, 0.01]
step_epochs = [2,4]
exp_decay = [0.1, 0.01, 0.001]
i_s=[5,10]
directory = "./test/figures/SGD/Lab/"

#%%
# Run experiment
lab = MBGDLab()
lab.gridsearch(X=X, y=y, X_val=X_val, y_val=y_val, theta=theta, learning_rate=learning_rate, 
               learning_rate_sched=learning_rate_sched, time_decay=time_decay, 
               step_decay=step_decay, step_epochs=step_epochs, exp_decay=exp_decay,
               batch_size=batch_size, precision=precision, maxiter=maxiter, 
               i_s=i_s)
#%%%           
dfs = lab.summary()
dfd = lab.detail()
lab.figure(data=dfs, x='learning_rate', y='final_mse', z='learning_rate_sched',
           func=lab.barplot, directory=directory, show=True)
report = lab.report(directory=directory, filename='Minibatch Gradient Descent Report.csv')
print(report)