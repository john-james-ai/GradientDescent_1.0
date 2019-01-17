# =========================================================================== #
#                       BATCH GRADIENT DESCENT TEST                           #
# =========================================================================== #
# %%
import numpy as np 
import pandas as pd 
import sys
srcdir = "c:\\Users\\John\\Documents\\Data Science\\Libraries\\GradientDescent\\src"
sys.path.append(srcdir)

from GradientDescent import BGD, SGD 
import data 
#%%


# Data
X, X_val, y, y_val = data.demo()

# Parameters
theta = np.array([-1,-1])
learning_rate = 0.1
learning_rate_sched = 'c'
stop_metric = 't'
time_decay = 0.5
step_decay=0.5
step_epochs=2
exp_decay=0.1
precision = 0.01
maxiter = 10000
i_s = 5
directory = "./test/figures/BGD/"
gd = BGD(theta=theta,X_val=X_val, y_val=y_val, learning_rate=learning_rate,
        learning_rate_sched=learning_rate_sched, time_decay=time_decay, step_decay=step_decay,
        step_epochs=step_epochs, exp_decay=exp_decay,
        maxiter=maxiter, precision=precision, i_s=i_s, stop_metric=stop_metric)
#%%        
gd.get_params()