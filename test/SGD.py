# =========================================================================== #
#                       STOCHASTIC GRADIENT DESCENT TEST                      #
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
from GradientDescent import BGD, SGD
import data
#%%
# --------------------------------------------------------------------------- #
#                               TEST DRIVER                                   #
# --------------------------------------------------------------------------- #
X, X_val, y, y_val = data.demo()
theta = np.array([-1,-1])
learning_rate = 0.01
learning_rate_sched = 't'
stop_metric = 'g'
time_decay = 0.5
step_decay=0.5
step_epochs=2
exp_decay=0.1
precision = 0.01
maxiter = 5000
i_s = 5
directory = "./test/figures/SGD/"
gd = SGD()
gd.fit(X=X, y=y, theta=theta,X_val=X_val, y_val=y_val, learning_rate=learning_rate,
        learning_rate_sched=learning_rate_sched, time_decay=time_decay, step_decay=step_decay,
        step_epochs=step_epochs, exp_decay=exp_decay,
        maxiter=maxiter, precision=precision, i_s=i_s, stop_metric=stop_metric)
print(gd.summary())   
print(gd.detail())
print(gd.eval())

