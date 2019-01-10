# =========================================================================== #
#                    MINI-BATCH GRADIENT DESCENT TEST                         #
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
from GradientDescent import MBGD
import data
#%%
# --------------------------------------------------------------------------- #
#                             SEARCH FUNCTION                                 #  
# --------------------------------------------------------------------------- #
def mbgd():
        X, X_val, y, y_val = data.demo()
        theta = np.array([-1,-1])
        learning_rate = 0.01
        learning_rate_sched = 't'
        time_decay = 0.1
        step_decay=0.5
        step_epochs=2
        exp_decay=0.1             
        precision = 0.001
        maxiter = 10000
        directory = "./test/figures/BGD/"
        no_improvement_stop=5
        batch_size = 5
        gd = MBGD()
        gd.fit(X=X, y=y, theta=theta,X_val=X_val, y_val=y_val, learning_rate=learning_rate,
               learning_rate_sched=learning_rate_sched, time_decay=time_decay, step_decay=step_decay,
               step_epochs=step_epochs, exp_decay=exp_decay,
               maxiter=maxiter, precision=precision, no_improvement_stop=no_improvement_stop) 
        return(gd)
#%%        
gd = mbgd()  
print(gd.summary())      
print(gd.detail())

