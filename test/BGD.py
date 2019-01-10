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
# --------------------------------------------------------------------------- #
#                             SEARCH FUNCTION                                 #  
# --------------------------------------------------------------------------- #
def bgd():
        X, X_val, y, y_val = data.demo()
        theta = np.array([-1,-1])
        learning_rate = 0.1
        learning_rate_sched = 'e'
        time_decay = None
        step_decay=0.5
        step_epochs=2
        exp_decay=0.1
        precision = 0.0001
        maxiter = 10000
        no_improvement_stop = 5
        directory = "./test/figures/BGD/"
        gd = BGD()
        gd.fit(X=X, y=y, theta=theta,X_val=X_val, y_val=y_val, learning_rate=learning_rate,
               learning_rate_sched=learning_rate_sched, time_decay=time_decay, step_decay=step_decay,
               step_epochs=step_epochs, exp_decay=exp_decay,
               maxiter=maxiter, precision=precision, no_improvement_stop=no_improvement_stop)
        return(gd)
#%%        
gd = bgd()        
print(gd.summary())
print(gd.detail())
