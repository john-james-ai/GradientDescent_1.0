# =========================================================================== #
#                       BATCH GRADIENT DESCENT TEST                           #
# =========================================================================== #
# %%
import numpy as np 
import pandas as pd 
import sys
srcdir = "c:\\Users\\John\\Documents\\Data Science\\Libraries\\GradientDescent\\src"
sys.path.append(srcdir)

from GradientDescent import BGD 
import data 
#%%
# --------------------------------------------------------------------------- #
#                             SEARCH FUNCTION                                 #  
# --------------------------------------------------------------------------- #

X, X_val, y, y_val = data.demo()
params = {'X_val': X_val, 'y_val':y_val, 'theta': np.array([-1,-1]),
          'learning_rate': 0.01, 'learning_rate_sched': 'c', 
          'stop_metric' : 'v', 'time_decay' : 0.5,
          'step_decay' : 0.5, 'step_epochs': 2, 'exp_decay': 0.1,
          'precision' : 0.01, 'maxiter' : 5000, 'i_s':5}

gd = BGD()
gd.fit(X=X, y=y, **params)
print(gd.summary())   
print(gd.detail())
print(gd.eval())
#%%
