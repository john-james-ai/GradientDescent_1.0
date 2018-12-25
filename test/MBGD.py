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
        alpha = 0.01
        precision = 0.001
        maxiter = 10000
        stop_parameter = 'g'
        stop_metric = 'a'
        directory = "./test/figures/BGD/"
        batch_size = 5
        gd = MBGD()
        gd.fit(X=X, y=y, theta=theta,X_val=X_val, y_val=y_val, alpha=alpha,
                maxiter=maxiter, precision=precision, stop_parameter=stop_parameter,
                stop_metric=stop_metric, batch_size=batch_size)
        gd.plot(directory=directory)
        gd.animate(directory=directory)
        rpt = gd.summary()
        return(rpt)
#%%        
report = mbgd()        
print(report)
