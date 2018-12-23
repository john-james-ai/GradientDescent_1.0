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
def sgd():
        X, X_val, y, y_val = data.demo()
        theta = np.array([-1,-1])
        alpha = 0.01
        precision = 0.01
        maxiter = 10000
        check_point = .1
        stop_parameter = 't'
        stop_metric = 'a'
        directory = "./test/figures/SGD/"
        gd = SGD()
        gd.fit(X=X, y=y, theta=theta,X_val=X_val, y_val=y_val, 
                check_point=check_point, alpha=alpha,
                maxiter=maxiter, precision=precision, stop_parameter=stop_parameter,
                stop_metric=stop_metric)
        gd.plot(directory=directory)
        gd.animate(directory=directory)
        rpt = gd.summary()
        return(rpt)
#%%
report = sgd()        
print(report)
