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
        alpha = 0.01
        precision = 0.01
        maxiter = 10000
        stop_parameter = 'g'
        stop_metric = 'a'
        directory = "./test/figures/BGD/"
        gd = BGD()
        gd.fit(X=X, y=y, theta=theta,X_val=X_val, y_val=y_val, alpha=alpha,
                maxiter=maxiter, precision=precision, stop_parameter=stop_parameter,
                stop_metric=stop_metric)
        gd.plot(directory=directory)
        gd.animate(directory=directory)
        rpt = gd.summary()
        return(rpt)
#%%        
report = bgd()        
print(report)
