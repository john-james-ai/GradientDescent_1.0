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
        precision = 0.0001
        maxiter = 10000
        n_iter_no_change = 5
        directory = "./test/figures/BGD/"
        gd = BGD()
        gd.fit(X=X, y=y, theta=theta,X_val=X_val, y_val=y_val, alpha=alpha,
                maxiter=maxiter, precision=precision, n_iter_no_change=n_iter_no_change)
        gd.plot(directory=directory)        
        rpt = gd.summary()
        return(rpt)
#%%        
report = bgd()        
print(report)
