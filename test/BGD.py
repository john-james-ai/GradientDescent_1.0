# =========================================================================== #
#                       BATCH GRADIENT DESCENT TEST                           #
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
#                             SEARCH FUNCTION                                 #  
# --------------------------------------------------------------------------- #
def fit(X, y, theta, alpha, precision, maxiter, stop_parameter, stop_metric,
           directory, X_val=None, y_val=None):
        gd = BGD()
        gd.fit(X=X, y=y, theta=theta,X_val=X_val, y_val=y_val, alpha=alpha,
                maxiter=maxiter, precision=precision, stop_parameter=stop_parameter,
                stop_metric=stop_metric)
        gd.plot(directory=directory)
        gd.animate(directory=directory)
        rpt = gd.summary()
        return(rpt)
#%%        
# --------------------------------------------------------------------------- #
#                            BRIEF TEST DRIVER                                #
# --------------------------------------------------------------------------- #
def brief():
        X, X_val, y, y_val = data.demo()
        theta = np.array([-1,-1])
        alpha = 0.01
        precision = 0.01
        maxiter = 10000
        stop_parameter = 'g'
        stop_metric = 'a'
        directory = "./test/figures/BGD/"
   
        rpt = fit(X,y, theta, alpha, precision, maxiter, stop_parameter, stop_metric,
                directory, X_val, y_val,)
        return(rpt)

#%%
# --------------------------------------------------------------------------- #
#                             FULL TEST DRIVER                                #
# --------------------------------------------------------------------------- #
def full():
        X, X_val, y, y_val = data.demo()
        # Parameters
        X = [X] * 12
        y = [y] * 12
        theta = [np.array([-1,-1])] * 12
        alpha = [0.01] * 12
        precision = [0.001] * 12
        maxiter = [10000] * 12
        stop_parameter = [['t'] * 4, ['g'] * 4,['v'] * 4]
        stop_parameter = [item for sublist in stop_parameter for item in sublist]
        stop_metric = ['a','a','r','r','a','a','r','r','a','a','r','r']
        X_v = [None,X_val,None,X_val,None,X_val,None,X_val,X_val,X_val,X_val,X_val]
        y_v = [None,y_val,None,y_val,None,y_val,None,y_val,y_val,y_val,y_val,y_val]
        directory = ["./test/figures/BGD/"] * 12


        #%%
        #Instantiate and execute search
        report = pd.DataFrame()
        for x,y,xv,yv,t,a,p,m,sp,met,d in zip(
        X,y, X_v, y_v, theta, alpha, precision, maxiter, stop_parameter, 
        stop_metric, directory):
                rpt = fit(X=x, y=y, X_val=xv, y_val=yv, theta=t, alpha=a, maxiter=m, precision=p, stop_parameter=sp,
                        stop_metric=met, directory=d)
                report = pd.concat([report, rpt], axis=0, sort=True)
        return(report)
#%%        
report = full()        
print(report)
