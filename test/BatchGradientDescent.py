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

# Data
X, X_val, y, y_val = data.demo()
# --------------------------------------------------------------------------- #
#                             SEARCH FUNCTION                                 #  
# --------------------------------------------------------------------------- #
def search(X, y, theta, alpha, precision, maxiter, stop_measure, stop_metric,
           X_val=None, y_val=None, path_plot=None, path_animate=None):
        gd = BGD()
        gd.search(X=X, y=y, theta=theta,X_val=X_val, y_val=y_val, alpha=alpha,
                maxiter=maxiter, precision=precision, stop_measure=stop_measure,
                stop_metric=stop_metric)
        gd.plot(path=path_plot)
        gd.animate(path=path_animate)
        rpt = gd.summary()
        return(rpt)


# theta = np.array([-1,-1])
# alpha = 0.01
# precision = 0.01
# maxiter = 10000
# stop_measure = 'g'
# stop_metric = 'p'
# path_plot = "/BDG/figures/bgd_rel_chg_grad_cv.png"
# path_animate = "/BDG/figures/bgd_rel_chg_grad_cv.gif"
# rpt = search(X,y, theta, alpha, precision, maxiter, stop_measure, stop_metric,
#              X_val, y_val, path_plot, path_animate)
# print(rpt)




# --------------------------------------------------------------------------- #
#                               TEST DRIVER                                   #
# --------------------------------------------------------------------------- #
# Parameters
X = [X] * 12
y = [y] * 12
theta = [np.array([-1,-1])] * 12
alpha = [0.01] * 12
precision = [0.0001] * 12
maxiter = [10000] * 12
stop_measure = [['j'] * 4, ['g'] * 4,['v'] * 4]
stop_measure = [item for sublist in stop_measure for item in sublist]
stop_metric = ['a','a','p','p','a','a','p','p','a','a','p','p']
X_v = [None,X_val,None,X_val,None,X_val,None,X_val,None,X_val,None,X_val]
y_v = [None,y_val,None,y_val,None,y_val,None,y_val,None,y_val,None,y_val]
path_plot = ["./test/BGD/figures/bgd_abs_chg_cost.png",
             "./test/BGD/figures/bgd_abs_chg_cost_cv.png",
             "./test/BGD/figures/bgd_rel_chg_cost.png",
             "./test/BGD/figures/bgd_rel_chg_cost_cv.png",

             "./test/BGD/figures/bgd_abs_chg_grad.png",
             "./test/BGD/figures/bgd_abs_chg_grad_cv.png",
             "./test/BGD/figures/bgd_rel_chg_grad.png",
             "./test/BGD/figures/bgd_rel_chg_grad_cv.png",

             "./test/BGD/figures/bgd_abs_chg_mse.png",
             "./test/BGD/figures/bgd_abs_chg_mse_cv.png",
             "./test/BGD/figures/bgd_rel_chg_mse.png",
             "./test/BGD/figures/bgd_rel_chg_mse_cv.png"]

path_animation = ["./test/BGD/figures/bgd_abs_chg_cost.gif",
             "./test/BGD/figures/bgd_abs_chg_cost_cv.gif",
             "./test/BGD/figures/bgd_rel_chg_cost.gif",
             "./test/BGD/figures/bgd_rel_chg_cost_cv.gif",

             "./test/BGD/figures/bgd_abs_chg_grad.gif",
             "./test/BGD/figures/bgd_abs_chg_grad_cv.gif",
             "./test/BGD/figures/bgd_rel_chg_grad.gif",
             "./test/BGD/figures/bgd_rel_chg_grad_cv.gif",

             "./test/BGD/figures/bgd_abs_chg_mse.gif",
             "./test/BGD/figures/bgd_abs_chg_mse_cv.gif",
             "./test/BGD/figures/bgd_rel_chg_mse.gif",
             "./test/BGD/figures/bgd_rel_chg_mse_cv.gif"]


#%%
#Instantiate and execute search
report = pd.DataFrame()
for x,y,xv,yv,t,a,p,m,meas,met,pp, pa in zip(
    X,y, X_v, y_v, theta, alpha, precision, maxiter, stop_measure, 
    stop_metric, path_plot, path_animation):

    rpt = search(X=x, y=y, X_val=xv, y_val=yv, theta=t, alpha=a, maxiter=m, precision=p, stop_measure=meas,
            stop_metric=met, path_plot=pp, path_animate=pa)
    report = pd.concat([report, rpt], axis=0)
print(report)
