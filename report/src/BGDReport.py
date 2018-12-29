# =========================================================================== #
#                    BATCH GRADIENT DESCENT LAB TEST                          #
# =========================================================================== #
# %%
import numpy as np
import pandas as pd
import sys
src = "c:\\Users\\John\\Documents\\Data Science\\Libraries\\GradientDescent\\src"
sys.path.append(src)
from GradientLab import BGDLab
from GradientDemo import BGDDemo
from utils import save_csv
import data

# Data
X, X_val, y, y_val = data.demo(n=100)

# Process flags
show = False

# --------------------------------------------------------------------------- #
#                                 GRIDSEARCH                                  #
# --------------------------------------------------------------------------- #
def bgd_gs():
        theta = np.array([-1,-1]) 
        alpha = [0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1]
        #alpha = [0.02, 0.04, 0.06, 0.08, 0.1]
        precision = [0.1, 0.01, 0.001, 0.0001]
        miniter=100
        maxiter = 10000
        stop_parameter = ['t', 'v', 'g']
        stop_metric = ['a', 'r']
        directory = "./report/figures/BGD/"
        filename = 'Batch Gradient Descent - Gridsearch Report.csv'

        # Run experiment
        lab = BGDLab()
        lab.gridsearch(X=X, y=y, X_val=X_val, y_val=y_val, theta=theta, alpha=alpha, precision=precision,
                miniter=miniter, maxiter=maxiter, stop_parameter=stop_parameter, stop_metric=stop_metric)
        fig, featured = lab.plot(directory=directory, show=show)
        report = lab.report(directory=directory, filename=filename)
        return(report, featured)


#%%
# --------------------------------------------------------------------------- #
#                               Longest Solution                              #
# --------------------------------------------------------------------------- #
def bgd_longest(params):
        theta = np.array([-1,-1]) 
        alpha = params.alpha
        precision = params.precision
        miniter=100
        maxiter = 10000
        stop_parameter = params.stop_parameter[0].lower()
        stop_metric = params.stop_metric[0].lower()
        directory = "./report/figures/BGD/"
        filename_search =  "slowest search.gif"
        filename_fit = "slowest fit.gif"

        # Run Demo Animation
        bgd = BGDDemo()
        bgd.fit(X=X, y=y, theta=theta, X_val=X_val, y_val=y_val, alpha=alpha, precision=precision,
                miniter=miniter, maxiter=maxiter, stop_parameter=stop_parameter, stop_metric=stop_metric)
        bgd.show_search(directory=directory, filename=filename_search, fps=10)
        bgd.show_fit(directory=directory, filename=filename_fit, fps=10)
        
# --------------------------------------------------------------------------- #
#                               Fastest Solution                              #
# --------------------------------------------------------------------------- #
def bgd_fastest(params):
        theta = np.array([-1,-1]) 
        alpha = params.alpha
        precision = params.precision
        miniter=100
        maxiter = 10000
        stop_parameter = params.stop_parameter[0].lower()
        stop_metric = params.stop_metric[0].lower()
        directory = "./report/figures/BGD/"
        filename_search =  "fastest search.gif"
        filename_fit = "fastest fit.gif"

        # Run Demo Animation
        bgd = BGDDemo()
        bgd.fit(X=X, y=y, theta=theta, X_val=X_val, y_val=y_val, alpha=alpha, 
                precision=precision, miniter=miniter, maxiter=maxiter, 
                stop_parameter=stop_parameter, stop_metric=stop_metric)
        bgd.show_search(directory=directory, filename=filename_search, fps=10)
        bgd.show_fit(directory=directory, filename=filename_fit, fps=10)

# --------------------------------------------------------------------------- #
#                                Worst Solution                               #
# --------------------------------------------------------------------------- #
def bgd_worst(params):
        theta = np.array([-1,-1]) 
        alpha = params.alpha
        precision = params.precision
        miniter=100
        maxiter = 10000
        stop_parameter = params.stop_parameter[0].lower()
        stop_metric = params.stop_metric[0].lower()
        directory = "./report/figures/BGD/"
        filename_search =  "worst search.gif"
        filename_fit = "worst fit.gif"

        # Run Demo Animation
        bgd = BGDDemo()
        bgd.fit(X=X, y=y, theta=theta, X_val=X_val, y_val=y_val, alpha=alpha, precision=precision,
                miniter=miniter, maxiter=maxiter, stop_parameter=stop_parameter, stop_metric=stop_metric)
        bgd.show_search(directory=directory, filename=filename_search, fps=10)
        bgd.show_fit(directory=directory, filename=filename_fit, fps=10)

#%%        
# --------------------------------------------------------------------------- #
#                                 Best Solution                               #
# --------------------------------------------------------------------------- #
def bgd_best(params):
        theta = np.array([-1,-1]) 
        alpha = params.alpha
        precision = params.precision
        miniter=100
        maxiter = 10000
        stop_parameter = params.stop_parameter[0].lower()
        stop_metric = params.stop_metric[0].lower()
        directory = "./report/figures/BGD/"
        filename_search =  "best search.gif"
        filename_fit = "best fit.gif"        

        # Run Demo Animation
        bgd = BGDDemo()
        bgd.fit(X=X, y=y, theta=theta, X_val=X_val, y_val=y_val, alpha=alpha, precision=precision,
                miniter=miniter, maxiter=maxiter, stop_parameter=stop_parameter, stop_metric=stop_metric)
        bgd.show_search(directory=directory, filename=filename_search, fps=10)
        bgd.show_fit(directory=directory, filename=filename_fit, fps=10)

report, featured = bgd_gs()
print(featured)
bgd_best(featured.iloc[0])
bgd_worst(featured.iloc[1])
bgd_longest(featured.iloc[2])
bgd_fastest(featured.iloc[3])

