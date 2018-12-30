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
X, X_val, y, y_val = data.demo(n=500)

# Process flags
show = False

# --------------------------------------------------------------------------- #
#                                 GRIDSEARCH                                  #
# --------------------------------------------------------------------------- #
def bgd_gs(alpha=None, precision=None, miniter=None, maxiter=None, 
           stop_parameter=None, stop_metric=None, directory=None, filename=None):
    theta = np.array([-1,-1]) 
    if not alpha:
        alpha = [0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1]
    if not precision:
        precision = [0.1, 0.01, 0.001, 0.0001]
    if not miniter:
        miniter=0
    if not maxiter:
        maxiter = 10000
    if not stop_parameter:
        stop_parameter = ['t', 'v', 'g']
    if not stop_metric:
        stop_metric = ['a', 'r']
    if not directory:
        directory = "./report/figures/BGD/"
    if not filename:
        filename = 'Batch Gradient Descent - Gridsearch Report.csv'

    # Run experiment
    lab = BGDLab()
    lab.gridsearch(X=X, y=y, X_val=X_val, y_val=y_val, theta=theta, alpha=alpha, precision=precision,
            miniter=miniter, maxiter=maxiter, stop_parameter=stop_parameter, stop_metric=stop_metric)
    fig = lab.plot(directory=directory, show=show)
    report = lab.report(directory=directory, filename=filename)
    return(report)

# --------------------------------------------------------------------------- #
#                         GET FEATURED OBSERVATIONS                           #
# --------------------------------------------------------------------------- #
def bgd_featured(report, alpha):
    featured = pd.DataFrame()
    for a in alpha:
        f = report[report.alpha==a].nsmallest(1, 'final_costs_val')
        featured = pd.concat([featured, f], axis=0)
    return(featured)
    
#%%        
# --------------------------------------------------------------------------- #
#                                    DEMO                                     #
# --------------------------------------------------------------------------- #
def bgd_demo(alpha, precision, maxiter, miniter, stop_parameter, stop_metric, 
            filename=None, search=True, fit=True, fontsize=None, cache=False):
    theta = np.array([-1,-1]) 
    directory = "./report/figures/BGD/"
    if miniter:
        stub = ' Minimum Iterations - ' + str(miniter)
    else:
        stub = ''
    if filename:
        filename_search = filename + ' Search' + stub + '.gif'
        filename_fit = filename + ' Fit' + stub + '.gif'
    else:

        filename_search =  'Batch Gradient Descent Search - Alpha ' + str(alpha) + stub + '.gif'
        filename_fit = 'Batch Gradient Descent Fit - Alpha ' + str(alpha) + stub + '.gif'

    # Run Demo Animation
    if not cache:
        bgd = BGDDemo()
        bgd.fit(X=X, y=y, theta=theta, X_val=X_val, y_val=y_val, alpha=alpha, precision=precision,
                miniter=miniter, maxiter=maxiter, stop_parameter=stop_parameter, stop_metric=stop_metric)
        if search:
            bgd.show_search(directory=directory, filename=filename_search, fontsize=fontsize, fps=10)
        if fit:
            bgd.show_fit(directory=directory, filename=filename_fit, fontsize=fontsize, fps=10)

# --------------------------------------------------------------------------- #
#                              ANIMATIONS                                     #
# --------------------------------------------------------------------------- #
def bgd_ani(featured, filename=None, miniter=0, fontsize=None, cache=False):
    if cache is False:
        for idx, row in featured.iterrows():
            bgd_demo(alpha=row.alpha, precision=row.precision,
            maxiter=row.maxiter, miniter=row.miniter,
            stop_parameter=row.stop_parameter, 
            stop_metric=row.stop_metric,
            filename=filename, search=True, fit=True,
            fontsize=fontsize)


# report = bgd_gs()
# alpha = [0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1]
# best_by_alpha = bgd_featured(report, alpha)
# print(best_by_alpha.info())
# print(best_by_alpha)
# alpha =[0.02, 0.1, 0.8]
# featured = bgd_featured(report, alpha)
# print(featured)
# bgd_ani(featured)

