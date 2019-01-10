# =========================================================================== #
#                     STOCHASTIC GRADIENT DESCENT REPORT                      #
# =========================================================================== #
# %%
import numpy as np
import pandas as pd
import sys
src = "c:\\Users\\John\\Documents\\Data Science\\Libraries\\GradientDescent\\src"
sys.path.append(src)
from GradientLab import SGDLab
from GradientDemo import SGDDemo
from utils import save_csv
import data

# Data
X, X_val, y, y_val = data.demo(n=500)

# Process flags
cache = False
show = False
#%%
# --------------------------------------------------------------------------- #
#                                 GRIDSEARCH                                  #
# --------------------------------------------------------------------------- #
def sgd_gs(learning_rate=None, precision=None, miniter=None, maxiter=None, 
           stop_parameter=None, stop_metric=None, check_point=None,
           directory=None, filename=None):
    theta = np.array([-1,-1]) 
    if not learning_rate:
        learning_rate = [0.02, 0.04, 0.06, 0.08, 0.2, 0.4, 0.6, 0.8, 1, 1.5]
    if not precision:
        precision = [0.1, 0.01, 0.001, 0.0001]
    if not check_point:
        check_point = [0.01, 0.05, 0.1, 0.2]
    if not miniter:
        miniter=0
    if not maxiter:
        maxiter = 10000
    if not stop_parameter:
        stop_parameter = ['t', 'v', 'g']
    if not stop_metric:
        stop_metric = ['a', 'r']
    if not directory:
        directory = "./report/figures/SGD/"
    if not filename:
        filename = 'Stochastic Gradient Descent - Gridsearch Report.csv'

    # Run experiment
    if cache is False:
        lab = SGDLab()
        lab.gridsearch(X=X, y=y, X_val=X_val, y_val=y_val, theta=theta, learning_rate=learning_rate, precision=precision,
                    check_point=check_point, miniter=miniter, maxiter=maxiter, 
                    stop_parameter=stop_parameter, stop_metric=stop_metric)
        fig = lab.plot(directory=directory, show=show)
        lab.plot_learning_rate(directory=directory, show=show)
        lab.plot_check_point(directory=directory, show=show)
        lab.plot_precision(directory=directory, show=show)
        lab.plot_curves(stop_parameter='Training Set Costs', 
                        stop_metric='Relative Change',
                        directory=directory, show=show)
        report = lab.report(directory=directory, filename=filename)
        return(report)

# --------------------------------------------------------------------------- #
#                         GET FEATURED OBSERVATIONS                           #
# --------------------------------------------------------------------------- #
def sgd_featured(report, learning_rate):
    featured = pd.DataFrame()
    for a in learning_rate:
        f = report[report.learning_rate==a].nsmallest(1, 'final_mse', 'duration')
        featured = pd.concat([featured, f], axis=0)
    return(featured)
    
#%%        
# --------------------------------------------------------------------------- #
#                                    DEMO                                     #
# --------------------------------------------------------------------------- #
def sgd_demo(learning_rate, precision, maxiter, miniter, stop_parameter, stop_metric, 
             check_point=0.1, filename=None, search=True, fit=True, 
             fontsize=None, cache=False):
    theta = np.array([-1,-1]) 
    directory = "./report/figures/SGD/"
    if miniter:
        stub = ' Minimum Iterations - ' + str(miniter)
    else:
        stub = ''
    if filename:
        filename_search = filename + ' Search' + stub + '.gif'
        filename_fit = filename + ' Fit' + stub + '.gif'
    else:

        filename_search =  'Stochastic Gradient Descent Search - learning_rate ' + str(learning_rate) + stub + '.gif'
        filename_fit = 'Stochastic Gradient Descent Fit - learning_rate ' + str(learning_rate) + stub + '.gif'

    # Run Demo Animation
    if not cache:
        sgd = SGDDemo()
        sgd.fit(X=X, y=y, theta=theta, X_val=X_val, y_val=y_val, learning_rate=learning_rate, precision=precision, 
                check_point=check_point, miniter=miniter, maxiter=maxiter, 
                stop_parameter=stop_parameter, stop_metric=stop_metric)
        if search:
            sgd.show_search(directory=directory, filename=filename_search, 
                            fontsize=fontsize, fps=10, maxframes=500)
        if fit:
            sgd.show_fit(directory=directory, filename=filename_fit, 
                         fontsize=fontsize, fps=10, maxframes=500)

# --------------------------------------------------------------------------- #
#                              ANIMATIONS                                     #
# --------------------------------------------------------------------------- #
def sgd_ani(featured, filename=None, fontsize=None, cache=False):
    if cache is False:    
        sgd_demo(learning_rate=featured.learning_rate, precision=featured.precision,
        maxiter=featured.maxiter, miniter=featured.miniter,
        stop_parameter=featured.stop_parameter, 
        check_point=featured.check_point,
        stop_metric=featured.stop_metric,
        filename=filename, search=True, fit=True,
        fontsize=fontsize)

#%%

report = sgd_gs()
#%%
print(report.head())
sgd_ani(report.iloc[0])

