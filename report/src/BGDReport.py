# =========================================================================== #
#                     BATCH GRADIENT DESCENT REPORT                           #
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
show = True
#%%
# --------------------------------------------------------------------------- #
#                                 GRIDSEARCH                                  #
# --------------------------------------------------------------------------- #
def bgd_gs(learning_rate=None, precision=None, maxiter=None, 
           i_s=None, directory=None, filename=None):
    theta = np.array([-1,-1]) 
    if not learning_rate:
        learning_rate = [0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1]
    if not precision:
        precision = [0.1, 0.01, 0.001, 0.0001]
    if not maxiter:
        maxiter = 10000
    if not i_s:
        i_s = [5,10,20]
    if not directory:
        directory = "./report/figures/BGD/"
    if not filename:
        filename = 'Batch Gradient Descent - Gridsearch Report.csv'

    # Run experiment
    lab = BGDLab()
    lab.gridsearch(X=X, y=y, X_val=X_val, y_val=y_val, theta=theta, learning_rate=learning_rate, precision=precision,
                   maxiter=maxiter, i_s=i_s)
    # Obtain plot data    
    dfs = lab.summary()
    dfd = lab.detail()

    # Plot Scatterplot of Costs and Time by learning_rate Group By i_s and precision
    lab.figure(data=dfs, x='duration', y='final_mse', z='learning_rate',
               func=lab.scatterplot, directory=directory, show=show)
    # Plot Costs by learning_rate and Precision Group By i_s
    lab.figure(data=dfs, x='learning_rate', y='final_mse', z='precision', 
               groupby=['i_s'],
               func=lab.barplot, directory=directory, show=show)
    # Plot Time by learning_rate and Precision Group By i_s
    lab.figure(data=dfs, x='learning_rate', y='duration', z='precision', 
               groupby=['i_s'],
               func=lab.barplot, directory=directory, show=show)               
    # Plot Learning Curves 
    lab.figure(data=dfd, x='iterations', y='cost', z='learning_rate', 
               groupby=['i_s','precision'],
               func=lab.lineplot, directory=directory, show=show)               
    # Plot i_s Parameter Costs
    lab.figure(data=dfs, x='i_s', y='final_mse', z='precision',                
               func=lab.barplot, directory=directory, show=show)               
    # Plot i_s Parameter Computation
    lab.figure(data=dfs, x='i_s', y='duration', z='precision',                
               func=lab.barplot, directory=directory, show=show)        
    report = lab.report(directory=directory, filename=filename)
    return(report)

# --------------------------------------------------------------------------- #
#                         GET FEATURED OBSERVATIONS                           #
# --------------------------------------------------------------------------- #
def bgd_featured(report, learning_rate):
    featured = pd.DataFrame()
    for a in learning_rate:
        f = report[report.learning_rate==a].nsmallest(1, 'final_mse')
        featured = pd.concat([featured, f], axis=0)
    return(featured)
    
#%%        
# --------------------------------------------------------------------------- #
#                                    DEMO                                     #
# --------------------------------------------------------------------------- #
def bgd_demo(learning_rate, precision, maxiter, miniter, stop_parameter, stop_metric, 
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

        filename_search =  'Batch Gradient Descent Search - learning_rate ' + str(learning_rate) + stub + '.gif'
        filename_fit = 'Batch Gradient Descent Fit - learning_rate ' + str(learning_rate) + stub + '.gif'

    # Run Demo Animation
    if not cache:
        bgd = BGDDemo()
        bgd.fit(X=X, y=y, theta=theta, X_val=X_val, y_val=y_val, learning_rate=learning_rate, precision=precision,
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
            bgd_demo(learning_rate=row.learning_rate, precision=row.precision,
            maxiter=row.maxiter, miniter=row.miniter,
            stop_parameter=row.stop_parameter, 
            stop_metric=row.stop_metric,
            filename=filename, search=True, fit=True,
            fontsize=fontsize)

#%%
# report = bgd_gs()
# print(report.head())
# learning_rate = [0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1]
# best_by_learning_rate = bgd_featured(report, learning_rate)
# print(best_by_learning_rate.info())
# print(best_by_learning_rate)
# learning_rate =[0.02, 0.1, 0.8]
# featured = bgd_featured(report, learning_rate)
# print(featured)
# bgd_ani(featured)

