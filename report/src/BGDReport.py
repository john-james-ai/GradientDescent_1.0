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
run = False
show = False
# Parameters
theta = np.array([-1,-1]) 
alpha = [0.01, 0.02, 0.04, 0.08, 0.1, 0.2, 0.4, 0.8, 1.0, 1.5, 1.9]
precision = [0.1, 0.01, 0.001, 0.0001]
maxiter = 10000
stop_parameter = ['t', 'v', 'g']
stop_metric = ['a', 'r']
directory = "./report/figures/BGD/"

if run:
    # Run experiment
    lab = BGDLab()
    lab.gridsearch(X=X, y=y, X_val=X_val, y_val=y_val, theta=theta, alpha=alpha, precision=precision,
            maxiter=maxiter, stop_parameter=stop_parameter, stop_metric=stop_metric)
    fig, annotations = lab.plot(directory=directory, show=show)
    report = lab.report(directory=directory)
    experiments = annotations.values
    featured = pd.merge(annotations, report, how='inner', on='experiment')
    save_csv(featured, directory, filename='BGD Featured Experiments.csv')

#%% 
# Worst Cost
# alpha = 1.9
# precision = 0.1
# maxiter = 10000
# max_costs = 100
# stop_parameter = 'g'
# stop_metric = 'a'
# bgd = BGDDemo()
# bgd.fit(X=X, y=y, theta=theta, alpha=alpha,
#         precision=precision, maxiter=maxiter, stop_parameter=stop_parameter,
#         stop_metric=stop_metric)
# bgd.show_search(directory=directory, fps=10)
#%%
# Median Cost
alpha = 0.02
precision = 0.01
maxiter = 10000
max_costs = 100
stop_parameter = 'g'
stop_metric = 'a'
bgd = BGDDemo()
bgd.fit(X=X, y=y, theta=theta, alpha=alpha,
        precision=precision, maxiter=maxiter, stop_parameter=stop_parameter,
        stop_metric=stop_metric)
#bgd.show_search(directory=directory, fps=10)
bgd.show_fit(directory=directory, fps=10)
