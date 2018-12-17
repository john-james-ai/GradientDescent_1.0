# =========================================================================== #
#                       BATCH GRADIENT DESCENT TEST                           #
# =========================================================================== #
# %%
import inspect
import os
import sys
import numpy as np

src = "c:\\Users\\John\\Documents\\Data Science\\Libraries\\GradientDescent\\src"
sys.path.append(src)
from GradientDescent import BGD, SGD
import data

# Data
X, X_val, y, y_val = data.demo()

# Parameters
theta = np.array([-1,-1])
alpha = 0.01
precision = 0.0001
maxiter = 10000
stop_measure = 'j'
stop_metric = 'a'

# Instantiate and execute search
gd = SGD()
gd.search(X=X, y=y, theta=theta,X_val=X_val, y_val=y_val, alpha=alpha,
          maxiter=maxiter, precision=precision, stop_measure=stop_measure,
          stop_metric=stop_metric, check_grad=100)
request = gd.get_request()
result = gd.get_result()

print(request)
print(result)