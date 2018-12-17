
# %%
# =========================================================================== #
#                             GRADIENT DESCENT                                #
# =========================================================================== #
'''
Class creates gradient descent solutions for each of the following gradient
descent variants and optimization algorithms.
- Batch gradient descent
- Stochastic gradient descent
- Momentum
- Nesterov Accelerated Gradient
- Adagrad
- Adadelta
- RMSProp
- Adam
- Adamax
- Nadam
'''

# --------------------------------------------------------------------------- #
#                                LIBRARIES                                    #
# --------------------------------------------------------------------------- #

import inspect
import os
import sys

import datetime
import numpy as np
from numpy import array, newaxis
import pandas as pd
import seaborn as sns

from GradientSearch import BGDSearch, SGDSearch
# --------------------------------------------------------------------------- #
#                       GRADIENT DESCENT BASE CLASS                           #
# --------------------------------------------------------------------------- #


class GradientDescent:
    '''Base class for Gradient Descent'''

    def __init__(self):
        self._request = None
        self._result = None

    def _todf(self, x, stub):
        n = len(x[0])
        df = pd.DataFrame()
        for i in range(n):
            colname = stub + str(i)
            vec = [item[i] for item in x]
            df_vec = pd.DataFrame(vec, columns=[colname])
            df = pd.concat([df, df_vec], axis=1)
        return(df)        

    def get_request(self):
        if self._request is None:
            raise Exception('No search request to report.')        
        else:
            return(self._request)

    def get_result(self):
        if self._result is None:
            raise Exception('No search results to report.')
        else:
            return(self._result)
  
    def search(self, X, y, theta, X_val=None, y_val=None, 
               alpha=0.01, maxiter=0, precision=0.001,
               stop_measure='j', stop_metric='a', scaler='minmax'):

        # Set initial request parameters
        cross_validated = all(v is not None for v in [X_val, y_val])

        # Package request
        self._request = dict()
        self._request['data'] = dict()
        self._request['hyper'] = dict()

        self._request['data']['X'] = X
        self._request['data']['y'] = y
        self._request['data']['X_val'] = X_val
        self._request['data']['y_val'] = y_val        
        self._request['data']['scaler'] = scaler        
        self._request['hyper']['alpha'] = alpha
        self._request['hyper']['theta'] = theta
        self._request['hyper']['maxiter'] = maxiter
        self._request['hyper']['precision'] = precision
        self._request['hyper']['stop_measure'] = stop_measure
        self._request['hyper']['stop_metric'] = stop_metric
        self._request['hyper']['cross_validated'] = cross_validated

        # Run search and obtain result        
        gd = BGDSearch()
        start = datetime.datetime.now()
        gd.search(self._request)
        end = datetime.datetime.now()

        # Extract search log
        epochs = pd.DataFrame(gd.get_epochs(), columns=['Epochs'])
        iterations = pd.DataFrame(gd.get_iterations(), columns=['Epochs'])
        thetas = self._todf(gd.get_thetas(), stub='theta_')
        costs = pd.DataFrame(gd.get_costs(), columns=['Cost'])
        if cross_validated:
            mse = pd.DataFrame(gd.get_mse(), columns=['MSE'])
        search_log = pd.concat([iterations, thetas, costs], axis=1)

        # Package results
        self._result = dict()        
        self._result['detail'] = search_log

        self._result['summary'] = dict()
        self._result['summary']['Algorithm'] = gd.get_alg()
        self._result['summary']['Start'] = start
        self._result['summary']['End'] = end
        self._result['summary']['Duration'] = end-start
        self._result['summary']['Epochs'] = epochs
        self._result['summary']['X_transformed'] = gd.get_data()[0]
        self._result['summary']['y_transformed'] = gd.get_data()[1]
        self._result['summary']['Iterations'] = iterations
        self._result['summary']['Theta_Init'] = thetas.iloc[0]
        self._result['summary']['Theta_Final'] = thetas.iloc[-1]
        self._result['summary']['Cost_Init'] = costs.iloc[0]
        self._result['summary']['Cost_Final'] = costs.iloc[-1]

        if cross_validated:
            self._result['summary']['MSE_Init'] = mse.iloc[0]
            self._result['summary']['MSE_Final'] = mse.iloc[-1]


# --------------------------------------------------------------------------- #
#                       BATCH GRADIENT DESCENT CLASS                          #
# --------------------------------------------------------------------------- #

class BGD(GradientDescent):
    '''Batch Gradient Descent'''

    def __init__(self):
        pass
# --------------------------------------------------------------------------- #
#                     STOCHASTIC GRADIENT DESCENT CLASS                       #
# --------------------------------------------------------------------------- #

class SGD(GradientDescent):
    '''Stochastic Gradient Descent'''

    def __init__(self):
        pass
  
    def search(self, X, y, theta, X_val=None, y_val=None, 
               alpha=0.01, maxiter=0, precision=0.001,
               stop_measure='j', stop_metric='a', check_grad=100,
               scaler='minmax'):

        # Set initial request parameters
        cross_validated = all(v is not None for v in [X_val, y_val])

        # Package request
        self._request = dict()
        self._request['data'] = dict()
        self._request['hyper'] = dict()

        self._request['data']['X'] = X
        self._request['data']['y'] = y
        self._request['data']['X_val'] = X_val
        self._request['data']['y_val'] = y_val        
        self._request['data']['scaler'] = scaler        
        self._request['hyper']['alpha'] = alpha
        self._request['hyper']['theta'] = theta
        self._request['hyper']['maxiter'] = maxiter
        self._request['hyper']['precision'] = precision
        self._request['hyper']['stop_measure'] = stop_measure
        self._request['hyper']['stop_metric'] = stop_metric
        self._request['hyper']['check_grad'] = check_grad
        self._request['hyper']['cross_validated'] = cross_validated

        # Run search and obtain result        
        gd = SGDSearch()
        start = datetime.datetime.now()
        gd.search(self._request)
        end = datetime.datetime.now()

        # Extract search log
        epochs = pd.DataFrame(gd.get_epochs(), columns=['Epochs'])
        iterations = pd.DataFrame(gd.get_iterations(), columns=['Epochs'])
        thetas = self._todf(gd.get_thetas(), stub='theta_')
        costs = pd.DataFrame(gd.get_costs(), columns=['Cost'])
        if cross_validated:
            mse = pd.DataFrame(gd.get_mse(), columns=['MSE'])
        search_log = pd.concat([iterations, thetas, costs], axis=1)

        # Package results
        self._result = dict()        
        self._result['detail'] = search_log

        self._result['summary'] = dict()
        self._result['summary']['Algorithm'] = gd.get_alg()
        self._result['summary']['Start'] = start
        self._result['summary']['End'] = end
        self._result['summary']['Duration'] = end-start
        self._result['summary']['Epochs'] = epochs
        self._result['summary']['X_transformed'] = gd.get_data()[0]
        self._result['summary']['y_transformed'] = gd.get_data()[1]
        self._result['summary']['Iterations'] = iterations
        self._result['summary']['Theta_Init'] = thetas.iloc[0]
        self._result['summary']['Theta_Final'] = thetas.iloc[-1]
        self._result['summary']['Cost_Init'] = costs.iloc[0]
        self._result['summary']['Cost_Final'] = costs.iloc[-1]

        if cross_validated:
            self._result['summary']['MSE_Init'] = mse.iloc[0]
            self._result['summary']['MSE_Final'] = mse.iloc[-1]
