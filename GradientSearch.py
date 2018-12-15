
# %%
# =========================================================================== #
#                             GRADIENT SEARCH                                 #
# =========================================================================== #
'''
Class that standardizes a gradient descent search across all gradient descent
variants and optimization algorithms. Thisincludes a standard interface for
defining a search, and implementing a search through the GradientDescent class.

The family of classes include:
- GradientSearch: The base GradientSearch class
- SearchBGD: Gradient Search for batch gradient descent
- SearchSGD: Gradient Search for stochastic gradient descent
- SearchMomentum: Gradient Search for momentum
- SearchNAG: Gradient Search for Nesterov Accelerated Gradient
- SearchAdagrad: Gradient Search for Adagrad
- SearchAdadelta: Gradient Search for Adadelta
- SearchRMSProp: Gradient Search for RMSProp
- SearchAdam: Gradient Search for Adam
- SearchAdamax: Gradient Search for Adamax
- SearchNadam: Gradient Search for Nadam

The high-level data structure:
- Algorithm - name of gradient descent algorithm
- Metadata - date and time started and concluded, completion status
- Hyperparameters - Parameters used during the search
- Log - Thetas gradients, cost history per iteration
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

import GradientDescent
# --------------------------------------------------------------------------- #
#                          GRADIENT SEARCH CLASS                              #
# --------------------------------------------------------------------------- #


class GradientSearch:
    '''Base class for Gradient Search'''

    def __init__(self):
        self._final = None
        pass

    def _todf(self, x, stub):
        n = len(x[0])
        df = pd.DataFrame()
        for i in range(n):
            colname = stub + str(i)
            vec = [item[i] for item in x]
            df_vec = pd.DataFrame(vec, columns=[colname])
            df = pd.concat([df, df_vec], axis=1)
        return(df)        

    def results(self):
        if self._final is None:
            raise Exception('No search results to report.')
        return(self._final)
  
    def search(self, X, y, theta, X_val=None, y_val=None, 
               alpha=0.01, maxiter=0, precision=0.001,
               stop_measure='j', stop_metric='a', n_val=0):

        # Set initial request parameters
        cross_validated = X_val is not None & y_val is not None

        # Package request
        request = dict()
        request['data'] = dict()
        request['hyper'] = dict()

        request['data']['X'] = X
        request['data']['y'] = y
        request['data']['X_val'] = X_val
        request['data']['y_val'] = y_val

        request['hyper']['alpha'] = alpha
        request['hyper']['theta'] = theta
        request['hyper']['maxiter'] = maxiter
        request['hyper']['precision'] = precision
        request['hyper']['stop_measure'] = stop_measure
        request['hyper']['stop_metric'] = stop_metric
        request['hyper']['n_val'] = n_val
        request['hyper']['cross_validated'] = cross_validated

        # Run search and obtain result        
        gd = GradientDescent.BGD()
        start = datetime.datetime.now()
        gd.search(request)
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
        result = dict()        
        result['detail'] = search_log

        result['summary'] = dict()
        result['summary']['Algorithm'] = gd.get_alg()
        result['summary']['Start'] = start
        result['summary']['End'] = end
        result['summary']['Duration'] = end-start
        result['summary']['Epochs'] = gd.get_epochs()
        result['summary']['Iterations'] = gd.get_iterations()
        result['summary']['Theta_Init'] = thetas.iloc[0]
        result['summary']['Theta_Final'] = thetas.iloc[-1]
        result['summary']['Cost_Init'] = costs.iloc[0]
        result['summary']['Cost_Final'] = costs.iloc[-1]

        if cross_validated:
            result['summary']['MSE_Init'] = mse.iloc[0]
            result['summary']['MSE_Final'] = mse.iloc[-1]

        # Package request and results, then return       
        self._final = dict() 
        self._final['request'] = request
        self._final['result'] = result
        return(self._final)

