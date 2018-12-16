
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
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from GradientDescent import GradientDescent, BGD
# --------------------------------------------------------------------------- #
#                          GRADIENT SEARCH CLASS                              #
# --------------------------------------------------------------------------- #


class GradientSearch:
    '''Base class for Gradient Search'''

    def __init__(self):
        self._request = None
        self._result = None
        pass

    def _encode_labels(self, X, y):
        le = LabelEncoder()
        if isinstance(X, pd.core.frame.DataFrame):
            X = X.apply(le.fit_transform)
        else:
            x = le.fit_transform(X)
        if isinstance(y, pd.core.frame.DataFrame):
            y = y.apply(le.fit_transform)
        else:
            y = le.fit_transform(y)
        return(X, y)

    def _scale(self, X, y, scaler='minmax', bias=False):
        # Select scaler
        if scaler == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()

        # Put X and y into a dataframe
        if isinstance(X, pd.core.frame.DataFrame):                
            y = pd.DataFrame({'y': y})
            df = pd.concat([X, y], axis=1)
        else:
            df = pd.DataFrame({'X':X, 'y':y})

        # Scale then return X and y
        df_scaled = scaler.fit_transform(df)
        df = pd.DataFrame(df_scaled, columns=df.columns)

        # Add bias term
        if bias:
            X0 = pd.DataFrame(np.ones((df.shape[0], 1)), columns=['X0'])
            df = pd.concat([X0, df], axis=1)
        X = df.drop(columns=['y']).values
        y = df['y']
        return(X, y)

    def prep_data(self, X,y, scaler='minmax', bias=True):        
        X, y = self._encode_labels(X,y)
        X, y = self._scale(X,y, scaler, bias)
        return(X,y)         

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
               stop_measure='j', stop_metric='a', n_val=0):

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

        self._request['hyper']['alpha'] = alpha
        self._request['hyper']['theta'] = theta
        self._request['hyper']['maxiter'] = maxiter
        self._request['hyper']['precision'] = precision
        self._request['hyper']['stop_measure'] = stop_measure
        self._request['hyper']['stop_metric'] = stop_metric
        self._request['hyper']['n_val'] = n_val
        self._request['hyper']['cross_validated'] = cross_validated

        # Run search and obtain result        
        gd = BGD()
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
        self._result['summary']['Iterations'] = iterations
        self._result['summary']['Theta_Init'] = thetas.iloc[0]
        self._result['summary']['Theta_Final'] = thetas.iloc[-1]
        self._result['summary']['Cost_Init'] = costs.iloc[0]
        self._result['summary']['Cost_Final'] = costs.iloc[-1]

        if cross_validated:
            self._result['summary']['MSE_Init'] = mse.iloc[0]
            self._result['summary']['MSE_Final'] = mse.iloc[-1]


