
# %%
# =========================================================================== #
#                             Gradient Descent                                #
# =========================================================================== #
import inspect
import os
import sys

import datetime
import itertools
import math
import numpy as np
from numpy import array, newaxis
import pandas as pd

# --------------------------------------------------------------------------- #
#                       Gradient Descent Base Class                           #
# --------------------------------------------------------------------------- #


class GradientDescent:
    '''Base class for Gradient Descent'''

    def __init__(self):
        self._alg = "Batch Gradient Descent"
        self._request = dict()
        self._J_history = []
        self._theta_history = []
        self._g_history = []
        self._mse_history = []
        self._iterations = []
        self._epochs = []
        self._state = dict()

    def get_alg(self):
        return(self._alg)

    def get_epochs(self):
        return(self._epochs)

    def get_iterations(self):
        return(self._iterations)

    def get_thetas(self):
        return(self._theta_history)

    def get_costs(self):
        return(self._J_history)

    def get_mse(self):
        if self._request['hyper']['cross_validated']:
            return(self._mse_history)
        else:
            raise Exception("Search was not cross-validated. No MSE data available.")

    def _hypothesis(self, theta):
        return(self._request['data']['X'].dot(theta))

    def _error(self, h):
        return(h-self._request['data']['y'])

    def _cost(self, e):
        return(1/2 * np.mean(e**2))

    def _mse(self, theta):
        h = self._hypothesis(theta)
        e = self._error(h)
        return(np.mean(e**2))

    def _gradient(self, e):
        return(self._request['data']['X'].T.dot(e)/self._request['data']['X'].shape[0])

    def _update(self, theta, gradient):
        return(theta-(self._request['hyper']['alpha'] * gradient))

    def _zeros(self, d):
        for k, v in d.items():
            for i, j in v.items():
                if type(j) is float or type(j) is int:
                    if (j<10**-10) & (j>0):
                        j = 10**-10
                    elif (j>-10**-10) & (j<0):
                        j = -10**-10
                else: 
                    for l in j:
                        if (l<10**-10) & (l>0):
                            l = 10**-10
                        elif (l>-10**-10) & (l<0):
                            l = -10**-10
        return(d)

    def _finished_grad(self):

        if self._request['hyper']['stop_metric'] == 'a':
            result = (all(abs(y-x)<self._request['hyper']['precision'] for x,y in zip(self._state['g']['prior'], self._state['g']['current'])))
        else:    
            result = (all(abs((y-x)/x)<self._request['hyper']['precision'] for x,y in zip(self._state['g']['prior'], self._state['g']['current'])))
        self._state['g']['prior'] = self._state['g']['current']
        return(result)

    
    def _finished_v(self):

        if self._request['hyper']['stop_metric'] == 'a':
            result = (abs(self._state['v']['current']-self._state['v']['prior']) < self._request['hyper']['precision'])                        
        else:
            result = (abs((self._state['v']['current']-self._state['v']['prior'])/self._state['v']['prior']) < self._request['hyper']['precision'])
        self._state['v']['prior'] = self._state['v']['current']
        return(result)

    def _finished_J(self):

        if self._request['hyper']['stop_metric'] == 'a':
            result = (abs(self._state['j']['current']-self._state['j']['prior']) < self._request['hyper']['precision'])                        
        else:
            result = (abs((self._state['j']['current']-self._state['j']['prior'])/self._state['j']['prior']) < self._request['hyper']['precision'])
        self._state['j']['prior'] = self._state['j']['current']
        return(result)


    def _maxed_out(self, iteration):
        if self._request['hyper']['maxiter']:
            if iteration == self._request['hyper']['maxiter']:
                return(True)  

    def _finished(self, iteration):
        self._state = self._zeros(self._state)
        if self._maxed_out(iteration):
            return(True)
        elif self._request['hyper']['stop_measure'] == 'j':
            return(self._finished_J())
        elif self._request['hyper']['stop_measure'] == 'g':
            return(self._finished_grad())    
        else:
            return(self._finished_v())

  

    def search(self, request):

        self._request = request

        # Initialize search variables
        iteration = 0
        theta = self._request['hyper']['theta']
        self._J_history = []
        self._theta_history = []
        self._g_history = []
        self._iterations = []
        self._epochs = []
        self._mse_history = []
        self._state = {'j':{'prior':10**10, 'current':1},
                       'g':{'prior':np.repeat(10**10, self._request['data']['X'].shape[1]), 
                            'current':np.repeat(1, self._request['data']['X'].shape[1])},
                       'v':{'prior':10**10, 'current':1}}      

        while not self._finished(iteration):
            iteration += 1

            # Compute the costs and validation set error (if required)
            h = self._hypothesis(theta)
            e = self._error(h)
            J = self._cost(e)
            g = self._gradient(e)

            if self._request['hyper']['cross_validated']:
                mse = self._mse(theta)
                self._mse_history.append(mse)
                self._state['v']['current'] = mse
            
            # Save current computations in state
            self._state['j']['current'] = J
            self._state['g']['current'] = g.tolist()
            
            # Save iterations, costs and thetas in history 
            self._theta_history.append(theta.tolist())
            self._J_history.append(J)            
            self._g_history.append(g.tolist())
            self._iterations.append(iteration)
            self._epochs.append(iteration)

            theta = self._update(theta, g)

# --------------------------------------------------------------------------- #
#                       Batch Gradient Descent Class                          #
# --------------------------------------------------------------------------- #            
class BGD(GradientDescent):
    '''Batch Gradient Descent'''

    def __init__(self):
        self._alg = "Batch Gradient Descent"    

# %%
# Get Data
# from data import data
# ames = data.read()
# ames = ames[['Area', 'SalePrice']]
# df = ames.sample(n=100, random_state=50, axis=0)
# test = ames.loc[~ames.index.isin(df.index),:]
# test = test.dropna()
# df = df.reset_index(drop=True)
# test = test.reset_index(drop=True)
# X = df[['Area']]
# y = df['SalePrice']
# X_val = test[['Area']]
# y_val = test[['SalePrice']]

# # Prep Data
# import GradientSearch
# gd = GradientSearch()
# X, y = gd.prep_data(X, y)
# X_val, y_val = gd.prep_data(X_val, y_val)

# # Create request object
# request = dict()
# request['data'] = dict()
# request['data']['X'] = X
# request['data']['y'] = y
# request['data']['X_val'] = X_val
# request['data']['y_val'] = y_val

# request['hyper']['alpha'] = 0.01
# request['hyper']['theta'] = np.array([-1,-1])
# request['hyper']['maxiter'] = 10000
# request['hyper']['precision'] = 0.0001
# request['hyper']['stop_measure'] = 'j'
# request['hyper']['stop_metric'] = 'a'
# request['hyper']['n_val'] = 0
# request['hyper']['cross_validated'] = True

# # Instantiate and execute search
# gd = BGD()
# gd.search(request)
# request_2 = gd.get_request()
# result_2 = gd.get_result()

# request_2.shape
# result_2.shape