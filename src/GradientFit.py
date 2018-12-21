
# %%
# =========================================================================== #
#                             Gradient Search                                 #
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
from sklearn.metrics import r2_score

# --------------------------------------------------------------------------- #
#                        Gradient Search Base Class                           #
# --------------------------------------------------------------------------- #


class GradientFit:
    '''Base class for Gradient Search'''

    def __init__(self):
        self._alg = "Batch Gradient Descent"
        self._request = dict()
        self._J_history = []
        self._J_history_val = []
        self._theta_history = []
        self._g_history = []
        self._iterations = []
        self._time = []
        self._epochs = []
        self._r2 = None
        self._r2_val = None
        self._X = None
        self._y = None

    def get_alg(self):
        return(self._alg)

    def get_data(self):
        return(self._X, self._y)

    def get_epochs(self):
        return(self._epochs)

    def get_times(self):
        return(self._time)

    def get_iterations(self):
        return(self._iterations)

    def get_thetas(self):
        return(self._theta_history)

    def get_costs(self, dataset='t', init_final=None):
        if dataset == 't' :
            if init_final is None:
                return(self._J_history)
            elif init_final == 'i':
                return(self._J_history[0])
            else:
                return(self._J_history[-1])
        else:
            if init_final is None:
                return(self._J_history_val)
            elif init_final == 'i':
                return(self._J_history_val[0])
            else:
                return(self._J_history_val[-1])

    def _hypothesis(self, X, theta):
        return(X.dot(theta))

    def _error(self, h,y):
        return(h-y)

    def _cost(self, e):
        return(1/2 * np.mean(e**2))

    def _compute_r2(self, X, y, theta):
        h = self._hypothesis(X, theta)
        return(r2_score(y, h))

    def _gradient(self, X, e):
        return(X.T.dot(e)/X.shape[0])

    def _update(self, theta, gradient):
        return(theta-(self._request['hyper']['alpha'] * gradient))

    def _maxed_out(self, iteration):
        if self._request['hyper']['maxiter']:
            if iteration == self._request['hyper']['maxiter']:
                return(True)  

    def _finished(self, state):
        if self._maxed_out(state['iteration']):
            return(True)
        elif self._request['hyper']['stop_metric'] == 'a':
            return(abs(state['prior']-state['current']) < self._request['hyper']['precision'])
        else:
            if state['prior'] == 0:
                state['prior'] = 10**-10
            return(abs(state['prior']-state['current'])/abs(state['prior'])*100 < self._request['hyper']['precision'])    

    def _update_state(self,state, iteration, J,J_val, g):
        state['iteration'] = iteration
        state['prior'] = state['current']
        stop = self._request['hyper']['stop_measure']
        if stop == 't':
            state['current'] = J
        elif stop == 'v':
            state['current'] = J_val
        else:
            state['current'] = np.sqrt(np.sum(abs(g)**2))
        return(state)

    def fit(self, request):

        self._request = request

        # Initialize search variables
        iteration = 0
        theta = self._request['hyper']['theta']
        self._J_history = []
        self._J_history_val = []
        self._theta_history = []
        self._g_history = []
        self._iterations = []
        self._time = []
        self._epochs = []
        self._r2 = None
        self._r2_val = None

        # Extract data
        self._X = self._request['data']['X']
        self._y = self._request['data']['y']
        self._X_val = self._request['data']['X_val']
        self._y_val = self._request['data']['y_val']

        # Initialize State 
        state = {'prior':10**10, 'current':0, 'iteration':0}

        while not self._finished(state):
            iteration += 1

            # Compute the costs and validation set error (if required)
            h = self._hypothesis(self._X, theta)
            e = self._error(h, self._y)
            J = self._cost(e)
            J_val = None
            g = self._gradient(self._X, e)

            # Save iterations, costs and thetas in history 
            self._theta_history.append(theta.tolist())
            self._J_history.append(J)            
            self._g_history.append(g)
            self._iterations.append(iteration)
            self._epochs.append(iteration)            

            if self._request['hyper']['cross_validated']:
                h_val = self._hypothesis(self._X_val, theta)
                e_val = self._error(h_val, self._y_val)
                J_val = self._cost(e_val)                
                self._J_history_val.append(J_val)

            self._time.append(datetime.datetime.now())

            state = self._update_state(state, iteration, J, J_val, g)
                        
            theta = self._update(theta, g)

# --------------------------------------------------------------------------- #
#                       Batch Gradient Descent Search                         #
# --------------------------------------------------------------------------- #            
class BGDFit(GradientFit):
    '''Batch Gradient Descent'''

    def __init__(self):
        self._alg = "Batch Gradient Descent"    

# --------------------------------------------------------------------------- #
#                     Stochastic Gradient Descent Search                      #
# --------------------------------------------------------------------------- #            
class SGDFit(GradientFit):

    def __init__(self):
        self._alg = "Stochastic Gradient Descent"
        self._unit = "Iteration"
        self._J_history = []
        self._J_history_smooth = []
        self._theta_history = []
        self._theta_history_smooth = []
        self._h_history = []
        self._g_history = []
        self._stop = False
        self._epochs = []
        self._epochs_smooth = []
        self._iteration = 0
        self._max_iterations = 0
        self._iterations = []
        self._iterations_smooth = []
        self._X = []
        self._y = []
        self._X_i = []
        self._y_i = []
        self._X_i_smooth = []
        self._y_i_smooth = []               
        self._stop_criteria = "j"
        self._precision = 0.001
        self._stop_value = 'a'


    def _shuffle(self, X, y):
        y = np.expand_dims(y, axis=1)
        z = np.append(arr=X, values=y, axis=1)
        np.random.shuffle(z)
        X = np.delete(z, z.shape[1]-1, axis=1)
        y = z[:, z.shape[1]-1]
        return(X, y)

    def fit(self, request):

        self._request = request
        
        # Initialize search variables
        iteration = 0
        epoch = 0
        J_total = 0
        theta = self._request['hyper']['theta']
        self._J_history = []
        self._theta_history = []
        self._g_history = []
        self._iterations = []
        self._epochs = []
        self._rmse_history = []
        state = {'t':{'prior':10**10, 'current':1},
                 'g':{'prior':np.repeat(10**10, self._request['data']['X'].shape[1]), 
                      'current':np.repeat(1, self._request['data']['X'].shape[1])},
                 'v':{'prior':10**10, 'current':1}}                   

        # Extract data
        self._X = self._request['data']['X']
        self._y = self._request['data']['y']
        self._X_val = self._request['data']['X_val']
        self._y_val = self._request['data']['y_val']
        
        while not self._finished(state, iteration):
            epoch += 1            
            X, y = self._shuffle(self._X, self._y)

            for x_i, y_i in zip(X, y):
                iteration += 1

                h = self._hypothesis(x_i, theta)
                e = self._error(h, y_i)
                J = self._cost(e)
                J_total += J
                g = self._gradient(x_i, e)

                self._h_history.append(h)
                self._J_history.append(J)
                self._theta_history.append(theta.tolist())
                self._g_history.append(g)
                self._epochs.append(epoch)
                self._iterations.append(iteration)
                
                if self._iteration % self._request['hyper']['check_grad'] == 0:
                    state['t']['current'] = J_total / self._request['hyper']['check_grad']
                    state['g']['current'] = g

                    if self._request['hyper']['cross_validated']:
                        rmse = self._rmse(self._X_val, self._y_val, theta)
                        self._rmse_history.append(rmse)
                        state['v']['current'] = rmse

                    if self._finished(state, iteration):
                        break
                    J_total = 0

                theta = self._update(theta, g)
