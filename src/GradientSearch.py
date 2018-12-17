
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
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
#                        Gradient Search Base Class                           #
# --------------------------------------------------------------------------- #


class GradientSearch:
    '''Base class for Gradient Search'''

    def __init__(self):
        self._alg = "Batch Gradient Descent"
        self._request = dict()
        self._J_history = []
        self._theta_history = []
        self._g_history = []
        self._mse_history = []
        self._iterations = []
        self._epochs = []
        self._X = None
        self._y = None

    def get_alg(self):
        return(self._alg)

    def get_data(self):
        return(self._X, self._y)

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

    def _encode_labels(self, X, y):
        le = LabelEncoder()
        X = X.apply(le.fit_transform)        
        y = y.apply(le.fit_transform)
        return(X, y)

    def _scale(self, X, y, scaler='minmax', bias=True):
        # Select scaler
        if scaler == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()        

        # Put X and y into a dataframe
        df = pd.concat([X, y], axis=1)

        # Scale then recover dataframe with column names
        df_scaled = scaler.fit_transform(df)
        df = pd.DataFrame(df_scaled, columns=df.columns)

        # Add bias term
        if bias:
            X0 = pd.DataFrame(np.ones((df.shape[0], 1)), columns=['X0'])
            df = pd.concat([X0, df], axis=1)
        X = df.drop(columns=y.columns)
        y = df[y.columns].squeeze()
        return(X, y)

    def _prep_data(self, X,y, scaler='minmax', bias=True):        
        X, y = self._encode_labels(X,y)
        X, y = self._scale(X,y, scaler, bias)
        return(X,y)         

    def _hypothesis(self, X, theta):
        return(X.dot(theta))

    def _error(self, h,y):
        return(h-y)

    def _cost(self, e):
        return(1/2 * np.mean(e**2))

    def _mse(self, X, y, theta):
        h = self._hypothesis(X, theta)
        e = self._error(h,y)
        return(np.mean(e**2))

    def _gradient(self, X, e):
        return(X.T.dot(e)/X.shape[0])

    def _update(self, theta, gradient):
        return(theta-(self._request['hyper']['alpha'] * gradient))

    def _zeros(self, d):
        for k, v in d.items():
            for i, j in v.items():
                if isinstance(j, (float, int)):
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

    def _finished_grad(self, state):

        if self._request['hyper']['stop_metric'] == 'a':
            result = (all(abs(y-x)<self._request['hyper']['precision'] for x,y in zip(state['g']['prior'], state['g']['current'])))
        else:    
            result = (all(abs((y-x)/x)<self._request['hyper']['precision'] for x,y in zip(state['g']['prior'], state['g']['current'])))
        state['g']['prior'] = state['g']['current']
        return(result)

    
    def _finished_v(self, state):

        if self._request['hyper']['stop_metric'] == 'a':
            result = (abs(state['v']['current']-state['v']['prior']) < self._request['hyper']['precision'])                        
        else:
            result = (abs((state['v']['current']-state['v']['prior'])/state['v']['prior']) < self._request['hyper']['precision'])
        state['v']['prior'] = state['v']['current']
        return(result)

    def _finished_J(self, state):

        if self._request['hyper']['stop_metric'] == 'a':
            result = (abs(state['j']['current']-state['j']['prior']) < self._request['hyper']['precision'])                        
        else:
            result = (abs((state['j']['current']-state['j']['prior'])/state['j']['prior']) < self._request['hyper']['precision'])
        state['j']['prior'] = state['j']['current']
        return(result)


    def _maxed_out(self, iteration):
        if self._request['hyper']['maxiter']:
            if iteration == self._request['hyper']['maxiter']:
                return(True)  

    def _finished(self, state, iteration):
        state = self._zeros(state)
        if self._maxed_out(iteration):
            return(True)
        elif self._request['hyper']['stop_measure'] == 'j':
            return(self._finished_J(state))
        elif self._request['hyper']['stop_measure'] == 'g':
            return(self._finished_grad(state))    
        else:
            return(self._finished_v(state))  

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

        # Prepare data
        self._X, self._y = self._prep_data(X=self._request['data']['X'],
                                           y=self._request['data']['y'], 
                                           scaler=self._request['data']['scaler'])
        if self._request['hyper']['cross_validated']:
            self._X_val, self._y_val = self._prep_data(X=self._request['data']['X_val'],
                                                       y=self._request['data']['y_val'], 
                                                       scaler=self._request['data']['scaler'])       

        # Initialize State Variables
        state = {'j':{'prior':10**10, 'current':1},
                 'g':{'prior':np.repeat(10**10, self._request['data']['X'].shape[1]), 
                      'current':np.repeat(1, self._request['data']['X'].shape[1])},
                 'v':{'prior':10**10, 'current':1}}                                                             

        while not self._finished(state, iteration):
            iteration += 1

            # Compute the costs and validation set error (if required)
            h = self._hypothesis(self._X, theta)
            e = self._error(h, self._y)
            J = self._cost(e)
            g = self._gradient(self._X, e)

            if self._request['hyper']['cross_validated']:
                mse = self._mse(self._X, self._y, theta)
                self._mse_history.append(mse)
                state['v']['current'] = mse
            
            # Save current computations in state
            state['j']['current'] = J
            state['g']['current'] = g
            
            # Save iterations, costs and thetas in history 
            self._theta_history.append(theta.tolist())
            self._J_history.append(J)            
            self._g_history.append(g)
            self._iterations.append(iteration)
            self._epochs.append(iteration)

            theta = self._update(theta, g)

# --------------------------------------------------------------------------- #
#                       Batch Gradient Descent Search                         #
# --------------------------------------------------------------------------- #            
class BGDSearch(GradientSearch):
    '''Batch Gradient Descent'''

    def __init__(self):
        self._alg = "Batch Gradient Descent"    

# --------------------------------------------------------------------------- #
#                     Stochastic Gradient Descent Search                      #
# --------------------------------------------------------------------------- #            
class SGDSearch(GradientSearch):

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

    def search(self, request):

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
        self._mse_history = []
        state = {'j':{'prior':10**10, 'current':1},
                 'g':{'prior':np.repeat(10**10, self._request['data']['X'].shape[1]), 
                      'current':np.repeat(1, self._request['data']['X'].shape[1])},
                 'v':{'prior':10**10, 'current':1}}                   

        # Prepare data
        self._X, self._y = self._prep_data(X=self._request['data']['X'],
                                           y=self._request['data']['y'], 
                                           scaler=self._request['data']['scaler'])
        if self._request['hyper']['cross_validated']:
            self._X_val, self._y_val = self._prep_data(X=self._request['data']['X_val'],
                                                       y=self._request['data']['y_val'], 
                                                       scaler=self._request['data']['scaler'])  

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
                    state['j']['current'] = J_total / self._request['hyper']['check_grad']
                    state['g']['current'] = g

                    if self._request['hyper']['cross_validated']:
                        mse = self._mse(self._X_val, self._y_val, theta)
                        self._mse_history.append(mse)
                        state['v']['current'] = mse

                    if self._finished(state, iteration):
                        break
                    J_total = 0

                theta = self._update(theta, g)
