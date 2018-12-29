
# %%
# =========================================================================== #
#                             Gradient Search                                 #
# =========================================================================== #
from abc import ABC, abstractmethod
import datetime
import math
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
#                        Gradient Search Base Class                           #
# --------------------------------------------------------------------------- #


class GradientFit(ABC):
    '''Abstract base class for Gradient Search'''

    def __init__(self):
        self._alg = "Batch Gradient Descent"
        self._request = dict()   
        self._J_history = []     
        self._J_history_val = [] 
        self._theta_history = [] 
        self._g_history = []     
        self._iterations = []    
        self._epochs = []        
        self._X = None           
        self._y = None
        self._X_val = None
        self._y_val = None

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

    def _error(self, h, y):
        return(h-y)

    def _cost(self, e):
        return(1/2 * np.mean(e**2))

    def _gradient(self, X, e):
        return(X.T.dot(e)/X.shape[0])

    def _update(self, theta, gradient):        
        return(theta-(self._request['hyper']['alpha'] * gradient))

    def _maxed_out(self, iteration):
        if self._request['hyper']['maxiter']:
            if iteration == self._request['hyper']['maxiter']:
                return(True)  

    def _zeros(self, x):
        if (x<1**-10) & (x>0):
            x = 1**-10
        elif (x>-1**-10) & (x<0):
            x = -1**-10
        return(x)

    def _finished(self, state):
        state['current'] = self._zeros(state['current'])
        state['prior'] = self._zeros(state['prior'])

        if self._request['hyper']['miniter']:
            if self._request['hyper']['miniter'] <= state['iteration']:
                if self._maxed_out(state['iteration']):
                    return(True)
                elif self._request['hyper']['stop_metric'] == 'a':
                    return(abs(state['prior']-state['current']) < self._request['hyper']['precision'])
                else:
                    return(abs(state['prior']-state['current'])/abs(state['prior']) < self._request['hyper']['precision'])     
        else:                    
            if self._maxed_out(state['iteration']):
                return(True)
            elif self._request['hyper']['stop_metric'] == 'a':
                return(abs(state['prior']-state['current']) < self._request['hyper']['precision'])
            else:
                return(abs(state['prior']-state['current'])/abs(state['prior']) < self._request['hyper']['precision'])    

    def _update_state(self,state, iteration, J, J_val, g):
        state['iteration'] = iteration
        state['prior'] = state['current']
        stop = self._request['hyper']['stop_parameter']
        if stop == 't':
            state['current'] = J
        elif stop == 'v':
            state['current'] = J_val
        else:
            state['current'] = np.sqrt(np.sum(abs(g)**2))
        return(state)

    @abstractmethod
    def fit(self, request):
        pass

# --------------------------------------------------------------------------- #
#                       Batch Gradient Descent Search                         #
# --------------------------------------------------------------------------- #            
class BGDFit(GradientFit):
    '''Batch Gradient Descent'''

    def __init__(self)->None:
        self._alg = "Batch Gradient Descent"    

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
        self._epochs = []

        # Extract data
        self._X = self._request['data']['X']
        self._y = self._request['data']['y']
        self._X_val = self._request['data']['X_val']
        self._y_val = self._request['data']['y_val']

        # Initialize State 
        state = {'prior':1, 'current':10, 'iteration':0}

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

            state = self._update_state(state, iteration, J, J_val, g)
                        
            theta = self._update(theta, g)



# --------------------------------------------------------------------------- #
#                     Stochastic Gradient Descent Search                      #
# --------------------------------------------------------------------------- #            
class SGDFit(GradientFit):

    def __init__(self)->None:
        self._alg = "Stochastic Gradient Descent"
        self._request = dict()
        self._J_history = []
        self._J_history_val = []
        self._theta_history = []
        self._g_history = []
        self._iterations = []
        self._epochs = []
        self._X = None
        self._y = None
        self._X_val = None
        self._y_val = None


    def _shuffle(self, X, y):
        y_var = y.name
        df = pd.concat([X,y], axis=1)
        df = df.sample(frac=1, replace=False, axis=0)
        X = df.drop(labels = y_var, axis=1)
        y = df[y_var]
        return(X, y)

    def fit(self, request):

        self._request = request

        # Initialize search variables
        iteration = 0
        epoch = 0
        J_total = 0
        theta = self._request['hyper']['theta']
        self._J_history = []
        self._J_history_val = []
        self._theta_history = []
        self._g_history = []
        self._iterations = []
        self._epochs = []        

        # Extract data
        self._X = self._request['data']['X']
        self._y = self._request['data']['y']
        self._X_val = self._request['data']['X_val']
        self._y_val = self._request['data']['y_val']

        # Initialize State 
        state = {'prior':10**10, 'current':0, 'iteration':0}
        
        # Compute number of iterations in each batch
        if self._request['hyper']['check_point'] > 1:
            iterations_per_batch = self._request['hyper']['check_point']
        else:
            iterations_per_batch = math.floor(self._X.shape[0] * self._request['hyper']['check_point'])
        
        while not self._finished(state):
            epoch += 1            
            X, y = self._shuffle(self._X, self._y)

            for x_i, y_i in zip(X.values, y):
                iteration += 1

                h = self._hypothesis(x_i, theta)
                e = self._error(h, y_i)
                J = self._cost(e)
                J_total = J_total + J
                J_val = None
                g = self._gradient(x_i, e)

                if iteration % iterations_per_batch == 0:
                    J_avg = J_total / iterations_per_batch
                    J_total = 0                    
                    self._J_history.append(J_avg)
                    self._theta_history.append(theta.tolist())
                    self._g_history.append(g)
                    self._epochs.append(epoch)
                    self._iterations.append(iteration)

                    if self._request['hyper']['cross_validated']:
                        h_val = self._hypothesis(self._X_val, theta)
                        e_val = self._error(h_val, self._y_val)
                        J_val = self._cost(e_val)                
                        self._J_history_val.append(J_val)

                    state = self._update_state(state, iteration, J, J_val, g)

                    if self._finished(state):
                        break

                theta = self._update(theta, g)


# --------------------------------------------------------------------------- #
#                     Mini-Batch Gradient Descent Search                      #
# --------------------------------------------------------------------------- #            
class MBGDFit(GradientFit):

    def __init__(self):
        self._alg = "Mini-Batch Gradient Descent"
        self._request = dict()
        self._J_history = []
        self._J_history_val = []
        self._theta_history = []
        self._g_history = []
        self._iterations = []
        self._epochs = []
        self._X = None
        self._y = None
        self._X_val = None
        self._y_val = None


    def _shuffle(self, X, y):
        y_var = y.name
        df = pd.concat([X,y], axis=1)
        df = df.sample(frac=1, replace=False, axis=0)
        X = df.drop(labels = y_var, axis=1)
        y = df[y_var]
        return(X, y)

    def _get_batches(self, X, y, batch_size):        
        df = pd.concat([X,y], axis=1)
        if batch_size >1:
            pass
        else:
            batch_size = df.shape[0]*batch_size
        df_group = df.groupby(pd.qcut(np.arange(df.shape[0]), q=int(batch_size)))
        X_list = []
        y_list = []
        for name, group in df_group:
            X_list.append(group.drop(labels=y.name, axis=1))
            y_list.append(group[y.name])

        return(X_list,y_list)


    def fit(self, request):

        self._request = request

        # Initialize search variables
        iteration = 0
        epoch = 0        
        theta = self._request['hyper']['theta']
        self._J_history = []
        self._J_history_val = []
        self._theta_history = []
        self._g_history = []
        self._iterations = []
        self._epochs = []        

        # Extract data
        self._X = self._request['data']['X']
        self._y = self._request['data']['y']
        self._X_val = self._request['data']['X_val']
        self._y_val = self._request['data']['y_val']

        # Initialize State 
        state = {'prior':10**10, 'current':0, 'iteration':0}
       
        while not self._finished(state):
            epoch += 1            
            J_total = 0
            X, y = self._shuffle(self._X, self._y)
            X_mb, y_mb = self._get_batches(X,y, self._request['hyper']['batch_size'])

            for x_i, y_i in zip(X_mb, y_mb):
                iteration += 1

                h = self._hypothesis(x_i.values, theta)
                e = self._error(h, y_i.values)
                J = self._cost(e)
                g = self._gradient(x_i.values, e)

                self._J_history.append(J)
                self._theta_history.append(theta.tolist())
                self._g_history.append(g)
                self._epochs.append(epoch)
                self._iterations.append(iteration)

                if self._request['hyper']['cross_validated']:
                    h_val = self._hypothesis(self._X_val, theta)
                    e_val = self._error(h_val, self._y_val)
                    J_val = self._cost(e_val)                
                    self._J_history_val.append(J_val)

                state = self._update_state(state, iteration, J, J_val, g)

                if self._finished(state):
                    break

                theta = self._update(theta, g)
