
# %%
# =========================================================================== #
#                             Gradient Search                                 #
# =========================================================================== #
from abc import ABC, abstractmethod
import datetime
import math
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
# --------------------------------------------------------------------------- #
#                        Gradient Search Base Class                           #
# --------------------------------------------------------------------------- #


class GradientFit(ABC):
    '''Abstract base class for Gradient Search'''

    def __init__(self):
        self._alg = "Batch Gradient Descent"
        self._request = dict()   
        self._learning_rate_history = []
        self._J_history = []     
        self._mse_history = [] 
        self._theta_history = [] 
        self._iterations = []    
        self._epochs = []        
        self._X = None           
        self._y = None
        self._X_val = None
        self._y_val = None
        self._i_s = 5
        self._iter_no_change = 0

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

    def get_learning_rates(self):
        return(self._learning_rate_history)

    def get_costs(self):
        return(self._J_history)

    def get_mse(self):
        return(self._mse_history)

    def _hypothesis(self, X, theta):
        return(X.dot(theta))

    def _error(self, h, y):
        return(h-y)

    def _cost(self, e):
        return(1/2 * np.mean(e**2))
    
    def _total_cost(self, X, y, theta):
        h = self._hypothesis(X, theta)
        e = self._error(h, y)
        return(self._cost(e))

    def _mse(self, X, y, theta):
        h = self._hypothesis(X, theta)
        e = self._error(h,y)
        return(np.mean(e**2))

    def _gradient(self, X, e):
        return(X.T.dot(e)/X.shape[0])

    def _update(self, theta, learning_rate, gradient):        
        return(theta-(learning_rate * gradient))

    def _update_learning_rate(self, learning_rate, epoch):
        learning_rate_init = self._request['hyper']['learning_rate']
        if self._request['hyper']['learning_rate_sched'] == 'c':
            learning_rate_new = learning_rate
        elif self._request['hyper']['learning_rate_sched'] == 't':
            k = self._request['hyper']['time_decay']
            learning_rate_new = learning_rate_init/(1+k*epoch)            
        elif self._request['hyper']['learning_rate_sched'] == 's':
            drop = self._request['hyper']['step_decay']
            epochs_drop = self._request['hyper']['step_epochs']
            learning_rate_new = learning_rate_init*math.pow(drop, math.floor((1+epoch)/epochs_drop))
        elif self._request['hyper']['learning_rate_sched'] == 'e':            
            k = self._request['hyper']['exp_decay']
            learning_rate_new = learning_rate_init * math.exp(-k*epoch)
        return(learning_rate_new)

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

        if self._maxed_out(state['iteration']):
            return(True)  
        if state['current'] > state['prior']:
            return(True)
        elif abs(state['prior']-state['current']) < self._request['hyper']['precision']:
            self._iter_no_change += 1
            if self._iter_no_change >= self._i_s:
                return(True)
            else:
                return(False)
        else:
            self._iter_no_change = 0
            return(False)

    def _update_state(self, state, iteration, J, mse):
        state['iteration'] = iteration
        state['prior'] = state['current']
        if self._request['hyper']['cross_validated']:
            state['current'] = mse
        else:      
            state['current'] = J
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
        self._request = dict()   
        self._learning_rate_history = []
        self._J_history = []     
        self._mse_history = [] 
        self._theta_history = [] 
        self._iterations = []    
        self._epochs = []        
        self._X = None           
        self._y = None
        self._X_val = None
        self._y_val = None
        self._i_s = 5
        self._iter_no_change = 0 

    def fit(self, request):

        self._request = request

        # Initialize lists
        self._learning_rate_history = []
        self._J_history = []
        self._mse_history = []
        self._theta_history = []
        self._iterations = []
        self._epochs = []

        # Initialize search variables
        iteration = 0
        theta = self._request['hyper']['theta']
        learning_rate = self._request['hyper']['learning_rate']
        self._i_s = self._request['hyper']['i_s']
        self._iter_no_change = 0
        
        # Extract data
        self._X = self._request['data']['X']
        self._y = self._request['data']['y']
        self._X_val = self._request['data']['X_val']
        self._y_val = self._request['data']['y_val']

        # Initialize State 
        state = {'prior':100, 'current':10, 'iteration':0}
        
        while not self._finished(state):
            iteration += 1
            mse = None

            # Compute the costs and validation set error (if required)
            h = self._hypothesis(self._X, theta)
            e = self._error(h, self._y)
            J = self._cost(e)
            if self._request['hyper']['cross_validated']:
                mse = self._mse(self._X_val, self._y_val, theta)            

            # Save iterations, costs and thetas in history 
            self._theta_history.append(theta.tolist())
            self._J_history.append(J)            
            self._iterations.append(iteration)
            self._epochs.append(iteration)
            self._mse_history.append(mse)
            self._learning_rate_history.append(learning_rate)


            # Compute gradient
            g = self._gradient(self._X, e)

            # Update thetas 
            theta = self._update(theta, learning_rate, g)

            # Update learning rate
            learning_rate = self._update_learning_rate(learning_rate, iteration)

            # Update state vis-a-vis training set costs and validation set mse 
            state = self._update_state(state, iteration, J, mse)



# --------------------------------------------------------------------------- #
#                     Stochastic Gradient Descent Search                      #
# --------------------------------------------------------------------------- #            
class SGDFit(GradientFit):

    def __init__(self)->None:
        self._alg = "Stochastic Gradient Descent"
        self._request = dict()
        self._J_history = []
        self._mse_history = []
        self._theta_history = []
        self._iterations = []
        self._epochs = []
        self._X = None
        self._y = None
        self._X_val = None
        self._y_val = None
        self._iter_no_change = 0
        self._i_s = 5        


    def _shuffle(self, X, y):
        y_var = y.name
        df = pd.concat([X,y], axis=1)
        df = df.sample(frac=1, replace=False, axis=0)
        X = df.drop(labels = y_var, axis=1)
        y = df[y_var]
        return(X, y)

    def fit(self, request):

        self._request = request

        # Initialize lists
        self._learning_rate_history = []
        self._J_history = []
        self._mse_history = []
        self._theta_history = []
        self._iterations = []
        self._epochs = []

        # Initialize search variables
        iteration = 0
        epoch = 0
        theta = self._request['hyper']['theta']
        learning_rate = self._request['hyper']['learning_rate']
        self._i_s = self._request['hyper']['i_s']
        self._iter_no_change = 0
        
        # Extract data
        self._X = self._request['data']['X']
        self._y = self._request['data']['y']
        self._X_val = self._request['data']['X_val']
        self._y_val = self._request['data']['y_val']

        # Initialize State 
        state = {'prior':1, 'current':5, 'iteration':0}
        
        while not self._finished(state):
            epoch += 1            
            mse = None

            X, y = self._shuffle(self._X, self._y)            

            for x_i, y_i in zip(X.values, y):
                iteration += 1

                h = self._hypothesis(x_i, theta)
                e = self._error(h, y_i)
                J = self._cost(e)                         
                g = self._gradient(x_i, e)
                theta = self._update(theta, learning_rate, g)
            
            # Compute total training set cost with latest theta
            J = self._total_cost(self._X, self._y, theta)

            # if cross_validated, compute validation set MSE 
            if self._request['hyper']['cross_validated']: 
                mse = self._mse(self._X_val, self._y_val, theta)
                self._mse_history.append(mse)

            # Save metrics in history
            self._J_history.append(J)
            self._theta_history.append(theta.tolist())            
            self._epochs.append(epoch)
            self._iterations.append(iteration)
            self._learning_rate_history.append(learning_rate)                

            # Update state to include current iteration, loss and mse (if cross_validated)
            state = self._update_state(state, iteration, J, mse)

            # Update learning rate
            learning_rate = self._update_learning_rate(learning_rate, epoch)



# --------------------------------------------------------------------------- #
#                     Mini-Batch Gradient Descent Search                      #
# --------------------------------------------------------------------------- #            
class MBGDFit(GradientFit):

    def __init__(self):
        self._alg = "Mini-Batch Gradient Descent"
        self._request = dict()
        self._J_history = []
        self._mse_history = []
        self._theta_history = []
        self._iterations = []
        self._epochs = []
        self._X = None
        self._y = None
        self._X_val = None
        self._y_val = None


    def _shuffle(self, X, y, random_state):
        y_var = y.name
        df = pd.concat([X,y], axis=1)
        df = df.sample(frac=1, replace=False, axis=0, random_state=random_state)
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
        learning_rate = self._request['hyper']['learning_rate']

        self._i_s = self._request['hyper']['i_s']
        self._iter_no_change = 0
        self._learning_rate_history = []        
        self._J_history = []
        self._mse_history = []
        self._theta_history = []
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
            epoch += 1        
            mse = None    
            X, y = self._shuffle(self._X, self._y, random_state=epoch)
            X_mb, y_mb = self._get_batches(X,y, self._request['hyper']['batch_size'])

            for x_i, y_i in zip(X_mb, y_mb):
                iteration += 1                

                h = self._hypothesis(x_i.values, theta)
                e = self._error(h, y_i.values)
                J = self._cost(e)
                g = self._gradient(x_i.values, e)
                theta = self._update(theta, learning_rate, g)

            # Compute total training set costs with latest theta
            J = self._total_cost(self._X, self._y, theta)

            # if cross_validated, compute validation set MSE 
            if self._request['hyper']['cross_validated']: 
                mse = self._mse(self._X_val, self._y_val, theta)  
                self._mse_history.append(mse)    

            # Save metrics in history
            self._J_history.append(J)
            self._theta_history.append(theta.tolist())
            self._epochs.append(epoch)
            self._iterations.append(iteration)
            self._learning_rate_history.append(learning_rate)                      

            # Update state to include iteration and current loss and mse (if cross_validated)
            state = self._update_state(state, iteration, J, mse)
            
            # Update learning rate
            learning_rate = self._update_learning_rate(learning_rate, epoch)

