# %%
# =========================================================================== #
#                             Gradient Descent                                #
# =========================================================================== #
from abc import ABC, abstractmethod
import datetime
import math
import numpy as np
from numpy import array, newaxis
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
# --------------------------------------------------------------------------- #
#                        Gradient Search Base Class                           #
# --------------------------------------------------------------------------- #


class GradientDescent(ABC):
    '''Abstract base class for Gradient Descent'''

    def __init__(self):

        self._alg = "Abstract Gradient Descent"

        # Parameters
        self._theta = None
        self._learning_rate = None
        self._learning_rate_sched = None
        self._precision = None
        self._batch_size = None
        self._time_decay = None
        self._step_decay = None
        self._step_epochs = None
        self._exp_decay = None
        self._maxiter = None
        self._i_s = None
        self._stop_metric = None

        
        self._learning_rate_detail_history = []
        self._J_detail_history = []             
        self._theta_detail_history = [] 
        self._iterations_detail_history = []    
        self._epochs_detail_history = []

        self._learning_rate_eval_history = []
        self._J_eval_history = []             
        self._mse_eval_history = []     
        self._theta_eval_history = [] 
        self._iterations_eval_history = []    
        self._epochs_eval_history = []
            
        # Data
        self._X = None           
        self._y = None
        self._X_val = None
        self._y_val = None

        self._i_s = 5
        self._iter_no_change = 0

        # Time variables
        self._start = None
        self._end =None        

    @property
    def theta(self):
        return(self._theta)

    @theta.setter
    def theta(self, theta):
        self._theta = theta 

    @property
    def learning_rate(self):
        return(self._learning_rate)

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self._learning_rate = learning_rate 

    @property
    def learning_rate_sched(self):
        return(self._learning_rate_sched)

    @learning_rate_sched.setter
    def learning_rate_sched(self, learning_rate_sched):
        self._learning_rate_sched = learning_rate_sched

    @property
    def precision(self):
        return(self.precision)

    @precision.setter
    def precision(self, precision):
        self._precision = precision         

    @property
    def time_decay(self):
        return(self._time_decay)

    @time_decay.setter
    def time_decay(self, time_decay):
        self._time_decay = time_decay 

    @property
    def step_decay(self):
        return(self._step_decay)

    @step_decay.setter
    def step_decay(self, step_decay):
        self._step_decay = step_decay 

    @property
    def step_epochs(self):
        return(self._step_epochs)

    @step_epochs.setter
    def step_epochs(self, step_epochs):
        self._step_epochs = step_epochs         

    @property
    def exp_decay(self):
        return(self._exp_decay)

    @exp_decay.setter
    def exp_decay(self, exp_decay):
        self._exp_decay = exp_decay   

    @property
    def maxiter(self):
        return(self._maxiter)

    @maxiter.setter
    def maxiter(self, maxiter):
        self._maxiter = maxiter  

    @property
    def i_s(self):
        return(self._i_s)

    @i_s.setter
    def i_s(self, i_s):
        self._i_s = i_s

    @property
    def stop_metric(self):
        return(self._stop_metric)

    @stop_metric.setter
    def stop_metric(self, stop_metric):
        self._stop_metric = stop_metric

    def get_params(self):
        params = {'alg': self._alg,
                  'theta': self._theta, 
                  'learning_rate_sched': self._learning_rate_sched,
                  'learning_rate_sched_label': self._get_label(self._learning_rate_sched),                  
                  'learning_rate': self._learning_rate,
                  'batch_size': self._batch_size,
                  'precision': self._precision,
                  "time_decay": self._time_decay,
                  "step_decay": self._step_decay,
                  "step_epochs": self._step_epochs,
                  "exp_decay": self._exp_decay,
                  "maxiter": self._maxiter,
                  "precision": self._precision,
                  "i_s": self._i_s,
                  "stop_metric": self._stop_metric,
                  'stop_metric_label': self._get_label(self._stop_metric)}
        return(params)

    def _get_train_log(self):
        log = {'epochs': self._epochs_detail_history,
               'iterations': self._iterations_detail_history,
               'learning_rates': self._learning_rate_detail_history,
               'cost': self._J_detail_history}
        return(log)

    def _get_eval_log(self):
        log = {'epochs': self._epochs_eval_history,
               'iterations': self._iterations_eval_history,
               'learning_rates': self._learning_rate_eval_history,
               'cost': self._J_eval_history,
               'mse': self._mse_eval_history}
        return(log)        

    def detail(self):
        params = self.get_params()
        params_df = pd.DataFrame.from_dict(params)

        train_log = self._get_train_log()
        train_df = pd.DataFrame.from_dict(train_log)

        thetas = self._todf(self._theta_detail_history, stub='theta_')
        
        log_df = pd.concat([params_df, train_df], axis=1)
        log_df = pd.concat([log_df, thetas], axis=1)
        
        return(log_df)        

    def summary(self):
        params = self.get_params()
        params_df = pd.DataFrame.from_dict(params)

        if self._mse_eval_history is None:
            initial_mse = None
            final_mse = None
        else:
            initial_mse = self._mse_eval_history[0]
            final_mse = self._mse_eval_history[-1]        

        summary = {'start' : self._start,
                   'end' : self._end,
                   'duration':(self._end-self._start).total_seconds(),                   
                   'epochs': self._epochs_detail_history[-1],
                   'iterations': self._iterations_detail_history[-1],
                   'initial_costs': self._J_detail_history[0],
                   'final_costs': self._J_detail_history[-1],
                   'initial_mse': initial_mse,
                   'final_mse': final_mse}
        summary_df = pd.DataFrame.from_dict(summary)

        thetas = self._todf(self._theta_detail_history, stub='theta_')
        
        summary_df = pd.concat([params_df, summary_df], axis=1)
        summary_df = pd.concat([summary_df, thetas], axis=1)
                
        return(summary_df)    
 

    def intercept(self):
        return(self._theta_detail_history[-1][0]) 

    def coef(self):
        return(np.array(self._theta_detail_history[-1][1:]))

    def _get_label(self, x):
        labels = {'c': 'Constant Learning Rate',
                  't': 'Time Decay Learning Rate',
                  's': 'Step Decay Learning Rate',
                  'e': 'Exponential Decay Learning Rate',
                  'j': 'Training Set Costs',
                  'v': 'Validation Set Error',
                  'g': 'Gradient Norm'}
        return(labels.get(x,x))        

    def _todf(self, x, stub):
        n = len(x[0])
        df = pd.DataFrame()
        for i in range(n):
            colname = stub + str(i)
            vec = [item[i] for item in x]
            df_vec = pd.DataFrame(vec, columns=[colname])
            df = pd.concat([df, df_vec], axis=1)
        return(df)         

    def _hypothesis(self, X, theta):
        return(X.dot(theta))

    def _error(self, h, y):
        return(h-y)

    def _cost(self, e):
        return(1/2 * np.mean(e**2))

    def _total_cost(self, X, y, theta):
        h = self._hypothesis(X, theta)
        e = self._error(h, y)
        return(1/2 * np.mean(e**2))  
    
    def _mse(self, X, y, theta):
        h = self._hypothesis(X, theta)
        e = self._error(h,y)
        return(np.mean(e**2))

    def _gradient(self, X, e):
        return(X.T.dot(e)/X.shape[0])

    def _gradient_norm(self, g):
        return(np.sqrt(g.dot(g)))    

    def _update(self, theta, learning_rate, gradient):        
        return(theta-(learning_rate * gradient))

    def _update_learning_rate(self, learning_rate, epoch):
        learning_rate_init = self._learning_rate
        if self._learning_rate_sched == 'c':
            learning_rate_new = learning_rate
        elif self._learning_rate_sched == 't':
            k = self._time_decay
            learning_rate_new = learning_rate_init/(1+k*epoch)            
        elif self._learning_rate_sched == 's':
            drop = self._step_decay
            epochs_drop = self._step_epochs
            learning_rate_new = learning_rate_init*math.pow(drop, math.floor((1+epoch)/epochs_drop))
        elif self._learning_rate_sched == 'e':            
            k = self._exp_decay
            learning_rate_new = learning_rate_init * math.exp(-k*epoch)
        return(learning_rate_new)

    def _maxed_out(self, iteration):
        if self._maxiter:
            if iteration == self._maxiter:
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
            self._iter_no_change += 1
            if self._iter_no_change >= self._i_s:
                return(True)
            else:
                return(False)

        elif state['prior']-state['current'] < self._precision:
            self._iter_no_change += 1
            if self._iter_no_change >= self._i_s:
                return(True)
            else:
                return(False)
        else:
            self._iter_no_change = 0
            return(False)

    def _update_state(self, state, iteration, J, mse, g):
        state['iteration'] = iteration
        state['prior'] = state['current']
        if self._stop_metric == 'j':
            state['current'] = J
        elif self._stop_metric == 'v':
            state['current'] = mse
        elif self._stop_metric == 'g':
            state['current'] = g            
        return(state)

    def _update_history(self, J, theta, epoch, iteration, learning_rate, mse, what='d'):

        if what == 'd':
            self._learning_rate_detail_history.append(learning_rate)
            self._J_detail_history.append(J)            
            self._theta_detail_history.append(theta.tolist())            
            self._iterations_detail_history.append(iteration)
            self._epochs_detail_history.append(epoch)
        else:
            self._J_eval_history.append(J)
            self._theta_eval_history.append(theta.tolist())            
            self._epochs_eval_history.append(epoch)
            self._iterations_eval_history.append(iteration)
            self._learning_rate_eval_history.append(learning_rate)   
            self._mse_eval_history.append(mse)             


    @abstractmethod
    def fit(self, X, y, **params):
        pass

# --------------------------------------------------------------------------- #
#                       Batch Gradient Descent Search                         #
# --------------------------------------------------------------------------- #            
class BGD(GradientDescent):
    '''Batch Gradient Descent'''

    def __init__(self, theta=None, X_val=None, y_val=None, learning_rate=0.01,
                 learning_rate_sched='c', batch_size=10, time_decay=None, 
                 step_decay=None, step_epochs=None,  exp_decay=None, maxiter=0, 
                 precision=0.001, i_s=5, stop_metric='j' ):

        self._alg = "Batch Gradient Descent"
        
        self._learning_rate_detail_history = []
        self._J_detail_history = []             
        self._theta_detail_history = [] 
        self._iterations_detail_history = []    
        self._epochs_detail_history = []

        self._learning_rate_eval_history = []
        self._J_eval_history = []             
        self._mse_eval_history = []     
        self._theta_eval_history = [] 
        self._iterations_eval_history = []    
        self._epochs_eval_history = []

        # Data
        self._X = None           
        self._y = None
        self._X_val = None
        self._y_val = None

        # Parameters
        self._theta = None
        self._learning_rate = None
        self._learning_rate_sched = None
        self._precision = None
        self._time_decay = None
        self._step_decay = None
        self._step_epochs = None
        self._exp_decay = None
        self._maxiter = None
        self._precision = None
        self._i_s = None
        self._stop_metric = None

        self._i_s = 5
        self._iter_no_change = 0

        self._start = None
        self._end =None      


    def fit(self,  X, y, **params): 

        self._start = datetime.datetime.now()

        # Store Data
        self._X = X
        self._y = y
        self._X_val = params.get('X_val', None)
        self._y_val = params.get('y_val', None)

        # Initialize lists
        self._learning_rate_detail_history = []
        self._J_detail_history = []             
        self._theta_detail_history = [] 
        self._iterations_detail_history = []    
        self._epochs_detail_history = []

        self._learning_rate_eval_history = []
        self._J_eval_history = []             
        self._mse_eval_history = []     
        self._theta_eval_history = [] 
        self._iterations_eval_history = []    
        self._epochs_eval_history = []

        # Parameters
        self._theta = params['theta']
        self._learning_rate = params.get('learning_rate', .01)
        self._learning_rate_sched = params.get('learning_rate_sched', 'c')
        self._time_decay = params.get('time_decay', 0)
        self._step_decay = params.get('step_decay',0)
        self._step_epochs = params.get('step_epochs',10)
        self._exp_decay = params.get('exp_decay',0)
        self._maxiter = params.get('maxiter',5000)
        self._precision = params.get('precision', 0.001)
        self._i_s = params.get('i_s',5)
        self._stop_metric = params.get('stop_metric','j')

        # Initialize counters
        iteration = 0
        
        # Initialize state variables       
        J = None
        g_norm = None
        mse = None       
        theta = self._theta
        learning_rate = self._learning_rate
        state = {'prior':100, 'current':10, 'iteration':0}
        
        while not self._finished(state):
            iteration += 1

            # Compute the costs and validation set error (if required)
            h = self._hypothesis(self._X, theta)
            e = self._error(h, self._y)
            J = self._cost(e)
            if self._X_val is not None:
                mse = self._mse(self._X_val, self._y_val, theta)            

            # Save iterations, costs and thetas in history 
            self._update_history(J, theta, iteration, iteration, learning_rate, mse, what='d')
            self._update_history(J, theta, iteration, iteration, learning_rate, mse, what='e')

            # Compute the gradient and its L2 Norm
            g = self._gradient(self._X, e)
            g_norm = self._gradient_norm(g)            

            # Update thetas 
            theta = self._update(theta, learning_rate, g)

            # Update learning rate
            learning_rate = self._update_learning_rate(learning_rate, iteration)

            # Update state vis-a-vis training set costs and validation set mse 
            state = self._update_state(state, iteration, J, mse, g_norm)

        self._end = datetime.datetime.now()
        return(theta)

