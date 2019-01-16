
# %%
# =========================================================================== #
#                             GRADIENT DESCENT                                #
# =========================================================================== #


# --------------------------------------------------------------------------- #
#                                LIBRARIES                                    #
# --------------------------------------------------------------------------- #

import inspect
import os
import sys

import datetime
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('seaborn-notebook')
from matplotlib import cm
from matplotlib import animation, rc
from matplotlib import colors as mcolors
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from numpy import array, newaxis
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from GradientFit import BGDFit, SGDFit, MBGDFit
from utils import save_fig, save_gif, save_csv
# --------------------------------------------------------------------------- #
#                       GRADIENT DESCENT BASE CLASS                           #
# --------------------------------------------------------------------------- #


class GradientDescent:
    '''Base class for Gradient Descent'''

    def __init__(self)->None:
        self._alg = "Gradient Descent"
        self._summary = None
        self._detail = None

    def get_hyper(self):
        return(self._request['hyper'])

    def get_transformed_data(self):
        return(self._request['data'])

    def detail(self, directory=None, filename=None):
        if self._detail is None:
            raise Exception('No search results to report.')
        else:
            if directory is not None:
                if filename is None:
                    filename = self._alg + ' Detail.csv'
                save_csv(self._detail, directory, filename)             
            return(self._detail)    

    def eval(self, directory=None, filename=None):
        if self._eval is None:
            raise Exception('No search results to report.')
        else:
            if directory is not None:
                if filename is None:
                    filename = self._alg + ' Evaluation.csv'
                save_csv(self._eval, directory, filename)             
            return(self._eval)    

    def summary(self, directory=None, filename=None):
        if self._summary is None:
            raise Exception('No search results to report.')
        else:
            if directory is not None:
                if filename is None:
                    filename = self._alg + ' Summary.csv'
                save_csv(self._summary, directory, filename) 
            return(self._summary)

    def _encode_labels(self, X, y):
        le = LabelEncoder()
        X = X.apply(le.fit_transform)        
        y = y.apply(le.fit_transform)
        return(X, y)

    def _scale(self, X, y, scaler, bias=True):
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

    def prep_data(self, X, y, scaler='minmax', bias=True):        
        X, y = self._encode_labels(X,y)
        X, y = self._scale(X,y, scaler, bias)
        return(X,y)         

    def _validate(self):
        if (self._request['hyper']['learning_rate_sched'] == 't' and 
           self._request['hyper']['time_decay'] is None):
           raise Exception("Time decay parameter required for time decay learning rate schedule.") 
        elif (self._request['hyper']['learning_rate_sched'] == 's' and  
           self._request['hyper']['step_decay'] is None):
           raise Exception("Step decay parameter required for step decay learning rate schedule.") 
        elif (self._request['hyper']['learning_rate_sched'] == 's' and 
           self._request['hyper']['step_epochs'] is None):
           raise Exception("Step epochs parameter required for step decay learning rate schedule.") 
        elif (self._request['hyper']['learning_rate_sched'] == 'e' and 
           self._request['hyper']['exp_decay'] is None):
           raise Exception("Exp decay parameter required for step decay learning rate schedule.") 
        elif self._request['hyper']['learning_rate_sched'] not in ['t', 's', 'e', 'c']:
            raise Exception("Learning rate schedule must be in ['c', 't', 'e', 's'].") 
        elif self._request['hyper']['stop_metric'] not in ['j', 'v', 'g']:
            raise Exception("Stop metric must be in  ['j', 'v', 'g'].")                         

    def _setup(self, X, y, theta, X_val=None, y_val=None, learning_rate=0.01, 
            batch_size=None, learning_rate_sched = 'c', time_decay=None, step_decay=None,
            step_epochs=None, exp_decay=None, maxiter=0, precision=0.001, 
            i_s=5, stop_metric='j', scaler='minmax'):

        # Set cross-validated flag if validation set included 
        cross_validated = all(v is not None for v in [X_val, y_val])

        # Prepare Data
        X, y = self.prep_data(X=X, y=y, scaler=scaler)
        if cross_validated:
            X_val, y_val = self.prep_data(X=X_val, y=y_val, scaler=scaler)                    

        # Package request
        self._request = dict()
        self._request['alg'] = self._alg        
        self._request['data'] = {'X': X, 'y':y, 'X_val':X_val, 'y_val':y_val} 
        self._request['hyper'] = {'learning_rate': learning_rate, 'theta': theta,
                                  'learning_rate_sched': learning_rate_sched,
                                  'batch_size': batch_size,
                                  'time_decay': time_decay,
                                  'step_decay': step_decay,
                                  'step_epochs': step_epochs,
                                  'exp_decay': exp_decay,
                                  'maxiter': maxiter, 
                                  'precision': precision,                                  
                                  'i_s': i_s,
                                  'stop_metric': stop_metric,
                                  'cross_validated': cross_validated}
        
        self._validate()


# --------------------------------------------------------------------------- #
#                       BATCH GRADIENT DESCENT CLASS                          #
# --------------------------------------------------------------------------- #

class BGD(GradientDescent):
    '''Batch Gradient Descent'''

    def __init__(self)->None:
        self._alg = "Batch Gradient Descent"
        self._request = None
        self._summary = None
        self._detail = None
    
    def fit (self, X, y, theta, X_val=None, y_val=None, learning_rate=0.01, 
            batch_size=None, learning_rate_sched = 'c', time_decay=None, step_decay=None,
            step_epochs=None, exp_decay=None, maxiter=0, precision=0.001, 
            i_s=5, stop_metric='j', scaler='minmax'):

        self._setup(X=X, y=y, theta=theta, X_val=X_val, y_val=y_val, 
                    learning_rate=learning_rate, batch_size=batch_size, 
                    learning_rate_sched = learning_rate_sched, 
                    time_decay=time_decay, step_decay=step_decay,
                    step_epochs=step_epochs, exp_decay=exp_decay, 
                    maxiter=maxiter, precision=precision, 
                    i_s=i_s, stop_metric=stop_metric, scaler=scaler)    

        gd = BGDFit()
        gd.fit(self._request)
        self._detail = gd.detail()
        self._summary = gd.summary()  
        self._eval = gd.eval()                  

        
# --------------------------------------------------------------------------- #
#                     STOCHASTIC GRADIENT DESCENT CLASS                       #
# --------------------------------------------------------------------------- #

class SGD(GradientDescent):
    '''Stochastic Gradient Descent'''

    def __init__(self)->None:
        self._alg = "Stochastic Gradient Descent"
        self._request = None
        self._summary = None
        self._detail = None
    
    def fit (self, X, y, theta, X_val=None, y_val=None, learning_rate=0.01, 
            batch_size=None, learning_rate_sched = 'c', time_decay=None, step_decay=None,
            step_epochs=None, exp_decay=None, maxiter=0, precision=0.001, 
            i_s=5, stop_metric='j', scaler='minmax'):

        self._setup(X=X, y=y, theta=theta, X_val=X_val, y_val=y_val, 
                    learning_rate=learning_rate, batch_size=batch_size, 
                    learning_rate_sched = learning_rate_sched, 
                    time_decay=time_decay, step_decay=step_decay,
                    step_epochs=step_epochs, exp_decay=exp_decay, 
                    maxiter=maxiter, precision=precision, 
                    i_s=i_s, stop_metric=stop_metric, scaler=scaler)    

        gd = SGDFit()
        gd.fit(self._request)
        self._detail = gd.detail()
        self._summary = gd.summary()   
        self._eval = gd.eval()


# --------------------------------------------------------------------------- #
#                     MINI-BATCH GRADIENT DESCENT CLASS                       #
# --------------------------------------------------------------------------- #

class MBGD(GradientDescent):
    '''Mini-Batch Gradient Descent'''

    def __init__(self)->None:
        self._alg = "Mini-Batch Gradient Descent"
        self._request = None
        self._summary = None
        self._detail = None
    
    def fit (self, X, y, theta, X_val=None, y_val=None, learning_rate=0.01, 
            batch_size=None, learning_rate_sched = 'c', time_decay=None, step_decay=None,
            step_epochs=None, exp_decay=None, maxiter=0, precision=0.001, 
            i_s=5, stop_metric='j', scaler='minmax'):

        self._setup(X=X, y=y, theta=theta, X_val=X_val, y_val=y_val, 
                    learning_rate=learning_rate, batch_size=batch_size, 
                    learning_rate_sched = learning_rate_sched, 
                    time_decay=time_decay, step_decay=step_decay,
                    step_epochs=step_epochs, exp_decay=exp_decay, 
                    maxiter=maxiter, precision=precision, 
                    i_s=i_s, stop_metric=stop_metric, scaler=scaler)    

        gd = MBGDFit()
        gd.fit(self._request)
        self._detail = gd.detail()
        self._summary = gd.summary()   
        self._eval = gd.eval()

