
# %%
# =========================================================================== #
#                             GRADIENT MODEL                                  #
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
#                       GRADIENT DESCENT MODEL CLASS                          #
# --------------------------------------------------------------------------- #


class GradientModel:
    '''Gradient Model'''

    def __init__(self):
        self._X = None
        self._y = None
        self._X_val = None
        self._y_val = None
        self._alg = None

    def coef(self):
        if self._alg is None:
            raise Exception('No algorithm selected.')
        else:
            return(self._coef)

    def intercept(self):
        if self._alg is None:
            raise Exception('No algorithm selected.')
        else:
            return(self._intercept)

    def detail(self, directory=None, filename=None):
        if self._alg is None:
            raise Exception('No algorithm selected.')
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

    def fit(self, X, y, algorithm, X_val=None, y_val=None, scaler='minmax'):

        # Set cross-validated flag if validation set included 
        cross_validated = all(v is not None for v in [X_val, y_val])

        # Prepare Data
        self._X, self._y = self.prep_data(X=X, y=y, scaler=scaler)
        if cross_validated:
            self._X_val, self._y_val = self.prep_data(X=X_val, y=y_val, scaler=scaler)                    

        # Package request
        self._request = dict()
        self._request['alg'] = self._alg        
        self._request['data'] = {'X': self._X, 'y':self._y, 'X_val':self._X_val, 'y_val':self._y_val} 
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


