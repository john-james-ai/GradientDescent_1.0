
# %%
# =========================================================================== #
#                             GRADIENT LAB                                    #
# =========================================================================== #

# --------------------------------------------------------------------------- #
#                                LIBRARIES                                    #
# --------------------------------------------------------------------------- #
import inspect
import os
import sys

from IPython.display import HTML
import datetime
import itertools
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('seaborn-notebook')
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from matplotlib import animation, rc
from matplotlib import colors as mcolors
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import rcParams
from mpl_toolkits.mplot3d import axes3d, Axes3D
import numpy as np
from numpy import array, newaxis
import pandas as pd
from scipy.stats.stats import pearsonr, spearmanr
import seaborn as sns
from textwrap import wrap
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from GradientDescent import BGD, SGD, MBGD
from utils import save_fig, save_csv

rcParams['animation.embed_limit'] = 60
rc('animation', html='jshtml')
rc
# --------------------------------------------------------------------------- #
#                          GRADIENTLAB BASE CLASS                             #  
# --------------------------------------------------------------------------- #
class GradientLab:
    '''
    Base class for gradient descent plots
    '''

    def __init__(self):
        self._alg = 'Gradient Descent'
        self._summary = None
        self._detail = None
        self._params = {}

    def _save_params(self, X, y, X_val, y_val, theta, learning_rate, 
                     learning_rate_sched, time_decay, step_decay,
                     step_epochs, exp_decay, precision, stop_metric,
                     i_s, maxiter, scaler):

        self._params = {'X': X, 'y':y, 'X_val': X_val, 'y_val':y_val,
                        'theta':theta, 'learning_rate':learning_rate, 
                        'stop_metric': stop_metric,
                        'learning_rate_sched': learning_rate_sched,
                        'time_decay': time_decay,
                        'step_decay': step_decay,
                        'step_epochs': step_epochs,
                        'exp_decay': exp_decay,
                        'precision':precision,
                        'i_s': i_s,
                        'maxiter':maxiter,
                        'scaler':scaler}

    def get_coef(self):
        if self._summary is None:
            raise Exception("No coefficients to report")
        else:
            best = self.summary(nbest=1)
            theta_cols = [col for col in best.columns if 'theta' in col]
            return(best[theta_cols])
            
    def summary(self, nbest=0, directory=None, filename=None):
        if self._summary is None:
            raise Exception("No summary to report")
        else:
            if directory is not None:
                if filename is None:
                    filename = self._alg + ' Lab Summary.csv'
                save_csv(self._summary, directory, filename) 
            if nbest:
                s = self._summary.sort_values(by=['final_costs', 'duration'])
                return(s.head(nbest))
            return(self._summary)

    def eval(self, nbest=0, directory=None, filename=None):
        if self._eval is None:
            raise Exception("No eval to report")
        else:
            if directory is not None:
                if filename is None:
                    filename = self._alg + ' Lab Evaluation.csv'
                save_csv(self._eval, directory, filename)             
            if nbest:
                s = self.summary(nbest=nbest)
                d = self._eval
                d = d.loc[d['experiment'].isin(s['experiment'])]
                return(d)
            return(self._eval)                

    def detail(self, nbest=0, directory=None, filename=None):
        if self._detail is None:
            raise Exception("No detail to report")
        else:
            if directory is not None:
                if filename is None:
                    filename = self._alg + ' Lab Detail.csv'
                save_csv(self._detail, directory, filename)             
            if nbest:
                s = self.summary(nbest=nbest)
                d = self._detail
                d = d.loc[d['experiment'].isin(s['experiment'])]
                return(d)
            return(self._detail)        

    def _runsearch(self, gd):
        experiment = 1
        self._summary = pd.DataFrame()
        self._eval = pd.DataFrame()
        self._detail = pd.DataFrame()
        
        # Extract parameters
        X = self._params['X']
        y = self._params['y']
        X_val = self._params['X_val']
        y_val = self._params['y_val']
        theta = self._params['theta']
        stop_metric = self._params['stop_metric']
        learning_rate = self._params['learning_rate']
        learning_rate_sched = self._params['learning_rate_sched']
        time_decay = self._params['time_decay']
        step_decay = self._params['step_decay']
        step_epochs = self._params['step_epochs']
        exp_decay = self._params['exp_decay']
        precision = self._params['precision']
        i_s = self._params['i_s']
        maxiter = self._params['maxiter']
        scaler = self._params['scaler']

        # Constant learning rates
        if 'c' in learning_rate_sched:
            for s in stop_metric:
                for n in i_s:
                    for p in precision:
                        for a in learning_rate:                    
                            gd.fit(X=X, y=y, theta=theta, X_val=X_val, y_val=y_val, 
                                    learning_rate=a, learning_rate_sched='c',                                 
                                    maxiter=maxiter, precision=p, stop_metric=s,
                                    i_s=n, scaler=scaler)
                            detail = gd.detail()
                            detail['experiment'] = experiment
                            eval = gd.eval()
                            eval['experiment'] = experiment                            
                            summary = gd.summary()
                            summary['experiment'] = experiment
                            self._detail = pd.concat([self._detail, detail], axis=0)    
                            self._eval = pd.concat([self._eval, eval], axis=0)    
                            self._summary = pd.concat([self._summary, summary], axis=0)    
                            experiment += 1               

        # Time Decay Learning Rates
        if 't' in learning_rate_sched:
            for s in stop_metric:
                for n in i_s:
                    for p in precision:
                        for a in learning_rate:
                            for d in time_decay:                    
                                gd.fit(X=X, y=y, theta=theta, X_val=X_val, y_val=y_val, 
                                        learning_rate=a, learning_rate_sched='t',
                                        time_decay=d, maxiter=maxiter, precision=p, 
                                        stop_metric=s, i_s=n, scaler=scaler)
                                detail = gd.detail()
                                detail['experiment'] = experiment
                                eval = gd.eval()
                                eval['experiment'] = experiment                            
                                summary = gd.summary()
                                summary['experiment'] = experiment
                                self._detail = pd.concat([self._detail, detail], axis=0)    
                                self._eval = pd.concat([self._eval, eval], axis=0)    
                                self._summary = pd.concat([self._summary, summary], axis=0)      
                                experiment += 1       

        # Step Decay Learning Rates
        if 's' in learning_rate_sched:
            for s in stop_metric:
                for n in i_s:
                    for p in precision:
                        for a in learning_rate:
                            for d in step_decay:                    
                                for e in step_epochs:
                                    gd.fit(X=X, y=y, theta=theta, X_val=X_val, y_val=y_val, 
                                            learning_rate=a, learning_rate_sched='s',
                                            step_decay=d, step_epochs=e, maxiter=maxiter, precision=p, 
                                            stop_metric=s, i_s=n, scaler=scaler)
                                    detail = gd.detail()
                                    detail['experiment'] = experiment
                                    eval = gd.eval()
                                    eval['experiment'] = experiment                            
                                    summary = gd.summary()
                                    summary['experiment'] = experiment
                                    self._detail = pd.concat([self._detail, detail], axis=0)    
                                    self._eval = pd.concat([self._eval, eval], axis=0)    
                                    self._summary = pd.concat([self._summary, summary], axis=0)      
                                    experiment += 1    

        # Exponential Decay Learning Rates
        if 'e' in learning_rate_sched:
            for s in stop_metric:
                for n in i_s:
                    for p in precision:
                        for a in learning_rate:
                            for d in exp_decay:                    
                                gd.fit(X=X, y=y, theta=theta, X_val=X_val, y_val=y_val, 
                                        learning_rate=a, learning_rate_sched='e',
                                        exp_decay=d, maxiter=maxiter, precision=p, 
                                        stop_metric=s, i_s=n, scaler=scaler)
                                detail = gd.detail()
                                detail['experiment'] = experiment
                                eval = gd.eval()
                                eval['experiment'] = experiment                            
                                summary = gd.summary()
                                summary['experiment'] = experiment
                                self._detail = pd.concat([self._detail, detail], axis=0)    
                                self._eval = pd.concat([self._eval, eval], axis=0)    
                                self._summary = pd.concat([self._summary, summary], axis=0)      
                                experiment += 1                                          

        
    def gridsearch(self, X, y, theta, X_val=None, y_val=None, learning_rate=0.01, 
            learning_rate_sched = 'c', time_decay=None, step_decay=None,
            step_epochs=None, exp_decay=None, maxiter=0, precision=0.001, stop_metric='j',
            i_s=5, scaler='minmax'):

        self._save_params(X=X, y=y, X_val=X_val, y_val=y_val, theta=theta, 
                         learning_rate=learning_rate, 
                         learning_rate_sched=learning_rate_sched, 
                         time_decay=time_decay, step_decay=step_decay,
                         step_epochs=step_epochs, exp_decay=exp_decay, 
                         precision=precision, i_s=i_s, 
                         stop_metric=stop_metric,
                         maxiter=maxiter, scaler=scaler)

        bgd = BGD()
        self._runsearch(bgd)

    def _get_params(self):
        x = ['learning_rate', 'learning_rate_sched', 'stop_metric', 
             'i_s', 'maxiter', 'time_decay', 'step_decay', 'step_epochs',
             'exp_decay', 'precision']
        return(x)


    def _get_label(self, x):
        labels = {'learning_rate': 'Learning Rate',
                  'learning_rates': 'Learning Rates',
                  'learning_rate_sched': 'Learning Rate Schedule',
                  'stop_metric': 'Stop Metric',
                  'i_s': 'Iterations No Change',
                  'batch_size': 'Batch Size',
                  'time_decay': 'Time Decay',
                  'step_decay': 'Step Decay',
                  'step_epochs': 'Epochs per Step',
                  'exp_decay': 'Exponential Decay',
                  'precision': 'Precision',
                  'theta': "Theta",
                  'duration': 'Computation Time (ms)',
                  'iterations': 'Iterations',
                  'cost': 'Training Set Costs',
                  'mse': 'Validation Set MSE',                  
                  'final_costs': 'Training Set Costs',
                  'final_mse': 'Validation Set MSE',
                  'c': 'Constant Learning Rate',
                  't': 'Time Decay Learning Rate',
                  's': 'Step Decay Learning Rate',
                  'e': 'Exponential Decay Learning Rate',
                  'j': 'Training Set Costs',
                  'v': 'Validation Set Error',
                  'g': 'Gradient Norm'}
        return(labels.get(x,x))

    def report(self,  n=None, sort='v', directory=None, filename=None):
        if self._detail is None:
            raise Exception('Nothing to report')
        else:
            vars = ['experiment', 'alg', 'learning_rate',
                    'learning_rate_sched', 'learning_rate_sched_label',
                    'stop_metric', 'stop_metric_label',
                    'time_decay', 'step_decay', 'step_epochs', 'exp_decay',
                    'precision', 'i_s', 'maxiter', 'stop_metric',
                    'epochs', 'iterations','duration',
                    'final_costs', 'final_mse']
            df = self._summary
            df = df[vars]
            if sort == 't':
                df = df.sort_values(by=['final_costs', 'duration'])
            else:
                df = df.sort_values(by=['final_mse', 'duration'])
            if directory:
                if filename is None:
                    filename = self._alg + ' Grid Search.csv'
                save_csv(df, directory, filename)                
            if n:
                df = df.iloc[:n]            
            return(df)

# --------------------------------------------------------------------------- #
#                              BGD Lab Class                                  #   
# --------------------------------------------------------------------------- #
class BGDLab(GradientLab):  

    def __init__(self):      
        self._alg = 'Batch Gradient Descent'
        self._summary = None
        self._detail = None
        
# --------------------------------------------------------------------------- #
#                              SGD Lab Class                                  #   
# --------------------------------------------------------------------------- #
class SGDLab(GradientLab):  

    def __init__(self):      
        self._alg = 'Stochastic Gradient Descent'
        self._summary = None
        self._detail = None

    def gridsearch(self, X, y, theta, X_val=None, y_val=None, learning_rate=0.01, 
            learning_rate_sched = 'c', time_decay=None, step_decay=None,
            step_epochs=None, exp_decay=None, maxiter=0, precision=0.001, 
            stop_metric='j', i_s=5, scaler='minmax'):
        self._save_params(X=X, y=y, X_val=X_val, y_val=y_val, 
                        theta=theta, learning_rate=learning_rate, 
                     learning_rate_sched=learning_rate_sched,
                     time_decay=time_decay, step_decay=step_decay,
                     step_epochs=step_epochs, exp_decay=exp_decay, 
                     precision=precision, stop_metric=stop_metric,
                     i_s=i_s, maxiter=maxiter, scaler=scaler)

        sgd = SGD()
        self._runsearch(sgd)

 
# --------------------------------------------------------------------------- #
#                              MBGD Lab Class                                 #   
# --------------------------------------------------------------------------- #
class MBGDLab(GradientLab):  

    def __init__(self):      
        self._alg = 'Minibatch Gradient Descent'
        self._summary = None
        self._detail = None

    def _save_params(self, X, y, X_val, y_val, batch_size, theta, learning_rate, 
                     learning_rate_sched, time_decay, step_decay,
                     step_epochs, exp_decay, precision, stop_metric,
                     i_s, maxiter, scaler):

        self._params = {'X': X, 'y':y, 'X_val': X_val, 'y_val':y_val,
                        'batch_size': batch_size,
                        'theta':theta, 'learning_rate':learning_rate, 
                        'learning_rate_sched': learning_rate_sched,
                        'stop_metric': stop_metric,
                        'time_decay': time_decay,
                        'step_decay': step_decay,
                        'step_epochs': step_epochs,
                        'exp_decay': exp_decay,
                        'precision':precision,
                        'i_s': i_s,
                        'maxiter':maxiter,
                        'scaler':scaler}

    def report(self,  n=None, sort='v', directory=None, filename=None):
        if self._detail is None:
            raise Exception('Nothing to report')
        else:
            vars = ['experiment', 'alg', 'batch_size', 'learning_rate',
                    'learning_rate_sched', 'learning_rate_sched_label',
                    'stop_metric', 'stop_metric_label',
                    'time_decay', 'step_decay', 'step_epochs', 'exp_decay',
                    'precision', 'i_s', 'maxiter', 'stop_metric',
                    'epochs', 'iterations','duration',
                    'final_costs', 'final_mse']
            df = self._summary
            df = df[vars]
            if sort == 't':
                df = df.sort_values(by=['final_costs', 'duration'])
            else:
                df = df.sort_values(by=['final_mse', 'duration'])
            if directory:
                if filename is None:
                    filename = self._alg + ' Grid Search.csv'
                save_csv(df, directory, filename)                
            if n:
                df = df.iloc[:n]            
            return(df)        

    def gridsearch(self, X, y, theta, X_val=None, y_val=None, learning_rate=0.01, 
            learning_rate_sched = 'c', time_decay=None, step_decay=None,
            step_epochs=None, exp_decay=None, maxiter=0, precision=0.001,
            batch_size=10, stop_metric='j', i_s=5, scaler='minmax'):

        self._save_params(X=X, y=y, X_val=X_val, y_val=y_val, 
                        theta=theta, learning_rate=learning_rate, 
                     learning_rate_sched=learning_rate_sched,
                     time_decay=time_decay, step_decay=step_decay,
                     step_epochs=step_epochs, exp_decay=exp_decay, 
                     precision=precision, batch_size=batch_size,
                     stop_metric=stop_metric, i_s=i_s, 
                     maxiter=maxiter, scaler=scaler)

        mbgd = MBGD()
        self._runsearch(mbgd)   
