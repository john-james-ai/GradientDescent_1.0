
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
from utils import save_fig, save_gif
# --------------------------------------------------------------------------- #
#                       GRADIENT DESCENT BASE CLASS                           #
# --------------------------------------------------------------------------- #


class GradientDescent:
    '''Base class for Gradient Descent'''

    def __init__(self)->None:
        self._alg = "Gradient Descent"
        self._summary = None
        self._detail = None

    def _todf(self, x, stub):
        n = len(x[0])
        df = pd.DataFrame()
        for i in range(n):
            colname = stub + str(i)
            vec = [item[i] for item in x]
            df_vec = pd.DataFrame(vec, columns=[colname])
            df = pd.concat([df, df_vec], axis=1)
        return(df) 

    def get_hyper(self):
        return(self._request['hyper'])

    def get_transformed_data(self):
        return(self._request['data'])

    def detail(self):
        if self._detail is None:
            raise Exception('No search results to report.')
        else:
            return(self._detail)

    def summary(self):
        if self._summary is None:
            raise Exception('No search results to report.')
        else:
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

    def _prep_data(self, X, y, scaler='minmax', bias=True):        
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

    def _get_label(self, x):
        labels = {'j': 'Training Set Cost',
                  'v': 'Validation Error',
                  'g': 'Gradient Norm',
                  'c': 'Constant Learning Rate',
                  't': 'Time Decay Learning Rate',
                  'e': 'Exponential Decay Learning Rate',
                  's': 'Step Decay Learning Rate'}
        return(labels[x])

    def fit(self, X, y, theta, X_val=None, y_val=None, learning_rate=0.01, 
            learning_rate_sched = 'c', time_decay=None, step_decay=None,
            step_epochs=None, exp_decay=None, maxiter=0, precision=0.001, 
            i_s=5, stop_metric='j', scaler='minmax'):

        # Set cross-validated flag if validation set included 
        cross_validated = all(v is not None for v in [X_val, y_val])

        # Prepare Data
        X, y = self._prep_data(X=X, y=y, scaler=scaler)
        if cross_validated:
            X_val, y_val = self._prep_data(X=X_val, y=y_val, scaler=scaler)                    

        # Package request
        self._request = dict()
        self._request['alg'] = self._alg        
        self._request['data'] = {'X': X, 'y':y, 'X_val':X_val, 'y_val':y_val} 
        self._request['hyper'] = {'learning_rate': learning_rate, 'theta': theta,
                                  'learning_rate_sched': learning_rate_sched,
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

        # Run search and obtain result        
        gd = BGDFit()
        start = datetime.datetime.now()
        gd.fit(self._request)
        end = datetime.datetime.now()        

        # Extract detail information
        epochs = pd.DataFrame(gd.get_epochs(), columns=['epochs'])        
        iterations = pd.DataFrame(gd.get_iterations(), columns=['iterations'])
        alphas = pd.DataFrame(gd.get_learning_rates(), columns=['learning_rates'])
        thetas = self._todf(gd.get_thetas(), stub='theta_')                
        J = pd.DataFrame(gd.get_costs(), columns=['cost'])   
        self._detail = pd.concat([epochs, iterations, alphas, thetas, J], axis=1)        
        if cross_validated:            
            mse = pd.DataFrame(gd.get_mse(), columns=['mse'])               
            self._detail = pd.concat([self._detail, mse], axis=1)
        self._detail['alg'] = self._alg
        self._detail['learning_rate_sched'] = learning_rate_sched
        self._detail['learning_rate_sched_label'] = self._get_label(learning_rate_sched)
        self._detail['learning_rate'] = learning_rate
        self._detail['time_decay'] = time_decay
        self._detail['step_decay'] = step_decay
        self._detail['step_epochs'] = step_epochs
        self._detail['exp_decay'] = exp_decay
        self._detail['precision'] = precision    
        self._detail['stop_metric'] = stop_metric
        self._detail['stop_metric_label'] = self._get_label(stop_metric)
        self._detail['i_s'] = self._request['hyper']['i_s']
        
        # Package summary results
        self._summary = pd.DataFrame({'alg': gd.get_alg(),
                                    'learning_rate_sched': learning_rate_sched,
                                    'learning_rate_sched_label': self._get_label(learning_rate_sched),
                                    'learning_rate': learning_rate,
                                    'time_decay': time_decay,
                                    'step_decay': step_decay,
                                    'step_epochs': step_epochs,
                                    'exp_decay': exp_decay,
                                    'precision': precision,
                                    'maxiter': maxiter,
                                    'stop_metric': stop_metric,
                                    'stop_metric_label': self._get_label(stop_metric),
                                    'i_s': i_s,
                                    'start':start,
                                    'end':end,
                                    'duration':(end-start).total_seconds(),
                                    'epochs': epochs.iloc[-1].item(),
                                    'iterations': iterations.iloc[-1].item(),
                                    'initial_costs': J.iloc[0].item(),
                                    'final_costs': J.iloc[-1].item()},
                                    index=[0])
        if cross_validated:                                    
            self._summary['initial_mse'] = mse.iloc[0].item()
            self._summary['final_mse'] = mse.iloc[-1].item()
        else:
            self._summary['initial_mse'] = None
            self._summary['final_mse'] = None
    
    def plot(self, directory=None, filename=None, show=True):

        # Obtain matplotlib figure
        fig = plt.figure(figsize=(12,6))
        gs = fig.add_gridspec(1,2)
        sns.set(style="whitegrid", font_scale=1)

        # Logic to format learning rate title
        def lr_title():
            title = 'Learning Rate Curve' + '\n' + \
                    self._get_label(self._request['hyper']['learning_rate_sched'])
            if self._request['hyper']['learning_rate_sched'] == 't':
                title = title + '\n' + 'Time Decay Factor: ' + \
                        str(self._request['hyper']['time_decay'])             
            elif self._request['hyper']['learning_rate_sched'] == 's':
                title = title + '\n' + 'Step Decay Factor: ' + \
                        str(self._request['hyper']['step_decay']) + '\n' + \
                        'Epochs per Step: ' + str(self._request['hyper']['step_epochs'])
            elif self._request['hyper']['learning_rate_sched'] == 'e':
                title = title + '\n' + 'Exponential Decay Factor: ' + \
                str(self._request['hyper']['exp_decay'])
            return(title)

        
        if self._request['hyper']['cross_validated']:

            # -----------------------  Cost bar plot  ------------------------ #
            # Data
            x = ['Initial', 'Initial', 'Final', 'Final',]
            y = [self._summary['initial_costs'].values.tolist(),
                 self._summary['initial_mse'].values.tolist(),
                 self._summary['final_costs'].values.tolist(),
                 self._summary['final_mse'].values.tolist()]
            z = ['Training Set Costs', 'Validation Set Error', 'Training Set Costs', 'Validation Set Error']
            y = [item for sublist in y for item in sublist]
           
            # Main plot
            ax0 = fig.add_subplot(gs[0,0])
            ax0 = sns.barplot(x,y, hue=z)  
            # Face, text, and label colors
            ax0.set_facecolor('w')
            ax0.tick_params(colors='k')
            ax0.xaxis.label.set_color('k')
            ax0.yaxis.label.set_color('k')        
            # Axes labels
            ax0.set_ylabel('Cost/Error')            
            # Values above bars
            rects = ax0.patches
            for rect, label in zip(rects, y):
                ax0.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + .03, str(round(rect.get_height(),3)),
                        ha='center', va='bottom')         
            ax0.set_title('Initial and Final Costs and Error', color='k', pad=15)
            
            # -----------------------  Cost line plot ------------------------ #
            df_train = pd.DataFrame({'Iteration': self._detail['iterations'],
                                     'Costs': self._detail['cost'],
                                     'Dataset': 'Train'})

            ax1 = fig.add_subplot(gs[0,1])
            ax1 = sns.lineplot(x='Iteration', y='Costs', data=df_train)
            ax1.set_facecolor('w')
            ax1.tick_params(colors='k')
            ax1.xaxis.label.set_color('k')
            ax1.yaxis.label.set_color('k')
            ax1.set_xlabel('Iterations')
            ax1.set_ylabel('Cost')
            title = lr_title()
            ax1.set_title(title, color='k', pad=15)
        else:
            # Cost bar plot
            x = ['Initial Cost', 'Final Costs']
            y = [self._summary['initial_costs'].values.tolist(),
                self._summary['final_costs'].values.tolist()]
            y = [item for sublist in y for item in sublist]
            ax0 = fig.add_subplot(gs[0,0])
            ax0 = sns.barplot(x,y)  
            ax0.set_facecolor('w')
            ax0.tick_params(colors='k')
            ax0.xaxis.label.set_color('k')
            ax0.yaxis.label.set_color('k')        
            ax0.set_ylabel('Cost')
            rects = ax0.patches
            for rect, label in zip(rects, y):
                height = rect.get_height()
                ax0.text(rect.get_x() + rect.get_width() / 2, height + .03, str(round(label,3)),
                        ha='center', va='bottom')                
            ax0.set_title('Initial and Final Costs and Error', color='k')

            # Cost Line Plot
            x = self._detail['iterations']
            y = self._detail['cost']
            ax1 = fig.add_subplot(gs[0,1])
            ax1 = sns.lineplot(x,y)
            ax1.set_facecolor('w')
            ax1.tick_params(colors='k')
            ax1.xaxis.label.set_color('k')
            ax1.yaxis.label.set_color('k')
            ax1.set_xlabel('Iterations')
            ax1.set_ylabel('Cost')
            title = lr_title()
            ax1.set_title(title, color='k')

        suptitle = self._alg + '\n' + 'Stop Metric: ' + self._get_label(self._request['hyper']['stop_metric']) + '\n' + \
                   'Learning Rate Schedule: ' + self._get_label(self._request['hyper']['learning_rate_sched']) + '\n' + \
                   r'$\alpha_0$' + " = " + str(round(self._request['hyper']['learning_rate'],3)) + ' ' +\
                   r'$\epsilon$' + " = " + str(round(self._request['hyper']['precision'],5)) 
        fig.suptitle(suptitle)   
        fig.tight_layout(rect=[0,0,1,.9])
        if show:
            plt.show()
        if directory is not None:
            if filename is None:
                filename = self._alg + ' Stop Metric ' + self._get_label(self._request['hyper']['stop_metric']) +  \
                   " Learning Rate Schedule " + self._get_label(self._request['hyper']['learning_rate_sched']) + \
                   " Learning Rate = " + str(self._request['hyper']['learning_rate']) +\
                   " Precision = " + str(round(self._request['hyper']['precision'],5)) 
                filename = filename.replace('\n', '')
                filename = filename.replace('  ', ' ')
                filename = filename.replace(':', '') + '.png'
            save_fig(fig, directory, filename)
        plt.close(fig)        
        return(fig)  

# --------------------------------------------------------------------------- #
#                       BATCH GRADIENT DESCENT CLASS                          #
# --------------------------------------------------------------------------- #

class BGD(GradientDescent):
    '''Batch Gradient Descent'''

    def __init__(self)->None:
        self._alg = "Batch Gradient Descent"
        self._summary = None
        self._detail = None
# --------------------------------------------------------------------------- #
#                     STOCHASTIC GRADIENT DESCENT CLASS                       #
# --------------------------------------------------------------------------- #

class SGD(GradientDescent):
    '''Stochastic Gradient Descent'''

    def __init__(self)->None:
        self._alg = "Stochastic Gradient Descent"        
        self._summary = None
        self._detail = None 

    def fit(self, X, y, theta,  X_val=None, y_val=None,  learning_rate=0.01, 
            learning_rate_sched = 'c', time_decay=None, step_decay=None,
            step_epochs=None, exp_decay=None, maxiter=0, precision=0.001, i_s=5, 
            stop_metric='j', scaler='minmax'):

        # Set cross-validated flag if validation set included 
        cross_validated = all(v is not None for v in [X_val, y_val])

        # Prepare Data
        X, y = self._prep_data(X=X, y=y, scaler=scaler)
        if cross_validated:
            X_val, y_val = self._prep_data(X=X_val, y=y_val, scaler=scaler)                    

        # Package request
        self._request = dict()
        self._request['alg'] = self._alg        
        self._request['data'] = {'X': X, 'y':y, 'X_val':X_val, 'y_val':y_val} 
        self._request['hyper'] = {'learning_rate': learning_rate, 'theta': theta,
                                  'learning_rate_sched': learning_rate_sched,
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

        # Run search and obtain result        
        gd = SGDFit()
        start = datetime.datetime.now()
        gd.fit(self._request)
        end = datetime.datetime.now()

        # Extract detail information
        epochs = pd.DataFrame(gd.get_epochs(), columns=['epochs'])        
        iterations = pd.DataFrame(gd.get_iterations(), columns=['iterations'])
        alphas = pd.DataFrame(gd.get_learning_rates(), columns=['learning_rates'])
        thetas = self._todf(gd.get_thetas(), stub='theta_')        
        J = pd.DataFrame(gd.get_costs(), columns=['cost'])   
        self._detail = pd.concat([epochs, iterations, alphas, thetas, J], axis=1)        
        if cross_validated:            
            mse = pd.DataFrame(gd.get_mse(), columns=['mse'])               
            self._detail = pd.concat([self._detail, mse], axis=1)
        self._detail['alg'] = self._alg
        self._detail['learning_rate_sched'] = learning_rate_sched
        self._detail['learning_rate_sched_label'] = self._get_label(learning_rate_sched)
        self._detail['learning_rate'] = learning_rate
        self._detail['time_decay'] = time_decay
        self._detail['step_decay'] = step_decay
        self._detail['step_epochs'] = step_epochs
        self._detail['exp_decay'] = exp_decay
        self._detail['precision'] = precision    
        self._detail['stop_metric'] = stop_metric
        self._detail['stop_metric_label'] = self._get_label(stop_metric)
        self._detail['i_s'] = self._request['hyper']['i_s']

        
        # Package summary results
        self._summary = pd.DataFrame({'alg': gd.get_alg(),
                                    'learning_rate_sched': learning_rate_sched,
                                    'learning_rate_sched_label': self._get_label(learning_rate_sched),
                                    'learning_rate': learning_rate,
                                    'time_decay': time_decay,
                                    'step_decay': step_decay,
                                    'step_epochs': step_epochs,
                                    'exp_decay': exp_decay,                                    
                                    'precision': precision,
                                    'maxiter': maxiter,
                                    'stop_metric': stop_metric,
                                    'stop_metric_label': self._get_label(stop_metric),
                                    'i_s': i_s,
                                    'start':start,
                                    'end':end,
                                    'duration':(end-start).total_seconds(),
                                    'epochs': epochs.iloc[-1].item(),
                                    'iterations': iterations.iloc[-1].item(),
                                    'initial_costs': J.iloc[0].item(),
                                    'final_costs': J.iloc[-1].item()},
                                    index=[0])
        if cross_validated:                                    
            self._summary['initial_mse'] = mse.iloc[0].item()
            self._summary['final_mse'] = mse.iloc[-1].item()
        else:
            self._summary['initial_mse'] = None
            self._summary['final_mse'] = None  

# --------------------------------------------------------------------------- #
#                     MINI-BATCH GRADIENT DESCENT CLASS                       #
# --------------------------------------------------------------------------- #

class MBGD(GradientDescent):
    '''Mini-Batch Gradient Descent'''

    def __init__(self)->None:
        self._alg = "Mini-Batch Gradient Descent"        
        self._summary = None
        self._detail = None 
    def fit(self, X, y, theta,  X_val=None, y_val=None,  batch_size=.1, 
            learning_rate=0.01, learning_rate_sched = 'c', time_decay=None, step_decay=None,
            step_epochs=None, exp_decay=None, maxiter=0, precision=0.001, i_s=5,
            stop_metric='j', scaler='minmax'):            

        # Set cross-validated flag if validation set included 
        cross_validated = all(v is not None for v in [X_val, y_val])
        
        # Prepare Data
        X, y = self._prep_data(X=X, y=y, scaler=scaler)
        if cross_validated:
            X_val, y_val = self._prep_data(X=X_val, y=y_val, scaler=scaler)                    

        # Package request
        self._request = dict()
        self._request['alg'] = self._alg        
        self._request['data'] = {'X': X, 'y':y, 'X_val':X_val, 'y_val':y_val} 
        self._request['hyper'] = {'learning_rate': learning_rate, 'theta': theta,
                                  'learning_rate_sched': learning_rate_sched,
                                  'time_decay': time_decay,
                                  'step_decay': step_decay,
                                  'step_epochs': step_epochs,
                                  'exp_decay': exp_decay,
                                  'maxiter': maxiter, 
                                  'precision': precision,
                                  'batch_size': batch_size,
                                  'i_s': i_s,
                                  'stop_metric': stop_metric,
                                  'cross_validated': cross_validated}

        self._validate()                                  

        # Run search and obtain result        
        gd = MBGDFit()
        start = datetime.datetime.now()
        gd.fit(self._request)
        end = datetime.datetime.now()

        # Extract detail information
        epochs = pd.DataFrame(gd.get_epochs(), columns=['epochs'])        
        iterations = pd.DataFrame(gd.get_iterations(), columns=['iterations'])
        alphas = pd.DataFrame(gd.get_learning_rates(), columns=['learning_rates'])
        thetas = self._todf(gd.get_thetas(), stub='theta_')        
        J = pd.DataFrame(gd.get_costs(), columns=['cost'])   
        self._detail = pd.concat([epochs, iterations, alphas, thetas, J], axis=1)        
        if cross_validated:            
            mse = pd.DataFrame(gd.get_mse(), columns=['mse'])               
            self._detail = pd.concat([self._detail, mse], axis=1)
        self._detail['alg'] = self._alg
        self._detail['learning_rate_sched'] = learning_rate_sched
        self._detail['learning_rate_sched_label'] = self._get_label(learning_rate_sched)
        self._detail['learning_rate'] = learning_rate
        self._detail['time_decay'] = time_decay
        self._detail['step_decay'] = step_decay
        self._detail['step_epochs'] = step_epochs
        self._detail['exp_decay'] = exp_decay
        self._detail['precision'] = precision    
        self._detail['i_s'] = self._request['hyper']['i_s']
        self._detail['stop_metric'] = stop_metric
        self._detail['stop_metric_label'] = self._get_label(stop_metric)
        self._detail['batch_size'] = batch_size
        
        # Package summary results
        self._summary = pd.DataFrame({'alg': gd.get_alg(),
                                    'learning_rate_sched': learning_rate_sched,
                                    'learning_rate_sched_label': self._get_label(learning_rate_sched),
                                    'learning_rate': learning_rate,
                                    'time_decay': time_decay,
                                    'step_decay': step_decay,
                                    'step_epochs': step_epochs,
                                    'exp_decay': exp_decay,                                    
                                    'precision': precision,
                                    'batch_size': batch_size,
                                    'maxiter': maxiter,
                                    'i_s': i_s,
                                    'stop_metric': stop_metric,
                                    'stop_metric_label': self._get_label(stop_metric),
                                    'start':start,
                                    'end':end,
                                    'duration':end-start,
                                    'epochs': epochs.iloc[-1].item(),
                                    'iterations': iterations.iloc[-1].item(),
                                    'initial_costs': J.iloc[0].item(),
                                    'final_costs': J.iloc[-1].item()},
                                    index=[0])
        if cross_validated:                                    
            self._summary['initial_mse'] = mse.iloc[0].item()
            self._summary['final_mse'] = mse.iloc[-1].item()
        else:
            self._summary['initial_mse'] = None
            self._summary['final_mse'] = None  

