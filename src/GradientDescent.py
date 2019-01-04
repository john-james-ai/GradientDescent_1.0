
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

    def get_detail(self):
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

    
    def fit(self, X, y, theta, X_val=None, y_val=None, alpha=0.01, 
            miniter=0, maxiter=0, precision=0.001, stop_parameter='t', 
            stop_metric='a', scaler='minmax', max_cost=100):

        # Set cross-validated flag if validation set included 
        cross_validated = all(v is not None for v in [X_val, y_val])

        # Confirm cross-validated of validation set cost is chosen as stopping 
        # condition
        if stop_parameter == 'v' and not cross_validated:
            raise Exception('Validation set must be provided for this stopping criteria')

        # Prepare Data
        X, y = self._prep_data(X=X, y=y, scaler=scaler)
        if cross_validated:
            X_val, y_val = self._prep_data(X=X_val, y=y_val, scaler=scaler)                    

        # Package request
        self._request = dict()
        self._request['alg'] = self._alg        
        self._request['data'] = {'X': X, 'y':y, 'X_val':X_val, 'y_val':y_val} 
        self._request['hyper'] = {'alpha': alpha, 'theta': theta,
                                  'miniter': miniter,
                                  'maxiter': maxiter, 
                                  'precision': precision,
                                  'stop_parameter': stop_parameter,
                                  'stop_metric': stop_metric,
                                  'max_cost': max_cost,
                                  'cross_validated': cross_validated}

        # Run search and obtain result        
        gd = BGDFit()
        start = datetime.datetime.now()
        gd.fit(self._request)
        end = datetime.datetime.now()

        # Format hyperparameters
        if self._request['hyper']['stop_parameter'] == 't':
            stop_parameter = 'Training Set Costs '
        if self._request['hyper']['stop_parameter'] == 'g':
            stop_parameter = 'Gradient Norm '
        if self._request['hyper']['stop_parameter'] == 'v':
            stop_parameter = 'Validation Set Costs '            
        if self._request['hyper']['stop_metric'] == 'a':
            stop_metric = 'Absolute Change'         
        if self._request['hyper']['stop_metric'] == 'r':
            stop_metric = 'Relative Change'         
        stop = stop_metric + " in " + stop_parameter + \
               ' less than ' + str(precision) 
        stop_condition = stop_metric + " in " + stop_parameter 

        # Extract detail information
        epochs = pd.DataFrame(gd.get_epochs(), columns=['epochs'])        
        iterations = pd.DataFrame(gd.get_iterations(), columns=['iterations'])
        thetas = self._todf(gd.get_thetas(), stub='theta_')        
        J = pd.DataFrame(gd.get_costs(dataset='t'), columns=['cost'])   
        self._detail = pd.concat([epochs, iterations, thetas, J], axis=1)        
        if cross_validated:            
            J_val = pd.DataFrame(gd.get_costs(dataset='v'), columns=['cost_val'])               
            self._detail = pd.concat([self._detail, J_val], axis=1)
        self._detail['alg'] = self._alg
        self._detail['alpha'] = self._request['hyper']['alpha']
        self._detail['stop'] = stop
        self._detail['stop_parameter'] = 'Stop Parameter: ' + stop_parameter
        self._detail['stop_condition'] = 'Stop Condition: ' + stop_condition
        self._detail['stop_metric'] = stop_metric
        self._detail['precision'] = self._request['hyper']['precision']
        
        # Package summary results
        self._summary = pd.DataFrame({'alg': gd.get_alg(),
                                    'alpha': self._request['hyper']['alpha'],
                                    'precision': self._request['hyper']['precision'],
                                    'miniter': self._request['hyper']['miniter'],
                                    'maxiter': self._request['hyper']['maxiter'],
                                    'stop_parameter': stop_parameter,
                                    'stop_metric': stop_metric,
                                    'stop_condition': stop_condition,
                                    'stop': stop,
                                    'start':start,
                                    'end':end,
                                    'duration':(end-start).total_seconds(),
                                    'epochs': epochs.iloc[-1].item(),
                                    'iterations': iterations.iloc[-1].item(),
                                    'initial_costs': J.iloc[0].item(),
                                    'final_costs': J.iloc[-1].item()},
                                    index=[0])
        if cross_validated:                                    
            self._summary['initial_costs_val'] = J_val.iloc[0].item()
            self._summary['final_costs_val'] = J_val.iloc[-1].item()
        else:
            self._summary['initial_costs_val'] = None
            self._summary['final_costs_val'] = None
    
    def plot(self, directory, show=True):

        # Obtain matplotlib figure
        fig = plt.figure(figsize=(12,4))
        gs = fig.add_gridspec(1,2)
        sns.set(style="whitegrid", font_scale=1)
        
        if self._request['hyper']['cross_validated']:

            # -----------------------  Cost bar plot  ------------------------ #
            # Data
            x = ['Initial Cost', 'Final Costs', 'Initial Cost', 'Final Costs',]
            y = [self._summary['initial_costs'].values.tolist(),
                 self._summary['final_costs'].values.tolist(),
                 self._summary['initial_costs_val'].values.tolist(),
                 self._summary['final_costs_val'].values.tolist()]
            z = ['Training Set', 'Training Set', 'Validation Set', 'Validation Set']
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
            ax0.set_ylabel('Cost')            
            # Values above bars
            rects = ax0.patches
            for rect, label in zip(rects, y):
                height = rect.get_height()
                ax0.text(rect.get_x() + rect.get_width() / 2, height + .03, str(round(label,3)),
                        ha='center', va='bottom')                
            # Title                        
            title = 'Initial and Final Costs ' + '\n' + \
                    r'$\alpha$' + " = " + str(self._request['hyper']['alpha']) + " " + \
                    r'$\epsilon$' + " = " + str(round(self._request['hyper']['precision'],5))                         
            ax0.set_title(title, color='k', pad=15)
            
            # -----------------------  Cost line plot ------------------------ #
            df = pd.DataFrame()
            df_train = pd.DataFrame({'Iteration': self._detail['iterations'],
                                     'Costs': self._detail['cost'],
                                     'Dataset': 'Train'})
            df_val = pd.DataFrame({'Iteration': self._detail['iterations'],
                                     'Costs': self._detail['cost_val'],
                                     'Dataset': 'Validation'})
            df = pd.concat([df_train, df_val], axis=0)
            ax1 = fig.add_subplot(gs[0,1])
            ax1 = sns.lineplot(x='Iteration', y='Costs', hue='Dataset', data=df)
            ax1.set_facecolor('w')
            ax1.tick_params(colors='k')
            ax1.xaxis.label.set_color('k')
            ax1.yaxis.label.set_color('k')
            ax1.set_xlabel('Iterations')
            ax1.set_ylabel('Cost')
            title = 'Costs by Iteration ' + '\n' + \
            r'$\alpha$' + " = " + str(self._request['hyper']['alpha']) + \
            r'$\epsilon$' + " = " + str(round(self._request['hyper']['precision'],5)) 
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
            title = 'Initial and Final Costs ' + '\n' + \
                    r'$\alpha$' + " = " + str(self._request['hyper']['alpha']) + " " + \
                    r'$\epsilon$' + " = " + str(round(self._request['hyper']['precision'],5)) 
            rects = ax0.patches
            for rect, label in zip(rects, y):
                height = rect.get_height()
                ax0.text(rect.get_x() + rect.get_width() / 2, height + .03, str(round(label,3)),
                        ha='center', va='bottom')                
            ax0.set_title(title, color='k', pad=15)

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
            title = 'Costs by Iteration ' + '\n' + \
            r'$\alpha$' + " = " + str(self._request['hyper']['alpha']) + \
            r'$\epsilon$' + " = " + str(round(self._request['hyper']['precision'],5)) 
            ax1.set_title(title, color='k', pad=15)

        fig.suptitle(self._alg)            
        footnote = "* Stop Criteria: " + self._summary['stop'].iloc[0]
        plt.figtext(0.3, -0.001, footnote, fontsize=12, color='k')

        fig.tight_layout()
        if show:
            plt.show()
        if directory is not None:
            filename = self._alg + ' ' + self._summary['stop'].item() + '.png'
            filename = filename.replace(':', '')
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

    def fit(self, X, y, theta,  X_val=None, y_val=None,  check_point=.1, 
            alpha=0.01, miniter=0, maxiter=0, precision=0.001, stop_parameter='t', 
            stop_metric='a', scaler='minmax', max_cost=100):

        # Set cross-validated flag if validation set included 
        cross_validated = all(v is not None for v in [X_val, y_val])

        # Confirm cross-validated of validation set cost is chosen as stopping 
        # condition
        if stop_parameter == 'v' and not cross_validated:
            raise Exception('Validation set must be provided for this stopping criteria')

        # Prepare Data
        X, y = self._prep_data(X=X, y=y, scaler=scaler)
        if cross_validated:
            X_val, y_val = self._prep_data(X=X_val, y=y_val, scaler=scaler)                    

        # Package request
        self._request = dict()
        self._request['alg'] = self._alg        
        self._request['data'] = {'X': X, 'y':y, 'X_val':X_val, 'y_val':y_val} 
        self._request['hyper'] = {'alpha': alpha, 'theta': theta,
                                  'miniter': miniter,
                                  'maxiter': maxiter, 
                                  'precision': precision,
                                  'check_point': check_point,
                                  'stop_parameter': stop_parameter,
                                  'stop_metric': stop_metric,
                                  'max_cost': max_cost,
                                  'cross_validated': cross_validated}

        # Run search and obtain result        
        gd = SGDFit()
        start = datetime.datetime.now()
        gd.fit(self._request)
        end = datetime.datetime.now()

        # Format hyperparameters
        if self._request['hyper']['stop_parameter'] == 't':
            stop_parameter = 'Training Set Costs'
        if self._request['hyper']['stop_parameter'] == 'g':
            stop_parameter = 'Gradient Norm'
        if self._request['hyper']['stop_parameter'] == 'v':
            stop_parameter = 'Validation Set Costs'            
        if self._request['hyper']['stop_metric'] == 'a':
            stop_metric = 'Absolute Change'         
        if self._request['hyper']['stop_metric'] == 'r':
            stop_metric = 'Relative Change'         
        stop = stop_metric + " in " + stop_parameter + \
               ' less than ' + str(precision) 
        stop_condition = stop_metric + " in " + stop_parameter 

        # Extract detail information
        epochs = pd.DataFrame(gd.get_epochs(), columns=['epochs'])        
        iterations = pd.DataFrame(gd.get_iterations(), columns=['iterations'])
        thetas = self._todf(gd.get_thetas(), stub='theta_')        
        J = pd.DataFrame(gd.get_costs(dataset='t'), columns=['cost'])   
        self._detail = pd.concat([epochs, iterations, thetas, J], axis=1)        
        if cross_validated:            
            J_val = pd.DataFrame(gd.get_costs(dataset='v'), columns=['cost_val'])               
            self._detail = pd.concat([self._detail, J_val], axis=1)
        self._detail['alg'] = self._alg
        self._detail['alpha'] = alpha
        self._detail['stop'] = stop
        self._detail['stop_parameter'] = stop_parameter
        self._detail['stop_condition'] = stop_condition
        self._detail['stop_metric'] = stop_metric
        self._detail['check_point'] = check_point
        self._detail['precision'] = precision

        
        # Package summary results
        self._summary = pd.DataFrame({'alg': gd.get_alg(),
                                    'alpha': alpha,
                                    'precision': precision,
                                    'check_point': check_point,
                                    'miniter': miniter,
                                    'maxiter': maxiter,
                                    'stop_parameter': stop_parameter,
                                    'stop_metric': stop_metric,
                                    'stop_condition': stop_condition,
                                    'stop': stop,
                                    'start':start,
                                    'end':end,
                                    'duration':(end-start).total_seconds(),
                                    'epochs': epochs.iloc[-1].item(),
                                    'iterations': iterations.iloc[-1].item(),
                                    'initial_costs': J.iloc[0].item(),
                                    'final_costs': J.iloc[-1].item()},
                                    index=[0])
        if cross_validated:                                    
            self._summary['initial_costs_val'] = J_val.iloc[0].item()
            self._summary['final_costs_val'] = J_val.iloc[-1].item()
        else:
            self._summary['initial_costs_val'] = None
            self._summary['final_costs_val'] = None  

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
            alpha=0.01, maxiter=0, precision=0.001, stop_parameter='t', 
            stop_metric='a', scaler='minmax'):            

        # Set cross-validated flag if validation set included 
        cross_validated = all(v is not None for v in [X_val, y_val])

        # Confirm cross-validated of validation set cost is chosen as stopping 
        # condition
        if stop_parameter == 'v' and not cross_validated:
            raise Exception('Validation set must be provided for this stopping criteria')

        # Prepare Data
        X, y = self._prep_data(X=X, y=y, scaler=scaler)
        if cross_validated:
            X_val, y_val = self._prep_data(X=X_val, y=y_val, scaler=scaler)                    

        # Package request
        self._request = dict()
        self._request['alg'] = self._alg        
        self._request['data'] = {'X': X, 'y':y, 'X_val':X_val, 'y_val':y_val} 
        self._request['hyper'] = {'alpha': alpha, 'theta': theta,
                                  'maxiter': maxiter, 
                                  'precision': precision,
                                  'batch_size': batch_size,
                                  'stop_parameter': stop_parameter,
                                  'stop_metric': stop_metric,
                                  'cross_validated': cross_validated}

        # Run search and obtain result        
        gd = MBGDFit()
        start = datetime.datetime.now()
        gd.fit(self._request)
        end = datetime.datetime.now()

        # Format hyperparameters
        if self._request['hyper']['stop_parameter'] == 't':
            stop_parameter = 'Training Set Costs'
        if self._request['hyper']['stop_parameter'] == 'g':
            stop_parameter = 'Gradient Norm'
        if self._request['hyper']['stop_parameter'] == 'v':
            stop_parameter = 'Validation Set Costs'            
        if self._request['hyper']['stop_metric'] == 'a':
            stop_metric = 'Absolute Change'         
        if self._request['hyper']['stop_metric'] == 'r':
            stop_metric = 'Relative Change'         
        stop = stop_metric + " in " + stop_parameter + \
               ' less than ' + str(precision) 
        stop_condition = stop_metric + " in " + stop_parameter 
        batch_size = 'Batch Size ' + str(batch_size)

        # Extract detail information
        epochs = pd.DataFrame(gd.get_epochs(), columns=['epochs'])        
        iterations = pd.DataFrame(gd.get_iterations(), columns=['iterations'])
        thetas = self._todf(gd.get_thetas(), stub='theta_')        
        J = pd.DataFrame(gd.get_costs(dataset='t'), columns=['cost'])   
        self._detail = pd.concat([epochs, iterations, thetas, J], axis=1)        
        if cross_validated:            
            J_val = pd.DataFrame(gd.get_costs(dataset='v'), columns=['cost_val'])               
            self._detail = pd.concat([self._detail, J_val], axis=1)
        self._detail['alg'] = self._alg
        self._detail['alpha'] = alpha
        self._detail['stop'] = stop
        self._detail['stop_parameter'] = stop_parameter
        self._detail['stop_condition'] = stop_condition
        self._detail['batch_size'] = batch_size
        
        # Package summary results
        self._summary = pd.DataFrame({'alg': gd.get_alg(),
                                    'alpha': alpha,
                                    'precision': precision,
                                    'batch_size': batch_size,
                                    'maxiter': maxiter,
                                    'stop_parameter': stop_parameter,
                                    'stop_metric': stop_metric,
                                    'stop_condition': stop_condition,
                                    'stop': stop,
                                    'start':start,
                                    'end':end,
                                    'duration':end-start,
                                    'epochs': epochs.iloc[-1].item(),
                                    'iterations': iterations.iloc[-1].item(),
                                    'initial_costs': J.iloc[0].item(),
                                    'final_costs': J.iloc[-1].item()},
                                    index=[0])
        if cross_validated:                                    
            self._summary['initial_costs_val'] = J_val.iloc[0].item()
            self._summary['final_costs_val'] = J_val.iloc[-1].item()
        else:
            self._summary['initial_costs_val'] = None
            self._summary['final_costs_val'] = None  

