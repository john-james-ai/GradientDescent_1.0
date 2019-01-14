
# %%
# =========================================================================== #
#                             GRADIENT VISUAL                                 #
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
    def summary(self):
        if self._summary is None:
            raise Exception("No summary to report")
        else:
            return(self._summary)

    def detail(self):
        if self._detail is None:
            raise Exception("No detail to report")
        else:
            return(self._detail)        

    def _runsearch(self, gd):
        experiment = 1
        self._summary = pd.DataFrame()
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
                            summary = gd.summary()
                            summary['experiment'] = experiment
                            self._detail = pd.concat([self._detail, detail], axis=0)    
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
                                summary = gd.summary()
                                summary['experiment'] = experiment
                                self._detail = pd.concat([self._detail, detail], axis=0)    
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
                                    summary = gd.summary()
                                    summary['experiment'] = experiment
                                    self._detail = pd.concat([self._detail, detail], axis=0)    
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
                                summary = gd.summary()
                                summary['experiment'] = experiment
                                self._detail = pd.concat([self._detail, detail], axis=0)    
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

    def _pearson(self, x, y):
        r, p = pearsonr(x,y)
        return(r, p)

    def _spearman(self, x, y):
        r, p = spearmanr(x,y)
        return(r, p)

    def _get_params(self):
        x = ['learning_rate', 'learning_rate_sched', 'stop_metric', 
             'i_s', 'maxiter', 'time_decay', 'step_decay', 'step_epochs',
             'exp_decay', 'precision']
        return(x)

    def _get_tests(self,x):
        tests = {'learning_rate': 'p',
                'learning_rate_sched': 's',
                'stop_metric': 's',
                'i_s': 's',
                'batch_size': 's',
                'time_decay': 'p',
                'step_decay': 'p',
                'step_epochs': 's',
                'exp_decay': 'p',
                'precision': 's'}
        return(tests.get(x,'s'))

    def associations(self, data, x=None, y='final_costs', directory=None, filename=None):
        if x is None:
            x = self._get_params()

        labels = [ self._get_label(p) for p in x]     
        tests = [ self._get_tests(p) for p in x]
        corr = []
        pval = []
        
        for param, test in zip(x, tests):

            df = data[data[param].notnull()]

            if test == 'p':
                r, p = self._pearson(df[param], df[y])
            else:
                r, p = self._spearman(df[param], df[y])
            corr.append(np.abs(r))
            pval.append(p)

        scores = pd.DataFrame({'Parameter': labels, 'Correlation': corr, 'p-Value': pval})
        scores['Strength'] = np.where(scores["Correlation"] < .1, 'Extremely Weak Correlation',
                                         np.where(scores["Correlation"] < .30, 'Small Correlation',
                                                  np.where(scores["Correlation"] < .5, 'Moderate Correlation',
                                                           'Strong Correlation')))
        scores = scores.sort_values(by=['Correlation'], ascending=False)
        scores = scores.dropna()

        if directory:
            if filename is None:
                filename = self._alg + ' Parameter Correlations.csv'
            save_csv(scores, directory, filename)    

        return(scores)    

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

    def scatterplot(self, ax,  data, x, y, z=None, title=None):

        # Initialize figure and settings        
        sns.set(style="whitegrid", font_scale=1)        

        # Plot time by learning rate 
        ax = sns.scatterplot(x=x, y=y, hue=z, data=data, ax=ax, legend='full')
        ax.set_facecolor('w')
        ax.tick_params(colors='k')
        ax.xaxis.label.set_color('k')
        ax.yaxis.label.set_color('k')
        ax.set_xlabel(self._get_label(x))
        ax.set_ylabel(self._get_label(y))
        ax.set_title(title, color='k')
        return(ax)              

    def barplot(self, ax,  data, x, y, z=None, title=None):

        # Initialize figure and settings        
        sns.set(style="whitegrid", font_scale=1)

        # Plot time by learning rate 
        ax = sns.barplot(x=x, y=y, hue=z, data=data, ax=ax)
        ax.set_facecolor('w')
        ax.tick_params(colors='k')
        ax.xaxis.label.set_color('k')
        ax.yaxis.label.set_color('k')
        ax.set_xlabel(self._get_label(x))
        ax.set_ylabel(self._get_label(y))
        ax.set_title(title, color='k')
        return(ax)             

    def boxplot(self, ax,  data, x, y, z=None, title=None):

        # Initialize figure and settings        
        sns.set(style="whitegrid", font_scale=1)  

        # Plot time by learning rate 
        ax = sns.boxplot(x=x, y=y, hue=z, data=data, ax=ax)
        ax.set_facecolor('w')
        ax.tick_params(colors='k')
        ax.xaxis.label.set_color('k')
        ax.yaxis.label.set_color('k')
        ax.set_xlabel(self._get_label(x))
        ax.set_ylabel(self._get_label(y))
        ax.set_title(title, color='k')
        return(ax) 

    def lineplot(self, ax, data, x, y, z=None, title=None):

        # Initialize figure and settings        
        sns.set(style="whitegrid", font_scale=1)

        # Plot time by learning rate 
        ax = sns.lineplot(x=x, y=y, hue=z, data=data, legend='full', ax=ax)
        ax.set_facecolor('w')
        ax.tick_params(colors='k')
        ax.xaxis.label.set_color('k')
        ax.yaxis.label.set_color('k')
        ax.set_xlabel(self._get_label(x))
        ax.set_ylabel(self._get_label(y))
        ax.set_title(title, color='k')
        return(ax) 

    def figure(self, data, x, y, z=None, groupby=None, func=None, cols=2,
               directory=None, filename=None, show=False, height=1, width=1):

        sns.set(style="whitegrid", font_scale=1)                
        if groupby:
            plots = data.groupby(groupby)
            # Set Grid Dimensions
            cols = cols
            rows = math.ceil(plots.ngroups/cols)
            odd = True if plots.ngroups % cols != 0 else False

            # Obtain and initialize matplotlib figure
            fig_width = math.floor(12*width)
            fig_height = math.floor(4*rows*height)
            fig = plt.figure(figsize=(fig_width, fig_height))       
         
            # Extract group title
            group_title = self._get_label(groupby)

            # Render rows
            i = 0
            for plot_name, plot_data in plots:
                if odd and i == plots.ngroups-1:
                    ax = plt.subplot2grid((rows,cols), (int(i/cols),0), colspan=cols)
                else:
                    ax = plt.subplot2grid((rows,cols), (int(i/cols),i%cols))
                ax = func(ax=ax, data=plot_data, x=x, y=y, z=z, 
                          title = group_title + ' = ' + str(self._get_label(plot_name)))
                i += 1
            
            # Format Suptitle
            suptitle = self._alg + '\n' + self._get_label(y) + '\n' + ' By ' + self._get_label(x)
            if z:
                suptitle = suptitle + ' and ' + self._get_label(z) 
            suptitle = suptitle + '\nGroup by: ' + group_title                
            fig.suptitle(suptitle)
          
            # Finalize, show and save
            fig.tight_layout(rect=[0,0,1,0.8])
            if show:
                plt.show()
            if directory is not None:
                if filename is None:
                    filename = suptitle.replace('\n', '')
                    filename = filename.replace('  ', ' ')
                    filename = filename.replace(':', '') + '.png'
                save_fig(fig, directory, filename)
            plt.close(fig)
        else:
            # Designate plot title
            title = self._alg + '\n' + self._get_label(y) + ' By ' + self._get_label(x)
            if z:
                title = title + ' and ' + self._get_label(z)             
            # Establish plot dimensions and initiate matplotlib objects
            fig_width = math.floor(12*width)
            fig_height = math.floor(4*height)                
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))    
            # Render plot    
            ax = func(ax=ax, data=data, x=x, y=y, z=z, 
                        title=title)            

            # Finalize and save
            fig.tight_layout()
            if show:
                plt.show()
            if directory is not None:
                if filename is None:
                    filename = title.replace('\n', '')
                    filename = filename.replace('  ', ' ')
                    filename = filename.replace(':', '') + '.png'
                save_fig(fig, directory, filename)
            plt.close(fig)      
         
        return(fig)                
            

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
