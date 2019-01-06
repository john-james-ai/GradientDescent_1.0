
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
import seaborn as sns
from textwrap import wrap

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

    def _save_params(self, X, y, X_val, y_val, theta, alpha, precision, 
               improvement, maxiter, scaler):
        self._params = {'theta':theta, 'alpha':alpha, 'precision':precision,
                        'improvement': improvement,
                        'maxiter':maxiter,
                        'scaler':scaler}
    def summary(self):
        if self._summary is None:
            raise Exception("No summary to report")
        else:
            return(self._summary)

    def get_detail(self):
        if self._detail is None:
            raise Exception("No detail to report")
        else:
            return(self._detail)        
        
    def gridsearch(self, X, y, X_val, y_val, theta, alpha, precision, 
                   improvement=5, maxiter=0, scaler='minmax'):

        self._save_params(X, y, X_val, y_val, theta, alpha, precision, 
               improvement, maxiter, scaler)

        bgd = BGD()
        experiment = 1
        self._summary = pd.DataFrame()
        self._detail = pd.DataFrame()
        for n in improvement:
            for a in alpha:
                for p in precision:
                    bgd.fit(X=X, y=y, theta=theta, X_val=X_val, y_val=y_val, 
                            alpha=a, maxiter=maxiter, precision=p, 
                            improvement=n, scaler=scaler)
                    detail = bgd.get_detail()
                    summary = bgd.summary()
                    summary['experiment'] = experiment
                    self._detail = pd.concat([self._detail, detail], axis=0)    
                    self._summary = pd.concat([self._summary, summary], axis=0)    
                    experiment += 1               

    def _get_label(self, x):
        labels = {'alpha': 'Learning Rate',
                  'precision': 'Precision',
                  'theta': "Theta",
                  'duration': 'Computation Time (ms)',
                  'iterations': 'Iterations',
                  'cost': 'Training Set Costs',
                  'cost_val': 'Validation Set Costs',
                  'improvement': 'No. Iterations No Improvement ',
                  'batch_size': 'Batch Size',
                  'final_costs': 'Training Set Costs',
                  'final_costs_val': 'Validation Set Costs'}
        return(labels[x])

    def report(self,  n=None, sort='v', directory=None, filename=None):
        if self._detail is None:
            raise Exception('Nothing to report')
        else:
            vars = ['experiment', 'alg', 'alpha', 'precision', 
                    'improvement', 'maxiter', 
                    'epochs', 'iterations','duration',
                    'final_costs', 'final_costs_val']
            df = self._summary
            df = df[vars]
            if sort == 't':
                df = df.sort_values(by=['final_costs', 'duration'])
            else:
                df = df.sort_values(by=['final_costs_val', 'duration'])
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
        ax = sns.boxplot(x=x, y=y, hue=z, data=data, ax=ax, legend='full')
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

    def figure(self, data, x, y, z=None, groupby=None, func=None, 
               directory=None, show=None):

        sns.set(style="whitegrid", font_scale=1)                
        if groupby:
            plots = data.groupby(groupby)
            # Set Grid Dimensions
            cols = 2  
            rows = math.ceil(plots.ngroups/cols)
            odd = True if plots.ngroups % cols != 0 else False

            # Obtain and initialize matplotlib figure
            fig = plt.figure(figsize=(12,4*rows))       
         
            # Extract group title
            group_title = ''
            for g in groupby:
                group_title = group_title + self._get_label(g)

            # Render rows
            i = 0
            for plot_name, plot_data in plots:
                if odd and i == plots.ngroups-1:
                    ax = plt.subplot2grid((rows,cols), (int(i/cols),0), colspan=cols)
                else:
                    ax = plt.subplot2grid((rows,cols), (int(i/cols),i%cols))
                ax = func(ax=ax, data=plot_data, x=x, y=y, z=z, 
                          title=group_title + ' : ' + str(plot_name))
                i += 1
            
            # Finalize and save plot
            suptitle = self._alg + '\n' + self._get_label(y) + '\n' + ' By ' + self._get_label(x)
            if z:
                suptitle = suptitle + ' and ' + self._get_label(z) 
            fig.suptitle(suptitle)
            fig.tight_layout()
            if show:
                plt.show()
            if directory is not None:
                filename = suptitle.replace('\n', '')
                filename = filename.replace('  ', ' ')
                filename = filename.replace(':', '') + '.png'
                save_fig(fig, directory, filename)
            plt.close(fig)
        else:
            # Obtain and initialize matplotlib figure
            title = self._alg + '\n' + self._get_label(y) + '\n' + ' By ' + self._get_label(x)
            if z:
                title = title + ' and ' + self._get_label(z) 
            fig, ax = plt.subplots(figsize=(12,4))        
            ax = func(ax=ax, data=data, x=x, y=y, z=z, 
                        title=title)
            fig.tight_layout(rect=[0,0,1,.85])
            if show:
                plt.show()
            if directory is not None:
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

    def report(self,  n=None, sort='v', directory=None, filename=None):
        if self._detail is None:
            raise Exception('Nothing to report')
        else:
            vars = ['experiment', 'alg', 'alpha', 'precision', 
                    'improvement',
                    'maxiter', 'epochs', 'iterations','duration',
                    'final_costs', 'final_costs_val']
            df = self._summary
            df = df[vars]
            if sort == 't':
                df = df.sort_values(by=['final_costs'])
            else:
                df = df.sort_values(by=['final_costs_val'])
            if directory:
                save_csv(df, directory, filename)                
            if n:
                df = df.iloc[:n]            
            return(df)

    def gridsearch(self, X, y, X_val, y_val, theta, alpha, precision, 
                   maxiter=0, improvement=5, scaler='minmax'):
        sgd = SGD()
        experiment = 1
        self._summary = pd.DataFrame()
        self._detail = pd.DataFrame()
        
        for n in improvement:
            for a in alpha:
                for p in precision:
                    sgd.fit(X=X, y=y, theta=theta, X_val=X_val, y_val=y_val, 
                            alpha=a, maxiter=maxiter, improvement=n,
                            precision=p, scaler=scaler)
                    detail = sgd.get_detail()
                    summary = sgd.summary()
                    summary['experiment'] = experiment
                    self._detail = pd.concat([self._detail, detail], axis=0)    
                    self._summary = pd.concat([self._summary, summary], axis=0)    
                    experiment += 1 

 
# --------------------------------------------------------------------------- #
#                              MBGD Lab Class                                 #   
# --------------------------------------------------------------------------- #
class MBGDLab(GradientLab):  

    def __init__(self):      
        self._alg = 'Minibatch Gradient Descent'
        self._summary = None
        self._detail = None

    def gridsearch(self, X, y, X_val, y_val, theta, alpha, precision, 
                  improvement, batch_size, maxiter=0, 
                  scaler='minmax'):
        mbgd = MBGD()
        self._summary = pd.DataFrame()
        self._detail = pd.DataFrame()

        for bs in batch_size:
            for n in improvement:
                for a in alpha:
                    for p in precision:
                        mbgd.fit(X=X, y=y, theta=theta, X_val=X_val, y_val=y_val, 
                                alpha=a, maxiter=maxiter, precision=p, improvement=n,
                                batch_size=bs, scaler=scaler)
                        detail = mbgd.get_detail()
                        summary = mbgd.summary()
                        self._detail = pd.concat([self._detail, detail], axis=0)    
                        self._summary = pd.concat([self._summary, summary], axis=0)    
