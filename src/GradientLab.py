
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

from GradientDescent import BGD, SGD
from utils import save_fig

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

    def report(self, n=None, sort='t'):
        if self._detail is None:
            raise Exception('Nothing to report')
        else:
            vars = ['alg', 'alpha', 'precision', 'maxiter',
                    'epochs', 'iterations','duration',
                    'stop_parameter', 'stop_metric', 
                    'final_costs', 'final_costs_val']
            df = self._summary
            df = df[vars]
            if sort == 't':
                df = df.sort_values(by=['final_costs'])
            else:
                df = df.sort_values(by=['final_costs_val'])
            if n:
                df = df.iloc[:n]

            return(df)
        
    def fit(self, X, y, X_val, y_val, theta, alpha, precision, 
               stop_parameter, stop_metric, maxiter=0, scaler='minmax'):

        bgd = BGD()
        self._summary = pd.DataFrame()
        self._detail = pd.DataFrame()
        for measure in stop_parameter:
            for metric in stop_metric:
                for a in alpha:
                    for p in precision:
                        bgd.fit(X=X, y=y, theta=theta, X_val=X_val, y_val=y_val, 
                                alpha=a, maxiter=maxiter, precision=p, stop_parameter=measure, 
                                stop_metric=metric, scaler=scaler)
                        detail = bgd.get_detail()
                        summary = bgd.summary()
                        self._detail = pd.concat([self._detail, detail], axis=0)    
                        self._summary = pd.concat([self._summary, summary], axis=0)    


    def _plot_costs(self, search_name, search,  directory=None, show=True):
        # Group data and obtain keys
        search_groups = search.groupby('stop_condition')   

        # Set Grid Dimensions
        cols = 2  
        rows = math.ceil(search_groups.ngroups)
        
        # Obtain and initialize matplotlib figure
        fig = plt.figure(figsize=(12,4*rows))        
        sns.set(style="whitegrid", font_scale=1)

        # Render plots
        i = 0
        for condition, search in search_groups:
            ax0 = plt.subplot2grid((rows,cols), (i,0))            
            ax0 = sns.barplot(x='alpha', y='final_costs', hue='precision', data=search)
            ax0.set_facecolor('w')
            ax0.tick_params(colors='k')
            ax0.xaxis.label.set_color('k')
            ax0.yaxis.label.set_color('k')
            ax0.set_xlabel('Learning Rate')
            ax0.set_ylabel('Training Set Costs')
            title = condition
            ax0.set_title(title, color='k')

            ax1 = plt.subplot2grid((rows,cols), (i,1))
            ax1 = sns.barplot(x='alpha', y='final_costs_val', hue='precision', data=search)
            ax1.set_facecolor('w')
            ax1.tick_params(colors='k')
            ax1.xaxis.label.set_color('k')
            ax1.yaxis.label.set_color('k')
            ax1.set_xlabel('Learning Rate')
            ax1.set_ylabel('Validation Set Costs')
            title = condition
            ax1.set_title(title, color='k')
            i += 1

        # Finalize plot and save
        if search_name:
            suptitle = self._alg + '\n' + 'Cost Analysis' + '\n' + ' '.join(search_name)
        else:
            suptitle = self._alg + '\n' + 'Cost Analysis' 
        fig.suptitle(suptitle)
        fig.tight_layout(rect=[0,0,1,.9])
        if show:
            plt.show()
        if directory is not None:
            filename = suptitle.replace('\n', '')
            filename = filename.replace(':', '') + '.png'
            save_fig(fig, directory, filename)
        plt.close(fig)
        return(fig)

    def _plot_times(self, search_name, search,  directory=None, show=True):
        # Group data and obtain keys
        search_groups = search.groupby('stop_condition')   

        # Set Grid Dimensions
        cols = 2  
        rows = math.ceil(search_groups.ngroups/cols)
        odd = True if search_groups.ngroups % cols != 0 else False

        # Obtain and initialize matplotlib figure
        fig = plt.figure(figsize=(12,4*rows))        
        sns.set(style="whitegrid", font_scale=1)
        
        # Render plots
        i = 0
        for condition, search in search_groups:
            if odd and i == search_groups.ngroups-1:
                ax = plt.subplot2grid((rows,cols), (int(i/cols),0), colspan=cols)
            else:
                ax = plt.subplot2grid((rows,cols), (int(i/cols),i%cols))
            ax = sns.barplot(x='alpha', y='duration', hue='precision', data=search)
            ax.set_facecolor('w')
            ax.tick_params(colors='k')
            ax.xaxis.label.set_color('k')
            ax.yaxis.label.set_color('k')
            ax.set_xlabel('Learning Rate')
            ax.set_ylabel('Duration (ms)')
            title = condition
            ax.set_title(title, color='k')
            i += 1

        # Finalize plot and save
        if search_name:
            suptitle = self._alg + '\n' + 'Computation Time' + '\n' + ' '.join(search_name)
        else:
            suptitle = self._alg + '\n' + 'Computation Time' 
        fig.suptitle(suptitle)
        fig.tight_layout(rect=[0,0,1,.9])
        if show:
            plt.show()
        if directory is not None:
            filename = suptitle.replace('\n', '')
            filename = filename.replace(':', '') + '.png'
            save_fig(fig, directory, filename)
        plt.close(fig)
        return(fig)

    def _plot_curves(self, search_name, search,  directory=None, show=True):

        # Group data and obtain keys
        search_groups = search.groupby('stop')   

        # Set Grid Dimensions
        cols = 2  
        rows = math.ceil(search_groups.ngroups/cols)
        odd = True if search_groups.ngroups % cols != 0 else False

        # Obtain and initialize matplotlib figure
        fig = plt.figure(figsize=(12,4*rows))        
        sns.set(style="whitegrid", font_scale=1)
        if search_name:
            suptitle = self._alg + '\n' + 'Learning Curves' + '\n' + ' '.join(search_name)
        else:
            suptitle = self._alg + '\n' + 'Learning Curves' 
        fig.suptitle(suptitle, y=1.1)

        # Render plots
        i = 0
        for condition, search in search_groups:
            if odd and i == search_groups.ngroups-1:
                ax = plt.subplot2grid((rows,cols), (int(i/cols),0), colspan=cols)
            else:
                ax = plt.subplot2grid((rows,cols), (int(i/cols),i%cols))
            ax = sns.lineplot(x='iterations', y='cost', hue='alpha', data=search, legend='full')
            ax.set_facecolor('w')
            ax.tick_params(colors='k')
            ax.xaxis.label.set_color('k')
            ax.yaxis.label.set_color('k')
            ax.set_xlabel('Iterations')
            ax.set_ylabel('Cost')
            title = condition
            ax.set_title(title, color='k')
            i += 1

        # Finalize plot and save
        fig.tight_layout()
        if show:
            plt.show()
        if directory is not None:
            filename = suptitle.replace('\n', '')
            filename = filename.replace(':', '') + '.png'
            save_fig(fig, directory, filename)
        plt.close(fig)
        return(fig)

    def plot_times(self, directory=None, show=True):
        search_name = None
        self._plot_times(search_name, self._summary,  directory, show)

    def plot_curves(self, directory=None, show=True):
        detail = self._detail.groupby(['stop_parameter'])
        for search_name, search in detail:
            self._plot_curves(search_name, search,  directory, show)

    def plot_costs(self, directory=None, show=True):
        summary = self._summary.groupby(['stop_parameter'])
        for search_name, search in summary:
            self._plot_costs(search_name, search,  directory, show)



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

    def fit(self, X, y, X_val, y_val, theta, alpha, precision, 
            stop_parameter, stop_metric, check_point, maxiter=0, 
            scaler='minmax'):
        sgd = SGD()
        self._summary = pd.DataFrame()
        self._detail = pd.DataFrame()

        for cp in check_point:
            for measure in stop_parameter:
                for metric in stop_metric:
                    for a in alpha:
                        for p in precision:
                            sgd.fit(X=X, y=y, theta=theta, X_val=X_val, y_val=y_val, 
                                    alpha=a, maxiter=maxiter, precision=p, stop_parameter=measure, 
                                    stop_metric=metric, check_point=cp, scaler=scaler)
                            detail = sgd.get_detail()
                            summary = sgd.summary()
                            self._detail = pd.concat([self._detail, detail], axis=0)    
                            self._summary = pd.concat([self._summary, summary], axis=0)    

    def plot_times(self, directory=None, show=True):
        summary = self._summary.groupby(['check_point'])
        for search_name, search in summary:
            self._plot_times(search_name, search, directory, show)
        

    def plot_curves(self, directory=None, show=True):
        detail = self._detail.groupby(['stop_parameter', 'check_point'])
        for search_name, search in detail:
            self._plot_curves(search_name, search, directory, show)

    def plot_costs(self, directory=None, show=True):
        summary = self._summary.groupby(['stop_parameter', 'check_point'])
        for search_name, search in summary:
            self._plot_costs(search_name, search, directory, show)