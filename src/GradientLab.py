
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

from GradientDescent import BGD

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
        self._searches = None

    def report(self):
        if self._searches is None:
            raise Exception('Nothing to report')
        else:
            return(self._searches)
        
    def fit(self, X, y, X_val, y_val, theta, alpha, precision, 
               stop_measure, stop_metric, maxiter=0, scaler='minmax'):

        bgd = BGD()
        self._searches = pd.DataFrame()
        for measure in stop_measure:
            for metric in stop_metric:
                for a in alpha:
                    for p in precision:
                        bgd.fit(X=X, y=y, theta=theta, X_val=X_val, y_val=y_val, 
                                alpha=a, maxiter=maxiter, precision=p, stop_measure=measure, 
                                stop_metric=metric, scaler=scaler)
                        search = bgd.get_detail()
                        self._searches = pd.concat([self._searches, search], axis=0)    

    def _plot_costs(self, searches, directory=None, show=True):
        # Obtain figure and gridspec objects
        fig = plt.figure(figsize=(12,4))
        gs = fig.add_gridspec(1,2)
        sns.set(style="whitegrid", font_scale=1)

        # Parse Data
        groups = searches.groupby(['alpha', 'precision'])
        df = pd.DataFrame()
        for group, data in groups:
            df_plot = pd.DataFrame({'alpha': data['alpha'].iloc[0],
                                    'precision': data['precision'].iloc[0],
                                    'cost': data['cost'].iloc[-1],
                                    'cost_val': data['cost_val'].iloc[-1]}, index=[0])
            df = pd.concat([df,df_plot], axis=0)

        # Configure Training Cost Plot
        ax0 = fig.add_subplot(gs[0,0])
        ax0 = sns.barplot(x='alpha', y='cost', hue='precision', data=df)
        # Face, text, and label colors
        ax0.set_facecolor('w')
        ax0.tick_params(colors='k')
        ax0.xaxis.label.set_color('k')
        ax0.yaxis.label.set_color('k') 
        # Axes labels
        ax0.set_xlabel('Alpha')
        ax0.set_ylabel('Cost')            
        ax0.set_title("Training Set Costs")

        # Configure Validation Cost Plot
        ax1 = fig.add_subplot(gs[0,1])
        ax1 = sns.barplot(x='alpha', y='cost_val', hue='precision', data=df)
        # Face, text, and label colors
        ax1.set_facecolor('w')
        ax1.tick_params(colors='k')
        ax1.xaxis.label.set_color('k')
        ax1.yaxis.label.set_color('k') 
        # Axes labels
        ax1.set_xlabel('Alpha')
        ax1.set_ylabel('Cost')            
        ax1.set_title("Validation Set Costs")

        # Finalize plot and save
        fig.subplots_adjust(top=0.8)
        suptitle = searches['alg'].iloc[0] + '\n' + \
                   'Final Costs' + '\n' + \
                   searches['stop_condition'].iloc[0]
        fig.suptitle(suptitle, y=1.1, fontsize=12)
        fig.tight_layout()
        if show:
            plt.show()
        if directory is not None:
            filename = 'Final Costs ' + searches['stop_condition'].iloc[0] + '.png'
            if os.path.exists(directory):
                path = os.path.join(os.path.abspath(directory), filename)
                fig.savefig(path, facecolor='w')
            else:
                os.makedirs(directory)
                path = os.path.join(os.path.abspath(directory),filename)
                fig.savefig(path, facecolor='w')
        plt.close(fig)
        return(fig)


    def _plot_time(self, searches, directory=None, show=True):
        # Obtain figure object
        fig, ax = plt.subplots(figsize=(12,4))
        sns.set(style="whitegrid", font_scale=1)

        # Parse Data
        groups = searches.groupby(['alpha', 'precision'])
        df = pd.DataFrame()
        for group, data in groups:
            df_plot = pd.DataFrame({'alpha': data['alpha'].iloc[0],
                                    'precision': data['precision'].iloc[0],
                                    'duration': data['time'].iloc[-1] - data['time'].iloc[0]}, 
                                    index=[0])
            df = pd.concat([df,df_plot], axis=0)
        
        # Configure time plot
        ax = sns.barplot(x='alpha', y='duration', hue='precision', data=df)        
        # Face, text, and label colors
        ax.set_facecolor('w')
        ax.tick_params(colors='k')
        ax.xaxis.label.set_color('k')
        ax.yaxis.label.set_color('k') 
        # Axes labels
        ax.set_xlabel('Alpha')
        ax.set_ylabel('Elapsed Time (ms)')            
        ax.set_title("Training Time")

        # Finalize figure and save
        fig.tight_layout()
        if show:
            plt.show()
        if directory is not None:
            filename = 'Elapsed Time ' + searches['stop_condition'].iloc[0] + '.png'
            if os.path.exists(directory):
                path = os.path.join(os.path.abspath(directory), filename)
                fig.savefig(path, facecolor='w')
            else:
                os.makedirs(directory)
                path = os.path.join(os.path.abspath(directory),filename)
                fig.savefig(path, facecolor='w')
        plt.close(fig)
        return(fig)

    def plot_costs(self, directory=None, show=True):
        searches = self._searches.groupby(['stop_condition'])
        for condition, data in searches:
            self._plot_costs(data, directory, show)

    def plot_times(self, directory=None, show=True):
        searches = self._searches.groupby(['stop_condition'])        
        for condition, data in searches:
            self._plot_time(data, directory, show)   

    def plot_curves(self, directory=None, show=True):
        # Render plot
        sns.set(style="whitegrid", font_scale=1)
        g = sns.FacetGrid(self._searches, col='stop', 
                          hue='alpha', col_wrap=2, 
                          sharex=False, sharey=False, legend_out=True,
                          height=2, aspect=2)
        g.map(sns.lineplot, 'iterations', 'cost', 'alpha')
        g.add_legend()
        suptitle = self._searches['alg'].iloc[0] + '\n' + \
                    'Learning Curves'        
        g.fig.suptitle(suptitle, y=1.1)
        g.fig.tight_layout()
        
        # Show and save plot
        if show:
            plt.show()
        if directory is not None:
            filename = 'Learning Curves.png'
            if os.path.exists(directory):
                path = os.path.join(os.path.abspath(directory), filename)
                g.savefig(path, facecolor='w')
            else:
                os.makedirs(directory)
                path = os.path.join(os.path.abspath(directory),filename)
                g.savefig(path, facecolor='w')




# --------------------------------------------------------------------------- #
#                              BGD Lab Class                                  #   
# --------------------------------------------------------------------------- #
class BGDLab(GradientLab):  

    def __init__(self):        
        self._searches = None  
        self._alg = 'Batch Gradient Descent'
        
