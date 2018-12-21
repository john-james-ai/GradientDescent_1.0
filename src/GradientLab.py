
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
        self._summary = None
        self._detail = None

    def report(self):
        if self._detail is None:
            raise Exception('Nothing to report')
        else:
            return(self._detail)
        
    def fit(self, X, y, X_val, y_val, theta, alpha, precision, 
               stop_measure, stop_metric, maxiter=0, scaler='minmax'):

        bgd = BGD()
        self._summary = pd.DataFrame()
        self._detail = pd.DataFrame()
        for measure in stop_measure:
            for metric in stop_metric:
                for a in alpha:
                    for p in precision:
                        bgd.fit(X=X, y=y, theta=theta, X_val=X_val, y_val=y_val, 
                                alpha=a, maxiter=maxiter, precision=p, stop_measure=measure, 
                                stop_metric=metric, scaler=scaler)
                        detail = bgd.get_detail()
                        summary = bgd.summary()
                        self._detail = pd.concat([self._detail, detail], axis=0)    
                        self._summary = pd.concat([self._summary, summary], axis=0)    

    def _save_fig(self, fig, directory, filename):
        if os.path.exists(directory):
            path = os.path.join(os.path.abspath(directory), filename)
            fig.savefig(path, facecolor='w')
        else:
            os.makedirs(directory)
            path = os.path.join(os.path.abspath(directory),filename)
            fig.savefig(path, facecolor='w')
                        

    def _plot_costs(self, search, directory=None, show=True):
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
            title = search['stop_condition'].iloc[0]
            ax0.set_title(title, color='k')

            ax1 = plt.subplot2grid((rows,cols), (i,1))
            ax1 = sns.barplot(x='alpha', y='final_costs_val', hue='precision', data=search)
            ax1.set_facecolor('w')
            ax1.tick_params(colors='k')
            ax1.xaxis.label.set_color('k')
            ax1.yaxis.label.set_color('k')
            ax1.set_xlabel('Learning Rate')
            ax1.set_ylabel('Validation Set Costs')
            title = search['stop_condition'].iloc[0]
            ax1.set_title(title, color='k')
            i += 1

        # Finalize plot and save
        suptitle = self._summary.alg.iloc[0] + '\n' + 'Cost Analysis' + '\n' + \
                   'Stop Criteria: ' + search.stop_measure.iloc[0]           
        fig.suptitle(suptitle)
        fig.tight_layout(rect=[0,0,1,.9])
        if show:
            plt.show()
        if directory is not None:
            filename = 'Cost Analysis.png'
            self._save_fig(fig, directory, filename)
        plt.close(fig)
        return(fig)

    def plot_times(self, directory=None, show=True):
        # Group data and obtain keys
        search_groups = self._summary.groupby('stop_condition')   

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
            title = search['stop_condition'].iloc[0]
            ax.set_title(title, color='k')
            i += 1

        # Finalize plot and save
        suptitle = self._summary.alg.iloc[0] + '\n' + 'Computation Time' 
        fig.suptitle(suptitle)
        fig.tight_layout(rect=[0,0,1,.9])
        if show:
            plt.show()
        if directory is not None:
            filename = 'Computation Time.png'
            self._save_fig(fig, directory, filename)
        plt.close(fig)
        return(fig)

    def _plot_curves(self, searches, directory=None, show=True):

        # Group data and obtain keys
        search_groups = searches.groupby('stop')   

        # Set Grid Dimensions
        cols = 2  
        rows = math.ceil(search_groups.ngroups/cols)
        odd = True if search_groups.ngroups % cols != 0 else False

        # Obtain and initialize matplotlib figure
        fig = plt.figure(figsize=(12,4*rows))        
        sns.set(style="whitegrid", font_scale=1)
        suptitle = searches.alg.iloc[0] + '\n' + 'Learning Curves' + '\n' + \
                   'Stopping Criteria: ' + searches.stop_measure.iloc[0] 
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
            title = search['stop'].iloc[0]
            ax.set_title(title, color='k')
            i += 1

        # Finalize plot and save
        fig.tight_layout()
        if show:
            plt.show()
        if directory is not None:
            filename = 'Learning Curves by Stop Condition' + searches['stop_condition'].iloc[0] + '.png'
            self._save_fig(fig, directory, filename)
        plt.close(fig)
        return(fig)

    def plot_curves(self, directory=None, show=True):
        detail = self._detail.groupby(['stop_measure'])
        for measure, data in detail:
            self._plot_curves(data, directory, show)

    def plot_costs(self, directory=None, show=True):
        searches = self._summary.groupby(['stop_measure'])
        for measure, data in searches:
            self._plot_costs(data, directory, show)



# --------------------------------------------------------------------------- #
#                              BGD Lab Class                                  #   
# --------------------------------------------------------------------------- #
class BGDLab(GradientLab):  

    def __init__(self):        
        self._detail = None  
        self._alg = 'Batch Gradient Descent'
        
