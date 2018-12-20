
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
        return(self._searches.sort_values(by='final r2_val', axis=0, ascending=False))
        
    def fit(self, X, y, X_val, y_val, theta, alpha, precision, 
               stop_measure='j', stop_metric='a', maxiter=0, scaler='minmax'):

        bgd = BGD()
        self._searches = pd.DataFrame()
        for a in alpha:
            for p in precision:
                bgd.fit(X=X, y=y, theta=theta, X_val=X_val, y_val=y_val, 
                           alpha=a, maxiter=maxiter, precision=p, stop_measure=stop_measure, 
                           stop_metric=stop_metric, scaler=scaler)
                result = bgd.summary()
                self._searches = pd.concat([self._searches, result], axis=0)

    def plot(self, path=None, show=True):
        # Obtain matplotlib figure
        fig = plt.figure(figsize=(12,8))
        gs = fig.add_gridspec(2,2)
        sns.set(style="whitegrid", font_scale=1)

        # Obtain the data
        df = self.report()
        
        # Training Set Bar Plot
        ax0 = fig.add_subplot(gs[0,0])
        ax0 = sns.barplot(x=df['alpha'], y=df['final r2'], hue=df['precision'])  
        ax0.set_facecolor('w')
        ax0.tick_params(colors='k')
        ax0.xaxis.label.set_color('k')
        ax0.yaxis.label.set_color('k')        
        ax0.set_ylabel('R2 Score')
        ax0.set_ylim(bottom=0)
        title = self._alg + '\n' + 'Training Set R2 Scores ' + '\n' + \
                'Stops When ' + df['stop'].iloc[0] 
        ax0.set_title(title, color='k', pad=15)

        # Validation Set Bar Plot
        ax1 = fig.add_subplot(gs[0,1])
        ax1 = sns.barplot(x=df['alpha'], y=df['final r2_val'], hue=df['precision'])  
        ax1.set_facecolor('w')
        ax1.tick_params(colors='k')
        ax1.xaxis.label.set_color('k')
        ax1.yaxis.label.set_color('k')        
        ax1.set_ylabel('R2 Score')
        ax1.set_ylim(bottom=0)
        title = self._alg + '\n' + 'Validation Set R2 Scores ' + '\n' + \
                'Stops When ' + df['stop'].iloc[0] 
        ax1.set_title(title, color='k', pad=15)

        # Training Set Time
        ax2 = fig.add_subplot(gs[1,0])
        ax2 = sns.barplot(x=df['alpha'], y=df['duration'], hue=df['precision'])  
        ax2.set_facecolor('w')
        ax2.tick_params(colors='k')
        ax2.xaxis.label.set_color('k')
        ax2.yaxis.label.set_color('k')        
        ax2.set_ylabel('R2 Score')
        title = self._alg + '\n' + 'Duration on Training Set ' + '\n' + \
                'Stops When ' + df['stop'].iloc[0] 
        ax2.set_title(title, color='k', pad=15)

        # Training Set Iterations
        ax3 = fig.add_subplot(gs[1,1])
        ax3 = sns.barplot(x=df['alpha'], y=df['iterations'], hue=df['precision'])  
        ax3.set_facecolor('w')
        ax3.tick_params(colors='k')
        ax3.xaxis.label.set_color('k')
        ax3.yaxis.label.set_color('k')        
        ax3.set_ylabel('R2 Score')
        title = self._alg + '\n' + 'Training Set Iterations ' + '\n' + \
                'Stops When ' + df['stop'].iloc[0] 
        ax3.set_title(title, color='k', pad=15)

        fig.tight_layout()
        if show:
            plt.show()
        if path is not None:
            if os.path.exists(os.path.dirname(path)):
                fig.savefig(path, facecolor='w')
            else:
                os.makedirs(os.path.dirname(path))
                fig.savefig(path, facecolor='w')
        plt.close(fig)        



# --------------------------------------------------------------------------- #
#                              BGD Lab Class                                  #   
# --------------------------------------------------------------------------- #
class BGDLab(GradientLab):  

    def __init__(self):        
        self._searches = None  
        self._alg = 'Batch Gradient Descent'
        
