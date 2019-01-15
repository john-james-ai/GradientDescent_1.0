
# %%
# =========================================================================== #
#                               GRADIENT VISUAL                               #
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
#                             GRADIENTVISUAL CLASS                            #  
# --------------------------------------------------------------------------- #
class GradientVisual:
    '''
    Base class for gradient descent plots
    '''

    def __init__(self):
        pass

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

    def figure(self, alg, data, x, y, z=None, groupby=None, func=None, cols=2,
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
            suptitle = alg + '\n' + self._get_label(y) + '\n' + ' By ' + self._get_label(x)
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
            title = alg + '\n' + self._get_label(y) + ' By ' + self._get_label(x)
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
            

