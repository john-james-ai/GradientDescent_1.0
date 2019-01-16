
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

from utils import save_fig, save_csv, save_gif

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
            

    def _cost_mesh(self,X, y, THETA):
        return(np.sum((X.dot(THETA) - y)**2)/(2*len(y)))        

    def show_search(self, alg, X, y, detail, summary, directory=None, filename=None, fontsize=None,
                    interval=200, fps=60, maxframes=500):
        '''Plots surface plot on two dimensional problems only 
        '''        
        # Designate plot area
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        sns.set(style="whitegrid", font_scale=1)

        # Create index for n <= maxframes number of points
        idx = np.arange(0,detail.shape[0])
        nth = math.floor(detail.shape[0]/maxframes)
        nth = max(nth,1) 
        idx = idx[::nth]

        # Create the x=theta0, y=theta1 grid for plotting
        iterations = detail['iterations']
        costs = detail['cost']     
        theta0 = detail['theta_0']
        theta1 = detail['theta_1']

        # Establish boundaries of plot
        theta0_min = min(-1, min(theta0))
        theta1_min = min(-1, min(theta1))
        theta0_max = max(1, max(theta0))
        theta1_max = max(1, max(theta1))
        theta0_mesh = np.linspace(theta0_min, theta0_max, 100)
        theta1_mesh = np.linspace(theta1_min, theta1_max, 100)
        theta0_mesh, theta1_mesh = np.meshgrid(theta0_mesh, theta1_mesh)

        # Create cost grid based upon x,y the grid of thetas
        Js = np.array([self._cost_mesh(X, y, THETA)
                    for THETA in zip(np.ravel(theta0_mesh), np.ravel(theta1_mesh))])
        Js = Js.reshape(theta0_mesh.shape)

        # Set Title
        title = alg + '\n' + r' $\alpha$' + " = " + str(summary['learning_rate'].item())
        if fontsize:
            ax.set_title(title, color='k', pad=30, fontsize=fontsize)                            
            display = ax.text2D(0.1,0.92, '', transform=ax.transAxes, color='k', fontsize=fontsize)
        else:
            ax.set_title(title, color='k', pad=30)               
            display = ax.text2D(0.3,0.92, '', transform=ax.transAxes, color='k')             
        # Set face, tick,and label colors 
        ax.set_facecolor('w')
        ax.tick_params(colors='k')
        ax.xaxis.label.set_color('k')
        ax.yaxis.label.set_color('k')
        # make the panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # make the grid lines transparent
        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        # Make surface plot
        ax.plot_surface(theta0_mesh, theta1_mesh, Js, rstride=1,
                cstride=1, cmap='jet', alpha=0.5, linewidth=0)
        ax.set_xlabel(r'Intercept($\theta_0$)')
        ax.set_ylabel(r'Slope($\theta_1$)')
        ax.set_zlabel(r'Cost $J(\theta)$')        
        ax.view_init(elev=30., azim=30)

        # Build the empty line plot at the initiation of the animation
        line3d, = ax.plot([], [], [], 'r-', label = 'Gradient descent', lw = 1.5)
        line2d, = ax.plot([], [], [], 'b-', label = 'Gradient descent', lw = 1.5)
        point3d, = ax.plot([], [], [], 'bo')
        point2d, = ax.plot([], [], [], 'bo')

        def init():

            # Initialize 3d line and point
            line3d.set_data([],[])
            line3d.set_3d_properties([])
            point3d.set_data([], [])
            point3d.set_3d_properties([])

            # Initialize 2d line and point
            line2d.set_data([],[])
            line2d.set_3d_properties([])
            point2d.set_data([], [])
            point2d.set_3d_properties([])

            # Initialize display
            display.set_text('')
            return (line2d, point2d, line3d, point3d, display,)

        # Animate the regression line as it converges
        def animate(i):
            # Animate 3d Line
            line3d.set_data(theta0[:idx[i]], theta1[:idx[i]])
            line3d.set_3d_properties(costs[:idx[i]])

            # Animate 3d points
            point3d.set_data(theta0[idx[i]], theta1[idx[i]])
            point3d.set_3d_properties(costs[idx[i]])

            # Animate 2d Line
            line2d.set_data(theta0[:idx[i]], theta1[:idx[i]])
            line2d.set_3d_properties(0)

            # Animate 2d points
            point2d.set_data(theta0[idx[i]], theta1[idx[i]])
            point2d.set_3d_properties(0)

            # Update display
            display.set_text('Iteration: '+ str(iterations[idx[i]]) + r'$\quad\theta_0=$ ' +
                            str(round(theta0[idx[i]],3)) + r'$\quad\theta_1=$ ' + str(round(theta1[idx[i]],3)) +
                            r'$\quad J(\theta)=$ ' + str(np.round(costs[idx[i]], 5)))


            return(line3d, point3d, line2d, point2d, display)

        # create animation using the animate() function
        surface_ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(idx),
                                            interval=interval, blit=True, repeat_delay=3000)
        if directory is not None:
            if filename is None:
                filename = alg + ' Search Path Learning Rate ' + str(summary['learning_rate'].item()) +  '.gif'
            save_gif(surface_ani, directory, filename, fps)
        plt.close(fig)
        return(surface_ani)

    def show_fit(self, alg, X, y, detail, summary, directory=None, filename=None, fontsize=None,
                 interval=50, fps=60, maxframes=500):
        '''Shows animation of regression line fit for 2D X Vector 
        '''

        # Create index for n <= maxframes number of points
        idx = np.arange(0,detail.shape[0])
        nth = math.floor(detail.shape[0]/maxframes)
        nth = max(nth,1) 
        idx = idx[::nth]

        # Extract data for plotting
        x = X[X.columns[1]]
        iterations = detail['iterations']
        costs = detail['cost']        
        theta0 = detail['theta_0']
        theta1 = detail['theta_1']
        theta = np.array([theta0, theta1])

        # Render scatterplot
        fig, ax = plt.subplots(figsize=(12,8))
        sns.set(style="whitegrid", font_scale=1)
        sns.scatterplot(x=x, y=y, ax=ax)
        # Set face, tick,and label colors 
        ax.set_facecolor('w')
        ax.tick_params(colors='k')
        ax.xaxis.label.set_color('k')
        ax.yaxis.label.set_color('k')
        ax.set_ylim(-2,2)
        # Initialize line
        line, = ax.plot([],[],'r-', lw=2)
        # Set Title, Annotations and label
        title = alg + '\n' + r' $\alpha$' + " = " + str(round(summary['learning_rate'].item(),3)) 
        if fontsize:
            ax.set_title(title, color='k', fontsize=fontsize)                                    
            display = ax.text(0.1, 0.9, '', transform=ax.transAxes, color='k', fontsize=fontsize)
        else:
            ax.set_title(title, color='k')                                    
            display = ax.text(0.35, 0.9, '', transform=ax.transAxes, color='k')
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        fig.tight_layout()

        # Build the empty line plot at the initiation of the animation
        def init():
            line.set_data([],[])
            display.set_text('')
            return (line, display,)

        # Animate the regression line as it converges
        def animate(i):

            # Animate Line
            y=X.dot(theta[:,idx[i]])
            line.set_data(x,y)

            # Animate text
            display.set_text('Iteration: '+ str(iterations[idx[i]]) + r'$\quad\theta_0=$ ' +
                            str(round(theta0[idx[i]],3)) + r'$\quad\theta_1=$ ' + str(round(theta1[idx[i]],3)) +
                            r'$\quad J(\theta)=$ ' + str(round(costs[idx[i]], 3)))
            return (line, display)

        # create animation using the animate() function
        line_gd = animation.FuncAnimation(fig, animate, init_func=init, frames=len(idx),
                                            interval=interval, blit=True, repeat_delay=3000)
        if directory is not None:
            if filename is None:
                filename = title = alg + ' Fit Plot Learning Rate ' + str(round(summary['learning_rate'].item(),3)) + '.gif'  
            save_gif(line_gd, directory, filename, fps)
        plt.close(fig)  
        return(line_gd)
