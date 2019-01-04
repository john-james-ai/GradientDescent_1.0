
# %%
# =========================================================================== #
#                               GRADIENT DEMO                                 #
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

src = "c:\\Users\\John\\Documents\\Data Science\\Libraries\\GradientDescent\\src"
sys.path.append(src)
import data
from GradientDescent import BGD, SGD
from utils import save_gif 

class GradientDemo():
    def __init__(self):
        self._alg = 'Gradient Descent'
        self._search = []
        self._summary = []
        self._X = None
        self._y = None        

    def fit(self, X, y, theta, X_val=None, y_val=None, n=500, alpha=0.01, 
            miniter=0, maxiter=10000, precision=0.001, stop_parameter='t',
            stop_metric='r'):

        # Fit to data
        gd = BGD()
        gd.fit(X=X, y=y, theta=theta, X_val=X_val, y_val=y_val, alpha=alpha, 
               miniter=miniter, maxiter=maxiter, 
               precision=precision, stop_parameter=stop_parameter,
               stop_metric=stop_metric)

        # Obtain search history detail
        self._search = gd.get_detail()
        self._summary = gd.summary()

        # Extract transformed data for plotting
        data = gd.get_transformed_data()
        self._X = data['X']
        self._y = data['y']

    def _cost_mesh(self,THETA):
        return(np.sum((self._X.dot(THETA) - self._y)**2)/(2*len(self._y)))        

    def show_search(self, directory=None, filename=None, fontsize=None,
                    interval=200, fps=60, maxframes=500):
        '''Plots surface plot on two dimensional problems only 
        '''        
        # Designate plot area
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Create index for n <= maxframes number of points
        idx = np.arange(0,self._search.shape[0])
        nth = math.floor(self._search.shape[0]/maxframes)
        nth = max(nth,1) 
        idx = idx[::nth]
        
        # Create the x=theta0, y=theta1 grid for plotting]
        theta0 = self._search['theta_0']
        theta1 = self._search['theta_1']

        # Establish boundaries of plot
        theta0_min = min(-1, min(theta0))
        theta1_min = min(-1, min(theta1))
        theta0_max = max(1, max(theta0))
        theta1_max = max(1, max(theta1))
        theta0_mesh = np.linspace(theta0_min, theta0_max, 100)
        theta1_mesh = np.linspace(theta1_min, theta1_max, 100)
        theta0_mesh, theta1_mesh = np.meshgrid(theta0_mesh, theta1_mesh)

        # Create cost grid based upon x,y the grid of thetas
        Js = np.array([self._cost_mesh(THETA)
                    for THETA in zip(np.ravel(theta0_mesh), np.ravel(theta1_mesh))])
        Js = Js.reshape(theta0_mesh.shape)

        # Set Title
        title = self._alg + '\n' + r' $\alpha$' + " = " + str(round(self._summary['alpha'].item(),3)) + "\n" + \
                'Stop Condition: ' + self._summary['stop'].item() 
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
            line3d.set_3d_properties(self._search['cost'][:idx[i]])

            # Animate 3d points
            point3d.set_data(theta0[idx[i]], theta1[idx[i]])
            point3d.set_3d_properties(self._search['cost'][idx[i]])

            # Animate 2d Line
            line2d.set_data(theta0[:idx[i]], theta1[:idx[i]])
            line2d.set_3d_properties(0)

            # Animate 2d points
            point2d.set_data(theta0[idx[i]], theta1[idx[i]])
            point2d.set_3d_properties(0)

            # Update display
            display.set_text('Epoch: '+ str(idx[i]) + r'$\quad\theta_0=$ ' +
                            str(round(theta0[idx[i]],3)) + r'$\quad\theta_1=$ ' + str(round(theta1[idx[i]],3)) +
                            r'$\quad J(\theta)=$ ' + str(np.round(self._search['cost'][idx[i]], 5)))


            return(line3d, point3d, line2d, point2d, display)

        # create animation using the animate() function
        surface_ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(idx),
                                            interval=interval, blit=True, repeat_delay=3000)
        if directory is not None:
            if filename is None:
                filename = self._alg + ' Search Plot Learning Rate ' + str(round(self._summary['alpha'].item(),3)) + \
                    ' Stop Condition ' + self._summary['stop'].item() + '.gif'              
            save_gif(surface_ani, directory, filename, fps)
        plt.close(fig)
        return(surface_ani)

    def show_fit(self, directory=None, filename=None, fontsize=None,
                 interval=50, fps=60, maxframes=500):
        '''Shows animation of regression line fit for 2D X Vector 
        '''

        # Create index for n <= maxframes number of points
        idx = np.arange(0,self._search.shape[0])
        nth = math.floor(self._search.shape[0]/maxframes)
        nth = max(nth,1) 
        idx = idx[::nth]

        # Extract data for plotting
        X = self._X
        y = self._y
        x = X[X.columns[1]]
        theta0 = self._search['theta_0']
        theta1 = self._search['theta_1']
        theta = np.array([theta0, theta1])

        # Render scatterplot
        fig, ax = plt.subplots(figsize=(12,8))
        sns.scatterplot(x=x, y=y, ax=ax)
        # Set face, tick,and label colors 
        ax.set_facecolor('w')
        ax.tick_params(colors='k')
        ax.xaxis.label.set_color('k')
        ax.yaxis.label.set_color('k')
        ax.set_ylim(-1,2)
        # Initialize line
        line, = ax.plot([],[],'r-', lw=2)
        # Set Title, Annotations and label
        title = self._alg + '\n' + r' $\alpha$' + " = " + str(round(self._summary['alpha'].item(),3)) + "\n" + \
                'Stop Condition: ' + self._summary['stop'].item()  
        if fontsize:
            ax.set_title(title, color='k', fontsize=fontsize)                                    
            display = ax.text(0.1, 0.9, '', transform=ax.transAxes, color='k', fontsize=fontsize)
        else:
            ax.set_title(title, color='k')                                    
            display = ax.text(0.2, 0.9, '', transform=ax.transAxes, color='k')
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
            display.set_text('Epoch: '+ str(idx[i]) + r'$\quad\theta_0=$ ' +
                            str(round(theta0[idx[i]],3)) + r'$\quad\theta_1=$ ' + str(round(theta1[idx[i]],3)) +
                            r'$\quad J(\theta)=$ ' + str(round(self._search['cost'][idx[i]], 3)))
            return (line, display)

        # create animation using the animate() function
        line_gd = animation.FuncAnimation(fig, animate, init_func=init, frames=len(idx),
                                            interval=interval, blit=True, repeat_delay=3000)
        if directory is not None:
            if filename is None:
                filename = title = self._alg + ' Fit Plot Learning Rate ' + str(round(self._summary['alpha'].item(),3)) + \
                    ' Stop Condition ' + self._summary['stop'].item() + '.gif'  
            save_gif(line_gd, directory, filename, fps)
        plt.close(fig)  
        return(line_gd)

# --------------------------------------------------------------------------- #
#                       BATCH GRADIENT DESCENT DEMO                           #
# --------------------------------------------------------------------------- #

class BGDDemo(GradientDemo):
    '''Batch Gradient Descent'''

    def __init__(self):
        self._alg = "Batch Gradient Descent"
        self._search = []
        self._summary = []
        self._X = None
        self._y = None         

# --------------------------------------------------------------------------- #
#                      STOCHASTIC GRADIENT DESCENT DEMO                       #
# --------------------------------------------------------------------------- #

class SGDDemo(GradientDemo):
    '''Stochastic Gradient Descent'''

    def __init__(self):
        self._alg = "Stochastic Gradient Descent"
        self._search = []
        self._summary = []
        self._X = None
        self._y = None     

    def fit(self, X, y, theta, X_val=None, y_val=None, n=500, alpha=0.01, 
            miniter=0, maxiter=10000, check_point=0.1, precision=0.001, stop_parameter='t',
            stop_metric='r'):

        # Fit to data
        gd = SGD()
        gd.fit(X=X, y=y, theta=theta, X_val=X_val, y_val=y_val, alpha=alpha, 
               miniter=miniter, maxiter=maxiter, check_point=check_point,
               precision=precision, stop_parameter=stop_parameter,
               stop_metric=stop_metric)

        # Obtain search history detail
        self._search = gd.get_detail()
        self._summary = gd.summary()

        # Extract transformed data for plotting
        data = gd.get_transformed_data()
        self._X = data['X']
        self._y = data['y']        
        
        