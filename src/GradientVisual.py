
# %%
# =========================================================================== #
#                             GRADIENT VISUAL                                 #
# =========================================================================== #
'''
    Module for visualizing and evaluating gradient descent algorithms. It is 
    comprised of an abstract base class that defines methods common among the 
    various gradient descent algorithms. It also has concrete classes for each
    gradient descent variant, including:
    - Batch Gradient Descent
    - Stochastic Gradient Descent
    - Mini-batch Gradient Descent

    Visualization classes also exist for several gradient descent optimization
    algorithms, including:
    - Momentum 
    - Nesterov Accelerated Gradient
    - Adagrad
    - Adadelta
    - RMSProp
    - Adam
    - Adamax
    - Nadam    
'''
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
rcParams['animation.embed_limit'] = 60
rc('animation', html='jshtml')
rc
# --------------------------------------------------------------------------- #
#                         GRADIENTVISUAL BASE CLASS                           #
# --------------------------------------------------------------------------- #
class GradientVisual:
    '''
    Base class for all gradient descent variants and optimization algorithms.
    '''

    def __init__(self):
        self._search = None

    def _get_search(self, search=None):
        if search is None:
            if self._search is None:
                raise Exception('No gradient search object provided')
        else:
            self._search = search
        return(self._search)

    def _todf(self, x, stub):
        n = len(x[0])
        df = pd.DataFrame()
        for i in range(n):
            colname = stub + str(i)
            vec = [item[i] for item in x]
            df_vec = pd.DataFrame(vec, columns=[colname])
            df = pd.concat([df, df_vec], axis=1)
        return(df)

    def report(self, search=None, thetas=False, gradients=False):

        search = self._get_search(search)

        # Format 
        iterations = pd.DataFrame(search['iteration'], columns=['Iteration'])
        costs = pd.DataFrame(search['J_history'], columns=['Cost'])
        thetas = self._todf(search['theta_history'], stub='theta_')
        gradients = self._todf(search['g_history'], stub='gradient_')
        result = pd.concat([iterations, thetas, costs, gradients], axis=1)
        if n:
            result = result.iloc[0:n]
        return(result)

    def display_costs(self, costs=None, nth=1,interval=100, path=None, fps=40):

        # Sets costs 
        if costs is None:
            costs = self._J_history

        # Create iteration vector
        iteration = list(range(0,len(costs)))

        # Extract 100 datapoints plus last for animation
        nth = math.floor(len(costs)/100)
        nth = max(nth,1)
        iteration_plot = iteration[::nth]
        iteration_plot.append(iteration[-1])
        cost_bar = costs[::nth]
        cost_bar.append(costs[-1])


        # Establish figure and axes objects and lines and points
        fig, ax = plt.subplots()
        line, = ax.plot([], [], 'r', lw=1.5)
        point, = ax.plot([], [], 'bo')
        epoch_display = ax.text(.7, 0.9, '',
                                transform=ax.transAxes, fontsize=16)
        J_display = ax.text(.7, 0.8, '', transform=ax.transAxes, fontsize=16)

        # Plot cost line
        ax.plot(np.arange(len(cost_bar)), cost_bar, c='r')

        # Set labels and title
        title = self._alg + r'$\alpha$' + " = " + str(round(self._alpha,3)) + " " + r'$\epsilon$' + " = " + str(round(self._precision,5)) 
        ax.set_xlabel(self._unit)
        ax.set_ylabel(r'$J(\theta)$')
        ax.set_title(title)

        def init():
            line.set_data([], [])
            point.set_data([], [])
            epoch_display.set_text('')
            J_display.set_text('')

            return(line, point, epoch_display, J_display)

        def animate(i):
            # Animate points
            point.set_data(iteration_plot[i], cost_bar[i])

            # Animate value display
            epoch_display.set_text(self._unit + " = " + str(iteration_plot[i]))
            J_display.set_text(r'     $J(\theta)=$' +
                               str(round(cost_bar[i], 4)))

            return(line, point, epoch_display, J_display)

        display = animation.FuncAnimation(fig, animate, init_func=init,
                                          frames=len(cost_bar), interval=interval,
                                          blit=True, repeat_delay=500)
        if path:
            face_edge_colors = {'facecolor': 'w', 'edgecolor': 'w'}
            display.save(path, writer='imagemagick', fps=fps, savefig_kwargs = face_edge_colors)

        plt.close(fig)
        return(display)

    def surface(self, theta_history=None, cost_history=None, interval=50, path=None, fps=40):
        '''Plots surface plot on two dimensional problems only 
        '''
        if theta_history is None:
            theta_history = self._theta_history
            cost_history = self._J_history
        
        if len(theta_history[0]) > 2:
            raise Exception("Surface only works on 2-dimensional X arrays")  

        # Create iteration vector
        iteration_all = list(range(1,len(theta_history)+1))

        # Extract 100 datapoints for animation
        nth = math.floor(len(theta_history)/100)
        nth = max(nth,1)
        iteration = iteration_all[::nth]
        iteration.append(iteration_all[-1])
        theta = theta_history[::nth]
        theta.append(theta_history[-1])
        costs = cost_history[::nth]  
        costs.append(cost_history[-1])      

        # Designate plot area
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create the x=theta0, y=theta1 grid for plotting
        theta0 = [item[0] for item in theta]
        theta1 = [item[1] for item in theta]

        # Establish boundaries of plot
        theta0_min = min(-1, min(theta0))
        theta1_min = min(-1, min(theta1))
        theta0_max = max(1, max(theta0))
        theta1_max = max(1, max(theta1))
        theta0_mesh = np.linspace(theta0_min, theta0_max, 100)
        theta1_mesh = np.linspace(theta1_min, theta1_max, 100)
        theta0_mesh, theta1_mesh = np.meshgrid(theta0_mesh, theta1_mesh)

        # Create cost grid based upon x,y the grid of thetas
        Js = np.array([self.cost_mesh(self._X, self._y, THETA)
                    for THETA in zip(np.ravel(theta0_mesh), np.ravel(theta1_mesh))])
        Js = Js.reshape(theta0_mesh.shape)

        # make the panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # make the grid lines transparent
        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        # Make surface plot
        title = self._alg + " " + r'$\alpha$' + " = " + str(round(self._alpha,3)) + " " + r'$\epsilon$' + " = " + str(round(self._precision,5)) 
        ax.plot_surface(theta0_mesh, theta1_mesh, Js, rstride=1,
                cstride=1, cmap='jet', alpha=0.5, linewidth=0)
        ax.set_xlabel(r'Intercept($\theta_0$)')
        ax.set_ylabel(r'Slope($\theta_1$)')
        ax.set_zlabel(r'Cost $J(\theta)$')
        ax.set_title(title, pad=30)
        ax.view_init(elev=30., azim=30)

        # Build the empty line plot at the initiation of the animation
        line3d, = ax.plot([], [], [], 'r-', label = 'Gradient descent', lw = 1.5)
        line2d, = ax.plot([], [], [], 'b-', label = 'Gradient descent', lw = 1.5)
        point3d, = ax.plot([], [], [], 'bo')
        point2d, = ax.plot([], [], [], 'bo')
        display = ax.text2D(0.3,0.95, '', transform=ax.transAxes)

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
            line3d.set_data(theta0[:i], theta1[:i])
            line3d.set_3d_properties(costs[:i])

            # Animate 3d points
            point3d.set_data(theta0[i], theta1[i])
            point3d.set_3d_properties(costs[i])

            # Animate 2d Line
            line2d.set_data(theta0[:i], theta1[:i])
            line2d.set_3d_properties(0)

            # Animate 2d points
            point2d.set_data(theta0[i], theta1[i])
            point2d.set_3d_properties(0)

            # Update display
            display.set_text('Epoch: '+ str(iteration[i]) + r'$\quad\theta_0=$ ' +
                            str(round(theta0[i],3)) + r'$\quad\theta_1=$ ' + str(round(theta1[i],3)) +
                            r'$\quad J(\theta)=$ ' + str(np.round(costs[i], 5)))


            return(line3d, point3d, line2d, point2d, display)

        # create animation using the animate() function
        surface_ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(theta0),
                                            interval=interval, blit=True, repeat_delay=1000)
        if path:
            face_edge_colors = {'facecolor': 'w', 'edgecolor': 'w'}
            surface_ani.save(path, writer='imagemagick', fps=fps, savefig_kwargs=face_edge_colors)
        plt.close(fig)
        return(surface_ani)

    def show_fit(self, theta_history=None, J_history=None, interval=50, path=None, fps=40):
        '''Shows animation of regression line fit for 2D X Vector 
        '''
        if theta_history is None:
            theta_history = self._theta_history
            J_history = self._J_history

        if self._X.shape[1] > 2:
            raise Exception("Show_fit only works on 2-dimensional X arrays")    

        # Create iteration vector
        iteration_all = list(range(1, len(theta_history)+1))

        # Extract 100 datapoints for plotting
        nth = math.floor(len(theta_history)/100)
        nth = max(nth,1)
        iteration = iteration_all[::nth]
        iteration.append(iteration_all[-1])
        theta = theta_history[::nth]
        theta.append(theta_history[-1])
        costs = J_history[::nth]
        costs.append(J_history[-1])        

        # Extract data for plotting
        X = self._X
        y = self._y
        x = [item[1] for item in X]
        theta0 = [item[0] for item in theta]
        theta1 = [item[1] for item in theta]

        # Render scatterplot
        fig, ax = plt.subplots()
        sns.scatterplot(x=x, y=y, ax=ax)

        # Initialize line
        line, = ax.plot([],[],'r-', lw=2)

        # Annotations and labels
        title = "Model Fit By " + self._alg + r'$\alpha$' + " = " + str(round(self._alpha,3)) + " " + r'$\epsilon$' + " = " + str(round(self._precision,5))
        display = ax.text(0.1, 0.9, '', transform=ax.transAxes, fontsize=16)
        ax.set_xlabel('X', fontsize='large')
        ax.set_ylabel('y', fontsize='large')
        ax.set_title(title, fontsize='x-large')

        # Build the empty line plot at the initiation of the animation
        def init():
            line.set_data([],[])
            display.set_text('')
            return (line, display,)

        # Animate the regression line as it converges
        def animate(i):

            # Animate Line
            y=X.dot(theta[i])
            line.set_data(x,y)

            # Animate text
            display.set_text('Epoch: '+ str(iteration[i]) + r'$\quad\theta_0=$ ' +
                            str(round(theta0[i],3)) + r'$\quad\theta_1=$ ' + str(round(theta1[i],3)) +
                            r'$\quad J(\theta)=$ ' + str(round(costs[i], 3)))
            return (line, display)

        # create animation using the animate() function
        line_gd = animation.FuncAnimation(fig, animate, init_func=init, frames=len(theta),
                                            interval=interval, blit=True, repeat_delay=1000)
        if path:
            face_edge_colors = {'facecolor': 'w', 'edgecolor': 'w'}
            line_gd.save(path, writer='imagemagick', fps=fps, savefig_kwargs=face_edge_colors)
        plt.close(fig)
        return(line_gd)  

    def _cost_line(self):
        df = pd.DataFrame()
        for s in self._searches:
            df2 = pd.DataFrame({'Iteration': s['iterations'],
                                'Cost': s['J_history'],
                                'Learning Rate': round(s['alpha'],2)})
            df = df.append(df2)
        sns.set(style="whitegrid", font_scale=1)
        p = sns.lineplot(x=df.iloc[:,0], y=df.iloc[:,1], hue=df.iloc[:,2], 
                         legend='full')          
        return(p)        

    def _mse_line(self):
        df = pd.DataFrame()
        for s in self._searches:
            df2 = pd.DataFrame({'Iteration': s['iterations'],
                                'MSE': s['mse_history'],
                                'Learning Rate': round(s['alpha'],2)})
            df = df.append(df2)
        sns.set(style="whitegrid", font_scale=1)
        p = sns.lineplot(x=df.iloc[:,0], y=df.iloc[:,1], hue=df.iloc[:,2], 
                         legend='full')          
        return(p)          

    def _cost_bar(self):
        df = pd.DataFrame()        
        for s in self._searches:
            df2 = pd.DataFrame({'Learning Rate': round(s['alpha'],2),
                                'Cost': s['J_history'][-1]}, index=[0])
            df = df.append(df2)
        sns.set(style="whitegrid", font_scale=1)
        p = sns.barplot(x=df.iloc[:,0], y=df.iloc[:,1]) 
        return(p)      

    def _mse_bar(self):
        df = pd.DataFrame()        
        for s in self._searches:
            df2 = pd.DataFrame({'Learning Rate': round(s['alpha'],2),
                                'MSE': s['mse_history'][-1]}, index=[0])
            df = df.append(df2)
        sns.set(style="whitegrid", font_scale=1)
        p = sns.barplot(x=df.iloc[:,0], y=df.iloc[:,1]) 
        return(p)        


    def _iterations_plot(self):
        df = pd.DataFrame()
        for s in self._searches:
            df2 = pd.DataFrame({'Alpha': round(s['alpha'],2),
                                'Iterations': s['iterations'][-1]}, index=[0])
            df = df.append(df2)
        sns.set(style="whitegrid", font_scale=1)
        p = sns.barplot(x=df.iloc[:,0], y=df.iloc[:,1])
        return(p) 

    def _duration_plot(self):
        df = pd.DataFrame()
        for s in self._searches:
            df2 = pd.DataFrame({'Alpha': round(s['alpha'],2),
                                'Elapsed (ms)': s['elapsed_ms']}, index=[0])
            df = df.append(df2)
        sns.set(style="whitegrid", font_scale=1)
        p = sns.barplot(x=df.iloc[:,0], y=df.iloc[:,1])
        return(p)      

    def _evaluation_plot_mse_time(self, df):
        fig = plt.figure(figsize=(12,4))
        gs = fig.add_gridspec(1,2)
        sns.set(style="whitegrid", font_scale=1)
        
        # Create MSE v Time, controlling for learning rate
        ax0 = fig.add_subplot(gs[0])
        ax0 = sns.scatterplot(x=df['Elapsed'], y=df['MSE'], hue=df['Learning Rate']) 
        ax0.set_facecolor('w')
        ax0.tick_params(colors='k')
        ax0.xaxis.label.set_color('k')
        ax0.yaxis.label.set_color('k')        
        ax0.set_xlabel('Elapsed Time (ms)')
        ax0.set_ylabel('MSE')  
        ax0.set(ylim=(0,0.5))   
        title = 'MSE and Elapsed Time by Learning Rate'
        ax0.set_title("\n".join(wrap(title, 30)), color='k')        

        # Create MSE v Time, controlling for precision
        ax1 = fig.add_subplot(gs[1])
        ax1 = sns.scatterplot(x=df['Elapsed'], y=df['MSE'], hue=df['Precision']) 
        ax1.set_facecolor('w')
        ax1.tick_params(colors='k')
        ax1.xaxis.label.set_color('k')
        ax1.yaxis.label.set_color('k')        
        ax1.set_xlabel('Elapsed Time (ms)')
        ax1.set_ylabel('MSE')  
        ax1.set(ylim=(0,0.5))   
        title = 'MSE and Elapsed Time by Precision'
        ax1.set_title("\n".join(wrap(title, 30)), color='k')       

        plt.close()
        return(fig)

    def _evaluation_plot_time(self, df):
        fig, ax = plt.subplots(figsize=(12,8))        
        sns.set(style="whitegrid", font_scale=1)
        # Create Elapsed Time by Learning Rate and Precision
        ax = sns.barplot(x=df['Learning Rate'], y=df['Elapsed'], hue=df['Precision']) 
        ax.set_facecolor('w')
        ax.tick_params(colors='k')
        ax.xaxis.label.set_color('k')
        ax.yaxis.label.set_color('k')        
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Elapsed Time (ms)')  
        ax.set(ylim=(0,4500))      
        title = 'Elapsed Time by Learning Rate and Precision'
        ax.set_title("\n".join(wrap(title, 30)), color='k')        
        plt.close()
        return(fig)


    def _evaluation_plots_time(self, df):
        # Prepare figure object and defaults
        fig = plt.figure(figsize=(16,12))
        gs = fig.add_gridspec(3,2)         
        sns.set(style="whitegrid", font_scale=1)

        # Create Elapsed Time by Learning Rate and Precision - Abs Change in Cost
        d = df[df['Stop Condition'] == 'Absolute Change in Cost']
        ax0 = fig.add_subplot(gs[0])
        ax0 = sns.barplot(x=d['Learning Rate'], y=d['Elapsed'], hue=d['Precision']) 
        ax0.set_facecolor('w')
        ax0.tick_params(colors='k')
        ax0.xaxis.label.set_color('k')
        ax0.yaxis.label.set_color('k')        
        ax0.set_xlabel('Learning Rate')
        ax0.set_ylabel('Elapsed')
        ax0.set(ylim=(0,150))
        title = 'Elapsed Time by Learning Rate and Precision on Absolute Change in Costs'
        ax0.set_title("\n".join(wrap(title, 30)), color='k')        

        # Create Elapsed Time by Learning Rate and Precision - Pct Change in Cost
        d = df[df['Stop Condition'] == 'Percent Change in Cost']
        ax1 = fig.add_subplot(gs[1])
        ax1 = sns.barplot(x=d['Learning Rate'], y=d['Elapsed'], hue=d['Precision']) 
        ax1.set_facecolor('w')
        ax1.tick_params(colors='k')
        ax1.xaxis.label.set_color('k')
        ax1.yaxis.label.set_color('k')        
        ax1.set_xlabel('Learning Rate')
        ax1.set_ylabel('Elapsed')
        ax1.set(ylim=(0,2000))
        title = 'Elapsed Time by Learning Rate and Precision on Percent Change in Costs'
        ax1.set_title("\n".join(wrap(title, 30)), color='k')        

        # Create Elapsed Time by Learning Rate and Precision - Abs Change in Gradient
        d = df[df['Stop Condition'] == 'Absolute Change in Gradient']
        ax2 = fig.add_subplot(gs[2])
        ax2 = sns.barplot(x=d['Learning Rate'], y=d['Elapsed'], hue=d['Precision']) 
        ax2.set_facecolor('w')
        ax2.tick_params(colors='k')
        ax2.xaxis.label.set_color('k')
        ax2.yaxis.label.set_color('k')        
        ax2.set_xlabel('Learning Rate')
        ax2.set_ylabel('Elapsed')
        ax2.set(ylim=(0,200))
        title = 'Elapsed Time by Learning Rate and Precision on Absolute Change in Gradient'
        ax2.set_title("\n".join(wrap(title, 30)), color='k')  
        
        # Create Elapsed Time by Learning Rate and Precision - Pct Change in Gradient
        d = df[df['Stop Condition'] == 'Percent Change in Gradient']
        ax3 = fig.add_subplot(gs[3])
        ax3 = sns.barplot(x=d['Learning Rate'], y=d['Elapsed'], hue=d['Precision']) 
        ax3.set_facecolor('w')
        ax3.tick_params(colors='k')
        ax3.xaxis.label.set_color('k')
        ax3.yaxis.label.set_color('k')        
        ax3.set_xlabel('Learning Rate')
        ax3.set_ylabel('Elapsed')
        ax3.set(ylim=(0,8000))
        title = 'Elapsed Time by Learning Rate and Precision on Percent Change in Gradient'
        ax3.set_title("\n".join(wrap(title, 30)), color='k')  
        
        # Create Elapsed Time by Learning Rate and Precision - Abs Change in Elapsed
        d = df[df['Stop Condition'] == 'Absolute Change in Test MSE']
        ax4 = fig.add_subplot(gs[4])
        ax4 = sns.barplot(x=d['Learning Rate'], y=d['Elapsed'], hue=d['Precision']) 
        ax4.set_facecolor('w')
        ax4.tick_params(colors='k')
        ax4.xaxis.label.set_color('k')
        ax4.yaxis.label.set_color('k')        
        ax4.set_xlabel('Learning Rate')
        ax4.set_ylabel('Elapsed')
        ax4.set(ylim=(0,125))
        title = 'Elapsed Time by Learning Rate and Precision on Absolute Change in Test MSE'
        ax4.set_title("\n".join(wrap(title, 30)), color='k')  
        
        # Create Elapsed Time by Learning Rate and Precision - Pct Change in Elapsed
        d = df[df['Stop Condition'] == 'Percent Change in Test MSE']
        ax5 = fig.add_subplot(gs[5])
        ax5 = sns.barplot(x=d['Learning Rate'], y=d['Elapsed'], hue=d['Precision']) 
        ax5.set_facecolor('w')
        ax5.tick_params(colors='k')
        ax5.xaxis.label.set_color('k')
        ax5.yaxis.label.set_color('k')        
        ax5.set_xlabel('Learning Rate')
        ax5.set_ylabel('Elapsed')
        ax5.set(ylim=(0,300))
        title = 'Elapsed Time by Learning Rate and Precision on Percent Change in Test MSE'
        ax5.set_title("\n".join(wrap(title, 30)), color='k')          
                
        # Close and wrap-up
        fig.tight_layout()
        plt.close()
        return(fig)        

    def _evaluation_plot_MSE(self, df):
        fig, ax = plt.subplots(figsize=(12,8))        
        sns.set(style="whitegrid", font_scale=1)
        # Create MSE by Learning Rate and Precision
        ax = sns.barplot(x=df['Learning Rate'], y=df['MSE'], hue=df['Precision']) 
        ax.set_facecolor('w')
        ax.tick_params(colors='k')
        ax.xaxis.label.set_color('k')
        ax.yaxis.label.set_color('k')        
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('MSE')        
        ax.set(ylim=(0,1))
        title = 'MSE by Learning Rate and Precision'
        ax.set_title("\n".join(wrap(title, 30)), color='k')        
        plt.close()
        return(fig)


    def _evaluation_plots_MSE(self, df):
        # Prepare figure object and defaults
        fig = plt.figure(figsize=(16,12))
        gs = fig.add_gridspec(3,2)         
        sns.set(style="whitegrid", font_scale=1)

        # Create MSE by Learning Rate and Precision - Abs Change in Cost
        d = df[df['Stop Condition'] == 'Absolute Change in Cost']
        ax0 = fig.add_subplot(gs[0])
        ax0 = sns.barplot(x=d['Learning Rate'], y=d['MSE'], hue=d['Precision']) 
        ax0.set_facecolor('w')
        ax0.tick_params(colors='k')
        ax0.xaxis.label.set_color('k')
        ax0.yaxis.label.set_color('k')        
        ax0.set_xlabel('Learning Rate')
        ax0.set_ylabel('MSE')
        ax0.set(ylim=(0,.3))
        title = 'MSE by Learning Rate and Precision on Absolute Change in Costs'
        ax0.set_title("\n".join(wrap(title, 30)), color='k')        

        # Create MSE by Learning Rate and Precision - Pct Change in Cost
        d = df[df['Stop Condition'] == 'Percent Change in Cost']
        ax1 = fig.add_subplot(gs[1])
        ax1 = sns.barplot(x=d['Learning Rate'], y=d['MSE'], hue=d['Precision']) 
        ax1.set_facecolor('w')
        ax1.tick_params(colors='k')
        ax1.xaxis.label.set_color('k')
        ax1.yaxis.label.set_color('k')        
        ax1.set_xlabel('Learning Rate')
        ax1.set_ylabel('MSE')
        ax1.set(ylim=(0,.3))
        title = 'MSE by Learning Rate and Precision on Percent Change in Costs'
        ax1.set_title("\n".join(wrap(title, 30)), color='k')        

        # Create MSE by Learning Rate and Precision - Abs Change in Gradient
        d = df[df['Stop Condition'] == 'Absolute Change in Gradient']
        ax2 = fig.add_subplot(gs[2])
        ax2 = sns.barplot(x=d['Learning Rate'], y=d['MSE'], hue=d['Precision']) 
        ax2.set_facecolor('w')
        ax2.tick_params(colors='k')
        ax2.xaxis.label.set_color('k')
        ax2.yaxis.label.set_color('k')        
        ax2.set_xlabel('Learning Rate')
        ax2.set_ylabel('MSE')
        ax2.set(ylim=(0,.3))
        title = 'MSE by Learning Rate and Precision on Absolute Change in Gradient'
        ax2.set_title("\n".join(wrap(title, 30)), color='k')  
        
        # Create MSE by Learning Rate and Precision - Pct Change in Gradient
        d = df[df['Stop Condition'] == 'Percent Change in Gradient']
        ax3 = fig.add_subplot(gs[3])
        ax3 = sns.barplot(x=d['Learning Rate'], y=d['MSE'], hue=d['Precision']) 
        ax3.set_facecolor('w')
        ax3.tick_params(colors='k')
        ax3.xaxis.label.set_color('k')
        ax3.yaxis.label.set_color('k')        
        ax3.set_xlabel('Learning Rate')
        ax3.set_ylabel('MSE')
        ax3.set(ylim=(0,.3))
        title = 'MSE by Learning Rate and Precision on Percent Change in Gradient'
        ax3.set_title("\n".join(wrap(title, 30)), color='k')  
        
        # Create MSE by Learning Rate and Precision - Abs Change in MSE
        d = df[df['Stop Condition'] == 'Absolute Change in Test MSE']
        ax4 = fig.add_subplot(gs[4])
        ax4 = sns.barplot(x=d['Learning Rate'], y=d['MSE'], hue=d['Precision']) 
        ax4.set_facecolor('w')
        ax4.tick_params(colors='k')
        ax4.xaxis.label.set_color('k')
        ax4.yaxis.label.set_color('k')        
        ax4.set_xlabel('Learning Rate')
        ax4.set_ylabel('MSE')
        ax4.set(ylim=(0,.3))
        title = 'MSE by Learning Rate and Precision on Absolute Change in Test MSE'
        ax4.set_title("\n".join(wrap(title, 30)), color='k')  
        
        # Create MSE by Learning Rate and Precision - Pct Change in MSE
        d = df[df['Stop Condition'] == 'Percent Change in Test MSE']
        ax5 = fig.add_subplot(gs[5])
        ax5 = sns.barplot(x=d['Learning Rate'], y=d['MSE'], hue=d['Precision']) 
        ax5.set_facecolor('w')
        ax5.tick_params(colors='k')
        ax5.xaxis.label.set_color('k')
        ax5.yaxis.label.set_color('k')        
        ax5.set_xlabel('Learning Rate')
        ax5.set_ylabel('MSE')
        ax5.set(ylim=(0,.3))
        title = 'MSE by Learning Rate and Precision on Percent Change in Test MSE'
        ax5.set_title("\n".join(wrap(title, 30)), color='k')          
                
        # Close and wrap-up
        fig.tight_layout()
        plt.close()
        return(fig)

    def evaluation_plots(self, df):
        eval = dict()
        eval['mse_summary'] = self._evaluation_plot_MSE(df)
        eval['mse_detail'] = self._evaluation_plots_MSE(df)
        eval['time_summary'] = self._evaluation_plot_time(df)
        eval['time_detail'] = self._evaluation_plots_time(df)
        eval['mse_time'] = self._evaluation_plot_mse_time(df)
        return(eval)


    def _evaluation_summary(self):
        df = pd.DataFrame()
        for s in self._searches:
            measure = np.where(s['stop_measure']=='g', "Gradient",
                                 np.where(s['stop_measure']=='j', "Cost", "Test MSE"))
            metric = np.where(s['stop_metric']=='p', "Percent Change", "Absolute Change")
            condition = str(metric) + " in " + str(measure)
            df2 = pd.DataFrame({'Algorithm': self._alg,
                                'Stop Condition': condition,
                                'Learning Rate': round(s['alpha'],3),
                                'Precision' : s['precision'],
                                'Iterations': len(s['iterations']),
                                'Elapsed': s['elapsed_ms'],
                                'Cost': s['J_history'][-1],
                                'MSE': s['mse_history'][-1]}, index=[0])
            df = df.append(df2)            
        return(df)

    def _diagnostic_brief(self):

        fig = plt.figure(figsize=(16,4))
        gs = fig.add_gridspec(1,4)

        # Cost Line Plot
        ax0 = fig.add_subplot(gs[0])
        ax0 = self._cost_line()
        ax0.set_facecolor('w')
        ax0.tick_params(colors='k')
        ax0.xaxis.label.set_color('k')
        ax0.yaxis.label.set_color('k')        
        ax0.set_xlabel('Iterations')
        ax0.set_ylabel('Cost')
        title = 'Costs by Learning Rate and Precision ' + r'$\epsilon$' + " = " + str(round(self._precision,5)) 
        ax0.set_title("\n".join(wrap(title, 30)), color='k')

        # Cost Bar Plot
        ax1 = fig.add_subplot(gs[1])
        ax1 = self._cost_bar()
        ax1.set_xlabel('Learning Rate')
        ax1.set_ylabel('Final Cost')
        title = 'Final Costs by Learning Rate and Precision ' + r'$\epsilon$' + " = " + str(round(self._precision,5)) 
        ax1.set_title("\n".join(wrap(title, 30)), color='k')

        # Duration Plot
        ax2 = fig.add_subplot(gs[2])
        ax2 = self._duration_plot()  
        ax2.set_xlabel('Learning Rate')
        ax2.set_ylabel('Duration (ms)')
        title = 'Computation Time by Learning Rate and Precision ' + r'$\epsilon$' + " = " + str(round(self._precision,5)) 
        ax2.set_title("\n".join(wrap(title, 30)), color='k')

        # Iterations Plot
        ax3 = fig.add_subplot(gs[3])
        ax3 = self._iterations_plot()
        ax3.set_xlabel('Learning Rate')
        ax3.set_ylabel('Iterations')
        title = 'Iterations by Learning Rate and Precision ' + r'$\epsilon$' + " = " + str(round(self._precision,5)) 
        ax3.set_title("\n".join(wrap(title, 30)), color='k')

        # Suptitle and wrap up
        suptitle = self._alg + ' Diagnostics'
        fig.suptitle(suptitle, y=1.05)
        fig.tight_layout()
        plt.close()
        return(fig)

    def _diagnostic_full(self):

        fig = plt.figure(figsize=(16,8))
        gs = fig.add_gridspec(2,4)

        # Cost Line Plot
        ax0 = fig.add_subplot(gs[0,0:2])
        ax0 = self._cost_line()
        ax0.set_facecolor('w')
        ax0.tick_params(colors='k')
        ax0.xaxis.label.set_color('k')
        ax0.yaxis.label.set_color('k')        
        ax0.set_xlabel('Iterations')
        ax0.set_ylabel('Cost')
        title = 'Costs by Learning Rate and Precision ' + r'$\epsilon$' + " = " + str(round(self._precision,5)) 
        ax0.set_title("\n".join(wrap(title, 30)), color='k')

        # Validation MSE Line Plot
        ax1 = fig.add_subplot(gs[0,2:4])
        ax1 = self._mse_line()
        ax0.set_facecolor('w')
        ax0.tick_params(colors='k')
        ax0.xaxis.label.set_color('k')
        ax0.yaxis.label.set_color('k')
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Mean Squared Error')
        title = 'Validation Set MSE by Learning Rate and Precision ' + r'$\epsilon$' + " = " + str(round(self._precision,5)) 
        ax1.set_title("\n".join(wrap(title, 30)), color='k')

        # Cost Bar Plot
        ax2 = fig.add_subplot(gs[1,0])
        ax2 = self._cost_bar()
        ax2.set_xlabel('Learning Rate')
        ax2.set_ylabel('Final Cost')
        title = 'Final Costs by Learning Rate and Precision ' + r'$\epsilon$' + " = " + str(round(self._precision,5)) 
        ax2.set_title("\n".join(wrap(title, 30)), color='k')

        # MSE Bar Plot
        ax3 = fig.add_subplot(gs[1,1])
        ax3 = self._cost_bar()
        ax3.set_xlabel('Learning Rate')
        ax3.set_ylabel('Final Mean Squared Error')
        title = 'Final Validation Set MSE by Learning Rate and Precision ' + r'$\epsilon$' + " = " + str(round(self._precision,5)) 
        ax3.set_title("\n".join(wrap(title, 30)), color='k')

        # Duration Plot
        ax4 = fig.add_subplot(gs[1,2])
        ax4 = self._duration_plot()  
        ax4.set_xlabel('Learning Rate')
        ax4.set_ylabel('Duration (ms)')
        title = 'Computation Time by Learning Rate and Precision ' + r'$\epsilon$' + " = " + str(round(self._precision,5)) 
        ax4.set_title("\n".join(wrap(title, 30)), color='k')

        # Iterations Plot
        ax5 = fig.add_subplot(gs[1,3])
        ax5 = self._iterations_plot()
        ax5.set_xlabel('Learning Rate')
        ax5.set_ylabel('Iterations')
        title = 'Iterations by Learning Rate and Precision ' + r'$\epsilon$' + " = " + str(round(self._precision,5)) 
        ax5.set_title("\n".join(wrap(title, 30)), color='k')

        # Suptitle and wrap up
        suptitle = self._alg + ' Diagnostics'
        fig.suptitle(suptitle, y=1.05)
        fig.tight_layout()
        plt.close()
        return(fig)


    def evaluate(self, X, y, theta, alpha, X_val=None, y_val=None, precision=0.001,
               stop_measure='j', stop_metric='a', maxiter=10000):

        # Run search
        self._searches = [self.search(X=X,y=y,X_val=X_val, y_val=y_val, theta=theta, alpha=ALPHA, 
                    precision=precision, stop_measure=stop_measure, stop_metric=stop_metric, 
                    maxiter=maxiter)
                    for ALPHA in alpha]

        if X_val is None:
            diagnostic = self._diagnostic_brief()   
        else:
            diagnostic = self._diagnostic_full()
          
        eval = self._evaluation_summary()                
        return(eval, diagnostic)
    

