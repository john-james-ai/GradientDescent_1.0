
# %%
# =========================================================================== #
#                             Gradient Descent                                #
# =========================================================================== #
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
from sklearn import linear_model
from sklearn.datasets import make_regression
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from textwrap import wrap
rcParams['animation.embed_limit'] = 60
rc('animation', html='jshtml')
rc
# --------------------------------------------------------------------------- #
#                       Gradient Descent Base Class                           #
# --------------------------------------------------------------------------- #


class GradientDescent:
    '''Base class for Gradient Descent'''

    def __init__(self):
        self._alg = "Batch Gradient Descent"
        self._unit = "Epoch"
        self._J_history = []
        self._theta_history = []
        self._g_history = []
        self._mse_history = []
        self._iterations = []
        self._precision = 0.001
        self._alpha = 0.01
        self._searches = []

    def _hypothesis(self, X, theta):
        return(X.dot(theta))

    def _error(self, h, y):
        return(h-y)

    def _cost(self, e):
        return(1/2 * np.mean(e**2))

    def _mse(self, X, y, theta):
        h = self._hypothesis(X, theta)
        e = self._error(h, y)
        return(np.mean(e**2))

    def cost_mesh(self, X, y, THETA):
        return(np.sum((X.dot(THETA) - y)**2)/(2*len(y)))

    def _gradient(self, X, e):
        return(X.T.dot(e)/X.shape[0])

    def _update(self, alpha, theta, gradient):
        return(theta-(alpha * gradient))

    def _encode_labels(self, X, y):
        le = LabelEncoder()
        if isinstance(X, pd.core.frame.DataFrame):
            X = X.apply(le.fit_transform)
        else:
            x = le.fit_transform(X)
        if isinstance(y, pd.core.frame.DataFrame):
            y = y.apply(le.fit_transform)
        else:
            y = le.fit_transform(y)
        return(X, y)

    def _scale(self, X, y, scaler='minmax', bias=False):
        # Select scaler
        if scaler == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()

        # Put X and y into a dataframe
        if isinstance(X, pd.core.frame.DataFrame):                
            y = pd.DataFrame({'y': y})
            df = pd.concat([X, y], axis=1)
        else:
            df = pd.DataFrame({'X':X, 'y':y})

        # Scale then return X and y
        df_scaled = scaler.fit_transform(df)
        df = pd.DataFrame(df_scaled, columns=df.columns)

        # Add bias term
        if bias:
            X0 = pd.DataFrame(np.ones((df.shape[0], 1)), columns=['X0'])
            df = pd.concat([X0, df], axis=1)
        X = df.drop(columns=['y']).values
        y = df['y']
        return(X, y)

    def prep_data(self, X,y, scaler='minmax', bias=True):        
        X, y = self._encode_labels(X,y)
        X, y = self._scale(X,y, scaler, bias)
        return(X,y)        

    def _zeros(self, d):
        for k, v in d.items():
            for i, j in v.items():
                if type(j) is float or type(j) is int:
                    if (j<10**-10) & (j>0):
                        j = 10**-10
                    elif (j>-10**-10) & (j<0):
                        j = -10**-10
                else: 
                    for l in j:
                        if (l<10**-10) & (l>0):
                            l = 10**-10
                        elif (l>-10**-10) & (l<0):
                            l = -10**-10
        return(d)

    def _finished_grad(self):

        if self._stop_metric == 'a':
            result = (all(abs(y-x)<self._precision for x,y in zip(self._state['g']['prior'], self._state['g']['current'])))
        else:    
            result = (all(abs((y-x)/x)<self._precision for x,y in zip(self._state['g']['prior'], self._state['g']['current'])))
        self._state['g']['prior'] = self._state['g']['current']
        return(result)

    
    def _finished_v(self):

        if self._stop_metric == 'a':
            result = (abs(self._state['v']['current']-self._state['v']['prior']) < self._precision)                        
        else:
            result = (abs((self._state['v']['current']-self._state['v']['prior'])/self._state['v']['prior']) < self._precision)
        self._state['v']['prior'] = self._state['v']['current']
        return(result)

    def _finished_J(self):

        if self._stop_metric == 'a':
            result = (abs(self._state['j']['current']-self._state['j']['prior']) < self._precision)                        
        else:
            result = (abs((self._state['j']['current']-self._state['j']['prior'])/self._state['j']['prior']) < self._precision)
        self._state['j']['prior'] = self._state['j']['current']
        return(result)


    def _maxed_out(self):
        if self._max_iterations:
            if self._iteration == self._max_iterations:
                return(True)  

    def _finished(self):
        self._state = self._zeros(self._state)
        if self._maxed_out():
            return(True)
        elif self._stop_measure == 'j':
            return(self._finished_J())
        elif self._stop_measure == 'g':
            return(self._finished_grad())    
        else:
            return(self._finished_v())

  

    def search(self, X, y, theta, X_val=None, y_val=None, 
               alpha=0.01, maxiter=0, precision=0.001,
               stop_measure='j', stop_metric='a', n_val=0):

        self._J_history = []
        self._theta_history = []
        self._g_history = []
        self._iterations = []
        self._mse_history = []
        self._alpha = alpha
        self._precision = precision
        self._stop_measure = stop_measure
        self._stop_metric = stop_metric 
        self._iteration = 0
        self._max_iterations = maxiter
        self._X = X
        self._y = y
        self._state = {'j':{'prior':10**10, 'current':1},
                       'g':{'prior':np.repeat(10**10, X.shape[1]), 'current':np.repeat(1, X.shape[1])},
                       'v':{'prior':10**10, 'current':1}}
      
        start = datetime.datetime.now()

        while not self._finished():
            self._iteration += 1

            # Compute the costs and validation set error (if required)
            h = self._hypothesis(X, theta)
            e = self._error(h, y)
            J = self._cost(e)
            g = self._gradient(X, e)

            if X_val is not None:
                mse = self._mse(X_val, y_val, theta)
                self._mse_history.append(mse)
                self._state['v']['current'] = mse
            
            # Save current computations in state
            self._state['j']['current'] = J
            self._state['g']['current'] = g.tolist()
            
            # Save iterations, costs and thetas in history 
            self._theta_history.append(theta.tolist())
            self._J_history.append(J)            
            self._g_history.append(g.tolist())
            self._iterations.append(self._iteration)

            theta = self._update(alpha, theta, g)

        end = datetime.datetime.now()
        diff = end-start

        d = dict()
        d['alg'] = self._alg
        d['X'] = X
        d['y'] = y
        d['elapsed_ms'] = diff.total_seconds() * 1000
        d['alpha'] = alpha
        d['precision'] = precision
        d['stop_measure'] = self._stop_measure
        d['stop_metric'] = self._stop_metric
        d['iterations'] = self._iterations
        d['theta_history'] = self._theta_history
        d['J_history'] = self._J_history
        d['mse_history'] = self._mse_history
        d['gradient_history'] = self._g_history

        return(d)

    def _todf(self, x, stub):
        n = len(x[0])
        df = pd.DataFrame()
        for i in range(n):
            colname = stub + str(i)
            vec = [item[i] for item in x]
            df_vec = pd.DataFrame(vec, columns=[colname])
            df = pd.concat([df, df_vec], axis=1)
        return(df)

    def report(self, n=0):

        # Epochs dataframe
        iterations = pd.DataFrame(self._iterations, columns=[self._unit])
        costs = pd.DataFrame(self._J_history, columns=['Cost'])
        thetas = self._todf(self._theta_history, stub='theta_')
        gradients = self._todf(self._g_history, stub='gradient_')
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
    

class BGD(GradientDescent):
    def __init__(self):
        self._alg = "Batch Gradient Descent"
        self._unit = "Epoch"
        self._J_history = []
        self._theta_history = []
        self._h_history = []
        self._g_history = []
        self._stop = False
        self._epochs = []
        self._X = []
        self._y = []
        self._iteration = 0
        self._max_iterations = 0
        self._stop_measure = "j"
        self._precision = 0.001
        self._stop_metric = 'a'
        self._searches = []


class SGD(GradientDescent):

    def __init__(self):
        self._alg = "Stochastic Gradient Descent"
        self._unit = "Iteration"
        self._J_history = []
        self._J_history_smooth = []
        self._theta_history = []
        self._theta_history_smooth = []
        self._h_history = []
        self._g_history = []
        self._stop = False
        self._epochs = []
        self._epochs_smooth = []
        self._iteration = 0
        self._max_iterations = 0
        self._iterations = []
        self._iterations_smooth = []
        self._X = []
        self._y = []
        self._X_i = []
        self._y_i = []
        self._X_i_smooth = []
        self._y_i_smooth = []               
        self._stop_measure = "j"
        self._precision = 0.001
        self._stop_metric = 'a'
        self._searches = []


    def _shuffle(self, X, y):
        y = np.expand_dims(y, axis=1)
        z = np.append(arr=X, values=y, axis=1)
        np.random.shuffle(z)
        X = np.delete(z, z.shape[1]-1, axis=1)
        y = z[:, z.shape[1]-1]
        return(X, y)

    def search(self, X, y, theta, alpha=0.01, maxiter=0, 
               stop_measure='j', stop_metric='a',
               precision=0.0001, check_grad=100):
        self._J_history = []
        self._J_history_smooth = []        
        self._theta_history = []
        self._theta_history_smooth = []
        self._h_history = []
        self._h_history_smooth = []
        self._g_history = []
        self._g_history_smooth = []
        self._stop = False
        self._epochs = []
        self._epochs_smooth = []
        self._iteration = 0
        self._max_iterations = maxiter
        self._iterations = []
        self._iterations_smooth = []
        self._X = X
        self._y = y
        self._X_i = []
        self._y_i = []
        self._X_i_smooth = []
        self._y_i_smooth = []    

        self._stop_measure = stop_measure
        self._precision = precision
        self._stop_metric = stop_metric  

        epoch = 0
        g_prior = np.repeat(1, X.shape[1])
        J_prior = math.inf
        J_total = 0

        start = datetime.datetime.now()

        while not self._stop:
            epoch += 1
            X, y = self._shuffle(X, y)

            for x_i, y_i in zip(X, y):
                self._iteration += 1

                h = self._hypothesis(x_i, theta)
                e = self._error(h, y_i)
                J = self._cost(e)
                J_total += J
                g = self._gradient(x_i, e)

                self._h_history.append(h)
                self._J_history.append(J)
                self._theta_history.append(theta.tolist())
                self._g_history.append(g.tolist())
                self._epochs.append(epoch)
                self._iterations.append(self._iteration)
                self._X_i.append(x_i)
                self._y_i.append(y_i)

                if self._iteration % check_grad == 0:
                    J_smooth = J_total / check_grad
                    g_smooth = g
                    self._h_history_smooth.append(h)
                    self._J_history_smooth.append(J_smooth)
                    self._theta_history_smooth.append(theta.tolist())
                    self._g_history_smooth.append(g_smooth.tolist())
                    self._epochs_smooth.append(epoch)
                    self._iterations_smooth.append(self._iteration)
                    self._X_i_smooth.append(x_i)
                    self._y_i_smooth.append(y_i)                    

                    if self._maxed_out(self._iteration):
                        self._stop = True
                    elif self._stop_measure == 'j':
                        self._stop = self._finished(J_smooth, J_prior)
                    elif self._stop_measure == 'g':
                        self._stop = self._finished(g_smooth, g_prior)
                    if self._stop:
                        break
                    
                    J_prior = J_smooth
                    g_prior = g_smooth
                    J_total = 0

                theta = self._update(alpha, theta, g)

        end = datetime.datetime.now()
        diff = end-start

        d = dict()
        d['alg'] = self._alg
        d['X'] = X
        d['y'] = y
        d['X_i'] = self._X_i
        d['y_i'] = self._y_i
        d['elapsed_ms'] = diff.total_seconds() * 1000
        d['alpha'] = alpha
        d['epochs'] = self._epochs
        d['epochs_smooth'] = self._epochs_smooth
        d['iterations'] = self._iterations
        d['iterations_smooth'] = self._iterations_smooth
        d['h_history'] = self._h_history
        d['theta_history'] = self._theta_history
        d['theta_history_smooth'] = self._theta_history_smooth
        d['J_history'] = self._J_history
        d['J_history_smooth'] = self._J_history_smooth
        d['g_history'] = self._g_history
        d['g_history_smooth'] = self._g_history_smooth
        

        return(d)

    def report_detail(self, data,n=0):

        epochs = pd.DataFrame(data['Epoch'], columns=['Epoch'])
        iterations = pd.DataFrame(data['Iteration'], columns=['Iteration'])
        y = pd.DataFrame(data['y'], columns=['y'])
        h = pd.DataFrame(data['h'], columns=['h'])
        J = pd.DataFrame(data['J'], columns=['Cost'])

        # Create thetas dataframe columns
        thetas = self._todf(data['Theta'], stub='theta_')
        gradients = self._todf(data['g'], stub='gradient_')
        
        result = pd.concat([epochs, iterations, thetas, y,
                            h, J, gradients], axis=1, sort=False)
        if n:
            result = result.iloc[0:n]
        return(result)

    def report(self, n=0, smooth=False):
        if smooth:
            data = {'Epoch': self._epochs_smooth,
                    'Iteration': self._iterations_smooth,
                    'Theta': self._theta_history_smooth,
                    'h': self._h_history_smooth,
                    'y': self._y_i_smooth,
                    'J': self._J_history_smooth,
                    'g': self._g_history_smooth
                    }
            return(self.report_detail(data, n))
        else:
            data = {'Epoch': self._epochs,
                    'Iteration': self._iterations,
                    'Theta': self._theta_history,
                    'h': self._h_history,
                    'y': self._y_i,
                    'J': self._J_history,
                    'g': self._g_history
                    }
            return(self.report_detail(data, n))

class MBGD(GradientDescent):

    def __init__(self):
        self._alg = "Mini-Batch Gradient Descent"
        self._unit = "Batch"
        self._J_history = []
        self._theta_history = []
        self._g_history = []
        self._stop = False
        self._epochs = []
        self._X = []
        self._y = []
        self._iteration = 0
        self._iterations = []
        self._max_iterations = 0
        self._stop_measure = "j"
        self._precision = 0.001
        self._stop_metric = 'a'

    def _shuffle(self, X, y):
        y = np.expand_dims(y, axis=1)
        z = np.append(arr=X, values=y, axis=1)
        np.random.shuffle(z)
        X = np.delete(z, z.shape[1]-1, axis=1)
        y = z[:, z.shape[1]-1]
        return(X, y)

    def _get_batch(self, X, y, batch):
        X = X[batch:batch+self._batch_size,:]
        y = y[batch:batch+self._batch_size]
        return(X, y)

    def search(self, X, y, theta, alpha=0.01, batch_size = 50,
               stop_measure = 'j', stop_metric = 'a',
               maxiter=0, precision=0.0001):
        self._J_history = []
        self._theta_history = []
        self._g_history = []
        self._stop = False
        self._epochs = []
        self._batches = []
        self._iterations = []
        self._iteration = 0
        self._stop_measure = stop_measure
        self._stop_metric = stop_metric
        self._max_iterations = maxiter
        self._batch_size = batch_size
        self._X = X
        self._y = y        

        epoch = 0
        n_batches = math.ceil(X.shape[0] / batch_size)
        J_prior = math.inf
        g_prior = np.repeat(1, X.shape[1])

        start = datetime.datetime.now()

        while not self._stop:
            epoch += 1
            X, y = self._shuffle(X, y)

            for batch in np.arange(n_batches):
                X_batch, y_batch = self._get_batch(X,y, batch)
                self._iteration += 1
                h = self._hypothesis(X_batch, theta)
                e = self._error(h, y_batch)
                J = self._cost(e)
                g = self._gradient(X_batch, e)

                self._iterations.append(self._iteration)
                self._theta_history.append(theta.tolist())
                self._J_history.append(J)
                self._g_history.append(g.tolist())
                self._epochs.append(epoch)
                self._batches.append(batch+1)

                if self._maxed_out(self._iteration):
                    self._stop = True
                elif self._stop_measure == 'j':
                    self._stop = self._finished(J, J_prior)
                elif self._stop_measure == 'g':
                    self._stop = self._finished(g, g_prior)
                if self._stop:
                    break

                theta = self._update(alpha, theta, g)                
                J_prior = J
                g_prior = g

        end = datetime.datetime.now()
        diff = end-start

        d = dict()
        d['alg'] = self._alg
        d['X'] = X
        d['y'] = y
        d['elapsed_ms'] = diff.total_seconds() * 1000
        d['alpha'] = alpha
        d['batches'] = self._batches
        d['epochs'] = self._epochs
        d['iterations'] = self._iterations
        d['theta_history'] = self._theta_history
        d['J_history'] = self._J_history
        d['gradient_history'] = self._g_history

        return(d)

    def report(self, n=0):

        iterations = pd.DataFrame(self._iterations, columns=['Iteration'])
        epochs = pd.DataFrame(self._epochs, columns=['Epoch'])
        batches = pd.DataFrame(self._batches, columns=['Batch'])
        J = pd.DataFrame(self._J_history, columns=['Cost'])
        thetas = self._todf(self._theta_history, stub='theta_')
        gradients = self._todf(self._g_history, stub='gradient_')

        result = pd.concat([iterations, epochs, batches, thetas, 
                            J, gradients], axis=1, sort=False)
        if n:
            result = result.iloc[0:n]
        return(result)
# %%
# from data import data
# ames = data.read()
# ames = ames[['Area', 'SalePrice']]
# df = ames.sample(n=100, random_state=50, axis=0)
# test = ames.loc[~ames.index.isin(df.index),:]
# test = test.dropna()
# df = df.reset_index(drop=True)
# test = test.reset_index(drop=True)
# X = df[['Area']]
# y = df['SalePrice']
# X_val = test[['Area']]
# y_val = test[['SalePrice']]


# gd = BGD()
# np.random.seed(50)
# X, y = gd.encode_labels(X, y)
# X, y = gd.scale(X, y, 'minmax', bias=True)
# X_val, y_val = gd.encode_labels(X_val, y_val)
# X_val, y_val = gd.scale(X_val, y_val, 'minmax', bias=True)
# theta = np.array([-1,0])
# #gd.search(X,y, theta)
# alpha = np.arange(0.01,0.1,0.02)
# ts, p = gd.tune(X, y, theta, X_val = X_val, y_val=y_val, maxiter = 100, alpha=alpha, stop_measure=None)
# ts
# p
# plt.show()
# report = gd.report(n=20)
# report



# %%
# ts
# ani = gd.display_costs(costs= search['J_history'], interval=100)
# HTML(ani.to_jshtml())
# rc('animation', html='jshtml')
# rc
# ani = gd.surface(theta_history = search['theta_history'], 
#                  cost_history=search['J_history'], interval=100)
# HTML(ani.to_jshtml())
# rc('animation', html='jshtml')
# rc
# ani = gd.show_fit(theta_history = search['theta_history'], 
#                   J_history = search['J_history'],
#                   interval=100)
# HTML(ani.to_jshtml())


