
# %%
# =========================================================================== #
#                             Gradient Descent                                #
# =========================================================================== #
import inspect
import os
import sys

from IPython.display import HTML

import itertools
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
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
        self._stop = False
        self._J_history = []
        self._theta_history = []
        self._g_history = []
        self._epochs = []
        self._X = []
        self._y = []
        self._alpha = None
        self._precision = None

    def _hypothesis(self, X, theta):
        return(X.dot(theta))

    def _error(self, h, y):
        return(h-y)

    def _cost(self, e):
        return(1/2 * np.mean(e**2))

    def cost_mesh(self, X, y, THETA):
        return(np.sum((X.dot(THETA) - y)**2)/(2*len(y)))

    def _gradient(self, X, e):
        return(X.T.dot(e)/X.shape[0])

    def _update(self, alpha, theta, gradient):
        return(theta-(alpha * gradient))

    def encode_labels(self, X, y):
        le = LabelEncoder()
        X = X.apply(le.fit_transform)
        y = le.fit_transform(y)
        return(X, y)

    def scale(self, X, y, scaler='minmax', bias=False):
        # Select scaler
        if scaler == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()

        # Combine X and y into a dataframe
        y = pd.DataFrame({'y': y})
        df = pd.concat([X, y], axis=1)

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

    def _finished(self, J, J_prior, iter, maxiter, precision):
        if maxiter:
            if iter == maxiter:
                self._stop = True
                return(True)
        elif abs(J-J_prior) < precision:
            self.converged = True
            return(True)
        else:
            return(False)

    def search(self, X, y, theta, alpha=0.01, maxiter=0, precision=0.001):

        self._stop = False
        self._J_history = []
        self._theta_history = []
        self._g_history = []
        self._epochs = []
        self._X = X
        self._y = y
        self._alpha = alpha
        self._precision = precision
        J = math.inf
        epoch = 0
        g = np.repeat(1, X.shape[1])

        while not self._stop:
            epoch += 1
            J_prior = J

            h = self._hypothesis(X, theta)
            e = self._error(h, y)
            J = self._cost(e)
            g = self._gradient(X, e)

            self._theta_history.append(theta.tolist())
            self._J_history.append(J)
            self._g_history.append(g.tolist())
            self._epochs.append(epoch)

            if self._finished(J, J_prior, epoch, maxiter, precision):
                break

            theta = self._update(alpha, theta, g)

        d = dict()
        d['X'] = X
        d['y'] = y
        d['epochs'] = self._epochs
        d['theta_history'] = self._theta_history
        d['J_history'] = self._J_history
        d['gradient_history'] = self._g_history

        return(d)

    def report(self, n=0):

        # Epochs dataframe
        epochs = pd.DataFrame(self._epochs, columns=['Epoch'])
        costs = pd.DataFrame(self._J_history, columns=['Cost'])

        # Create thetas dataframe columns
        n_thetas = len(self._theta_history[0])
        thetas = pd.DataFrame()
        for i in range(n_thetas):
            colname = 'theta_' + str(i)
            theta = [item[i] for item in self._theta_history]
            df_theta = pd.DataFrame(theta, columns=[colname])
            thetas = pd.concat([thetas, df_theta], axis=1)

        # Create gradients dataframe columns
        n_gradients = len(self._g_history[0])
        gradients = pd.DataFrame()
        for i in range(n_gradients):
            colname = 'gradient_' + str(i)
            gradient = [item[i] for item in self._g_history]
            df_gradient = pd.DataFrame(gradient, columns=[colname])
            gradients = pd.concat([gradients, df_gradient], axis=1)

        result = pd.concat([epochs, thetas, costs, gradients], axis=1)
        if n:
            result = result.iloc[0:n]
        return(result)

    def display_costs(self, costs=None, interval=100, path=None, fps=60):

        # Sets costs 
        if costs is None:
            costs = self._J_history

        # Establish figure and axes objects and lines and points
        fig, ax = plt.subplots()
        line, = ax.plot([], [], 'r', lw=1.5)
        point, = ax.plot([], [], 'bo')
        epoch_display = ax.text(.7, 0.9, '',
                                transform=ax.transAxes, fontsize=16)
        J_display = ax.text(.7, 0.8, '', transform=ax.transAxes, fontsize=16)

        # Plot cost line
        ax.plot(np.arange(len(costs)), costs, c='r')

        # Set labels and title
        ax.set_xlabel(self._unit)
        ax.set_ylabel(r'$J(\theta)$')
        ax.set_title(self._alg)

        def init():
            line.set_data([], [])
            point.set_data([], [])
            epoch_display.set_text('')
            J_display.set_text('')

            return(line, point, epoch_display, J_display)

        def animate(i):
            # Animate points
            point.set_data(i, costs[i])

            # Animate value display
            epoch_display.set_text(self._unit + " = " + str(i+1))
            J_display.set_text(r'     $J(\theta)=$' +
                               str(round(costs[i], 4)))

            return(line, point, epoch_display, J_display)

        display = animation.FuncAnimation(fig, animate, init_func=init,
                                          frames=len(costs), interval=interval,
                                          blit=True)
        if path:
            display.save(path, writer='imagemagick', fps=fps)

        plt.close(fig)
        return(display)

    def surface(self, theta_history, cost_history, interval=50, path=None, fps=60):
        '''Plots surface plot on two dimensional problems only 
        '''
        if len(theta_history[0]) > 2:
            raise Exception("Surface only works on 2-dimensional X arrays")        

        # Designate plot area
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create the x=theta0, y=theta1 grid for plotting
        theta0 = [item[0] for item in theta_history]
        theta1 = [item[1] for item in theta_history]

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
        display = ax.text2D(0.2,0.95, '', transform=ax.transAxes)

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
            line3d.set_3d_properties(cost_history[:i])

            # Animate 3d points
            point3d.set_data(theta0[i], theta1[i])
            point3d.set_3d_properties(cost_history[i])

            # Animate 2d Line
            line2d.set_data(theta0[:i], theta1[:i])
            line2d.set_3d_properties(0)

            # Animate 2d points
            point2d.set_data(theta0[i], theta1[i])
            point2d.set_3d_properties(0)

            # Update display
            display.set_text('Epoch: '+ str(i) + r'$\quad\theta_0=$ ' +
                            str(round(theta0[i],3)) + r'$\quad\theta_1=$ ' + str(round(theta1[i],3)) +
                            r'$\quad J(\theta)=$ ' + str(np.round(cost_history[i], 5)))


            return(line3d, point3d, line2d, point2d, display)

        # create animation using the animate() function
        surface_ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(theta0),
                                            interval=interval, blit=True)
        if path:
            surface_ani.save(path, writer='imagemagick', fps=fps)
        plt.close(fig)
        return(surface_ani)

    def show_fit(self, interval=50, path=None, fps=60):
        '''Shows animation of regression line fit for 2D X Vector 
        '''
        if self._X.shape[1] > 2:
            raise Exception("Show_fit only works on 2-dimensional X arrays")    

        # Extract data for plotting
        X = self._X
        y = self._y
        x = [item[1] for item in X]
        theta0 = [item[0] for item in self._theta_history]
        theta1 = [item[1] for item in self._theta_history]

        # Render scatterplot
        fig, ax = plt.subplots()
        sns.scatterplot(x=x, y=y, ax=ax)

        # Initialize line
        line, = ax.plot([],[],'r-', lw=2)

        # Annotations and labels
        display = ax.text(0.1, 0.9, '', transform=ax.transAxes)
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.set_title('Model Fit by Gradient Descent')

        # Build the empty line plot at the initiation of the animation
        def init():
            line.set_data([],[])
            display.set_text('')
            return (line, display,)

        # Animate the regression line as it converges
        def animate(i):

            # Animate Line
            y=X.dot(self._theta_history[i])
            line.set_data(x,y)

            # Animate text
            display.set_text('Epoch: '+ str(i) + r'$\quad\theta_0=$ ' +
                            str(round(theta0[i],3)) + r'$\quad\theta_1=$ ' + str(round(theta1[i],3)) +
                            r'$\quad J(\theta)=$ ' + str(round(self._J_history[i], 3)))
            return (line, display)

        # create animation using the animate() function
        line_gd = animation.FuncAnimation(fig, animate, init_func=init, frames=len(self._theta_history),
                                            interval=interval, blit=True)
        if path:
            line_gd.save(path, writer='imagemagick', fps=fps)
        plt.close(fig)
        return(line_gd)
        



class BGD(GradientDescent):
    def __init__(self):
        pass


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
        self._iterations = []
        self._iterations_smooth = []
        self._X = []
        self._y = []
        self._X_i = []
        self._y_i = []
        self._alpha = None
        self._precision = None            
        self._check_grad = None        

    def _shuffle(self, X, y):
        y = np.expand_dims(y, axis=1)
        z = np.append(arr=X, values=y, axis=1)
        np.random.shuffle(z)
        X = np.delete(z, z.shape[1]-1, axis=1)
        y = z[:, z.shape[1]-1]
        return(X, y)

    def search(self, X, y, theta, alpha=0.01, maxiter=0,
               precision=0.001, check_grad=100):
        self._J_history = []
        self._J_history_smooth = []
        self._theta_history = []
        self._theta_history_smooth = []
        self._h_history = []
        self._g_history = []
        self._stop = False
        self._epochs = []
        self._epochs_smooth = []
        self._iterations = []
        self._iterations_smooth = []
        self._X = X
        self._y = y
        self._X_i = []
        self._y_i = []
        self._alpha = alpha
        self._precision = precision            
        self._check_grad = check_grad

        J = math.inf
        J_total = 0
        epoch = 0
        iteration = 0
        g = np.repeat(1, X.shape[1])
        while not self._stop:
            epoch += 1
            J_prior = J
            X, y = self._shuffle(X, y)

            for x_i, y_i in zip(X, y):
                iteration += 1

                h = self._hypothesis(x_i, theta)
                e = self._error(h, y_i)
                J = self._cost(e)
                J_total += J
                g = self._gradient(x_i, e)

                self._J_history.append(J)
                self._theta_history.append(theta.tolist())
                self._h_history.append(h)
                self._g_history.append(g.tolist())
                self._epochs.append(epoch)
                self._iterations.append(iteration)
                self._X_i.append(x_i)
                self._y_i.append(y_i)

                if maxiter:
                    if iteration == maxiter:
                        self._epochs_smooth.append(epoch)
                        self._iterations_smooth.append(iteration)
                        self._J_history_smooth.append(J)
                        self._theta_history_smooth.append(theta)
                        self._stop = True
                        break

                if iteration % check_grad == 0:
                    J_smooth = J_total / check_grad
                    self._epochs_smooth.append(epoch)
                    self._iterations_smooth.append(iteration)
                    self._J_history_smooth.append(J_smooth)
                    self._theta_history_smooth.append(theta)
                    if abs(J_prior - J_smooth) < precision:
                        self._stop = True
                        break
                    J_prior = J_smooth
                    J_total = 0

                theta = self._update(alpha, theta, g)

        d = dict()
        d['X'] = X
        d['y'] = y
        d['X_i'] = self._X_i
        d['y_i'] = self._y_i
        d['epochs'] = self._epochs
        d['epochs_smooth'] = self._epochs_smooth
        d['iterations'] = self._iterations
        d['iterations_smooth'] = self._iterations_smooth
        d['h_history'] = self._h_history
        d['theta_history'] = self._theta_history
        d['theta_history_smooth'] = self._theta_history_smooth
        d['J_history'] = self._J_history
        d['J_history_smooth'] = self._J_history_smooth
        d['gradient_history'] = self._g_history

        return(d)

    def report_detail(self, n=0):

        epochs = pd.DataFrame(self._epochs, columns=['Epoch'])
        iterations = pd.DataFrame(self._iterations, columns=['Iteration'])
        y = pd.DataFrame(self._y_i, columns=['y'])
        h = pd.DataFrame(self._h_history, columns=['h'])
        J = pd.DataFrame(self._J_history, columns=['Cost'])

        # Create thetas dataframe columns
        n_thetas = len(self._theta_history[0])
        thetas = pd.DataFrame()
        for i in range(n_thetas):
            colname = 'theta_' + str(i)
            theta = [item[i] for item in self._theta_history]
            df_theta = pd.DataFrame(theta, columns=[colname])
            thetas = pd.concat([thetas, df_theta], axis=1)

        # Create gradients dataframe columns
        n_gradients = len(self._g_history[0])
        gradients = pd.DataFrame()
        for i in range(n_gradients):
            colname = 'gradient_' + str(i)
            gradient = [item[i] for item in self._g_history]
            df_gradient = pd.DataFrame(gradient, columns=[colname])
            gradients = pd.concat([gradients, df_gradient], axis=1)

        result = pd.concat([epochs, iterations, thetas, y,
                            h, J, gradients], axis=1, sort=False)
        if n:
            result = result.iloc[0:n]
        return(result)

    def report_smooth(self, n=0):

        result = pd.DataFrame(
            {'Epoch': self._epochs_smooth,
             'Iteration': self._iterations_smooth,
             'Cost': self._J_history_smooth
             }
        )
        if n:
            result = result.iloc[0:n]
        return(result)

    def report(self, n=0, smooth=False):
        if smooth:
            return(self.report_smooth(n))
        else:
            return(self.report_detail(n))


# %%
from data import data
df = data.read()
df = df[['Area', 'SalePrice']]
df = df.sample(n=100, random_state=50, axis=0)
df = df.reset_index(drop=True)
X = df[['Area']]
y = df['SalePrice']

#%%
gd = SGD()
np.random.seed(50)
X, y = gd.encode_labels(X, y)
X, y = gd.scale(X, y, 'minmax', bias=True)
theta = np.array([-1,-1])
search = gd.search(X, y, theta, check_grad=100)
print(search['theta_history_smooth'][0])

#%%
ani = gd.surface(theta_history = search['theta_history_smooth'], 
           cost_history = search['J_history_smooth'])
HTML(ani.to_jshtml())


