
# %%
# =========================================================================== #
#                             Gradient Descent                                #
# =========================================================================== #
import inspect
import os
import sys

from IPython.display import HTML

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
# --------------------------------------------------------------------------- #
#                       Gradient Descent Base Class                           #
# --------------------------------------------------------------------------- #


class GradientDescent:
    '''Base class for Gradient Descent'''

    def __init__(self):
        self._stop = False
        self._J_history = []
        self._theta_history = []
        self._g_history = []
        self._epochs = []
        pass

    def _hypothesis(self, X, theta):
        return(X.dot(theta))

    def _error(self, h, y):
        return(h-y)

    def _cost(self, e):
        return(1/2 * np.mean(e**2))

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

    def search(self, X, y, theta, alpha=0.001, maxiter=0, precision=0.0001):

        self._stop = False
        self._J_history = []
        self._theta_history = []
        self._g_history = []
        self._epochs = []
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

    def display_costs(self, costs=None, title="Gradient Descent",
                      xaxis_label="Epoch", interval=100, path=None):

        if costs is None:
            costs = self._J_history

        fig, ax = plt.subplots()
        line, = ax.plot([], [], 'r', lw=1.5)
        point, = ax.plot([], [], 'bo')
        epoch_display = ax.text(.7, 0.9, '',
                                transform=ax.transAxes, fontsize=16)
        J_display = ax.text(.7, 0.8, '', transform=ax.transAxes, fontsize=16)

        # Concatenate np array of thetas into a single list
        ax.plot(np.arange(len(costs)), costs, c='r')
        ax.set_xlabel(xaxis_label, fontsize=20)
        ax.set_ylabel(r'$J(\theta)$', fontsize=20)
        ax.set_title(title, fontsize=24)

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
            epoch_display.set_text(xaxis_label + " = " + str(i+1))
            J_display.set_text(r'     $J(\theta)=$' +
                               str(round(costs[i], 4)))

            return(line, point, epoch_display, J_display)

        display = animation.FuncAnimation(fig, animate, init_func=init,
                                          frames=len(costs), interval=interval,
                                          blit=True)
        if path:
            display.save(path, writer='imagemagick', fps=60)

        plt.close(fig)
        return(display)


class BGD(GradientDescent):
    def __init__(self):
        pass


class SGD(GradientDescent):

    def __init__(self):
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

    def _shuffle(self, X, y):
        y = np.expand_dims(y, axis=1)
        z = np.append(arr=X, values=y, axis=1)
        np.random.shuffle(z)
        X = np.delete(z, z.shape[1]-1, axis=1)
        y = z[:, z.shape[1]-1]
        return(X, y)

    def search(self, X, y, theta, alpha=0.001, maxiter=0,
               precision=0.0001, check_grad=1000):
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
                self._X.append(x_i)
                self._y.append(y_i)

                if maxiter:
                    if iteration == maxiter:
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
        d['X'] = self._X
        d['y'] = self._y
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
        y = pd.DataFrame(self._y, columns=['y'])
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
# from data import data
# df = data.read()
# df = df.iloc[0:100]
# X = df.drop(columns='SalePrice')
# y = df['SalePrice']

# gd = SGD()
# np.random.seed(5)
# X, y = gd.encode_labels(X, y)
# X, y = gd.scale(X, y, 'minmax')
# theta = np.random.rand(X.shape[1])
# search = gd.search(X, y, theta)

# %%
# gd.report(n=10, smooth=True)
# %%
# ani = gd.display_costs(interval=10)
# rc('animation', html='jshtml')
# rc
# HTML(ani.to_jshtml())
