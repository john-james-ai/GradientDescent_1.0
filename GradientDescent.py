
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
        self._maxiter = 0
        self._regression = True
        self._precision = 0.0001
        self._stop_condition = 'i'
        self._stop = False
        self.J_train_history = []
        self.theta_history = []
        self.epochs = []

    def _hypothesis(self, X, theta):
        if self._regression:
            return(X.dot(theta))
        else:
            # TODO: add prediction based upon probability of class membership
            pass

    def _error(self, h, y):
        return(h-y)

    def _cost(self, e):
        if self._regression:
            return(1/2 * np.mean(e**2))
        else:
            # TODO: compute cost using cross-entropy
            pass

    def _gradient(self, X, e):
        return(X.T.dot(e)/X.shape[0])

    def _update(self, alpha, theta, gradient):
        return(theta-(alpha * gradient))

    def encode_labels(self, X, y):
        le = LabelEncoder()
        X = X.apply(le.fit_transform)
        y = le.fit_transform(y)
        return(X, y)

    def scale(self, X, y, scaler='minmax'):
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
        X = df.drop(columns=['y']).values
        y = df['y']
        return(X, y)

    def _finished(self, J, J_prior, g, g_prior, epoch):
        if epoch == self._maxiter:
            self._stop = True
            return(True)
        elif self._stop_condition == 'i' and abs(J-J_prior) < self._precision:
            self._stop = True
            return(True)
        else:
            g_diff = np.absolute(g-g_prior)
            g_close = g_diff < self._precision
            if (np.all(g_close)):
                self._stop = True
                return(True)
        return(False)

    def search(self, X, y, theta, alpha=0.001, maxiter=0, regression=True, precision=0.001, stop_condition='i'):
        self.J_train_history = []
        self.theta_history = []
        self.epochs = []
        self._maxiter = maxiter
        self._regression = regression
        self._precision = precision
        self._stop = False
        self._stop_condition = stop_condition

        J = 0
        epoch = 0
        g = np.repeat(1, X.shape[1])
        converged = False

        while not self._stop:
            epoch += 1
            J_prior = J
            g_prior = g

            h = self._hypothesis(X, theta)
            e = self._error(h, y)
            J = self._cost(e)
            g = self._gradient(X, e)
            theta = self._update(alpha, theta, g)

            self.theta_history.append(theta)
            self.J_train_history.append(J)
            self.epochs.append(epoch)

            if self._finished(J, J_prior, g, g_prior, epoch):
                break

        return(X, y, self.epochs, self.theta_history, self.J_train_history)


class BGD(GradientDescent):
    def __init__(self):
        pass


class SGD(GradientDescent):

    def __init__(self):
        self._regression = True
        self._precision = 0.001
        self._maxiter = 0
        self._validate = 100
        self._stop = False
        self.J_train_history = []
        self.J_val_history = []
        self.theta_history = []
        self.iterations = []
        self.epochs = []

    def _shuffle(self, X, y):
        y = np.expand_dims(y, axis=1)
        z = np.append(arr=X, values=y, axis=1)
        np.random.shuffle(z)
        X = np.delete(z, z.shape[1]-1, axis=1)
        y = z[:, z.shape[1]-1]
        return(X, y)

    def _split(self, X, y, p_train=0.8):
        if X.shape[0] != y.shape[0]:
            raise Exception("The length X, {}".format(X.shape[0]) +
                            "does not match the length of y,{}".format(y.shape[0]))

        point = math.floor(X.shape[0]*p_train)
        X_splits = np.split(X, [point])
        y_splits = np.split(y, [point])

        X_train = X_splits[0]
        X_val = X_splits[1]
        y_train = y_splits[0]
        y_val = y_splits[1]

        return(X_train, y_train, X_val, y_val)

    def _finished(self, X_val, y_val, theta, iteration):
        if iteration == self._maxiter:
            self._stop = True
            return(True)
        elif iteration % self._validate == 0:
            h = self._hypothesis(X_val, theta)
            e = self._error(h, y_val)
            J = self._cost(e)
            if abs(J-self.J_val_history[-1]) < self._precision:
                self.J_val_history.append(J)
                self._stop = True
                return(True)
            else:
                self.J_val_history.append(J)
        else:
            return(False)

    def search(self, X, y, theta, alpha=0.001, maxiter=0, regression=True,
               precision=0.0001, validate=100):
        self._regression = regression
        self._precision = precision
        self._maxiter = maxiter
        self._validate = validate
        self._stop = False
        self.J_train_history = []
        self.J_val_history = []
        self.theta_history = []
        self.iterations = []
        self.epochs = []

        J = math.inf
        epoch = 0
        iteration = 0
        g = np.repeat(1, X.shape[1])

        while not self._stop:
            epoch += 1
            self.epochs.append(epoch)

            X, y = self._shuffle(X, y)
            X_train, y_train, X_val, y_val = self._split(X, y)

            for x_i, y_i in zip(X_train, y_train):
                iteration += 1
                self.iterations.append(iteration)

                h = self._hypothesis(x_i, theta)
                e = self._error(h, y_i)
                J = self._cost(e)
                g = self._gradient(x_i, e)

                self.theta_history.append(theta)
                self.J_train_history.append(J)

                if self._finished(X_val, y_val, theta, iteration):
                    break
                theta = self._update(alpha, theta, g)

        return(X, y, self.epochs, self.iterations, self.theta_history, self.J_train_history, self.J_val_history)


# %%
from data import data
df = data.read()
df = df.iloc[0:10, :]
X = df.drop(columns='SalePrice')
y = df['SalePrice']

gd = SGD()
theta = np.random.rand(X.shape[1])
X, y = gd.encode_labels(X, y)
X, y = gd.scale(X, y, 'minmax')
X, y, epochs, iterations, theta_history, J_train_history, J_val_history = gd.search(
    X, y, theta, maxiter=100000)
print(J_val_history)
print(iterations[-1])
