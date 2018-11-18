
# %%
# =========================================================================== #
#                             Gradient Descent                                #
# =========================================================================== #
import inspect
import os
import sys

from IPython.display import HTML

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
    _hypothesis = None

    def __linear_hypothesis(self, X, theta):
        return((X.dot(theta))[0])

    def __logistic_hypothesis(self, X, theta):
        # TODO: Add sigmoid function
        return(X.dot(theta))

    def __error(self, h, y):
        return(h-y)

    def __mse(self, e):
        return(1/2 * np.mean(e**2))

    def __cost_mesh(self, X, y, THETA):
        return(np.sum((X.dot(THETA) - y)**2)/(2*len(y)))

    def __gradient(self, X, e):
        return(X.T.dot(e)/X.shape[0])

    def __update(self, alpha, theta, gradient):
        return(theta - alpha * gradient)

    def __encode_labels(self, X, y):
        le = LabelEncoder()
        X = X.apply(le.fit_transform)
        y = le.fit_transform(y)
        return(X, y)

    def __scale(self, X, y, scaler):
        # Select scaler
        if scaler == 'std':
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()

        # Combine X and y into a dataframe
        y = pd.DataFrame({'y': y})
        df = pd.concat([X, y], axis=1)

        # Scale then return X and y
        df_scaled = scaler.fit_transform(df)
        df = pd.DataFrame(df_scaled, columns=df.columns)
        X = df.drop(columns=['y']).values
        y = df['y']
        return(X, y)

    def search(self, X, y, theta, alpha=0.001, hypothesis='linear', maxiter=0, precision=0.001, stop='i',
               label_encoder=True, scaler='minmax'):
        J_history = []
        theta_history = []
        epochs = 0
        J = 0
        g = np.repeat(1, X.shape[1])
        converged = False

        # Prepare Data
        if label_encoder:
            X, y = self.__encode_labels(X, y)
        if scaler:
            X, y = self.__scale(X, y, scaler)

        while not converged:
            J_prior = J
            g_prior = g

            if hypothesis == 'linear':
                h = self.__linear_hypothesis(X, theta)
            else:
                h = self.__logistic_hypothesis(X, theta)
            e = self.__error(h[0], y)
            J = self.__mse(e)
            print("Cost is " + str(J))
            g = self.__gradient(X, e)
            theta = self.__update(alpha, theta, g)

            theta_history.append(theta)
            J_history.append(J)

            if (stop == 'i'):
                if abs(J-J_prior) <= precision:
                    converged = True
            else:
                g_diff = np.absolute(g-g_prior)
                g_close = g_diff < precision
                if (np.all(g_close)):
                    converged = True

            epochs += 1
            if maxiter:
                if epochs == maxiter:
                    break

        return(X, y, epochs, theta_history, J_history)

    def __init__(self):
        pass


# %%
from data import data
df = data.read()
X = df.drop(columns='SalePrice')
y = df['SalePrice']
np.random.seed(5)
theta = np.random.rand(X.shape[1], 1)
gd = GradientDescent()
X, y, epochs, theta_history, J_history = gd.search(X, y, theta, stop='g')
