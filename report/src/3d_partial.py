# %%
# ============================================================================ #
#                                 LIBRARIES                                    #
# ============================================================================ #
import inspect
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))

import matplotlib.pyplot as plt
plt.style.use('seaborn-notebook')
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# ============================================================================ #

def partial():
    def z(x, y):
        return(x**2 + x*y - y**2)


    def zprimex(x, y):
        return(2*x + y)


    def zprimey(x, y):
        return(x-2*y)


    def tangent_x(a, x, y):
        return(z(a, y)+zprimex(a, y)*(x-a))


    def tangent_y(a, x, y):
        return(z(a, x)+zprimey(a, x)*(y-a))


    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')

    # 3D Plot
    x = np.arange(-5, 5, .05)
    y = np.arange(-5, 5, .05)
    X, Y = np.meshgrid(x, y)
    zs = np.array([z(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    ax.plot_surface(X, Y, Z, alpha=0.2)


    # 2D Partial plot with respect to x
    xx, zz = np.meshgrid(x, z(x, y))
    yy = xx*0
    #ax.plot_surface(xx, yy, zz, alpha=0.5)

    # 2D Partial plot with respect to y
    yy, zz = np.meshgrid(y, z(x, y))
    xx = yy*0
    #ax.plot_surface(xx, yy, zz, alpha=0.5)

    # Plot point
    point_x = -2
    point_y = 0
    point_z = z(point_x, point_y)
    #ax.scatter(point_x, point_y, point_z)
    ax.text(x=point_x, y=point_y, z=z(point_x, point_y)*1.5, s='P')

    # Plot tangent to point w.r.t. x
    a = 0
    tan_x = np.linspace(-4, 4, 10)
    tan_y = tan_x * 0
    tan_z = tangent_x(a, tan_x, tan_y)
    ax.plot(tan_x, tan_y, tan_z, color='steelblue')

    # Plot tangent to point w.r.t. y
    a = 0
    tan_x = np.linspace(-2, -2, 10)
    tan_y = np.linspace(-4, 4, 10)
    tan_z = tangent_y(a, tan_x, tan_y)
    ax.plot(tan_x, tan_y, tan_z, color='steelblue')


    # Get rid of colored axes planes
    # Remove grid
    ax.grid(False)
    # Remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    ax.set_xlabel('X')
    ax.set_ylabel('Y ')
    ax.set_zlabel('Z')

    plt.tight_layout()
    fig.savefig("./report/figures/partial.png", facecolor='w')
    plt.close(fig)
partial()
