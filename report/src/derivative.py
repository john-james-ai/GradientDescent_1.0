# %%
# ============================================================================ #
#                                 LIBRARIES                                    #
# ============================================================================ #
import inspect
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('seaborn-notebook')
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import UnivariateSpline

# %%
def derivative():

        def f(x):
                return(x**2)

        # Data
        x = np.linspace(-5, 5, 100)
        y = f(x)
        x_points = [1, 3.5]
        y_points = [1, 3.5**2]
        
        # Secant and Tangent Lines
        secant_x = np.linspace(0, 5, 10)
        s = UnivariateSpline(x_points, y_points, k=1)
        secant_y = s(secant_x)
        a = 1
        h = 0.1
        fprime = (f(a+h)-f(a))/h
        tan_x = np.linspace(-1, 3, 10)

        def tan_y(x):
                return(f(a)+fprime*(x-a))
        
        # Vertical and horizontal lines
        x0_x = [1, 1]
        x0_y = [-5, 1]
        x1_x = [3.5, 3.5]
        x1_y = [-5, 3.5**2]
        x2_x = [1, 4.5]
        x2_y = [1, 1]
        x3_x = [3.5, 4.5]
        x3_y = [3.5**2, 3.5**2]
        pq_x = [3.5]
        pq_y = [1]
        
        # Plot
        sns.set(style="white", font_scale=2)
        fig, ax = plt.subplots(figsize=(12,8))
        ax = sns.scatterplot(x=x_points, y=y_points)
        ax = sns.scatterplot(x=pq_x, y=pq_y, color='0.75')
        ax = sns.lineplot(x=x, y=y, ci=None)
        ax = sns.lineplot(x=secant_x, y=secant_y, ci=None)
        ax = sns.lineplot(x=tan_x, y=tan_y(tan_x), ci=None)
        ax.lines[1].set_linestyle("--")

        ax.text(x=1*.8, y=1*1.2, s='P')
        ax.text(x=3.5*.9, y=f(3.5)*1.05, s='Q')
        ax.text(x=2.2, y=-2.5, s=r'$h$', color='dimgrey', fontsize=14)
        ax.text(x=3.75, y=(3**2-((3**2-1)/2)), s=r'$f(a+h)-f(a)$', color='dimgrey',
                fontsize=14)

        ax.annotate(r'$y=x^2$', xy=(-4, f(4)), xytext=(-3, f(4.5)),
                arrowprops=dict(facecolor='black', shrink=0.05))
        ax.annotate('Secant', xy=(2, s(2)), xytext=(0, 8.5),
                arrowprops=dict(facecolor='black', shrink=0.05))
        ax.annotate('Tangent', xy=(-1, tan_y(-1)), xytext=(-3, tan_y(0)),
                arrowprops=dict(facecolor='black', shrink=0.05))
        ax.annotate('', xy=(1, -2), xytext=(2, -2),
                arrowprops=dict(facecolor='grey', shrink=0.05,
                                width=2, headwidth=5))
        ax.annotate('', xy=(3.5, -2), xytext=(2.5, -2),
                arrowprops=dict(facecolor='grey', shrink=0.05,
                                width=2, headwidth=5))
        ax.annotate('', xy=(4, 3.5**2), xytext=(4, 7),
                arrowprops=dict(facecolor='grey', shrink=0.05,
                                width=2, headwidth=5))

        ax.annotate('', xy=(4, 1), xytext=(4, 5),
                arrowprops=dict(facecolor='grey', shrink=0.05,
                                width=2, headwidth=5))

        ax.plot(x0_x, x0_y, color='grey', lw=1, linestyle='--')
        ax.plot(x1_x, x1_y, color='grey', lw=1, linestyle='--')
        ax.plot(x2_x, x2_y, color='grey', lw=1, linestyle='--')
        ax.plot(x3_x, x3_y, color='grey', lw=1, linestyle='--')

        x0_tick = r'$a$'
        x1_tick = r'$a+h$'
        ax.set_xticks((1, 3.5), (x0_tick, x1_tick))

        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_title("Derivative", color='k')
        plt.tight_layout()

        fig.savefig("./report/figures/derivative.png", facecolor='w')
        plt.close(fig)
derivative()       

