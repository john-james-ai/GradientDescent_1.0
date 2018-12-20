# %%
# ============================================================================ #
#                                 LIBRARIES                                    #
# ============================================================================ #
import sys, os, inspect
sys.path.append(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from math import sqrt

# %%
def polynomial():
    def f(x):
        return(x**3-x)


    def fprime(x):
        return(3*(x**2)-1)


    def tan_y(x_0, x_1):
        return(f(x_0)+fprime(x_0)*(x_1-x_0))


    x = np.linspace(-2, 2, 100)
    y = f(x)
    
    # Derivative Function
    fprime_x = x
    fprime_y = fprime(fprime_x)

    # Tangent Line
    ta_x = np.linspace(-1.7, -1.3, 10)
    ta_y = tan_y(-1.5, ta_x)
    tb_x = np.linspace(-1.2, -.8, 10)
    tb_y = tan_y(-1, tb_x)
    tc_x = np.linspace(-.7, -.3, 10)
    tc_y = tan_y(-sqrt(1/3), tc_x)
    td_x = np.linspace(-.15, .15, 10)
    td_y = tan_y(0, td_x)
    te_x = np.linspace(.3, .7, 10)
    te_y = tan_y(sqrt(1/3), te_x)
    tf_x = np.linspace(.8, 1.2, 10)
    tf_y = tan_y(1, tf_x)
    tg_x = np.linspace(1.3, 1.7, 10)
    tg_y = tan_y(1.5, tg_x)

    # Plot
    sns.set(style="white", font_scale=1)
    fig, ax = plt.subplots(figsize=(12,8))

    ax = sns.lineplot(x=x, y=y, size=1)
    #ax = sns.lineplot(x=fprime_x, y=fprime_y)
    ax = sns.lineplot(x=ta_x, y=ta_y, color='black')
    ax = sns.lineplot(x=tb_x, y=tb_y, color='black')
    ax = sns.lineplot(x=tc_x, y=tc_y, color='black')
    ax = sns.lineplot(x=td_x, y=td_y, color='black')
    ax = sns.lineplot(x=te_x, y=te_y, color='black')
    ax = sns.lineplot(x=tf_x, y=tf_y, color='black')
    ax = sns.lineplot(x=tg_x, y=tg_y, color='black')
    ax.legend_.remove()

    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title(r'$y=x^3-x$')
    plt.tight_layout()
    fig.savefig("./report/figures/polynomial.png", facecolor='w')
    plt.close(fig)
polynomial()