# %%
# ============================================================================ #
#                                 LIBRARIES                                    #
# ============================================================================ #
import sys, os, inspect
import math
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))

import numpy as np
import pandas as pd
import seaborn as sns

def non_differentiable():
    # ============================================================================ #
    x = np.arange(-5, 5, .05)
    y1 = np.abs(x)
    y2 = np.sqrt(np.abs(x))+0.5
    # ============================================================================ #
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,8))
    axes[0].plot(x, y1)
    axes[0].text(-1.5, 4.5, r'$y=\mid{x}\mid$')

    axes[1].plot(x, y2)
    axes[1].text(-2.5, 2.5, r'$y=\sqrt{\mid{x}\mid}+0.5$')

    plt.tight_layout()
    fig.savefig("./report/figures/non_differentiable.png", facecolor='w')
    plt.close(fig)
non_differentiable()