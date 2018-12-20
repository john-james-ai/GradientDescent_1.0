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
import numpy as np
import pandas as pd
import seaborn as sns

def ames_line():
    # Prepare Data
    ames = pd.read_csv("./report/data/sample.csv", encoding="Latin-1", low_memory=False)
    ames = ames[['GrLivArea', 'SalePrice']]
    area = ames['GrLivArea']
    intercept = pd.DataFrame({'X0': np.repeat(1, len(area))})
    price = ames['SalePrice']
    ames_x0 = pd.concat([price, intercept, area], axis=1)

    # Render Lineplot
    sns.set(style="whitegrid", font_scale=1)
    sns.set_palette("GnBu_d")
    fig, ax = plt.subplots(figsize=(12,8))
    ax = sns.scatterplot(x='GrLivArea', y='SalePrice', data=ames)
    ax = sns.lineplot(x='GrLivArea', y='SalePrice', data=ames, ci=None)
    ax.get_xaxis().set_major_formatter(
        mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.get_yaxis().set_major_formatter(
        mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.xlabel(r'Living Area $(feet^2)$')
    plt.ylabel("Sales Price ($)")
    plt.title("Housing Price by Area")
    plt.tight_layout()
    fig.savefig("./report/figures/ames_line.png", facecolor='w')
    plt.show()
    plt.close(fig)
    return(ames, fig)
ames, fig = ames_line()
