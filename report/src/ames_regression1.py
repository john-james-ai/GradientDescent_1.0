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
from matplotlib import colors as mcolors
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from numpy import array, newaxis
import pandas as pd
import seaborn as sns
from sklearn import linear_model
from sklearn.datasets import make_regression

# ============================================================================ #
#                                 DATA                                         #
# ============================================================================ #
ames = pd.read_csv("./report/data/sample.csv",
                   encoding="Latin-1", low_memory=False)
ames = ames[['GrLivArea', 'SalePrice']]
ames.columns = [r'Living Area $(feet^2)$', 'Sales Price ($)']

# Create intercept point in data for regression line
X = pd.Series(ames[r'Living Area $(feet^2)$'])
X = X[:, np.newaxis]

# Train, fit and predict a linear model
lr = linear_model.LinearRegression()
lr.fit(ames[[r'Living Area $(feet^2)$']], ames[['Sales Price ($)']])
yHat = lr.predict(X)

# Create error table
se = (yHat-X)**2
mse = np.mean(se)
df1 = pd.DataFrame(ames['Sales Price ($)'])
df2 = pd.DataFrame(yHat, columns=['h(x)'])
df3 = pd.DataFrame(se, columns=['Squared Error'])
e_table_1 = pd.concat([df1, df2, df3], axis=1)
mse_row = pd.Series({"Sales Price ($)": "MSE", "Squared Error": mse},
                    index=e_table_1.columns).fillna("")
e_table_1 = e_table_1.append(mse_row, ignore_index=True)

# Extract slope and intercept
intercept = lr.intercept_[0]
slope = lr.coef_[0][0]

# ============================================================================ #
#                                   PLOT                                       #
# ============================================================================ #
# Create error line
error1x = ames.iloc[7][r'Living Area $(feet^2)$']
error1y = intercept + slope * ames.iloc[7][r'Living Area $(feet^2)$']
error2x = ames.iloc[7][r'Living Area $(feet^2)$']
error2y = ames.iloc[7]['Sales Price ($)']
errorlinex = [error1x, error2x]
errorliney = [error1y, error2y]

# Create error text coordinates
error_x = error1x
error_y = (error1y+error2y)/2
error_text_x = 1600
error_text_y = error_y

# Scatterplot of observations
fig, ax = plt.subplots(figsize=(6, 4))
plt.scatter(ames[[r'Living Area $(feet^2)$']],
            ames[['Sales Price ($)']], color='steelblue', s=100)

# Regression Line
plt.plot(X, yHat, color='firebrick')
ax.get_xaxis().set_major_formatter(
    mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
ax.get_yaxis().set_major_formatter(
    mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))


# Annotate Error
plt.plot(errorlinex, errorliney, color='steelblue')
ax.annotate("Error", xy=(error_x, error_y), xycoords='data',
            xytext=(error_text_x, error_text_y),
            arrowprops=dict(arrowstyle="simple"))


# Plot Hypothesis
ax.text(x=1000, y=220000, s=r'$h(x)=19,442.49 + 101.04x$')

# Plot MSE
s = "Mean Squared Error = {:.2E}".format(mse)
ax.text(x=1000, y=200000, s=s)

plt.xlabel(r'Living Area $(feet^2)$')
plt.ylabel("Sales Price ($)")
plt.title("Housing Price by Area - Hypothesis 1")

plt.tight_layout()
# plt.show()
fig.savefig("./report/figures/ames_regression1.png", facecolor='w')
plt.close(fig)
