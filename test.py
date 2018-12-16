# =========================================================================== #
#                             GRADIENT TEST PLATFORM                          #
# =========================================================================== #
# --------------------------------------------------------------------------- #
#                                    LIBRARIES                                #
# --------------------------------------------------------------------------- #
#%%
import inspect
import os
import sys

import numpy as np
from numpy import array, newaxis
import pandas as pd

from GradientSearch import GradientSearch
# --------------------------------------------------------------------------- #
#                                      DATA                                   #
# --------------------------------------------------------------------------- #
from data import data
df = data.read()
X = df['Area']
y = df['SalePrice']

# Prep Data
gs = GradientSearch()
X, y = gs.prep_data(X,y)
print(X.shape)

# Initialize parameters
theta = np.array([-1,-1])

# Conduct Search
gs.search(X,y, theta)
#%%
gs.get_result()
