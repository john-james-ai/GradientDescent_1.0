# =========================================================================== #
#                   BATCH GRADIENT DESCENT DEMO TEST                          #
# =========================================================================== #
# %%
import inspect
import os
import sys

from IPython.display import HTML
from matplotlib import animation, rc
from matplotlib import rcParams
import numpy as np
import pandas as pd

src = "c:\\Users\\John\\Documents\\Data Science\\Libraries\\GradientDescent\\src"
sys.path.append(src)
from GradientDemo import BGDDemo
import data

# Data
X, X_val, y, y_val = data.demo(n=500)

# Parameters
theta = np.array([-1,-1]) 
learning_rate_sched = 's'
learning_rate = 1.6
step_decay=0.75
step_epochs=20
precision = 0.001
maxiter = 5000
i_s=10
directory = "./test/figures/BGD/"

# Run experiment
demo = BGDDemo()
demo.fit(X=X,y=y, theta=theta, learning_rate_sched=learning_rate_sched,
         learning_rate=learning_rate, precision=precision, step_decay=step_decay,
         step_epochs=step_epochs,maxiter=maxiter, i_s=i_s)
#%%           
demo.show_search(directory=directory, fps=1)
demo.show_fit(directory=directory, fps=1)

