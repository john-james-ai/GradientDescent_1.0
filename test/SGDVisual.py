# =========================================================================== #
#                  STOCHASTIC GRADIENT DESCENT VISUAL TEST                    #
# =========================================================================== #
# %%
import numpy as np 
import pandas as pd 
import sys
srcdir = "c:\\Users\\John\\Documents\\Data Science\\Libraries\\GradientDescent\\src"
sys.path.append(srcdir)

from GradientDescent import BGD, SGD 
from GradientVisual import GradientVisual
import data 
#%%
# --------------------------------------------------------------------------- #
#                             SEARCH FUNCTION                                 #  
# --------------------------------------------------------------------------- #

X, X_val, y, y_val = data.demo()
alg = 'Stochastic Gradient Descent'
theta = np.array([-1,-1])
learning_rate = 0.01
learning_rate_sched = 'c'
stop_metric = 'j'
time_decay = 0.5
step_decay=0.5
step_epochs=2
exp_decay=0.1
precision = 0.01
maxiter = 5000
i_s = 5
directory = "./test/figures/SGD/"
gd = SGD()
gd.fit(X=X, y=y, theta=theta,X_val=X_val, y_val=y_val, learning_rate=learning_rate,
        learning_rate_sched=learning_rate_sched, time_decay=time_decay, step_decay=step_decay,
        step_epochs=step_epochs, exp_decay=exp_decay,
        maxiter=maxiter, precision=precision, i_s=i_s, stop_metric=stop_metric)

viz = GradientVisual()
summary = gd.summary()
detail = gd.detail()
eval = gd.eval()

# Line Plot
viz.figure(alg=alg, data=detail, x='iterations', y='cost',
           func=viz.lineplot, width=1, show=True)
viz.figure(alg=alg, data=eval, x='iterations', y='cost',
           func=viz.lineplot, width=1, show=True)           
X, y = gd.prep_data(X, y)
viz.show_fit(alg, X, y, detail, summary, directory=directory) 
