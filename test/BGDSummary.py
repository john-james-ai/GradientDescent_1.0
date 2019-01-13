# =========================================================================== #
#                       BATCH GRADIENT DESCENT TEST                           #
# =========================================================================== #
# %%
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import seaborn as sns
import sys
srcdir = "c:\\Users\\John\\Documents\\Data Science\\Libraries\\GradientDescent\\src"
sys.path.append(srcdir)

from GradientDescent import BGD
from utils import save_fig
import data 
#%%
# --------------------------------------------------------------------------- #
#                             SEARCH FUNCTION                                 #  
# --------------------------------------------------------------------------- #
X, X_val, y, y_val = data.demo()
theta = np.array([-1,-1]) 
learning_rate_sched = 'c'
learning_rate = 1.39
time_decay=0.01
step_decay=0.88
step_epochs=20
exp_decay=0.01
precision = 0.001
maxiter = 5000
i_s=20

# Perform best performing searches
gd = BGD()
gd.fit(X=X, y=y, theta=theta, X_val=None, y_val=None, learning_rate=learning_rate, 
            learning_rate_sched = learning_rate_sched, time_decay=time_decay, 
            step_decay=step_decay, step_epochs=step_epochs, exp_decay=exp_decay, 
            maxiter=maxiter, precision=precision, i_s=i_s, scaler='minmax')
final_cost_constant = gd.summary()['final_costs']
duration_constant = gd.summary()['duration']

learning_rate_sched = 't'
gd.fit(X=X, y=y, theta=theta, X_val=None, y_val=None, learning_rate=learning_rate, 
            learning_rate_sched = learning_rate_sched, time_decay=time_decay, 
            step_decay=step_decay, step_epochs=step_epochs, exp_decay=exp_decay, 
            maxiter=maxiter, precision=precision, i_s=i_s, scaler='minmax')
final_cost_time = gd.summary()['final_costs']
duration_time = gd.summary()['duration']

learning_rate_sched = 's'
gd.fit(X=X, y=y, theta=theta, X_val=None, y_val=None, learning_rate=learning_rate, 
            learning_rate_sched = learning_rate_sched, time_decay=time_decay, 
            step_decay=step_decay, step_epochs=step_epochs, exp_decay=exp_decay, 
            maxiter=maxiter, precision=precision, i_s=i_s, scaler='minmax')
final_cost_step = gd.summary()['final_costs']
duration_step = gd.summary()['duration']

learning_rate_sched = 'e'
gd.fit(X=X, y=y, theta=theta, X_val=None, y_val=None, learning_rate=learning_rate, 
            learning_rate_sched = learning_rate_sched, time_decay=time_decay, 
            step_decay=step_decay, step_epochs=step_epochs, exp_decay=exp_decay, 
            maxiter=maxiter, precision=precision, i_s=i_s, scaler='minmax')
final_cost_exp = gd.summary()['final_costs']
duration_exp = gd.summary()['duration']

# Format data
schedules = ['Constant', 'Time Decay', 'Step Decay', 'Exponential Decay']
final_costs = [final_cost_constant, final_cost_time, final_cost_step, final_cost_exp]
final_times = [duration_constant, duration_time, duration_step, duration_exp]
data = pd.DataFrame({'Schedule':schedules, 'Costs': final_costs, 'Computation Time':final_times})
print(data.info())
print(data)

# Plot
fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(12,4))
sns.set(style="whitegrid", font_scale=1)
fig.suptitle('Batch Gradient Descent Performance Summary by Learning Rate Schedule')

# Plot Costs 
title = 'Costs by Learning Rate Schedule'
ax0 = sns.barplot(x='Schedule', y='Costs', data=data, ax=ax0)
ax0.set_facecolor('w')
ax0.tick_params(colors='k')
ax0.xaxis.label.set_color('k')
ax0.yaxis.label.set_color('k')
ax0.set_xlabel('Learning Rate Schedule')
ax0.set_ylabel('Costs')
ax0.set_title(title, color='k')

# Plot Times
title = 'Computation Times by Learning Rate Schedule'
ax1 = sns.barplot(x='Schedule', y='Computation Time', data=data, ax=ax1)
ax1.set_facecolor('w')
ax1.tick_params(colors='k')
ax1.xaxis.label.set_color('k')
ax1.yaxis.label.set_color('k')
ax1.set_xlabel('Learning Rate Schedule')
ax1.set_ylabel('Computation Time')
ax1.set_title(title, color='k')

fig.tight_layout(rect=[0, 0, 1, 0.9])
#%%
# Save
directory = "./test/figures/BGD/Lab/"
filename = 'Batch Gradient Descent Performance Summary.png'
save_fig(fig, directory, filename)
plt.close(fig)