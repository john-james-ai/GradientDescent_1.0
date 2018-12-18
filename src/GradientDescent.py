
# %%
# =========================================================================== #
#                             GRADIENT DESCENT                                #
# =========================================================================== #
'''
Class creates gradient descent solutions for each of the following gradient
descent variants and optimization algorithms.
- Batch gradient descent
- Stochastic gradient descent
- Momentum
- Nesterov Accelerated Gradient
- Adagrad
- Adadelta
- RMSProp
- Adam
- Adamax
- Nadam
'''

# --------------------------------------------------------------------------- #
#                                LIBRARIES                                    #
# --------------------------------------------------------------------------- #

import inspect
import os
import sys

import datetime
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('seaborn-notebook')
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from matplotlib import animation, rc
from matplotlib import colors as mcolors
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import numpy as np
from numpy import array, newaxis
import pandas as pd
import seaborn as sns

from GradientSearch import BGDSearch, SGDSearch
# --------------------------------------------------------------------------- #
#                       GRADIENT DESCENT BASE CLASS                           #
# --------------------------------------------------------------------------- #


class GradientDescent:
    '''Base class for Gradient Descent'''

    def __init__(self):
        self._alg = "Gradient Descent"
        self._summary = None
        self._detail = None

    def _todf(self, x, stub):
        n = len(x[0])
        df = pd.DataFrame()
        for i in range(n):
            colname = stub + str(i)
            vec = [item[i] for item in x]
            df_vec = pd.DataFrame(vec, columns=[colname])
            df = pd.concat([df, df_vec], axis=1)
        return(df) 

    def detail(self):
        if self._summary is None:
            raise Exception('No search results to report.')
        else:
            return(self._detail)

    def summary(self):
        if self._summary is None:
            raise Exception('No search results to report.')
        else:
            return(self._summary)
  
    def search(self, X, y, theta, X_val=None, y_val=None, 
               alpha=0.01, maxiter=0, precision=0.001,
               stop_measure='j', stop_metric='a', scaler='minmax'):

        # Set cross-validated flag if validation set included 
        cross_validated = all(v is not None for v in [X_val, y_val])

        # Package request
        self._request = dict()
        self._request['alg'] = self._alg        
        self._request['data'] = {'X': X, 'y':y, 'X_val':X_val, 'y_val':y_val,
                                 'scaler': scaler} 
        self._request['hyper'] = {'alpha': alpha, 'theta': theta,
                                  'maxiter': maxiter, 
                                  'precision': precision,
                                  'stop_measure': stop_measure,
                                  'stop_metric': stop_metric,
                                  'cross_validated': cross_validated}

        # Run search and obtain result        
        gd = BGDSearch()
        start = datetime.datetime.now()
        gd.search(self._request)
        end = datetime.datetime.now()

        # Extract Detail Results
        epochs = pd.DataFrame(gd.get_epochs(), columns=['epochs'])        
        iterations = pd.DataFrame(gd.get_iterations(), columns=['iterations'])        
        thetas = self._todf(gd.get_thetas(), stub='theta_')        
        costs = pd.DataFrame(gd.get_costs(), columns=['cost'])        
        self._detail = pd.concat([epochs, iterations, thetas, costs], axis=1)
        
        if cross_validated:            
            mse = pd.DataFrame(gd.get_mse(), columns=['mse'])
            self._detail = pd.concat([epochs, iterations, thetas, costs, mse], axis=1)

        #Format stop condition
        if self._request['hyper']['stop_measure'] == 'j':
            stop_measure = 'Costs'
        if self._request['hyper']['stop_measure'] == 'g':
            stop_measure = 'Gradient'
        if self._request['hyper']['stop_measure'] == 'v':
            stop_measure = 'Validation Set MSE'            
        if self._request['hyper']['stop_metric'] == 'a':
            stop_metric = 'Absolute Change'         
        if self._request['hyper']['stop_metric'] == 'p':
            stop_metric = 'Relative Change'         
        stop = stop_metric + " in " + stop_measure

        # Format validation set MSE 
        mse_init = 'NA'
        mse_final = 'NA'
        if cross_validated:
            mse_init = mse.iloc[0].item(),
            mse_final = mse.iloc[-1].item()
        
        # Package summary results
        self._summary = pd.DataFrame({'algorithm': gd.get_alg(),
                                    'start':start,
                                    'end':end,
                                    'duration':end - start,
                                    'epochs': epochs.shape[0],
                                    'iterations': iterations.shape[0],
                                    'alpha': alpha,
                                    'stop': stop,
                                    'precision': precision,
                                    'initial costs': costs.iloc[0].item(),
                                    'final costs': costs.iloc[-1].item(),
                                    'initial mse':mse_init,
                                    'final mse': mse_final},
                                    index=[0])
    
    def plot(self, path=None, show=True):

        # Set figure dimensions 
        if self._request['hyper']['cross_validated']:
            height = 8
            nrow = 2
        else:
            height=4
            nrow = 1

        # Obtain matplotlib figure
        fig = plt.figure(figsize=(12,height))
        gs = fig.add_gridspec(nrow,2)
        sns.set(style="whitegrid", font_scale=1)
        
        # Cost bar plot
        x = ['Initial Cost', 'Final Costs']
        y = [self._summary['initial costs'].values.tolist(),
             self._summary['final costs'].values.tolist()]
        y = [item for sublist in y for item in sublist]
        ax0 = fig.add_subplot(gs[0,0])
        ax0 = sns.barplot(x,y)  
        ax0.set_facecolor('w')
        ax0.tick_params(colors='k')
        ax0.xaxis.label.set_color('k')
        ax0.yaxis.label.set_color('k')        
        ax0.set_ylabel('Cost')
        title = 'Initial and Final Costs ' + \
                r'$\alpha$' + " = " + str(self._request['hyper']['alpha']) + " " + \
                r'$\epsilon$' + " = " + str(round(self._request['hyper']['precision'],5)) 
        rects = ax0.patches
        for rect, label in zip(rects, y):
            height = rect.get_height()
            ax0.text(rect.get_x() + rect.get_width() / 2, height + .03, str(round(label,3)),
                    ha='center', va='bottom')                
        ax0.set_title(title, color='k')

        # Cost Line Plot
        x = self._detail['iterations']
        y = self._detail['cost']
        ax1 = fig.add_subplot(gs[0,1])
        ax1 = sns.lineplot(x,y)
        ax1.set_facecolor('w')
        ax1.tick_params(colors='k')
        ax1.xaxis.label.set_color('k')
        ax1.yaxis.label.set_color('k')
        ax1.set_xlabel('Iterations')
        ax1.set_ylabel('Cost')
        title = 'Costs by Iteration ' + \
        r'$\alpha$' + " = " + str(self._request['hyper']['alpha']) + \
        r'$\epsilon$' + " = " + str(round(self._request['hyper']['precision'],5)) 
        ax1.set_title(title, color='k')

        if self._request['hyper']['cross_validated']:
            # MSE bar plot
            x = ['Initial MSE', 'Final MSE']
            y = [self._summary['initial mse'].values.tolist(),
                self._summary['final mse'].values.tolist()]
            y = [item for sublist in y for item in sublist]
            ax2 = fig.add_subplot(gs[1,0])
            ax2 = sns.barplot(x,y)  
            ax2.set_facecolor('w')
            ax2.tick_params(colors='k')
            ax2.xaxis.label.set_color('k')
            ax2.yaxis.label.set_color('k')
            ax2.set_ylabel('Mean Squared Error (MSE)')
            rects = ax2.patches
            for rect, label in zip(rects, y):
                height = rect.get_height()
                ax2.text(rect.get_x() + rect.get_width() / 2, height + .03, str(round(label,3)),
                        ha='center', va='bottom')              
            title = 'Initial and Final Validation Set MSE ' + \
                    r'$\alpha$' + " = " + str(self._request['hyper']['alpha']) + " " + \
                    r'$\epsilon$' + " = " + str(round(self._request['hyper']['precision'],5)) 
            ax2.set_title(title, color='k')

            # MSE Line Plot
            x = self._detail['iterations']
            y = self._detail['mse']
            ax3 = fig.add_subplot(gs[1,1])
            ax3 = sns.lineplot(x,y)
            ax3.set_facecolor('w')
            ax3.tick_params(colors='k')
            ax3.xaxis.label.set_color('k')
            ax3.yaxis.label.set_color('k')
            ax3.set_xlabel('Iterations')
            ax3.set_ylabel('MSE')
            title = 'Mean Squared Error (MSE) by Iteration ' + \
            r'$\alpha$' + " = " + str(self._request['hyper']['alpha']) + \
            r'$\epsilon$' + " = " + str(round(self._request['hyper']['precision'],5)) 
            ax3.set_title(title, color='k')

        fig.tight_layout()
        if show:
            plt.show()
        if path is not None:
            if os.path.exists(os.path.dirname(path)):
                fig.savefig(path, facecolor='w')
            else:
                os.makedirs(os.path.dirname(path))
                fig.savefig(path, facecolor='w')
        plt.close(fig)        

    def animate(self, path=None, nth=1,interval=100,  fps=30):

        # Obtain data
        costs = list(self._detail['cost'])

        # Create iteration vector
        iteration = list(range(0,len(costs)))

        # Extract 100 datapoints plus last for animation
        nth = math.floor(len(costs)/100)
        nth = max(nth,1)
        iteration_plot = iteration[::nth]
        iteration_plot.append(iteration[-1])
        costs_plot = costs[::nth]
        costs_plot.append(costs[-1])


        # Establish figure and axes objects and lines and points
        fig, ax = plt.subplots()
        line, = ax.plot([], [], 'r', lw=1.5)
        point, = ax.plot([], [], 'bo')
        epoch_display = ax.text(.5, 0.9, '', color='k',
                                transform=ax.transAxes, fontsize=12)
        J_display = ax.text(.5, 0.85, '', color='k',transform=ax.transAxes, 
                            fontsize=12)

        # Plot cost line
        ax.plot(iteration_plot, costs_plot, c='r')

        # Set labels and title
        title = self._alg + r' $\alpha$' + " = " + str(round(self._request['hyper']['alpha'],3)) + " " + \
                            r'$\epsilon$' + " = " + str(round(self._request['hyper']['precision'],5)) 
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r'$J(\theta)$')
        ax.set_facecolor('w')
        ax.tick_params(colors='k')
        ax.xaxis.label.set_color('k')
        ax.yaxis.label.set_color('k')
        ax.set_title(title, color='k')

        def init():
            line.set_data([], [])
            point.set_data([], [])
            epoch_display.set_text('')
            J_display.set_text('')

            return(line, point, epoch_display, J_display)

        def animate(i):
            # Animate points
            point.set_data(iteration_plot[i], costs_plot[i])

            # Animate value display
            epoch_display.set_text("Iterations  = " + str(iteration_plot[i]))
            J_display.set_text(r'     $J(\theta)=$' +
                               str(round(costs_plot[i], 4)))

            return(line, point, epoch_display, J_display)

        display = animation.FuncAnimation(fig, animate, init_func=init,
                                          frames=len(costs_plot), interval=interval,
                                          blit=True, repeat_delay=100)
        if path is not None:
            face_edge_colors = {'facecolor': 'w', 'edgecolor': 'w'}
            if os.path.exists(os.path.dirname(path)):
                display.save(path, writer='imagemagick', fps=fps, savefig_kwargs = face_edge_colors)
            else:
                os.makedirs(os.path.dirname(path))                
                display.save(path, writer='imagemagick', fps=fps, savefig_kwargs = face_edge_colors)
        plt.close(fig)   
        return(display)
  

# --------------------------------------------------------------------------- #
#                       BATCH GRADIENT DESCENT CLASS                          #
# --------------------------------------------------------------------------- #

class BGD(GradientDescent):
    '''Batch Gradient Descent'''

    def __init__(self):
        self._alg = "Batch Gradient Descent"
        self._summary = None
        self._detail = None
# --------------------------------------------------------------------------- #
#                     STOCHASTIC GRADIENT DESCENT CLASS                       #
# --------------------------------------------------------------------------- #

class SGD(GradientDescent):
    '''Stochastic Gradient Descent'''

    def __init__(self):
        pass
  
    def search(self, X, y, theta, X_val=None, y_val=None, 
               alpha=0.01, maxiter=0, precision=0.001,
               stop_measure='j', stop_metric='a', check_grad=100,
               scaler='minmax'):

        # Set initial request parameters
        cross_validated = all(v is not None for v in [X_val, y_val])

        # Package request
        self._request = dict()
        self._request['data'] = dict()
        self._request['hyper'] = dict()

        self._request['data']['X'] = X
        self._request['data']['y'] = y
        self._request['data']['X_val'] = X_val
        self._request['data']['y_val'] = y_val        
        self._request['data']['scaler'] = scaler        
        self._request['hyper']['alpha'] = alpha
        self._request['hyper']['theta'] = theta
        self._request['hyper']['maxiter'] = maxiter
        self._request['hyper']['precision'] = precision
        self._request['hyper']['stop_measure'] = stop_measure
        self._request['hyper']['stop_metric'] = stop_metric
        self._request['hyper']['check_grad'] = check_grad
        self._request['hyper']['cross_validated'] = cross_validated

        # Run search and obtain result        
        gd = SGDSearch()
        start = datetime.datetime.now()
        gd.search(self._request)
        end = datetime.datetime.now()

        # Extract search log
        epochs = pd.DataFrame(gd.get_epochs(), columns=['Epochs'])
        iterations = pd.DataFrame(gd.get_iterations(), columns=['Epochs'])
        thetas = self._todf(gd.get_thetas(), stub='theta_')
        costs = pd.DataFrame(gd.get_costs(), columns=['Cost'])
        if cross_validated:
            mse = pd.DataFrame(gd.get_mse(), columns=['MSE'])
        search_log = pd.concat([iterations, thetas, costs], axis=1)

        # Package results
        self._summary = dict()        
        self._summary['detail'] = search_log

        self._summary['summary'] = dict()
        self._summary['summary']['Algorithm'] = gd.get_alg()
        self._summary['summary']['Start'] = start
        self._summary['summary']['End'] = end
        self._summary['summary']['Duration'] = end-start
        self._summary['summary']['Epochs'] = epochs
        self._summary['summary']['X_transformed'] = gd.get_data()[0]
        self._summary['summary']['y_transformed'] = gd.get_data()[1]
        self._summary['summary']['Iterations'] = iterations
        self._summary['summary']['Theta_Init'] = thetas.iloc[0]
        self._summary['summary']['Theta_Final'] = thetas.iloc[-1]
        self._summary['summary']['Cost_Init'] = costs.iloc[0]
        self._summary['summary']['Cost_Final'] = costs.iloc[-1]

        if cross_validated:
            self._summary['summary']['MSE_Init'] = mse.iloc[0]
            self._summary['summary']['MSE_Final'] = mse.iloc[-1]
