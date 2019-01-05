
# %%
# =========================================================================== #
#                             GRADIENT VISUAL                                 #
# =========================================================================== #

# --------------------------------------------------------------------------- #
#                                LIBRARIES                                    #
# --------------------------------------------------------------------------- #
import inspect
import os
import sys

from IPython.display import HTML
import datetime
import itertools
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('seaborn-notebook')
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from matplotlib import animation, rc
from matplotlib import colors as mcolors
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import rcParams
from mpl_toolkits.mplot3d import axes3d, Axes3D
import numpy as np
from numpy import array, newaxis
import pandas as pd
import seaborn as sns
from textwrap import wrap

from GradientDescent import BGD, SGD, MBGD
from utils import save_fig, save_csv

rcParams['animation.embed_limit'] = 60
rc('animation', html='jshtml')
rc
# --------------------------------------------------------------------------- #
#                          GRADIENTLAB BASE CLASS                             #  
# --------------------------------------------------------------------------- #
class GradientLab:
    '''
    Base class for gradient descent plots
    '''

    def __init__(self):
        self._alg = 'Gradient Descent'
        self._summary = None
        self._detail = None
        self._params = {}

    def _save_params(self, X, y, X_val, y_val, theta, alpha, precision, 
               stop_parameter, stop_metric, miniter, maxiter, scaler):
        self._params = {'theta':theta, 'alpha':alpha, 'precision':precision,
                        'stop_parameter': stop_parameter,
                        'stop_metric': stop_metric, 
                        'miniter': miniter,
                        'maxiter':maxiter,
                        'scaler':scaler}
    def summary(self):
        if self._summary is None:
            raise Exception("No summary to report")
        else:
            return(self._summary)

    def get_detail(self):
        if self._detail is None:
            raise Exception("No detail to report")
        else:
            return(self._detail)        
        
    def gridsearch(self, X, y, X_val, y_val, theta, alpha, precision, 
                   n_iter_no_change=5, maxiter=0, scaler='minmax'):

        self._save_params(X, y, X_val, y_val, theta, alpha, precision, 
               n_iter_no_change, maxiter, scaler)

        bgd = BGD()
        experiment = 1
        self._summary = pd.DataFrame()
        self._detail = pd.DataFrame()
        for n in n_iter_no_change:
            for a in alpha:
                for p in precision:
                    bgd.fit(X=X, y=y, theta=theta, X_val=X_val, y_val=y_val, 
                            alpha=a, maxiter=maxiter, precision=p, 
                            n_iter_no_change=n, scaler=scaler)
                    detail = bgd.get_detail()
                    summary = bgd.summary()
                    summary['experiment'] = experiment
                    self._detail = pd.concat([self._detail, detail], axis=0)    
                    self._summary = pd.concat([self._summary, summary], axis=0)    
                    experiment += 1               

    def _get_label(self, x):
        labels = {'alpha': 'Learning Rate',
                  'precision': 'Precision',
                  'theta': "Theta",
                  'batch_size': 'Batch Size',
                  'final_costs': 'Training Set Costs',
                  'final_costs_val': 'Validation Set Costs'}
        return(labels[x])

    def report(self,  n=None, sort='v', directory=None, filename=None):
        if self._detail is None:
            raise Exception('Nothing to report')
        else:
            vars = ['experiment', 'alg', 'alpha', 'precision', 
                    'n_iter_no_change', 'maxiter', 
                    'epochs', 'iterations','duration',
                    'final_costs', 'final_costs_val']
            df = self._summary
            df = df[vars]
            if sort == 't':
                df = df.sort_values(by=['final_costs', 'duration'])
            else:
                df = df.sort_values(by=['final_costs_val', 'duration'])
            if directory:
                if filename is None:
                    filename = self._alg + ' Grid Search.csv'
                save_csv(df, directory, filename)                
            if n:
                df = df.iloc[:n]            
            return(df)

    def plot_alpha(self, directory=None, filename=None, show=True):
        df = self._summary

        # Obtain and initialize figure and settings
        sns.set(style="whitegrid", font_scale=1)        

        # Plot costs by learning rate and stop condition
        fig = plt.figure(figsize=(12,4))
        ax0 = plt.subplot2grid((1,2), (0,0))  
        ax0 = sns.boxplot(x='alpha', y='final_costs_val', data=df)
        ax0.set_facecolor('w')
        ax0.tick_params(colors='k')
        ax0.xaxis.label.set_color('k')
        ax0.yaxis.label.set_color('k')
        ax0.set_xlabel('Alpha')
        ax0.set_ylabel('Costs')

        # Zoom in top 1/2
        df2 = df.groupby('alpha')['final_costs_val'].min()        
        n_smallest=int(max(1,math.floor(df2.shape[0]/2)))
        alphas = df2.nsmallest(n=n_smallest)
        df2 = df[df.alpha.isin(list(alphas.index))]        
        ax1 = plt.subplot2grid((1,2), (0,1))  
        ax1 = sns.boxplot(x='alpha', y='final_costs_val', data=df2)
        ax1.set_facecolor('w')
        ax1.tick_params(colors='k')
        ax1.xaxis.label.set_color('k')
        ax1.yaxis.label.set_color('k')
        ax1.set_xlabel('Alpha')
        ax1.set_ylabel('Costs')

        suptitle = self._alg + '\n' + ' Validation Set Costs ' + '\n' + 'By Learning Rate'
        fig.suptitle(suptitle)
        fig.tight_layout(rect=[0,0,1,0.85])        
        if show:
            plt.show()
        if directory is not None:
            if filename is None:
                filename = suptitle.replace('\n', '')
                filename = filename.replace('  ', ' ')
                filename = filename.replace(':', '') + '.png'
                save_fig(fig, directory, filename)     
                filename=None                    
        plt.close(fig)       
        
        # Plot time by learning rate and stop condition         
        fig = plt.figure(figsize=(12,4))
        ax0 = plt.subplot2grid((1,2), (0,0))  
        ax0 = sns.boxplot(x='alpha', y='duration', data=df)
        ax0.set_facecolor('w')
        ax0.tick_params(colors='k')
        ax0.xaxis.label.set_color('k')
        ax0.yaxis.label.set_color('k')
        ax0.set_xlabel('Alpha')
        ax0.set_ylabel('Time (ms)')

        # Zoom in on top 1/2         
        df2 = df.groupby('alpha')['duration'].min()        
        n_smallest=int(max(1,math.floor(df2.shape[0]/2)))
        alphas = df2.nsmallest(n=n_smallest, keep='last')
        df2 = df[df.alpha.isin(list(alphas.index))]           
        ax1 = plt.subplot2grid((1,2), (0,1))  
        ax1 = sns.boxplot(x='alpha', y='duration', data=df2)
        ax1.set_facecolor('w')
        ax1.tick_params(colors='k')
        ax1.xaxis.label.set_color('k')
        ax1.yaxis.label.set_color('k')
        ax1.set_xlabel('Alpha')
        ax1.set_ylabel('Time (ms)')        

        # Finalize plot and save
        suptitle = self._alg + '\n' + ' Computation Time ' + '\n' + 'By Learning Rate'
        fig.suptitle(suptitle)
        fig.tight_layout(rect=[0,0,1,0.85])           
        if show:
            plt.show()
        if directory is not None:
            if filename is None:
                filename = suptitle.replace('\n', '')
                filename = filename.replace('  ', ' ')
                filename = filename.replace(':', '') + '.png'
                save_fig(fig, directory, filename)     
                filename=None                    
        plt.close(fig)       
        return(fig) 

    def plot_precision(self, directory=None, filename=None, show=True):
        df = self._summary

        # Obtain and initialize figure and settings
        sns.set(style="whitegrid", font_scale=1)
        fig = plt.figure(figsize=(12,6))

        # Plot costs by learning rate 
        ax0 = plt.subplot2grid((1,2), (0,0))  
        ax0 = sns.boxplot(x='precision', y='final_costs_val', data=df)
        ax0.set_facecolor('w')
        ax0.tick_params(colors='k')
        ax0.xaxis.label.set_color('k')
        ax0.yaxis.label.set_color('k')
        ax0.set_xlabel('Precision')
        ax0.set_ylabel('Costs')

        # Zoom in
        ax1 = plt.subplot2grid((1,2), (0,1))  
        ax1 = sns.boxplot(x='precision', y='final_costs_val', data=df)
        ax1.set_facecolor('w')
        ax1.tick_params(colors='k')
        ax1.xaxis.label.set_color('k')
        ax1.yaxis.label.set_color('k')
        ax1.set_xlabel('Precision')
        ax1.set_ylabel('Costs')
        ax1.set_ylim(0,0.2)

        suptitle = self._alg + '\n' + ' Validation Set Costs '+ \
                    '\n' + ' By Precision'
        fig.suptitle(suptitle)
        fig.tight_layout(rect=[0,0,1,.85])
        if show:
            plt.show()
        if directory is not None:
            if filename is None:
                filename = suptitle.replace('\n', '')
                filename = filename.replace('  ', ' ')
                filename = filename.replace(':', '') + '.png'
                save_fig(fig, directory, filename)     
                filename=None    
        plt.close(fig)   

        fig = plt.figure(figsize=(12,6))
        # Plot computation time by learning rate         
        ax0 = plt.subplot2grid((1,2), (0,0))  
        ax0 = sns.boxplot(x='precision', y='duration', data=df)
        ax0.set_facecolor('w')
        ax0.tick_params(colors='k')
        ax0.xaxis.label.set_color('k')
        ax0.yaxis.label.set_color('k')
        ax0.set_xlabel('Precision')
        ax0.set_ylabel('Time (ms)')

        # Zoom in
        ax1 = plt.subplot2grid((1,2), (0,1))  
        ax1 = sns.boxplot(x='precision', y='duration', data=df)
        ax1.set_facecolor('w')
        ax1.tick_params(colors='k')
        ax1.xaxis.label.set_color('k')
        ax1.yaxis.label.set_color('k')
        ax1.set_xlabel('Precision')
        ax1.set_ylabel('Time (ms)')
        ax1.set_ylim(0,0.015)

        # Finalize plot and save
        suptitle = self._alg + '\n' + ' Computation Time '+ \
                    '\n' + ' By Precision'
        fig.suptitle(suptitle)
        fig.tight_layout(rect=[0,0,1,.85])
        if show:
            plt.show()
        if directory is not None:
            if filename is None:
                filename = suptitle.replace('\n', '')
                filename = filename.replace('  ', ' ')
                filename = filename.replace(':', '') + '.png'
                save_fig(fig, directory, filename)      
                filename=None   
        plt.close(fig)            

    def plot_costs(self, x='alpha', y='final_costs_val', z='precision', directory=None, 
                   show=True):
        
        for row_name, row_data in plots:
            ax = plt.subplot2grid((1,cols), (0,i))            
            ax = sns.barplot(x=x, y='final_costs_val', hue=z, data=row_data)
            ax.set_facecolor('w')
            ax.tick_params(colors='k')
            ax.xaxis.label.set_color('k')
            ax.yaxis.label.set_color('k')
            ax.set_xlabel(self._get_label(x))
            ax.set_ylabel('Costs')
            title = 'Validation Set Costs' + '\n' + row_name + ' in ' + fig_name
            ax.set_title(title, color='k')
            i += 1
        
        # Finalize plot and save
        suptitle = self._alg + '\n' + ' Validation Set Cost Analysis' + '\n' + ' Stop Parameter: ' + \
                    fig_name 
        fig.suptitle(suptitle)
        fig.tight_layout(rect=[0,0,1,.9])
        if show:
            plt.show()
        if directory is not None:
            filename = suptitle.replace('\n', '')
            filename = filename.replace('  ', ' ')
            filename = filename.replace(':', '') + '.png'
            save_fig(fig, directory, filename)                

        plt.close(fig)
        return(fig)

    def plot_times(self, x, z, fig_key='stop_parameter', row_key='stop_condition',
                     directory=None, filename=None, show=True):
        # Group data and obtain keys
        figures = self._summary.groupby(fig_key)
        for fig_name, fig_data in figures:
            plots = fig_data.groupby(row_key)   

            # Set Grid Dimensions
            cols = 2  
            rows = plots.ngroups
            
            # Obtain and initialize matplotlib figure
            fig = plt.figure(figsize=(12,4*rows))        
            sns.set(style="whitegrid", font_scale=1)

            # Render plots
            i = 0
            for row_name, row_data in plots:
                ax0 = plt.subplot2grid((rows,cols), (i,0))            
                ax0 = sns.barplot(x=x, y='duration', hue=z, data=row_data)
                ax0.set_facecolor('w')
                ax0.tick_params(colors='k')
                ax0.xaxis.label.set_color('k')
                ax0.yaxis.label.set_color('k')
                ax0.set_xlabel(self._get_label(x))
                ax0.set_ylabel('(ms)')
                title = 'Elapsed Time (ms)' + '\n' + row_name
                ax0.set_title(title, color='k')

                ax1 = plt.subplot2grid((rows,cols), (i,1))
                ax1 = sns.barplot(x=x, y='iterations', hue=z, data=row_data)
                ax1.set_facecolor('w')
                ax1.tick_params(colors='k')
                ax1.xaxis.label.set_color('k')
                ax1.yaxis.label.set_color('k')
                ax1.set_xlabel(self._get_label(x))
                ax1.set_ylabel('Iterations')
                title = 'Iterations' + '\n' + row_name
                ax1.set_title(title, color='k')
                i += 1
            
            # Finalize plot and save
            suptitle = self._alg + '\n' + 'Time Analysis' 
            fig.suptitle(suptitle, y=1.1) 
            fig.tight_layout()
            if show:
                plt.show()
            if directory is not None:
                if filename is None:
                    filename = suptitle.replace('\n', '')
                    filename = filename.replace('  ', ' ')
                    filename = filename.replace(':', '') + '.png'
                save_fig(fig, directory, filename)
            plt.close(fig)
        return(fig)

    def plot_curves(self, stop_parameter=None, stop_metric=None,
                    precision=None, directory=None, filename=None, show=True):

        # Filter data
        df = self._detail
        if stop_parameter:
            df = df[df.stop_parameter == stop_parameter]
        if stop_metric:
            df = df[df.stop_metric == stop_metric]
        if precision:
            df = df[df.precision == precision]
        # Group data and obtain keys            
        figures = df.groupby('stop_parameter')
        for fig_name, fig_data in figures:
            plots = fig_data.groupby('stop_metric')

            # Set Grid Dimensions
            cols = 2  
            rows = math.ceil(plots.ngroups/cols)
            odd = True if plots.ngroups % cols != 0 else False

            # Obtain and initialize matplotlib figure
            fig = plt.figure(figsize=(12,6*rows))        
            sns.set(style="whitegrid", font_scale=1)
            palette = sns.color_palette("muted")

            # Render plots
            i = 0
            for row_name, row_data in plots:
                if odd and i == plots.ngroups-1:
                    ax = plt.subplot2grid((rows,cols), (int(i/cols),0), colspan=cols)
                else:
                    ax = plt.subplot2grid((rows,cols), (int(i/cols),i%cols))
                ax = sns.lineplot(x='iterations', y='cost', hue='alpha', 
                                  palette=palette,
                                  data=row_data, legend='full')
                ax.set_facecolor('w')
                ax.tick_params(colors='k')
                ax.xaxis.label.set_color('k')
                ax.yaxis.label.set_color('k')
                ax.set_xlabel('Iterations')
                ax.set_ylabel('Cost')
                title = row_data.stop.iloc[0]
                ax.set_title(title, color='k')
                i += 1

            # Finalize plot and save            
            suptitle = self._alg + '\n' + ' Learning Curves ' 
            if i == 1:
                title = suptitle + '\n' + title
                ax.set_title(title, color='k')
            else:
                fig.suptitle(suptitle)
            # Finalize plot and save
            fig.tight_layout(rect=[0,0,1,.85])
            if show:
                plt.show()
            if directory is not None:
                if filename is None:
                    filename = suptitle.replace('\n', '')
                    filename = filename.replace('  ', ' ')
                    filename = filename.replace(':', '') + '.png'
                save_fig(fig, directory, filename)
                filename=None
            plt.close(fig)
        return(fig)






# --------------------------------------------------------------------------- #
#                              BGD Lab Class                                  #   
# --------------------------------------------------------------------------- #
class BGDLab(GradientLab):  

    def __init__(self):      
        self._alg = 'Batch Gradient Descent'
        self._summary = None
        self._detail = None
        
# --------------------------------------------------------------------------- #
#                              SGD Lab Class                                  #   
# --------------------------------------------------------------------------- #
class SGDLab(GradientLab):  

    def __init__(self):      
        self._alg = 'Stochastic Gradient Descent'
        self._summary = None
        self._detail = None

    def plot_check_point(self, directory=None, filename=None, show=True):
        # Group data and obtain keys
        df = self._summary

        # Obtain and initialize figure and settings
        sns.set(style="whitegrid", font_scale=1)
        fig = plt.figure(figsize=(12,6))

        # Plot costs by learning rate and stop condition
        ax0 = plt.subplot2grid((1,2), (0,0))  
        ax0 = sns.boxplot(x='check_point', y='final_costs_val', hue='stop_condition', data=df)
        ax0.set_facecolor('w')
        ax0.tick_params(colors='k')
        ax0.xaxis.label.set_color('k')
        ax0.yaxis.label.set_color('k')
        ax0.set_xlabel('Check Point')
        ax0.set_ylabel('Costs')

        # Zoom in
        ax1 = plt.subplot2grid((1,2), (0,1))  
        ax1 = sns.boxplot(x='check_point', y='final_costs_val', hue='stop_condition', data=df)
        ax1.set_facecolor('w')
        ax1.tick_params(colors='k')
        ax1.xaxis.label.set_color('k')
        ax1.yaxis.label.set_color('k')
        ax1.set_xlabel('Check Point')
        ax1.set_ylabel('Costs')
        ax1.set_ylim(0,0.2)

        suptitle = self._alg + '\n' + ' Validation Set Costs '+ \
                    '\n' + ' By Check Point and Stop Condition'
        fig.suptitle(suptitle)
        fig.tight_layout(rect=[0,0,1,.85])
        if show:
            plt.show()
        if directory is not None:
            if filename is None:
                filename = suptitle.replace('\n', '')
                filename = filename.replace('  ', ' ')
                filename = filename.replace(':', '') + '.png'
                save_fig(fig, directory, filename)     
                filename=None    
        plt.close(fig)   

        fig = plt.figure(figsize=(12,6))
        # Plot computation time by learning rate and stop condition         
        ax0 = plt.subplot2grid((1,2), (0,0))  
        ax0 = sns.boxplot(x='check_point', y='duration', hue='stop_condition', data=df)
        ax0.set_facecolor('w')
        ax0.tick_params(colors='k')
        ax0.xaxis.label.set_color('k')
        ax0.yaxis.label.set_color('k')
        ax0.set_xlabel('Check Point')
        ax0.set_ylabel('Time (ms)')

        # Zoom in
        ax1 = plt.subplot2grid((1,2), (0,1))  
        ax1 = sns.boxplot(x='check_point', y='duration', hue='stop_condition', data=df)
        ax1.set_facecolor('w')
        ax1.tick_params(colors='k')
        ax1.xaxis.label.set_color('k')
        ax1.yaxis.label.set_color('k')
        ax1.set_xlabel('Check Point')
        ax1.set_ylabel('Time (ms)')
        ax1.set_ylim(0,0.015)

        # Finalize plot and save
        suptitle = self._alg + '\n' + ' Computation Time '+ \
                    '\n' + ' By Check Point and Stop Condition'
        fig.suptitle(suptitle)
        fig.tight_layout(rect=[0,0,1,.85])
        if show:
            plt.show()
        if directory is not None:
            if filename is None:
                filename = suptitle.replace('\n', '')
                filename = filename.replace('  ', ' ')
                filename = filename.replace(':', '') + '.png'
                save_fig(fig, directory, filename)      
                filename=None   
        plt.close(fig)         

    def report(self,  n=None, sort='v', directory=None, filename=None):
        if self._detail is None:
            raise Exception('Nothing to report')
        else:
            vars = ['experiment', 'alg', 'alpha', 'precision', 
                    'miniter', 'maxiter', 'check_point',
                    'epochs', 'iterations','duration',
                    'stop_parameter', 'stop_metric', 
                    'final_costs', 'final_costs_val']
            df = self._summary
            df = df[vars]
            if sort == 't':
                df = df.sort_values(by=['final_costs'])
            else:
                df = df.sort_values(by=['final_costs_val'])
            if directory:
                save_csv(df, directory, filename)                
            if n:
                df = df.iloc[:n]            
            return(df)

    def gridsearch(self, X, y, X_val, y_val, theta, alpha, precision, 
            stop_parameter, stop_metric, check_point, miniter=0, maxiter=0, 
            scaler='minmax'):
        sgd = SGD()
        experiment = 1
        self._summary = pd.DataFrame()
        self._detail = pd.DataFrame()
        
        for cp in check_point:
            for measure in stop_parameter:
                for metric in stop_metric:
                    for a in alpha:
                        for p in precision:
                            sgd.fit(X=X, y=y, theta=theta, X_val=X_val, y_val=y_val, 
                                    alpha=a, miniter=miniter, maxiter=maxiter, 
                                    precision=p, stop_parameter=measure, 
                                    stop_metric=metric, check_point=cp, scaler=scaler)
                            detail = sgd.get_detail()
                            summary = sgd.summary()
                            summary['experiment'] = experiment
                            self._detail = pd.concat([self._detail, detail], axis=0)    
                            self._summary = pd.concat([self._summary, summary], axis=0)    
                            experiment += 1 

    def plot_curves(self, stop_parameter=None, stop_metric=None, precision=None, 
                    check_point=None, directory=None, filename=None, show=True):

        # Filter data
        df = self._detail
        if stop_parameter:
            df = df[df.stop_parameter == stop_parameter]
        if stop_metric:
            df = df[df.stop_metric == stop_metric]
        if precision:
            df = df[df.precision == precision]
        if check_point:
            df = df[df.check_point == check_point]
        # Group data and obtain keys            
        figures = df.groupby('stop_parameter')
        for fig_name, fig_data in figures:
            plots = fig_data.groupby('stop_metric')
            for plot_name, plot_data in plots:
                prec = plot_data.groupby('precision')  
                for prec_name, prec_data in prec:
                    cp = prec_data.groupby('check_point')

                    # Set Grid Dimensions
                    cols = 2  
                    rows = math.ceil(cp.ngroups/cols)
                    odd = True if cp.ngroups % cols != 0 else False

                    # Obtain and initialize matplotlib figure
                    fig = plt.figure(figsize=(12,6*rows))        
                    sns.set(style="whitegrid", font_scale=1)

                    # Render plots
                    i = 0
                    for cp_name, cp_data in cp:
                        if odd and i == cp.ngroups-1:
                            ax = plt.subplot2grid((rows,cols), (int(i/cols),0), colspan=cols)
                        else:
                            ax = plt.subplot2grid((rows,cols), (int(i/cols),i%cols))
                        ax = sns.lineplot(x='iterations', y='cost', hue='alpha', data=cp_data, legend='full')
                        ax.set_facecolor('w')
                        ax.tick_params(colors='k')
                        ax.xaxis.label.set_color('k')
                        ax.yaxis.label.set_color('k')
                        ax.set_xlabel('Iterations')
                        ax.set_ylabel('Cost')
                        title = ' Check Point: ' + str(cp_data.check_point.iloc[0])
                        ax.set_title(title, color='k')
                        i += 1

                    # Finalize plot and save            
                    suptitle = self._alg + '\n' + ' Learning Curves ' + '\n' + cp_data.stop.iloc[0]
                    if i == 1:
                        title = suptitle + '\n' + title
                        ax.set_title(title, color='k')
                    else:
                        fig.suptitle(suptitle)
                    # Finalize plot and save
                    fig.tight_layout(rect=[0,0,1,.9])
                    if show:
                        plt.show()
                    if directory is not None:
                        if filename is None:
                            filename = suptitle.replace('\n', '')
                            filename = filename.replace('  ', ' ')
                            filename = filename.replace(':', '') + '.png'
                        save_fig(fig, directory, filename)
                        filename=None
                    plt.close(fig)

# --------------------------------------------------------------------------- #
#                              MBGD Lab Class                                 #   
# --------------------------------------------------------------------------- #
class MBGDLab(GradientLab):  

    def __init__(self):      
        self._alg = 'Minibatch Gradient Descent'
        self._summary = None
        self._detail = None

    def gridsearch(self, X, y, X_val, y_val, theta, alpha, precision, 
            stop_parameter, stop_metric, batch_size, miniter=0, maxiter=0, 
            scaler='minmax'):
        mbgd = MBGD()
        self._summary = pd.DataFrame()
        self._detail = pd.DataFrame()

        for bs in batch_size:
            for measure in stop_parameter:
                for metric in stop_metric:
                    for a in alpha:
                        for p in precision:
                            mbgd.fit(X=X, y=y, theta=theta, X_val=X_val, y_val=y_val, 
                                    alpha=a, miniter=miniter, maxiter=maxiter, precision=p, stop_parameter=measure, 
                                    stop_metric=metric, batch_size=bs, scaler=scaler)
                            detail = mbgd.get_detail()
                            summary = mbgd.summary()
                            self._detail = pd.concat([self._detail, detail], axis=0)    
                            self._summary = pd.concat([self._summary, summary], axis=0)    
