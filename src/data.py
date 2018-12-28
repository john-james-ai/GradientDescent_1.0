# %%
# ============================================================================ #
#                                 LIBRARIES                                    #
# ============================================================================ #
import inspect
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# %%
# ---------------------------------------------------------------------------- #
#                                   LOAD                                       #
# ---------------------------------------------------------------------------- #
def load(vars=None):
    '''
    Loads the raw training data, selects some features, does some basic data
    preprocessing and saves  the data in the interim folder.    
    Args:
        vars: None or list of column names to load.  If None all columns will
                be loaded
    Returns:
        pd.DataFrame 
    '''
    df = pd.read_csv("./data/raw/train.csv",
                     encoding="Latin-1", low_memory=False)

    # Select Variables
    if vars is not None:            
        df = df[vars]
        df = df.dropna()

    # Create interim directory if not exists and save data
    if not os.path.exists("./data/interim"):
        os.makedirs("./data/interim")

    # Save the training data
    df.to_csv(os.path.join("./data/interim/train.csv"),
              index=False, index_label=False)

# ---------------------------------------------------------------------------- #
#                                   AMES                                       #
# ---------------------------------------------------------------------------- #    

def ames(train=0.7, test=None):
    '''
    Splits training data into a training and test.
    Args:
        train(num): Int or float.  If int, the training set size, if float, 
                    the proportion of the training set to obtain
        test(num):  Int or float.  If int, the test set size, if float, 
                    the proportion of the training set to allocate to test
    Return:
        X_train (pd.Series): Training set input data
        y_train (pd.Series): Training set output data
        X_test (pd.Series): Test set input data
        y_test (pd.Series): Test set output data
    '''
    X =  ["MSSubClass",	"MSZoning",	"LotFrontage",	"LotArea",	"LotConfig",
     	 "Neighborhood","BldgType",	"HouseStyle",	"OverallQual",	"OverallCond",	
         "YearBuilt",	"YearRemodAdd",	"ExterQual",	"ExterCond",	
         "1stFlrSF",	"GrLivArea",	"FullBath",	"HalfBath",
         "BedroomAbvGr","KitchenAbvGr",	"KitchenQual",	"TotRmsAbvGrd",	"GarageCars"]
    y = ['SalePrice']

    df = pd.read_csv("./data/interim/train.csv",
                     encoding="Latin-1", low_memory=False)

    if train < 1:
        test = 1 - train
    else:
        test = df.shape[0] - train

    X = df[X]
    y = df[y]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train, test_size=test, shuffle=True, random_state=5)
    return(X_train, X_test, y_train, y_test)

# ---------------------------------------------------------------------------- #
#                                   AMES                                       #
# ---------------------------------------------------------------------------- #    

def demo(n=500, cv=True):
    '''
    Reads data from file into a dataframe and splits the dataframe into 
    a training and an optional validation set.
    Args:
        n(num): Int or float.  If int, the training set size, if float, 
                the proportion of the observations to return as part of 
                the training set. 
        cv(bool): Yes if a validation set should be returned for 
                  cross-validation. The validation set will be the complement
                  of the training set.
    Return:
        X_train (pd.DataFrame): Training set input data
        y_train (pd.DataFrame): Training set output data
        X_val (pd.DataFrame): Validation set input data (optional)
        y_val (pd.DataFrame): Validation set output data (optional)
    '''
    X = ["GrLivArea"]
    y = ['SalePrice']

    df = pd.read_csv("./data/interim/train.csv",
                     encoding="Latin-1", low_memory=False)

    X = df[X]
    y = df[y]

    if cv:
        if n < 1:
            test = 1 - n
        else:
            test = df.shape[0] - n        
        X_train, X_val, y_train, y_val = train_test_split(
                X, y, train_size=n, test_size=test, shuffle=True, random_state=5)
        return(X_train, X_val, y_train, y_val)    
    else:
        test = None
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, train_size=n, test_size=test, shuffle=True, random_state=5)
        return(X_train, X_val, y_train, y_val)    
#%%        
