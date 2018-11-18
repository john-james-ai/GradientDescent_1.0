# %%
import pandas as pd


def read(what='linear'):
    if what == 'linear':
        path = "./data/linear.csv"
    else:
        path = "./data/logistic.csv"
    df = pd.read_csv(path, encoding="Latin-1", low_memory=False)
    return(df)
