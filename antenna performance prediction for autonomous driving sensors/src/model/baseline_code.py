import pandas as pd
import random
import os
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
from sklearn.multioutput import MultiOutputRegressor


class CFG:
    datapath = "antenna performance prediction for autonomous driving sensors/data/"
    trainpath = datapath+'raw/train.csv'
    testpath = datapath+'raw/test.csv'
    submission = datapath+'raw/sample_submission.csv'
    outpath = datapath+'processed/'

def seedEverything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
def standardization(df):
    return (df-df.mean(numeric_only=True))/df.std(numeric_only=True)

def normalization(df):
    return (df-df.min())/(df.max()-df.min())
    
def baseLine(train_x, train_y):
    LR = MultiOutputRegressor(LinearRegression()).fit(train_x, train_y)
    test_x = pd.read_csv(CFG.testpath).drop(columns=['ID'])
    preds = LR.predict(test_x)
    submit = pd.read_csv(CFG.submission)
    for idx, col in enumerate(submit.columns):
        if col=='ID':
            continue
        submit[col] = preds[:,idx-1]
    return submit

def baseLine2(train_x, train_y):
    train_x = train_x.drop(columns=['X_02','X_04','X_10','X_11','X_23','X_46','X_47','X_48',
                                    'X_19','X_20','X_21','X_22'])
    LR = MultiOutputRegressor(LinearRegression()).fit(train_x, train_y)
    test_x = pd.read_csv(CFG.testpath).drop(columns=['ID','X_02','X_04','X_10','X_11','X_23','X_46','X_47','X_48',   
                                                     'X_19','X_20','X_21','X_22'])
    preds = LR.predict(test_x)
    submit = pd.read_csv(CFG.submission)
    for idx, col in enumerate(submit.columns):
        if col=='ID':
            continue
        submit[col] = preds[:,idx-1]
    return submit

def report():
    # {BayesianRidge-> 1.9749297768
    # ['X_02','X_04','X_10','X_11','X_23','X_46','X_47','X_48'] Drop -> 1.9746318023
    # }
    pass

if __name__ == "__main__":
    seedEverything(42) # Seed 고정
    train_df = pd.read_csv(CFG.trainpath)

    train_x = train_df.filter(regex='X') # Input : X Featrue
    train_y = train_df.filter(regex='Y') # Output : Y Feature

    submit = baseLine2(train_x, train_y)
    submit.to_csv(CFG.outpath+'submit.csv', index=False)

