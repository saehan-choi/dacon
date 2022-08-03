
import pandas as pd
import random
import os
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

import xgboost as xgb
import random


import datetime as dt
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import random
import lightgbm as lgb
import re
from sklearn.metrics import *
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings(action='ignore')

from pycaret.regression import * 

SEED = 42

class CFG:
    datapath = "antenna performance prediction for autonomous driving sensors/data/"
    trainpath = datapath+'raw/train.csv'
    testpath = datapath+'raw/test.csv'
    submission = datapath+'raw/sample_submission.csv'
    outpath = datapath+'processed/'
    
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(SEED) # Seed 고정

train_df = pd.read_csv(CFG.trainpath)

train_x = train_df.filter(regex='X') # Input : X Featrue
train_y = train_df.filter(regex='Y') # Output : Y Feature

train_01 = pd.concat([train_x, train_y['Y_01']], axis=1)
train_02 = pd.concat([train_x, train_y['Y_02']], axis=1)
train_03 = pd.concat([train_x, train_y['Y_03']], axis=1)
train_04 = pd.concat([train_x, train_y['Y_04']], axis=1)
train_05 = pd.concat([train_x, train_y['Y_05']], axis=1)
train_06 = pd.concat([train_x, train_y['Y_06']], axis=1)
train_07 = pd.concat([train_x, train_y['Y_07']], axis=1)
train_08 = pd.concat([train_x, train_y['Y_08']], axis=1)
train_09 = pd.concat([train_x, train_y['Y_09']], axis=1)
train_10 = pd.concat([train_x, train_y['Y_10']], axis=1)
train_11 = pd.concat([train_x, train_y['Y_11']], axis=1)
train_12 = pd.concat([train_x, train_y['Y_12']], axis=1)
train_13 = pd.concat([train_x, train_y['Y_13']], axis=1)
train_14 = pd.concat([train_x, train_y['Y_14']], axis=1)

# model = setup(train_01, target = 'Y_01', session_id=SEED )
# compare_models()

lgbm_01 = create_model('lightgbm', fold=10)
