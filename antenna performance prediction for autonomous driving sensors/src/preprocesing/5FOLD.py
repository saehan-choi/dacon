import pandas as pd
import random
import os
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

import matplotlib.pyplot as plt
import seaborn as sns

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

def heatmap(df):
    plt.pcolor(df)  
    plt.colorbar()
    plt.show()


    
df = pd.read_csv(CFG.trainpath)

print(df.describe())

print(df.head())


print(df.describe())

# 5FOLD 만들어주징

# 편차^2 / 평균 = 분산