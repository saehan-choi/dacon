import pandas as pd
import random
import os
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

class CFG:
    trainsetPath = './antenna performance prediction for autonomous driving sensors/dataset/train.csv'
    testsetPath = './antenna performance prediction for autonomous driving sensors/dataset/test.csv'
    
    
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
if __name__ == "__main__":
    seed_everything(42) # Seed 고정

    train_df = pd.read_csv(CFG.trainsetPath)

    train_x = train_df.filter(regex='X') # Input : X Featrue
    train_y = train_df.filter(regex='Y') # Output : Y Feature

    LR = MultiOutputRegressor(LinearRegression()).fit(train_x, train_y)
    print('Done.')
    test_x = pd.read_csv(CFG.testsetPath).drop(columns=['ID'])
    preds = LR.predict(test_x)
    print('Done.')
    submit = pd.read_csv('./antenna performance prediction for autonomous driving sensors/dataset/sample_submission.csv')
    for idx, col in enumerate(submit.columns):
        if col=='ID':
            continue
        submit[col] = preds[:,idx-1]
    print('Done.')
    submit.to_csv('./submit.csv', index=False)