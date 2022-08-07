import pandas as pd
import random
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_validate, train_test_split

from sklearn.metrics import mean_squared_error


from flaml import AutoML
from flaml.data import load_openml_dataset
from flaml.ml import sklearn_metric_loss_score

class CFG:
    dataPath = "aa/data/"
    trainPath = dataPath+'raw/train.csv'
    testPath = dataPath+'raw/test.csv'
    submission = dataPath+'raw/sample_submission.csv'
    outPath = dataPath+'processed/'
    weightsavePath = dataPath+'weights/'

    seed = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def data_preprosessing(train_df, test_df):

    train_X = train_df.filter(regex='X') # Input : X Featrue
    train_Y = train_df.filter(regex='Y') # Output : Y Feature

    train_X = train_X.drop(['X_04', 'X_23', 'X_47', 'X_48'], axis=1)
    test_Y = test_df.drop(['X_04', 'X_23', 'X_47', 'X_48'], axis=1)

    train_X, test_X, train_Y, test_Y = train_test_split(train_X, train_Y, test_size=0.2, random_state=42)
    train_X, val_X, train_Y, val_Y = train_test_split(train_X, train_Y, test_size=0.2, random_state=42)

    return train_X, train_Y, val_X, val_Y, test_X, test_Y

seed_everything(CFG.seed) # Seed 고정

train_df = pd.read_csv(CFG.trainPath) # train.csv load
test_df = pd.read_csv(CFG.testPath).drop(columns=['ID']) # test.csv load

train_X, train_Y, val_X, val_Y, test_X, test_Y = data_preprosessing(train_df, test_df)

submit = pd.read_csv(CFG.submission)
submit_sample = pd.read_csv(CFG.submission).iloc[0:len(test_Y),:]


for i in range(1, 15):
    y_number = 'Y_'+ str(format(i, '02'))

    automl = AutoML()
    settings = {
        "time_budget": 6000,  # total running time in seconds
        # "time_budget": 6200s -> spend 1 day,  # total running time in seconds
        "metric": 'rmse',  # primary metrics for regression can be chosen from: ['mae','mse','r2']
        "estimator_list": ['lgbm'],  # list of ML learners; we tune lightgbm in this example
        "task": 'regression',  # task type
        "log_file_name": (CFG.outPath+y_number+'.log'),  # flaml log file
        "seed": CFG.seed,    # random seed
    }

    automl.fit(X_train=train_X, y_train=train_Y[y_number], **settings)

    # 결과 샘플 csv로 추출
    y_pred = automl.predict(test_X)
    y_pred_df = pd.DataFrame(y_pred, columns=[y_number])
    for idx, col in enumerate(submit_sample.columns):
        if col==y_number:   
            submit_sample[col] = y_pred_df
            break

    submit_sample.to_csv(CFG.outPath+'submit_sample_LightGBM.csv', index=False)
    submit_sample = pd.read_csv(CFG.outPath+'submit_sample_LightGBM.csv')

    # 제출 csv 추출
    y_pred_final = automl.predict(test_X)
    y_pred_final_df = pd.DataFrame(y_pred_final, columns=[y_number])
    for idx, col in enumerate(submit.columns):
        if col==y_number:
            submit[col] = y_pred_final_df
            break
    submit.to_csv(CFG.outPath+'submit_LightGBM.csv', index=False)
    submit = pd.read_csv(CFG.outPath+'submit_LightGBM.csv')

    # print('Predicted labels', y_pred)
    # print('rmse', '=', sklearn_metric_loss_score('rmse', y_pred, test_Y[y_number]))
    print(y_number + ' Done **************************************************************************************************************')


submit_sample_2 = submit_sample.drop(['ID'], axis=1)

def lg_nrmse(gt, preds):
    # 각 Y Feature별 NRMSE 총합
    # Y_01 ~ Y_08 까지 20% 가중치 부여
    all_nrmse = []
    for idx in range(0,14): # ignore 'ID'
        rmse = mean_squared_error(gt.iloc[:,idx], preds.iloc[:,idx], squared=False)
        nrmse = rmse/np.mean(np.abs(gt.iloc[:,idx]))
        all_nrmse.append(nrmse)
    score = 1.2 * np.sum(all_nrmse[:8]) + 1.0 * np.sum(all_nrmse[8:])
    return score

sample_score = lg_nrmse(test_X, submit_sample_2)

print(f"sample_score:{sample_score}")