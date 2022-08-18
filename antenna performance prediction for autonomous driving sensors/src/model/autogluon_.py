import shutil
import pandas as pd
import random
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_validate, train_test_split

from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error

import torch

import gc

class CFG:
    dataPath = "antenna performance prediction for autonomous driving sensors/data/"
    # trainPath = dataPath+'raw/new_train.csv'
    # testPath = dataPath+'raw/new_test.csv'
    trainPath = dataPath+'raw/train.csv'
    testPath = dataPath+'raw/test.csv'

    submission = dataPath+'raw/sample_submission.csv'
    outPath = dataPath+'processed/'
    drop_list = ['X_04', 'X_23', 'X_47', 'X_48', 'X_10', 'X_11']
    fold_num = 10
    seed = 42

# class CFG:
#     trainPath = 'new_train.csv'
#     testPath = 'new_test.csv'
#     submission = 'sample_submission.csv'
#     drop_list = ['X_04', 'X_23', 'X_47', 'X_48', 'X_10', 'X_11']
#     seed = 42


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def lg_nrmse(gt, preds):
    # 각 Y Feature별 NRMSE 총합
    # Y_01 ~ Y_08 까지 20% 가중치 부여
    all_nrmse = []
    for idx in range(0,14): # ignore 'ID'
        rmse = mean_squared_error(gt.iloc[:,idx], preds.iloc[:,idx], squared=False)
        nrmse = rmse/np.mean(np.abs(gt.iloc[:,idx]))
        print(f'Y_{idx+1} nrmse:{nrmse}')
        all_nrmse.append(nrmse)
    score = 1.2 * np.sum(all_nrmse[:8]) + 1.0 * np.sum(all_nrmse[8:])
    return score


if __name__ == "__main__":
    seed_everything(CFG.seed) # Seed 고정

    train_df = pd.read_csv(CFG.trainPath).drop('ID', axis=1) # train.csv load
    test_df = pd.read_csv(CFG.testPath).drop('ID', axis=1)
    test_x = test_df.drop(CFG.drop_list, axis=1)
    train_x = train_df.filter(regex='X') # Input : X Featrue
    train_x = train_x.drop(CFG.drop_list, axis=1)
    train_y = train_df.filter(regex='Y') # Output : Y Feature

    kf = KFold(n_splits=CFG.fold_num, shuffle=True, random_state=CFG.seed)
    
    for f_idx, (train_idx, val_idx) in enumerate(kf.split(train_x)):
    # ##TabularDataset 가 csv를 df 로 읽는 구조라 csv -> 그냥 pandas dataframe은 읽어올 수 있다고 하네요
        train_input, train_target = train_x.iloc[train_idx, :].copy(), train_y.iloc[train_idx, :].copy()
        val_input, val_target = train_x.iloc[val_idx, :].copy(), train_y.iloc[val_idx, :].copy()
        submit_test = pd.read_csv(CFG.submission).copy()
        submit_val = pd.DataFrame()
        submit_val['ID'] = pd.read_csv(CFG.trainPath)['ID'].copy()

        fold_save_path = f'./FOLD{f_idx+1}/'
        os.makedirs(fold_save_path, exist_ok=True)
        for i in range(1, 15):
            number = str(i).zfill(2)
            y_number = 'Y_'+ number

            train_data = pd.concat([train_input, train_target.iloc[:,i-1]], axis=1)
            val_data = pd.concat([val_input, val_target.iloc[:,i-1]], axis=1)

            y_true = val_data[y_number]
            val_data = val_data.drop(columns=[y_number])

            save_path = fold_save_path + y_number + 'Models-predict'
            predictor = TabularPredictor(label=y_number,  eval_metric='root_mean_squared_error', path=save_path).fit(train_data, presets='best_quality',  ag_args_fit={'num_gpus': 1})

            y_pred_val = predictor.predict(val_data)
            # perf = predictor.evaluate_predictions(y_true=y_true, y_pred=y_pred, auxiliary_metrics=True)
            y_pred_val_df = pd.DataFrame(y_pred_val, columns=[y_number])

            # 제출 csv 추출
            y_pred_test = predictor.predict(test_x)
            y_pred_test_df = pd.DataFrame(y_pred_test, columns=[y_number])

            for col in submit_test.columns:
                if col==y_number:
                    submit_val[col] = y_pred_val_df
                    submit_test[col] = y_pred_test_df
                    break
            
            # we have to save fold validation csv! T.T to compare with KDE plot
            # index True for compare with trainset
            submit_val.to_csv(f'./fold{f_idx+1}_submit_val.csv', index=False)
            submit_val = pd.read_csv(f'./fold{f_idx+1}_submit_val.csv')

            submit_test.to_csv(f'./fold{f_idx+1}_submit_test.csv', index=False)
            submit_test = pd.read_csv(f'./fold{f_idx+1}_submit_test.csv')
            print(y_number + ' Done **************************************************************************************************************')
        
        # 용량이 너무커서 fold파일을 삭제해야하면 이것을 이용하세요 용량 안되면 저장하다가 에러납니다...!
        if os.path.exists(fold_save_path):
            shutil.rmtree(fold_save_path)
        # seed 고정되었는지 확인하세요 -> 고정됨
        gc.collect()