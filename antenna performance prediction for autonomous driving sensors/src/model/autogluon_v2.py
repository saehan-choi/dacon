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

from sklearn.metrics import mean_squared_error

import torch

class CFG:
    dataPath = "antenna performance prediction for autonomous driving sensors/data/"
    # trainPath = dataPath+'raw/new_train.csv'
    # testPath = dataPath+'raw/new_test.csv'
    trainPath = dataPath+'raw/train.csv'
    testPath = dataPath+'raw/test.csv'    

    submission = dataPath+'raw/sample_submission.csv'
    outPath = dataPath+'processed/'
    weightsavePath = dataPath+'weights/'
    drop_list = ['X_04', 'X_23', 'X_47', 'X_48', 'X_10', 'X_11']
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
    submit = pd.read_csv(CFG.submission)

    test_final_x = test_df.drop(CFG.drop_list, axis=1)

    train_x = train_df.filter(regex='X') # Input : X Featrue
    train_x = train_x.drop(CFG.drop_list, axis=1)

    train_y = train_df.filter(regex='Y') # Output : Y Feature
    train_input, test_input, train_target, test_target = train_test_split(train_x, train_y, test_size=0.2, random_state=CFG.seed)

    ##TabularDataset 가 csv를 df 로 읽는 구조라 csv -> 그냥 pandas dataframe은 읽어올 수 있다고 하네요
    for i in range(1, 15):
        number = str(i).zfill(2)
        y_number = 'Y_'+ number

        train_data = pd.concat([train_input, train_target.iloc[:,i-1]], axis=1)
        test_data = pd.concat([test_input, test_target.iloc[:,i-1]], axis=1)

        y_test = test_data[y_number]
        test_data_nolab = test_data.drop(columns=[y_number])

        save_path = y_number + 'Models-predict'
        # predictor = TabularPredictor(label=y_number,  eval_metric='root_mean_squared_error', path=save_path).fit(train_data, presets='best_quality',  ag_args_fit={'num_gpus': 1})
        # predictor = TabularPredictor.load(save_path)  # unnecessary, just demonstrates how to load previously-trained predictor from file

        predictor = TabularPredictor(label=y_number,  eval_metric='root_mean_squared_error').fit(train_data, presets='best_quality',  ag_args_fit={'num_gpus': 1})
        # predictor = TabularPredictor.load(save_path)  # unnecessary, just demonstrates how to load previously-trained predictor from file
        # 이거 이렇게 load 없애도 동작하는지 실험중입니다 submit_AutoGluon_Tabuler_defual.csv 이게 생성되고 Y가 2개 이상 나오면 예상되로 된것.

        y_pred = predictor.predict(test_data_nolab)
        perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)

        # 제출 csv 추출
        y_pred_final = predictor.predict(TabularDataset(test_final_x))
        y_pred_final_df = pd.DataFrame(y_pred_final, columns=[y_number])
        for idx, col in enumerate(submit.columns):
            if col==y_number:
                submit[col] = y_pred_final_df
                break
        submit.to_csv('./submit_AutoGluon_Tabuler_defual.csv', index=False)
        submit = pd.read_csv('./submit_AutoGluon_Tabuler_defual.csv')

        print(y_number + ' Done **************************************************************************************************************')




    print(f"val_score : {lg_nrmse(test_target, submit_sample_2)}")

