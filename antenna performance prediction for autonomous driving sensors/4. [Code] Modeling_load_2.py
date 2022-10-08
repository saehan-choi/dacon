import pandas as pd
import random
import os
import numpy as np
# from tqdm import tqdm

from autogluon.tabular import TabularDataset, TabularPredictor

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import mean_squared_error

import torch


class CFG:
    dataPath = "./data/"

    trainYPath = dataPath+'train.csv'
    trainXPath = dataPath+'train_x_engineered.csv'
    testXPath = dataPath+'test_x_engineered.csv'

    weights_path = './weights/' # 모델 상위 폴더 경로
    submission = dataPath+'sample_submission.csv'

    fold_num = 10
    seed = 42



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

    test_x = pd.read_csv(CFG.testXPath)
    train_x = pd.read_csv(CFG.trainXPath) # Input : X Featrue
    train_y = pd.read_csv(CFG.trainYPath).filter(regex='Y') # Output : Y Feature

    train_y['Class'] = [0 if i<14000 else 1 for i in range(len(train_x))]


    kf = StratifiedKFold(n_splits=CFG.fold_num, shuffle=True, random_state=CFG.seed)

    for f_idx, (train_idx, val_idx) in enumerate(kf.split(train_x, train_y['Class'])):

        # fold1 만 돌리려면 -> f_idx == 0
        if f_idx == 0:
            train_input, train_target = train_x.iloc[train_idx, :].copy(), train_y.iloc[train_idx, :].copy()
            val_input, val_target = train_x.iloc[val_idx, :].copy(), train_y.iloc[val_idx, :].copy()
            submit_test = pd.read_csv(CFG.submission).copy()
            submit_val = pd.DataFrame()
            submit_val['ID'] = pd.read_csv(CFG.trainYPath)['ID'].copy()

            fold_save_path = CFG.weights_path
            os.makedirs(fold_save_path, exist_ok=True)
            for i in range(1, 15):
                number = str(i).zfill(2)
                y_number = 'Y_'+ number

                train_data = pd.concat([train_input, train_target.iloc[:,i-1]], axis=1)
                val_data = pd.concat([val_input, val_target.iloc[:,i-1]], axis=1)
                print(val_data)
                val_data.to_csv('val_data_for_leader_board.csv')
                y_true = val_data[y_number]
                val_data = val_data.drop(columns=[y_number])

                save_path = fold_save_path + y_number + 'Models-predict'

                predictor = TabularPredictor.load(save_path)

                # 제출 csv 추출
                y_pred_test = predictor.predict(test_x)
                y_pred_test_df = pd.DataFrame(y_pred_test, columns=[y_number])
          
                for col in submit_test.columns:
                    if col==y_number:
                        submit_test[col] = y_pred_test_df
                        break
                
                # we have to save fold validation csv! T.T to compare with KDE plot
                # index True for compare with trainset

                submit_test.to_csv('./weights_load_submit_test.csv', index=False)
                submit_test = pd.read_csv('./weights_load_submit_test.csv')
                print(y_number + ' Done **************************************************************************************************************')
        else:
            pass

            