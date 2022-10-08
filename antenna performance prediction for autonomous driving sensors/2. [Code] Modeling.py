import os
import torch
import random
import numpy as np
import pandas as pd

from autogluon.tabular import TabularPredictor

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error


# configuration
class CFG:
    trainYPath = './data/train.csv'
    trainXPath = './data/train_x_engineered.csv'
    testXPath = './data/test_x_engineered.csv'
    submission = './data/sample_submission.csv'

    fold_num = 10
    seed = 42

# seed setting
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# evalutation function
def lg_nrmse(gt, preds):
    all_nrmse = []
    for idx in range(0,14):
        rmse = mean_squared_error(gt.iloc[:,idx], preds.iloc[:,idx], squared=False)
        nrmse = rmse/np.mean(np.abs(gt.iloc[:,idx]))
        print(f'Y_{idx+1} nrmse:{nrmse}')
        all_nrmse.append(nrmse)
    score = 1.2 * np.sum(all_nrmse[:8]) + 1.0 * np.sum(all_nrmse[8:])
    return score

if __name__ == "__main__":
    # seed 고정 및 dataset read
    seed_everything(CFG.seed)
    test_x = pd.read_csv(CFG.testXPath)
    train_x = pd.read_csv(CFG.trainXPath) 
    train_y = pd.read_csv(CFG.trainYPath).filter(regex='Y') 

    # feature 주기성을 반영한 StratifiedKFold 적용
    train_y['Class'] = [0 if i<14000 else 1 for i in range(len(train_x))]
    kf = StratifiedKFold(n_splits=CFG.fold_num, shuffle=True, random_state=CFG.seed)

    for f_idx, (train_idx, val_idx) in enumerate(kf.split(train_x, train_y['Class'])):
        # if f_idx == 0: -> FOLD1 실행
        if f_idx == 0:
            train_input, train_target = train_x.iloc[train_idx, :], train_y.iloc[train_idx, :]
            val_input, val_target = train_x.iloc[val_idx, :], train_y.iloc[val_idx, :]

            submit_test = pd.read_csv(CFG.submission)
            submit_val = pd.DataFrame()
            submit_val['ID'] = pd.read_csv(CFG.trainYPath)['ID']

            fold_save_path = f'./FOLD{f_idx+1}/'
            os.makedirs(fold_save_path, exist_ok=True)
            
            # Y01~Y14 학습
            for i in range(1, 15): # y_06~y_14 예측모델 만들고 싶으면 range(6, 15)로 변경
                number = str(i).zfill(2)
                y_number = 'Y_'+ number
                save_path = fold_save_path + y_number + 'Models-predict'

                # autogluon은 train과 label이 같은 dataframe으로 들어가는 구조입니다. 
                # TabularPredictor를 거치면 train_target(Y feature)는 라벨로 사용된 후 Dataframe에서 삭제됩니다.
                train_data = pd.concat([train_input, train_target.iloc[:,i-1]], axis=1)
                predictor = TabularPredictor(label=y_number,  eval_metric='root_mean_squared_error', path=save_path).fit(train_data, presets='high_quality',  ag_args_fit={'num_gpus': 1})

                # 학습된 모델을 바탕으로 값 예측
                y_pred_val = predictor.predict(val_input)
                y_pred_val_df = pd.DataFrame(y_pred_val, columns=[y_number])

                y_pred_test = predictor.predict(test_x)
                y_pred_test_df = pd.DataFrame(y_pred_test, columns=[y_number])
                
                # 예측값 csv 저장 작업
                for col in submit_test.columns:
                    if col==y_number:
                        submit_val[col] = y_pred_val_df
                        submit_test[col] = y_pred_test_df
                        break
                
                submit_val.to_csv(f'./fold{f_idx+1}_submit_val.csv', index=False)
                submit_val = pd.read_csv(f'./fold{f_idx+1}_submit_val.csv')

                submit_test.to_csv(f'./fold{f_idx+1}_submit_test.csv', index=False)
                submit_test = pd.read_csv(f'./fold{f_idx+1}_submit_test.csv')
                print(y_number + ' Done **************************************************************************************************************')

        else:
            pass