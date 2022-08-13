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

class CFG:
    # dataPath = "aa/data/"
    # trainPath = dataPath+'raw/train.csv'

    trainPath = 'new_train.csv'
    testPath = 'new_test.csv'

    # trainPath = dataPath+'raw/train.csv'
    # testPath = dataPath+'raw/test.csv'    
    submission = 'sample_submission.csv'
    # outPath = 'processed/'
    # weightsavePath = dataPath+'weights/'
    seed = 42

seed = 42
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(seed) # Seed 고정

drop_list = ['X_04', 'X_23', 'X_47', 'X_48', 'X_10', 'X_11']

train_df = pd.read_csv(CFG.trainPath).iloc[:,1:] # train.csv load
test_df = pd.read_csv(CFG.testPath).iloc[:,1:]

test_final_x = test_df.drop(drop_list, axis=1)
test_final_x.to_csv('test_final_x.csv')

#1
train_x = train_df.filter(regex='X') # Input : X Featrue
train_x = train_x.drop(drop_list, axis=1)

train_y = train_df.filter(regex='Y') # Output : Y Feature
train_input, test_input, train_target, test_target = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

##TabularDataset 가 csv를 df 로 읽는 구조라 csv 로 다시만듬;
train_input.to_csv('train_input.csv')
test_input.to_csv('test_input.csv')
train_target.to_csv('train_target.csv')
test_target.to_csv('test_target.csv')

#2
# train_set_input, val_set_input, train_set_target, val_set_target = train_test_split(train_input, train_target, test_size=0.2, random_state=42)



submit = pd.read_csv(CFG.submission)
# submit_sample = pd.read_csv(CFG.submission).iloc[0:len(test_target),:]

for i in range(1, 15):
    number = str(i).zfill(2)
    y_number = 'Y_'+ number

    # train data
    train_data = pd.concat([train_input, train_target.iloc[:,i-1]], axis=1)
    train_data.to_csv('train_data.csv')
    train_data = TabularDataset('./train_data.csv')

    # test data
    test_data = pd.concat([test_input, test_target.iloc[:,i-1]], axis=1)
    test_data.to_csv('test_data.csv')
    test_data = TabularDataset('./test_data.csv')
    y_test = test_data[y_number]  # values to predict
    test_data_nolab = test_data.drop(columns=[y_number])  # delete label column to prove we're not cheating

    # subsample_size = 500  # subsample subset of data for faster demo, try setting this to much larger values
    # train_data = train_data.sample(n=subsample_size, random_state=42)

    save_path = y_number + 'Models-predict'  # specifies folder to store trained models
    predictor = TabularPredictor(label=y_number,  eval_metric='root_mean_squared_error', path=save_path).fit(train_data, presets='best_quality',  ag_args_fit={'num_gpus': 1})
    predictor = TabularPredictor.load(save_path)  # unnecessary, just demonstrates how to load previously-trained predictor from file
    y_pred = predictor.predict(test_data_nolab)
    # print("Predictions:  \n", y_pred)
    perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)


    # 결과 샘플 csv로 추출
    # y_pred_df = pd.DataFrame(y_pred, columns=[y_number])
    # for idx, col in enumerate(submit_sample.columns):
    #     if col==y_number:   
    #         submit_sample[col] = y_pred_df
    #         break

    # submit_sample.to_csv('./submit_sample_AutoGluon_Tabuler_defual.csv', index=False)
    # submit_sample = pd.read_csv('./submit_sample_AutoGluon_Tabuler_defual.csv')

    # 제출 csv 추출
    y_pred_final = predictor.predict(TabularDataset('./test_final_x.csv'))
    y_pred_final_df = pd.DataFrame(y_pred_final, columns=[y_number])
    for idx, col in enumerate(submit.columns):
        if col==y_number:
            submit[col] = y_pred_final_df
            break
    submit.to_csv('./submit_AutoGluon_Tabuler_defual.csv', index=False)
    submit = pd.read_csv('./submit_AutoGluon_Tabuler_defual.csv')

    print(y_number + ' Done **************************************************************************************************************')


from sklearn.metrics import mean_squared_error

# submit_sample_2 = submit_sample.drop(['ID'], axis=1)

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



# print(f"val_score : {lg_nrmse(test_target, submit_sample_2)}")

