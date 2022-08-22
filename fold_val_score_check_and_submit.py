
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error


train = pd.read_csv('train.csv')

fold1_pred = pd.read_csv('fold1_submit_val.csv').dropna().drop(columns='ID')
fold2_pred = pd.read_csv('fold2_submit_val.csv').dropna().drop(columns='ID')
fold3_pred = pd.read_csv('fold3_submit_val.csv').dropna().drop(columns='ID')
fold4_pred = pd.read_csv('fold4_submit_val.csv').dropna().drop(columns='ID')
fold5_pred = pd.read_csv('fold5_submit_val.csv').dropna().drop(columns='ID')
fold6_pred = pd.read_csv('fold6_submit_val.csv').dropna().drop(columns='ID')
fold7_pred = pd.read_csv('fold7_submit_val.csv').dropna().drop(columns='ID')

def lg_nrmse(gt, preds):
    # 각 Y Feature별 NRMSE 총합
    # Y_01 ~ Y_08 까지 20% 가중치 부여
    all_nrmse = []
    for idx in range(0,14): # ignore columns='ID'
        rmse = mean_squared_error(gt.iloc[:,idx], preds.iloc[:,idx], squared=False)
        nrmse = rmse/np.mean(np.abs(gt.iloc[:,idx]))
        # print(f'Y_{idx+1} nrmse:{nrmse}')
        all_nrmse.append(nrmse)
    score = 1.2 * np.sum(all_nrmse[:8]) + 1.0 * np.sum(all_nrmse[8:])
    return score

fold1_gt = train.iloc[fold1_pred.index,:].drop(columns='ID').filter(regex='Y')
fold2_gt = train.iloc[fold2_pred.index,:].drop(columns='ID').filter(regex='Y')
fold3_gt = train.iloc[fold3_pred.index,:].drop(columns='ID').filter(regex='Y')
fold4_gt = train.iloc[fold4_pred.index,:].drop(columns='ID').filter(regex='Y')
fold5_gt = train.iloc[fold5_pred.index,:].drop(columns='ID').filter(regex='Y')
fold6_gt = train.iloc[fold6_pred.index,:].drop(columns='ID').filter(regex='Y')
fold7_gt = train.iloc[fold7_pred.index,:].drop(columns='ID').filter(regex='Y')

print(lg_nrmse(fold1_gt, fold1_pred))
print(lg_nrmse(fold2_gt, fold2_pred))
print(lg_nrmse(fold3_gt, fold3_pred))
print(lg_nrmse(fold4_gt, fold4_pred))
print(lg_nrmse(fold5_gt, fold5_pred))
print(lg_nrmse(fold6_gt, fold6_pred))
print(lg_nrmse(fold7_gt, fold7_pred))




def make_mean_submit():
    fold1_test = pd.read_csv('fold1_submit_test.csv')
    fold2_test = pd.read_csv('fold2_submit_test.csv')
    fold3_test = pd.read_csv('fold6_submit_test.csv')
    
    fold1_submit = fold1_test.drop(columns="ID")
    fold2_submit = fold2_test.drop(columns="ID")
    fold3_submit = fold3_test.drop(columns="ID")
    
    fold_mean_submit = (fold1_submit+fold2_submit+fold3_submit)/3
    
    fold_mean_submit.insert(0, 'ID', fold1_test['ID'])
    
    fold_mean_submit.to_csv('high_qulity_val_score_1.923_fold_1_2_6.csv', index=False)
    
def make_median_submit():
    fold1_test = pd.read_csv('fold1_submit_test.csv')
    fold2_test = pd.read_csv('fold2_submit_test.csv')
    fold3_test = pd.read_csv('fold6_submit_test.csv')
    

make_mean_submit()