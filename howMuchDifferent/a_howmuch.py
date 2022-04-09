import pandas as pd


path = './howMuchDifferent/'

pred_csv = 'baseline_seed_0.csv'
gt_csv = 'human_label.csv'

pred_label = pd.read_csv(path+pred_csv)
gt_label = pd.read_csv(path+gt_csv)

pred = pred_label['label']
gt = gt_label['label']

compare = gt.compare(pred, align_axis=0)

gt = compare.loc[:,'self'].reset_index()
pred = compare.loc[:,'other'].reset_index()


gt.columns = ['index','gt']
pred.columns = ['index','pred']

# print(pred['pred'])
df = pd.concat([gt,pred['pred']],axis=1)

df['file_name'] = df['index']+20000

# gt = gt[gt!=pred]
# # gt에서 예측한 값들
# print("\n\n")
# pred = pred[pred!=gt]
# # pred에서 예측한 값들
