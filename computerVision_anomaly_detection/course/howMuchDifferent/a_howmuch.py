import pandas as pd


path = './howMuchDifferent/'

# csv만 옮기고 파일쓰면 됩니다.
pred_csv = 'train_with_affine_aug_16_lr_0.0003.csv'
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

df = pd.concat([gt,pred['pred']],axis=1)
df['file_name'] = df['index']+20000

pd.set_option('display.max_rows',None)
print(df['gt'].value_counts())

# 여기서 제일많은 순서대로 고쳐 나가야 합니다.
# 제가 라벨링 한게 False postive or False Negative일 확률이 있기 때문입니다.

df.to_csv(path+f"howMuchDiffrent.csv", index = False)

# gt = gt[gt!=pred]
# # gt에서 예측한 값들
# print("\n\n")
# pred = pred[pred!=gt]
# # pred에서 예측한 값들
