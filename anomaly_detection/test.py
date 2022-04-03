# import random

# print(random.randint(0,2))

# scale = random.uniform(-1.1,-1)

# print(scale)


# Datasplit2

import os
import shutil
import pandas as pd

pathTrainwithLabel = './anomaly_detection/dataAnalysis/train_with_label/'
pathTestwithLabel = './anomaly_detection/dataAnalysis/train_with_label/'

pathLabel = './anomaly_detection/dataset/'

pathTrain = './anomaly_detection/dataset/train/train/'
pathTest = './anomaly_detection/dataset/test/test/'

df = pd.read_csv(pathLabel+'train_df.csv')


class_ = df['class'].unique()
# state_ = df['state'].unique()


# 나중에 여기 for i in class_ 넣고 class_[0] -> class_i로 변경할것

# for cls in class_:
#     df_anyclass = df[df['class']==cls]
#     df_anyclass = df_anyclass['state'].unique()

#     for k in df_anyclass:
#         # os.mkdir(pathTrainwithLabel+df_anyclass+f'/{k}')
#         print(k)

for a in class_:
    cls = a
    df_anyclass = df[df['class']==cls]
    df_anyclass = df_anyclass['state'].unique()

    for k in df_anyclass:
        res_path = pathTrainwithLabel+cls+'/'+k+'/'
        os.mkdir(res_path)
        kf = df[df['state'] == k]
        kf = kf[kf['class']== cls]
        kf = kf['file_name'].to_list()

        for m in kf:
            shutil.copy(pathTrain+m, res_path+m)
