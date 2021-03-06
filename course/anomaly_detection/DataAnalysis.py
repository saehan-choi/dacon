
# DataAnalysis
import pandas as pd

DataPath = './anomaly_detection/dataset/'

df = pd.read_csv(DataPath+'train_df.csv')

# index,   file_name,   class,        state,   label
# 0,       10000.png,   transistor,   good,    transistor-good
# print(df)

# set Dataset
setClass = df['class'].unique()
# ['transistor' 'capsule' 'wood' 'bottle' 'screw' 'cable' 'carpet'
#  'hazelnut' 'pill' 'metal_nut' 'zipper' 'leather' 'toothbrush' 'tile'
#  'grid']

add = 0
for k in setClass:
    checkState = df[df['class']==k]
    checkState = checkState['state'].unique()
    print(k)
    print(len(checkState))
    add += len(checkState)
    # print(checkState)

print(add)



# DataSplit

# import pandas as pd
# import shutil
# import os

# DataPath = './anomaly_detection/dataset/'
# trainDataPath = './anomaly_detection/dataset/train/train/'
# resultPath = './anomaly_detection/dataAnalysis/'

# df = pd.read_csv(DataPath+'train_df.csv')

# # index,   file_name,   class,        state,   label
# # 0,       10000.png,   transistor,   good,    transistor-good
# # print(df)

# # set Dataset
# setClass = df['class'].unique()

# for i in setClass:
#     # os.mkdir(resultPath+i)
#     fileName = df[df['class'] == i]
#     fileName = fileName['file_name'].to_list()
    
#     for j in fileName:
#         shutil.copy(trainDataPath+j, resultPath+i)
