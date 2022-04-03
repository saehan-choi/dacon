import pandas as pd
import shutil
import os

DataPath = './anomaly_detection/dataset/'
trainDataPath = './anomaly_detection/dataset/train/train/'
resultPath = './anomaly_detection/dataAnalysis/'

df = pd.read_csv(DataPath+'train_df.csv')

# index,   file_name,   class,        state,   label
# 0,       10000.png,   transistor,   good,    transistor-good
# print(df)

# set Dataset
setClass = df['class'].unique()

for i in setClass:
    # os.mkdir(resultPath+i)
    fileName = df[df['class'] == i]
    fileName = fileName['file_name'].to_list()
    
    for j in fileName:
        shutil.copy(trainDataPath+j, resultPath+i)
