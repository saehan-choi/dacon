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


for k in setClass:
    checkState = df[df['class']==k]
    checkState = checkState['state'].unique()
    print(k)
    print(len(checkState))
    # print(checkState)

