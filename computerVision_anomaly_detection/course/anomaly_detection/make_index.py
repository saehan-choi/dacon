from matplotlib.pyplot import axis
import pandas as pd

df = pd.read_csv('./anomaly_detection/dataset/baseline_test.csv')
pathLabel = './anomaly_detection/dataset/'

df['index'] = 1

for i in range(df.shape[0]):
    df.loc[i,'index'] = i

df = df.loc[:,['index','label']]


df.to_csv(pathLabel+f"submissions/baseline_test2.csv", index = False)

