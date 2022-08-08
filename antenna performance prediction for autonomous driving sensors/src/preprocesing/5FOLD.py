import pandas as pd
import random
import os
import numpy as np


class CFG:
    dataPath = "antenna performance prediction for autonomous driving sensors/data/"
    trainPath = dataPath+'raw/train.csv'
    testPath = dataPath+'raw/test.csv'
    submission = dataPath+'raw/sample_submission.csv'
    outPath = dataPath+'processed/'
    weightsavePath = dataPath+'weights/'

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
df = pd.read_csv(CFG.trainPath)

print(len(df)//5)
print(len(df)//5 * 5)

