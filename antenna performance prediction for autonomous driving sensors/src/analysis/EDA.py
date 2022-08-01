import pandas as pd
import matplotlib.pyplot as plt

class CFG:
    datapath = "antenna performance prediction for autonomous driving sensors/data/"
    trainpath = datapath+'raw/train.csv'
    testpath = datapath+'raw/test.csv'
    submission = datapath+'raw/sample_submission.csv'
    outpath = datapath+'processed/'
    
# pd.set_option('display.max_columns', None)

df = pd.read_csv(CFG.trainpath)

def standardization(df):
    return (df-df.mean(numeric_only=True))/df.std(numeric_only=True)

def normalization(df):
    return (df-df.min())/(df.max()-df.min())

def distribution_visualization(df):
    for col in df:
        if col == "ID":
            continue
        
        df_list = df[col].dropna().to_list()
        print(df_list)
        df_list = plt.hist(df_list, bins=100)
        plt.show()

def drop_column_ID(df):
    return df.drop(columns=['ID'])



df = drop_column_ID(df)


# df = standardization(df)
df = normalization(df)



distribution_visualization(df)
