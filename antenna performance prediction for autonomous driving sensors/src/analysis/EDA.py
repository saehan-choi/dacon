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
df = pd.read_csv(CFG.testpath)

def standardization(df):
    return (df-df.mean(numeric_only=True))/df.std(numeric_only=True)

def normalization(df):
    return (df-df.min())/(df.max()-df.min())

def distribution_visualization(df):
    for col in df:
        if col == "ID":
            continue
        df_list = df[col].dropna().to_list()
        df_list = plt.hist(df_list, bins=200)
        
        plt.title(f'{col} standardized histogram')
        # plt.xlabel(f'{col} standardization')
        plt.ylabel('amount')
        plt.savefig(f'{col}.png')
        # plt.show()
        plt.clf()

def drop_column_ID(df):
    return df.drop(columns=['ID'])

def values_check(df):
    # 정수로 표현된 데이터 value_counts
    print(df['X_04'].value_counts())
    print('\n\n')
    print(df['X_10'].value_counts())
    print('\n\n')
    print(df['X_11'].value_counts())
    print('\n\n')
    print(df['X_23'].value_counts())
    print('\n\n')
    print(df['X_47'].value_counts())
    print('\n\n')
    print(df['X_48'].value_counts())
    print('\n\n')    

df = drop_column_ID(df)

df = standardization(df)
# df = normalization(df)
distribution_visualization(df)

