import pandas as pd



# # print(CFG.trainsetPath)
# pd.set_option('display.max_columns', None)
# pd.set_option("max_rows", None)

# pd.set_option('max_columns', None)

df = pd.read_csv(CFG.trainsetPath)

df = df.describe()

df.to_csv('./EDA.csv')