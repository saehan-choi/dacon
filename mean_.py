import pandas as pd

df_1 = pd.read_csv('fold1_submit_test.csv')
df_2 = pd.read_csv('fold2_submit_test.csv')
df_3 = pd.read_csv('fold3_submit_test.csv')
df_4 = pd.read_csv('fold4_submit_test.csv')
df_5 = pd.read_csv('fold5_submit_test.csv')

df_all = df_1+df_2+df_3+df_4+df_5

df_all = df_all.drop(columns='ID')
df_all = df_all/5

df_all.insert(0, 'ID', df_1['ID'])

# df_all['ID'] = df_1['ID']
df_all.to_csv('all_fold_mean_csv.csv', index=False)

print(df_all)