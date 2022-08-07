# import sns
# train_df = 0
# import plt

import pandas as pd

pathX = 'antenna performance prediction for autonomous driving sensors\data/raw\meta/x_feature_info.csv'
pathY = 'antenna performance prediction for autonomous driving sensors\data/raw\meta/y_feature_info.csv'

df_x = pd.read_csv(pathX)
df_y = pd.read_csv(pathY)
df_x = df_x.iloc[:,1].to_list()
df_y = df_y.iloc[:,1].to_list()

# 원하는 fature를 해주시면 됩니다
cnt_X = '01'
for idx, y in enumerate(df_y):
    cnt_y = str(idx+1).zfill(2)
    print(f'X_'+cnt_X')
    sns.regplot(x='X_'+cnt_X, y='Y_'+cnt_y, data=train_df)
    plt.show()
    print(f"['{cnt_y} : {y}']")
    print("")
    print("")

# 전체 다 하는코드
for idx, x in enumerate(df_x):
    cnt_X = str(idx+1).zfill(2)
    X_exp = f"['{cnt_X}  : {x}']"
    print(X_exp)
    for idx, y in enumerate(df_y):
        cnt_y = str(idx+1).zfill(2)
        sns.regplot(x='X_'+cnt_X, y='Y_'+cnt_y, data=train_df)
        plt.show()
        print(f"['{cnt_y} : {y}']")
        print("")
        print("")

# X feature마다 하는코드
# 원하는 fature를 해주시면 됩니다
cnt_X = '01'
for idx, y in enumerate(df_y):
    cnt_y = str(idx+1).zfill(2)
    sns.regplot(x='X_'+cnt_X, y='Y_'+cnt_y, data=train_df)
    plt.show()
    print(f"['{cnt_y} : {y}']")
    print("")
    print("")



        

# X_56
cnt_Y = 1
X_exp = ["X_56  : RF7 부분 SMT 납 량"]
sns.regplot(x='X_01', y='Y_01', data=train_df)
plt.show()
print(X_exp)
print("['Y_%02d : 안테나 Gain 평균 (각도1)']"%cnt_Y)
print("")
print("")
cnt_Y+=1

sns.regplot(x='X_01', y='Y_02', data=train_df)
plt.show()
print(X_exp)
print("['Y_%02d : 안테나 1 Gain 편차']"%cnt_Y)
print("")
print("")
cnt_Y+=1

sns.regplot(x='X_01', y='Y_03', data=train_df)
plt.show()
print(X_exp)
print("['Y_%02d : 안테나 2 Gain 편차']"%cnt_Y)
print("")
print("")
cnt_Y+=1

sns.regplot(x='X_01', y='Y_04', data=train_df)
plt.show()
print(X_exp)
print("['Y_%02d : 평균 신호대 잡음비']"%cnt_Y)
print("")
print("")
cnt_Y+=1

sns.regplot(x='X_01', y='Y_05', data=train_df)
plt.show()
print(X_exp)
print("['Y_%02d : 안테나 Gain 평균 (각도2)']"%cnt_Y)
print("")
print("")
cnt_Y+=1

sns.regplot(x='X_01', y='Y_06', data=train_df)
plt.show()
print(X_exp)
print("['Y_%02d : 신호대 잡음비  (각도1)']"%cnt_Y)
print("")
print("")
cnt_Y+=1

sns.regplot(x='X_01', y='Y_07', data=train_df)
plt.show()
print(X_exp)
print("['Y_%02d : 안테나 Gain 평균 (각도3)  ']"%cnt_Y)
print("")
print("")
cnt_Y+=1

sns.regplot(x='X_01', y='Y_08', data=train_df)
plt.show()
print(X_exp)
print("['Y_%02d : 신호대 잡음비  (각도2)']"%cnt_Y)
print("")
print("")
cnt_Y+=1

sns.regplot(x='X_01', y='Y_09', data=train_df)
plt.show()
print(X_exp)
print("['Y_%02d : 신호대 잡음비  (각도3)']"%cnt_Y)
print("")
print("")
cnt_Y+=1

sns.regplot(x='X_01', y='Y_10', data=train_df)
plt.show()
print(X_exp)
print("['Y_%02d : 신호대 잡음비  (각도4)']"%cnt_Y)
print("")
print("")
cnt_Y+=1

sns.regplot(x='X_01', y='Y_11', data=train_df)
plt.show()
print(X_exp)
print("['Y_%02d : 안테나 Gain 평균 (각도4)']"%cnt_Y)
print("")
print("")
cnt_Y+=1

sns.regplot(x='X_01', y='Y_12', data=train_df)
plt.show()
print(X_exp)
print("['Y_%02d : 신호대 잡음비  (각도5)']"%cnt_Y)
print("")
print("")
cnt_Y+=1

sns.regplot(x='X_01', y='Y_13', data=train_df)
plt.show()
print(X_exp)
print("['Y_%02d :신호대 잡음비  (각도6)']"%cnt_Y)
print("")
print("")
cnt_Y+=1

sns.regplot(x='X_01', y='Y_14', data=train_df)
plt.show()
print(X_exp)
print("['Y_%02d : 신호대 잡음비  (각도7)']"%cnt_Y)
print("")
print("")
cnt_Y+=1
