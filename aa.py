### X_01 ###
cnt_X = 1
cnt_Y = 1
X_exp = ["X_01 : PCB 체결 시 단계별 누름량(Step 1)"]
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


-> 

### X_01 ###
cnt_X = 1
cnt_Y = 1
X_exp = ["X_01 : PCB 체결 시 단계별 누름량(Step 1)"]
sns.regplot(x='X_01', y='Y_01', data=train_df)
plt.show()
print(X_exp)
print("['Y_%02d : 안테나 Gain 평균 (각도1)']"%cnt_Y)
print("")
print("")
cnt_Y+=1
X_exp = ["X_02 : PCB 체결 시 단계별 누름량(Step 2)"]
sns.regplot(x='X_01', y='Y_02', data=train_df)
plt.show()
print(X_exp)
print("['Y_%02d : 안테나 1 Gain 편차']"%cnt_Y)
print("")
print("")
cnt_Y+=1