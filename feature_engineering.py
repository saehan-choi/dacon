import pandas as pd

class CFG:
    datapath = "antenna performance prediction for autonomous driving sensors/data/"
    trainpath = datapath+'raw/test.csv'

path = CFG.trainpath
df = pd.read_csv(path)

df_X = df.filter(regex="X")
df_Y = df.filter(regex="Y")
# 누름량
PUSH = ['X_01', 'X_02', 'X_05', 'X_06']

# 방열재료
MATERIAL = ['X_03', 'X_07', 'X_08', 'X_09', 'X_10', 'X_11']

# 통과여부
PASS = ['X_04', 'X_23', 'X_47', 'X_48']

# 안테나 (애매한 부분 섞임)
ANTENNA = ['X_13', 'X_14', 'X_15', 'X_16', 'X_17', 'X_18',
         'X_41', 'X_42', 'X_43', 'X_44', 'X_45']

# 스크류 (애매하지는 않으나, 상단부분 feature들과 하단부분 feature들의 차이를 모르겠음)
SCREW = ['X_19', 'X_20', 'X_21', 'X_22',
        'X_30', 'X_31', 'X_32', 'X_33']

# 스크류 분당회전수
SCREW_ROTATION = ['X_34', 'X_35', 'X_36', 'X_37']

# 커넥터핀치수
PIN_SIZE = ['X_24', 'X_25', 'X_26', 'X_27', 'X_28', 'X_29']

# PCB치수
PCB_SIZE = ['X_38', 'X_39', 'X_40']

# RF납량
RF_AMOUNT = ['X_50', 'X_51', 'X_52', 'X_53', 'X_54', 'X_55', 'X_56']

# X_12,커넥터 위치 기준 좌표, X_46,실란트 본드 소요량은 2개는 단일변수로서 포함되지않음 

eachRange = [PUSH, MATERIAL, PASS, ANTENNA, SCREW, SCREW_ROTATION, PIN_SIZE, PCB_SIZE, RF_AMOUNT]

for idx, e in enumerate(eachRange):
    df_add = df.loc[:, e].copy()
    # 추후 mean말고 median으로 해보는 방법도 해보겠습니다.
    df_X[f'N_{str(idx).zfill(2)}'] = df_add.mean(axis=1)


for idx, values in enumerate(df_Y.iloc[:,]):
    df_X[values] = df_Y[values]

print(df_X)
df_X.to_csv('./new.csv')


# 이렇게하면 feature 늘어납니다 이걸로 저장하고 학습시키면 됩니다.

# Feature,설명
# X_01,PCB 체결 시 단계별 누름량(Step 1) o
# X_02,PCB 체결 시 단계별 누름량(Step 2) o
# X_03,방열 재료 1 무게 o
# X_04,1차 검사 통과 여부 o
# X_05,PCB 체결 시 단계별 누름량(Step 3) o
# X_06,PCB 체결 시 단계별 누름량(Step 4) o
# X_07,방열 재료 1 면적 o
# X_08,방열 재료 2 면적 o
# X_09,방열 재료 3 면적 o
# X_10,방열 재료 2 무게 o
# X_11,방열 재료 3 무게 o
# X_12,커넥터 위치 기준 좌표
# X_13,각 안테나 패드 위치(높이) 차이 o
# X_14,1번 안테나 패드 위치 o
# X_15,2번 안테나 패드 위치 o
# X_16,3번 안테나 패드 위치 o
# X_17,4번 안테나 패드 위치 o
# X_18,5번 안테나 패드 위치 o
# X_19,1번 스크류 삽입 깊이 o
# X_20,2번 스크류 삽입 깊이 o
# X_21,3번 스크류 삽입 깊이 o 
# X_22,4번 스크류 삽입 깊이 o
# X_23,2차 검사 통과 여부 o
# X_24,커넥터 1번 핀 치수 o 
# X_25,커넥터 2번 핀 치수 o
# X_26,커넥터 3번 핀 치수 o
# X_27,커넥터 4번 핀 치수 o
# X_28,커넥터 5번 핀 치수 o
# X_29,커넥터 6번 핀 치수 o
# X_30,스크류 삽입 깊이1 o
# X_31,스크류 삽입 깊이2 o
# X_32,스크류 삽입 깊이3 o
# X_33,스크류 삽입 깊이4 o
# X_34,스크류 체결 시 분당 회전수 1 o
# X_35,스크류 체결 시 분당 회전수 2 o
# X_36,스크류 체결 시 분당 회전수 3 o
# X_37,스크류 체결 시 분당 회전수 4 o
# X_38,하우징 PCB 안착부 1 치수 o
# X_39,하우징 PCB 안착부 2 치수 o
# X_40,하우징 PCB 안착부 3 치수 o
# X_41,레이돔 치수 (안테나 1번 부위) o  -> 이부분 안테나에 포함시킴
# X_42,레이돔 치수 (안테나 2번 부위) o
# X_43,레이돔 치수 (안테나 3번 부위) o
# X_44,레이돔 치수 (안테나 4번 부위) o
# X_45,안테나 부분 레이돔 기울기 o
# X_46,실란트 본드 소요량
# X_47,3차 검사 통과 여부 o
# X_48,4차 검사 통과 여부 o
# X_49,Cal 투입 전 대기 시간
# X_50,RF1 부분 SMT 납 량 o
# X_51,RF2 부분 SMT 납 량 o
# X_52,RF3 부분 SMT 납 량 o
# X_53,RF4 부분 SMT 납 량 o
# X_54,RF5 부분 SMT 납 량 o
# X_55,RF6 부분 SMT 납 량 o
# X_56,RF7 부분 SMT 납 량 o