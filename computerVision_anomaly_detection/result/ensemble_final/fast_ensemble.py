import pandas as pd
import numpy as np
import time

ensemble_label_1 = pd.read_csv('./ensemble_final/OPTUNA19_19_lr_4.956248851278424e-05_val_f1_0.8139666488774808__LB_0.8300102652.csv')
ensemble_label_2 = pd.read_csv('./ensemble_final/OPTUNA23_14_lr_0.00012099442459783506_val_f1_0.8261092405068314.csv')
ensemble_label_3 = pd.read_csv('./ensemble_final/OPTUNA13_11_lr_0.025847566789958795_val_f1_0.8175652239826874__LB_0.8180084031.csv')
ensemble_label_4 = pd.read_csv('./ensemble_final/OPTUNA18_15_lr_0.00014567432582401115_val_f1_0.8155812599500116.csv')

# arange_arr1 = np.arange(0,1,0.2)
# arange_arr1 = np.arange(0,1,0.2)
arange_arr1 = [1]
missed_arr  = [1000]
# 에러방지용 1000넣음
l_k_m_value = []
# # 이거 앙상블할때 
st = time.time()
for l in arange_arr1:
    for k in arange_arr1:
        for m in arange_arr1:
            for a in arange_arr1:
                all_arr = []
                for j in range(len(ensemble_label_1)):
                    arr1 = ensemble_label_1['score_88'][j]
                    arr1 = l*np.array(eval(arr1))

                    arr2 = ensemble_label_2['score_88'][j]
                    arr2 = k*np.array(eval(arr2))

                    arr3 = ensemble_label_3['score_88'][j]
                    arr3 = m*np.array(eval(arr3))

                    arr4 = ensemble_label_4['score_88'][j]
                    arr4 = m*np.array(eval(arr4))
                    # 이게 and로 바꾸는게 효과가 더 있네요 arr1이랑 모두달라야 시작을하는게 옳게받아들이네.....흠 ㅠ
                    # 모두 다른건 문제가 있나보다..
                    # 하나라도 다르면 실행되게 만들어놓긴했음 지금 .

                    if np.argmax(arr1)!=np.argmax(arr2) or np.argmax(arr1)!=np.argmax(arr3) or np.argmax(arr1)!=np.argmax(arr4) :
                        if max(arr1)>max(arr2) and max(arr1)>max(arr3) and max(arr1)>max(arr4):
                            all_arr.append(ensemble_label_1['label'][j])
                        
                        elif max(arr2)>max(arr1) and max(arr2)>max(arr3) and max(arr2)>max(arr4):
                            all_arr.append(ensemble_label_2['label'][j])
                        
                        elif max(arr3)>max(arr1) and max(arr3)>max(arr2) and max(arr3)>max(arr4):
                            all_arr.append(ensemble_label_3['label'][j])
                        
                        elif max(arr4)>max(arr1) and max(arr4)>max(arr2) and max(arr4)>max(arr3):
                            all_arr.append(ensemble_label_4['label'][j])

                        # 에러방지용 append 입니다. 알아서 잘처리하셈 아, 세개가 서로 다 같으면 걍 이거넣어도 똑같으니깐. 
                        # max값이 두개가똑같고 하나가 다르면 이걸로처리하라 이뜻이네요. 흠 이것도 수정이 필요해보임.
                        else:
                            all_arr.append(ensemble_label_1['label'][j])

                    else:
                        all_arr.append(ensemble_label_1['label'][j])

df = pd.DataFrame()
df['index'] = ensemble_label_1['index']
df['label'] = all_arr
ed = time.time()
print(ed-st)