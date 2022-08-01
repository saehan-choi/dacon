from concurrent.futures import thread
import pandas as pd
# from sklearn import ensemble
import numpy as np
from howMuchDifferent.a_howmuch import how_much_diffrent
import threading

# def ensemble_test(path, pred_csv, gt_csv):

#     ensemble_label_1 = pd.read_csv('./ensemble_final/OPTUNA19_19_lr_4.956248851278424e-05_val_f1_0.8139666488774808__LB_0.8300102652.csv')
#     ensemble_label_2 = pd.read_csv('./ensemble_final/OPTUNA23_14_lr_0.00012099442459783506_val_f1_0.8261092405068314.csv')
#     ensemble_label_3 = pd.read_csv('./ensemble_final/OPTUNA13_11_lr_0.025847566789958795_val_f1_0.8175652239826874__LB_0.8180084031.csv')
#     ensemble_label_4 = pd.read_csv('./ensemble_final/OPTUNA18_15_lr_0.00014567432582401115_val_f1_0.8155812599500116.csv')
#     ensemble_label_5 = pd.read_csv('./ensemble_final/train_with_affine_aug_epochs8_16_lr_0.0003_val_f1_0.8102715125929859.csv')

#     arange_arr1 = np.arange(0,1,0.1)
#     missed_arr  = [1000]
#     # 에러방지용 1000넣음
#     l_k_m_value = []
#     # # 이거 앙상블할때 
#     for l in arange_arr1:
#         for k in arange_arr1:
#             for m in arange_arr1:
#                 for a in arange_arr1:
#                     for g in arange_arr1:
#                         all_arr = []
#                         for j in range(len(ensemble_label_1)):
#                             arr1 = ensemble_label_1['score_88'][j]
#                             arr1 = l*np.array(eval(arr1))

#                             arr2 = ensemble_label_2['score_88'][j]
#                             arr2 = k*np.array(eval(arr2))

#                             arr3 = ensemble_label_3['score_88'][j]
#                             arr3 = m*np.array(eval(arr3))

#                             arr4 = ensemble_label_4['score_88'][j]
#                             arr4 = m*np.array(eval(arr4))

#                             arr4 = ensemble_label_4['score_88'][j]
#                             arr4 = m*np.array(eval(arr4))

#                             arr5 = ensemble_label_4['score_88'][j]
#                             arr5 = m*np.array(eval(arr5))

#                             if np.argmax(arr1)!=np.argmax(arr2) and np.argmax(arr1)!=np.argmax(arr3) and np.argmax(arr1)!=np.argmax(arr4) and  :
#                                 if max(arr1)>max(arr2) and max(arr1)>max(arr3) and max(arr1)>max(arr4):
#                                     all_arr.append(ensemble_label_1['label'][j])
                                
#                                 elif max(arr2)>max(arr1) and max(arr2)>max(arr3) and max(arr2)>max(arr4):
#                                     all_arr.append(ensemble_label_2['label'][j])
                                
#                                 elif max(arr3)>max(arr1) and max(arr3)>max(arr2) and max(arr3)>max(arr4):
#                                     all_arr.append(ensemble_label_3['label'][j])
                                
#                                 elif max(arr4)>max(arr1) and max(arr4)>max(arr2) and max(arr4)>max(arr3):
#                                     all_arr.append(ensemble_label_4['label'][j])

#                                 # 에러방지용 append 입니다. 알아서 잘처리하셈 아, 세개가 서로 다 같으면 걍 이거넣어도 똑같으니깐. 
#                                 # max값이 두개가똑같고 하나가 다르면 이걸로처리하라 이뜻이네요. 흠 이것도 수정이 필요해보임.
#                                 else:
#                                     all_arr.append(ensemble_label_1['label'][j])

#                             else:
#                                 all_arr.append(ensemble_label_1['label'][j])

#                         df = pd.DataFrame()
#                         df['index'] = ensemble_label_1['index']
#                         df['label'] = all_arr
#                         df.to_csv(path+'ensemble_.csv', index=False)

#                         value_count__ = how_much_diffrent(path=path,
#                                         pred_csv=pred_csv,
#                                         gt_csv=gt_csv)


#                 # print(value_count__)
#                 # print(f'l:{l} k:{k}')
#                 if min(missed_arr)>value_count__:
#                     print(value_count__)
#                     print(f'min l 값:{l}')
#                     print(f'min k 값:{k}')
#                     print(f'min m 값:{m}')
#                     print(f'min m 값:{a}')
#                     l_k_m_value.append(l)
#                     l_k_m_value.append(k)
#                     l_k_m_value.append(m)
#                     l_k_m_value.append(a)
#                 missed_arr.append(value_count__)

#     print(f'min_l      값:{l_k_m_value[-4]}')
#     print(f'min_k      값:{l_k_m_value[-3]}')
#     print(f'min_m      값:{l_k_m_value[-2]}')
#     print(f'min_a      값:{l_k_m_value[-1]}')
#     print(f'min_missed 값:{min(missed_arr)}')
    
    # df = pd.DataFrame()
    # df['index'] = ensemble_label_1['index']
    # df['label'] = all_arr
    # ensemble_csv 저장하려면 이거 다시키셈
    # df.to_csv('./anomaly_detection/ensemble/ensemble_.csv', index=False)


def previous_test(path, pred_csv, gt_csv):

    path = path
    # 비교대상 여기넣기
    pred_csv = pred_csv

    gt_csv = gt_csv
    
    how_much_diffrent(path,pred_csv,gt_csv)

if __name__ == '__main__':
    path='./ensemble_final/'
    pred_csv='ensemble_.csv'
    gt_csv='human_label.csv'

    # ensemble_test(path, pred_csv, gt_csv)

    # './')
    # 이것도 파일 여기에 저장하기 ensemble1으로 ㅎ...
    # './anomaly_detection/ensemble/')

    # pred_csv='ensemble_.csv'
    # gt_csv='human_label.csv'
    # path='./anomaly_detection/ensemble/'
    # gt_csv='human_label.csv'
    # pred_csv='OPTUNA13_11_lr_0.025847566789958795_val_f1_0.8175652239826874__LB_0.8180084031.csv'
    # previous_test(path, pred_csv, gt_csv)



# 앙상블저장할때사용
save_min_ensemble = True
if save_min_ensemble:
    l, k, m, a, f = 1, 1, 1, 1, 1
    ensemble_label_1 = pd.read_csv('./ensemble_final/OPTUNA19_19_lr_4.956248851278424e-05_val_f1_0.8139666488774808__LB_0.8300102652.csv')
    ensemble_label_2 = pd.read_csv('./ensemble_final/OPTUNA23_14_lr_0.00012099442459783506_val_f1_0.8261092405068314.csv')
    ensemble_label_3 = pd.read_csv('./ensemble_final/OPTUNA13_11_lr_0.025847566789958795_val_f1_0.8175652239826874__LB_0.8180084031.csv')
    ensemble_label_4 = pd.read_csv('./ensemble_final/OPTUNA18_15_lr_0.00014567432582401115_val_f1_0.8155812599500116.csv')
    ensemble_label_5 = pd.read_csv('./ensemble_final/train_with_affine_aug_epochs8_16_lr_0.0003_val_f1_0.8102715125929859.csv')

    all_arr = []
    for j in range(len(ensemble_label_1)):
        arr1 = ensemble_label_1['score_88'][j]
        arr1 = l*np.array(eval(arr1))

        arr2 = ensemble_label_2['score_88'][j]
        arr2 = k*np.array(eval(arr2))

        arr3 = ensemble_label_3['score_88'][j]
        arr3 = m*np.array(eval(arr3))

        arr4 = ensemble_label_4['score_88'][j]
        arr4 = a*np.array(eval(arr4))

        arr5 = ensemble_label_5['score_88'][j]
        arr5 = f*np.array(eval(arr5))

        if np.argmax(arr1)!=np.argmax(arr2) and np.argmax(arr1)!=np.argmax(arr3) and np.argmax(arr1)!=np.argmax(arr4) and np.argmax(arr1)!=np.argmax(arr5):
            if max(arr1)>max(arr2) and max(arr1)>max(arr3) and max(arr1)>max(arr4) and max(arr1)>max(arr5):
                all_arr.append(ensemble_label_1['label'][j])

            elif max(arr2)>max(arr1) and max(arr2)>max(arr3) and max(arr2)>max(arr4) and max(arr1)>max(arr5):
                all_arr.append(ensemble_label_2['label'][j])
            
            elif max(arr3)>max(arr1) and max(arr3)>max(arr2) and max(arr3)>max(arr4) and max(arr3)>max(arr5):
                all_arr.append(ensemble_label_3['label'][j])
            
            elif max(arr4)>max(arr1) and max(arr4)>max(arr2) and max(arr4)>max(arr3) and max(arr4)>max(arr5):
                all_arr.append(ensemble_label_4['label'][j])

            elif max(arr5)>max(arr1) and max(arr5)>max(arr2) and max(arr5)>max(arr3) and max(arr5)>max(arr4):
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
    df.to_csv(path+'ensemble_final.csv', index=False)
    pred_csv='ensemble_final.csv'
    how_much_diffrent(path,pred_csv,gt_csv)
