from concurrent.futures import thread
import pandas as pd
# from sklearn import ensemble
import numpy as np
from howMuchDifferent.a_howmuch import how_much_diffrent
import threading

def ensemble_test(path, pred_csv, gt_csv):

    ensemble_label_1 = pd.read_csv('./ensemble_final/OPTUNA23_16_lr_0.0003_val_f1_0.8436755696791693_model_f1_84_checked.csv')
    ensemble_label_2 = pd.read_csv('./ensemble_final/model_test_25_17_lr_0.0001_val_f1_0.83341946235041.csv')
    ensemble_label_3 = pd.read_csv('./ensemble_final/OPTUNA22_16_lr_0.0003_val_f1_0.8254278458417583_model_f1_82_checked.csv')
                                    
    # ensemble_label_3 = pd.read_csv('./ensemble_final/828TEST12_32_lr_0.0003_val_f1_0.7923809410329947.csv')



    arange_arr1 = np.arange(0,1,0.1)
    missed_arr  = [1000]
    # 에러방지용 1000넣음
    l_k_m_value = []
    # # 이거 앙상블할때 
    for l in arange_arr1:
        for k in arange_arr1:
            for m in arange_arr1:
                all_arr = []
                for j in range(len(ensemble_label_1)):
                    arr1 = ensemble_label_1['score_88'][j]
                    arr1 = l*np.array(eval(arr1))

                    arr2 = ensemble_label_2['score_88'][j]
                    arr2 = k*np.array(eval(arr2))

                    arr3 = ensemble_label_3['score_88'][j]
                    arr3 = m*np.array(eval(arr3))

                    if np.argmax(arr1)!=np.argmax(arr2) or np.argmax(arr1)!=np.argmax(arr3):
                        if max(arr1)>max(arr2) and max(arr1)>max(arr3):
                            all_arr.append(ensemble_label_1['label'][j])
                        
                        elif max(arr2)>max(arr1) and max(arr2)>max(arr3):
                            all_arr.append(ensemble_label_2['label'][j])
                        
                        elif max(arr3)>max(arr1) and max(arr3)>max(arr2):
                            all_arr.append(ensemble_label_3['label'][j])
                        
                        # 에러방지용 append 입니다. 알아서 잘처리하셈 아, 세개가 서로 다 같으면 걍 이거넣어도 똑같으니깐.
                        else:
                            all_arr.append(ensemble_label_1['label'][j])

                    else:
                        all_arr.append(ensemble_label_1['label'][j])

                df = pd.DataFrame()
                df['index'] = ensemble_label_1['index']
                df['label'] = all_arr
                df.to_csv(path+'ensemble_.csv', index=False)

                value_count__ = how_much_diffrent(path=path,
                                pred_csv=pred_csv,
                                gt_csv=gt_csv)
                

                # print(value_count__)
                # print(f'l:{l} k:{k}')
                if min(missed_arr)>value_count__:
                    print(value_count__)
                    print(f'min l 값:{l}')
                    print(f'min k 값:{k}')
                    print(f'min m 값:{m}')
                    l_k_m_value.append(l)
                    l_k_m_value.append(k)
                    l_k_m_value.append(m)
                missed_arr.append(value_count__)

    print(f'min_l      값:{l_k_m_value[-3]}')
    print(f'min_k      값:{l_k_m_value[-2]}')
    print(f'min_m      값:{l_k_m_value[-1]}')    
    print(f'min_missed 값:{min(missed_arr)}')
    



    # all_arr = []
    # for j in range(len(ensemble_label_1)):
    #     arr1 = ensemble_label_1['score_88'][j]
    #     # arr1 = np.array(eval(arr1))
    #     arr1 = np.array(eval(arr1))

    #     arr2 = ensemble_label_2['score_88'][j]
    #     # arr2 = np.array(eval(arr2))
    #     arr2 = np.array(eval(arr2))

    #     arr3 = ensemble_label_3['score_88'][j]
    #     # arr2 = np.array(eval(arr2))
    #     arr3 = np.array(eval(arr3))
        
# 0.4, 0.3, 0.4 
# 1 2 3 등순으로 나열시 이게 베스트

    #     if np.argmax(arr1)!=np.argmax(arr2) and np.argmax(arr1)!=np.argmax(arr3):
    #         if max(arr1)>max(arr2) and max(arr1)>max(arr3):
    #             all_arr.append(ensemble_label_1['label'][j])
    #             # print(ensemble_label_1['label'][j])
    #             pass
    #         elif max(arr2)>max(arr1) and max(arr2)>max(arr3):
    #             all_arr.append(ensemble_label_2['label'][j])
    #             # print(ensemble_label_2['label'][j])
    #             pass
    #         elif max(arr3)>max(arr1) and max(arr3)>max(arr2):
    #             all_arr.append(ensemble_label_3['label'][j])
    #             # print(ensemble_label_3['label'][j])
                
    #     else:
    #         all_arr.append(ensemble_label_1['label'][j])
    #         # print(ensemble_label_1['label'][j])


    # df = pd.DataFrame()
    # df['index'] = ensemble_label_1['index']
    # df['label'] = all_arr
    # # ensemble_csv 저장하려면 이거 다시키셈
    # df.to_csv('./ensemble_final/ensemble_.csv', index=False)


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

    ensemble_test(path, pred_csv, gt_csv)

    # './')
    # 이것도 파일 여기에 저장하기 ensemble1으로 ㅎ...
    # './anomaly_detection/ensemble/')

    # pred_csv='ensemble_.csv'
    # gt_csv='human_label.csv'
    # path='./ensemble_final/'
    # gt_csv='human_label.csv'
    # pred_csv='ensemble_.csv'
    # previous_test(path, pred_csv, gt_csv)
