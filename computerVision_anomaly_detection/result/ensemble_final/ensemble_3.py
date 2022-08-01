from concurrent.futures import thread
import pandas as pd
# from sklearn import ensemble
import numpy as np
from howMuchDifferent.a_howmuch import how_much_diffrent


# ensemble_label_1 = pd.read_csv('./ensemble_final/ensemble_data/albumentations_aug_05-06-_53_32_lr_0.0003_val_f1_0.8257859092504767.csv')
# ensemble_label_2 = pd.read_csv('./ensemble_final/ensemble_data/OPTUNA23_16_lr_0.0003_val_f1_0.8436755696791693_model_f1_84_checked.csv')
# ensemble_label_3 = pd.read_csv('./ensemble_final/ensemble_data/OPTUNA24_16_lr_0.0003_val_f1_0.8231543342945248_model_f1_82_checked.csv')

# min_l      값:0.1
# min_k      값:0.4
# min_m      값:0.30000000000000004
# min_missed 값:213


def ensemble_test(path, pred_csv, gt_csv):

    ensemble_label_1 = pd.read_csv('./ensemble_final/ensemble_data/albumentations_aug_05-06-_53_32_lr_0.0003_val_f1_0.8257859092504767.csv')
    # ==> model_f1_82_albumentations_1_checked.py   -> 1080it 에서 체크 완료
    ensemble_label_2 = pd.read_csv('./ensemble_final/ensemble_data/OPTUNA23_16_lr_0.0003_val_f1_0.8436755696791693_model_f1_84_checked.csv')
    # ==> model_f1_84_checked.py   -> 1080ti 에서 체크 완료
    ensemble_label_3 = pd.read_csv('./ensemble_final/ensemble_data/OPTUNA24_16_lr_0.0003_val_f1_0.8231543342945248_model_f1_82_checked.csv')
    # ==> model_f1_82_checked.py

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

                    if np.argmax(arr1)!=np.argmax(arr2) and np.argmax(arr1)!=np.argmax(arr3):
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



    # best _ value 저장시 사용
    all_arr = []
    for j in range(len(ensemble_label_1)):
        arr1 = ensemble_label_1['score_88'][j]
        arr1 = l_k_m_value[-3]*np.array(eval(arr1))

        arr2 = ensemble_label_2['score_88'][j]
        arr2 = l_k_m_value[-2]*np.array(eval(arr2))

        arr3 = ensemble_label_3['score_88'][j]
        arr3 = l_k_m_value[-1]*np.array(eval(arr3))

        if np.argmax(arr1)!=np.argmax(arr2) and np.argmax(arr1)!=np.argmax(arr3):
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
    # ensemble_csv 저장하려면 이거 다시키셈
    df.to_csv('./ensemble_final/ensemble_.csv', index=False)

    # value_count__ = how_much_diffrent(path=path,
    #                 pred_csv=pred_csv,
    #                 gt_csv=gt_csv)    



    # # 이거 두개 기본앙상블할때 
    # ensemble_label_1 = pd.read_csv('./anomaly_detection/ensemble/OPTUNA19_19_lr_4.956248851278424e-05_val_f1_0.8139666488774808__LB_0.8300102652.csv')
    # ensemble_label_2 = pd.read_csv('./anomaly_detection/ensemble/OPTUNA13_11_lr_0.025847566789958795_val_f1_0.8175652239826874__LB_0.8180084031.csv')
    # ensemble_label_3 = pd.read_csv('./anomaly_detection/ensemble/OPTUNA13_11_lr_0.025847566789958795_val_f1_0.8175652239826874__LB_0.8180084031.csv')
    # # 앙상블 세번째거 넣기.

    # all_arr = []
    # for j in range(len(ensemble_label_1)):
    #     arr1 = ensemble_label_1['score_88'][j]
    #     # arr1 = np.array(eval(arr1))
    #     arr1 = 0.9*np.array(eval(arr1))

    #     arr2 = ensemble_label_2['score_88'][j]
    #     # arr2 = np.array(eval(arr2))
    #     arr2 = 0.7*np.array(eval(arr2))

    #     arr3 = ensemble_label_3['score_88'][j]
    #     # arr2 = np.array(eval(arr2))
    #     arr3 = 0.7*np.array(eval(arr3))


    #     if np.argmax(arr1)!=np.argmax(arr2) or np.argmax(arr1)!=np.argmax(arr3):
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
    # path='./anomaly_detection/ensemble/'
    # gt_csv='human_label.csv'
    # pred_csv='OPTUNA13_11_lr_0.025847566789958795_val_f1_0.8175652239826874__LB_0.8180084031.csv'
    # previous_test(path, pred_csv, gt_csv)
