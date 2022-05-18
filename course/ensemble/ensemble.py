import pandas as pd
# from sklearn import ensemble
import numpy as np
from howMuchDifferent.a_howmuch import how_much_diffrent

def ensemble_test(path, pred_csv, gt_csv):

    ensemble_label_1 = pd.read_csv('./anomaly_detection/ensemble/holy8_32_lr_0.0003_val_f1_0.8106405826933956.csv')
    # 이것도 파일 여기에 저장하기 ensemble1으로 ㅎ...
    ensemble_label_1 = pd.read_csv('./anomaly_detection/ensemble/howMuchDifferent/NEW_2022_04_16_epochs17_32_lr_0.0003_val_f1_0.810311179175168.csv')
    ensemble_label_2 = pd.read_csv('./anomaly_detection/ensemble/train_with_affine_aug_epochs8_16_lr_0.0003_val_f1_0.8102715125929859.csv')

    arange_arr1 = np.arange(0.6,0.8,0.01)
    missed_arr  = [1000]
    # 에러방지용 1000넣음
    l_k_value = []
    # # 이거 앙상블할때 
    for l in arange_arr1:
        for k in arange_arr1:
            all_arr = []
            for j in range(len(ensemble_label_1)):
                arr1 = ensemble_label_1['score_88'][j]
                # arr1 = np.array(eval(arr1))
                arr1 = l*np.array(eval(arr1))

                arr2 = ensemble_label_2['score_88'][j]
                # arr2 = np.array(eval(arr2))
                arr2 = k*np.array(eval(arr2))

                if np.argmax(arr1)!=np.argmax(arr2):
                    if max(arr1)>max(arr2):
                        all_arr.append(ensemble_label_1['label'][j])
                        # print(ensemble_label_1['label'][j])
                        pass
                    else:
                        all_arr.append(ensemble_label_2['label'][j])
                        # print(ensemble_label_2['label'][j])
                        pass
                else:
                    all_arr.append(ensemble_label_1['label'][j])
                    # print(ensemble_label_1['label'][j])

            df = pd.DataFrame()
            df['index'] = ensemble_label_1['index']
            df['label'] = all_arr
            # ensemble_csv 저장하려면 이거 다시키셈
            df.to_csv('./anomaly_detection/ensemble/ensemble_.csv', index=False)

            value_count__ = how_much_diffrent(path=path,
                            pred_csv=pred_csv,
                            gt_csv=gt_csv)
            

            # print(value_count__)
            # print(f'l:{l} k:{k}')
            if min(missed_arr)>value_count__:
                print(value_count__)
                print(f'min l 값:{l}')
                print(f'min k 값:{k}')
                l_k_value.append(l)
                l_k_value.append(k)
            missed_arr.append(value_count__)

    print(f'min_l      값:{l_k_value[-2]}')
    print(f'min_k      값:{l_k_value[-1]}')
    print(f'min_missed 값:{min(missed_arr)}')
    
    # all_arr = []
    # for j in range(len(ensemble_label_1)):
    #     arr1 = ensemble_label_1['score_88'][j]
    #     # arr1 = np.array(eval(arr1))
    #     arr1 = 0.7*np.array(eval(arr1))

    #     arr2 = ensemble_label_2['score_88'][j]
    #     # arr2 = np.array(eval(arr2))
    #     arr2 = 0.6*np.array(eval(arr2))

    #     if np.argmax(arr1)!=np.argmax(arr2):
    #         if max(arr1)>max(arr2):
    #             all_arr.append(ensemble_label_1['label'][j])
    #             # print(ensemble_label_1['label'][j])
    #             pass
    #         else:
    #             all_arr.append(ensemble_label_2['label'][j])
    #             # print(ensemble_label_2['label'][j])
    #             pass
    #     else:
    #         all_arr.append(ensemble_label_1['label'][j])
    #         # print(ensemble_label_1['label'][j])

    df = pd.DataFrame()
    df['index'] = ensemble_label_1['index']
    df['label'] = all_arr
    # ensemble_csv 저장하려면 이거 다시키셈
    df.to_csv('./anomaly_detection/ensemble/ensemble_.csv', index=False)

    value_count__ = how_much_diffrent(path=path,
                    pred_csv=pred_csv,
                    gt_csv=gt_csv)    

def previous_test(path, pred_csv, gt_csv):

    path = path
    # 비교대상 여기넣기
    pred_csv = pred_csv

    gt_csv = gt_csv
    
    how_much_diffrent(path,pred_csv,gt_csv)

if __name__ == '__main__':
    path='./anomaly_detection/ensemble/'
    pred_csv='ensemble_.csv'
    gt_csv='human_label.csv'

    ensemble_test(path, pred_csv, gt_csv)


    # './')
    # 이것도 파일 여기에 저장하기 ensemble1으로 ㅎ...
    # './anomaly_detection/ensemble/')

    # pred_csv='ensemble_.csv'
    # gt_csv='human_label.csv'
    # previous_test(path, pred_csv, gt_csv)