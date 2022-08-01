import pandas as pd




def how_much_diffrent(path, pred_csv, gt_csv):
    global value_count__
    pred_label = pd.read_csv(path+pred_csv)
    gt_label = pd.read_csv(path+gt_csv)

    pred = pred_label['label']
    gt = gt_label['label']

    compare = gt.compare(pred, align_axis=0)

    gt = compare.loc[:,'self'].reset_index()
    pred = compare.loc[:,'other'].reset_index()


    gt.columns = ['index','gt']
    pred.columns = ['index','pred']

    df = pd.concat([gt,pred['pred']],axis=1)
    df['file_name'] = df['index']+20000

    pd.set_option('display.max_rows',None)
    print(df['gt'].value_counts())
    print(f"{df['gt'].value_counts().sum()} missed")
    # 여기서 제일많은 순서대로 고쳐 나가야 합니다.
    # 제가 라벨링 한게 False postive or False Negative일 확률이 있기 때문입니다.

    

    df.to_csv(path+f"howMuchDiffrent.csv", index = False)

    return df['gt'].value_counts().sum()
    # gt = gt[gt!=pred]
    # # gt에서 예측한 값들
    # print("\n\n")
    # pred = pred[pred!=gt]
    # # pred에서 예측한 값들

if __name__ == '__main__':
    
    path = './anomaly_detection/ensemble/'
    # csv만 옮기고 파일쓰면 됩니다.
    pred_csv = 'howMuchDifferent/NEW_2022_04_16_epochs17_32_lr_0.0003_val_f1_0.810311179175168.csv'
    gt_csv = 'human_label.csv'
    how_much_diffrent(path,pred_csv,gt_csv)