import pandas as pd

from autogluon.tabular import TabularPredictor

from multiprocessing import Process, Manager

from tqdm import tqdm


# configuration
class CFG:
    testXPath = './data/test_x_engineered.csv'
    sample_submission_path = './data/sample_submission.csv'
    weights_path = './weights/'
    pred_amount = 10000 # 한번에 예측하는 갯수 (조절가능)

# 가중치 로드 및 예측 결과값 반출
def pred(y_number, test_x, queue):
    load_path = CFG.weights_path+y_number+'Models-predict'
    predictor = TabularPredictor.load(load_path, require_version_match=False)
    queue.put(predictor.predict(test_x))


if __name__ == "__main__":
    # testdata 로드 및 멀티프로세싱 값 반환을 위한 queue 생성
    test_x = pd.read_csv(CFG.testXPath)
    q = Manager().Queue()


    # pred_amount의 값만큼씩 예측 진행
    for i in tqdm(range(0,len(test_x),CFG.pred_amount)):
        test_x_part = test_x[i:i+CFG.pred_amount]

        # multiprocessing 을 통한 Y01~Y14 동시예측
        for j in range(14):
            number = str(j+1).zfill(2)
            y_number = 'Y_'+ number
            output = Process(target=pred, args=(y_number, test_x_part, q,))
            output.start()
        output.join()

        # 결과값 생성
        if i==0:
            result = pd.read_csv(CFG.sample_submission_path)
        else:
            result = pd.read_csv('weights_load_result.csv')

        # multiprocessing 예측 결과값 반환
        for j in range(14):
            df = q.get()
            result.loc[i:i+CFG.pred_amount, df.name] = df

        result.to_csv('weights_load_result.csv',index=False)
    