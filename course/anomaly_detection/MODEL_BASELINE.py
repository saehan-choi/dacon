import imghdr
import warnings
from PIL import Image
import random
warnings.filterwarnings('ignore')

# 데이터에 더 신경을쓰자.
# class imbalance도 해결해야함.
# 없는애들은 클래스를 augmentation 으로 다 맞추고 난뒤에 그다음에 평가할것
# 필요없는 이미지들은 빼야함. testset과 비교후 쓰레기 데이터들은 빼라.

from glob import glob
import pandas as pd
import numpy as np 
from tqdm import tqdm
import cv2

import timm

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.metrics import f1_score
import time

from utils import set_seed, transform_album
from augmentation import transform

# 파이토치에서는 amp를 쓴다는 것은 학습할때
# torch.cuda.amp.autocast 와 torch.cuda.amp.GradScaler를 같이 쓰는 것을 의미한다.
# 여튼 두개를 쓰면, 학습할때 성능은 유지(신경망의 수렴이 잘되고)되면서, 
# GPU 메모리 소모도 줄고, GPU 연산속도도 증가한다. 
# 얼마나 빨라지냐는 GPU의 architecure마다 다르다. 
# 가령 Volta, Turing, Ampere는 2~3배. 최신의 Kepler, Maxwell, Pascal은 1~2배정도 빨라진다.

for se_ed in range(100,1000):
    # 이거 자꾸 에러 쳐떠서 다시한다.
    try:
        seed = se_ed
        batch_size = 16
        epochs = 15

        # seed별 기록 남겨놓기!
        # seed만 바꿔놓고 재현가능한지 한번 더 확인해야함.



        ########## 실제 자료 이용시에는 train,test ORIGINAL을 이용하세요.###########
        ########## 테스트 자료 이용시에는 train,test 이용하세요.###########
        # pathTrain = './anomaly_detection/dataset/train/'
        pathTrain = './anomaly_detection/dataset/train_original/'
        # pathTest = './anomaly_detection/dataset/test/'
        pathTest = './anomaly_detection/dataset/test_original/'
        pathLabel = './anomaly_detection/dataset/'

        device = torch.device('cuda')

        train_png = sorted(glob(pathTrain+'/*.png'))
        val_png = sorted(glob(pathTest+'/*.png'))



        train_y = pd.read_csv(pathLabel+"train_df.csv")
        train_labels = train_y["label"]


        val_y = pd.read_csv(pathLabel+"validationSET.csv")
        val_labels = val_y["label"]


        label_unique = sorted(np.unique(train_labels))
        label_unique = {key:value for key,value in zip(label_unique, range(len(label_unique)))}
        # print(train_labels)
        # 0       transistor-good
        # 1          capsule-good

        # print(label_unique)
        # {'bottle-broken_large': 0, 'bottle-broken_small': 1,...
        # 라벨별로 unique하게 쪼개 놓은거임
        train_labels = [label_unique[k] for k in train_labels]


        label_unique = sorted(np.unique(val_labels))

        label_unique = {key:value for key,value in zip(label_unique, range(len(label_unique)))}
        val_labels = [label_unique[k] for k in val_labels]

        # print(train_labels)
        # [72, 15, 72, 76, 3, 76, 15, 55, 4...]
        # 단순히 이 숫자들은 label_unique의 숫자에 불과함 ex) 72 -> in label_unique -> transistor-good


        def img_load(path):
            img = cv2.imread(path)[:,:,::-1]
            # img = cv2.imread(path) -> 이거는 BGR 로 읽어들임
            # BGR 을 RGB로 변환

            # print(img.shape)
            # image shape은 여기임
            img = cv2.resize(img, (512, 512))
            # 지금은 512로바꾸는데 나중에 이미지 별로 사이즈도 조절해바라.


            return img

        # CROSS VALIDATION 생성 할 것

        train_imgs = [img_load(m) for m in tqdm(train_png)]
        val_imgs = [img_load(n) for n in tqdm(val_png)]

        class Custom_dataset(Dataset):

            def __init__(self, img_paths, labels, mode='train'):
                self.img_paths = img_paths
                self.labels = labels
                self.mode=mode
            def __len__(self):
                return len(self.img_paths)
            def __getitem__(self, idx):

                img = self.img_paths[idx]

                if self.mode=='train':
                    augmentation = random.randint(0,2)
                    # 0,1,2의 수 중 하나를 반환 augmentation==0 이면 아무것도 안하는듯
                    if augmentation==1:
                        img = img[::-1].copy()
                        # 수평변환

                    elif augmentation==2:
                        img = img[:,::-1].copy()
                        # 수직변환


                if self.mode=='val':
                    pass

                img = transforms.ToTensor()(img)
                if self.mode=='test':
                    pass
                
                label = self.labels[idx]
                return img, label
            
        class Network(nn.Module):
            def __init__(self):
                super(Network, self).__init__()
                self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=88)
                
            def forward(self, x):
                x = self.model(x)
                return x

        # Train
        train_dataset = Custom_dataset(np.array(train_imgs), np.array(train_labels), mode='train')
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

        # Test
        val_dataset = Custom_dataset(np.array(val_imgs), np.array(val_labels), mode='val')
        val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)


        def score_function(real, pred):
            score = f1_score(real, pred, average="macro")

            return score

        model = Network().to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler() 


        model.train()
        best=0
        for epoch in range(epochs):
            performance = open('./anomaly_detection/performance_record.txt','a')

            start=time.time()
            train_loss = 0
            train_pred=[]
            train_y=[]

            model.train()
            for batch in (train_loader):
                # print(f'train:{len(batch[0])} {len(batch[1])}')

                optimizer.zero_grad()
                x = torch.tensor(batch[0], dtype=torch.float32, device=device)
                y = torch.tensor(batch[1], dtype=torch.long, device=device)
                with torch.cuda.amp.autocast():
                    pred = model(x)
                loss = criterion(pred, y)


                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()/len(train_loader)
                train_pred += pred.argmax(1).detach().cpu().numpy().tolist()
                train_y += y.detach().cpu().numpy().tolist()
                
                # val은 1860까지 늘어났음.
            
            train_f1 = score_function(train_y, train_pred)

            TIME = time.time() - start
            print(f'epoch : {epoch+1}/{epochs}    time : {TIME:.0f}s/{TIME*(epochs-epoch-1):.0f}s')
            print(f'TRAIN    loss : {train_loss:.5f}    f1 : {train_f1:.5f}')
            performance.write(f'epochs: {epoch} train loss: {train_loss} train f1: {train_f1:.5f} seed:{seed}\n')

            val_loss = 0
            val_pred = []
            val_y = []
            print('validation 진행중')

            model.eval()
            with torch.no_grad():
                for batch in (val_loader):
                    x = torch.tensor(batch[0], dtype = torch.float32, device = device)
                    y = torch.tensor(batch[1], dtype=torch.long, device=device)

                    with torch.cuda.amp.autocast():
                        pred = model(x)

                    loss = criterion(pred, y)

                    val_loss += loss.item()/len(val_loader)
                    val_pred += pred.argmax(1).detach().cpu().numpy().tolist()
                    val_y += y.detach().cpu().numpy().tolist()

            
            val_f1 = score_function(val_y, val_pred)
            # print('이거 계산에서 에러남')
            # 아 이거 에러나는 이유가 testset에 이미지의 라벨이 없어서 못알아 먹는거네
            # 나중에 이거 이미지갯수, 라벨링갯수 맞는지 확인한다음에 고칠것
            print(f'epoch    : {epoch+1}/{epochs}')
            print(f'VAL loss : {val_loss:.5f}        f1   : {val_f1:.5f}')
            performance.write(f'epochs: {epoch} val loss: {val_loss} val f1: {val_f1:.5f} seed:{seed}\n')
            performance.close()

        f_pred = []

        val_dataset = Custom_dataset(np.array(val_imgs), np.array(["tmp"]*len(val_imgs)), mode='test')
        val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)

        with torch.no_grad():
            for batch in (val_loader):
                x = torch.tensor(batch[0], dtype = torch.float32, device = device)
                with torch.cuda.amp.autocast():
                    pred = model(x)
                f_pred.extend(pred.argmax(1).detach().cpu().numpy().tolist())

        label_decoder = {val:key for key, val in label_unique.items()}

        f_result = [label_decoder[result] for result in f_pred]


        # 제출물 생성
        submission = pd.read_csv(pathLabel+"sample_submission.csv")

        # print(submission)
        submission["label"] = f_result

        # 이거 index가 있어야 평가가되네;;
            # 인덱스 있어야해서 놔둘게

        submission.to_csv(pathLabel+f"submissions/test_seed_{seed}.csv", index = False)

    except:
        pass