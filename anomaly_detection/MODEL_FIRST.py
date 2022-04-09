import imghdr
import warnings
from PIL import Image
import random
warnings.filterwarnings('ignore')

from glob import glob
import pandas as pd
import numpy as np 
from tqdm import tqdm
import cv2
import gc

import timm

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, accuracy_score
import time

from utils import set_seed, transform_album
from augmentation import transform

# 파이토치에서는 amp를 쓴다는 것은 학습할때
# torch.cuda.amp.autocast 와 torch.cuda.amp.GradScaler를 같이 쓰는 것을 의미한다.
# 여튼 두개를 쓰면, 학습할때 성능은 유지(신경망의 수렴이 잘되고)되면서, 
# GPU 메모리 소모도 줄고, GPU 연산속도도 증가한다. 
# 얼마나 빨라지냐는 GPU의 architecure마다 다르다. 
# 가령 Volta, Turing, Ampere는 2~3배. 최신의 Kepler, Maxwell, Pascal은 1~2배정도 빨라진다.

batch_size = 32
epochs = 21

# 보고 15 epochs 도 괜찮기도 한듯
# 21~22epochs로 가야함 여기서 성능이 제일좋고 그다음부터 떨어짐

# set_seed(seed)
# 지금 시드없앴음 seed를 정하면 augmentation이 똑같은 augmentation만 
# 반복되서 다양성을 죽일거같음 ㅠㅠ
# seed만 바꿔놓고 재현가능한지 한번 더 확인해야함.

########## 실제 자료 이용시에는 train,test ORIGINAL을 이용하세요.###########
########## 테스트 자료 이용시에는 train,test 이용하세요.###########
# pathTrain = './anomaly_detection/dataset/train/'
# pathTrain = './anomaly_detection/dataset/train_original/'
# pathTest = './anomaly_detection/dataset/test/'
# pathTest = './anomaly_detection/dataset/test_original/'


# # 변경됨
# pathTrain = './anomaly_detection/dataset/train_original/'
pathTrain = './anomaly_detection/dataset/train_with_label_transistor_aug/'

pathTest = './anomaly_detection/dataset/test_original/'

# 단순히 lr만 3e-4로 바꿨을뿐인데 0.61 -> 0.71로 성능향상 ㄷㄷ?;;


pathLabel = './anomaly_detection/dataset/'

device = torch.device('cuda')

train_png = sorted(glob(pathTrain+'/*.png'))
val_png = sorted(glob(pathTest+'/*.png'))

train_y = pd.read_csv(pathLabel+"label_train_with_label_transistor_aug.csv")
# 이거 augmentation 한게아니라 class balanced 하게 맞춰준거밖에없음

# 변경됨
train_labels = train_y["label"]

val_y = pd.read_csv(pathLabel+"baseline_test.csv")
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
    img = cv2.resize(img, (512, 512))
    return img

train_imgs = [img_load(m) for m in tqdm(train_png)]
val_imgs = [img_load(n) for n in tqdm(val_png)]

for se_ed in range(100):
    seed = se_ed


    class Custom_dataset(Dataset):

        def __init__(self, img_paths, labels, mode='train'):
            self.img_paths = img_paths
            self.labels = labels


            self.mode=mode

        def __len__(self):
            return len(self.img_paths)

        def __getitem__(self, idx):
            img = self.img_paths[idx]
            # index가 엄청많긌네 그럼 여기 위에있어야하는게 맞긴하겠다 잠심나

            # 이거 넘파이? 이루어져있을듯

            if self.mode=='train':
                augmentation = random.randint(0,2)
                # 0,1,2의 수 중 하나를 반환 augmentation==0 이면 아무것도 안하는듯
                if augmentation==1:
                    img = img[::-1].copy()
                    # 수평변환

                elif augmentation==2:
                    img = img[:,::-1].copy()
                    # 수직변환
                
                elif augmentation==3:
                    img = img[::-1].copy()
                    img = img[:,::-1].copy()
                    # 수평수직변환

                # 이거 random(0,4)로바꾸고 수평변환 수직변환 동시에 적용된거 넣으면 어케될까

                #  아 이거 random.randint해도 random_seed잡혀있어서 안먹힘.
                # augmentation = random.random()
                # print(f'random value : {augmentation}')
                # if augmentation<0.1:
                    # img = cv2.flip(img, -1)
                    # print('now is flip')
                    # cv2.imshow('flip',img)
                    # cv2.waitKey(0)

                # if 0.1<augmentation<0.2:
                #     img = transform(img, seed)
                    # print('now is affine')
                    # cv2.imshow('affine',img)
                    # cv2.waitKey(0)

                # if 0<augmentation<0.3:
                #     RandomCrop_P, HorizontalFlip_P, VerticalFlip_P = 0.7, 0.7, 0.7
                #     img = transform_album(img, RandomCrop_P, HorizontalFlip_P, VerticalFlip_P)
                    # print('now is horizontal, vertical')
                    # cv2.imshow('horizontal, vertical',img)
                    # cv2.waitKey(0)

                # elif 0.3<augmentation<0.4:
                #     RandomCrop_P, HorizontalFlip_P, VerticalFlip_P = 0.6, 0.5, 0.5
                #     img = cv2.flip(img, -1)
                #     img = transform_album(img, RandomCrop_P, HorizontalFlip_P, VerticalFlip_P)
                #     img = transform(img, seed)
                    # print('now is all of setting')
                    # cv2.imshow('all',img)
                    # cv2.waitKey(0)

                # elif 0.4<augmentation<1:
                #     pass

            if self.mode=='val':
                pass


            if self.mode=='test':
                pass

            # img = np.array(img)
            # 여기서 이렇게 normalize하면 안되지않나? dataloader에서 해야하는거같은데..
            img = transforms.ToTensor()(img)
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
    train_dataset = Custom_dataset(train_imgs, train_labels, mode='train')
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

    # Test
    val_dataset = Custom_dataset(val_imgs, val_labels, mode='val')
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)


    def score_function(real, pred):
        score = f1_score(real, pred, average="macro")

        return score

    model = Network().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler() 


    model.train()
    best=0
    for epoch in range(epochs):
        start=time.time()
        train_loss = 0
        train_pred=[]
        train_y=[]
        performance = open('./anomaly_detection/performance_record.txt','a')
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
        performance.write(f'epochs: {epoch} train loss: {train_loss}  train f1: {train_f1:.5f}  seed: {seed} \n')

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
        performance.write(f'epochs: {epoch} val loss: {val_loss}  val f1:{val_f1:.5f}  seed: {seed}\n')
        # 나중에 찾기쉽게하기위해 seed랑 val f1이랑 자리바꿨다.
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

    submission["label"] = f_result

    submission.to_csv(pathLabel+f"submissions/baseline_seed_{se_ed}.csv", index = False)
    gc.collect()

