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

# from utils import set_seed, transform_album
# from augmentation import transform

# 파이토치에서는 amp를 쓴다는 것은 학습할때
# torch.cuda.amp.autocast 와 torch.cuda.amp.GradScaler를 같이 쓰는 것을 의미한다.
# 여튼 두개를 쓰면, 학습할때 성능은 유지(신경망의 수렴이 잘되고)되면서, 
# GPU 메모리 소모도 줄고, GPU 연산속도도 증가한다. 
# 얼마나 빨라지냐는 GPU의 architecure마다 다르다. 
# 가령 Volta, Turing, Ampere는 2~3배. 최신의 Kepler, Maxwell, Pascal은 1~2배정도 빨라진다.

# 밑에나온 epochs보다 1이작아야함 epoch가 1부터 시작해서
batch_size_arr = [16]
lr_arr = [3e-4]
epochs = 23
seed = 24
# 보고 15 epochs 도 괜찮기도 한듯
# 21~22epochs로 가야함 여기서 성능이 제일좋고 그다음부터 떨어짐

# 지금 시드없앴음 seed를 정하면 augmentation이 똑같은 augmentation만 
# 반복되서 다양성을 죽일거같음 ㅠㅠ
# seed만 바꿔놓고 재현가능한지 한번 더 확인해야함.

########## 실제 자료 이용시에는 train,test ORIGINAL을 이용하세요.###########
########## 테스트 자료 이용시에는 train,test 이용하세요.###########
# pathTrain = './anomaly_detection/dataset/train/'
# pathTrain = './anomaly_detection/dataset/train_original/'
# pathTest = './anomaly_detection/dataset/test/'
# pathTest = './anomaly_detection/dataset/test_original/'


# # # 변경됨
pathTrain = './anomaly_detection/dataset/train_with_affine_aug/'
pathTest = './anomaly_detection/dataset/test_original/'


# 단순히 lr만 3e-4로 바꿨을뿐인데 0.61 -> 0.71로 성능향상 ㄷㄷ?;;
# -> 거기에 image augmentation 적용 -> 0.83 ㄷㄷ


pathLabel = './anomaly_detection/dataset/'

device = torch.device('cuda')

train_png = sorted(glob(pathTrain+'/*.png'))
val_png = sorted(glob(pathTest+'/*.png'))


train_y = pd.read_csv(pathLabel+"train_with_affine_aug.csv")
# 이거 augmentation 한게아니라 class balanced 하게 맞춰준거밖에없음

# 변경됨
train_labels = train_y["label"]

val_y = pd.read_csv(pathLabel+"human_label.csv")
val_labels = val_y["label"]

# np.unique를 사용할 때 unique한 라벨들을 얻어낼 수 있다.

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
    # efficient net test

    # img = cv2.resize(img, (224, 224))
    # vit base test
    return img

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
        # index가 엄청많긌네 그럼 여기 위에있어야하는게 맞긴하겠다 잠심나

        # 이거 넘파이? 이루어져있을듯

        if self.mode=='train':
            augmentation = random.randint(0,3)

            # 0,1,2의 수 중 하나를 반환 augmentation==0 이면 아무것도 안하는듯
            if augmentation==1:
                img = img[::-1].copy()
                # 수평변환

            elif augmentation==2:
                img = img[:,::-1].copy()
                # 수직변환
            
            elif augmentation==3:
                # pass
                transform_ = transforms.Compose([
                                    transforms.ToTensor(),
                                    # transforms.RandomErasing(scale=(0.01,0.05),ratio=(0.01,0.05), p=1),
                                    transforms.RandomErasing(scale=(0.01,0.04),ratio=(0.01,0.09), p=1),
                                    # transforms.Pad(40, fill=0),
                                    transforms.RandomCrop(512),
                                    transforms.RandomPerspective(distortion_scale=0.1,p=1),
                                    # pad 변경했음
                                    transforms.ToPILImage()
                ])
                img = transform_(img)
                # img.show()

        if self.mode=='val':
            pass
        if self.mode=='test':
            pass

        # img = np.array(img)
        # 여기서 이렇게 normalize하면 안되지않나? dataloader에서 해야하는거같은데..
        # transforms.Pad(2), transforms.RandomCrop(28),
        # 이것도 알아보셈.

        
        # 이거돌려보고 바로 pad이거 돌리기.

        img = transforms.ToTensor()(img)
        img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
        label = self.labels[idx]
        return img, label
    
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # self.model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=88)
        self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=88)
        # 나중에이거 efficientnet b4으로 바꾸고도 테스트해보깅
        # self.model = timm.create_model('vit_base_patch8_224', pretrained=True, num_classes=88)
        
        
    def forward(self, x):
        x = self.model(x)
        return x

for batch_size in batch_size_arr:
    for lr in lr_arr:
        # Train
        train_dataset = Custom_dataset(train_imgs, train_labels, mode='train')
        # 만약 normalize 한다면 train 뿐만아니라 validation도 nomalize 해야함.
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

        # Test
        val_dataset = Custom_dataset(val_imgs, val_labels, mode='val')
        val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)


        def score_function(real, pred):
            score = f1_score(real, pred, average="macro")

            return score

        model = Network().to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
            print(f'TRAIN loss : {train_loss:.5f}  f1: {train_f1:.5f} lr: {lr} batch: {batch_size}')
            performance.write(f'epochs: {epoch} train loss: {train_loss}  train f1: {train_f1:.5f}  lr: {lr} batch: {batch_size} \n')

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
            print(f'VAL loss : {val_loss:.5f}   f1: {val_f1:.5f} lr: {lr} batch: {batch_size}')
            performance.write(f'epochs: {epoch} val loss: {val_loss}  val f1:{val_f1:.5f}  lr: {lr} batch: {batch_size}\n')
            # 나중에 찾기쉽게하기위해 seed랑 val f1이랑 자리바꿨다.
            performance.close()

            f_pred = []
            own_pred = []

            val_dataset_2 = Custom_dataset(np.array(val_imgs), np.array(["tmp"]*len(val_imgs)), mode='test')
            val_loader_2 = DataLoader(val_dataset_2, shuffle=False, batch_size=batch_size)

            with torch.no_grad():
                for batch in (val_loader_2):
                    x = torch.tensor(batch[0], dtype = torch.float32, device = device)
                    with torch.cuda.amp.autocast():
                        pred = model(x)

                    for i in range(len(batch[0])):
                        own_pred.extend([pred[i].detach().cpu().numpy().tolist()])
                    f_pred.extend(pred.argmax(1).detach().cpu().numpy().tolist())

            label_decoder = {val:key for key, val in label_unique.items()}
            f_result = [label_decoder[result] for result in f_pred]

            add_arr = []
            for j in range(len(own_pred)):
                arr = own_pred[j]
                arr = [str(i) for i in arr]
                arr = (','.join(arr))
                add_arr.append(arr)

            submission = pd.read_csv(pathLabel+"sample_submission.csv")
            submission["label"] = f_result
            submission["score_88"] = add_arr
            submission.to_csv(pathLabel+f"submissions/828TEST{epoch}_{batch_size}_lr_{lr}_val_f1_{val_f1}.csv", index = False)
            gc.collect()

# epoch : 1/25    time : 394s/9446s
# TRAIN loss : 0.61711  f1: 0.41985 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 1/25
# VAL loss : 1.10604   f1: 0.55915 lr: 0.0003 batch: 16
# epoch : 2/25    time : 296s/6797s
# TRAIN loss : 0.19704  f1: 0.73594 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 2/25
# VAL loss : 0.83345   f1: 0.67058 lr: 0.0003 batch: 16
# epoch : 3/25    time : 325s/7144s
# TRAIN loss : 0.11322  f1: 0.84920 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 3/25
# VAL loss : 0.75337   f1: 0.71252 lr: 0.0003 batch: 16
# epoch : 4/25    time : 306s/6424s
# TRAIN loss : 0.08870  f1: 0.89780 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 4/25
# VAL loss : 0.93626   f1: 0.72712 lr: 0.0003 batch: 16
# epoch : 5/25    time : 305s/6106s
# TRAIN loss : 0.07747  f1: 0.91355 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 5/25
# VAL loss : 0.76917   f1: 0.76809 lr: 0.0003 batch: 16
# epoch : 6/25    time : 312s/5928s
# TRAIN loss : 0.05136  f1: 0.94431 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 6/25
# VAL loss : 0.89933   f1: 0.76464 lr: 0.0003 batch: 16
# epoch : 7/25    time : 294s/5297s
# TRAIN loss : 0.05225  f1: 0.94628 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 7/25
# VAL loss : 1.03727   f1: 0.75414 lr: 0.0003 batch: 16
# epoch : 8/25    time : 304s/5172s
# TRAIN loss : 0.06280  f1: 0.93330 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 8/25
# VAL loss : 0.78247   f1: 0.79631 lr: 0.0003 batch: 16
# epoch : 9/25    time : 294s/4699s
# TRAIN loss : 0.04472  f1: 0.95771 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 9/25
# VAL loss : 0.90712   f1: 0.78042 lr: 0.0003 batch: 16
# epoch : 10/25    time : 291s/4366s
# TRAIN loss : 0.03404  f1: 0.96626 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 10/25
# VAL loss : 0.98368   f1: 0.79224 lr: 0.0003 batch: 16
# epoch : 11/25    time : 291s/4074s
# TRAIN loss : 0.03708  f1: 0.96105 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 11/25
# VAL loss : 1.15623   f1: 0.76501 lr: 0.0003 batch: 16
# epoch : 12/25    time : 295s/3834s
# TRAIN loss : 0.02671  f1: 0.96781 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 12/25
# VAL loss : 0.84600   f1: 0.77390 lr: 0.0003 batch: 16
# epoch : 13/25    time : 296s/3555s
# TRAIN loss : 0.03390  f1: 0.96841 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 13/25
# VAL loss : 0.93520   f1: 0.79964 lr: 0.0003 batch: 16
# epoch : 14/25    time : 294s/3233s
# TRAIN loss : 0.03760  f1: 0.96298 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 14/25
# VAL loss : 0.98873   f1: 0.79181 lr: 0.0003 batch: 16
# epoch : 15/25    time : 292s/2921s
# TRAIN loss : 0.03609  f1: 0.96422 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 15/25
# VAL loss : 0.98919   f1: 0.76527 lr: 0.0003 batch: 16
# epoch : 16/25    time : 293s/2635s
# TRAIN loss : 0.01819  f1: 0.98850 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 16/25
# VAL loss : 0.92797   f1: 0.79205 lr: 0.0003 batch: 16
# epoch : 17/25    time : 292s/2333s
# TRAIN loss : 0.02294  f1: 0.97538 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 17/25
# VAL loss : 0.84701   f1: 0.80384 lr: 0.0003 batch: 16
# epoch : 18/25    time : 326s/2282s
# TRAIN loss : 0.02826  f1: 0.97124 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 18/25
# VAL loss : 0.90895   f1: 0.81791 lr: 0.0003 batch: 16
# epoch : 19/25    time : 294s/1764s
# TRAIN loss : 0.03051  f1: 0.97624 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 19/25
# VAL loss : 1.14192   f1: 0.76485 lr: 0.0003 batch: 16
# epoch : 20/25    time : 298s/1489s
# TRAIN loss : 0.02261  f1: 0.98092 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 20/25
# VAL loss : 1.15190   f1: 0.79264 lr: 0.0003 batch: 16
# epoch : 21/25    time : 298s/1190s
# TRAIN loss : 0.02721  f1: 0.97052 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 21/25
# VAL loss : 0.98495   f1: 0.80663 lr: 0.0003 batch: 16
# epoch : 22/25    time : 297s/890s
# TRAIN loss : 0.01948  f1: 0.98219 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 22/25
# VAL loss : 0.92115   f1: 0.78599 lr: 0.0003 batch: 16
# epoch : 23/25    time : 298s/596s
# TRAIN loss : 0.02264  f1: 0.97855 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 23/25
# VAL loss : 1.03723   f1: 0.77863 lr: 0.0003 batch: 16
# epoch : 24/25    time : 296s/296s
# TRAIN loss : 0.01949  f1: 0.98021 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 24/25
# VAL loss : 0.78868   f1: 0.82576 lr: 0.0003 batch: 16
# epoch : 25/25    time : 295s/0s
# TRAIN loss : 0.01941  f1: 0.97886 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 25/25
# VAL loss : 0.91196   f1: 0.77406 lr: 0.0003 batch: 16

# seed는 코드가 하나만달라져도 바뀌구나....... 더 많은 epochs로 실험할것