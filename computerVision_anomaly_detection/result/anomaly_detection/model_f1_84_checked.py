import warnings
warnings.filterwarnings('ignore')

from glob import glob
import pandas as pd
import numpy as np 
from tqdm import tqdm
import cv2
import gc

import timm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import f1_score
from utils import *
import time

# pathTrain = './anomaly_detection/dataset/train_with_affine_aug/'
# pathTest = './anomaly_detection/dataset/test_original/'

# 'Adam', 'epochs': 24, 'batch_size': 15, 'randomErasing_p': 0.46807186533474215, 
# 'randomAdjustSharpness_factor': 0.4955186920103012, 
# 'randomRandomApply_p': 0.05804358305331545, 
# 'randomResizedCrop_scale_min': 0.9464080515555006, 
# 'randomResizedCrop_scale_max': 1.0337455027110465, 
# 'randomResizedCrop_ratio_min': 0.9082391076406687, 
# 'randomResizedCrop_ratio_max': 1.091251678330364, 
# 'randomGrayscale_p': 0.1739219736134851}. 

class CFG:
    seed = 42
    lr = 3e-4
    epochs = 23
    batch_size = 16
    
    randomErasing_p = 0.5
    randomAdjustSharpness_factor = 0.45
    randomRandomApply_p = 0
    randomResizedCrop_scale_min = 0.9
    randomResizedCrop_scale_max = 1
    randomResizedCrop_ratio_min = 0.9
    randomResizedCrop_ratio_max = 1
    randomGrayscale_p = 0.2

pathTrain = './anomaly_detection/dataset/train_with_affine_aug/'
pathTest = './anomaly_detection/dataset/test_original/'

# pathTrain = './anomaly_detection/dataset/train/'
# pathTest = './anomaly_detection/dataset/test/'


pathLabel = './anomaly_detection/dataset/'

device = torch.device('cuda')
seed = CFG.seed
set_seed(seed)
# seed 설정을 맨 위에해야 적용이 제대로 되네요.

train_png = sorted(glob(pathTrain+'/*.png'))
val_png = sorted(glob(pathTest+'/*.png'))

train_y = pd.read_csv(pathLabel+"train_with_affine_aug.csv")
train_labels = train_y["label"]

val_y = pd.read_csv(pathLabel+"human_label.csv")
val_labels = val_y["label"]

label_unique = sorted(np.unique(train_labels))
label_unique = {key:value for key,value in zip(label_unique, range(len(label_unique)))}
train_labels = [label_unique[k] for k in train_labels]

label_unique = sorted(np.unique(val_labels))
label_unique = {key:value for key,value in zip(label_unique, range(len(label_unique)))}
val_labels = [label_unique[k] for k in val_labels]

def img_load(path):
    img = cv2.imread(path)[:,:,::-1]
    img = cv2.resize(img, (512, 512))
    return img

train_imgs = [img_load(m) for m in tqdm(train_png)]
val_imgs = [img_load(n) for n in tqdm(val_png)]



class Custom_dataset(Dataset):
    def __init__(self, img_paths, labels, probability, mode='train'):
        self.img_paths = img_paths
        self.labels = labels
        self.mode=mode
        self.probablity = probability

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = self.img_paths[idx]
        
        if self.mode=='train':
            randomErasing_p, randomAdjustSharpness_factor = self.probablity[0], self.probablity[1]
            randomResizedCrop_scale_min, randomResizedCrop_scale_max, randomResizedCrop_ratio_min, randomResizedCrop_ratio_max=self.probablity[3], self.probablity[4], self.probablity[5], self.probablity[6]
            randomGrayscale_p = self.probablity[7]
            augmentation = random.randint(0,3)
            
            if augmentation==1:
                img = img[::-1].copy()

            elif augmentation==2:
                img = img[:,::-1].copy()

            elif augmentation==3:
                transform_ = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.RandomErasing(scale=(0.01,0.05),ratio=(0.01,0.05), p=randomErasing_p),
                                    transforms.RandomAdjustSharpness(randomAdjustSharpness_factor,p=1),
                                    transforms.RandomResizedCrop(512,scale=(randomResizedCrop_scale_min, randomResizedCrop_scale_max),ratio=(randomResizedCrop_ratio_min, randomResizedCrop_ratio_max)),
                                    transforms.RandomGrayscale(p=randomGrayscale_p),
                                    transforms.ToPILImage(),
                ])
                img = transform_(img)

        if self.mode=='val':
            pass

        img = transforms.ToTensor()(img)
        img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
        label = self.labels[idx]
        return img, label

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=88)

    def forward(self, x):
        x = self.model(x)
        return x


def score_function(real, pred):
    score = f1_score(real, pred, average="macro")
    return score


if __name__ == "__main__":
    model = Network().to(device)

    lr = CFG.lr
    epochs = CFG.epochs
    batch_size = CFG.batch_size

    probability = [CFG.randomErasing_p, CFG.randomAdjustSharpness_factor, CFG.randomRandomApply_p,
                CFG.randomResizedCrop_scale_min, CFG.randomResizedCrop_scale_max, CFG.randomResizedCrop_ratio_min, CFG.randomResizedCrop_ratio_max,
                CFG.randomGrayscale_p]

    optimizer = optim.Adam(model.parameters(),lr=lr)


    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler() 
    # Train
    train_dataset = Custom_dataset(train_imgs, train_labels, probability, mode='train')
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

    # Test
    val_dataset = Custom_dataset(val_imgs, val_labels, probability=None, mode='val')
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)

    model.train()
    best=0
    for epoch in range(epochs):
        start=time.time()
        train_loss = 0
        train_pred=[]
        train_y=[]
        model.train()
        for batch in (train_loader):

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

        train_f1 = score_function(train_y, train_pred)
        TIME = time.time() - start
        print(f'epoch : {epoch+1}/{epochs}    time : {TIME:.0f}s/{TIME*(epochs-epoch-1):.0f}s')
        print(f'TRAIN loss : {train_loss:.5f}  f1: {train_f1:.5f} lr: {lr} batch: {batch_size}')

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
                # 아 .. 이게 배치랑 관련이 있는듯.. 제대로되게 안나눠줘서 그런듯하다.
                # f1 score를 보면될듯 ㅎㅎ
                val_pred += pred.argmax(1).detach().cpu().numpy().tolist()
                val_y += y.detach().cpu().numpy().tolist()


        val_f1 = score_function(val_y, val_pred)

        print(f'epoch    : {epoch+1}/{epochs}')
        print(f'VAL loss : {val_loss:.5f}   f1: {val_f1:.5f} lr: {lr} batch: {batch_size}')
            
        f_pred = []
        own_pred = []

        val_dataset_2 = Custom_dataset(np.array(val_imgs), np.array(["tmp"]*len(val_imgs)), probability=None, mode='test')
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
        submission.to_csv(pathLabel+f"submissions/OPTUNA{epoch}_{batch_size}_lr_{lr}_val_f1_{val_f1}.csv", index = False)
        gc.collect()


# epoch : 1/25    time : 348s/8347s
# TRAIN loss : 0.63695  f1: 0.40591 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 1/25
# VAL loss : 1.01970   f1: 0.54500 lr: 0.0003 batch: 16
# epoch : 2/25    time : 341s/7849s
# TRAIN loss : 0.21432  f1: 0.71750 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 2/25
# VAL loss : 0.82250   f1: 0.70244 lr: 0.0003 batch: 16
# epoch : 3/25    time : 344s/7567s
# TRAIN loss : 0.12503  f1: 0.84131 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 3/25
# VAL loss : 0.74010   f1: 0.73939 lr: 0.0003 batch: 16
# epoch : 4/25    time : 343s/7206s
# TRAIN loss : 0.09486  f1: 0.89776 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 4/25
# VAL loss : 0.80755   f1: 0.73440 lr: 0.0003 batch: 16
# epoch : 5/25    time : 342s/6833s
# TRAIN loss : 0.07703  f1: 0.90765 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 5/25
# VAL loss : 0.99642   f1: 0.74736 lr: 0.0003 batch: 16
# epoch : 6/25    time : 340s/6459s
# TRAIN loss : 0.07306  f1: 0.91708 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 6/25
# VAL loss : 0.80904   f1: 0.78158 lr: 0.0003 batch: 16
# epoch : 7/25    time : 342s/6160s
# TRAIN loss : 0.05642  f1: 0.94097 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 7/25
# VAL loss : 0.72473   f1: 0.78733 lr: 0.0003 batch: 16
# epoch : 8/25    time : 344s/5844s
# TRAIN loss : 0.04114  f1: 0.95598 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 8/25
# VAL loss : 1.02501   f1: 0.76706 lr: 0.0003 batch: 16
# epoch : 9/25    time : 338s/5406s
# TRAIN loss : 0.04682  f1: 0.95285 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 9/25
# VAL loss : 0.83362   f1: 0.78656 lr: 0.0003 batch: 16
# epoch : 10/25    time : 330s/4946s
# TRAIN loss : 0.03098  f1: 0.96689 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 10/25
# VAL loss : 0.77673   f1: 0.81342 lr: 0.0003 batch: 16
# epoch : 11/25    time : 341s/4781s
# TRAIN loss : 0.03357  f1: 0.97171 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 11/25
# VAL loss : 0.98191   f1: 0.76931 lr: 0.0003 batch: 16
# epoch : 12/25    time : 342s/4441s
# TRAIN loss : 0.04376  f1: 0.94485 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 12/25
# VAL loss : 0.89212   f1: 0.79043 lr: 0.0003 batch: 16
# epoch : 13/25    time : 339s/4071s
# TRAIN loss : 0.04283  f1: 0.96533 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 13/25
# VAL loss : 0.97693   f1: 0.79110 lr: 0.0003 batch: 16
# epoch : 14/25    time : 341s/3755s
# TRAIN loss : 0.04188  f1: 0.95599 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 14/25
# VAL loss : 0.93215   f1: 0.78507 lr: 0.0003 batch: 16
# epoch : 15/25    time : 338s/3378s
# TRAIN loss : 0.03109  f1: 0.96794 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 15/25
# VAL loss : 0.82090   f1: 0.79425 lr: 0.0003 batch: 16
# epoch : 16/25    time : 341s/3070s
# TRAIN loss : 0.03056  f1: 0.97256 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 16/25
# VAL loss : 0.97863   f1: 0.80544 lr: 0.0003 batch: 16
# epoch : 17/25    time : 341s/2726s
# TRAIN loss : 0.02497  f1: 0.97688 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 17/25
# VAL loss : 0.87907   f1: 0.78991 lr: 0.0003 batch: 16
# epoch : 18/25    time : 343s/2400s
# TRAIN loss : 0.02845  f1: 0.97296 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 18/25
# VAL loss : 0.88649   f1: 0.81905 lr: 0.0003 batch: 16
# epoch : 19/25    time : 340s/2043s
# TRAIN loss : 0.03413  f1: 0.96896 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 19/25
# VAL loss : 0.89453   f1: 0.81187 lr: 0.0003 batch: 16
# epoch : 20/25    time : 343s/1716s
# TRAIN loss : 0.02685  f1: 0.97412 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 20/25
# VAL loss : 1.10253   f1: 0.79300 lr: 0.0003 batch: 16
# epoch : 21/25    time : 328s/1313s
# TRAIN loss : 0.02854  f1: 0.96868 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 21/25
# VAL loss : 0.82550   f1: 0.83257 lr: 0.0003 batch: 16
# epoch : 22/25    time : 327s/982s
# TRAIN loss : 0.02321  f1: 0.97780 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 22/25
# VAL loss : 0.91075   f1: 0.81983 lr: 0.0003 batch: 16
# epoch : 23/25    time : 324s/649s
# TRAIN loss : 0.02673  f1: 0.97227 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 23/25
# VAL loss : 0.84046   f1: 0.81240 lr: 0.0003 batch: 16
# epoch : 24/25    time : 327s/327s
# TRAIN loss : 0.02075  f1: 0.98629 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 24/25
# VAL loss : 0.87298   f1: 0.84368 lr: 0.0003 batch: 16
# epoch : 25/25    time : 327s/0s

# 이 상태에서 0.81 찍었습네다 !!!!!!!!!!!!!!!!!!!!!!!!!



# 일케나오던데 확인부탁드립니다......
# 십노답이네요 계속이렇게 나옵니다..
# 이거 컴퓨터마다 시드가 달라지네요?;;
# epoch : 1/30    time : 286s/8286s
# TRAIN loss : 0.65526  f1: 0.37972 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 1/30
# VAL loss : 1.06564   f1: 0.54704 lr: 0.0003 batch: 16
# epoch : 2/30    time : 284s/7955s
# TRAIN loss : 0.21800  f1: 0.71032 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 2/30
# VAL loss : 0.82544   f1: 0.70814 lr: 0.0003 batch: 16
# epoch : 3/30    time : 534s/14418s
# TRAIN loss : 0.13150  f1: 0.83176 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 3/30
# VAL loss : 0.76339   f1: 0.75119 lr: 0.0003 batch: 16
# epoch : 4/30    time : 277s/7191s
# TRAIN loss : 0.08711  f1: 0.88919 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 4/30
# VAL loss : 0.92094   f1: 0.73302 lr: 0.0003 batch: 16
# epoch : 5/30    time : 275s/6883s
# TRAIN loss : 0.07662  f1: 0.90904 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 5/30
# VAL loss : 0.80236   f1: 0.74740 lr: 0.0003 batch: 16
# epoch : 6/30    time : 277s/6648s
# TRAIN loss : 0.06701  f1: 0.92352 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 6/30
# VAL loss : 0.87004   f1: 0.74260 lr: 0.0003 batch: 16
# epoch : 7/30    time : 364s/8374s
# TRAIN loss : 0.04534  f1: 0.95424 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 7/30
# VAL loss : 0.81746   f1: 0.77506 lr: 0.0003 batch: 16
# epoch : 8/30    time : 282s/6194s
# TRAIN loss : 0.05392  f1: 0.94160 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 8/30
# VAL loss : 0.91864   f1: 0.78243 lr: 0.0003 batch: 16
# epoch : 9/30    time : 298s/6268s
# TRAIN loss : 0.05004  f1: 0.94409 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 9/30
# VAL loss : 0.87233   f1: 0.76035 lr: 0.0003 batch: 16
# epoch : 10/30    time : 511s/10220s
# TRAIN loss : 0.04450  f1: 0.95321 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 10/30
# VAL loss : 0.91240   f1: 0.77334 lr: 0.0003 batch: 16
# epoch : 11/30    time : 280s/5328s
# TRAIN loss : 0.03780  f1: 0.96218 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 11/30
# VAL loss : 0.94724   f1: 0.77952 lr: 0.0003 batch: 16
# epoch : 12/30    time : 279s/5019s
# TRAIN loss : 0.04849  f1: 0.94652 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 12/30
# VAL loss : 0.92898   f1: 0.79211 lr: 0.0003 batch: 16
# epoch : 13/30    time : 279s/4741s
# TRAIN loss : 0.02510  f1: 0.97634 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 13/30
# VAL loss : 0.84819   f1: 0.79543 lr: 0.0003 batch: 16
# epoch : 14/30    time : 278s/4449s
# TRAIN loss : 0.04486  f1: 0.95703 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 14/30
# VAL loss : 0.95020   f1: 0.77429 lr: 0.0003 batch: 16
# epoch : 15/30    time : 282s/4223s
# TRAIN loss : 0.02816  f1: 0.96985 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 15/30
# VAL loss : 0.98994   f1: 0.77286 lr: 0.0003 batch: 16
# epoch : 16/30    time : 279s/3911s
# TRAIN loss : 0.03127  f1: 0.97351 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 16/30
# VAL loss : 0.89356   f1: 0.78911 lr: 0.0003 batch: 16
# epoch : 17/30    time : 280s/3636s
# TRAIN loss : 0.02481  f1: 0.97531 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 17/30
# VAL loss : 0.91406   f1: 0.79494 lr: 0.0003 batch: 16
# epoch : 18/30    time : 280s/3363s
# TRAIN loss : 0.02487  f1: 0.97252 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 18/30
# VAL loss : 0.95494   f1: 0.78667 lr: 0.0003 batch: 16
# epoch : 19/30    time : 279s/3066s
# TRAIN loss : 0.02653  f1: 0.97641 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 19/30
# VAL loss : 0.98413   f1: 0.80660 lr: 0.0003 batch: 16
# epoch : 20/30    time : 282s/2816s
# TRAIN loss : 0.03439  f1: 0.96216 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 20/30
# VAL loss : 0.82983   f1: 0.80417 lr: 0.0003 batch: 16
# epoch : 21/30    time : 279s/2510s
# TRAIN loss : 0.01880  f1: 0.98232 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 21/30
# VAL loss : 1.00484   f1: 0.79576 lr: 0.0003 batch: 16
# epoch : 22/30    time : 281s/2252s
# TRAIN loss : 0.01673  f1: 0.98502 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 22/30
# VAL loss : 0.81907   f1: 0.82057 lr: 0.0003 batch: 16
# epoch : 23/30    time : 327s/2289s
# TRAIN loss : 0.02254  f1: 0.97837 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 23/30
# VAL loss : 1.01589   f1: 0.79337 lr: 0.0003 batch: 16
# epoch : 24/30    time : 277s/1663s
# TRAIN loss : 0.02826  f1: 0.96998 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 24/30
# VAL loss : 0.95701   f1: 0.79600 lr: 0.0003 batch: 16
# epoch : 25/30    time : 278s/1392s
# TRAIN loss : 0.02385  f1: 0.97817 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 25/30
# VAL loss : 1.13609   f1: 0.77793 lr: 0.0003 batch: 16
# epoch : 26/30    time : 279s/1114s
# TRAIN loss : 0.02284  f1: 0.97562 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 26/30
# VAL loss : 1.07964   f1: 0.79684 lr: 0.0003 batch: 16
# epoch : 27/30    time : 280s/841s
# TRAIN loss : 0.02631  f1: 0.97676 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 27/30
# VAL loss : 0.88629   f1: 0.79882 lr: 0.0003 batch: 16
# epoch : 28/30    time : 277s/554s
# TRAIN loss : 0.01834  f1: 0.98200 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 28/30
# VAL loss : 0.97340   f1: 0.80678 lr: 0.0003 batch: 16
# epoch : 29/30    time : 278s/278s
# TRAIN loss : 0.01976  f1: 0.98126 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 29/30
# VAL loss : 1.01783   f1: 0.79286 lr: 0.0003 batch: 16
# epoch : 30/30    time : 282s/0s
# TRAIN loss : 0.02182  f1: 0.97768 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 30/30
# VAL loss : 1.04270   f1: 0.81020 lr: 0.0003 batch: 16