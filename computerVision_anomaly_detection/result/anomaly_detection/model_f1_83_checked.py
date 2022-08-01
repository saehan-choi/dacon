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

import optuna
from optuna.trial import TrialState


# pathTrain = './anomaly_detection/dataset/train/'
# pathTest = './anomaly_detection/dataset/test/'

pathTrain = './anomaly_detection/dataset/train_with_affine_aug/'
pathTest = './anomaly_detection/dataset/test_original/'

class CFG:
    seed = 2
    lr = 1e-4
    # 밑에나온 epochs보다 1이작아야함 epoch가 1부터 시작해서
    
    epochs = 21
    batch_size = 17

set_seed(CFG.seed)


pathLabel = './anomaly_detection/dataset/'
device = torch.device('cuda')

train_png = sorted(glob(pathTrain+'/*.png'))
val_png = sorted(glob(pathTest+'/*.png'))

# train_y = pd.read_csv(pathLabel+"train_with_affine_aug.csv")
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
    def __init__(self, img_paths, labels, mode='train'):
        self.img_paths = img_paths
        self.labels = labels
        self.mode=mode

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = self.img_paths[idx]
        
        if self.mode=='train':

            augmentation = random.randint(0,3)
            if augmentation==1:
                img = img[::-1].copy()

            elif augmentation==2:
                img = img[:,::-1].copy()

            elif augmentation==3:
                transform_ = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.RandomErasing(scale=(0.01,0.05),ratio=(0.01,0.05), p=1),
                                    transforms.RandomPerspective(distortion_scale=0.1,p=1),
                                    transforms.ToPILImage(),
                ])
                img = transform_(img)

        if self.mode=='val':
            pass
        if self.mode=='test':
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



model = Network().to(device)

# [I 2022-04-23 06:07:23,887] Trial 10 finished with value: 0.8327445729532524 and parameters: {'optimizer': 'Adam', 'lr': 0.0001013575098983134, 'epochs': 20, 'batch_size': 17, 'seed': 2}. Best is trial 10 with value: 0.8327445729532524.

# 검증시 사용

optimizer = optim.Adam(model.parameters(),lr=CFG.lr)
criterion = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler() 

# Train
train_dataset = Custom_dataset(train_imgs, train_labels, mode='train')
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=CFG.batch_size)

# Test
val_dataset = Custom_dataset(val_imgs, val_labels, mode='val')
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=CFG.batch_size)

model.train()
for epoch in range(CFG.epochs):
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
    print(f'epoch : {epoch+1}/{CFG.epochs}    time : {TIME:.0f}s/{TIME*(CFG.epochs-epoch-1):.0f}s')
    print(f'TRAIN loss : {train_loss:.5f}  f1: {train_f1:.5f} lr: {CFG.lr} batch: {CFG.batch_size}')

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

    print(f'epoch    : {epoch+1}/{CFG.epochs}')
    print(f'VAL loss : {val_loss:.5f}   f1: {val_f1:.5f} lr: {CFG.lr} batch: {CFG.batch_size}')
                
    f_pred = []
    
    own_pred = []

    val_dataset_2 = Custom_dataset(np.array(val_imgs), np.array(["tmp"]*len(val_imgs)), mode='test')
    val_loader_2 = DataLoader(val_dataset_2, shuffle=False, batch_size=CFG.batch_size)

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
    submission.to_csv(pathLabel+f"submissions/model_test_{CFG.epochs}_{CFG.batch_size}_lr_{CFG.lr}_val_f1_{val_f1}.csv", index = False)

    gc.collect()

# epoch : 1/25    time : 366s/8789s
# TRAIN loss : 0.74857  f1: 0.28300 lr: 0.0001 batch: 17
# validation 진행중
# epoch    : 1/25
# VAL loss : 1.46996   f1: 0.36616 lr: 0.0001 batch: 17
# epoch : 2/25    time : 367s/8437s
# TRAIN loss : 0.27573  f1: 0.63571 lr: 0.0001 batch: 17
# validation 진행중
# epoch    : 2/25
# VAL loss : 0.84013   f1: 0.66169 lr: 0.0001 batch: 17
# epoch : 3/25    time : 347s/7641s
# TRAIN loss : 0.13180  f1: 0.82536 lr: 0.0001 batch: 17
# validation 진행중
# epoch    : 3/25
# VAL loss : 0.74321   f1: 0.72686 lr: 0.0001 batch: 17
# epoch : 4/25    time : 348s/7304s
# TRAIN loss : 0.06956  f1: 0.90692 lr: 0.0001 batch: 17
# validation 진행중
# epoch    : 4/25
# VAL loss : 0.69430   f1: 0.74295 lr: 0.0001 batch: 17
# epoch : 5/25    time : 348s/6954s
# TRAIN loss : 0.05623  f1: 0.93422 lr: 0.0001 batch: 17
# validation 진행중
# epoch    : 5/25
# VAL loss : 0.57569   f1: 0.77921 lr: 0.0001 batch: 17
# epoch : 6/25    time : 346s/6566s
# TRAIN loss : 0.03767  f1: 0.95664 lr: 0.0001 batch: 17
# validation 진행중
# epoch    : 6/25
# VAL loss : 0.70601   f1: 0.76768 lr: 0.0001 batch: 17
# epoch : 7/25    time : 346s/6233s
# TRAIN loss : 0.03413  f1: 0.96034 lr: 0.0001 batch: 17
# validation 진행중
# epoch    : 7/25
# VAL loss : 0.63323   f1: 0.80711 lr: 0.0001 batch: 17
# epoch : 8/25    time : 347s/5892s
# TRAIN loss : 0.01807  f1: 0.97982 lr: 0.0001 batch: 17
# validation 진행중
# epoch    : 8/25
# VAL loss : 0.62683   f1: 0.81250 lr: 0.0001 batch: 17
# epoch : 9/25    time : 348s/5565s
# TRAIN loss : 0.02288  f1: 0.97989 lr: 0.0001 batch: 17
# validation 진행중
# epoch    : 9/25
# VAL loss : 0.67574   f1: 0.81265 lr: 0.0001 batch: 17
# epoch : 10/25    time : 346s/5191s
# TRAIN loss : 0.02235  f1: 0.97951 lr: 0.0001 batch: 17
# validation 진행중
# epoch    : 10/25
# VAL loss : 0.76198   f1: 0.81219 lr: 0.0001 batch: 17
# epoch : 11/25    time : 347s/4854s
# TRAIN loss : 0.02801  f1: 0.97186 lr: 0.0001 batch: 17
# validation 진행중
# epoch    : 11/25
# VAL loss : 0.67656   f1: 0.82006 lr: 0.0001 batch: 17
# epoch : 12/25    time : 348s/4521s
# TRAIN loss : 0.02182  f1: 0.97839 lr: 0.0001 batch: 17
# validation 진행중
# epoch    : 12/25
# VAL loss : 0.68538   f1: 0.82612 lr: 0.0001 batch: 17
# epoch : 13/25    time : 347s/4164s
# TRAIN loss : 0.01728  f1: 0.98136 lr: 0.0001 batch: 17
# validation 진행중
# epoch    : 13/25
# VAL loss : 0.70443   f1: 0.82789 lr: 0.0001 batch: 17
# epoch : 14/25    time : 344s/3788s
# TRAIN loss : 0.01558  f1: 0.98863 lr: 0.0001 batch: 17
# validation 진행중
# epoch    : 14/25
# VAL loss : 0.69993   f1: 0.82279 lr: 0.0001 batch: 17
# epoch : 15/25    time : 345s/3445s
# TRAIN loss : 0.01868  f1: 0.98235 lr: 0.0001 batch: 17
# validation 진행중
# epoch    : 15/25
# VAL loss : 0.76874   f1: 0.81723 lr: 0.0001 batch: 17
# epoch : 16/25    time : 346s/3111s
# TRAIN loss : 0.01848  f1: 0.98416 lr: 0.0001 batch: 17
# validation 진행중
# epoch    : 16/25
# VAL loss : 0.78420   f1: 0.81359 lr: 0.0001 batch: 17
# epoch : 17/25    time : 347s/2775s
# TRAIN loss : 0.01564  f1: 0.98358 lr: 0.0001 batch: 17
# validation 진행중
# epoch    : 17/25
# VAL loss : 0.84089   f1: 0.81300 lr: 0.0001 batch: 17
# epoch : 18/25    time : 347s/2432s
# TRAIN loss : 0.00896  f1: 0.99418 lr: 0.0001 batch: 17
# validation 진행중
# epoch    : 18/25
# VAL loss : 0.74879   f1: 0.82434 lr: 0.0001 batch: 17
# epoch : 19/25    time : 346s/2077s
# TRAIN loss : 0.01811  f1: 0.98293 lr: 0.0001 batch: 17
# validation 진행중
# epoch    : 19/25
# VAL loss : 0.79922   f1: 0.80128 lr: 0.0001 batch: 17
# epoch : 20/25    time : 347s/1737s
# TRAIN loss : 0.01246  f1: 0.98598 lr: 0.0001 batch: 17
# validation 진행중
# epoch    : 20/25
# VAL loss : 0.84572   f1: 0.80837 lr: 0.0001 batch: 17
# epoch : 21/25    time : 345s/1381s
# TRAIN loss : 0.01381  f1: 0.98819 lr: 0.0001 batch: 17
# validation 진행중
# epoch    : 21/25
# VAL loss : 0.83212   f1: 0.80939 lr: 0.0001 batch: 17
# epoch : 22/25    time : 346s/1037s
# TRAIN loss : 0.01029  f1: 0.99131 lr: 0.0001 batch: 17
# validation 진행중
# epoch    : 22/25
# VAL loss : 0.79271   f1: 0.83342 lr: 0.0001 batch: 17
# epoch : 23/25    time : 347s/694s
# TRAIN loss : 0.01029  f1: 0.99260 lr: 0.0001 batch: 17
# validation 진행중
# epoch    : 23/25
# VAL loss : 0.81409   f1: 0.82996 lr: 0.0001 batch: 17
# epoch : 24/25    time : 345s/345s
# TRAIN loss : 0.01093  f1: 0.99101 lr: 0.0001 batch: 17
# validation 진행중
# epoch    : 24/25
# VAL loss : 0.87295   f1: 0.82104 lr: 0.0001 batch: 17
# epoch : 25/25    time : 345s/0s
# TRAIN loss : 0.01425  f1: 0.98452 lr: 0.0001 batch: 17
# validation 진행중
# epoch    : 25/25
# VAL loss : 0.87978   f1: 0.81760 lr: 0.0001 batch: 17