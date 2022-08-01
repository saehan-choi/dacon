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
from utils import *
import time

# from utils import set_seed, transform_album
# from augmentation import transform

# 파이토치에서는 amp를 쓴다는 것은 학습할때
# torch.cuda.amp.autocast 와 torch.cuda.amp.GradScaler를 같이 쓰는 것을 의미한다.
# 여튼 두개를 쓰면, 학습할때 성능은 유지(신경망의 수렴이 잘되고)되면서, 
# GPU 메모리 소모도 줄고, GPU 연산속도도 증가한다. 
# 얼마나 빨라지냐는 GPU의 architecure마다 다르다. 
# 가령 Volta, Turing, Ampere는 2~3배. 최신의 Kepler, Maxwell, Pascal은 1~2배정도 빨라진다.

# 14 epoch까지 돌리기

pathTrain = './anomaly_detection/dataset/train_with_affine_aug/'
pathTest = './anomaly_detection/dataset/test_original/'
# pathTrain = './anomaly_detection/dataset/train/'
# pathTest = './anomaly_detection/dataset/test/'

pathLabel = './anomaly_detection/dataset/'

seed = 24
set_seed(seed)

class CFG:
    batch_size = 16
    lr = 3e-4
    # 밑에나온 epochs보다 1이작아야함 epoch가 1부터 시작해서
    epochs = 30

device = torch.device('cuda')



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
                                    transforms.Pad(10, fill=0),
                                    transforms.RandomCrop(512),
                                    transforms.ToPILImage()
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

if __name__ == "__main__":

    # Train
    train_dataset = Custom_dataset(train_imgs, train_labels, mode='train')
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=CFG.batch_size)

    # Test
    val_dataset = Custom_dataset(val_imgs, val_labels, mode='val')
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=CFG.batch_size)


    def score_function(real, pred):
        score = f1_score(real, pred, average="macro")
        return score

    model = Network().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler() 


    model.train()
    best=0
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
        submission.to_csv(pathLabel+f"submissions/OPTUNA{epoch}_{CFG.batch_size}_lr_{CFG.lr}_val_f1_{val_f1}.csv", index = False)
        gc.collect()

# epoch    : 1/25
# VAL loss : 0.96185   f1: 0.57956 lr: 0.0003 batch: 16
# epoch : 2/25    time : 305s/7017s
# TRAIN loss : 0.19425  f1: 0.74805 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 2/25
# VAL loss : 0.91502   f1: 0.67695 lr: 0.0003 batch: 16
# epoch : 3/25    time : 316s/6961s
# TRAIN loss : 0.11187  f1: 0.85212 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 3/25
# VAL loss : 0.86104   f1: 0.72034 lr: 0.0003 batch: 16
# epoch : 4/25    time : 360s/7560s
# TRAIN loss : 0.08311  f1: 0.89503 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 4/25
# VAL loss : 0.68380   f1: 0.78472 lr: 0.0003 batch: 16
# epoch : 5/25    time : 418s/8366s
# TRAIN loss : 0.06162  f1: 0.92990 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 5/25
# VAL loss : 0.93089   f1: 0.73778 lr: 0.0003 batch: 16
# epoch : 6/25    time : 413s/7852s
# TRAIN loss : 0.06305  f1: 0.92951 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 6/25
# VAL loss : 0.72572   f1: 0.79752 lr: 0.0003 batch: 16
# epoch : 7/25    time : 419s/7542s
# TRAIN loss : 0.04586  f1: 0.95023 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 7/25
# VAL loss : 0.77657   f1: 0.77393 lr: 0.0003 batch: 16
# epoch : 8/25    time : 416s/7077s
# TRAIN loss : 0.05503  f1: 0.94702 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 8/25
# VAL loss : 0.87910   f1: 0.78050 lr: 0.0003 batch: 16
# epoch : 9/25    time : 411s/6573s
# TRAIN loss : 0.03358  f1: 0.96836 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 9/25
# VAL loss : 0.83835   f1: 0.78902 lr: 0.0003 batch: 16
# epoch : 10/25    time : 404s/6054s
# TRAIN loss : 0.04696  f1: 0.95645 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 10/25
# VAL loss : 0.98780   f1: 0.75059 lr: 0.0003 batch: 16
# epoch : 11/25    time : 409s/5726s
# TRAIN loss : 0.03456  f1: 0.96238 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 11/25
# VAL loss : 0.96102   f1: 0.80304 lr: 0.0003 batch: 16
# epoch : 12/25    time : 408s/5302s
# TRAIN loss : 0.03382  f1: 0.96318 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 12/25
# VAL loss : 0.88356   f1: 0.79803 lr: 0.0003 batch: 16
# epoch : 13/25    time : 410s/4924s
# TRAIN loss : 0.02930  f1: 0.97036 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 13/25
# VAL loss : 0.98824   f1: 0.75585 lr: 0.0003 batch: 16
# epoch : 14/25    time : 410s/4513s
# TRAIN loss : 0.03553  f1: 0.96548 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 14/25
# VAL loss : 0.85689   f1: 0.79833 lr: 0.0003 batch: 16
# epoch : 15/25    time : 413s/4129s
# TRAIN loss : 0.03406  f1: 0.96610 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 15/25
# VAL loss : 1.00547   f1: 0.78614 lr: 0.0003 batch: 16
# epoch : 16/25    time : 407s/3667s
# TRAIN loss : 0.01916  f1: 0.98153 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 16/25
# VAL loss : 0.97392   f1: 0.80048 lr: 0.0003 batch: 16
# epoch : 17/25    time : 405s/3238s
# TRAIN loss : 0.02197  f1: 0.97947 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 17/25
# VAL loss : 0.90204   f1: 0.80016 lr: 0.0003 batch: 16
# epoch : 18/25    time : 406s/2839s
# TRAIN loss : 0.03195  f1: 0.97236 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 18/25
# VAL loss : 0.85345   f1: 0.81107 lr: 0.0003 batch: 16
# epoch : 19/25    time : 405s/2429s
# TRAIN loss : 0.02697  f1: 0.97445 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 19/25
# VAL loss : 0.91981   f1: 0.81437 lr: 0.0003 batch: 16
# epoch : 20/25    time : 406s/2029s
# TRAIN loss : 0.02090  f1: 0.98089 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 20/25
# VAL loss : 0.87387   f1: 0.78941 lr: 0.0003 batch: 16
# epoch : 21/25    time : 406s/1624s
# TRAIN loss : 0.01644  f1: 0.98596 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 21/25
# VAL loss : 0.95437   f1: 0.80751 lr: 0.0003 batch: 16
# epoch : 22/25    time : 406s/1219s
# TRAIN loss : 0.03455  f1: 0.96942 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 22/25
# VAL loss : 1.06130   f1: 0.79395 lr: 0.0003 batch: 16
# epoch : 23/25    time : 409s/818s
# TRAIN loss : 0.02576  f1: 0.97885 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 23/25
# VAL loss : 0.86844   f1: 0.82543 lr: 0.0003 batch: 16
# epoch : 24/25    time : 408s/408s
# TRAIN loss : 0.01725  f1: 0.98136 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 24/25
# VAL loss : 0.96383   f1: 0.79402 lr: 0.0003 batch: 16
# epoch : 25/25    time : 409s/0s
# TRAIN loss : 0.01172  f1: 0.98985 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 25/25
# VAL loss : 0.84565   f1: 0.82315 lr: 0.0003 batch: 16