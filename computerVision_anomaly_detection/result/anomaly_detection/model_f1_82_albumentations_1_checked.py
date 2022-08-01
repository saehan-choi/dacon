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

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class CFG:
    seed = 42
    lr = 3e-4
    epochs = 55
    batch_size = 32

    img_resize = (512, 512)
    
    transform = A.Compose([
                        A.ToGray(p=0.2),
                        A.GaussianBlur(blur_limit=(3, 7), p=0.05),
                        A.Sharpen(alpha=(0,0.45), p=0.2),
                        A.Posterize(p=0.2),
                        A.Cutout(num_holes=16, max_h_size=16, max_w_size=16, p=0.5),
                        A.RandomResizedCrop(img_resize[0], img_resize[1], scale=(0.9, 1.0), ratio=(0.9, 1)),
                        A.Normalize(),
                        ])


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
                transformed = CFG.transform
                img = transformed(image=img)["image"]
                
        if self.mode=='val':
            pass
        


        img = ToTensorV2()(image=img)["image"]
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

    optimizer = optim.Adam(model.parameters(),lr=lr)


    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler() 

    # Train
    train_dataset = Custom_dataset(train_imgs, train_labels, mode='train')
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

    # Test
    val_dataset = Custom_dataset(val_imgs, val_labels, mode='val')
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
                val_pred += pred.argmax(1).detach().cpu().numpy().tolist()
                val_y += y.detach().cpu().numpy().tolist()


        val_f1 = score_function(val_y, val_pred)

        print(f'epoch    : {epoch+1}/{epochs}')
        print(f'VAL loss : {val_loss:.5f}   f1: {val_f1:.5f} lr: {lr} batch: {batch_size}')
            
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
        submission.to_csv(pathLabel+f"submissions/albumentations_aug_05-06-_{epoch}_{batch_size}_lr_{lr}_val_f1_{val_f1}.csv", index = False)
        gc.collect()



# epoch : 1/200    time : 260s/51773s
# TRAIN loss : 0.77986  f1: 0.31711 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 1/200
# VAL loss : 1.08077   f1: 0.47424 lr: 0.0003 batch: 32
# epoch : 2/200    time : 262s/51797s
# TRAIN loss : 0.31018  f1: 0.61949 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 2/200
# VAL loss : 0.69671   f1: 0.67194 lr: 0.0003 batch: 32
# epoch : 3/200    time : 262s/51653s
# TRAIN loss : 0.18918  f1: 0.77976 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 3/200
# VAL loss : 0.69663   f1: 0.71540 lr: 0.0003 batch: 32
# epoch : 4/200    time : 262s/51264s
# TRAIN loss : 0.13951  f1: 0.84572 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 4/200
# VAL loss : 0.72656   f1: 0.74989 lr: 0.0003 batch: 32
# epoch : 5/200    time : 262s/51059s
# TRAIN loss : 0.11237  f1: 0.86719 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 5/200
# VAL loss : 0.58056   f1: 0.79573 lr: 0.0003 batch: 32
# epoch : 6/200    time : 256s/49709s
# TRAIN loss : 0.09080  f1: 0.91030 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 6/200
# VAL loss : 0.65581   f1: 0.78557 lr: 0.0003 batch: 32
# epoch : 7/200    time : 254s/49071s
# TRAIN loss : 0.07156  f1: 0.92856 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 7/200
# VAL loss : 0.70841   f1: 0.77746 lr: 0.0003 batch: 32
# epoch : 8/200    time : 255s/48895s
# TRAIN loss : 0.07868  f1: 0.92120 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 8/200
# VAL loss : 0.66522   f1: 0.79805 lr: 0.0003 batch: 32
# epoch : 9/200    time : 254s/48423s
# TRAIN loss : 0.05956  f1: 0.93582 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 9/200
# VAL loss : 0.70302   f1: 0.79714 lr: 0.0003 batch: 32
# epoch : 10/200    time : 253s/48113s
# TRAIN loss : 0.04912  f1: 0.94747 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 10/200
# VAL loss : 0.71603   f1: 0.77352 lr: 0.0003 batch: 32
# epoch : 11/200    time : 254s/48074s
# TRAIN loss : 0.05352  f1: 0.94838 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 11/200
# VAL loss : 0.86838   f1: 0.79181 lr: 0.0003 batch: 32
# epoch : 12/200    time : 258s/48517s
# TRAIN loss : 0.05479  f1: 0.94581 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 12/200
# VAL loss : 0.85512   f1: 0.77026 lr: 0.0003 batch: 32
# epoch : 13/200    time : 251s/46936s
# TRAIN loss : 0.04930  f1: 0.94661 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 13/200
# VAL loss : 0.75486   f1: 0.78469 lr: 0.0003 batch: 32
# epoch : 14/200    time : 257s/47807s
# TRAIN loss : 0.05048  f1: 0.94141 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 14/200
# VAL loss : 0.83927   f1: 0.78724 lr: 0.0003 batch: 32
# epoch : 15/200    time : 252s/46596s
# TRAIN loss : 0.04206  f1: 0.96044 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 15/200
# VAL loss : 0.83056   f1: 0.77425 lr: 0.0003 batch: 32
# epoch : 16/200    time : 251s/46146s
# TRAIN loss : 0.03430  f1: 0.96831 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 16/200
# VAL loss : 0.85952   f1: 0.78730 lr: 0.0003 batch: 32
# epoch : 17/200    time : 253s/46302s
# TRAIN loss : 0.03472  f1: 0.96746 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 17/200
# VAL loss : 0.91388   f1: 0.76641 lr: 0.0003 batch: 32
# epoch : 18/200    time : 256s/46616s
# TRAIN loss : 0.04798  f1: 0.95495 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 18/200
# VAL loss : 0.86225   f1: 0.78125 lr: 0.0003 batch: 32
# epoch : 19/200    time : 251s/45449s
# TRAIN loss : 0.03261  f1: 0.96698 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 19/200
# VAL loss : 0.85591   f1: 0.78456 lr: 0.0003 batch: 32
# epoch : 20/200    time : 251s/45095s
# TRAIN loss : 0.03175  f1: 0.96895 lr: 0.0003 batch: 32
# validation 진행중
5859# epoch    : 20/200
# VAL loss : 0.98179   f1: 0.75389 lr: 0.0003 batch: 32
# epoch : 21/200    time : 250s/44765s
# TRAIN loss : 0.02766  f1: 0.97801 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 21/200
# VAL loss : 0.77269   f1: 0.79277 lr: 0.0003 batch: 32
# epoch : 22/200    time : 251s/44620s
# TRAIN loss : 0.02204  f1: 0.97526 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 22/200
# VAL loss : 0.82371   f1: 0.80247 lr: 0.0003 batch: 32
# epoch : 23/200    time : 256s/45226s
# TRAIN loss : 0.03124  f1: 0.97311 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 23/200
# VAL loss : 0.82121   f1: 0.79929 lr: 0.0003 batch: 32
# epoch : 24/200    time : 257s/45260s
# TRAIN loss : 0.03018  f1: 0.97425 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 24/200
# VAL loss : 0.84259   f1: 0.77165 lr: 0.0003 batch: 32
# epoch : 25/200    time : 258s/45139s
# TRAIN loss : 0.02927  f1: 0.97085 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 25/200
# VAL loss : 1.05201   f1: 0.76074 lr: 0.0003 batch: 32
# epoch : 26/200    time : 252s/43765s
# TRAIN loss : 0.03595  f1: 0.96344 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 26/200
# VAL loss : 0.74095   f1: 0.81234 lr: 0.0003 batch: 32
# epoch : 27/200    time : 252s/43601s
# TRAIN loss : 0.02234  f1: 0.97761 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 27/200
# VAL loss : 0.88629   f1: 0.79910 lr: 0.0003 batch: 32
# epoch : 28/200    time : 254s/43700s
# TRAIN loss : 0.02582  f1: 0.97840 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 28/200
# VAL loss : 0.91526   f1: 0.78206 lr: 0.0003 batch: 32
# epoch : 29/200    time : 254s/43498s
# TRAIN loss : 0.02445  f1: 0.97994 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 29/200
# VAL loss : 0.87707   f1: 0.79534 lr: 0.0003 batch: 32
# epoch : 30/200    time : 254s/43189s
# TRAIN loss : 0.02050  f1: 0.97954 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 30/200
# VAL loss : 0.77290   f1: 0.80800 lr: 0.0003 batch: 32
# epoch : 31/200    time : 255s/43046s
# TRAIN loss : 0.02417  f1: 0.97925 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 31/200
# VAL loss : 0.77488   f1: 0.80805 lr: 0.0003 batch: 32
# epoch : 32/200    time : 253s/42504s
# TRAIN loss : 0.02085  f1: 0.97767 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 32/200
# VAL loss : 0.78270   f1: 0.81543 lr: 0.0003 batch: 32
# epoch : 33/200    time : 255s/42509s
# TRAIN loss : 0.02171  f1: 0.98036 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 33/200
# VAL loss : 0.86489   f1: 0.80124 lr: 0.0003 batch: 32
# epoch : 34/200    time : 258s/42807s
# TRAIN loss : 0.01931  f1: 0.98161 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 34/200
# VAL loss : 0.84289   f1: 0.81838 lr: 0.0003 batch: 32
# epoch : 35/200    time : 258s/42602s
# TRAIN loss : 0.02654  f1: 0.98117 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 35/200
# VAL loss : 0.85316   f1: 0.78432 lr: 0.0003 batch: 32
# epoch : 36/200    time : 254s/41638s
# TRAIN loss : 0.02527  f1: 0.97564 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 36/200
# VAL loss : 0.84606   f1: 0.79524 lr: 0.0003 batch: 32
# epoch : 37/200    time : 259s/42257s
# TRAIN loss : 0.01599  f1: 0.98362 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 37/200
# VAL loss : 0.79193   f1: 0.80165 lr: 0.0003 batch: 32
# epoch : 38/200    time : 254s/41191s
# TRAIN loss : 0.01883  f1: 0.98007 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 38/200
# VAL loss : 0.80616   f1: 0.81496 lr: 0.0003 batch: 32
# epoch : 39/200    time : 260s/41843s
# TRAIN loss : 0.01503  f1: 0.98857 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 39/200
# VAL loss : 0.89317   f1: 0.80456 lr: 0.0003 batch: 32
# epoch : 40/200    time : 253s/40529s
# TRAIN loss : 0.02439  f1: 0.97894 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 40/200
# VAL loss : 0.96637   f1: 0.78640 lr: 0.0003 batch: 32
# epoch : 41/200    time : 257s/40941s
# TRAIN loss : 0.02512  f1: 0.97569 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 41/200
# VAL loss : 0.90963   f1: 0.81127 lr: 0.0003 batch: 32
# epoch : 42/200    time : 255s/40326s
# TRAIN loss : 0.01446  f1: 0.98735 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 42/200
# VAL loss : 0.90421   f1: 0.80392 lr: 0.0003 batch: 32
# epoch : 43/200    time : 257s/40346s
# TRAIN loss : 0.02607  f1: 0.97418 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 43/200
# VAL loss : 0.90923   f1: 0.81336 lr: 0.0003 batch: 32
# epoch : 44/200    time : 252s/39267s
# TRAIN loss : 0.01516  f1: 0.98292 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 44/200
# VAL loss : 0.90889   f1: 0.81790 lr: 0.0003 batch: 32
# epoch : 45/200    time : 252s/38998s
# TRAIN loss : 0.01531  f1: 0.98717 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 45/200
# VAL loss : 0.93815   f1: 0.81004 lr: 0.0003 batch: 32
# epoch : 46/200    time : 252s/38735s
# TRAIN loss : 0.01868  f1: 0.97976 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 46/200
# VAL loss : 0.89596   f1: 0.79370 lr: 0.0003 batch: 32
# epoch : 47/200    time : 252s/38630s
# TRAIN loss : 0.01953  f1: 0.98275 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 47/200
# VAL loss : 0.81931   f1: 0.81845 lr: 0.0003 batch: 32
# epoch : 48/200    time : 252s/38345s
# TRAIN loss : 0.02691  f1: 0.97863 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 48/200
# VAL loss : 0.82944   f1: 0.80267 lr: 0.0003 batch: 32
# epoch : 49/200    time : 251s/37952s
# TRAIN loss : 0.01079  f1: 0.98973 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 49/200
# VAL loss : 0.89994   f1: 0.79300 lr: 0.0003 batch: 32
# epoch : 50/200    time : 251s/37628s
# TRAIN loss : 0.01430  f1: 0.98575 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 50/200
# VAL loss : 0.82286   f1: 0.82071 lr: 0.0003 batch: 32
# epoch : 51/200    time : 252s/37484s
# TRAIN loss : 0.01506  f1: 0.98627 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 51/200
# VAL loss : 0.88061   f1: 0.82063 lr: 0.0003 batch: 32
# epoch : 52/200    time : 251s/37113s
# TRAIN loss : 0.01442  f1: 0.98898 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 52/200
# VAL loss : 0.85983   f1: 0.80463 lr: 0.0003 batch: 32
# epoch : 53/200    time : 252s/37075s
# TRAIN loss : 0.01616  f1: 0.98163 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 53/200
# VAL loss : 1.01208   f1: 0.79829 lr: 0.0003 batch: 32
# epoch : 54/200    time : 251s/36621s
# TRAIN loss : 0.01304  f1: 0.98717 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 54/200
# VAL loss : 0.88250   f1: 0.82579 lr: 0.0003 batch: 32
# epoch : 55/200    time : 253s/36746s
# TRAIN loss : 0.01879  f1: 0.98472 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 55/200
# VAL loss : 0.88522   f1: 0.81587 lr: 0.0003 batch: 32
# epoch : 56/200    time : 257s/36982s
# TRAIN loss : 0.01362  f1: 0.98831 lr: 0.0003 batch: 32
# validation 진행중
# epoch    : 56/200
# VAL loss : 0.80269   f1: 0.82527 lr: 0.0003 batch: 32
# epoch : 57/200    time : 257s/36798s