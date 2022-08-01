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
    epochs = 100
    batch_size = 16

    img_resize = (700, 700)

    transform = A.Compose([
                        A.ToGray(p=0.2),
                        A.GaussianBlur(blur_limit=(3, 7), p=0.05),
                        A.Sharpen(alpha=(0,0.45), p=0.2),
                        A.Posterize(p=1),
                        A.Cutout(num_holes=16, max_h_size=16, max_w_size=16, p=1),
                        A.RandomResizedCrop(img_resize[0], img_resize[1], scale=(0.9, 1.0), ratio=(0.9, 1)),
                        # !!!!!!!!!!!!!!!!!!!!!!!!!!! 각도를 일정각도만 지정해서 해보기 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
    img = cv2.resize(img, (CFG.img_resize[0], CFG.img_resize[1]))
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
        submission.to_csv(pathLabel+f"submissions/700x700_album_{epoch}_{batch_size}_lr_{lr}_val_f1_{val_f1}.csv", index = False)
        gc.collect()


# epoch : 1/100    time : 564s/55857s
# TRAIN loss : 0.78927  f1: 0.30499 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 1/100
# VAL loss : 1.26720   f1: 0.42624 lr: 0.0003 batch: 16
# epoch : 2/100    time : 529s/51804s
# TRAIN loss : 0.33816  f1: 0.57898 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 2/100
# VAL loss : 0.90093   f1: 0.62153 lr: 0.0003 batch: 16
# epoch : 3/100    time : 561s/54432s
# TRAIN loss : 0.21618  f1: 0.72731 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 3/100
# VAL loss : 0.79068   f1: 0.68679 lr: 0.0003 batch: 16
# epoch : 4/100    time : 555s/53248s
# TRAIN loss : 0.15752  f1: 0.79988 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 4/100
# VAL loss : 0.65697   f1: 0.75663 lr: 0.0003 batch: 16
# epoch : 5/100    time : 513s/48726s
# TRAIN loss : 0.13250  f1: 0.84442 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 5/100
# VAL loss : 0.75782   f1: 0.74176 lr: 0.0003 batch: 16
# epoch : 6/100    time : 516s/48470s
# TRAIN loss : 0.10160  f1: 0.87795 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 6/100
# VAL loss : 0.81284   f1: 0.74962 lr: 0.0003 batch: 16
# epoch : 7/100    time : 518s/48142s
# TRAIN loss : 0.09836  f1: 0.88012 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 7/100
# VAL loss : 0.70829   f1: 0.77070 lr: 0.0003 batch: 16
# epoch : 8/100    time : 511s/46995s
# TRAIN loss : 0.08588  f1: 0.89949 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 8/100
# VAL loss : 0.86307   f1: 0.75845 lr: 0.0003 batch: 16
# epoch : 9/100    time : 509s/46321s
# TRAIN loss : 0.07852  f1: 0.91107 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 9/100
# VAL loss : 0.83028   f1: 0.77182 lr: 0.0003 batch: 16
# epoch : 10/100    time : 509s/45840s
# TRAIN loss : 0.06415  f1: 0.92718 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 10/100
# VAL loss : 0.91276   f1: 0.75253 lr: 0.0003 batch: 16
# epoch : 11/100    time : 510s/45357s
# TRAIN loss : 0.06343  f1: 0.93110 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 11/100
# VAL loss : 0.94514   f1: 0.74928 lr: 0.0003 batch: 16
# epoch : 12/100    time : 511s/44982s
# TRAIN loss : 0.05112  f1: 0.95180 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 12/100
# VAL loss : 0.91028   f1: 0.75653 lr: 0.0003 batch: 16
# epoch : 13/100    time : 513s/44598s
# TRAIN loss : 0.05791  f1: 0.94274 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 13/100
# VAL loss : 0.84802   f1: 0.76436 lr: 0.0003 batch: 16
# epoch : 14/100    time : 514s/44210s
# TRAIN loss : 0.05945  f1: 0.93986 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 14/100
# VAL loss : 0.78149   f1: 0.78202 lr: 0.0003 batch: 16
# epoch : 15/100    time : 512s/43522s
# TRAIN loss : 0.04254  f1: 0.95691 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 15/100
# VAL loss : 0.89366   f1: 0.77675 lr: 0.0003 batch: 16
# epoch : 16/100    time : 510s/42857s
# TRAIN loss : 0.05812  f1: 0.95032 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 16/100
# VAL loss : 0.78896   f1: 0.76886 lr: 0.0003 batch: 16
# epoch : 17/100    time : 511s/42450s
# TRAIN loss : 0.04495  f1: 0.95310 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 17/100
# VAL loss : 0.82484   f1: 0.79480 lr: 0.0003 batch: 16
# epoch : 18/100    time : 510s/41793s
# TRAIN loss : 0.03300  f1: 0.96650 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 18/100
# VAL loss : 0.88414   f1: 0.77982 lr: 0.0003 batch: 16
# epoch : 19/100    time : 512s/41482s
# TRAIN loss : 0.04474  f1: 0.95990 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 19/100
# VAL loss : 0.80502   f1: 0.80095 lr: 0.0003 batch: 16
# epoch : 20/100    time : 513s/41077s
# TRAIN loss : 0.03857  f1: 0.95993 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 20/100
# VAL loss : 0.86498   f1: 0.78410 lr: 0.0003 batch: 16
# epoch : 21/100    time : 510s/40258s
# TRAIN loss : 0.04171  f1: 0.96219 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 21/100
# VAL loss : 0.69591   f1: 0.81574 lr: 0.0003 batch: 16
# epoch : 22/100    time : 510s/39814s
# TRAIN loss : 0.03643  f1: 0.96245 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 22/100
# VAL loss : 0.94853   f1: 0.78532 lr: 0.0003 batch: 16
# epoch : 23/100    time : 510s/39232s
# TRAIN loss : 0.03255  f1: 0.96977 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 23/100
# VAL loss : 0.93039   f1: 0.76814 lr: 0.0003 batch: 16
# epoch : 24/100    time : 514s/39076s
# TRAIN loss : 0.03222  f1: 0.96668 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 24/100
# VAL loss : 0.98635   f1: 0.76944 lr: 0.0003 batch: 16
# epoch : 25/100    time : 511s/38309s
# TRAIN loss : 0.03116  f1: 0.96891 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 25/100
# VAL loss : 0.93813   f1: 0.80218 lr: 0.0003 batch: 16
# epoch : 26/100    time : 514s/38061s
# TRAIN loss : 0.03238  f1: 0.96863 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 26/100
# VAL loss : 0.95797   f1: 0.77409 lr: 0.0003 batch: 16
# epoch : 27/100    time : 510s/37257s
# TRAIN loss : 0.02322  f1: 0.97868 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 27/100
# VAL loss : 1.04078   f1: 0.76977 lr: 0.0003 batch: 16
# epoch : 28/100    time : 509s/36628s
# TRAIN loss : 0.03695  f1: 0.96723 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 28/100
# VAL loss : 0.90183   f1: 0.78217 lr: 0.0003 batch: 16
# epoch : 29/100    time : 513s/36448s
# TRAIN loss : 0.02120  f1: 0.98200 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 29/100
# VAL loss : 0.99116   f1: 0.78977 lr: 0.0003 batch: 16
# epoch : 30/100    time : 511s/35757s
# TRAIN loss : 0.03472  f1: 0.96520 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 30/100
# VAL loss : 1.01821   f1: 0.77050 lr: 0.0003 batch: 16
# epoch : 31/100    time : 511s/35254s
# TRAIN loss : 0.02270  f1: 0.97763 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 31/100
# VAL loss : 1.18349   f1: 0.73445 lr: 0.0003 batch: 16
# epoch : 32/100    time : 510s/34680s
# TRAIN loss : 0.02374  f1: 0.96873 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 32/100
# VAL loss : 1.08513   f1: 0.77635 lr: 0.0003 batch: 16
# epoch : 33/100    time : 510s/34169s
# TRAIN loss : 0.03084  f1: 0.97219 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 33/100
# VAL loss : 0.97642   f1: 0.76982 lr: 0.0003 batch: 16
# epoch : 34/100    time : 509s/33599s
# TRAIN loss : 0.03410  f1: 0.96663 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 34/100
# VAL loss : 0.82915   f1: 0.79807 lr: 0.0003 batch: 16
# epoch : 35/100    time : 509s/33101s
# TRAIN loss : 0.02183  f1: 0.97766 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 35/100
# VAL loss : 0.93884   f1: 0.78211 lr: 0.0003 batch: 16
# epoch : 36/100    time : 509s/32579s
# TRAIN loss : 0.01723  f1: 0.98453 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 36/100
# VAL loss : 0.82842   f1: 0.78835 lr: 0.0003 batch: 16
# epoch : 37/100    time : 510s/32115s
# TRAIN loss : 0.02814  f1: 0.97173 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 37/100
# VAL loss : 0.86891   f1: 0.80108 lr: 0.0003 batch: 16
# epoch : 38/100    time : 510s/31623s
# TRAIN loss : 0.02539  f1: 0.97764 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 38/100
# VAL loss : 0.93414   f1: 0.79448 lr: 0.0003 batch: 16
# epoch : 39/100    time : 512s/31221s
# TRAIN loss : 0.01554  f1: 0.98222 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 39/100
# VAL loss : 0.99849   f1: 0.78672 lr: 0.0003 batch: 16
# epoch : 40/100    time : 510s/30623s
# TRAIN loss : 0.02304  f1: 0.97741 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 40/100
# VAL loss : 1.15376   f1: 0.77584 lr: 0.0003 batch: 16
# epoch : 41/100    time : 510s/30071s
# TRAIN loss : 0.01537  f1: 0.98602 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 41/100
# VAL loss : 1.26336   f1: 0.78541 lr: 0.0003 batch: 16
# epoch : 42/100    time : 511s/29622s
# TRAIN loss : 0.01841  f1: 0.98093 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 42/100
# VAL loss : 1.10931   f1: 0.78070 lr: 0.0003 batch: 16
# epoch : 43/100    time : 512s/29211s
# TRAIN loss : 0.02248  f1: 0.97601 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 43/100
# VAL loss : 1.06473   f1: 0.80652 lr: 0.0003 batch: 16
# epoch : 44/100    time : 509s/28506s
# TRAIN loss : 0.01318  f1: 0.98953 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 44/100
# VAL loss : 1.17826   f1: 0.79139 lr: 0.0003 batch: 16
# epoch : 45/100    time : 510s/28071s
# TRAIN loss : 0.02547  f1: 0.97689 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 45/100
# VAL loss : 1.14678   f1: 0.76537 lr: 0.0003 batch: 16
# epoch : 46/100    time : 511s/27591s
# TRAIN loss : 0.02170  f1: 0.97826 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 46/100
# VAL loss : 1.17137   f1: 0.80118 lr: 0.0003 batch: 16
# epoch : 47/100    time : 515s/27301s
# TRAIN loss : 0.01933  f1: 0.98406 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 47/100
# VAL loss : 1.12632   f1: 0.76738 lr: 0.0003 batch: 16
# epoch : 48/100    time : 510s/26546s
# TRAIN loss : 0.01869  f1: 0.98376 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 48/100
# VAL loss : 1.14580   f1: 0.78907 lr: 0.0003 batch: 16
# epoch : 49/100    time : 510s/26001s
# TRAIN loss : 0.01678  f1: 0.98668 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 49/100
# VAL loss : 1.02885   f1: 0.81354 lr: 0.0003 batch: 16
# epoch : 50/100    time : 513s/25627s
# TRAIN loss : 0.02128  f1: 0.98020 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 50/100
# VAL loss : 0.90246   f1: 0.80253 lr: 0.0003 batch: 16
# epoch : 51/100    time : 510s/25006s
# TRAIN loss : 0.01877  f1: 0.98301 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 51/100
# VAL loss : 1.07149   f1: 0.77903 lr: 0.0003 batch: 16
# epoch : 52/100    time : 509s/24454s
# TRAIN loss : 0.01086  f1: 0.99085 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 52/100
# VAL loss : 1.09618   f1: 0.78303 lr: 0.0003 batch: 16
# epoch : 53/100    time : 511s/24000s
# TRAIN loss : 0.02345  f1: 0.97934 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 53/100
# VAL loss : 1.08979   f1: 0.76328 lr: 0.0003 batch: 16
# epoch : 54/100    time : 513s/23577s
# TRAIN loss : 0.01762  f1: 0.98235 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 54/100
# VAL loss : 1.02093   f1: 0.77849 lr: 0.0003 batch: 16
# epoch : 55/100    time : 509s/22910s
# TRAIN loss : 0.01890  f1: 0.97943 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 55/100
# VAL loss : 1.11859   f1: 0.77219 lr: 0.0003 batch: 16