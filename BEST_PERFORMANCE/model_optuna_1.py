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


# batch_size = 32
# lr = 3e-4
# epochs = 20


########## 실제 자료 이용시에는 train,test ORIGINAL을 이용하세요.###########
########## 테스트 자료 이용시에는 train,test 이용하세요.###########
# pathTrain = './anomaly_detection/dataset/train/'
# pathTrain = './anomaly_detection/dataset/train_original/'
# pathTest = './anomaly_detection/dataset/test/'
# pathTest = './anomaly_detection/dataset/test_original/'

pathTrain = './anomaly_detection/dataset/train_with_affine_aug/'

# # # 변경됨
# pathTrain = './anomaly_detection/dataset/train_original/'
pathTest = './anomaly_detection/dataset/test_original/'
# 단순히 lr만 3e-4로 바꿨을뿐인데 0.61 -> 0.71로 성능향상
# -> 거기에 image augmentation 적용 -> 0.83 
# -> ensemble 적용 0.835


pathLabel = './anomaly_detection/dataset/'
device = torch.device('cuda')

train_png = sorted(glob(pathTrain+'/*.png'))
val_png = sorted(glob(pathTest+'/*.png'))

# train_y = pd.read_csv(pathLabel+"train_with_affine_aug.csv")
train_y = pd.read_csv(pathLabel+"train_with_affine_aug.csv")
train_labels = train_y["label"]

val_y = pd.read_csv(pathLabel+"baseline_test.csv")
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
                # 수평변환

            elif augmentation==2:
                img = img[:,::-1].copy()
                # 수직변환

            elif augmentation==3:
                transform_ = transforms.Compose([
                                    transforms.ToTensor(),
                                    # transforms.RandomResizedCrop(512,scale=(0.85,1),ratio=(1,1.2)),
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


def objective(trial):
    
    

    model = Network().to(device)

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    epochs = trial.suggest_int('epochs',5,20)
    batch_size = trial.suggest_int('batch_size',8,32)
    seed = trial.suggest_int('seed',1,100)
    # [I 2022-04-23 06:07:23,887] Trial 10 finished with value: 0.8327445729532524 and parameters: {'optimizer': 'Adam', 'lr': 0.0001013575098983134, 'epochs': 20, 'batch_size': 17, 'seed': 2}. Best is trial 10 with value: 0.8327445729532524.
    
    # 검증시 사용
    # lr = 0.0006833364758030707
    # epochs = 8
    # optimizer = optim.RMSprop(model.parameters(),lr=lr)
    # batch_size = 32
    # seed = 69
    
    set_seed(seed)
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
        performance = open('./anomaly_detection/performance_record.txt','a')
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

        print(f'epoch    : {epoch+1}/{epochs}')
        print(f'VAL loss : {val_loss:.5f}   f1: {val_f1:.5f} lr: {lr} batch: {batch_size}')
        performance.write(f'epochs: {epoch} val loss: {val_loss}  val f1:{val_f1:.5f}  lr: {lr} batch: {batch_size}\n')
        performance.close()
        
        trial.report(val_f1, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
        # f_pred = []
        
        # own_pred = []

        # val_dataset_2 = Custom_dataset(np.array(val_imgs), np.array(["tmp"]*len(val_imgs)), mode='test')
        # val_loader_2 = DataLoader(val_dataset_2, shuffle=False, batch_size=batch_size)

        # with torch.no_grad():
        #     for batch in (val_loader_2):
        #         x = torch.tensor(batch[0], dtype = torch.float32, device = device)
        #         with torch.cuda.amp.autocast():
        #             pred = model(x)

        #         for i in range(len(batch[0])):
        #             own_pred.extend([pred[i].detach().cpu().numpy().tolist()])
        #         f_pred.extend(pred.argmax(1).detach().cpu().numpy().tolist())

        # label_decoder = {val:key for key, val in label_unique.items()}
        # f_result = [label_decoder[result] for result in f_pred]

        # add_arr = []
        # for j in range(len(own_pred)):
        #     arr = own_pred[j]
        #     arr = [str(i) for i in arr]
        #     arr = (','.join(arr))
        #     add_arr.append(arr)


        # submission = pd.read_csv(pathLabel+"sample_submission.csv")
        # submission["label"] = f_result
        # submission["score_88"] = add_arr
        # submission.to_csv(pathLabel+f"submissions/add_clssifier_layer_epochs{epoch}_{batch_size}_lr_{lr}_val_f1_{val_f1}.csv", index = False)

        # gc.collect()
    return val_f1


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
