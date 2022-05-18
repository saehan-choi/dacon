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
import joblib



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
            randomErasing_p, randomAdjustSharpness_factor, randomRandomApply_p = self.probablity[0], self.probablity[1], self.probablity[2]
            randomResizedCrop_scale_min, randomResizedCrop_scale_max, randomResizedCrop_ratio_min, randomResizedCrop_ratio_max=self.probablity[3], self.probablity[4], self.probablity[5], self.probablity[6]
            randomGrayscale_p = self.probablity[7]
            augmentation = random.randint(0,3)
            
            if augmentation==1:
                img = img[::-1].copy()
                # 수평변환

            elif augmentation==2:
                img = img[:,::-1].copy()
                # 수직변환

            elif augmentation==3:
                # probability = [randomErasing_p, randomPerspective_p, randomAdjustSharpness_factor, 
                #             randomEqualize_p, randomRandomApply_p, randomInvert_p]

                transform_ = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.RandomErasing(scale=(0.01,0.05),ratio=(0.01,0.05), p=randomErasing_p),
                                    transforms.RandomAdjustSharpness(randomAdjustSharpness_factor,p=1),
                                    transforms.RandomApply(transforms=[transforms.GaussianBlur(kernel_size=(3,7))],p=randomRandomApply_p),
                                    transforms.RandomResizedCrop(512,scale=(randomResizedCrop_scale_min, randomResizedCrop_scale_max),ratio=(randomResizedCrop_ratio_min, randomResizedCrop_ratio_max)),
                                    transforms.RandomGrayscale(p=randomGrayscale_p),
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

    lr = trial.suggest_float("lr", 1e-4, 1e-2)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # epochs = trial.suggest_int('epochs',5,25)
    # epochs=20
    # batch_size = trial.suggest_int('batch_size',8,32)
    seed = 42

    # randomErasing_p = trial.suggest_float("randomErasing_p",0,0.5)
    # randomPerspective_p = trial.suggest_float("randomErasing_p",0,0.5)
    # randomAdjustSharpness_factor = trial.suggest_float("randomAdjustSharpness_factor", 0, 0.5)
    # randomRandomApply_p = trial.suggest_float("randomRandomApply_p",0,0.1)

    # randomResizedCrop_scale_min = trial.suggest_float("randomResizedCrop_scale_min",0.9,1)
    # randomResizedCrop_scale_max = trial.suggest_float("randomResizedCrop_scale_max",1,1.1)
    # randomResizedCrop_ratio_min = trial.suggest_float("randomResizedCrop_ratio_min",0.9,1)
    # randomResizedCrop_ratio_max = trial.suggest_float("randomResizedCrop_ratio_max",1,1.1)

    # randomGrayscale_p = trial.suggest_float("randomGrayscale_p",0, 0.2)


    # # 검증시 사용
    lr = 3e-4
    epochs = 23
    optimizer = optim.Adam(model.parameters(),lr=lr)
    batch_size = 16
    
    randomErasing_p = 0.5
    randomAdjustSharpness_factor = 0.45
    randomRandomApply_p = 0
    randomResizedCrop_scale_min = 0.9
    randomResizedCrop_scale_max = 1
    randomResizedCrop_ratio_min = 0.9
    randomResizedCrop_ratio_max = 1
    randomGrayscale_p = 0.2

    probability = [randomErasing_p, randomAdjustSharpness_factor, randomRandomApply_p,
                randomResizedCrop_scale_min, randomResizedCrop_scale_max, randomResizedCrop_ratio_min, randomResizedCrop_ratio_max,
                randomGrayscale_p]

    set_seed(seed)
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
    print(f'erasing_{randomErasing_p}_Sharpness_factor{randomAdjustSharpness_factor}_RandomApply_{randomRandomApply_p}_scale_min_{randomResizedCrop_scale_min}_scale_max_{randomResizedCrop_scale_min}_ratio_min_{randomResizedCrop_ratio_min}_ratio_max_{randomResizedCrop_ratio_max}_randomGrayscale_p_{randomGrayscale_p}')
    submission.to_csv(pathLabel+f"submissions/OPTUNA{epoch}_{batch_size}_lr_{lr}_val_f1_{val_f1}.csv", index = False)
    gc.collect()

    return val_f1


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    joblib.dump(study, "optuna_result/optuna_result1.pkl")

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


# TRAIN loss : 0.60812  f1: 0.41184 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 1/23
# VAL loss : 0.90750   f1: 0.53561 lr: 0.0003 batch: 16
# epoch : 2/23    time : 290s/6084s
# TRAIN loss : 0.20431  f1: 0.73368 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 2/23
# VAL loss : 0.88480   f1: 0.69070 lr: 0.0003 batch: 16
# epoch : 3/23    time : 288s/5751s
# TRAIN loss : 0.13413  f1: 0.84065 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 3/23
# VAL loss : 0.75611   f1: 0.74470 lr: 0.0003 batch: 16
# epoch : 4/23    time : 288s/5479s
# TRAIN loss : 0.09752  f1: 0.88612 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 4/23
# VAL loss : 0.71638   f1: 0.76978 lr: 0.0003 batch: 16
# epoch : 5/23    time : 286s/5143s
# TRAIN loss : 0.06693  f1: 0.91787 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 5/23
# VAL loss : 0.73802   f1: 0.77991 lr: 0.0003 batch: 16
# epoch : 6/23    time : 288s/4888s
# TRAIN loss : 0.05901  f1: 0.93823 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 6/23
# VAL loss : 0.69865   f1: 0.79289 lr: 0.0003 batch: 16
# epoch : 7/23    time : 283s/4525s
# TRAIN loss : 0.06401  f1: 0.93055 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 7/23
# VAL loss : 0.83336   f1: 0.79149 lr: 0.0003 batch: 16
# epoch : 8/23    time : 284s/4265s
# TRAIN loss : 0.05000  f1: 0.93704 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 8/23
# VAL loss : 0.97006   f1: 0.73424 lr: 0.0003 batch: 16
# epoch : 9/23    time : 283s/3968s
# TRAIN loss : 0.04096  f1: 0.95622 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 9/23
# VAL loss : 0.83207   f1: 0.76834 lr: 0.0003 batch: 16
# epoch : 10/23    time : 284s/3697s
# TRAIN loss : 0.04665  f1: 0.95057 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 10/23
# VAL loss : 0.81400   f1: 0.79986 lr: 0.0003 batch: 16
# epoch : 11/23    time : 281s/3372s
# TRAIN loss : 0.03194  f1: 0.95966 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 11/23
# VAL loss : 0.85843   f1: 0.79010 lr: 0.0003 batch: 16
# epoch : 12/23    time : 281s/3094s
# TRAIN loss : 0.04813  f1: 0.94919 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 12/23
# VAL loss : 0.86018   f1: 0.77750 lr: 0.0003 batch: 16
# epoch : 13/23    time : 283s/2827s
# TRAIN loss : 0.03225  f1: 0.96880 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 13/23
# VAL loss : 1.05335   f1: 0.75303 lr: 0.0003 batch: 16
# epoch : 14/23    time : 283s/2544s
# TRAIN loss : 0.03146  f1: 0.96964 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 14/23
# VAL loss : 0.99126   f1: 0.77666 lr: 0.0003 batch: 16
# epoch : 15/23    time : 283s/2265s
# TRAIN loss : 0.03404  f1: 0.96368 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 15/23
# VAL loss : 0.91137   f1: 0.76593 lr: 0.0003 batch: 16
# epoch : 16/23    time : 280s/1958s
# TRAIN loss : 0.03618  f1: 0.96917 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 16/23
# VAL loss : 0.88563   f1: 0.76421 lr: 0.0003 batch: 16
# epoch : 17/23    time : 280s/1683s
# TRAIN loss : 0.02540  f1: 0.97885 lr: 0.0003 batch: 16
# validation 진행중
# epoch    : 17/23
# VAL loss : 0.76994   f1: 0.81341 lr: 0.0003 batch: 16

# 이 상태에서 0.81 찍었습네다 !!!!!!!!!!!!!!!!!!!!!!!!!