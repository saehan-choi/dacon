import random
import os

import numpy as np
import pandas as pd
from torch import optim
import torch

import torch.nn as nn

from sklearn.model_selection import KFold

from tqdm import tqdm

import pandas as pd

class CFG:
    dataPath = "antenna performance prediction for autonomous driving sensors/data/"
    trainPath = dataPath+'raw/train.csv'
    testPath = dataPath+'raw/test.csv'
    submission = dataPath+'raw/sample_submission.csv'
    outPath = dataPath+'processed/'
    weightsavePath = dataPath+'weights/'
    
    device = 'cuda'

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(56, 512)
        self.layer1 = self.make_layers(512, num_repeat=300)
        # num_repeat 4로 시작해볼것
        self.relu = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(512, 14)


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = nn.Dropout(0.2)(x)
        x = self.fc5(x)
        return x

    def make_layers(self, value, num_repeat):
        layers = []
        for _ in range(num_repeat):
            layers.append(nn.Linear(value, value))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))

def seedEverything(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    np.random.seed(random_seed)    
    # np.random.seed(random_seed)

def numpy_to_tensor(variable):
    x = variable.values
    x = np.array(x, dtype=np.float32)
    x = torch.from_numpy(x)
    return x

def pandas_to_tensor(variable):
    return torch.tensor(variable.values)

def train_one_epoch(model, train_batch, criterion, optimizer, train_X, train_Y, device):
    running_loss = 0
    dataset_size = 0

    model.train()
    for i in range(train_batch+1):
        
        start = i * batch_size
        end = start + batch_size
        input = train_X[start:end].to(device, dtype=torch.float)
        label = train_Y[start:end].to(device, dtype=torch.float)

        input, label = input.to(device), label.to(device)
        outputs = model(input).squeeze()
        loss = criterion(outputs, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        output_size = len(label)
        running_loss += loss.item()*output_size
        dataset_size += output_size
        train_loss = running_loss/dataset_size

    # 이거 loss 구할때 batch로 나눠줘야함 지금 train_loss랑 val_loss랑 차이가 날수밖에없네
    print(f"train_loss : {train_loss}")

def val_one_epoch(model, val_batch, criterion, val_X, val_Y, device):
    running_loss = 0
    dataset_size = 0
    
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for i in range(val_batch+1):
            start = i * batch_size
            end = start + batch_size
            input = val_X[start:end].to(device, dtype=torch.float)
            label = val_Y[start:end].to(device, dtype=torch.float)

            input, label = input.to(device), label.to(device)
            outputs = model(input).squeeze()
            loss = criterion(outputs, label)

            output_size = len(label)
            running_loss += loss.item()*output_size
            dataset_size += output_size
            val_loss = running_loss/dataset_size

    print(f"val_loss : {val_loss}")

def datapreparation_with_FOLD(train_df_fold, val_df_fold):

    train_X = train_df_fold.filter(regex='X')
    train_Y = train_df_fold.filter(regex='Y')

    val_X = val_df_fold.filter(regex='X')
    val_Y = val_df_fold.filter(regex='Y')

    train_X = pandas_to_tensor(train_X)
    train_Y = pandas_to_tensor(train_Y)    
    val_X = pandas_to_tensor(val_X)
    val_Y = pandas_to_tensor(val_Y)

    return train_X, train_Y, val_X, val_Y

def report_txt():
    f = open(CFG.outPath+'report.txt', 'a')
    # pd.
    f.write()
    # csv로 남길수 있도록 할것..!
    # epoch 기록남길것... 이거 안남기니깐 금방사라짐.
    pass

if __name__ == '__main__':
    seedEverything(52)
    train_df = pd.read_csv(CFG.trainPath)    
    # train_df_X, train_df_Y, val_df_X, val_df_Y = datapreparation(train_df)

    kfold = KFold(n_splits=5, shuffle=True)

    for train, val in kfold.split(train_df):
        train_df_fold = train_df.iloc[train]
        val_df_fold = train_df.iloc[val]

        train_X, train_Y, val_X, val_Y = datapreparation_with_FOLD(train_df_fold, val_df_fold)


        model = NeuralNet()
        model = model.to(CFG.device)
        optimizer = optim.Adam(model.parameters(),lr=0.0008121375255305587)
        criterion = RMSELoss().cuda()

        num_epochs = 50
        batch_size = 4096
        train_batch = len(train_X) // batch_size
        val_batch = len(val_X) // batch_size

        for epoch in range(num_epochs):
            print(f"epoch:{epoch}")
            train_one_epoch(model, train_batch, criterion, optimizer, train_X, train_Y, CFG.device)
            val_one_epoch(model, val_batch, criterion, val_X, val_Y, CFG.device)
            # torch.save(model.state_dict(), CFG.weightsavePath+f'{epoch}_neuralnet.pt')
            print('\n')

        # PATH = './weights/'
