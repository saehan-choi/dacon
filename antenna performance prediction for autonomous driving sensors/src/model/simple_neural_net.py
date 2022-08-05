import random
import os

import numpy as np
import pandas as pd
from torch import optim
import torch

import torch.nn as nn
from torch.nn.modules.container import Sequential

from tqdm import tqdm

import gc

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
    model.train()
    train_loss = 0.0
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
        train_loss += loss.item()
    train_loss = train_loss/(train_batch+1)
    # 이거 loss 구할때 batch로 나눠줘야함 지금 train_loss랑 val_loss랑 차이가 날수밖에없네
    print(f"train_loss : {train_loss}")
    
def val_one_epoch(model, val_batch, criterion, val_X, val_Y, device):
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for i in range(val_batch+1):
            start = i * batch_size
            end = start + batch_size
            input = val_X[start:end].to(device, dtype=torch.float)
            label = val_Y[start:end].to(device, dtype=torch.float)

            input, label = input.to(device), label.to(device)
            outputs = model(input).squeeze()
            loss = criterion(outputs, label)
            val_loss += loss.item()
    val_loss = val_loss/(val_batch+1)
    print(f"val_loss : {val_loss}")

def datapreparation(train_df):
    # shuffle
    valset_ratio = 0.2
    train_df = train_df.sample(frac=1)

    train_df_X = train_df.filter(regex='X')
    train_df_Y = train_df.filter(regex='Y')

    valset_num = round(len(train_df_Y)*valset_ratio)

    val_df_X = pandas_to_tensor(train_df_X.iloc[:valset_num])
    val_df_Y = pandas_to_tensor(train_df_Y.iloc[:valset_num])    
    train_df_X = pandas_to_tensor(train_df_X.iloc[valset_num:])
    train_df_Y = pandas_to_tensor(train_df_Y.iloc[valset_num:])

    return train_df_X, train_df_Y, val_df_X, val_df_Y

def report_txt():
    f = open(CFG.outPath+'report.txt', 'a')
    # pd.
    f.write()

    # epoch 기록남길것... 이거 안남기니깐 금방사라짐.
    pass

if __name__ == '__main__':
    seedEverything(52)
    train_df = pd.read_csv(CFG.trainPath)    
    train_df_X, train_df_Y, val_df_X, val_df_Y = datapreparation(train_df)

    model = NeuralNet()
    model = model.to(CFG.device)
    optimizer = optim.Adam(model.parameters(),lr=0.00010547491234286884)
    criterion = nn.MSELoss().cuda()

    num_epochs = 50
    batch_size = 4096
    
    train_batch = len(train_df_X) // batch_size
    val_batch = len(val_df_X) // batch_size
    gc.collect()
    
    for epoch in range(num_epochs):
        print(f"epoch:{epoch}")
        train_one_epoch(model, train_batch, criterion, optimizer, train_df_X, train_df_Y, CFG.device)
        val_one_epoch(model, val_batch, criterion, val_df_X, val_df_Y, CFG.device)
        # report_txt()
        
        torch.save(model.state_dict(), CFG.weightsavePath+f'{epoch}_neuralnet.pt')

        print('\n')
    
    # PATH = './weights/'
