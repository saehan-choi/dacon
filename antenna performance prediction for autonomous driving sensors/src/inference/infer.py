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

class CFG:
    dataPath = "antenna performance prediction for autonomous driving sensors/data/"
    trainPath = dataPath+'raw/train.csv'
    testPath = dataPath+'raw/test.csv'
    submission = dataPath+'raw/sample_submission.csv'
    outPath = dataPath+'processed/'
    weightsavePath = dataPath+'weights/'
    
    device = 'cuda'
    
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
        x = nn.Dropout(0.5)(x)
        # we higher up the drop ratio, because over fitting..!
        # x = nn.Dropout(0.2)(x)
        x = self.fc5(x)
        return x

    def make_layers(self, value, num_repeat):
        layers = []
        for _ in range(num_repeat):
            layers.append(nn.Linear(value, value))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

def numpy_to_tensor(variable):
    x = variable.values
    x = np.array(x, dtype=np.float32)
    x = torch.from_numpy(x)
    return x

def pandas_to_tensor(variable):
    return torch.tensor(variable.values)

def datapreparation(train_df):
    # shuffle
    valset_ratio = 0.15
    train_df = train_df.sample(frac=1)

    train_df_X = train_df.filter(regex='X')
    train_df_Y = train_df.filter(regex='Y')

    valset_num = round(len(train_df_Y)*valset_ratio)

    val_df_X = pandas_to_tensor(train_df_X.iloc[:valset_num])
    val_df_Y = pandas_to_tensor(train_df_Y.iloc[:valset_num])    
    train_df_X = pandas_to_tensor(train_df_X.iloc[valset_num:])
    train_df_Y = pandas_to_tensor(train_df_Y.iloc[valset_num:])

    return train_df_X, train_df_Y, val_df_X, val_df_Y

def testdata_prepation(test_df):
    test_df_X = test_df.filter(regex='X')
    test_df_X = pandas_to_tensor(test_df_X)
    
    return test_df_X

def model_import(weight_path):
    model = NeuralNet()
    model.load_state_dict(torch.load(CFG.weightsavePath+weight_path))
    model = model.to(CFG.device)

    return model
    
def submission_report(output):
    submit = pd.read_csv(CFG.submission)
    for idx, col in enumerate(submit.columns):
        if col=='ID':
            continue
        submit[col] = output[:,idx-1]
    return submit

if __name__ == '__main__':
    seedEverything(52)

    test_df = pd.read_csv(CFG.testPath)
    test_df_X = testdata_prepation(test_df)
    
    # weight_name = '179_neuralnet.pt' -> 1.82 best val set
    weight_name = '49_neuralnet.pt'
    
    model = model_import(weight_name)

    with torch.no_grad():
        model.eval()
        batch = test_df_X.to(CFG.device, dtype=torch.float)
        output = model(batch)
    
    output = output.detach().cpu().numpy()
    
    submission = submission_report(output)
    submission.to_csv(CFG.outPath+'submit_nn.csv', index=False)

