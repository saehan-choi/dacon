import random
import os

import numpy as np
import pandas as pd
from torch import optim
import torch

import torch.nn as nn
from torch.nn.modules.container import Sequential


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
        self.fc1 = nn.Linear(51, 512)
        self.layer1 = self.make_layers(512, num_repeat=4)
        self.fc5 = nn.Linear(512, 1)
        self.relu = nn.ReLU(inplace=True)

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

def numpy_to_tensor(variable):
    x = variable.values
    x = np.array(x, dtype=np.float32)
    x = torch.from_numpy(x)
    return x

class CFG:
    datapath = "antenna performance prediction for autonomous driving sensors/data/"
    trainpath = datapath+'raw/train.csv'
    testpath = datapath+'raw/test.csv'
    submission = datapath+'raw/sample_submission.csv'
    outpath = datapath+'processed/'

if __name__ == '__main__':
    seedEverything(42)
    train_df = pd.read_csv(CFG.trainpath)
    # shuffle 
    train_df = train_df.sample(frac=1)
    # only X
    train_df_X = train_df.filter(regex='X')
    train_df_Y = train_df.filter(regex='Y')

    valset_ratio = round(len(train_df_X)*0.15)
    print(train_df_X(round(len(train_df_X)*0.15)))


# train_data = train_data.iloc[:]
# train_label_data = train_label_data.iloc[:]

# train_data = numpy_to_tensor(train_data)
# train_label_data = numpy_to_tensor(train_label_data)

# device = torch.device('cuda')
# net = NeuralNet()
# net = net.to(device)


# criterion = nn.L1Loss().cuda()
# # in this competition loss function is MAE Loss so I used it.
# optimizer = optim.Adam(net.parameters(),lr=1e-3)
# num_epochs = 25
# batch_size = 80
# batch_num_train = len(train_data) // batch_size
# batch_num_val = len(val_data) // batch_size

# for epochs in range(num_epochs):
#     train_loss = 0.0
#     test_loss = 0.0
#     for i in range(batch_num_train):
#         start = i * batch_size
#         end = start + batch_size
#         pred = train_data[start:end][:]
#         label = train_label_data[start:end]
#         pred, label = pred.to(device), label.to(device)
#         outputs = net(pred).squeeze()
#         loss = criterion(outputs, label)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()

#     print(f'epochs : {epochs}')
#     print(f'train_loss : {train_loss/batch_num_train}')
    
# PATH = './weights/'
# torch.save(net.state_dict(), PATH+'model_alot_of_feature.pt')
