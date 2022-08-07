import os
import random
import optuna
import joblib
import torch
import numpy as np
import pandas as pd
import torch.nn as nn

from torch import optim
from optuna.trial import TrialState
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

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
        # x = nn.Dropout(0.4)(x)
        # we higher up the drop ratio, because over fitting..!
        x = nn.Dropout(0.2)(x)
        x = self.fc5(x)
        return x

    def make_layers(self, value, num_repeat):
        layers = []
        for _ in range(num_repeat):
            layers.append(nn.Linear(value, value))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

def lg_nrmse(preds, gt):
    gt, preds = gt.detach().cpu().numpy(), preds.detach().cpu().numpy()
    assert gt.shape[1] == 14 or preds.shape[1] == 14

    all_nrmse = []
    for idx in range(14): # ignore 'ID'
        rmse = mean_squared_error(gt[:,idx], preds[:,idx], squared=False)
        nrmse = rmse/np.mean(np.abs(gt[:,idx]))
        all_nrmse.append(nrmse)

    score = 1.2 * np.sum(all_nrmse[:8]) + 1.0 * np.sum(all_nrmse[8:])
    score = torch.tensor(score.item(), requires_grad=True)
    return score


def numpy_to_tensor(variable):
    x = variable.values
    x = np.array(x, dtype=np.float32)
    x = torch.from_numpy(x)
    return x

def pandas_to_tensor(variable):
    return torch.tensor(variable.values)

def train_one_epoch(model, train_batch, criterion, optimizer, train_X, train_Y, batch_size, device):
    running_loss = 0
    dataset_size = 0
    
    model.train()
    for i in range(train_batch+1):
        start = i * batch_size
        end = start + batch_size
        inputs = train_X[start:end].to(device, dtype=torch.float)
        labels = train_Y[start:end].to(device, dtype=torch.float)
        
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs).squeeze()        
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output_size = len(labels)
        running_loss += loss.item()*output_size
        dataset_size += output_size
    train_loss = running_loss/dataset_size
        
    print(f"train_loss : {train_loss}")
    
def val_one_epoch(model, val_batch, criterion, val_X, val_Y, batch_size, device):
    running_loss = 0
    dataset_size = 0

    model.eval()
    with torch.no_grad():
        for i in range(val_batch+1):
            start = i * batch_size
            end = start + batch_size

            inputs = val_X[start:end].to(device, dtype=torch.float)
            labels = val_Y[start:end].to(device, dtype=torch.float)

            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)

            output_size = len(labels)
            running_loss += loss.item()*output_size
            dataset_size += output_size
        val_loss = running_loss/dataset_size

        print(f"val_loss : {val_loss}")
    return val_loss

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

def tunning(trial):
    seedEverything(42)

    train_df = pd.read_csv(CFG.trainPath)    
    kfold = KFold(n_splits=5, shuffle=True)
    val_loss_sum = 0
    for train, val in kfold.split(train_df):
        train_df_fold = train_df.iloc[train]
        val_df_fold = train_df.iloc[val]

        train_X, train_Y, val_X, val_Y = datapreparation_with_FOLD(train_df_fold, val_df_fold)

        num_epochs = trial.suggest_int("num_epochs", 5, 100)
        batch_size = 4096

        train_batch = len(train_X) // batch_size
        val_batch = len(val_X) // batch_size

        model = NeuralNet()
        model = model.to(CFG.device)
        lr = trial.suggest_float("lr", 1e-5, 1e-3)
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
        
        criterion = lg_nrmse

        for epoch in range(num_epochs):
            print(f"epoch:{epoch}")
            train_one_epoch(model, train_batch, criterion, optimizer, train_X, train_Y, batch_size, CFG.device)
            val_loss_result = val_one_epoch(model, val_batch, criterion, val_X, val_Y, batch_size, CFG.device)
            # torch.save(model.state_dict(), CFG.weightsavePath+f'{epoch}_neuralnet_optuna.pt')

        val_loss_sum += val_loss_result
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()        
    # 5fold divide
    val_loss_final = val_loss_sum/5
    return val_loss_final

if __name__ == '__main__':


    study = optuna.create_study(direction='minimize')
    study.optimize(tunning)
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