
import numpy as np
from torch.nn.modules.container import Sequential

import sklearn.metrics as metrics

import torch.nn as nn
import torch

def lg_nrmse(gt, preds):
    # 각 Y Feature별 NRMSE 총합
    # Y_01 ~ Y_08 까지 20% 가중치 부여
    all_nrmse = []
    for idx in range(1,15): # ignore 'ID'
        rmse = metrics.mean_squared_error(gt[:,idx], preds[:,idx], squared=False)
        print(f"rmse:{rmse}")
        nrmse = rmse/np.mean(np.abs(gt[:,idx]))
        all_nrmse.append(nrmse)
    print(len(all_nrmse))
    score = 1.2 * np.sum(all_nrmse[:8]) + 1.0 * np.sum(all_nrmse[8:])
    return score

class LG_NRMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, yhat, y):
        all_nrmse = []
        for idx in range(14):
            rmse = self.mse(yhat[:, idx], y[:, idx])
            print(f"rmse:{rmse}")
            nrmse = rmse/torch.mean(torch.abs(y[:,idx]))
            all_nrmse.append(nrmse.item())

        loss = 1.2 * sum(all_nrmse[:8]) + 1.0 * sum(all_nrmse[8:])
        return loss


def torch_lg_nrmse(gt, preds):
    gt, preds = gt.detach().cpu().numpy(), preds.detach().cpu().numpy()
    all_nrmse = []
    for idx in range(1,14): # ignore 'ID'
        rmse = metrics.mean_squared_error(gt[:,idx], preds[:,idx], squared=False)
        print(f"rmse:{rmse}")
        nrmse = rmse/np.mean(np.abs(gt[:,idx]))
        all_nrmse.append(nrmse)
    score = 1.2 * np.sum(all_nrmse[:8]) + 1.0 * np.sum(all_nrmse[8:14])
    print(len(all_nrmse))
    return score



size = (5,15)

gt = torch.ones(size).numpy()

preds = torch.ones(size).numpy()*1.2

score = lg_nrmse(gt, preds)

print(f'score:{score}')




size = (5,14)

gt = torch.ones(size)

preds = torch.ones(size)*1.2

score = torch_lg_nrmse(preds, gt)

print(f'score:{score}')