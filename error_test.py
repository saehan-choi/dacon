
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


size = (5,15)

gt = torch.ones(size).numpy()

preds = torch.ones(size).numpy()*1.2

score = lg_nrmse(gt, preds)

print(f'score:{score}')


class LG_NRMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        all_nrmse = []
        for idx in range(14):
            mse = self.mse(yhat[:, idx], y[:, idx])
            print(f"rmse:{mse}")
            # nrmse = rmse/torch.mean(torch.abs(y[:,idx]))
            all_nrmse.append(mse.item())

        loss = 1.2 * sum(all_nrmse[:8]) + 1.0 * sum(all_nrmse[8:])
        return loss


size = (5,14)

gt = torch.ones(size)

preds = torch.ones(size)*1.2

score = LG_NRMSELoss()
score = score(preds, gt)

print(f'score:{score}')


# from sklearn.metrics import mean_squared_error
# import math
# import numpy as np
# y_true = np.array([3, -0.5, 2, 7])
# y_pred = np.array([2.5, 0.0, 2, 8])
# print(mean_squared_error(y_true, y_pred))

# # root((3-2.5)^2 + (-0.5)^2 + 0 + (7-8)^2)/4

# print(math.sqrt((3-2.5)**2 + (-0.5)**2 + 0 + (7-8)**2)/4)

# print(np.average((y_true - y_pred) ** 2))


# k = (y_true-y_pred)**2
# print(sum(k)/len(k))
