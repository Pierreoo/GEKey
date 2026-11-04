import torch
import torch.nn as nn
from geomloss import SamplesLoss

import numpy as np
from scipy.optimize import linear_sum_assignment

from .builder import LOSSES


@LOSSES.register_module()
class KeypointLossMatch(nn.Module):
    def __init__(self, loss_weight=1.0, sinkhorn=False) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.sinkhorn = sinkhorn
        self.loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)

    def hungarian_loss(self, preds, targets):
        preds_np = preds.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        cost_matrix = np.linalg.norm(preds_np[:, None, :] - targets_np[None, :, :], axis=2)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_preds = preds[row_ind]
        matched_targets = targets[col_ind]
        return torch.mean((matched_preds - matched_targets).pow(2).sum(dim=1))

    def forward(self, all_pred_peaks, all_peaks):
        total_loss = 0.0
        for i in range(len(all_pred_peaks)):
            pred_peaks = all_pred_peaks[i]
            peaks = all_peaks[i]

            if self.sinkhorn:
                emd = self.loss_fn(pred_peaks, peaks)
            else:
                emd = self.hungarian_loss(pred_peaks, peaks)

            total_loss += emd

        return total_loss * self.loss_weight
