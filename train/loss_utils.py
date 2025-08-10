import torch.nn as nn
import torch

def get_loss(pos_weights=None):
    if pos_weights:
        weights = torch.tensor(pos_weights)
        return nn.BCEWithLogitsLoss(pos_weight=weights)
    return nn.BCEWithLogitsLoss()