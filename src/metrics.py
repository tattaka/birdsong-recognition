import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score


def f1(logit, target):
    logit = logit.detach().cpu().numpy()
    y_true = target.detach().cpu().numpy()
    y_pred = logit.argmax(axis=1)
    score = f1_score(y_true, y_pred, average="macro")
    return torch.tensor(score)


def mAP(logit, target):
    logit = logit.detach().cpu().numpy()
    y_true = target.detach().cpu().numpy()
    score = average_precision_score(y_true, logit, average=None)
    score = np.nan_to_num(score).mean()
    return torch.tensor(score)
