#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 19:05:51 2021

@author: ndthang
"""

import numpy as np
from sklearn.metrics import roc_auc_score, jaccard_score,accuracy_score, f1_score
import torch


def f1_loss(y_true:torch.Tensor, y_pred:torch.Tensor, is_training=False) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    
    '''
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2
    
    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)
        
    
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    return f1

def F1_score_mul(label, pred):
    #pred = (pred > 0.5)
    return f1_score(label, pred)

def accuracy_score_mul(label, pred):
    #pred = (pred > 0.5)
    return accuracy_score(label, pred)
def macro_multilabel_auc(label, pred, labels):
    aucs = []
    aucs_dt = {}
    for i in range(len(labels)):
        try:
            rl = roc_auc_score(label[:, i], pred[:, i])
            aucs.append(rl)
            aucs_dt[labels[i]] = rl
        except:
            aucs_dt[labels[i]] = -1
            
    return np.mean(aucs), aucs_dt

def jaccard(y_true, y_pred):
    smooth=0.001
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f,axis=-1)
    jac=(smooth+intersection)/(smooth-intersection+np.sum(y_pred_f,axis=-1)+np.sum(y_true_f, axis=-1))
    return jac
