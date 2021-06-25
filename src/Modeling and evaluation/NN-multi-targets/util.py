#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 19:05:51 2021

@author: ndthang
"""

import numpy as np
from sklearn.metrics import roc_auc_score, jaccard_score,accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F

class F1_Loss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true):
        
        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2* (precision*recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        return 1 - f1.mean()

def F1_score_mul(label, pred):
    
    return f1_score(label, pred, average='macro')

def accuracy_score_mul(label, pred):
    
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
