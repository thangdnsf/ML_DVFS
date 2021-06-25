#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 17:14:53 2021

@author: ndthang
"""

from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch
import numpy as np
import math
from collections import Counter
labels = [1200000,
     1300000,
     1400000,
     1500000,
     1600000,
     1700000,
     1800000,
     1900000,
     2000000,
     2100000,
     2200000,
     2300000,
     2400000]
labelsZ = [0,
     1200000,
     1300000,
     1400000,
     1500000,
     1600000,
     1700000,
     1800000,
     1900000,
     2000000,
     2100000,
     2200000,
     2300000,
     2400000]

def AddGaussianNoise(tensor, mean=0., std=0.01):
    return tensor + torch.Tensor(tensor.size()).random_(-1,1) * std + mean
    
def make_weights_for_balanced_classes(dflabels, nclasses):
    cnts = Counter(dflabels)
    total = len(dflabels)
    class_weights = [total/cnts[i] for i in labels]
    weight = [class_weights[labels.index(l)] for l in dflabels]
    return weight

class HPC_DVFS(Dataset):
    def __init__(self, df, mode,feature_list = None, sc = None,transform=None, augmentation = True):
        
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform
        self.augmentation = augmentation
        if feature_list is None:
            self.feature_list = list(set(df.columns) - {'target','targetZ'})
        else: 
            self.feature_list = feature_list
        self.sc = None
        if mode == 'training':
            if sc is not None:
                self.sc = sc
                self.df.loc[:,self.feature_list] = self.sc.fit_transform(self.df[self.feature_list])
        else:
            if sc is not None:
                self.sc = sc
                self.df.loc[:,self.feature_list] = self.sc.transform(self.df[self.feature_list])
        
    def StandardScaler(self):
        return self.sc
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        #print(index)
        row = self.df.loc[index]
        inputarr = row[self.feature_list]
        target = np.zeros(len(labels))
        if (row['target'] in labels):
            target[labels.index(row['target'])] = 1
        targetZ = np.zeros(len(labelsZ))
        if (row['targetZ'] in labelsZ):
            targetZ[labelsZ.index(row['targetZ'])] = 1
        
        label = {'target':torch.tensor(target).float(),
                 'targetZ':torch.tensor(targetZ).float()}
        
        if self.mode == 'test':
            return torch.tensor(inputarr).float()
        elif self.mode == 'valid':
            return torch.tensor(inputarr).float(), label
        else:
            if self.augmentation:
                return AddGaussianNoise(torch.tensor(inputarr),std=0.01).float(), label
            else:
                return torch.tensor(inputarr).float(), label