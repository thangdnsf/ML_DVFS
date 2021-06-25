#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 16:46:31 2021

@author: ndthang
"""
import pandas as pd
import numpy as np
import sys
import os
import time
import random
from model import DVFSModel
from datagenerator import HPC_DVFS, labels,labelsZ, make_weights_for_balanced_classes
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR 
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
import matplotlib.pyplot as plt
import util
from datetime import datetime
import json
from warnings import filterwarnings
filterwarnings("ignore")

# hyperparmeter
kfold = 0
seed = 42
warmup_epo = 5
init_lr = 1e-3
batch_size = 100
valid_batch_size = 100
n_epochs = 300#1000
num_batch = 400
warmup_factor = 10
num_workers = 4
use_amp = True
early_stop = 100
device = torch.device('cuda')
model_dir = 'logs7/'

class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)
    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True # for faster training, but not deterministic
    
seed_everything(seed)


#load dataset for training
df = pd.read_csv('../../data4train/train.csv', na_filter= False, index_col=0)

ignore_columns = ['typedata','expe','UID','target','targetZ','kfold', #'guest', 'guest_nice', 'irq',
                  #'steal','nice','emulation_faults','irxp', 'irxb', 
                  #'itxp', 'itxb', 'core0','core1','iowait','softirq','txp',
                 ]
feature_list = list(set(df.columns) - set(ignore_columns))
train_df = df[df.kfold != kfold]
train_df = train_df.sample(frac=1)
val_df = df[df.kfold == kfold]
#standarzation data training
sc = StandardScaler()

train_loader_= HPC_DVFS(df=train_df,mode='training',augmentation = True, feature_list=feature_list,sc=sc)
sc_train = train_loader_.StandardScaler()
val_loader_ = HPC_DVFS(df=val_df,mode='valid',feature_list= feature_list,sc=sc_train)

weights = make_weights_for_balanced_classes(train_df.target.values, len(labels))
weights = torch.DoubleTensor(weights)                                       
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))                     
                                                                                
train_loader = torch.utils.data.DataLoader(train_loader_, batch_size=batch_size,sampler = sampler, num_workers=num_workers, pin_memory=True)     
valid_loader = torch.utils.data.DataLoader(val_loader_,batch_size=valid_batch_size, num_workers=num_workers, pin_memory=True)

model = DVFSModel(n_cont=len(feature_list), out_sz=len(labels),out_szz=len(labelsZ), szs=[1024,512,256, 128, 64], drops=[0.001,0.01,0.05, 0.1,0.2])
model = model.to(device)

x,y = next(iter(train_loader))
x = x.to(device)
yhat = model(x)
from torchviz import make_dot
from torchsummary import summary
make_dot(yhat, params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")
summary(model, [190])
