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
model_dir = 'logs1/'

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

ignore_columns = ['typedata','expe','UID','target','targetZ','Kfold', #'guest', 'guest_nice', 'irq',
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

model = DVFSModel(n_cont=len(feature_list), out_sz=len(labelsZ), szs=[1024,512,256, 128, 64], drops=[0.001,0.01,0.05, 0.1,0.2])
model = model.to(device)
criterion = nn.BCEWithLogitsLoss()
criterion2 = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=init_lr/warmup_factor)
scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, eta_min=1e-7)
scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=warmup_epo, after_scheduler=scheduler_cosine)

def train_func(train_loader):
    model.train()
    bar = tqdm(train_loader)
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
    losses = []
    for batch_idx, (inputarr, targets) in enumerate(bar):
        inputarr, targetZ, target = inputarr.to(device), targets['target'].to(device),targets['targetZ'].to(device)
        
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(inputarr)
                loss_target = criterion(logits, target)
                #loss_target2 = criterion2(logits[0], target)
                #loss_targetZ= criterion(logits[1],targetZ)
                #loss_targetZ2= criterion2(logits[1],targetZ)
                loss = loss_target#*0.7 + loss_targetZ*0.3
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            logits = model(inputarr)
            loss_target = criterion(logits, target)
            #loss_targetZ= criterion(logits[1],targetZ)
            loss = loss_target#*0.7 + loss_targetZ*0.3
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        losses.append(loss.item())
        smooth_loss = np.mean(losses[-30:])

        bar.set_description(f'loss: {loss.item():.5f}, smth: {smooth_loss:.5f}')

    loss_train = np.mean(losses)
    return loss_train
def valid_func(valid_loader):
    model.eval()
    bar = tqdm(valid_loader)

    PROB = []
    TARGETS_target = []
    TARGETS_targetZ = []
    losses = []
    PREDS_target = []
    PREDS_targetZ = []
    
    with torch.no_grad():
        for batch_idx, (inputarr, targets) in enumerate(bar):
            inputarr, targetZ, target = inputarr.to(device), targets['target'].to(device),targets['targetZ'].to(device)
        
            logits = model(inputarr)
            PREDS_target += [logits]
            TARGETS_target += [target.detach().cpu()]
            
            loss_target = criterion(logits, target)
            
            loss = loss_target#*0.7 + loss_targetZ*0.3
            losses.append(loss.item())
            smooth_loss = np.mean(losses)
            bar.set_description(f'loss: {loss.item():.5f}, smth: {smooth_loss:.5f}')
            
    PREDS_target = torch.cat(PREDS_target).cpu().numpy()
    TARGETS_target = torch.cat(TARGETS_target).cpu().numpy()
    
    
    roc_auc_target,roc_roc_dt = util.macro_multilabel_auc(TARGETS_target, PREDS_target, labels=labels)
    F1_target = util.F1_score_mul(TARGETS_target, PREDS_target)
    accuracy_target = util.accuracy_score_mul(TARGETS_target, PREDS_target)
    loss_valid = np.mean(losses)
    return loss_valid, roc_auc_target,F1_target, {'target':roc_roc_dt}, accuracy_target


log = {}
roc_auc_max = 0
F1_max = 0
accuracy_max = 0
loss_min = 999
not_improving = 0
date_now = datetime.now().strftime("%m_%d_%Y-%H:%M:%S")
for epoch in range(1, n_epochs+1):
    scheduler_warmup.step(epoch-1)
    loss_train = train_func(train_loader)
    loss_valid, roc_auc_target,F1_target,roc_roc_dt, accuracy_target = valid_func(valid_loader)
    
    if (epoch%10 == 0):
        with open(f'{model_dir}/log_auc_dt.json', 'w') as fp:
            json.dump(roc_roc_dt, fp, sort_keys=True, indent=4)
    
    roc_auc = (roc_auc_target)
    F1_total =  ( F1_target)
    accuracy_total = (accuracy_target )
    content = time.ctime() + ' ' + f'Fold {kfold}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, loss_train: {loss_train:.5f}, \
        loss_valid: {loss_valid:.5f}, roc_auc_target: {roc_auc_target:.6f},\
        F1_target: {F1_target:.6f},accuracy_target: {accuracy_target:.6f} .'
    print(content)
    
    log['loss_train'] = log.get('loss_train', []) + [loss_train]
    log['loss_valid'] = log.get('loss_valid', []) + [loss_valid]
    log['lr'] = log.get('lr', []) + [optimizer.param_groups[0]["lr"]]
    log['roc_auc_target'] = log.get('roc_auc_target', []) + [roc_auc_target]
    #log['roc_auc_targetZ'] = log.get('roc_auc_targetZ', []) + [roc_auc_targetZ]
    log['F1_target'] = log.get('F1_target', []) + [F1_target]
    #log['F1_targetZ'] = log.get('F1_targetZ', []) + [F1_targetZ]
    log['accuracy_target'] = log.get('accuracy_target', []) + [accuracy_target]
    #log['accuracy_targetZ'] = log.get('accuracy_targetZ', []) + [accuracy_targetZ]
    
    
    not_improving += 1
    
    if roc_auc > roc_auc_max:
        print(f'roc_auc_max ({roc_auc_max:.6f} --> {roc_auc:.6f}). Saving model ...')
        torch.save(model.state_dict(), f'{model_dir}_fold{kfold}_best_AUC.pth')
        roc_auc_max = roc_auc
        not_improving = 0
        
    if F1_total > F1_max:
        print(f'F1_max ({F1_max:.6f} --> {F1_total:.6f}). Saving model ...')
        torch.save(model.state_dict(), f'{model_dir}_fold{kfold}_best_F1.pth')
        F1_max = F1_total
        not_improving = 0
    
    if accuracy_total > accuracy_max:
        print(f'accuracy_max ({accuracy_max:.6f} --> {accuracy_total:.6f}). Saving model ...')
        torch.save(model.state_dict(), f'{model_dir}_fold{kfold}_best_accuracy.pth')
        accuracy_max = accuracy_total
        not_improving = 0

    if loss_valid < loss_min:
        loss_min = loss_valid
        torch.save(model.state_dict(), f'{model_dir}_fold{kfold}_best_loss.pth')
        
    if not_improving == early_stop:
        print('Early Stopping...')
        break
    with open(f'{model_dir}/log_{date_now}.json', 'w') as fp:
        json.dump(log, fp, sort_keys=True, indent=4)
    