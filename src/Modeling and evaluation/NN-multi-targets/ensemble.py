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
num_fold = 5
seed = 42
warmup_epo = 5
batch_size = 100
num_workers = 4
device = torch.device('cuda')
model_dir = 'logs14/'

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
df = pd.read_csv('../../data4train/test.csv', na_filter= False, index_col=0)


from pickle import load, dump
#Load standarzation data training
sc_train = load(open(f'{model_dir}/0_scaler.pkl', 'rb'))
feature_list = load(open(f'{model_dir}/0_feature_list.pkl', 'rb'))
test_df = df

test_loader_ = HPC_DVFS(df=test_df,mode='valid',feature_list = feature_list,sc=sc_train)

test_loader = torch.utils.data.DataLoader(test_loader_,batch_size=batch_size, num_workers=num_workers, pin_memory=True)

models = []
for kfold in range(num_fold):
    model = DVFSModel(n_cont=len(feature_list), out_sz=len(labels),out_szz=len(labelsZ), szs=[1024,512,256, 128, 64], drops=[0.001,0.01,0.05, 0.1,0.2])
    model = model.to(device)
    model.load_state_dict(torch.load(f'{model_dir}_fold{kfold}_best_accuracy.pth',map_location='cuda:0'))
    model.eval()
    models.append(model)

bar = tqdm(test_loader)
PROB = []
TARGETS_target = []
TARGETS_targetZ = []
TARGETS_target_id = []
TARGETS_targetZ_id = []
losses = []
PREDS_target = []
PREDS_targetZ = []
PREDS_target_id = []
PREDS_targetZ_id = []

with torch.no_grad():
    for batch_idx, (inputarr, targets) in enumerate(bar):
        inputarr, target, targetZ = inputarr.to(device), targets['target'].to(device),targets['targetZ'].to(device)
        
        logits = []
        for i in range(num_fold):
            logit = models[i](inputarr)
            logits.append(logit)
        f_target = (logits[0][0]+logits[1][0]+logits[2][0]+logits[3][0]+logits[4][0])/num_fold
        PREDS_target += [f_target]
        PREDS_target_id += [torch.argmax(f_target,dim = 1)]
        
        TARGETS_target += [target.detach().cpu()]
        TARGETS_target_id += [torch.argmax(target,dim = 1).detach().cpu()]
        f_targetZ = (logits[0][1]+logits[1][1]+logits[2][1]+logits[3][1]+logits[4][1])/num_fold
        PREDS_targetZ += [f_targetZ]
        PREDS_targetZ_id += [torch.argmax(f_targetZ,dim = 1)]
        TARGETS_targetZ += [targetZ.detach().cpu()]
        TARGETS_targetZ_id += [torch.argmax(targetZ,dim = 1).detach().cpu()]
        
PREDS_target = torch.cat(PREDS_target).cpu().numpy()
PREDS_target_id = torch.cat(PREDS_target_id).cpu().numpy()
TARGETS_target = torch.cat(TARGETS_target).cpu().numpy()
TARGETS_target_id = torch.cat(TARGETS_target_id).cpu().numpy()

PREDS_targetZ = torch.cat(PREDS_targetZ).cpu().numpy()
TARGETS_targetZ = torch.cat(TARGETS_targetZ).cpu().numpy()
PREDS_targetZ_id = torch.cat(PREDS_targetZ_id).cpu().numpy()
TARGETS_targetZ_id = torch.cat(TARGETS_targetZ_id).cpu().numpy()

roc_auc_target,roc_roc_dt = util.macro_multilabel_auc(TARGETS_target, PREDS_target, labels=labels)
F1_target = util.F1_score_mul(TARGETS_target_id, PREDS_target_id)
accuracy_target = util.accuracy_score_mul(TARGETS_target_id, PREDS_target_id)

roc_auc_targetZ,roc_roc_dtZ = util.macro_multilabel_auc(TARGETS_targetZ, PREDS_targetZ, labels=labels)
F1_targetZ = util.F1_score_mul(TARGETS_targetZ_id, PREDS_targetZ_id)
accuracy_targetZ = util.accuracy_score_mul(TARGETS_targetZ_id, PREDS_targetZ_id)

loss_valid = np.mean(losses)
#return loss_valid, roc_auc_target, roc_auc_targetZ,F1_target, F1_targetZ, {'target':roc_roc_dt,'targetZ':roc_roc_dtZ}, accuracy_target,accuracy_targetZ

from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, f1_score
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns
def draw_cfm(y_test, preds):
    print(classification_report(y_test, preds))
    cfm = confusion_matrix(y_test, preds, labels = range(0,13))
    cm = cfm
    cfm = cfm.astype('float') / cfm.sum(axis=1)[:, np.newaxis]
    tmplabel = [i//100000 for i in labels]
    annot = np.empty_like(cfm).astype(str)
    nrows, ncols = cfm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cfm[i, j]*100
            if c == 0:
                annot[i, j] = ''
            else:
              annot[i, j] = '%.1f%%\n%d' % (c,cm[i, j])
    plt.figure(figsize=(12,10))
    ax = sns.heatmap(cfm, annot=annot, fmt='',cmap="YlGnBu")
    ax.set_xticklabels(tmplabel)
    ax.set_yticklabels(tmplabel)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()       

roc_auc = (roc_auc_target+roc_auc_targetZ)/2
F1_total = (F1_targetZ + F1_target)/2
accuracy_total = (accuracy_target + accuracy_targetZ)/2
content = time.ctime() + ' ' + f'Fold {kfold}, \
    loss_valid: {loss_valid:.5f}, roc_auc_target: {roc_auc_target:.6f}, roc_auc_targetZ: {roc_auc_targetZ:.6f}, total_roc_auc: {roc_auc:.6f},\
    F1_target: {F1_target:.6f},F1_targetZ: {F1_targetZ:.6f},accuracy_target: {accuracy_target:.6f}, accuracy_targetZ: {accuracy_targetZ:.6f} .'
print(content)
draw_cfm(TARGETS_target_id, PREDS_target_id)
draw_cfm(TARGETS_targetZ_id, PREDS_targetZ_id)
print('mean_squared_error:',mean_squared_error(TARGETS_target_id, PREDS_target_id))
