#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 11:22:37 2021

@author: thangdnsf
"""

import json
import numpy as np
import matplotlib.pyplot as plt
with open('logs1/log_06_17_2021-12:46:34.json') as fp:
    log = json.load(fp)
    

fig = plt.figure(figsize=(16, 18)) 
plt.subplot(511)
plt.title('Loss')
plt.plot(log['loss_train'], label='train')
i = np.argmin(log['loss_train'])
j = np.min(log['loss_train'])
plt.scatter(i,j,marker='X')
plt.text(i,j+0.15,s=str(j)[:6])
plt.plot(log['loss_valid'], label='test')
i = np.argmin(log['loss_valid'])
j = np.min(log['loss_valid'])
plt.scatter(i,j,marker='X')
plt.text(i,j+0.15,s=str(j)[:6])
plt.legend()
plt.subplot(512)
plt.title('AUC')
plt.plot(log['roc_auc_target'], label='target')
i = np.argmax(log['roc_auc_target'])
j = np.max(log['roc_auc_target'])
plt.scatter(i,j,marker='X')
plt.text(i,j,s=str(j)[:6])
plt.plot(log['roc_auc_targetZ'], label='targetZ')
i = np.argmax(log['roc_auc_targetZ'])
j = np.max(log['roc_auc_targetZ'])
plt.scatter(i,j,marker='X')
plt.text(i,j,s=str(j)[:6])
plt.legend()
plt.subplot(513)
plt.title('jaccard')
plt.plot(log['jaccard_target'], label='jaccard_target')
i = np.argmax(log['jaccard_target'])
j = np.max(log['jaccard_target'])
plt.scatter(i,j,marker='X')
plt.text(i,j,s=str(j)[:6])
plt.plot(log['jaccard_targetZ'], label='jaccard_targetZ')
i = np.argmax(log['jaccard_targetZ'])
j = np.max(log['jaccard_targetZ'])
plt.scatter(i,j,marker='X')
plt.text(i,j,s=str(j)[:6])
plt.legend()
plt.subplot(514)
plt.title('Accuracy')
plt.plot(log['accuracy_target'], label='accuracy_target')
i = np.argmax(log['accuracy_target'])
j = np.max(log['accuracy_target'])
plt.scatter(i,j,marker='X')
plt.text(i,j,s=str(j)[:6])
plt.plot(log['accuracy_targetZ'], label='accuracy_targetZ')
i = np.argmax(log['accuracy_targetZ'])
j = np.max(log['accuracy_targetZ'])
plt.scatter(i,j,marker='X')
plt.text(i,j,s=str(j)[:6])
plt.legend()

plt.subplot(515)
plt.title('learning rate')
plt.plot(log['lr'], label='lr')
plt.legend()
#plt.show()
plt.savefig('logs1/logs.png',dpi=fig.dpi)
