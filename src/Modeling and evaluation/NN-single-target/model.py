#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 16:47:45 2021

@author: ndthang
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DVFSModel(nn.Module):
    def __init__(self, n_cont, out_sz, szs, drops, use_bn=True):
        super().__init__()
        self.n_cont= n_cont
        
        szs = [n_cont] + szs
        self.lins = nn.ModuleList([nn.Linear(szs[i], szs[i+1]) for i in range(len(szs)-1)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(sz) for sz in szs[1:]])
        for o in self.lins: 
            nn.init.kaiming_normal_(o.weight.data)
        self.outp = nn.Linear(szs[-1], out_sz)
        nn.init.kaiming_normal_(self.outp.weight.data)
        

        self.drops = nn.ModuleList([nn.Dropout(drop) for drop in drops])
        self.bn = nn.BatchNorm1d(n_cont)
        self.use_bn = use_bn

    def forward(self, x_cont):
        
        x = self.bn(x_cont)
        for l,d,b in zip(self.lins, self.drops, self.bns):
            x = F.relu(l(x))
            if self.use_bn: 
                x = b(x)
            x = d(x)
        x1 = self.outp(x)
        x1 = torch.softmax(x1,dim = 1)
        return x1.squeeze()