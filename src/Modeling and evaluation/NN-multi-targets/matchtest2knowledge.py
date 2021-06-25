#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 13:27:33 2021

@author: ndthang
"""

import pandas as pd

#read and merge all dataset
all_knowledge = pd.read_csv("../../csvs/knowledge_allmetrics.csv")
all_vectors = pd.read_csv("../../csvs/vectors_allmetrics.csv")
all_vectors['typedata'] = 0

all_knowledge_old = pd.read_csv("../../csvs/old/knowledge_allmetrics.csv")
all_vectors_old = pd.read_csv("../../csvs/old/vectors_allmetrics.csv")
all_knowledge_old['expe'] = all_knowledge_old['expe'] + all_knowledge.expe.values[-1] + 1
all_vectors_old['typedata'] = 1

all_knowledge = pd.concat([all_knowledge, all_knowledge_old],ignore_index=True)
all_vectors = pd.concat([all_vectors, all_vectors_old],ignore_index=True)

all_knowledge_ = all_knowledge[all_knowledge.metric == 'energy']
all_vectors_ = all_vectors[all_knowledge.metric == 'energy']
all_knowledge_['freq'] = all_knowledge_.fmax
df  = pd.read_csv('../../../src/data4train/test.csv',na_filter= False, index_col=0)

commondf=pd.merge(df,all_knowledge_, on=['expe','freq'], left_index=True)