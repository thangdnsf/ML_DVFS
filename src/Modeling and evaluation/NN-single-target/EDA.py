#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 16:46:31 2021

@author: ndthang
"""
from expetator.tools import read_experiment, show_heatmap, add_objectives
from expetator.tools import prune_vectors, mojitos_to_vectors, show_pct_distribution
from expetator.monitors import mojitos
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, f1_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
import pandas as pd
from datapreparation import generate_en_features, outlierfill
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from random import sample 

all_knowledge = pd.read_csv("../../csvs/knowledge_allmetrics.csv")
all_vectors = pd.read_csv("../../csvs/vectors_allmetrics.csv")
all_vectors['typedata'] = 0

all_knowledge_old = pd.read_csv("../../csvs/old/knowledge_allmetrics.csv")
all_vectors_old = pd.read_csv("../../csvs/old/vectors_allmetrics.csv")
all_knowledge_old['expe'] = all_knowledge_old['expe'] + all_knowledge.expe.values[-1] + 1
all_vectors_old['typedata'] = 1

all_knowledge = pd.concat([all_knowledge, all_knowledge_old],ignore_index=True)
all_vectors = pd.concat([all_vectors, all_vectors_old],ignore_index=True)


# merge knowlage to vectors => data for training 
all_vectors['expe'] = all_knowledge.expe
all_vectors['target'] = all_knowledge['target']

ignore_columns = ['irxp', 'irxb','itxp', 'itxb', 'core0','core1','bpf_output',
                 'alignment_faults','page_faults_maj','dummy','emulation_faults','nice',
                 'irq','steal','guest','guest_nice']
all_vectors = all_vectors.loc[:,~ all_vectors.columns.isin(ignore_columns)]
all_vectors_outlier = outlierfill(all_vectors)
vectorbin = generate_en_features(all_vectors_outlier)
vectors = prune_vectors(vectorbin)


ignore_columns = ['typedata','expe','group','target','Kfold', #'guest', 'guest_nice', 'irq',
                  #'steal','nice','emulation_faults','irxp', 'irxb', 
                  #'itxp', 'itxb', 'core0','core1','iowait','softirq','txp',
                 ]

# convert categories to numerical
metric_mapping = {
    "energy":0,
    "duration":1,
    "etp":2
}
#vectors['metric'] = all_knowledge['metric'].map(metric_mapping)
vectors = vectors[all_knowledge['metric'] == 'energy']
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
vectors['target'] = vectors.target.apply(lambda row: labels.index(row))

testidx = []
exps = list(set(vectors[vectors['typedata'] == 1].expe))
testidx.extend(sample(exps, int(0.2*len(exps))))

exps = list(set(vectors[vectors['typedata'] == 0].expe))
testidx.extend(sample(exps, int(0.2*len(exps))))

train = vectors[~vectors.expe.isin(testidx)].reset_index(drop=True)
test = vectors[vectors.expe.isin(testidx)].reset_index(drop=True)
