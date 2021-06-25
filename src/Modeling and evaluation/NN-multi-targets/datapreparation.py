#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 17:04:08 2021

@author: ndthang
"""
import pandas as pd
import numpy as np

from expetator.tools import read_experiment, show_heatmap, add_objectives
from expetator.tools import prune_vectors, mojitos_to_vectors, show_pct_distribution
from expetator.monitors import mojitos
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, f1_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from random import sample 


def generate_en_features(df):
    df = df.copy()
    cols = set(df.columns)-{'metric','page_faults_min','expe','group','target'}
    for col in cols:
        for func in ['mean','std','min','max']:
            temp = df.groupby('expe')[col].agg([func]).rename({func:f"{col}_{func}"},axis=1)
            df = pd.merge(df,temp,on='expe',how='left')
        df[col+'_bin10'] = pd.cut(df[col], bins = 10, labels = False)
        df[col+'_bin100'] = pd.cut(df[col], bins = 100, labels = False)
        df[col+'_log'] = df[col].apply(lambda row: np.log(1+row))
        df[col+'_log+1'] = (df[col]+1).transform(np.log)
        df[col+'_log(x-min(x)+1)'] = (df[col]-df[col].min()+1) .transform(np.log)
    return df
def generate_aggregations(df, vectors):
    df = df.copy()
    for col in set(df.columns)-{'metric','expe','freq','group','target'}:
        for func in ['mean','std','min','max']:
          temp = vectors.groupby('expe')[col].agg([func]).rename({func:f"{col}_{func}"},axis=1)
          df = pd.merge(df,temp,on='expe',how='left')
    return df
def outlierdrop(df,factor):
    for col in set(df.columns) - {'metric','expe','freq','group','target'}:
        upper_lim = df[col].mean () + df[col].std () * factor
        lower_lim = df[col].mean () - df[col].std () * factor
        df = df[(df[col] < upper_lim) & (df[col] > lower_lim)]
    return df
def outlierfill(df):
    for column in set(df.columns) - {'metric','expe','freq','group','target'}:
        upper_lim = df[column].quantile(.95)
        lower_lim = df[column].quantile(0.05)
        df.loc[(df[column] > upper_lim),column] = upper_lim
        df.loc[(df[column] < lower_lim),column] = lower_lim
    return df

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


# merge knowlage to vectors => data for training 
all_vectors['expe'] = all_knowledge.expe
all_vectors['target'] = all_knowledge['target']
all_vectors['targetZ'] = all_knowledge['targetZ']

ignore_columns = ['irxp', 'irxb','itxp', 'itxb', 'core0','core1','bpf_output',
                 'alignment_faults','page_faults_maj','dummy','emulation_faults','nice',
                 'irq','steal','guest','guest_nice']
all_vectors = all_vectors.loc[:,~ all_vectors.columns.isin(ignore_columns)]
all_vectors_outlier = outlierfill(all_vectors)
vectorbin = generate_en_features(all_vectors_outlier)
vectors = prune_vectors(vectorbin)


#ignore_columns = ['typedata','expe','UID','target','Kfold', #'guest', 'guest_nice', 'irq',
                  #'steal','nice','emulation_faults','irxp', 'irxb', 
                  #'itxp', 'itxb', 'core0','core1','iowait','softirq','txp',
#                 ]

# convert categories to numerical
metric_mapping = {
    "energy":0,
    "duration":1,
    "etp":2
}
vectors['metric'] = all_knowledge['metric'].map(metric_mapping)
#vectors = vectors[all_knowledge['metric'] == 'energy']

testidx = []
exps = list(set(vectors[vectors['typedata'] == 1].expe))
testidx.extend(sample(exps, int(0.2*len(exps))))

exps = list(set(vectors[vectors['typedata'] == 0].expe))
testidx.extend(sample(exps, int(0.2*len(exps))))

#split data validation
train = vectors[~vectors.expe.isin(testidx)].reset_index(drop=True)
test = vectors[vectors.expe.isin(testidx)].reset_index(drop=True)

#split kfold for training dataset
skf = GroupKFold(n_splits=5)
train['UID'] = train.apply(lambda row: str(row['expe'])+str(row['typedata']),axis = 1)
train['kfold'] = -1
for i, (train_index, test_index) in enumerate(skf.split(train, train.target, groups = train['UID'])):
    train.loc[test_index,'kfold'] = i

test.to_csv('../../data4train/test.csv')
train.to_csv('../../data4train/train.csv')