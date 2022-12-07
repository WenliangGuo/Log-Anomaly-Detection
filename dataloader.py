import pandas as pd
import numpy as np
import re
import sys
from datetime import timezone,timedelta
import os, os.path
from fnmatch import fnmatch
from sklearn.utils import shuffle
from collections import OrderedDict
from datetime import datetime as dt
from dateutil.parser import parse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def _split_data(x_data, y_data=None, train_ratio=0, split_type='uniform'):
    if split_type == 'uniform' and y_data is not None:
        pos_idx = y_data > 0
        x_pos = x_data[pos_idx]
        y_pos = y_data[pos_idx]
        x_neg = x_data[~pos_idx]
        y_neg = y_data[~pos_idx]
        train_pos = int(train_ratio * x_pos.shape[0])
        train_neg = int(train_ratio * x_neg.shape[0])
        x_train = np.hstack([x_pos[0:train_pos], x_neg[0:train_neg]])
        y_train = np.hstack([y_pos[0:train_pos], y_neg[0:train_neg]])
        x_test = np.hstack([x_pos[train_pos:], x_neg[train_neg:]])
        y_test = np.hstack([y_pos[train_pos:], y_neg[train_neg:]])
    elif split_type == 'sequential':
        num_train = int(train_ratio * x_data.shape[0])
        x_train = x_data[0:num_train]
        x_test = x_data[num_train:]
        if y_data is None:
            y_train = None
            y_test = None
        else:
            y_train = y_data[0:num_train]
            y_test = y_data[num_train:]
    # Random shuffle
    indexes = shuffle(np.arange(x_train.shape[0]))
    x_train = x_train[indexes]
    if y_train is not None:
        y_train = y_train[indexes]
    return (x_train, y_train), (x_test, y_test)

def load_HDFS(log_file, label_file, template_file, window='session', train_ratio=0.7, split_type='sequential',
    save_csv=False):
    """ Load HDFS structured log into train and test data

    Arguments
    ---------
        log_file: str, the file path of structured log.
        label_file: str, the file path of anomaly labels, None for unlabeled data.
        template_file: str, the file path of structured templates of logs.
        window: str, the window options including `session` (default).
        train_ratio: float, the ratio of training data for train/test split.
        split_type: `uniform` or `sequential`, which determines how to split logs. `uniform` means
            to split positive samples and negative samples equally when setting label_file. `sequential`
            means to split the data sequentially without label_file. That is, the first part is for training,
            while the second part is for testing.
        save_csv : True or False

    Returns
        x_train, y_train: the training data
        x_test, y_test: the testing data
    """

    ## load logs
    struct_log = pd.read_csv(log_file, engine='c', na_filter=False, memory_map=True)
    data_dict = OrderedDict()  # ordered dictionary
    for idx, row in struct_log.iterrows():
        blkId_list = re.findall(r'(blk_-?\d+)', row['Content'])
        blkId_set = set(blkId_list)
        for blk_Id in blkId_set:
            if not blk_Id in data_dict:
                data_dict[blk_Id] = []
            data_dict[blk_Id].append(row['EventId'])
    data_df = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'EventSequence'])

    ## load labels
    label_data = pd.read_csv(label_file, engine='c', na_filter=False, memory_map=True)
    label_data = label_data.set_index('BlockId')
    label_dict = label_data['Label'].to_dict()
    data_df['Label'] = data_df['BlockId'].apply(lambda x: 1 if label_dict[x] == 'Anomaly' else 0)

    # Split train and test data
    (x_train, y_train), (x_test, y_test) = _split_data(data_df['EventSequence'].values,
        data_df['Label'].values, train_ratio, split_type)

    x_train = to_idx(x_train, template_file)
    x_test = to_idx(x_test, template_file)

    if save_csv:
        data_df.to_csv('data_instances.csv', index=False)

    return x_train, y_train, x_test, y_test

def to_idx(x, template_file):
    ## convert template to index by EventID
    vocab2idx = dict()
    template_file = pd.read_csv(template_file, engine='c', na_filter=False, memory_map=True)
    for idx, template_id in enumerate(template_file['EventId'], start=len(vocab2idx)):
        vocab2idx[template_id] = idx + 1

    max_len = 0
    x_idx = []
    for i in range(x.shape[0]):
        if len(x[i]) > max_len:
            max_len = len(x[i])

    for i in range(x.shape[0]):
        temp = []
        for j in range(len(x[i])):
            temp.append(vocab2idx[x[i][j]])
        temp += [0]*(max_len-len(x[i]))
        x_idx.append(temp)

    return np.array(x_idx,dtype=float)