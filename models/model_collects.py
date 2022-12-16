# -*- coding: utf-8 -*-
# @Time    : 12/16/22 1:04 PM
# @Author  : Wenliang Guo
# @email   : wg2397@columbia.edu
# @FileName: model_collects.py
# @Software: PyCharm

from models.transformer import Transformer
import torch
import torch.nn as nn

class TransLog(nn.Module):
    def __init__(self, max_length, in_dim=1, embed_dim=64, depth=6, heads=8,
                dim_head=64, dim_ratio=2, dropout=0.1):
        super(TransLog, self).__init__()
        self.transformer = Transformer(in_dim=in_dim, embed_dim=embed_dim,depth=depth,
                                       heads=heads,dim_head=dim_head,dim_ratio=dim_ratio,
                                       dropout=dropout)
        self.fc = nn.Linear(max_length*embed_dim, 2)

    def forward(self,x):
        x = self.transformer(x)
        x = self.fc(x)
        return x

class Large_TransLog(nn.Module):
    def __init__(self, max_length, in_dim=1, embed_dim=96, depth=8, heads=8,
                dim_head=64, dim_ratio=2, dropout=0.1):
        super(Large_TransLog, self).__init__()
        self.transformer = Transformer(in_dim=in_dim, embed_dim=embed_dim,depth=depth,
                                       heads=heads,dim_head=dim_head,dim_ratio=dim_ratio,
                                       dropout=dropout)
        self.fc = nn.Linear(max_length*embed_dim, 2)

    def forward(self,x):
        x = self.transformer(x)
        x = self.fc(x)
        return x

class Mini_TransLog(nn.Module):
    def __init__(self, max_length, in_dim=1, embed_dim=64, depth=4, heads=8,
                dim_head=64, dim_ratio=2, dropout=0.5):
        super(Mini_TransLog, self).__init__()
        self.transformer = Transformer(in_dim=in_dim, embed_dim=embed_dim,depth=depth,
                                       heads=heads,dim_head=dim_head,dim_ratio=dim_ratio,
                                       dropout=dropout)
        self.fc = nn.Linear(max_length*embed_dim, 2)

    def forward(self,x):
        x = self.transformer(x)
        x = self.fc(x)
        return x

class DeepLog(nn.Module):
    def __init__(self, input_size, device, hidden_size = 64, num_layers=4, num_classes = 2):
        super(DeepLog, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.device = device

    def forward(self,x):
        # 设置初始隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        # 前向传播 LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # 取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out

class DeepLog_GRU(nn.Module):
    def __init__(self, input_size, device, hidden_size = 64, num_layers=4, num_classes = 2):
        super(DeepLog_GRU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device

        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self,x):
        # 设置初始隐藏状态
        B = x.shape[0]
        h = torch.zeros(self.num_layers, B, self.hidden_size).to(self.device)

        # 前向传播 LSTM
        out, _ = self.gru(x, h)  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # 取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out
